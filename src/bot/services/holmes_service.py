"""Holmes Service - Full HolmesGPT integration with agents, tools, memory, and streaming."""
import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass

import structlog
from holmes.config import Config
from holmes.core.tool_calling_llm import ToolCallingLLM, LLMResult
from holmes.core.tools_utils.tool_executor import ToolExecutor
from holmes.core.tracing import TracingFactory
from holmes.core.tools import ToolsetTag, PrerequisiteCacheMode
from holmes.core.prompt import build_initial_ask_messages
from holmes.common.env_vars import DEFAULT_CLI_USER
from holmes.core.oauth_utils import enable_disk_token_store
from holmes.core.models import PendingToolApproval, ToolApprovalDecision

from src.bot.utils.config import config as bot_config
from src.bot.models.conversation import Conversation, Message
from src.bot.models.user import UserPermission

logger = structlog.get_logger(__name__)

# Request context for CLI-like usage
_CLI_REQUEST_CONTEXT = {"user_id": DEFAULT_CLI_USER}

# Dedicated thread pool for Holmes calls to avoid blocking the event loop
# and to allow concurrent processing of multiple requests
_holmes_executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix="holmes-")

# Store pending approval callbacks for Telegram integration
# Format: {approval_id: {"event": asyncio.Event, "result": (bool, str)}}
_pending_approvals: Dict[str, Dict[str, Any]] = {}


@dataclass
class ApprovalResult:
    """Result of a tool approval request."""
    approved: bool
    feedback: Optional[str] = None


class HolmesService:
    """Service for interacting with HolmesGPT."""

    def __init__(self):
        self._config: Optional[Config] = None
        self._tool_executor: Optional[ToolExecutor] = None
        self._tool_calling_llm: Optional[ToolCallingLLM] = None
        self._initialized = False

    def _get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration from environment variables."""
        # Support both Anthropic and OpenAI-compatible endpoints
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        openai_base = os.environ.get("OPENAI_API_BASE")
        openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o")

        # Determine which provider to use
        if openai_key and openai_base:
            # Custom OpenAI-compatible endpoint
            # Always prepend "openai/" for litellm to recognize OpenAI-compatible endpoints
            model = f"openai/{openai_model}"
            return {
                "provider": "openai",
                "model": model,
                "api_key": openai_key,
                "api_base": openai_base,
            }
        elif anthropic_key:
            # Anthropic (default)
            return {
                "provider": "anthropic",
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "api_key": anthropic_key,
                "api_base": None,
            }
        elif openai_key:
            # Standard OpenAI
            return {
                "provider": "openai",
                "model": f"openai/{openai_model}",
                "api_key": openai_key,
                "api_base": None,
            }
        else:
            logger.warning("No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY for Holmes to work.")
            return {
                "provider": "none",
                "model": "openai/gpt-4o",
                "api_key": "placeholder",
                "api_base": None,
            }

    async def initialize(self):
        """Initialize HolmesGPT components using native config."""
        if self._initialized:
            return

        try:
            enable_disk_token_store()

            # Try to load from config file first (like Holmes CLI does)
            config_path = os.environ.get("HOLMES_CONFIG_PATH", "/root/.holmes/config.yaml")

            if os.path.exists(config_path):
                # Load from config file - Holmes native way
                self._config = Config.load_from_file(Path(config_path))

                # Override model/api settings from env if provided
                llm_config = self._get_llm_config()
                if llm_config.get("model") and llm_config["model"] != "openai/gpt-4o":
                    self._config.model = llm_config["model"]
                if llm_config.get("api_key") and llm_config["api_key"] != "placeholder":
                    self._config.api_key = llm_config["api_key"]
                if llm_config.get("api_base"):
                    self._config.api_base = llm_config["api_base"]

                logger.info("Loaded Holmes config from file", path=config_path)
            else:
                # Fallback: create config manually from environment variables
                llm_config = self._get_llm_config()

                config_kwargs = {
                    "model": llm_config["model"],
                    "api_key": llm_config["api_key"],
                    "max_steps": 10,
                    "cluster_name": "telegram-bot",
                    "toolsets": {},
                    "mcp_servers": {},
                    "additional_toolsets": [],
                }

                if llm_config.get("api_base"):
                    config_kwargs["api_base"] = llm_config["api_base"]

                self._config = Config(**config_kwargs)
                logger.info("Created Holmes config from environment variables")

            # Create tracer for LLM wrapping
            tracer_factory = TracingFactory()
            tracer = tracer_factory.create_tracer("langchain")

            # Initialize tool executor using Config's built-in method to load proper toolsets
            self._tool_executor = self._config.create_tool_executor(
                toolset_tag_filter=[ToolsetTag.CORE, ToolsetTag.CLI],
                enable_all_toolsets_possible=True,
                prerequisite_cache=PrerequisiteCacheMode.ENABLED,
            )

            # Wrap TodoWrite tool to capture investigation task updates
            self._wrap_todo_write_tool()

            # Initialize tool calling LLM using Config's built-in method
            self._tool_calling_llm = self._config.create_toolcalling_llm(
                toolset_tag_filter=[ToolsetTag.CORE, ToolsetTag.CLI],
                enable_all_toolsets_possible=True,
                prerequisite_cache=PrerequisiteCacheMode.ENABLED,
                model=self._config.model,
                tracer=tracer,
                tool_results_dir=None,
            )

            self._initialized = True
            logger.info("Holmes service initialized successfully",
                       model=self._config.model,
                       api_base=getattr(self._config, 'api_base', None))
        except Exception as e:
            logger.error("Failed to initialize Holmes service", error=str(e))
            raise

    def _wrap_todo_write_tool(self):
        """Wrap the TodoWrite tool to capture investigation task updates."""
        if not self._tool_executor:
            return

        for toolset in self._tool_executor.toolsets:
            for i, tool in enumerate(toolset.tools):
                if tool.name == "TodoWrite":
                    original_invoke = tool.invoke

                    def make_wrapper(original_func):
                        async def wrapped_invoke(params, context=None):
                            # Call original tool
                            result = await original_func(params, context)

                            # Capture tasks from params or result
                            logger.info("TodoWrite called", params=params, result_type=type(result).__name__)

                            tasks = None
                            if isinstance(params, dict):
                                if "tasks" in params:
                                    tasks = params["tasks"]
                                elif "todos" in params:
                                    tasks = params["todos"]

                            # Also check result
                            if tasks is None and result:
                                if hasattr(result, 'data') and isinstance(result.data, dict):
                                    if "tasks" in result.data:
                                        tasks = result.data["tasks"]
                                    elif "todos" in result.data:
                                        tasks = result.data["todos"]
                                elif isinstance(result, dict):
                                    if "tasks" in result:
                                        tasks = result["tasks"]
                                    elif "todos" in result:
                                        tasks = result["todos"]

                            if tasks:
                                self._last_investigation_tasks = tasks
                                logger.info("Captured investigation tasks", num_tasks=len(tasks))
                                # Print task details
                                for t in tasks:
                                    logger.info("Task", id=t.get('id'), content=t.get('content'), status=t.get('status'))

                            return result
                        return wrapped_invoke

                    # Use object.__setattr__ for frozen Pydantic models
                    object.__setattr__(tool, 'invoke', make_wrapper(original_invoke))
                    logger.info("Wrapped TodoWrite tool for task tracking")
                    break

    def get_latest_investigation_tasks(self) -> List[Dict[str, Any]]:
        """Get the latest investigation tasks captured from TodoWrite."""
        return getattr(self, '_last_investigation_tasks', [])

    def extract_tasks_from_metadata(self, conversation: Conversation) -> List[Dict[str, Any]]:
        """Extract investigation tasks from Holmes conversation metadata."""
        if not conversation.metadata:
            return []

        # Try various metadata keys where Holmes might store tasks
        for key in ['investigation_tasks', 'tasks', 'todo_list', 'todo_tasks', 'agent_tasks']:
            if key in conversation.metadata:
                tasks = conversation.metadata[key]
                if isinstance(tasks, list) and tasks:
                    return tasks

        # Also check nested metadata
        if 'metadata' in conversation.metadata:
            nested = conversation.metadata['metadata']
            if isinstance(nested, dict):
                for key in ['investigation_tasks', 'tasks', 'todo_list', 'todo_tasks', 'agent_tasks']:
                    if key in nested:
                        tasks = nested[key]
                        if isinstance(tasks, list) and tasks:
                            return tasks

        return []

    async def close(self):
        """Close Holmes client."""
        self._tool_calling_llm = None
        self._tool_executor = None
        self._config = None
        self._initialized = False

    def _create_approval_callback(self, user_id: int, chat_id: int, topic_id: int, context: Any) -> Callable[[PendingToolApproval], Tuple[bool, Optional[str]]]:
        """Create an approval callback that sends request to Telegram and waits for response."""
        def approval_callback(approval: PendingToolApproval) -> Tuple[bool, Optional[str]]:
            try:
                logger.warning("=== APPROVAL CALLBACK TRIGGERED ===",
                           tool_name=approval.tool_name,
                           tool_description=approval.description,
                           params=approval.params,
                           user_id=user_id,
                           chat_id=chat_id,
                           topic_id=topic_id)
                # Generate unique approval ID
                import uuid
                approval_id = str(uuid.uuid4())

                # Store the approval request
                loop = asyncio.get_event_loop()
                future = loop.create_future()

                _pending_approvals[approval_id] = {
                    "approval": approval,
                    "future": future,
                    "user_id": user_id,
                    "chat_id": chat_id,
                    "topic_id": topic_id,
                }

                # Schedule the approval request to be sent to Telegram
                asyncio.run_coroutine_threadsafe(
                    self._send_approval_request(approval_id, approval, user_id, chat_id, topic_id, context),
                    loop
                )

                # Wait for the result (with timeout)
                try:
                    # Run the future in the thread pool context
                    result = asyncio.run_coroutine_threadsafe(future, loop).result(timeout=300)  # 5 min timeout
                    logger.warning("Approval callback completed", approval_id=approval_id, result=result)
                    return result
                except Exception as e:
                    logger.error("Approval callback error", approval_id=approval_id, error=str(e))
                    return False, str(e)
                finally:
                    _pending_approvals.pop(approval_id, None)
            except Exception as e:
                logger.error("Approval callback outer error", error=str(e))
                return False, str(e)

        return approval_callback

    async def _send_approval_request(self, approval_id: str, approval: PendingToolApproval, user_id: int, chat_id: int, topic_id: int, context: Any):
        """Send tool approval request to Telegram."""
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            # Format the approval message
            params_str = ""
            if approval.params:
                import json
                params_str = json.dumps(approval.params, indent=2, ensure_ascii=False)
                if len(params_str) > 500:
                    params_str = params_str[:500] + "... (truncated)"

            text = (
                f"🔐 **Tool Approval Required**\n\n"
                f"**Tool:** {approval.tool_name}\n"
                f"**Description:** {approval.description}\n"
                f"**Parameters:**\n```json\n{params_str}\n```\n\n"
                f"Approve this tool execution?"
            )

            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Approve", callback_data=f"tool_approve_{approval_id}"),
                    InlineKeyboardButton("❌ Deny", callback_data=f"tool_deny_{approval_id}"),
                ]
            ])

            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=keyboard,
                message_thread_id=topic_id if topic_id else None,
                parse_mode="Markdown"
            )

            logger.info("Sent tool approval request", approval_id=approval_id, user_id=user_id)
        except Exception as e:
            logger.error("Failed to send approval request", approval_id=approval_id, error=str(e))
            # Resolve the future with denial
            if approval_id in _pending_approvals:
                _pending_approvals[approval_id]["future"].set_result((False, str(e)))

    @staticmethod
    def handle_tool_approval_response(approval_id: str, approved: bool, feedback: Optional[str] = None):
        """Handle user's response to tool approval request."""
        logger.warning("handle_tool_approval_response CALLED", approval_id=approval_id, approved=approved, feedback=feedback)
        if approval_id in _pending_approvals:
            future = _pending_approvals[approval_id]["future"]
            if not future.done():
                future.set_result((approved, feedback))
            logger.info("Tool approval response received", approval_id=approval_id, approved=approved)
        else:
            logger.warning("handle_tool_approval_response: approval_id not found", approval_id=approval_id)

    async def _send_stream_approval_request(self, approval_id: str, tool_name: str, description: str, params: dict, user_id: int, chat_id: int, topic_id: int, context: Any, approval_token: str = None):
        """Send streaming tool approval request to Telegram."""
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            # Format the approval message
            params_str = ""
            if params:
                import json
                params_str = json.dumps(params, indent=2, ensure_ascii=False)
                if len(params_str) > 500:
                    params_str = params_str[:500] + "... (truncated)"

            text = (
                f"🔐 **Tool Approval Required (Streaming)**\n\n"
                f"**Tool:** {tool_name}\n"
                f"**Description:** {description}\n"
                f"**Parameters:**\n```json\n{params_str}\n```\n\n"
                f"Approve this tool execution?"
            )

            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Approve", callback_data=f"tool_approve_{approval_id}"),
                    InlineKeyboardButton("❌ Deny", callback_data=f"tool_deny_{approval_id}"),
                ]
            ])

            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=keyboard,
                message_thread_id=topic_id if topic_id else None,
                parse_mode="Markdown"
            )

            logger.info("Sent streaming tool approval request", approval_id=approval_id, user_id=user_id)
        except Exception as e:
            logger.error("Failed to send streaming approval request", approval_id=approval_id, error=str(e))
            # Resolve the future with denial
            if approval_id in _pending_approvals:
                _pending_approvals[approval_id]["future"].set_result((False, str(e)))

    def _get_user_permissions(self, user_permissions: List[UserPermission]) -> Dict[str, bool]:
        """Convert user permissions to Holmes feature flags."""
        return {
            "agents": UserPermission.AGENT_EXECUTION in user_permissions,
            "tools": UserPermission.TOOL_USAGE in user_permissions,
            "memory": UserPermission.CONVERSATION_MEMORY in user_permissions,
            "streaming": UserPermission.STREAMING_RESPONSES in user_permissions,
            "tool_approval": UserPermission.TOOL_APPROVAL in user_permissions,
        }

    def _convert_messages_to_holmes_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our Message model to Holmes message format (with tool_calls, tool_call_id, name)."""
        holmes_messages = []
        for msg in messages:
            holmes_msg = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.name:
                holmes_msg["name"] = msg.name
            if msg.tool_calls:
                holmes_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                holmes_msg["tool_call_id"] = msg.tool_call_id
            holmes_messages.append(holmes_msg)
        return holmes_messages

    def _normalize_tool_result(self, tool_call: Any) -> dict:
        """
        Normalize MCP tool result to OpenAI-compatible format.

        MCP tools return rich results with metadata. The LLM only needs:
        - tool_call_id (to match the call)
        - function.name (tool name)
        - function.arguments (params as JSON string)
        - content (result as string)
        """
        try:
            # If already in correct format, return as-is
            if isinstance(tool_call, dict) and "function" in tool_call:
                return tool_call

            # Extract from MCP result format
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("tool_name") or tool_call.get("name")
                params = tool_call.get("params", {})
                data = tool_call.get("data", "")
                tool_call_id = tool_call.get("tool_call_id") or tool_call.get("id")

                # If data is a JSON string, parse it to extract clean content
                content = data
                if isinstance(data, str):
                    try:
                        import json
                        parsed = json.loads(data)
                        # Extract meaningful content from structured results
                        if isinstance(parsed, dict) and "results" in parsed:
                            # For search_contacts and similar - summarize results
                            results = parsed["results"]
                            if isinstance(results, list):
                                content = f"Found {len(results)} results"
                                for r in results[:3]:  # First 3 items
                                    if isinstance(r, dict):
                                        name = r.get("name") or r.get("username") or str(r)
                                        content += f"\n- {name}"
                        elif isinstance(parsed, dict):
                            # Generic dict - stringify
                            content = json.dumps(parsed, ensure_ascii=False)
                    except json.JSONDecodeError:
                        pass  # Keep original string

                return {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(params, ensure_ascii=False) if params else "{}"
                    }
                }
        except Exception as e:
            logger.warning("Failed to normalize tool call, using fallback", error=str(e))

        # Fallback: return minimal valid structure
        return {
            "id": "unknown",
            "type": "function",
            "function": {
                "name": "unknown_tool",
                "arguments": "{}"
            }
        }

    def _convert_llm_result_to_message(self, result: LLMResult) -> Message:
        """Convert LLMResult to our Message model."""
        # Convert ToolCallResult objects to serializable dictionaries
        tool_calls = []
        if result.tool_calls:
            for tc in result.tool_calls:
                try:
                    if hasattr(tc, 'to_client_dict'):
                        tc_dict = tc.to_client_dict()
                    elif hasattr(tc, 'model_dump'):
                        tc_dict = tc.model_dump()
                    else:
                        tc_dict = tc

                    # Normalize to OpenAI format for LLM compatibility
                    normalized = self._normalize_tool_result(tc_dict)
                    tool_calls.append(normalized)
                except Exception as e:
                    logger.warning("Failed to serialize tool call, skipping", error=str(e))
                    # Skip malformed tool calls rather than crashing
                    continue

        return Message(
            role="assistant",
            content=result.result or "",
            tool_calls=tool_calls,
            metadata={
                "num_llm_calls": result.num_llm_calls,
                "finish_reason": result.finish_reason,
                "metadata": result.metadata,
            },
        )

    async def chat(
        self,
        user_id: int,
        user_message: str,
        conversation: Conversation,
        user_permissions: List[UserPermission],
        stream: bool = False,
        telegram_context: Any = None,
        chat_id: int = 0,
        topic_id: int = 0,
    ) -> str | AsyncGenerator[str, None]:
        """
        Send a message to Holmes and get response.

        Args:
            user_id: Telegram user ID
            user_message: User's message text
            conversation: Current conversation history
            user_permissions: User's permissions
            stream: Whether to stream the response
            telegram_context: Telegram context for sending approval requests
            chat_id: Chat ID for approval messages
            topic_id: Topic ID for approval messages

        Returns:
            Response text or async generator for streaming
        """
        await self.initialize()

        # Add user message to conversation
        user_msg = Message(role="user", content=user_message)
        conversation.add_message(user_msg)

        # Convert to Holmes format using build_initial_ask_messages for proper formatting with system prompt additions
        holmes_messages = self._build_holmes_messages(
            self._convert_messages_to_holmes_format(conversation.messages),
            user_id
        )

        # Get feature flags from permissions
        features = self._get_user_permissions(user_permissions)

        try:
            if stream and features.get("streaming", False):
                return self._stream_chat(holmes_messages, features, user_id, conversation, telegram_context, chat_id, topic_id)
            else:
                # Pass telegram context for approval callback
                response_text, llm_result = await self._non_stream_chat_with_memory(
                    holmes_messages, features, user_id, telegram_context, chat_id, topic_id
                )

                # Save Holmes response with tool calls and metadata to conversation
                try:
                    assistant_msg = self._convert_llm_result_to_message(llm_result)
                    conversation.add_message(assistant_msg)
                except Exception as e:
                    logger.warning("Failed to save assistant message, continuing without tool calls", error=str(e))
                    # Add a basic message without tool calls so conversation continues
                    conversation.add_message(Message(role="assistant", content=response_text or ""))

                # Update Holmes session memory
                self.update_conversation_memory(conversation, {
                    "session_id": llm_result.metadata.get("session_id") if llm_result.metadata else None,
                    "metadata": llm_result.metadata or {},
                })

                return response_text
        except Exception as e:
            logger.error("Holmes chat error", user_id=user_id, error=str(e))
            # Return a user-friendly error instead of crashing
            error_msg = "I encountered an error while processing your request. Please try again."
            conversation.add_message(Message(role="assistant", content=error_msg))
            return error_msg

    def _build_holmes_messages(self, messages: List[Dict[str, Any]], user_id: int) -> List[Dict[str, Any]]:
        """Build Holmes-compatible messages with proper system prompt and tool formatting."""
        # Use Holmes' native message builder for proper formatting
        return build_initial_ask_messages(
            initial_user_prompt="",  # We'll override with actual messages
            file_paths=None,
            tool_executor=self._tool_executor,
            skills=self._config.get_skill_catalog() if self._config else None,
            system_prompt_additions=bot_config.holmes_system_prompt_additions,
        ) + messages

    async def _non_stream_chat(
        self, messages: List[Dict[str, Any]], features: Dict[str, bool], user_id: int
    ) -> str:
        """Non-streaming chat with Holmes."""
        loop = asyncio.get_event_loop()

        # Run in dedicated executor since ToolCallingLLM.call is sync
        # Note: Non-streaming call uses approval_callback, not enable_tool_approval
        # For simplicity, we don't implement interactive approval in non-streaming mode
        result: LLMResult = await loop.run_in_executor(
            _holmes_executor,
            lambda: self._tool_calling_llm.call(
                messages=messages,
                request_context={**_CLI_REQUEST_CONTEXT, "user_id": str(user_id)},
            ),
        )

        return result.result or "I apologize, but I couldn't generate a response."

    async def _non_stream_chat_with_memory(
        self,
        messages: List[Dict[str, Any]],
        features: Dict[str, bool],
        user_id: int,
        telegram_context: Any = None,
        chat_id: int = 0,
        topic_id: int = 0
    ) -> tuple[str, LLMResult]:
        """Non-streaming chat with Holmes, returning full result for memory storage."""
        loop = asyncio.get_event_loop()

        # Use approval callback if tool_approval feature is enabled and we have telegram context
        enable_approval = features.get("tool_approval", False) and telegram_context is not None
        logger.warning("=== NON-STREAM CHAT ===",
                    user_id=user_id,
                    tool_approval_feature=features.get("tool_approval", False),
                    has_telegram_context=telegram_context is not None,
                    enable_approval=enable_approval,
                    features=features)

        if enable_approval:
            approval_callback = self._create_approval_callback(user_id, chat_id, topic_id, telegram_context)
            logger.warning("Created approval callback", user_id=user_id)
            result: LLMResult = await loop.run_in_executor(
                _holmes_executor,
                lambda: self._tool_calling_llm.call(
                    messages=messages,
                    request_context={**_CLI_REQUEST_CONTEXT, "user_id": str(user_id)},
                    approval_callback=approval_callback,
                ),
            )
        else:
            # No interactive approval
            logger.warning("Skipping approval callback - reason: tool_approval=%s, telegram_context=%s",
                          features.get("tool_approval", False), telegram_context is not None)
            result: LLMResult = await loop.run_in_executor(
                _holmes_executor,
                lambda: self._tool_calling_llm.call(
                    messages=messages,
                    request_context={**_CLI_REQUEST_CONTEXT, "user_id": str(user_id)},
                ),
            )

        return result.result or "I apologize, but I couldn't generate a response.", result

    async def _stream_chat(
        self, messages: List[Dict[str, Any]], features: Dict[str, bool], user_id: int, conversation: Conversation = None,
        telegram_context: Any = None, chat_id: int = 0, topic_id: int = 0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming chat with Holmes, yielding structured events including task updates and handling tool approval."""
        task_state = {"last_tasks": []}
        async for evt in self._run_holmes_stream(
            messages, None, features, user_id, conversation, telegram_context, chat_id, topic_id, task_state
        ):
            yield evt

    async def _run_holmes_stream(
        self,
        messages: List[Dict[str, Any]],
        tool_decisions: Optional[List["ToolApprovalDecision"]],
        features: Dict[str, bool],
        user_id: int,
        conversation: Optional[Conversation],
        telegram_context: Any,
        chat_id: int,
        topic_id: int,
        task_state: Dict[str, Any],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run (or resume) one Holmes stream call, recursing on APPROVAL_REQUIRED
        so any number of approval rounds are handled with the correctly
        updated message list each time."""
        loop = asyncio.get_event_loop()
        enable_approval = features.get("tool_approval", False)

        def stream_generator():
            logger.warning("Stream generator", enable_approval=enable_approval, has_tool_decisions=tool_decisions is not None)
            return self._tool_calling_llm.call_stream(
                msgs=messages,
                request_context={**_CLI_REQUEST_CONTEXT, "user_id": str(user_id)},
                enable_tool_approval=enable_approval,
                tool_decisions=tool_decisions,
            )

        stream = await loop.run_in_executor(_holmes_executor, stream_generator)

        last_task_check = 0

        for event in stream:
            event_name = event.event.name
            event_data = event.data

            # DEBUG: Log all event types to understand what's emitted
            logger.debug("Holmes stream event", event_name=event_name, data_keys=list(event_data.keys()) if isinstance(event_data, dict) else "non-dict")

            # Periodically check for investigation task updates (every ~5 events)
            last_task_check += 1
            if last_task_check >= 5:
                last_task_check = 0
                # Check both wrapped tool and conversation metadata
                current_tasks = self.get_latest_investigation_tasks()
                if conversation and not current_tasks:
                    current_tasks = self.extract_tasks_from_metadata(conversation)
                if current_tasks != task_state["last_tasks"] and current_tasks:
                    task_state["last_tasks"] = current_tasks
                    yield {
                        "type": "task_update",
                        "tasks": current_tasks
                    }

            if event_name == "AI_MESSAGE":
                content = event_data.get("content")
                reasoning = event_data.get("reasoning") or event_data.get("thinking")

                if content:
                    yield {
                        "type": "content",
                        "content": content
                    }
                # Also yield reasoning if present in the message
                if reasoning:
                    yield {
                        "type": "thinking",
                        "content": reasoning
                    }
            elif event_name == "TOOL_CALL":
                # Tool call started
                tool_name = event_data.get("tool_name", "unknown")
                tool_params = event_data.get("params", {})
                yield {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "params": tool_params
                }
                # If it's a TodoWrite call, also yield task update immediately
                if tool_name == "TodoWrite" and isinstance(tool_params, dict):
                    if "tasks" in tool_params:
                        yield {
                            "type": "task_update",
                            "tasks": tool_params["tasks"]
                        }
                    elif "todos" in tool_params:
                        yield {
                            "type": "task_update",
                            "tasks": tool_params["todos"]
                        }
            elif event_name == "TOOL_RESULT":
                # Tool completed
                tool_name = event_data.get("tool_name", "unknown")
                result = event_data.get("result", "")
                yield {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "result": result
                }
            elif event_name == "TASK_UPDATE":
                # Investigation task list updated
                tasks = event_data.get("tasks", [])
                yield {
                    "type": "task_update",
                    "tasks": tasks
                }
            elif event_name == "THINKING":
                # AI reasoning/thinking
                thinking = event_data.get("content", "")
                yield {
                    "type": "thinking",
                    "content": thinking
                }
            elif event_name == "ANSWER_END":
                # Final answer for this stream (e.g. after a tool-approval resume
                # with no further tool calls). Must surface content or the caller
                # never appends it to the response and ends up editing empty text.
                content = event_data.get("content")
                if content:
                    yield {
                        "type": "content",
                        "content": content
                    }
            elif event_name == "APPROVAL_REQUIRED":
                # Tool approval required - send to Telegram and wait for response.
                # Holmes marks the pending tool_calls (pending_approval + approval_token)
                # on the messages list it hands back here - we must resume with THIS
                # list, not the one we started with, or it won't find the approval.
                logger.info("APPROVAL_REQUIRED event received", pending_approvals_count=len(event_data.get("pending_approvals", [])))

                updated_messages = event_data.get("messages") or messages
                pending_approvals = event_data.get("pending_approvals", [])
                decisions: List[ToolApprovalDecision] = []

                for approval in pending_approvals:
                    tool_call_id = approval.get("tool_call_id")
                    tool_name = approval.get("tool_name", "unknown")
                    description = approval.get("description", "")
                    params = approval.get("params", {})

                    approval_id = str(uuid.uuid4())
                    future = loop.create_future()
                    _pending_approvals[approval_id] = {
                        "approval": approval,
                        "future": future,
                        "user_id": user_id,
                        "chat_id": chat_id,
                        "topic_id": topic_id,
                        "tool_call_id": tool_call_id,
                    }

                    if telegram_context:
                        asyncio.run_coroutine_threadsafe(
                            self._send_stream_approval_request(approval_id, tool_name, description, params, user_id, chat_id, topic_id, telegram_context),
                            loop
                        )

                    # Wait for user response
                    try:
                        result = await asyncio.wrap_future(future)
                        approved, feedback = result
                        logger.info("Approval received", approval_id=approval_id, approved=approved)
                        decisions.append(ToolApprovalDecision(
                            tool_call_id=tool_call_id,
                            approved=approved,
                            feedback=feedback or None,
                        ))
                        yield {
                            "type": "approval_result",
                            "tool_call_id": tool_call_id,
                            "approved": approved,
                            "feedback": feedback
                        }
                    except Exception as e:
                        logger.error("Approval error", approval_id=approval_id, error=str(e))
                        decisions.append(ToolApprovalDecision(
                            tool_call_id=tool_call_id,
                            approved=False,
                            feedback=str(e),
                        ))
                        yield {
                            "type": "approval_result",
                            "tool_call_id": tool_call_id,
                            "approved": False,
                            "feedback": str(e)
                        }
                    finally:
                        _pending_approvals.pop(approval_id, None)

                # Resume the stream with decisions, using the updated message
                # list Holmes gave us (the one with pending_approval flags set).
                # Recurse so any further approval rounds are handled too.
                if decisions:
                    async for resumed_evt in self._run_holmes_stream(
                        updated_messages, decisions, features, user_id, conversation,
                        telegram_context, chat_id, topic_id, task_state
                    ):
                        yield resumed_evt
                return
            else:
                yield {
                    "type": "event",
                    "event_name": event_name,
                    "data": event_data
                }

        # Final check for any remaining task updates
        current_tasks = self.get_latest_investigation_tasks()
        if conversation and not current_tasks:
            current_tasks = self.extract_tasks_from_metadata(conversation)
        if current_tasks != task_state["last_tasks"] and current_tasks:
            task_state["last_tasks"] = current_tasks
            yield {
                "type": "task_update",
                "tasks": current_tasks
            }

    async def execute_agent(
        self,
        user_id: int,
        agent_name: str,
        task: str,
        conversation: Conversation,
        user_permissions: List[UserPermission],
    ) -> str:
        """
        Execute a specific Holmes agent/task.

        Args:
            user_id: Telegram user ID
            agent_name: Name of the agent/task
            task: Task description
            conversation: Current conversation
            user_permissions: User's permissions

        Returns:
            Agent execution result
        """
        await self.initialize()

        if UserPermission.AGENT_EXECUTION not in user_permissions:
            raise PermissionError("User does not have agent execution permission")

        # Add task as user message
        user_msg = Message(role="user", content=f"[Agent Task: {agent_name}] {task}")
        conversation.add_message(user_msg)

        holmes_messages = self._build_holmes_messages(
            self._convert_messages_to_holmes_format(conversation.messages),
            user_id
        )
        features = self._get_user_permissions(user_permissions)

        loop = asyncio.get_event_loop()
        result: LLMResult = await loop.run_in_executor(
            _holmes_executor,
            lambda: self._tool_calling_llm.call(
                messages=holmes_messages,
                request_context={**_CLI_REQUEST_CONTEXT, "user_id": str(user_id)},
            ),
        )

        if result.result:
            # Add assistant response to conversation
            assistant_msg = self._convert_llm_result_to_message(result)
            conversation.add_message(assistant_msg)
            return result.result

        return "Agent execution completed without output."

    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from tool executor."""
        await self.initialize()

        if not self._tool_executor:
            return []

        tools = []
        for toolset in self._tool_executor.toolsets:
            for tool in toolset.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "toolset": toolset.name,
                })
        return tools

    def get_conversation_memory(self, conversation: Conversation) -> Dict[str, Any]:
        """Extract conversation memory for Holmes."""
        return {
            "messages": conversation.to_holmes_format(),
            "metadata": conversation.metadata,
            "session_id": conversation.holmes_session_id,
        }

    def update_conversation_memory(
        self, conversation: Conversation, memory: Dict[str, Any]
    ):
        """Update conversation with Holmes memory."""
        if "session_id" in memory:
            conversation.holmes_session_id = memory["session_id"]
        if "metadata" in memory:
            conversation.metadata.update(memory["metadata"])