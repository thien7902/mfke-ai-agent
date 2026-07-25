"""Holmes Service - Full HolmesGPT integration with agents, tools, memory, and streaming."""
import asyncio
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Dict, Any, Tuple

import structlog
from holmes.config import Config
from holmes.core.tool_calling_llm import ToolCallingLLM, LLMResult
from holmes.core.tools_utils.tool_executor import ToolExecutor
from holmes.core.tracing import TracingFactory
from holmes.core.tools import ToolsetTag, PrerequisiteCacheMode
from holmes.core.prompt import build_initial_ask_messages
from holmes.common.env_vars import DEFAULT_CLI_USER
from holmes.core.oauth_utils import enable_disk_token_store

from src.bot.utils.config import config as bot_config
from src.bot.models.conversation import Conversation, Message
from src.bot.models.user import UserPermission

logger = structlog.get_logger(__name__)

# Request context for CLI-like usage
_CLI_REQUEST_CONTEXT = {"user_id": DEFAULT_CLI_USER}


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

    async def close(self):
        """Close Holmes client."""
        self._tool_calling_llm = None
        self._tool_executor = None
        self._config = None
        self._initialized = False

    def _get_user_permissions(self, user_permissions: List[UserPermission]) -> Dict[str, bool]:
        """Convert user permissions to Holmes feature flags."""
        return {
            "agents": UserPermission.AGENT_EXECUTION in user_permissions,
            "tools": UserPermission.TOOL_USAGE in user_permissions,
            "memory": UserPermission.CONVERSATION_MEMORY in user_permissions,
            "streaming": UserPermission.STREAMING_RESPONSES in user_permissions,
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
    ) -> str | AsyncGenerator[str, None]:
        """
        Send a message to Holmes and get response.

        Args:
            user_id: Telegram user ID
            user_message: User's message text
            conversation: Current conversation history
            user_permissions: User's permissions
            stream: Whether to stream the response

        Returns:
            Response text or async generator for streaming
        """
        await self.initialize()

        # Add user message to conversation
        user_msg = Message(role="user", content=user_message)
        conversation.add_message(user_msg)

        # Convert to Holmes format using build_initial_ask_messages for proper formatting
        holmes_messages = self._convert_messages_to_holmes_format(conversation.messages)

        # Get feature flags from permissions
        features = self._get_user_permissions(user_permissions)

        try:
            if stream and features.get("streaming", False):
                return self._stream_chat(holmes_messages, features, user_id)
            else:
                response_text, llm_result = await self._non_stream_chat_with_memory(holmes_messages, features, user_id)

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
            system_prompt_additions=None,
        ) + messages

    async def _non_stream_chat(
        self, messages: List[Dict[str, Any]], features: Dict[str, bool], user_id: int
    ) -> str:
        """Non-streaming chat with Holmes."""
        loop = asyncio.get_event_loop()

        # Run in executor since ToolCallingLLM.call is sync
        result: LLMResult = await loop.run_in_executor(
            None,
            lambda: self._tool_calling_llm.call(
                messages=messages,
                request_context={**_CLI_REQUEST_CONTEXT, "user_id": str(user_id)},
            ),
        )

        return result.result or "I apologize, but I couldn't generate a response."

    async def _non_stream_chat_with_memory(
        self, messages: List[Dict[str, Any]], features: Dict[str, bool], user_id: int
    ) -> tuple[str, LLMResult]:
        """Non-streaming chat with Holmes, returning full result for memory storage."""
        loop = asyncio.get_event_loop()

        result: LLMResult = await loop.run_in_executor(
            None,
            lambda: self._tool_calling_llm.call(
                messages=messages,
                request_context={**_CLI_REQUEST_CONTEXT, "user_id": str(user_id)},
            ),
        )

        return result.result or "I apologize, but I couldn't generate a response.", result

    async def _stream_chat(
        self, messages: List[Dict[str, Any]], features: Dict[str, bool], user_id: int
    ) -> AsyncGenerator[str, None]:
        """Streaming chat with Holmes."""
        loop = asyncio.get_event_loop()

        def stream_generator():
            return self._tool_calling_llm.call_stream(
                msgs=messages,
                request_context={**_CLI_REQUEST_CONTEXT, "user_id": str(user_id)},
                enable_tool_approval=False,
            )

        stream = await loop.run_in_executor(None, stream_generator)

        # Yield chunks
        for event in stream:
            if event.event.name == "AI_MESSAGE":
                content = event.data.get("content")
                if content:
                    yield content

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

        holmes_messages = self._convert_messages_to_holmes_format(conversation.messages)
        features = self._get_user_permissions(user_permissions)

        loop = asyncio.get_event_loop()
        result: LLMResult = await loop.run_in_executor(
            None,
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