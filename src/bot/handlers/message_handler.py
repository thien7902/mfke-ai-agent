"""Message Handler - Process regular messages through Holmes."""
import asyncio
import logging
import random
import structlog
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from telegram import Update, ForumTopic
from telegram.ext import ContextTypes

from src.bot.services.holmes_service import HolmesService
from holmes.core.tool_calling_llm import LLMInterruptedError as _LLMInterrupted
from src.bot.services.conversation_service import ConversationService
from src.bot.services.permission_service import PermissionService
from src.bot.models.user import UserPermission
from src.bot.models.conversation import Conversation
from src.bot.utils.config import config
from src.bot.utils.telegram_helpers import send_long_message, edit_or_send_long_message

logger = structlog.get_logger(__name__)

# In-memory cache to prevent duplicate topic creation (user_id -> timestamp)
_recent_topic_creation: dict[int, float] = {}
_TOPIC_CREATION_COOLDOWN = 10  # seconds - increased from 5

# In-memory cache to prevent duplicate message processing (message_id -> timestamp)
_processed_messages: dict[int, float] = {}
_MESSAGE_PROCESSING_TTL = 30  # seconds

# Cache for progress topic IDs (main_topic_id -> progress_topic_id)
_progress_topic_cache: dict[tuple, int] = {}  # (chat_id, main_topic_id) -> progress_topic_id

# Per-topic locks to serialize message processing
_topic_locks: dict[tuple, asyncio.Lock] = {}  # (chat_id, topic_id) -> Lock


def _get_topic_lock(chat_id: int, topic_id: int) -> asyncio.Lock:
    """Get or create a lock for a specific topic."""
    key = (chat_id, topic_id)
    if key not in _topic_locks:
        _topic_locks[key] = asyncio.Lock()
    return _topic_locks[key]


# Per-topic message batching. Rapid consecutive messages in a forum topic are
# combined into a single Holmes request after a quiet period (debounce).
@dataclass
class _PendingBatch:
    updates: list = field(default_factory=list)   # Telegram Update objects for reply threading
    texts: list = field(default_factory=list)
    user_ids: list = field(default_factory=list)
    timer: Optional[asyncio.TimerHandle] = None
    started_at: float = 0.0
    cancel_event: Optional[threading.Event] = None  # set only while a dispatch is in flight


# (chat_id, topic_id) -> _PendingBatch
_topic_batches: dict[tuple, _PendingBatch] = {}

_STOPPED_FOOTER = "\n\n🛑 Stopped by user."


def _generate_topic_name(user_message: str, user_id: int) -> str:
    """Generate a topic name from user message: 2-3 word summary + random ID."""
    # Simple keyword extraction - take first few meaningful words
    words = user_message.strip().split()
    # Filter out common words, mentions, keep meaningful words
    meaningful_words = []
    for word in words:
        clean_word = word.strip('@#.,!?').lower()
        if len(clean_word) > 2 and clean_word not in {'the', 'and', 'for', 'you', 'are', 'who', 'what', 'how', 'why', 'when', 'where', 'can', 'please', 'help', 'bot', 'gpt', 'mfke'}:
            meaningful_words.append(clean_word)
            if len(meaningful_words) >= 3:
                break

    if meaningful_words:
        summary = ' '.join(meaningful_words[:3]).title()
    else:
        summary = "New Chat"

    # Add random 4-digit ID to prevent duplication
    random_id = random.randint(1000, 9999)
    return f"💬 {summary} #{random_id}"


class MessageHandler:
    """Handle regular text messages via Holmes."""

    def __init__(
        self,
        holmes_service: HolmesService,
        conversation_service: ConversationService,
        permission_service: PermissionService,
    ):
        self.holmes = holmes_service
        self.conversations = conversation_service
        self.permissions = permission_service

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text message, stickers, and emojis."""
        if not update.message:
            return

        # Extract text content from various message types
        message_text = None
        if update.message.text:
            message_text = update.message.text
        elif update.message.sticker:
            # Handle stickers - use emoji or description
            message_text = update.message.sticker.emoji or f"[Sticker: {update.message.sticker.set_name}]"
        elif update.message.emoji:
            # Handle raw emoji messages (rare, but possible)
            message_text = update.message.emoji
        elif update.message.animation:
            # Handle GIFs/animations
            message_text = f"[Animation: {update.message.animation.file_name or 'GIF'}]"
        elif update.message.photo:
            # Handle photos
            message_text = "[Photo]"
        else:
            # Ignore other message types (voice, video, document, etc.)
            return

        user = update.effective_user
        user_id = user.id
        message_id = update.message.message_id

        # Skip commands (they start with /)
        if message_text.startswith("/"):
            return

        # Deduplication: skip if we've processed this message recently
        current_time = time.time()
        if message_id in _processed_messages:
            elapsed = current_time - _processed_messages[message_id]
            if elapsed < _MESSAGE_PROCESSING_TTL:
                logger.warning("Duplicate message ignored", message_id=message_id, user_id=user_id, elapsed=elapsed)
                return
        _processed_messages[message_id] = current_time

        # Clean up old entries (older than TTL)
        cutoff = current_time - _MESSAGE_PROCESSING_TTL
        keys_to_remove = [k for k, v in _processed_messages.items() if v < cutoff]
        for k in keys_to_remove:
            del _processed_messages[k]

        # Get chat and topic info
        chat = update.effective_chat
        chat_id = chat.id
        topic_id = update.message.message_thread_id or 0

        # Only allow group/forum topic chats - DISABLE private chats
        if chat.type == "private":
            # Private chat - ignore with a message
            await update.message.reply_text(
                "🚫 Private chats are disabled. Please use the bot in a group forum topic.",
                message_thread_id=None
            )
            return
        elif chat.type in ("group", "supergroup") and chat.is_forum:
            # Forum topic - use topic_id
            if not topic_id:
                # Message in general chat, not a topic - create topic if mentioning bot
                await self._handle_general_chat_message(update, context, user, chat_id)
                return
        else:
            # Regular group (non-forum) - ignore
            return

        logger.info("Received message", user_id=user_id, chat_id=chat_id, topic_id=topic_id, text=message_text[:50])

        # Batch per topic: rapid consecutive messages combine into one Holmes request.
        await self._enqueue_for_batch(update, context, chat_id, topic_id, user, message_text)

    async def _enqueue_for_batch(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        topic_id: int,
        user,
        message_text: str,
    ):
        """Accumulate messages per topic and flush after a quiet period (debounce)."""
        key = (chat_id, topic_id)
        now = time.time()
        quiet = config.message_batch_quiet_period_seconds
        max_wait = config.message_batch_max_wait_seconds
        max_messages = config.message_batch_max_messages

        batch = _topic_batches.get(key)
        if batch is None or batch.cancel_event is not None:
            # No pending batch, or the previous one is already dispatching — start fresh.
            # A dispatching batch owns the topic lock, so a new batch will simply queue on it.
            batch = _PendingBatch(updates=[], texts=[], user_ids=[], started_at=now)
            _topic_batches[key] = batch

        batch.updates.append(update)
        batch.texts.append(message_text)
        batch.user_ids.append(user.id)

        # Cancel any previously armed timer (debounce: reset the quiet window).
        if batch.timer is not None:
            batch.timer.cancel()
            batch.timer = None

        force_flush = (
            getattr(config, "test_mode", False)
            or (now - batch.started_at) >= max_wait
            or len(batch.texts) >= max_messages
        )

        if force_flush:
            # Flush synchronously (tests) or when caps are hit — no timer.
            await self._flush_batch(context, chat_id, topic_id)
            return

        # Arm a timer to flush after the quiet period. loop.call_later returns a TimerHandle.
        loop = asyncio.get_event_loop()
        batch.timer = loop.call_later(
            quiet,
            lambda: asyncio.ensure_future(self._flush_batch(context, chat_id, topic_id)),
        )

    async def _flush_batch(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, topic_id: int):
        """Drain the pending batch for a topic and dispatch a single combined Holmes request."""
        key = (chat_id, topic_id)
        batch = _topic_batches.get(key)
        if batch is None or not batch.texts:
            return

        # Snapshot and clear the pending state so new messages start a new batch
        # while this one dispatches under the topic lock.
        updates = list(batch.updates)
        texts = list(batch.texts)
        user_ids = list(batch.user_ids)
        if batch.timer is not None:
            batch.timer.cancel()
        # Mark in-flight by setting cancel_event; /stop checks it.
        cancel_event = batch.cancel_event = threading.Event()

        combined_text = "\n\n".join(texts)
        first_update = updates[0]
        first_user_id = user_ids[0]
        user = first_update.effective_user

        # Acquire the per-topic lock to serialize dispatch across overlapping batches.
        lock = _get_topic_lock(chat_id, topic_id)
        logger.debug("Acquiring topic lock for batch flush", chat_id=chat_id, topic_id=topic_id)
        async with lock:
            logger.debug("Topic lock acquired for batch flush", chat_id=chat_id, topic_id=topic_id, batch_size=len(texts))
            try:
                # Get or create the first user (the one whose update we reply to).
                user_model = await self.permissions.get_or_create_user(
                    telegram_id=first_user_id,
                    username=user.username,
                    first_name=user.first_name,
                    last_name=user.last_name,
                )

                conversation = await self.conversations.get_conversation(
                    first_user_id, topic_id=topic_id, chat_id=chat_id
                )

                user_permissions = user_model.permissions

                await context.bot.send_chat_action(
                    chat_id=chat_id, action="typing", message_thread_id=topic_id or None
                )

                use_streaming = UserPermission.STREAMING_RESPONSES in user_permissions

                if use_streaming:
                    await self._handle_streaming_response(
                        context, first_update, conversation, user_permissions,
                        combined_text, chat_id, topic_id, cancel_event=cancel_event,
                    )
                else:
                    await self._handle_regular_response(
                        context, first_update, conversation, user_permissions,
                        combined_text, chat_id, topic_id, cancel_event=cancel_event,
                    )

                # If the user stopped mid-flight, surface it on the saved conversation.
                if cancel_event.is_set():
                    # Append a marker so future turns know this one was aborted.
                    try:
                        last = conversation.messages[-1] if conversation.messages else None
                        if last and last.role == "assistant" and _STOPPED_FOOTER not in last.content:
                            last.content = (last.content or "") + _STOPPED_FOOTER
                    except Exception:
                        pass
            except Exception as e:
                try:
                    error_str = str(e)
                except Exception:
                    error_str = f"<unserializable error: {type(e).__name__}>"
                logger.error("Error handling batch", chat_id=chat_id, topic_id=topic_id, error=error_str)
                try:
                    await first_update.message.reply_text(
                        "❌ An error occurred while processing your message. Please try again.",
                        message_thread_id=topic_id or None
                    )
                except Exception:
                    pass
            finally:
                # Clear the in-flight batch only if it's still the one we set.
                if _topic_batches.get(key) is batch:
                    _topic_batches.pop(key, None)

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop — cancel a pending batch or an in-flight request for the topic."""
        if not update.message:
            return
        chat = update.effective_chat
        topic_id = update.message.message_thread_id or 0

        # Only meaningful in a forum topic.
        if not (chat and chat.type in ("group", "supergroup") and chat.is_forum and topic_id):
            await update.message.reply_text(
                "ℹ️ /stop only works inside a forum topic.",
                message_thread_id=topic_id or None
            )
            return

        key = (chat.id, topic_id)
        batch = _topic_batches.get(key)

        if batch is None:
            await update.message.reply_text("ℹ️ Nothing to stop.", message_thread_id=topic_id)
            return

        if batch.cancel_event is not None:
            # A dispatch is in flight — signal Holmes to abort between iterations.
            batch.cancel_event.set()
            await update.message.reply_text("🛑 Stopping current request…", message_thread_id=topic_id)
            return

        # Pending batch (timer armed, not yet dispatching) — discard it.
        n = len(batch.texts)
        if batch.timer is not None:
            batch.timer.cancel()
        _topic_batches.pop(key, None)
        await update.message.reply_text(
            f"🛑 Stopped — {n} pending message(s) discarded.",
            message_thread_id=topic_id
        )

    async def handle_topic_created(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle when a new forum topic is created."""
        if not update.message or not update.message.forum_topic_created:
            return

        chat_id = update.effective_chat.id
        topic_id = update.message.message_thread_id
        topic_name = update.message.forum_topic_created.name

        logger.info("Forum topic created", chat_id=chat_id, topic_id=topic_id, name=topic_name)

        # Try to extract user_id from topic name (format: "💬 Username" or "User 123456")
        user_id = None
        if topic_name.startswith("💬 "):
            # Try to find user by name
            username = topic_name[3:].strip()
            # We can't easily get user_id from name, so we'll wait for first message
            pass

        # Just log it for now - the first message in the topic will create the conversation
        await update.message.reply_text(
            f"📝 New topic created: {topic_name}\n"
            f"Send a message here to start chatting with the AI!",
            message_thread_id=topic_id
        )

    async def _get_or_create_progress_topic(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, main_topic_id: int) -> int:
        """Get or create a progress tracking topic for a main conversation topic."""
        cache_key = (chat_id, main_topic_id)

        # Check cache first
        if cache_key in _progress_topic_cache:
            return _progress_topic_cache[cache_key]

        # Try to find existing progress topic
        # Progress topic name format: "📊 Progress: <main_topic_name>"
        try:
            # Get main topic info
            main_topic = await context.bot.get_forum_topic(chat_id=chat_id, message_thread_id=main_topic_id)
            main_topic_name = main_topic.name if main_topic else f"Topic {main_topic_id}"
        except Exception:
            main_topic_name = f"Topic {main_topic_id}"

        progress_topic_name = f"📊 Progress: {main_topic_name}"[:100]

        # Search for existing progress topic
        try:
            topics = await context.bot.get_forum_topics(chat_id=chat_id)
            for topic in topics:
                if topic.name == progress_topic_name:
                    _progress_topic_cache[cache_key] = topic.message_thread_id
                    return topic.message_thread_id
        except Exception:
            pass

        # Create new progress topic
        try:
            progress_topic = await context.bot.create_forum_topic(
                chat_id=chat_id,
                name=progress_topic_name,
                icon_color=0xFFA500  # Orange color for progress
            )
            progress_topic_id = progress_topic.message_thread_id
            _progress_topic_cache[cache_key] = progress_topic_id

            # Send initial message in progress topic
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"📊 **Progress Tracking for: {main_topic_name}**\n\n"
                    f"Real-time investigation status will appear here.\n"
                    f"Main conversation: #{main_topic_id}"
                ),
                message_thread_id=progress_topic_id,
                parse_mode="Markdown"
            )

            logger.info("Created progress topic",
                       chat_id=chat_id,
                       main_topic_id=main_topic_id,
                       progress_topic_id=progress_topic_id)

            return progress_topic_id
        except Exception as e:
            logger.error("Failed to create progress topic",
                        chat_id=chat_id,
                        main_topic_id=main_topic_id,
                        error=str(e))
            return 0  # Return 0 to indicate no progress topic

    def _get_progress_topic_id(self, chat_id: int, main_topic_id: int) -> int:
        """Get cached progress topic ID."""
        return _progress_topic_cache.get((chat_id, main_topic_id), 0)

    async def _handle_general_chat_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user, chat_id: int):
        """Handle messages in general chat (not in a topic) - create a new topic for each mention."""
        user_message = update.message.text
        bot_username = context.bot.username or "mfke_gpt_bot"

        # Only create topic if user mentions the bot (@mfke_gpt_bot)
        if f"@{bot_username}" not in user_message:
            # Not mentioning bot - ignore silently in general chat
            return

        # Check cooldown to prevent duplicate topic creation
        user_id = user.id
        current_time = time.time()
        if user_id in _recent_topic_creation:
            elapsed = current_time - _recent_topic_creation[user_id]
            if elapsed < _TOPIC_CREATION_COOLDOWN:
                logger.warning("Topic creation cooldown active", user_id=user_id, elapsed=elapsed)
                await update.message.reply_text(
                    f"⏳ Please wait a moment before creating another topic.",
                    message_thread_id=None
                )
                return

        # Remove the mention from the message for processing
        clean_message = user_message.replace(f"@{bot_username}", "").strip()

        # Mark topic creation time
        _recent_topic_creation[user_id] = current_time

        # Create a NEW forum topic for each mention (don't reuse existing topics)
        try:
            topic_name = _generate_topic_name(clean_message or "Hello", user.id)
            forum_topic = await context.bot.create_forum_topic(
                chat_id=chat_id,
                name=topic_name[:100],  # Max 100 chars
                icon_color=0x6FB9F0  # Blue color
            )

            topic_id = forum_topic.message_thread_id

            # Send welcome message in the new topic
            welcome_msg = await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"👋 Welcome {user.first_name or 'there'}! This is your new conversation topic.\n\n"
                    f"🤖 I'm ready to help. Just send me a message here!\n\n"
                    f"💡 Use /clear to reset conversation history."
                ),
                message_thread_id=topic_id
            )

            logger.info("Created new topic for user", user_id=user.id, topic_id=topic_id, chat_id=chat_id, topic_name=topic_name)

            # Process the user's message in the new topic by creating a synthetic update
            # We'll reply directly in the new topic instead of re-processing through handle_message
            # Get or create user
            user_model = await self.permissions.get_or_create_user(
                telegram_id=user.id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
            )

            # Get conversation for the new topic
            conversation = await self.conversations.get_conversation(
                user.id, topic_id=topic_id, chat_id=chat_id
            )

            # Get user permissions
            user_permissions = user_model.permissions

            # Send typing indicator in the new topic
            await context.bot.send_chat_action(
                chat_id=chat_id, action="typing", message_thread_id=topic_id
            )

            # Check if user has streaming permission
            # Force non-streaming if tool_approval is enabled (approval requires non-streaming)
            use_streaming = (UserPermission.STREAMING_RESPONSES in user_permissions and
                            UserPermission.TOOL_APPROVAL not in user_permissions)

            # Use clean message (without @mention) for processing
            process_message = clean_message if clean_message else "Hello"

            if use_streaming:
                await self._handle_streaming_response_in_topic(
                    context, conversation, user_permissions, process_message, chat_id, topic_id, user.id
                )
            else:
                await self._handle_regular_response_in_topic(
                    context, conversation, user_permissions, process_message, chat_id, topic_id, user.id
                )

        except Exception as e:
            logger.error("Failed to create forum topic", user_id=user.id, error=str(e))
            await update.message.reply_text(
                "❌ Failed to create conversation topic. Please contact admin.",
                message_thread_id=None
            )

    async def _handle_regular_response(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        update: Update,
        conversation: Conversation,
        user_permissions: list,
        message_text: str,
        chat_id: int,
        topic_id: int,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Handle non-streaming response."""
        response = await self.holmes.chat(
            user_id=update.effective_user.id,
            user_message=message_text,
            conversation=conversation,
            user_permissions=user_permissions,
            stream=False,
            telegram_context=context,
            chat_id=chat_id,
            topic_id=topic_id,
            cancel_event=cancel_event,
        )

        if cancel_event is not None and cancel_event.is_set():
            response = (response or "") + _STOPPED_FOOTER

        # Save conversation
        await self.conversations.save_conversation(conversation)

        # Send response (split if too long)
        try:
            await send_long_message(
                context.bot,
                chat_id,
                response,
                message_thread_id=topic_id or None
            )
        except Exception as e:
            logger.warning("Failed to deliver response", user_id=update.effective_user.id, error=str(e))
            await self._notify_delivery_failure(context.bot, chat_id, topic_id)

    async def _handle_streaming_response(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        update: Update,
        conversation: Conversation,
        user_permissions: list,
        message_text: str,
        chat_id: int,
        topic_id: int,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Handle streaming response with clean task progress updates."""
        import asyncio

        # Send initial empty message to edit
        message = await update.message.reply_text("🤔 Thinking...", message_thread_id=topic_id or None)

        # IMMEDIATELY send a status message so user sees something
        status_message = await update.message.reply_text(
            "📍 **Status**: 🔄 Working...",
            message_thread_id=topic_id or None,
            parse_mode="Markdown"
        )

        full_response = ""
        task_message = None
        current_status = "🔄 Working..."
        last_tool = None
        stream_completed = False
        try:
            async_gen = await self.holmes.chat(
                user_id=update.effective_user.id,
                user_message=message_text,
                conversation=conversation,
                user_permissions=user_permissions,
                stream=True,
                telegram_context=context,
                chat_id=chat_id,
                topic_id=topic_id,
                cancel_event=cancel_event,
            )

            # Add timeout to prevent hanging indefinitely
            async def process_stream():
                nonlocal full_response, task_message, current_status, last_tool, stream_completed
                async for event in async_gen:
                    event_type = event.get("type", "unknown")

                    if event_type == "content":
                        full_response += event.get("content", "")
                        # Edit message every few chunks to avoid rate limits
                        if len(full_response) % 100 == 0:
                            try:
                                await message.edit_text(full_response + "▌")
                            except Exception:
                                pass  # Ignore edit conflicts

                    elif event_type == "task_update":
                        # Display investigation task list
                        tasks = event.get("tasks", [])
                        task_text = self._format_task_list(tasks)
                        if task_text:
                            if task_message is None:
                                task_message = await update.message.reply_text(
                                    task_text,
                                    message_thread_id=topic_id or None,
                                    parse_mode="Markdown"
                                )
                            else:
                                try:
                                    await task_message.edit_text(task_text, parse_mode="Markdown")
                                except Exception:
                                    pass
                            # Update status based on current task
                            current_status = self._get_status_from_tasks(tasks)
                            try:
                                await status_message.edit_text(f"📍 **Status**: {current_status}", parse_mode="Markdown")
                            except Exception:
                                pass

                    elif event_type == "tool_call":
                        tool_name = event.get("tool_name", "unknown")
                        last_tool = tool_name
                        # Update status with current tool (clean, no params)
                        friendly_name = self._get_friendly_tool_name(tool_name)
                        current_status = f"🔧 Using {friendly_name}..."
                        try:
                            await status_message.edit_text(
                                f"📍 **Status**: {current_status}",
                                parse_mode="Markdown"
                            )
                        except Exception:
                            pass

                    elif event_type == "tool_result":
                        tool_name = event.get("tool_name", "unknown")
                        if tool_name == last_tool:
                            friendly_name = self._get_friendly_tool_name(tool_name)
                            current_status = f"✅ Completed {friendly_name}"
                            try:
                                await status_message.edit_text(
                                    f"📍 **Status**: {current_status}",
                                    parse_mode="Markdown"
                                )
                            except Exception:
                                pass

                stream_completed = True

            # Wait for stream with timeout (240 seconds max)
            try:
                await asyncio.wait_for(process_stream(), timeout=240.0)
            except asyncio.TimeoutError:
                logger.warning("Streaming response timed out after 240 seconds", user_id=update.effective_user.id)
                await status_message.edit_text("📍 **Status**: ⏱️ Timed out", parse_mode="Markdown")
                full_response += "\n\n⚠️ Response timed out. Please try again."
            except _LLMInterrupted:
                # User hit /stop mid-stream: Holmes aborted. Treat as graceful stop.
                logger.info("Stream interrupted by user", user_id=update.effective_user.id)
                if cancel_event is not None:
                    cancel_event.set()

            # Final edit of main response
            if full_response:
                if cancel_event is not None and cancel_event.is_set() and _STOPPED_FOOTER not in full_response:
                    full_response += _STOPPED_FOOTER
                try:
                    await message.edit_text(full_response)
                except Exception as e:
                    logger.warning("Failed to edit final message", error=str(e))
                    # Too long — split into multiple messages
                    try:
                        await send_long_message(
                            context.bot,
                            chat_id,
                            full_response,
                            message_thread_id=topic_id or None
                        )
                    except Exception as e2:
                        logger.warning("Failed to deliver streamed response", user_id=update.effective_user.id, error=str(e2))
                        await self._notify_delivery_failure(context.bot, chat_id, topic_id)

            # Final status
            if stream_completed and not (cancel_event is not None and cancel_event.is_set()):
                try:
                    await status_message.edit_text("📍 **Status**: ✅ Done", parse_mode="Markdown")
                except Exception:
                    pass
            elif cancel_event is not None and cancel_event.is_set():
                try:
                    await status_message.edit_text("📍 **Status**: 🛑 Stopped", parse_mode="Markdown")
                except Exception:
                    pass

        except Exception as e:
            logger.error("Streaming error", error=str(e), exc_info=True)
            try:
                await message.edit_text(full_response + "\n\n⚠️ Stream interrupted")
            except Exception:
                pass

        # Save conversation
        await self.conversations.save_conversation(conversation)

    async def _notify_delivery_failure(self, bot, chat_id: int, topic_id: int):
        """Best-effort notice when a generated response could not be delivered to Telegram."""
        try:
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ The response was generated but couldn't be delivered. Please try again.",
                message_thread_id=topic_id or None,
            )
        except Exception:
            pass

    def _get_friendly_tool_name(self, tool_name: str) -> str:
        """Convert technical tool names to user-friendly names."""
        friendly_names = {
            'get_shoot_kubeconfig': 'Getting cluster access',
            'kubectl_get': 'Querying Kubernetes resources',
            'kubectl_exec': 'Running commands in cluster',
            'search_contacts': 'Searching contacts',
            'get_cluster_info': 'Fetching cluster info',
            'TodoWrite': 'Planning investigation',
            'add_memory': 'Saving to memory',
            'bash': 'Running shell command',
            'read_file': 'Reading file',
            'write_file': 'Writing file',
        }
        return friendly_names.get(tool_name, tool_name.replace('_', ' ').title())

    def _get_status_from_tasks(self, tasks: list) -> str:
        """Generate status text from task list."""
        if not tasks:
            return "🤔 Starting investigation..."

        in_progress = [t for t in tasks if t.get('status', '').lower() in ('in_progress', '~', 'pending')]
        completed = [t for t in tasks if t.get('status', '').lower() in ('completed', '✓', 'done')]

        if in_progress:
            task = in_progress[0]
            content = task.get('Content', task.get('content', 'Working...'))
            return f"🔄 {content}"
        elif completed and len(completed) == len(tasks):
            return "✅ Investigation complete"
        else:
            return f"📋 {len(completed)}/{len(tasks)} tasks done"

    def _format_task_list(self, tasks: list) -> str:
        """Format task list for Telegram display."""
        if not tasks:
            return ""

        lines = ["📋 **Investigation Progress:**", ""]
        for task in tasks:
            task_id = task.get("ID", task.get("id", "?"))
            content = task.get("Content", task.get("content", ""))
            status = task.get("Status", task.get("status", ""))

            # Status icons
            if status in ("completed", "✓", "done"):
                icon = "✅"
            elif status in ("in_progress", "~", "pending"):
                icon = "🔄"
            else:
                icon = "⏳"

            lines.append(f"{icon} **Task {task_id}**: {content} `_({status})_`")

        return "\n".join(lines)

    async def _handle_regular_response_in_topic(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        conversation: Conversation,
        user_permissions: list,
        message_text: str,
        chat_id: int,
        topic_id: int,
        user_id: int,
    ):
        """Handle non-streaming response in a specific topic (without update object)."""
        response = await self.holmes.chat(
            user_id=user_id,
            user_message=message_text,
            conversation=conversation,
            user_permissions=user_permissions,
            stream=False,
            telegram_context=context,
            chat_id=chat_id,
            topic_id=topic_id,
        )

        # Save conversation
        await self.conversations.save_conversation(conversation)

        # Send response in the topic (split if too long)
        try:
            await send_long_message(
                context.bot,
                chat_id,
                response,
                message_thread_id=topic_id
            )
        except Exception as e:
            logger.warning("Failed to deliver response in topic", user_id=user_id, error=str(e))
            await self._notify_delivery_failure(context.bot, chat_id, topic_id)

    async def _handle_streaming_response_in_topic(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        conversation: Conversation,
        user_permissions: list,
        message_text: str,
        chat_id: int,
        topic_id: int,
        user_id: int,
    ):
        """Handle streaming response in a specific topic (without update object)."""
        import asyncio

        # Send initial empty message to edit
        message = await context.bot.send_message(
            chat_id=chat_id,
            text="🤔 Thinking...",
            message_thread_id=topic_id
        )

        # IMMEDIATELY send a status message so user sees something
        status_message = await context.bot.send_message(
            chat_id=chat_id,
            text="📍 **Status**: 🔄 Working...",
            message_thread_id=topic_id,
            parse_mode="Markdown"
        )

        full_response = ""
        task_message = None
        current_status = "🔄 Working..."
        last_tool = None
        stream_completed = False
        try:
            async_gen = await self.holmes.chat(
                user_id=user_id,
                user_message=message_text,
                conversation=conversation,
                user_permissions=user_permissions,
                stream=True,
                telegram_context=context,
                chat_id=chat_id,
                topic_id=topic_id,
            )

            # Add timeout to prevent hanging indefinitely
            async def process_stream():
                nonlocal full_response, task_message, current_status, last_tool, stream_completed
                async for event in async_gen:
                    event_type = event.get("type", "unknown")

                    if event_type == "content":
                        full_response += event.get("content", "")
                        # Edit message every few chunks to avoid rate limits
                        if len(full_response) % 100 == 0:
                            try:
                                await message.edit_text(full_response + "▌")
                            except Exception:
                                pass  # Ignore edit conflicts

                    elif event_type == "task_update":
                        # Display investigation task list
                        tasks = event.get("tasks", [])
                        task_text = self._format_task_list(tasks)
                        if task_text:
                            if task_message is None:
                                task_message = await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=task_text,
                                    message_thread_id=topic_id,
                                    parse_mode="Markdown"
                                )
                            else:
                                try:
                                    await task_message.edit_text(task_text, parse_mode="Markdown")
                                except Exception:
                                    pass
                            # Update status based on current task
                            current_status = self._get_status_from_tasks(tasks)
                            try:
                                await status_message.edit_text(f"📍 **Status**: {current_status}", parse_mode="Markdown")
                            except Exception:
                                pass

                    elif event_type == "tool_call":
                        tool_name = event.get("tool_name", "unknown")
                        last_tool = tool_name
                        # Update status with current tool (clean, no params)
                        friendly_name = self._get_friendly_tool_name(tool_name)
                        current_status = f"🔧 Using {friendly_name}..."
                        try:
                            await status_message.edit_text(
                                f"📍 **Status**: {current_status}",
                                parse_mode="Markdown"
                            )
                        except Exception:
                            pass

                    elif event_type == "tool_result":
                        tool_name = event.get("tool_name", "unknown")
                        if tool_name == last_tool:
                            friendly_name = self._get_friendly_tool_name(tool_name)
                            current_status = f"✅ Completed {friendly_name}"
                            try:
                                await status_message.edit_text(
                                    f"📍 **Status**: {current_status}",
                                    parse_mode="Markdown"
                                )
                            except Exception:
                                pass

                stream_completed = True

            # Wait for stream with timeout (240 seconds max)
            try:
                await asyncio.wait_for(process_stream(), timeout=240.0)
            except asyncio.TimeoutError:
                logger.warning("Streaming response timed out after 240 seconds", user_id=user_id)
                await status_message.edit_text("📍 **Status**: ⏱️ Timed out", parse_mode="Markdown")
                full_response += "\n\n⚠️ Response timed out. Please try again."

            # Final edit of main response
            if full_response:
                try:
                    await message.edit_text(full_response)
                except Exception as e:
                    logger.warning("Failed to edit final message", error=str(e))
                    # Too long — split into multiple messages
                    try:
                        await send_long_message(
                            context.bot,
                            chat_id,
                            full_response,
                            message_thread_id=topic_id
                        )
                    except Exception as e2:
                        logger.warning("Failed to deliver streamed response in topic", user_id=user_id, error=str(e2))
                        await self._notify_delivery_failure(context.bot, chat_id, topic_id)

            # Final status
            if stream_completed:
                try:
                    await status_message.edit_text("📍 **Status**: ✅ Done", parse_mode="Markdown")
                except Exception:
                    pass

        except Exception as e:
            logger.error("Streaming error", error=str(e), exc_info=True)
            try:
                await message.edit_text(full_response + "\n\n⚠️ Stream interrupted")
            except Exception:
                pass

        # Save conversation
        await self.conversations.save_conversation(conversation)