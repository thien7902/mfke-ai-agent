"""Message Handler - Process regular messages through Holmes."""
import asyncio
import base64
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
from src.bot.utils.file_extractor import (
    extract_text_from_file,
    extract_text_from_image,
    IMAGE_SENTINEL,
)
from src.bot.services import vision_service

logger = structlog.get_logger(__name__)


def _build_data_url(path: str, mime: str) -> str:
    """Read a file from disk and return a `data:<mime>;base64,...` URL suitable
    for `image_url` content blocks sent to OpenAI / Anthropic vision endpoints.
    """
    import os
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

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
    images: list = field(default_factory=list)    # OpenAI vision image dicts to attach
    timer: Optional[asyncio.TimerHandle] = None
    started_at: float = 0.0
    cancel_event: Optional[threading.Event] = None  # set only while a dispatch is in flight


# (chat_id, topic_id) -> _PendingBatch
_topic_batches: dict[tuple, _PendingBatch] = {}

_STOPPED_FOOTER = "\n\n🛑 Stopped by user."


# Per-topic active investigation registry. Streaming/regular handlers update
# status as they process events; /ping reads from here to report progress.
@dataclass
class _ActiveInvestigation:
    status_text: str = "🔄 Working..."
    started_at: float = 0.0
    tasks: list = field(default_factory=list)
    last_tool: Optional[str] = None
    response_preview: str = ""
    last_activity: str = ""


# (chat_id, topic_id) -> _ActiveInvestigation
_active_investigations: dict[tuple, _ActiveInvestigation] = {}


def _set_investigation_status(
    chat_id: int,
    topic_id: int,
    status_text: Optional[str] = None,
    tasks: Optional[list] = None,
    last_tool: Optional[str] = None,
    response_preview: Optional[str] = None,
    last_activity: Optional[str] = None,
):
    """Record latest status for the active investigation in this topic.

    Only fields that are not None are updated, so callers can refresh just
    the task list (or just the status text) without clobbering the other.
    last_activity overwrites the previous value — /ping shows only the
    most recent tool step, not a list.
    """
    key = (chat_id, topic_id)
    inv = _active_investigations.get(key)
    if inv is None:
        inv = _ActiveInvestigation(started_at=time.time())
        _active_investigations[key] = inv
    if status_text is not None:
        inv.status_text = status_text
    if tasks is not None:
        inv.tasks = tasks
    if last_tool is not None:
        inv.last_tool = last_tool
    if response_preview is not None:
        inv.response_preview = response_preview
    if last_activity is not None:
        inv.last_activity = last_activity


def _clear_investigation(chat_id: int, topic_id: int):
    """Remove the investigation entry once the request completes."""
    _active_investigations.pop((chat_id, topic_id), None)


def _get_investigation(chat_id: int, topic_id: int) -> Optional[_ActiveInvestigation]:
    return _active_investigations.get((chat_id, topic_id))


def _user_display_name(user) -> Optional[str]:
    """Build a concise display name from a Telegram user object.

    Prefers @username (always present in chat mentions) and falls back to
    first_name + last_name. Returns None when the user has neither — the
    caller treats None as "don't inject" so the LLM isn't told 'User: None'.
    """
    if user is None:
        return None
    parts = []
    if getattr(user, "username", None):
        parts.append(f"@{user.username}")
    if getattr(user, "first_name", None):
        parts.append(user.first_name)
    if getattr(user, "last_name", None):
        parts.append(user.last_name)
    return " ".join(parts) if parts else None


def _user_model_display_name(user_model) -> Optional[str]:
    """Same as _user_display_name but for the bot's own User model from
    permission_service.get_or_create_user."""
    if user_model is None:
        return None
    parts = []
    if getattr(user_model, "username", None):
        parts.append(f"@{user_model.username}")
    if getattr(user_model, "first_name", None):
        parts.append(user_model.first_name)
    if getattr(user_model, "last_name", None):
        parts.append(user_model.last_name)
    return " ".join(parts) if parts else None


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

        await self._enqueue_text(update, context, message_text)

    async def _enqueue_text(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        message_text: str,
        force_flush: bool = False,
        images: Optional[list] = None,
    ):
        """Validate chat context and enqueue a text turn into the per-topic batch.

        Shared by `handle_message` (plain text/sticker/etc.) and `handle_file`
        (file uploads whose contents have been extracted to text). Set
        `force_flush=True` to skip the debounce quiet window and dispatch
        immediately (used for file/image uploads). `images` (optional list of
        OpenAI vision image dicts) attaches the image to the user message so
        a multimodal main LLM can read it directly; text-only main LLMs should
        not pass `images` (they should extract text first).
        """
        user = update.effective_user
        user_id = user.id

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
        await self._enqueue_for_batch(
            update, context, chat_id, topic_id, user, message_text,
            force_flush=force_flush, images=images,
        )

    async def handle_file(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle a document or photo upload: download, extract text, enqueue as a text turn.

        Images go through a vision-model pre-step (if VISION_MODEL is set) before
        falling back to OCR. All other supported types are parsed directly.
        The extracted text is combined with the user's caption and handed to
        `_enqueue_text`, which reuses the normal batching/streaming/approval flow.
        """
        if not update.message:
            return

        # Deduplication: same logic as handle_message — /stop and /ping may
        # otherwise double-process the same upload.
        message_id = update.message.message_id
        current_time = time.time()
        if message_id in _processed_messages:
            elapsed = current_time - _processed_messages[message_id]
            if elapsed < _MESSAGE_PROCESSING_TTL:
                logger.warning("Duplicate file message ignored", message_id=message_id, elapsed=elapsed)
                return
        _processed_messages[message_id] = current_time
        cutoff = current_time - _MESSAGE_PROCESSING_TTL
        for k in [k for k, v in _processed_messages.items() if v < cutoff]:
            del _processed_messages[k]

        # Resolve attachment + caption. Document takes precedence; fall back to
        # the largest photo size. Photos arrive as a list of decreasing resolutions.
        doc = update.message.document
        if doc:
            file_id = doc.file_id
            filename = doc.file_name or "upload"
            file_size = doc.file_size or 0
            mime = doc.mime_type
        elif update.message.photo:
            # Largest size is last in the array
            photo = update.message.photo[-1]
            file_id = photo.file_id
            filename = f"photo_{photo.file_unique_id}.jpg"
            file_size = photo.file_size or 0
            mime = "image/jpeg"
        else:
            return

        caption = update.message.caption or ""

        # Reject very large raw uploads (Telegram bot limit is 20 MB) — extracting
        # text from >20 MB files would likely blow the LLM context anyway.
        if file_size and file_size > 20 * 1024 * 1024:
            await update.message.reply_text(
                "❌ File too large. Please keep uploads under 20 MB.",
                message_thread_id=update.message.message_thread_id or None,
            )
            return

        # Ensure tmp dir exists
        import os as _os
        import uuid as _uuid
        _os.makedirs(config.file_upload_tmp_dir, exist_ok=True)
        ext = _os.path.splitext(filename)[1]
        tmp_path = _os.path.join(config.file_upload_tmp_dir, f"{_uuid.uuid4().hex}{ext}")

        try:
            tg_file = await context.bot.get_file(file_id)
            await tg_file.download_to_drive(tmp_path)

            from pathlib import Path as _Path

            # Fast path: if the main Holmes LLM accepts image_url content
            # blocks (gpt-4o, claude-3-5-sonnet, qwen-vl-max, ...), attach the
            # raw image bytes directly to the user message and let the main
            # model read it — skip the vision pre-step / OCR entirely. Falls
            # through to the extraction path when the flag is off or the file
            # isn't an image.
            extract_preview = await asyncio.to_thread(extract_text_from_file, _Path(tmp_path), mime)
            is_image = extract_preview == IMAGE_SENTINEL

            if is_image and config.main_model_supports_images:
                try:
                    data_url = await asyncio.to_thread(_build_data_url, tmp_path, mime or "image/jpeg")
                except Exception as e:
                    logger.warning("Failed to build data URL for multimodal path; falling back to extraction", error=str(e))
                    data_url = None

                if data_url:
                    # Caption becomes the user-visible text; image is attached
                    # so the main model can inspect it directly. No "--- File ---"
                    # wrapper needed since the model sees the image itself.
                    combined = caption or "(_image uploaded — please inspect)"
                    await self._enqueue_text(
                        update, context, combined, force_flush=True,
                        images=[{"url": data_url}],
                    )
                    return
                # else: fall through to extraction

            extracted = extract_preview
            if is_image:
                # Vision pre-step if configured, else OCR directly.
                if config.vision_model:
                    try:
                        vision_text = await asyncio.to_thread(vision_service.describe_image, _Path(tmp_path))
                    except Exception as e:
                        logger.warning("Vision pre-step raised; falling back to OCR", error=str(e))
                        vision_text = ""
                    if vision_text:
                        extracted = vision_text
                    else:
                        extracted = await asyncio.to_thread(extract_text_from_image, _Path(tmp_path))
                else:
                    extracted = await asyncio.to_thread(extract_text_from_image, _Path(tmp_path))

            if not extracted or not extracted.strip():
                await update.message.reply_text(
                    f"❌ Couldn't extract text from `{filename}`. supported: txt/code, pdf, docx, xlsx, png/jpg/webp (image OCR or vision).",
                    message_thread_id=update.message.message_thread_id or None,
                    parse_mode="Markdown",
                )
                return

            combined = f"{caption}\n\n--- File: {filename} ---\n{extracted}".strip() if caption else f"--- File: {filename} ---\n{extracted}"
            # File uploads skip the debounce quiet window — extracted text is
            # already a complete turn; no benefit to waiting for more messages
            # to batch together.
            await self._enqueue_text(update, context, combined, force_flush=True)
        except Exception as e:
            logger.error("File upload handling failed", message_id=message_id, filename=filename, error=str(e))
            try:
                await update.message.reply_text(
                    "❌ Failed to process the uploaded file. Please try again.",
                    message_thread_id=update.message.message_thread_id or None,
                )
            except Exception:
                pass
        finally:
            try:
                if _os.path.exists(tmp_path):
                    _os.remove(tmp_path)
            except Exception:
                pass

    async def _enqueue_for_batch(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        topic_id: int,
        user,
        message_text: str,
        force_flush: bool = False,
        images: Optional[list] = None,
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
        if images:
            batch.images.extend(images)

        # Cancel any previously armed timer (debounce: reset the quiet window).
        if batch.timer is not None:
            batch.timer.cancel()
            batch.timer = None

        # File/image uploads skip the debounce quiet window — extracted text
        # is already a complete turn; no benefit to waiting for more messages.
        if force_flush:
            await self._flush_batch(context, chat_id, topic_id)
            return

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
        images = list(batch.images) if batch.images else []
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
                _set_investigation_status(chat_id, topic_id, "🔄 Working...")
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
                        images=images,
                    )
                else:
                    await self._handle_regular_response(
                        context, first_update, conversation, user_permissions,
                        combined_text, chat_id, topic_id, cancel_event=cancel_event,
                        images=images,
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
                _clear_investigation(chat_id, topic_id)

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

    async def ping_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ping — report current agent status in this topic."""
        if not update.message:
            return
        chat = update.effective_chat
        topic_id = update.message.message_thread_id or 0

        # /ping only meaningful inside a forum topic (matching /stop).
        if not (chat and chat.type in ("group", "supergroup") and chat.is_forum and topic_id):
            await update.message.reply_text(
                "ℹ️ /ping chỉ hoạt động trong forum topic.",
                message_thread_id=topic_id or None,
            )
            return

        inv = _get_investigation(chat.id, topic_id)
        if inv is None:
            # Maybe a batch is queued but not yet dispatched.
            batch = _topic_batches.get((chat.id, topic_id))
            if batch is not None and batch.cancel_event is None:
                n = len(batch.texts)
                await update.message.reply_text(
                    f"⏳ Queued — {n} message(s) chờ batch, chưa dispatch.",
                    message_thread_id=topic_id,
                )
            else:
                await update.message.reply_text(
                    "💤 Không có investigation nào đang chạy trong topic này.",
                    message_thread_id=topic_id,
                )
            return

        elapsed = int(time.time() - inv.started_at)
        lines = [
            "📍 Agent Status",
            "",
            f"Trạng thái: {inv.status_text}",
            f"Thời gian: ⏱ {elapsed}s",
        ]
        if inv.last_activity:
            lines.append(f"Hoạt động cuối: {inv.last_activity}")
        if inv.tasks:
            lines.append("")
            task_lines = self._format_task_list(inv.tasks)
            # _format_task_list uses Markdown (bold/italic); strip those markers
            # for the ping reply since status_text may contain underscores that
            # would otherwise break Markdown parsing.
            task_lines = (task_lines
                           .replace("**", "")
                           .replace("`", ""))
            lines.append(task_lines)
        lines.append("")
        lines.append("📝 Latest LLM response:")
        if inv.response_preview:
            preview = inv.response_preview.strip()
            if len(preview) > 800:
                preview = preview[-800:]
            # When preview is the tail of a longer response, it usually starts
            # mid-sentence (the head was truncated). Drop everything up to the
            # first sentence terminator so the first bullet is a full sentence,
            # then split the rest.
            import re as _re
            first_term = _re.search(r'[.?!]\s+', preview)
            if first_term and first_term.start() < 200:
                preview = preview[first_term.end():]
            sentences = _re.split(r'(?<=[.?!])\s+', preview)
            for s in sentences:
                s = s.strip()
                if s:
                    lines.append(f"• {s}")
        else:
            lines.append("(chưa có — agent đang gọi tools, LLM chưa emit text)")

        await update.message.reply_text(
            "\n".join(lines),
            message_thread_id=topic_id,
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
                    context, conversation, user_permissions, process_message, chat_id, topic_id, user.id,
                    user_display_name=_user_display_name(user),
                )
            else:
                await self._handle_regular_response_in_topic(
                    context, conversation, user_permissions, process_message, chat_id, topic_id, user.id,
                    user_display_name=_user_display_name(user),
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
        images: Optional[list] = None,
    ):
        """Handle non-streaming response.

        Internally runs Holmes in stream mode so we can update the /ping
        registry with the latest LLM content as it arrives, but delivers only
        the final assembled text to the user — no typing indicator or
        status-message editing like the visible streaming path.
        """
        _set_investigation_status(chat_id, topic_id, "🔄 Working...")
        response = ""
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
                images=images,
                user_display_name=_user_display_name(update.effective_user),
            )
            async for event in async_gen:
                event_type = event.get("type", "unknown")
                if event_type == "content":
                    chunk = event.get("content", "")
                    response += chunk
                    _set_investigation_status(
                        chat_id, topic_id,
                        response_preview=response[-300:],
                    )
                elif event_type == "task_update":
                    _set_investigation_status(
                        chat_id, topic_id, None, tasks=event.get("tasks", []),
                    )
                elif event_type == "tool_call":
                    tool_name = event.get("tool_name", "unknown")
                    friendly = self._get_friendly_tool_name(tool_name)
                    _set_investigation_status(
                        chat_id, topic_id,
                        status_text=f"🔧 Using {friendly}...",
                        last_tool=tool_name,
                        last_activity=self._format_activity(tool_name, friendly, event, done=False),
                    )
                elif event_type == "tool_result":
                    tool_name = event.get("tool_name", "unknown")
                    friendly = self._get_friendly_tool_name(tool_name)
                    _set_investigation_status(
                        chat_id, topic_id,
                        status_text=f"✅ Completed {friendly}",
                        last_activity=self._format_activity(tool_name, friendly, event, done=True),
                    )
        except _LLMInterrupted:
            # User stopped mid-call — keep what we have so far.
            if cancel_event is not None:
                cancel_event.set()

        if cancel_event is not None and cancel_event.is_set():
            response = (response or "") + _STOPPED_FOOTER

        _clear_investigation(chat_id, topic_id)

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
        images: Optional[list] = None,
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
        current_tasks: list = []
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
                images=images,
                user_display_name=_user_display_name(update.effective_user),
            )
            _set_investigation_status(chat_id, topic_id, current_status, tasks=current_tasks, last_tool=last_tool)

            # Add timeout to prevent hanging indefinitely
            async def process_stream():
                nonlocal full_response, task_message, current_status, last_tool, current_tasks, stream_completed
                async for event in async_gen:
                    event_type = event.get("type", "unknown")

                    if event_type == "content":
                        chunk = event.get("content", "")
                        full_response += chunk
                        _set_investigation_status(
                            chat_id, topic_id,
                            response_preview=full_response[-300:],
                        )
                        # Edit message every few chunks to avoid rate limits
                        if len(full_response) % 100 == 0:
                            try:
                                await message.edit_text(full_response + "▌")
                            except Exception:
                                pass  # Ignore edit conflicts

                    elif event_type == "task_update":
                        # Display investigation task list
                        tasks = event.get("tasks", [])
                        current_tasks = tasks
                        _set_investigation_status(chat_id, topic_id, None, tasks=tasks)
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
                            _set_investigation_status(chat_id, topic_id, current_status)
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
                        _set_investigation_status(
                            chat_id, topic_id, current_status, last_tool=tool_name,
                            last_activity=self._format_activity(tool_name, friendly_name, event, done=False),
                        )
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
                            _set_investigation_status(
                                chat_id, topic_id, current_status,
                                last_activity=self._format_activity(tool_name, friendly_name, event, done=True),
                            )
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

        _clear_investigation(chat_id, topic_id)

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

    def _extract_tool_command(self, event: dict) -> str:
        """Extract a short human-readable command string from a tool event.

        Holmes START_TOOL emits only `{tool_name, id}` — no params. The real
        params arrive on the TOOL_RESULT event via ToolCallResult.to_client_dict(),
        which includes a top-level `description` (the tool's rendered one-liner,
        e.g. "kubectl get shoot -A -o json"), plus `result.invocation` (the
        actual executed command for bash/kubectl tools) and `result.params`.
        Returns "" if nothing useful is available.
        """
        # Top-level `description` from ToolCallResult.to_client_dict() — Holmes
        # sets this to tool.get_parameterized_one_liner(tool_params), which for
        # bash/kubectl is the full command string.
        description = event.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()

        result = event.get("result")
        if isinstance(result, dict):
            invocation = result.get("invocation")
            if isinstance(invocation, str) and invocation.strip():
                return invocation.strip()
            params = result.get("params")
            if isinstance(params, dict):
                for key in ("command", "query", "url", "path", "name", "expression"):
                    val = params.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                import json as _json
                try:
                    return _json.dumps(params, ensure_ascii=False)[:200]
                except Exception:
                    return ""
        return ""

    def _format_activity(self, tool_name: str, friendly: str, event: dict, done: bool) -> str:
        """Format a single-line `Hoạt động cuối` entry.

        `done=False` → tool_call (🔧 prefix); `done=True` → tool_result (✅ prefix).
        Appends the actual command extracted from the event when available.
        """
        icon = "✅" if done else "🔧"
        cmd = self._extract_tool_command(event)
        if cmd:
            display = cmd if len(cmd) <= 200 else cmd[:200] + "…"
            return f"{icon} {friendly}: {display}"
        return f"{icon} {friendly}"

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
        user_display_name: Optional[str] = None,
    ):
        """Handle non-streaming response in a specific topic (without update object).

        Streamed internally so /ping can report progress; only the final
        assembled message is delivered to the user.
        """
        _set_investigation_status(chat_id, topic_id, "🔄 Working...")
        response = ""
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
                user_display_name=user_display_name,
            )
            async for event in async_gen:
                event_type = event.get("type", "unknown")
                if event_type == "content":
                    chunk = event.get("content", "")
                    response += chunk
                    _set_investigation_status(
                        chat_id, topic_id,
                        response_preview=response[-300:],
                    )
                elif event_type == "task_update":
                    _set_investigation_status(
                        chat_id, topic_id, None, tasks=event.get("tasks", []),
                    )
                elif event_type == "tool_call":
                    tool_name = event.get("tool_name", "unknown")
                    friendly = self._get_friendly_tool_name(tool_name)
                    _set_investigation_status(
                        chat_id, topic_id,
                        status_text=f"🔧 Using {friendly}...",
                        last_tool=tool_name,
                        last_activity=self._format_activity(tool_name, friendly, event, done=False),
                    )
                elif event_type == "tool_result":
                    tool_name = event.get("tool_name", "unknown")
                    friendly = self._get_friendly_tool_name(tool_name)
                    _set_investigation_status(
                        chat_id, topic_id,
                        status_text=f"✅ Completed {friendly}",
                        last_activity=self._format_activity(tool_name, friendly, event, done=True),
                    )
        except _LLMInterrupted:
            # No cancel_event in this path (general-chat topic creation)
            # but keep partial content if it happens.
            pass

        _clear_investigation(chat_id, topic_id)

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
        user_display_name: Optional[str] = None,
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
        current_tasks: list = []
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
                user_display_name=user_display_name,
            )
            _set_investigation_status(chat_id, topic_id, current_status, tasks=current_tasks, last_tool=last_tool)

            # Add timeout to prevent hanging indefinitely
            async def process_stream():
                nonlocal full_response, task_message, current_status, last_tool, current_tasks, stream_completed
                async for event in async_gen:
                    event_type = event.get("type", "unknown")

                    if event_type == "content":
                        chunk = event.get("content", "")
                        full_response += chunk
                        _set_investigation_status(
                            chat_id, topic_id,
                            response_preview=full_response[-300:],
                        )
                        # Edit message every few chunks to avoid rate limits
                        if len(full_response) % 100 == 0:
                            try:
                                await message.edit_text(full_response + "▌")
                            except Exception:
                                pass  # Ignore edit conflicts

                    elif event_type == "task_update":
                        # Display investigation task list
                        tasks = event.get("tasks", [])
                        current_tasks = tasks
                        _set_investigation_status(chat_id, topic_id, None, tasks=tasks)
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
                            _set_investigation_status(chat_id, topic_id, current_status)
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
                        _set_investigation_status(
                            chat_id, topic_id, current_status, last_tool=tool_name,
                            last_activity=self._format_activity(tool_name, friendly_name, event, done=False),
                        )
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
                            _set_investigation_status(
                                chat_id, topic_id, current_status,
                                last_activity=self._format_activity(tool_name, friendly_name, event, done=True),
                            )
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

        # Mark investigation complete so /ping stops reporting it.
        _clear_investigation(chat_id, topic_id)

        # Save conversation
        await self.conversations.save_conversation(conversation)