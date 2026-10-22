"""`/addmemory` command — admin-only ingestion into the graph memory.

Usage:
- `/addmemory <text>` — ingest the text after the command.
- Reply to a message with `/addmemory` — ingest the replied message's text.
- Reply to a document with `/addmemory` — extract the file's text and ingest it.
- `/addmemory` with an attached document — extract and ingest the attachment.
- Attach a file with caption `/addmemory` — extract and ingest the attachment.

Large inputs (>~4 KB) are split into chunks along markdown header boundaries
before ingesting — each chunk becomes its own episode. Graphiti's LLM-driven
entity extraction per episode scales poorly with input size, so keeping
chunks small not only avoids LLM timeouts but also improves retrieval quality.

When memory is disabled (GRAPHITI_MEMORY_ENABLED=false), replies with a
notice and exits without raising.
"""
import asyncio
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional

import structlog
from telegram import Update
from telegram.ext import ContextTypes

from src.bot.services.memory_service import MemoryService
from src.bot.services.permission_service import PermissionService
from src.bot.utils.config import config
from src.bot.utils.decorators import admin_only
from src.bot.utils.file_extractor import extract_text_from_file, extract_text_from_image

logger = structlog.get_logger(__name__)

# Above this size (rough heuristic) we split before calling add_episode.
# Graphiti's AnthropicClient extracts entities per-episode — a 25 KB blob
# produces a huge extraction response and tends to hit LLM timeout / context
# limits. Chunking keeps each episode compact and improves search quality.
_CHUNK_THRESHOLD = 4000
# Soft cap on chunk size — split paragraphs/lines if a single section exceeds it.
_CHUNK_SOFT_CAP = 6000
# Minimum chunk size — we'd rather merge tiny sections than produce 50 episodes.
_CHUNK_MIN = 400


class AddMemoryCommand:
    """Handler for `/addmemory`."""

    def __init__(
        self,
        memory_service: MemoryService,
        permission_service: Optional[PermissionService] = None,
    ):
        self._memory = memory_service
        self._permissions = permission_service

    @admin_only
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Entry point for `/addmemory`."""
        if not config.graphiti_memory_enabled:
            await update.message.reply_text(
                "⚠️ Memory is disabled. Set GRAPHITI_MEMORY_ENABLED=true to use /addmemory."
            )
            return

        message = update.message
        topic_id = message.message_thread_id or None

        text = await self._collect_text(update, context)
        if not text or not text.strip():
            await message.reply_text(
                "ℹ️ Usage:\n"
                "• `/addmemory <text>` — ingest the given text\n"
                "• Reply to a message with `/addmemory` — ingest that message's text\n"
                "• Reply to a document with `/addmemory` — ingest the document's extracted text\n"
                "• Send `/addmemory` with an attached file — ingest the attachment\n"
                "• Attach a file with caption `/addmemory` — ingest the attachment",
                message_thread_id=topic_id,
                parse_mode="Markdown",
            )
            return

        await self._ingest(update, text, topic_id)

    @admin_only
    async def handle_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Entry point for file uploads whose caption starts with `/addmemory`.

        Telegram sets `message.text=None` and `message.caption="/addmemory ..."`
        when a file is attached with a command-style caption, so PTB's
        CommandHandler (which only matches `message.text`) never fires. This
        handler is registered as a MessageHandler on documents/photos so we
        catch that case.
        """
        if not config.graphiti_memory_enabled:
            await update.message.reply_text(
                "⚠️ Memory is disabled. Set GRAPHITI_MEMORY_ENABLED=true to use /addmemory."
            )
            return

        message = update.message
        topic_id = message.message_thread_id or None

        # Pull any text after the /addmemory command in the caption.
        caption = message.caption or ""
        caption_extra = ""
        parts = caption.split(maxsplit=1)
        if len(parts) > 1:
            caption_extra = parts[1].strip()

        extracted = await self._extract_from_attachment(message, context)
        if not extracted:
            await message.reply_text(
                "❌ Could not extract text from the attached file.",
                message_thread_id=topic_id,
            )
            return

        text = f"{caption_extra}\n\n{extracted}".strip() if caption_extra else extracted
        await self._ingest(update, text, topic_id)

    async def _ingest(self, update: Update, text: str, topic_id) -> None:
        """Shared finish path: chunk if large, add each as an episode, report result."""
        message = update.message
        if not text or not text.strip():
            return

        source = self._build_source_description(update)
        chunks = self._chunk_text(text)
        if len(chunks) > 1:
            await message.reply_text(
                f"⏳ Ingesting {len(chunks)} chunks (text too large for a single "
                f"episode)...",
                message_thread_id=topic_id,
            )

        ok_ids: List[str] = []
        failures = 0
        for i, chunk in enumerate(chunks, 1):
            result_id = await self._memory.add_episode(
                text=chunk,
                source_description=f"{source} chunk={i}/{len(chunks)}",
                name=f"addmemory-{uuid.uuid4().hex[:8]}",
            )
            if result_id:
                ok_ids.append(result_id)
            else:
                failures += 1

        if ok_ids:
            preview = text[:80].replace("\n", " ")
            summary = (
                f"✅ Added {len(ok_ids)}/{len(chunks)} chunk(s) to memory "
                f"({len(text)} chars total, {len(chunks)} chunk(s)).\n"
                f"Preview: {preview}{'…' if len(text) > 80 else ''}"
            )
            if failures:
                summary += f"\n⚠️ {failures} chunk(s) failed — see logs."
            await message.reply_text(
                summary,
                message_thread_id=topic_id,
            )
        else:
            await message.reply_text(
                f"❌ Failed to add to memory ({len(text)} chars, {len(chunks)} "
                f"chunk(s) all failed). Common causes: LLM/embedder timeout, "
                f"FalkorDB connection issue, or missing OPENAI_API_KEY for "
                f"embedder. Check logs for the failure type.",
                message_thread_id=topic_id,
            )

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        """Split a long blob into chunks along markdown section boundaries.

        Strategy:
        - If under _CHUNK_THRESHOLD, return the whole thing as one chunk.
        - Otherwise split on `^#` markdown headers (atx style), keeping the
          header with its section.
        - Sections ≤ _CHUNK_SOFT_CAP are kept whole.
        - Over-long sections are split further by paragraphs (blank line).
        - Tiny adjacent sections are merged back together so we don't end up
          with 50 100-char episodes.
        """
        text = text.strip()
        if len(text) <= _CHUNK_THRESHOLD:
            return [text] if text else []

        # Split on lines beginning with `#` (markdown header). Drop empty leading.
        section_re = re.compile(r"^(?=#{1,6}\s+\S)", re.MULTILINE)
        raw_sections = section_re.split(text)
        sections = [s.strip() for s in raw_sections if s and s.strip()]

        # If we got a single section back (no headers), fall back to paragraph
        # splitting so a long unstructured blob still gets chunked.
        if len(sections) <= 1:
            paragraphs = re.split(r"\n\s*\n", text)
            sections = [p.strip() for p in paragraphs if p.strip()]

        # Further split any section that exceeds the soft cap — by paragraph.
        chunks: List[str] = []
        for section in sections:
            if len(section) <= _CHUNK_SOFT_CAP:
                chunks.append(section)
                continue
            buf = ""
            for para in re.split(r"\n\s*\n", section):
                para = para.strip()
                if not para:
                    continue
                if len(buf) + len(para) + 2 > _CHUNK_SOFT_CAP and buf:
                    chunks.append(buf.strip())
                    buf = para
                else:
                    buf = f"{buf}\n\n{para}" if buf else para
            if buf.strip():
                chunks.append(buf.strip())

        # Merge tiny sections back together so we don't produce 50 100-char
        # episodes. Stop merging if the next chunk would exceed _CHUNK_SOFT_CAP.
        merged: List[str] = []
        for chunk in chunks:
            if merged and len(merged[-1]) + len(chunk) + 2 <= _CHUNK_SOFT_CAP \
                    and (len(merged[-1]) < _CHUNK_MIN or len(chunk) < _CHUNK_MIN):
                merged[-1] = f"{merged[-1]}\n\n{chunk}"
            else:
                merged.append(chunk)

        return merged or ([text] if text else [])

    async def _collect_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
        """Gather text from: (1) explicit argument, (2) replied message, (3) attachment."""
        message = update.message

        # 1. Explicit text after the command
        if context.args:
            return " ".join(context.args).strip()

        # 2. Reply to a message
        replied = message.reply_to_message
        if replied is not None:
            # 2a. Reply to a document/photo — extract text from the attachment
            if replied.document or replied.photo:
                extracted = await self._extract_from_attachment(replied, context)
                if extracted:
                    caption = replied.caption or ""
                    return f"{caption}\n\n{extracted}".strip() if caption else extracted
                return ""
            # 2b. Reply to a text message
            return (replied.text or replied.caption or "").strip()

        # 3. Attachment on the /addmemory message itself
        if message.document or message.photo:
            extracted = await self._extract_from_attachment(message, context)
            caption = message.caption or ""
            return f"{caption}\n\n{extracted}".strip() if caption else extracted

        return ""

    async def _extract_from_attachment(self, message, context: ContextTypes.DEFAULT_TYPE) -> str:
        """Download a message's attachment and extract its text content."""
        if message.document:
            file_id = message.document.file_id
            filename = message.document.file_name or "upload"
            mime = message.document.mime_type
        elif message.photo:
            file_id = message.photo[-1].file_id
            filename = f"photo_{message.photo[-1].file_unique_id}.jpg"
            mime = "image/jpeg"
        else:
            return ""

        os.makedirs(config.file_upload_tmp_dir, exist_ok=True)
        ext = os.path.splitext(filename)[1]
        tmp_path = os.path.join(config.file_upload_tmp_dir, f"{uuid.uuid4().hex}{ext}")

        try:
            tg_file = await context.bot.get_file(file_id)
            await tg_file.download_to_drive(tmp_path)

            if mime and mime.startswith("image/"):
                extracted = await asyncio.to_thread(extract_text_from_image, Path(tmp_path))
            else:
                extracted = await asyncio.to_thread(extract_text_from_file, Path(tmp_path), mime)
            return extracted or ""
        except Exception as e:
            logger.error("addmemory_extract_failed", error=str(e), filename=filename)
            return ""
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _build_source_description(update: Update) -> str:
        user = update.effective_user
        chat = update.effective_chat
        topic = update.message.message_thread_id or 0
        return (
            f"telegram-addmemory user_id={user.id if user else 0} "
            f"username={user.username if user else ''} "
            f"chat_id={chat.id if chat else 0} topic_id={topic}"
        )
