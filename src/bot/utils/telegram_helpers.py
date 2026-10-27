"""Telegram Helper Utilities."""
import asyncio
import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from telegram.error import NetworkError, RetryAfter, TimedOut

from src.bot.utils.file_extractor import extract_text_from_file

logger = logging.getLogger(__name__)


TELEGRAM_MAX_MESSAGE_LENGTH = 4096

# Number of retry attempts for transient network errors (TimedOut/NetworkError/RetryAfter).
# Telegram's own guidance: a timeout doesn't necessarily mean the request didn't reach the
# server, so a short backoff retry is safe and avoids losing a long LLM response to a
# single transient blip.
_MAX_SEND_RETRIES = 2
_RETRY_BACKOFF_BASE = 2.0  # seconds; doubled per attempt, plus any RetryAfter value


def split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> List[str]:
    """
    Split a long message into chunks that fit within Telegram's message length limit.

    Attempts to split at natural boundaries (newlines, spaces) to avoid breaking words.

    Args:
        text: The message text to split
        max_length: Maximum length per chunk (default: Telegram's 4096 limit)

    Returns:
        List of message chunks, each <= max_length characters
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_length:
        # Try to find a good split point within the limit
        split_idx = max_length

        # Prefer splitting at double newline (paragraph break)
        double_newline_idx = remaining.rfind('\n\n', 0, max_length)
        if double_newline_idx > max_length * 0.5:  # Only if it's not too early
            split_idx = double_newline_idx + 2  # Include the newlines
        else:
            # Try single newline
            newline_idx = remaining.rfind('\n', 0, max_length)
            if newline_idx > max_length * 0.5:
                split_idx = newline_idx + 1  # Include the newline
            else:
                # Try space
                space_idx = remaining.rfind(' ', 0, max_length)
                if space_idx > max_length * 0.5:
                    split_idx = space_idx + 1  # Include the space

        chunk = remaining[:split_idx]
        chunks.append(chunk)
        remaining = remaining[split_idx:]

    # Add the remaining part
    if remaining:
        chunks.append(remaining)

    return chunks


async def _retry_send_message(bot, **send_kwargs):
    """Call bot.send_message with backoff retry on transient Telegram errors."""
    last_exc = None
    for attempt in range(_MAX_SEND_RETRIES + 1):
        try:
            return await bot.send_message(**send_kwargs)
        except RetryAfter as e:
            last_exc = e
            if attempt >= _MAX_SEND_RETRIES:
                break
            await asyncio.sleep(float(e.retry_after))
        except (TimedOut, NetworkError) as e:
            last_exc = e
            if attempt >= _MAX_SEND_RETRIES:
                break
            await asyncio.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
    raise last_exc if last_exc else RuntimeError("send_message failed without exception")


async def _retry_edit_message_text(bot, **edit_kwargs):
    """Call bot.edit_message_text with backoff retry on transient Telegram errors."""
    last_exc = None
    for attempt in range(_MAX_SEND_RETRIES + 1):
        try:
            return await bot.edit_message_text(**edit_kwargs)
        except RetryAfter as e:
            last_exc = e
            if attempt >= _MAX_SEND_RETRIES:
                break
            await asyncio.sleep(float(e.retry_after))
        except (TimedOut, NetworkError) as e:
            last_exc = e
            if attempt >= _MAX_SEND_RETRIES:
                break
            await asyncio.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
    raise last_exc if last_exc else RuntimeError("edit_message_text failed without exception")


async def send_long_message(
    bot,
    chat_id: int,
    text: str,
    message_thread_id: int = None,
    **kwargs
) -> List:
    """
    Send a potentially long message by splitting it into multiple Telegram messages.

    Args:
        bot: Telegram bot instance
        chat_id: Chat ID to send to
        text: Message text (will be split if > 4096 chars)
        message_thread_id: Optional forum topic thread ID
        **kwargs: Additional arguments passed to send_message (parse_mode, etc.)

    Returns:
        List of sent Message objects
    """
    chunks = split_message(text)
    sent_messages = []

    for i, chunk in enumerate(chunks):
        # For the first chunk, we might want to reply to a specific message
        # For subsequent chunks, just send as new messages
        if i == 0 and 'reply_to_message_id' in kwargs:
            # Only reply to the original message for the first chunk
            message = await _retry_send_message(
                bot,
                chat_id=chat_id,
                text=chunk,
                message_thread_id=message_thread_id,
                **kwargs
            )
        else:
            # Remove reply_to_message_id for subsequent chunks
            send_kwargs = {k: v for k, v in kwargs.items() if k != 'reply_to_message_id'}
            message = await _retry_send_message(
                bot,
                chat_id=chat_id,
                text=chunk,
                message_thread_id=message_thread_id,
                **send_kwargs
            )
        sent_messages.append(message)

    return sent_messages


async def edit_or_send_long_message(
    bot,
    chat_id: int,
    text: str,
    message_id: int = None,
    message_thread_id: int = None,
    **kwargs
) -> List:
    """
    Edit an existing message or send a new one, handling long messages by splitting.

    If message_id is provided, attempts to edit that message with the first chunk,
    then sends remaining chunks as new messages.

    Args:
        bot: Telegram bot instance
        chat_id: Chat ID
        text: Message text (will be split if > 4096 chars)
        message_id: Optional message ID to edit (for streaming responses)
        message_thread_id: Optional forum topic thread ID
        **kwargs: Additional arguments

    Returns:
        List of Message objects (edited/sent)
    """
    chunks = split_message(text)
    sent_messages = []

    if not chunks:
        return sent_messages

    if message_id:
        # Edit the first chunk
        try:
            await _retry_edit_message_text(
                bot,
                chat_id=chat_id,
                message_id=message_id,
                text=chunks[0],
                **kwargs
            )
            # Create a dummy message object for the edited message
            # We can't get the actual message object from edit_message_text easily
            sent_messages.append(None)  # Placeholder
        except Exception:
            # If edit fails, send as new message
            message = await _retry_send_message(
                bot,
                chat_id=chat_id,
                text=chunks[0],
                message_thread_id=message_thread_id,
                **kwargs
            )
            sent_messages.append(message)

        # Send remaining chunks as new messages
        for chunk in chunks[1:]:
            message = await _retry_send_message(
                bot,
                chat_id=chat_id,
                text=chunk,
                message_thread_id=message_thread_id,
                **kwargs
            )
            sent_messages.append(message)
    else:
        # No message_id, send all as new messages
        sent_messages = await send_long_message(
            bot, chat_id, text, message_thread_id, **kwargs
        )

    return sent_messages

# ---------- image attachments for vision-capable models ----------

# Telegram bot download limit for a file produced by get_file().
_MAX_IMAGE_DOWNLOAD_BYTES = 20 * 1024 * 1024


def attachment_target(message):
    """Return (file_id, filename, mime) for a file attached to `message`.

    Handles a document of any type and a photo (largest size). Returns None
    when the message carries no attachment.
    """
    if message is None:
        return None

    doc = getattr(message, "document", None)
    if doc:
        return (
            doc.file_id,
            doc.file_name or "upload",
            (doc.mime_type or "").lower() or None,
        )

    photo = getattr(message, "photo", None)
    if photo:
        largest = photo[-1]
        return largest.file_id, f"photo_{largest.file_unique_id}.jpg", "image/jpeg"

    return None


def _image_target_from_message(message):
    """Return (file_id, mime, suffix) for an image attached to `message`.

    Handles a photo (largest size) or an image document (.png/.jpg/...).
    Returns None when the message carries no image, so callers can no-op.
    """
    target = attachment_target(message)
    if target is None:
        return None

    file_id, filename, mime = target
    if (mime or "").startswith("image/"):
        suffix = os.path.splitext(filename)[1] or ".img"
        return file_id, mime, suffix
    return None


async def download_image_as_data_url(bot, message) -> Optional[dict]:
    """Download the image on `message` and return a vision `{"url": ...}` dict.

    Returns an OpenAI-style `data:<mime>;base64,...` URL wrapped as
    `{"url": ...}` (the shape `build_vision_content` and Holmes' multimodal
    path expect), or None when the message has no image / the download fails.
    The temp file is always removed.
    """
    target = _image_target_from_message(message)
    if target is None:
        return None

    file_id, mime, suffix = target
    tmp_path = None
    try:
        tg_file = await bot.get_file(file_id)
        size = getattr(tg_file, "file_size", 0) or 0
        if size and size > _MAX_IMAGE_DOWNLOAD_BYTES:
            logger.warning(
                "Image too large to attach: %s bytes (limit %s)",
                size, _MAX_IMAGE_DOWNLOAD_BYTES,
            )
            return None

        fd, tmp_path = tempfile.mkstemp(prefix="img-", suffix=suffix)
        os.close(fd)
        await tg_file.download_to_drive(tmp_path)

        with open(tmp_path, "rb") as f:
            payload = base64.b64encode(f.read()).decode("ascii")
        return {"url": f"data:{mime};base64,{payload}"}
    except Exception as e:
        logger.warning("Failed to download image for vision model: %s", e)
        return None
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


async def collect_vision_images(bot, message) -> List[dict]:
    """Images to attach for a command turn.

    Looks at the command's own message (an image uploaded *with* the command as
    its caption) and, additionally, at the message it replies to. Dedupes by
    file_id so an unlikely overlap is attached once.
    """
    images: List[dict] = []
    seen = set()

    for candidate in (message, getattr(message, "reply_to_message", None)):
        target = _image_target_from_message(candidate)
        if target is None or target[0] in seen:
            continue
        seen.add(target[0])
        image = await download_image_as_data_url(bot, candidate)
        if image:
            images.append(image)

    return images


# Hard cap on text extracted from an attachment attached to a command turn.
# The enterprise model has a large context, but a 10 MB xlsx dumped in full
# would still crowd out the conversation and the tool results.
MAX_ATTACHMENT_TEXT_CHARS = 200_000

# Telegram's bot download limit for a file fetched via get_file().
_MAX_ATTACHMENT_DOWNLOAD_BYTES = 20 * 1024 * 1024


async def extract_attachment_text(bot, message) -> Optional[str]:
    """Download the document on `message` and return its extracted text.

    Returns None when the message has no *non-image* attachment or extraction
    yields nothing — images are handled by `collect_vision_images` instead, and
    an image reaching this function (e.g. an image document) is left alone so
    the caller does not paste OCR text alongside the native image.

    The temp download is always removed. Extraction runs off-thread because
    pypdf/openpyxl on a large workbook are CPU-bound and would stall the loop.
    """
    target = attachment_target(message)
    if target is None:
        return None

    file_id, filename, mime = target
    if (mime or "").startswith("image/"):
        return None

    tmp_path = None
    try:
        tg_file = await bot.get_file(file_id)
        size = getattr(tg_file, "file_size", 0) or 0
        if size and size > _MAX_ATTACHMENT_DOWNLOAD_BYTES:
            logger.warning(
                "Attachment too large to extract: %s bytes (limit %s)",
                size, _MAX_ATTACHMENT_DOWNLOAD_BYTES,
            )
            return None

        suffix = os.path.splitext(filename)[1]
        fd, tmp_path = tempfile.mkstemp(prefix="att-", suffix=suffix)
        os.close(fd)
        await tg_file.download_to_drive(tmp_path)

        text = await asyncio.to_thread(extract_text_from_file, Path(tmp_path), mime)
        if not text or not text.strip():
            return None
        if len(text) > MAX_ATTACHMENT_TEXT_CHARS:
            text = (
                text[:MAX_ATTACHMENT_TEXT_CHARS]
                + f"\n\n[... truncated at {MAX_ATTACHMENT_TEXT_CHARS} characters ...]"
            )
        return text
    except Exception as e:
        logger.warning("Failed to extract attachment text: %s", e)
        return None
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


async def save_attachment_for_agent(bot, message) -> Optional[tuple]:
    """Save a non-image attachment to disk and return (path, filename).

    Used by `/claude` only: instead of flattening the file to text, the bytes
    are kept in the outbox's `received/` dir and the path is handed to the
    enterprise model, which has bash + python and can inspect the real file
    (formulas, hidden sheets, embedded media, scanned PDFs, archives) far more
    thoroughly than a text extraction ever could.

    Reusing the outbox means the file inherits its lifetime rules: pruned after
    `FILE_SEND_OUTBOX_TTL_HOURS`, on the persistent volume, and deliverable back
    to the chat with `chat_send_file` / `/sendfile`.

    Returns None when there is no attachment, it is an image (handled natively
    by the vision path), or it could not be stored.
    """
    target = attachment_target(message)
    if target is None:
        return None

    file_id, filename, mime = target
    if (mime or "").startswith("image/"):
        return None

    from src.bot.utils import file_sender as _file_sender

    tmp_path = None
    try:
        tg_file = await bot.get_file(file_id)
        size = getattr(tg_file, "file_size", 0) or 0
        if size and size > _MAX_ATTACHMENT_DOWNLOAD_BYTES:
            logger.warning(
                "Attachment too large to save: %s bytes (limit %s)",
                size, _MAX_ATTACHMENT_DOWNLOAD_BYTES,
            )
            return None

        suffix = os.path.splitext(filename)[1]
        fd, tmp_path = tempfile.mkstemp(prefix="att-", suffix=suffix)
        os.close(fd)
        await tg_file.download_to_drive(tmp_path)

        stored = await asyncio.to_thread(_file_sender.stash_upload, tmp_path, filename)
        if stored is None:
            return None
        return stored, filename
    except Exception as e:
        logger.warning("Failed to save attachment for agent: %s", e)
        return None
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
