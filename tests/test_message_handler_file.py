"""Tests for MessageHandler.handle_file — document/photo upload flow."""
import asyncio
import importlib
import os
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.handlers.message_handler import MessageHandler
from src.bot.utils.file_extractor import IMAGE_SENTINEL

# `src/bot/utils/__init__.py` re-exports the `config` instance, shadowing the
# submodule of the same name — import the module explicitly to read settings.
config_module = importlib.import_module("src.bot.utils.config")


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear in-message dedup caches between tests so message_id=1 doesn't
    collide across tests in the same module."""
    from src.bot.handlers import message_handler as mh
    from src.bot.utils import file_sender
    mh._processed_messages.clear()
    mh._topic_batches.clear()
    mh._topic_locks.clear()
    file_sender.reset_file_sender()
    yield


@pytest.fixture
def msg_handler(mock_holmes_service, mock_conversation_service, mock_permission_service):
    return MessageHandler(mock_holmes_service, mock_conversation_service, mock_permission_service)


@pytest.fixture
def mock_holmes_service():
    svc = MagicMock()
    svc.initialize = AsyncMock()
    return svc


@pytest.fixture
def mock_conversation_service():
    svc = MagicMock()
    return svc


@pytest.fixture
def mock_permission_service():
    svc = MagicMock()
    svc.get_or_create_user = AsyncMock()
    return svc


def _file_update(*, file_id="file123", filename="doc.txt", mime="text/plain",
                  size=100, caption="", chat_type="supergroup", is_forum=True,
                  topic_id=123, message_id=1, photo=None):
    """Build a mock Update for a document/photo upload."""
    update = MagicMock()
    update.effective_user = MagicMock(id=42, username="u", first_name="F", last_name="L")
    update.effective_chat = MagicMock()
    update.effective_chat.id = -100
    update.effective_chat.type = chat_type
    update.effective_chat.is_forum = is_forum
    update.message = MagicMock()
    update.message.message_id = message_id
    update.message.message_thread_id = topic_id if (is_forum and topic_id) else None
    update.message.text = None
    update.message.caption = caption
    update.message.photo = photo
    update.message.reply_text = AsyncMock()
    if photo is None:
        doc = MagicMock()
        doc.file_id = file_id
        doc.file_name = filename
        doc.file_size = size
        doc.mime_type = mime
        update.message.document = doc
    else:
        update.message.document = None
    return update


def _mock_context(tmp_path):
    """Return a mock context whose bot.get_file returns a fake TG file object
    that writes its bytes to the path we choose."""
    context = MagicMock()
    context.bot = MagicMock()

    async def _get_file(file_id):
        class FakeFile:
            async def download_to_drive(self, dest):
                p = pathlib.Path(dest)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"hello world, this is extracted text")
        return FakeFile()

    context.bot.get_file = AsyncMock(side_effect=_get_file)
    return context


@pytest.mark.asyncio
async def test_handle_file_text_upload_enqueues_text(msg_handler, tmp_path, monkeypatch):
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    update = _file_update(filename="note.txt", mime="text/plain", caption="summarize this")
    context = _mock_context(tmp_path)

    enqueue_calls = []
    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        enqueue_calls.append((text, kwargs))
    msg_handler._enqueue_text = fake_enqueue_text

    await msg_handler.handle_file(update, context)

    assert len(enqueue_calls) == 1
    combined, kwargs = enqueue_calls[0]
    assert "summarize this" in combined
    assert "--- File: note.txt ---" in combined
    assert "hello world, this is extracted text" in combined
    # File uploads must skip the debounce quiet window.
    assert kwargs.get("force_flush") is True
    # Tmp file cleaned up
    leftover = list(pathlib.Path(str(tmp_path)).glob("*"))
    assert leftover == []


@pytest.mark.asyncio
async def test_handle_file_stashes_upload_and_notes_resend_path(msg_handler, tmp_path, monkeypatch):
    """A document upload is copied into the outbox and the turn tells the model
    where the original lives, so it can be sent back later."""
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    from src.bot.utils import file_sender
    outbox = pathlib.Path(config_module.config.file_send_outbox_dir)

    update = _file_update(filename="note.txt", mime="text/plain")
    context = _mock_context(tmp_path)

    enqueued = []
    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        enqueued.append(text)
    msg_handler._enqueue_text = fake_enqueue_text

    await msg_handler.handle_file(update, context)

    stored = outbox / "received" / "note.txt"
    assert stored.is_file()
    assert stored.read_text() == "hello world, this is extracted text"
    assert f"[Original file kept for re-sending: {stored}]" in enqueued[0]
    # The stashed copy is what makes the re-send possible.
    assert file_sender.resolve_sendable_path("received/note.txt") == stored.resolve()


@pytest.mark.asyncio
async def test_handle_file_does_not_stash_photos(msg_handler, tmp_path, monkeypatch):
    """Photos already render in chat; stashing them would only bloat the outbox."""
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    from src.bot.utils import file_sender
    outbox = pathlib.Path(config_module.config.file_send_outbox_dir)

    async def _get_file(file_id):
        class FakeFile:
            async def download_to_drive(self, dest):
                p = pathlib.Path(dest)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x89PNG\r\n\x1a\n")
        return FakeFile()

    photo = MagicMock()
    photo.file_id = "p1"
    photo.file_unique_id = "u1"
    photo.file_size = 10
    update = _file_update(filename=None, photo=[photo], caption="look")
    context = MagicMock()
    context.bot = MagicMock()
    context.bot.get_file = AsyncMock(side_effect=_get_file)

    enqueued = []
    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        enqueued.append(text)
    msg_handler._enqueue_text = fake_enqueue_text

    with patch("src.bot.handlers.message_handler.extract_text_from_image", return_value="ocr"):
        await msg_handler.handle_file(update, context)

    assert not (outbox / "received").exists()
    assert "Original file kept" not in enqueued[0]


@pytest.mark.asyncio
async def test_handle_file_unsupported_type_replies_error(msg_handler, tmp_path, monkeypatch):
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))

    async def _get_file(file_id):
        class FakeFile:
            async def download_to_drive(self, dest):
                p = pathlib.Path(dest)
                p.parent.mkdir(parents=True, exist_ok=True)
                # PK zip magic
                p.write_bytes(b"PK\x03\x04unsupported")
        return FakeFile()

    update = _file_update(filename="thing.zip", mime="application/zip")
    context = MagicMock()
    context.bot = MagicMock()
    context.bot.get_file = AsyncMock(side_effect=_get_file)

    called = False
    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        nonlocal called
        called = True
    msg_handler._enqueue_text = fake_enqueue_text

    await msg_handler.handle_file(update, context)

    assert not called, "should not enqueue for unsupported files"
    update.message.reply_text.assert_called_once()
    reply_text = update.message.reply_text.call_args[0][0]
    assert "Couldn't extract" in reply_text or "Couldn't" in reply_text


@pytest.mark.asyncio
async def test_handle_file_image_routes_through_vision(msg_handler, tmp_path, monkeypatch):
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    monkeypatch.setattr("src.bot.handlers.message_handler.config.vision_model", "gpt-4o")

    async def _get_file(file_id):
        class FakeFile:
            async def download_to_drive(self, dest):
                p = pathlib.Path(dest)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x89PNG\r\n\x1a\n")
        return FakeFile()

    update = _file_update(filename="pic.png", mime="image/png", caption="what is this?")
    context = MagicMock()
    context.bot = MagicMock()
    context.bot.get_file = AsyncMock(side_effect=_get_file)

    enqueued = []
    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        enqueued.append(text)
    msg_handler._enqueue_text = fake_enqueue_text

    with patch("src.bot.services.vision_service.describe_image", return_value="a cat sitting on a chair"):
        await msg_handler.handle_file(update, context)

    assert len(enqueued) == 1
    combined = enqueued[0]
    assert "what is this?" in combined
    assert "a cat sitting on a chair" in combined
    assert "--- File: pic.png ---" in combined


@pytest.mark.asyncio
async def test_handle_file_image_falls_back_to_ocr_on_vision_failure(msg_handler, tmp_path, monkeypatch):
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    monkeypatch.setattr("src.bot.handlers.message_handler.config.vision_model", "gpt-4o")

    async def _get_file(file_id):
        class FakeFile:
            async def download_to_drive(self, dest):
                p = pathlib.Path(dest)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x89PNG\r\n\x1a\n")
        return FakeFile()

    update = _file_update(filename="pic.png", mime="image/png")
    context = MagicMock()
    context.bot = MagicMock()
    context.bot.get_file = AsyncMock(side_effect=_get_file)

    enqueued = []
    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        enqueued.append(text)
    msg_handler._enqueue_text = fake_enqueue_text

    with patch("src.bot.services.vision_service.describe_image", return_value=""), \
         patch("src.bot.handlers.message_handler.extract_text_from_image", return_value="ocr result text"):
        await msg_handler.handle_file(update, context)

    assert len(enqueued) == 1
    assert "ocr result text" in enqueued[0]


@pytest.mark.asyncio
async def test_handle_file_image_straight_to_ocr_when_no_vision_model(msg_handler, tmp_path, monkeypatch):
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    monkeypatch.setattr("src.bot.handlers.message_handler.config.vision_model", None)

    async def _get_file(file_id):
        class FakeFile:
            async def download_to_drive(self, dest):
                p = pathlib.Path(dest)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x89PNG\r\n\x1a\n")
        return FakeFile()

    update = _file_update(filename="pic.png", mime="image/png")
    context = MagicMock()
    context.bot = MagicMock()
    context.bot.get_file = AsyncMock(side_effect=_get_file)

    enqueued = []
    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        enqueued.append(text)
    msg_handler._enqueue_text = fake_enqueue_text

    with patch("src.bot.handlers.message_handler.extract_text_from_image", return_value="plain ocr text"):
        await msg_handler.handle_file(update, context)

    assert len(enqueued) == 1
    assert "plain ocr text" in enqueued[0]


@pytest.mark.asyncio
async def test_handle_file_too_large_rejected_early(msg_handler, tmp_path, monkeypatch):
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    update = _file_update(filename="big.bin", size=30 * 1024 * 1024)  # 30 MB
    context = _mock_context(tmp_path)

    enqueued = []
    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        enqueued.append(text)
    msg_handler._enqueue_text = fake_enqueue_text

    await msg_handler.handle_file(update, context)

    assert enqueued == []
    update.message.reply_text.assert_called_once()
    assert "too large" in update.message.reply_text.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_handle_file_deduplicates(msg_handler, tmp_path, monkeypatch):
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    update = _file_update(filename="note.txt", message_id=555)
    context = _mock_context(tmp_path)

    count = 0
    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        nonlocal count
        count += 1
    msg_handler._enqueue_text = fake_enqueue_text

    await msg_handler.handle_file(update, context)
    await msg_handler.handle_file(update, context)

    assert count == 1


@pytest.mark.asyncio
async def test_handle_file_tmp_cleanup_on_extraction_error(msg_handler, tmp_path, monkeypatch):
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))

    async def _get_file(file_id):
        class FakeFile:
            async def download_to_drive(self, dest):
                p = pathlib.Path(dest)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"hello")
        return FakeFile()

    update = _file_update(filename="note.txt", mime="text/plain")
    context = MagicMock()
    context.bot = MagicMock()
    context.bot.get_file = AsyncMock(side_effect=_get_file)

    with patch("src.bot.handlers.message_handler.extract_text_from_file",
               side_effect=RuntimeError("boom")):
        await msg_handler.handle_file(update, context)

    leftover = list(pathlib.Path(str(tmp_path)).glob("*"))
    assert leftover == [], "tmp file should be cleaned up even on extraction error"
    update.message.reply_text.assert_called_once()
