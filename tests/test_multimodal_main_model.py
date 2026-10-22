"""Tests for the multimodal main-model path.

When `config.main_model_supports_images` is True, image uploads skip the
vision pre-step / OCR and are attached directly to the user message as an
`image_url` OpenAI content block. These tests exercise that path plus the
`HolmesService.chat` plumbing that converts an `images` list into a
multimodal content array via Holmes' `build_vision_content`.
"""
import base64
import os
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.handlers import message_handler as mh
from src.bot.handlers.message_handler import MessageHandler
from src.bot.models.conversation import Conversation, Message
from src.bot.utils.file_extractor import IMAGE_SENTINEL


@pytest.fixture(autouse=True)
def _clear_caches():
    mh._processed_messages.clear()
    mh._topic_batches.clear()
    mh._topic_locks.clear()
    yield


@pytest.fixture
def msg_handler(mock_holmes_service, mock_conversation_service, mock_permission_service):
    return MessageHandler(mock_holmes_service, mock_conversation_service, mock_permission_service)


@pytest.fixture
def mock_holmes_service():
    svc = MagicMock()
    svc.initialize = AsyncMock()
    # Treat .chat as an async function returning an async generator
    svc.chat = AsyncMock()
    return svc


@pytest.fixture
def mock_conversation_service():
    return MagicMock()


@pytest.fixture
def mock_permission_service():
    svc = MagicMock()
    svc.get_or_create_user = AsyncMock()
    return svc


def _file_update(*, filename="pic.png", mime="image/png", caption="", size=100, message_id=1):
    update = MagicMock()
    update.effective_user = MagicMock(id=42, username="u", first_name="F", last_name="L")
    update.effective_chat = MagicMock(id=-100, type="supergroup", is_forum=True)
    update.message = MagicMock()
    update.message.message_id = message_id
    update.message.message_thread_id = 123
    update.message.text = None
    update.message.caption = caption
    update.message.photo = None
    update.message.reply_text = AsyncMock()
    doc = MagicMock()
    doc.file_id = "file1"
    doc.file_name = filename
    doc.file_size = size
    doc.mime_type = mime
    update.message.document = doc
    return update


def _mock_context_writing_bytes(bytes_payload):
    context = MagicMock()
    context.bot = MagicMock()

    async def _get_file(file_id):
        class FakeFile:
            async def download_to_drive(self, dest):
                p = pathlib.Path(dest)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(bytes_payload)
        return FakeFile()

    context.bot.get_file = AsyncMock(side_effect=_get_file)
    return context


@pytest.mark.asyncio
async def test_image_upload_attaches_data_url_when_main_model_supports_images(
    msg_handler, tmp_path, monkeypatch
):
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    monkeypatch.setattr("src.bot.handlers.message_handler.config.main_model_supports_images", True)
    monkeypatch.setattr("src.bot.handlers.message_handler.config.vision_model", "Qwen3.8-27B")

    # ensure vision_service.describe_image would NOT be called in this path
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    update = _file_update(filename="pic.png", mime="image/png", caption="what is this?")
    context = _mock_context_writing_bytes(image_bytes)

    enqueued = []

    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        enqueued.append((text, kwargs))
    msg_handler._enqueue_text = fake_enqueue_text

    with patch("src.bot.services.vision_service.describe_image") as mock_vision, \
         patch("src.bot.handlers.message_handler.extract_text_from_image") as mock_ocr:
        await msg_handler.handle_file(update, context)
        mock_vision.assert_not_called()
        mock_ocr.assert_not_called()

    assert len(enqueued) == 1
    text, kwargs = enqueued[0]
    # Caption forwarded verbatim — no "--- File ---" wrapper
    assert text == "what is this?"
    assert kwargs.get("force_flush") is True
    images = kwargs.get("images")
    assert isinstance(images, list) and len(images) == 1
    data_url = images[0]["url"]
    assert data_url.startswith("data:image/png;base64,")
    # round-trip the b64 to verify original bytes preserved
    b64 = data_url.split(",", 1)[1]
    assert base64.b64decode(b64) == image_bytes
    # Tmp file cleaned up
    assert list(pathlib.Path(str(tmp_path)).glob("*")) == []


@pytest.mark.asyncio
async def test_image_upload_falls_back_to_extraction_when_flag_off(
    msg_handler, tmp_path, monkeypatch
):
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    monkeypatch.setattr("src.bot.handlers.message_handler.config.main_model_supports_images", False)
    monkeypatch.setattr("src.bot.handlers.message_handler.config.vision_model", None)

    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    update = _file_update(filename="pic.png", mime="image/png", caption="")
    context = _mock_context_writing_bytes(image_bytes)

    enqueued = []

    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        enqueued.append((text, kwargs))
    msg_handler._enqueue_text = fake_enqueue_text

    with patch("src.bot.handlers.message_handler.extract_text_from_image", return_value="ocr text"):
        await msg_handler.handle_file(update, context)

    assert len(enqueued) == 1
    text, kwargs = enqueued[0]
    assert kwargs.get("images") is None
    assert "ocr text" in text


@pytest.mark.asyncio
async def test_non_image_uses_extraction_path_even_when_flag_on(
    msg_handler, tmp_path, monkeypatch
):
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    monkeypatch.setattr("src.bot.handlers.message_handler.config.main_model_supports_images", True)
    monkeypatch.setattr("src.bot.handlers.message_handler.config.vision_model", None)

    # A text file — even though main model supports images, this is not an image.
    update = _file_update(filename="note.txt", mime="text/plain", caption="summarize")
    context = _mock_context_writing_bytes(b"hello world, this is some text content")

    enqueued = []
    async def fake_enqueue_text(upd, ctx, text, **kwargs):
        enqueued.append((text, kwargs))
    msg_handler._enqueue_text = fake_enqueue_text

    await msg_handler.handle_file(update, context)
    assert len(enqueued) == 1
    text, kwargs = enqueued[0]
    assert kwargs.get("images") is None
    assert "hello world" in text


@pytest.mark.asyncio
async def test_images_passed_through_flush_batch_to_holmes_chat(
    msg_handler, tmp_path, monkeypatch
):
    """End-to-end: _flush_batch passes images into holmes.chat for both paths."""
    monkeypatch.setattr("src.bot.handlers.message_handler.config.file_upload_tmp_dir", str(tmp_path))
    monkeypatch.setattr("src.bot.handlers.message_handler.config.test_mode", True)
    monkeypatch.setattr("src.bot.handlers.message_handler.config.message_batch_quiet_period_seconds", 10)
    monkeypatch.setattr("src.bot.handlers.message_handler.config.message_batch_max_wait_seconds", 30)
    monkeypatch.setattr("src.bot.handlers.message_handler.config.message_batch_max_messages", 20)

    # Wire a fake user/permissions/conversation so the dispatch reaches holmes.chat
    msg_handler.permissions.get_or_create_user = AsyncMock(return_value=MagicMock(permissions=[]))
    conv = Conversation(user_id=42, topic_id=123, chat_id=-100)
    msg_handler.conversations.get_conversation = AsyncMock(return_value=conv)

    # holmes.chat returns an async generator; we need a real coroutine
    async def fake_chat(**kwargs):
        # Ensure images reach here
        assert kwargs.get("images"), "images should reach holmes.chat"
        assert kwargs["images"][0]["url"].startswith("data:image/png;base64,")

        async def gen():
            yield {"type": "content", "content": "ok"}
        return gen()

    msg_handler.holmes.chat = AsyncMock(side_effect=fake_chat)
    msg_handler.holmes.initialize = AsyncMock()

    # Patch save to avoid DB
    msg_handler.conversations.save_conversation = AsyncMock()

    # Send the message directly via _enqueue_for_batch
    update = MagicMock()
    update.effective_user = MagicMock(id=42, username="u", first_name="F", last_name="L")
    update.effective_chat = MagicMock(id=-100, type="supergroup", is_forum=True)
    update.message = MagicMock()
    update.message.message_thread_id = 123
    update.message.reply_text = AsyncMock()
    update.message.message_id = 1

    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()

    user = update.effective_user
    data_url = "data:image/png;base64,iVBORw0KGgo="

    # Monkey-patch send_long_message and the streaming path to avoid extra deps
    with patch("src.bot.handlers.message_handler.send_long_message", new=AsyncMock()):
        await msg_handler._enqueue_for_batch(
            update, context, -100, 123, user, "what is this?",
            force_flush=True, images=[{"url": data_url}],
        )

    # holmes.chat was called with images
    args, kwargs = msg_handler.holmes.chat.call_args
    assert kwargs.get("images") == [{"url": data_url}]


def test_convert_messages_to_holmes_format_builds_vision_content():
    """HolmesService._convert_messages_to_holmes_format should turn a user msg
    with metadata['images'] into a content-array with text+image_url blocks."""
    from src.bot.services.holmes_service import HolmesService
    svc = HolmesService()
    msg = Message(role="user", content="hello see this image")
    msg.metadata["images"] = [{"url": "data:image/png;base64,iVBOR="}]
    out = svc._convert_messages_to_holmes_format([msg])
    assert len(out) == 1
    content = out[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "hello see this image"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,iVBOR="


def test_convert_messages_leaves_plain_user_message_as_string():
    """Without images, content stays a plain string (no regression)."""
    from src.bot.services.holmes_service import HolmesService
    svc = HolmesService()
    msg = Message(role="user", content="just text")
    out = svc._convert_messages_to_holmes_format([msg])
    assert out[0]["content"] == "just text"


def test_convert_messages_leaves_tool_messages_unaffected():
    from src.bot.services.holmes_service import HolmesService
    svc = HolmesService()
    # Even if a tool message weirdly has images metadata, it must stay plain content.
    msg = Message(role="tool", content="result", tool_call_id="1", name="bash")
    msg.metadata["images"] = [{"url": "data:image/png;base64,x"}]
    out = svc._convert_messages_to_holmes_format([msg])
    assert out[0]["content"] == "result"


@pytest.mark.asyncio
async def test_holmes_chat_persists_images_on_user_message_metadata(monkeypatch):
    """chat() should store images on the Message that it appends to conversation,
    so a subsequent history replay rehydrates the multimodal content array."""
    from src.bot.services.holmes_service import HolmesService
    svc = HolmesService()
    svc._initialized = True
    # Stub _build_holmes_messages + _get_user_permissions + streaming path
    svc._build_holmes_messages = lambda msgs, uid: msgs
    svc._get_user_permissions = lambda perms: {
        "agents": False, "tools": False, "memory": False,
        "streaming": False, "tool_approval": False,
    }

    # Non-streaming path returns a quick response
    async def nonstream(messages, features, user_id, *args, **kwargs):
        return "ok", MagicMock()
    svc._non_stream_chat_with_memory = AsyncMock(side_effect=nonstream)
    svc._convert_llm_result_to_messages = lambda result: [Message(role="assistant", content="ok")]
    svc.update_conversation_memory = MagicMock()

    conv = Conversation(user_id=1, topic_id=1, chat_id=1)
    img = [{"url": "data:image/png;base64,iVBOR="}]
    await svc.chat(
        user_id=1, user_message="see this", conversation=conv,
        user_permissions=[], stream=False,
        telegram_context=None, chat_id=1, topic_id=1,
        images=img,
    )

    # The last appended user message carries images in metadata
    last_user_msg = [m for m in conv.messages if m.role == "user"][-1]
    assert last_user_msg.metadata.get("images") == img
