"""Tests for /claude reading images natively (enterprise model vision path).

`/claude` attaches image bytes as `image_url` content blocks, independently of
`MAIN_MODEL_SUPPORTS_IMAGES` — that flag governs the *default* model's path in
MessageHandler.handle_file. The enterprise model reads images by default; set
ENTERPRISE_MODEL_SUPPORTS_IMAGES=false for a text-only endpoint.
"""
import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.handlers.command_handler import CommandHandler
from src.bot.utils import telegram_helpers
from src.bot.utils.telegram_helpers import collect_vision_images

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-payload"


class FakeTgFile:
    def __init__(self, payload=PNG_BYTES):
        self.file_size = len(payload)
        self._payload = payload

    async def download_to_drive(self, dest):
        with open(dest, "wb") as f:
            f.write(self._payload)


class FakeBot:
    def __init__(self, payload=PNG_BYTES):
        self._payload = payload
        self.requested = []

    async def get_file(self, file_id):
        self.requested.append(file_id)
        return FakeTgFile(self._payload)


def _message(file_id=None, *, reply_to=None, photo=False, caption=None):
    m = MagicMock()
    m.reply_to_message = reply_to
    m.caption = caption
    if photo:
        m.photo = [MagicMock(file_id=f"{file_id}-small"), MagicMock(file_id=file_id)]
        m.document = None
    elif file_id:
        m.photo = None
        doc = MagicMock()
        doc.file_id = file_id
        doc.mime_type = "image/png"
        doc.file_name = "screenshot.png"
        m.document = doc
    else:
        m.photo = None
        m.document = None
    return m


# ---------- collect_vision_images ----------

@pytest.mark.asyncio
async def test_collects_image_document():
    images = await collect_vision_images(FakeBot(), _message("F1"))
    assert len(images) == 1
    url = images[0]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG_BYTES


@pytest.mark.asyncio
async def test_collects_photo_largest_size():
    bot = FakeBot()
    images = await collect_vision_images(bot, _message("F2", photo=True))
    assert len(images) == 1
    # Largest photo size is the last entry — only that one is downloaded.
    assert bot.requested == ["F2"]
    assert images[0]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_collects_image_from_replied_message():
    images = await collect_vision_images(FakeBot(), _message(None, reply_to=_message("F3")))
    assert len(images) == 1
    assert images[0]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_no_image_returns_empty():
    assert await collect_vision_images(FakeBot(), _message(None)) == []


@pytest.mark.asyncio
async def test_download_failure_is_not_fatal(monkeypatch):
    class BrokenBot:
        async def get_file(self, file_id):
            raise RuntimeError("telegram down")

    assert await collect_vision_images(BrokenBot(), _message("F4")) == []


@pytest.mark.asyncio
async def test_oversize_image_skipped(monkeypatch):
    monkeypatch.setattr(telegram_helpers, "_MAX_IMAGE_DOWNLOAD_BYTES", 4)
    assert await collect_vision_images(FakeBot(), _message("F5")) == []


@pytest.mark.asyncio
async def test_temp_file_removed(monkeypatch):
    import os
    created = []
    real_mkstemp = telegram_helpers.tempfile.mkstemp

    def tracking_mkstemp(*a, **kw):
        fd, path = real_mkstemp(*a, **kw)
        created.append(path)
        return fd, path

    monkeypatch.setattr(telegram_helpers.tempfile, "mkstemp", tracking_mkstemp)
    await collect_vision_images(FakeBot(), _message("F6"))
    assert created and all(not os.path.exists(p) for p in created)


# ---------- /claude prompt + image detection ----------

def test_prompt_from_command_args():
    ctx = MagicMock(args=["describe", "this"])
    assert CommandHandler._claude_prompt(_message(None), ctx) == "describe this"


def test_prompt_from_caption():
    ctx = MagicMock(args=[])
    assert CommandHandler._claude_prompt(_message("F", caption="/claude mô tả ảnh"), ctx) == "mô tả ảnh"


def test_prompt_from_caption_with_bot_suffix():
    ctx = MagicMock(args=[])
    assert CommandHandler._claude_prompt(
        _message("F", caption="/claude@MyBot giải thích lỗi"), ctx
    ) == "giải thích lỗi"


def test_prompt_empty_when_only_command():
    ctx = MagicMock(args=[])
    assert CommandHandler._claude_prompt(_message("F", caption="/claude"), ctx) == ""


def test_has_image_detects_photo_and_reply():
    assert CommandHandler._claude_has_image(_message("F", photo=True)) is True
    assert CommandHandler._claude_has_image(_message("F")) is True
    assert CommandHandler._claude_has_image(_message(None)) is False
    assert CommandHandler._claude_has_image(_message(None, reply_to=_message("F"))) is True


# ---------- enterprise chat carries the images ----------

@pytest.mark.asyncio
async def test_chat_enterprise_persists_images_in_metadata():
    from src.bot.services.holmes_service import HolmesService
    from src.bot.models.conversation import Conversation

    svc = HolmesService()
    conversation = Conversation(user_id=1)
    captured = {}

    async def fake_stream(holmes_messages, features, user_id, conv, *a, **kw):
        captured["messages"] = holmes_messages
        captured["tcll"] = kw.get("tcll")
        return
        yield  # pragma: no cover - makes this an async generator

    svc._build_holmes_messages = lambda msgs, uid: msgs
    svc._get_user_permissions = lambda perms: {}
    svc._stream_chat = fake_stream
    svc._get_enterprise_tcll = AsyncMock(return_value="enterprise-llm")

    image = {"url": "data:image/png;base64,AAAA"}
    async_gen = await svc.chat_enterprise(
        user_id=1,
        user_message="what is this?",
        conversation=conversation,
        user_permissions=[],
        stream=True,
        images=[image],
    )
    async for _ in async_gen:
        pass

    # Persisted for history replay...
    assert conversation.messages[-1].metadata["images"] == [image]
    # ...and converted into an OpenAI multimodal content array for the model.
    content = captured["messages"][-1]["content"]
    assert isinstance(content, list)
    assert [c["type"] for c in content] == ["text", "image_url"]
    assert captured["tcll"] == "enterprise-llm"
