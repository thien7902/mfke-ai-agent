"""Tests for /claude handling attachments.

Images go in natively as image_url blocks, independently of
`MAIN_MODEL_SUPPORTS_IMAGES` — that flag governs the *default* model's path in
MessageHandler.handle_file. The enterprise model reads images by default; set
ENTERPRISE_MODEL_SUPPORTS_IMAGES=false for a text-only endpoint.

Documents are saved to the outbox and handed to the agent as a PATH, so the
enterprise model can inspect the real file with bash/python instead of relying
on a lossy text extraction.
"""
import pathlib
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


# ---------- non-image attachments (xlsx/pdf/docx/csv) ----------

def _xlsx_bytes():
    import io
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Thu chi"
    ws.append(["Ngày", "Nội dung", "Số tiền"])
    ws.append(["20-10-2026", "Mua vật tư", 1500000])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _doc_message(file_id="F1", *, name="Thu_chi_20-10_2026_.xlsx", mime=XLSX_MIME, caption=None, reply_to=None):
    m = MagicMock()
    m.reply_to_message = reply_to
    m.photo = None
    m.caption = caption
    doc = MagicMock()
    doc.file_id = file_id
    doc.file_name = name
    doc.mime_type = mime
    m.document = doc
    return m


@pytest.mark.asyncio
async def test_extract_attachment_text_from_xlsx():
    from src.bot.utils.telegram_helpers import extract_attachment_text

    text = await extract_attachment_text(FakeBot(_xlsx_bytes()), _doc_message())
    assert text is not None
    assert "Sheet: Thu chi" in text
    assert "Mua vật tư" in text
    assert "1500000" in text


@pytest.mark.asyncio
async def test_extract_attachment_text_skips_images():
    """Images go through the native vision path; do not double up with OCR."""
    from src.bot.utils.telegram_helpers import extract_attachment_text

    assert await extract_attachment_text(
        FakeBot(PNG_BYTES), _doc_message(mime="image/png", name="shot.png")
    ) is None


@pytest.mark.asyncio
async def test_extract_attachment_text_unsupported_returns_none():
    from src.bot.utils.telegram_helpers import extract_attachment_text

    assert await extract_attachment_text(
        FakeBot(b"PK\x03\x04zip"), _doc_message(mime="application/zip", name="a.zip")
    ) is None


@pytest.mark.asyncio
async def test_extract_attachment_text_without_attachment():
    from src.bot.utils.telegram_helpers import extract_attachment_text

    assert await extract_attachment_text(FakeBot(), _message(None)) is None


def test_has_attachment_covers_documents_and_images():
    assert CommandHandler._claude_has_attachment(_doc_message()) is True
    assert CommandHandler._claude_has_attachment(_message("F", photo=True)) is True
    assert CommandHandler._claude_has_attachment(_message(None)) is False
    assert CommandHandler._claude_has_attachment(
        _message(None, reply_to=_doc_message())
    ) is True


@pytest.mark.asyncio
async def test_claude_command_appends_xlsx_text_to_prompt():
    """The reported bug: an xlsx captioned `/claude ...` must be read, not
    answered with the usage text."""
    import importlib
    from src.bot.models.user import User

    cfgmod = importlib.import_module("src.bot.utils.config")
    cfgmod.config.test_mode = True
    cfgmod.config.enterprise_model_supports_images = True

    captured = {}

    async def chat_enterprise(**kwargs):
        captured.update(kwargs)

        async def gen():
            yield {"type": "content", "content": "ok"}

        return gen()

    holmes = MagicMock()
    holmes.enterprise_model_configured = lambda: True
    holmes.chat_enterprise = chat_enterprise

    perms = MagicMock()
    perms.get_or_create_user = AsyncMock(return_value=User(telegram_id=1))
    convs = MagicMock()
    convs.get_conversation = AsyncMock(return_value=MagicMock(messages=[]))
    convs.save_conversation = AsyncMock()

    handler = CommandHandler(perms, convs, holmes_service=holmes)

    msg = _doc_message(caption="/claude phân tích chỗ này")
    msg.message_thread_id = 99
    msg.reply_text = AsyncMock(return_value=AsyncMock(edit_text=AsyncMock(), delete=AsyncMock()))

    update = MagicMock()
    update.message = msg
    update.effective_chat = MagicMock(id=-100, type="supergroup", is_forum=True)
    update.effective_user = MagicMock(id=1, username="u", first_name="F", last_name="L")

    sent = []

    async def fake_send_long(bot, chat_id, text, **kwargs):
        sent.append(text)

    with patch("src.bot.handlers.command_handler.send_long_message", fake_send_long):
        await handler.claude_command(
            update, MagicMock(args=[], bot=FakeBot(_xlsx_bytes()))
        )

    prompt = captured["user_message"]
    # The user's request leads, so the ack preview and the model both see it.
    assert prompt.startswith("phân tích chỗ này")
    # The file is handed over as a PATH for the agent to inspect with its own
    # bash/python tools — not pre-flattened to text. This is what lets it read
    # formulas, hidden sheets and embedded media that extraction drops.
    assert "Thu_chi_20-10_2026_.xlsx" in prompt
    assert "Path: " in prompt
    assert "received" in prompt
    assert "pandas" in prompt
    # The saved copy actually exists and is where the prompt says it is.
    saved = pathlib.Path(prompt.split("Path: ", 1)[1].splitlines()[0].strip())
    assert saved.is_file()
    assert saved.read_bytes() == _xlsx_bytes()
    # No extracted text is inlined any more.
    assert "--- File:" not in prompt
    assert not captured.get("images")
    assert sent == ["ok"]
    # Usage text must never be emitted for a turn that had an attachment.
    for call in msg.reply_text.call_args_list:
        assert "Usage: /claude" not in str(call)


@pytest.mark.asyncio
async def test_claude_command_hands_any_file_type_to_agent():
    """A zip (or any other type) is no longer refused — the agent runs code on
    it and can unzip/parse it itself. Only an unsaveable file is rejected."""
    import importlib
    from src.bot.models.user import User

    cfgmod = importlib.import_module("src.bot.utils.config")
    cfgmod.config.test_mode = True
    cfgmod.config.enterprise_model_supports_images = True

    captured = {}

    async def chat_enterprise(**kwargs):
        captured.update(kwargs)

        async def gen():
            yield {"type": "content", "content": "ok"}

        return gen()

    holmes = MagicMock()
    holmes.enterprise_model_configured = lambda: True
    holmes.chat_enterprise = chat_enterprise

    perms = MagicMock()
    perms.get_or_create_user = AsyncMock(return_value=User(telegram_id=1))
    convs = MagicMock()
    convs.get_conversation = AsyncMock(return_value=MagicMock(messages=[]))
    convs.save_conversation = AsyncMock()
    handler = CommandHandler(perms, convs, holmes_service=holmes)

    msg = _doc_message(name="logs.zip", mime="application/zip", caption="/claude")
    msg.message_thread_id = 99
    msg.reply_text = AsyncMock(return_value=AsyncMock(edit_text=AsyncMock(), delete=AsyncMock()))
    update = MagicMock()
    update.message = msg
    update.effective_chat = MagicMock(id=-100, type="supergroup", is_forum=True)
    update.effective_user = MagicMock(id=1, username="u", first_name="F", last_name="L")

    with patch("src.bot.handlers.command_handler.send_long_message", AsyncMock()):
        await handler.claude_command(
            update, MagicMock(args=[], bot=FakeBot(b"PK\x03\x04zipbytes"))
        )

    prompt = captured["user_message"]
    assert "logs.zip" in prompt
    assert "Path: " in prompt
    saved = pathlib.Path(prompt.split("Path: ", 1)[1].splitlines()[0].strip())
    assert saved.is_file() and saved.read_bytes() == b"PK\x03\x04zipbytes"
    # The model was actually asked to analyse it.
    assert captured["user_message"].startswith(
        "Describe what this file contains and flag anything notable."
    )


@pytest.mark.asyncio
async def test_claude_command_reports_unsaveable_attachment():
    """Too large to store → tell the user, do not call the model."""
    import importlib
    from src.bot.models.user import User

    cfgmod = importlib.import_module("src.bot.utils.config")
    cfgmod.config.test_mode = True

    holmes = MagicMock()
    holmes.enterprise_model_configured = lambda: True
    holmes.chat_enterprise = AsyncMock()

    perms = MagicMock()
    perms.get_or_create_user = AsyncMock(return_value=User(telegram_id=1))
    convs = MagicMock()
    convs.get_conversation = AsyncMock(return_value=MagicMock(messages=[]))
    convs.save_conversation = AsyncMock()
    handler = CommandHandler(perms, convs, holmes_service=holmes)

    msg = _doc_message(name="kubeconfig.enc", mime="application/octet-stream", caption="/claude")
    msg.message_thread_id = 99
    msg.reply_text = AsyncMock(return_value=AsyncMock(edit_text=AsyncMock(), delete=AsyncMock()))
    update = MagicMock()
    update.message = msg
    update.effective_chat = MagicMock(id=-100, type="supergroup", is_forum=True)
    update.effective_user = MagicMock(id=1, username="u", first_name="F", last_name="L")

    with patch("src.bot.handlers.command_handler.send_long_message", AsyncMock()):
        await handler.claude_command(
            update, MagicMock(args=[], bot=FakeBot(b"secret"))
        )

    replies = [str(c) for c in msg.reply_text.call_args_list]
    assert any("Couldn't save" in r for r in replies)
    holmes.chat_enterprise.assert_not_called()
