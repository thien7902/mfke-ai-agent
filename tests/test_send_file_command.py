"""Tests for the /sendfile command."""
import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.commands.send_file import SendFileCommand
from src.bot.utils import file_sender

# `src/bot/utils/__init__.py` re-exports the `config` instance, shadowing the
# submodule of the same name — import the module explicitly to patch it.
config_module = importlib.import_module("src.bot.utils.config")


@pytest.fixture(autouse=True)
def outbox(tmp_path, monkeypatch):
    d = tmp_path / "outbox"
    d.mkdir()
    monkeypatch.setattr(config_module.config, "file_send_outbox_dir", str(d))
    monkeypatch.setattr(config_module.config, "file_send_max_bytes", 50 * 1024 * 1024)
    monkeypatch.setattr(config_module.config, "file_send_outbox_ttl_hours", 24)
    monkeypatch.setattr(config_module.config, "test_mode", True)
    file_sender.reset_file_sender()
    yield d
    file_sender.reset_file_sender()


def _update(chat_type="supergroup", is_forum=True, topic_id=55, args=None):
    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = -100
    update.effective_chat.type = chat_type
    update.effective_chat.is_forum = is_forum
    update.effective_user = MagicMock(id=42)
    update.message = MagicMock()
    update.message.message_thread_id = topic_id
    update.message.reply_text = AsyncMock()
    context = MagicMock()
    context.args = args or []
    context.bot = MagicMock()
    context.bot.send_document = AsyncMock()
    return update, context


@pytest.mark.asyncio
async def test_private_chat_rejected():
    handler = SendFileCommand()
    update, context = _update(chat_type="private", is_forum=False)
    await handler.handle(update, context)
    assert "Private chats are disabled" in update.message.reply_text.call_args[0][0]
    context.bot.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_non_topic_group_rejected():
    handler = SendFileCommand()
    update, context = _update(topic_id=None)
    await handler.handle(update, context)
    assert "forum topic" in update.message.reply_text.call_args[0][0]


@pytest.mark.asyncio
async def test_no_args_lists_empty_outbox(outbox):
    handler = SendFileCommand()
    update, context = _update()
    await handler.handle(update, context)
    text = update.message.reply_text.call_args[0][0]
    assert "empty" in text.lower()
    assert str(outbox) in text


@pytest.mark.asyncio
async def test_no_args_lists_files(outbox):
    (outbox / "report.csv").write_text("a,b")
    handler = SendFileCommand()
    update, context = _update()
    await handler.handle(update, context)
    text = update.message.reply_text.call_args[0][0]
    assert "report.csv" in text


@pytest.mark.asyncio
async def test_sends_named_file(outbox):
    (outbox / "note.txt").write_text("hello")
    handler = SendFileCommand()
    update, context = _update(args=["note.txt"])

    await handler.handle(update, context)

    context.bot.send_document.assert_called_once()
    kwargs = context.bot.send_document.call_args.kwargs
    assert kwargs["chat_id"] == -100
    assert kwargs["message_thread_id"] == 55
    assert kwargs["document"] == str((outbox / "note.txt").resolve())


@pytest.mark.asyncio
async def test_outside_outbox_replies_error(tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    handler = SendFileCommand()
    update, context = _update(args=[str(outside)])

    await handler.handle(update, context)

    assert "outside the sendable directory" in update.message.reply_text.call_args[0][0]
    context.bot.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_missing_file_replies_error(outbox):
    handler = SendFileCommand()
    update, context = _update(args=["nope.txt"])
    await handler.handle(update, context)
    assert "File not found" in update.message.reply_text.call_args[0][0]
    context.bot.send_document.assert_not_called()


@pytest.mark.asyncio
async def test_telegram_failure_reported(outbox):
    (outbox / "note.txt").write_text("hello")
    handler = SendFileCommand()
    update, context = _update(args=["note.txt"])
    context.bot.send_document = AsyncMock(side_effect=RuntimeError("chat not found"))

    await handler.handle(update, context)

    assert "Couldn't send" in update.message.reply_text.call_args[0][0]


@pytest.mark.asyncio
async def test_path_with_spaces_joined_from_args(outbox):
    (outbox / "my report.txt").write_text("hi")
    handler = SendFileCommand()
    update, context = _update(args=["my", "report.txt"])

    await handler.handle(update, context)

    kwargs = context.bot.send_document.call_args.kwargs
    assert kwargs["document"] == str((outbox / "my report.txt").resolve())
