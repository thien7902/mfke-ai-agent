"""Tests for the /addmemory command handler."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.commands.add_memory import AddMemoryCommand
from src.bot.services.memory_service import MemoryService


def _make_update(args=None, reply_to=None, document=None, photo=None, caption=None, text="/addmemory"):
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = 987654321
    update.effective_user.username = "admin"
    update.effective_chat = MagicMock()
    update.effective_chat.id = -100
    update.message = MagicMock()
    update.message.text = text
    update.message.message_thread_id = 123
    update.message.reply_text = AsyncMock()
    update.message.reply_to_message = reply_to
    update.message.document = document
    update.message.photo = photo
    update.message.caption = caption
    return update


def _make_context(args=None):
    ctx = MagicMock()
    ctx.args = args or []
    ctx.bot = MagicMock()
    ctx.bot.get_file = AsyncMock()
    return ctx


@pytest.mark.asyncio
async def test_addmemory_disabled_replies_with_notice():
    """When memory is disabled, /addmemory short-circuits with a config notice."""
    with patch("src.bot.commands.add_memory.config") as cfg:
        cfg.graphiti_memory_enabled = False
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        cmd = AddMemoryCommand(svc)
        update = _make_update(args=["hello"])
        ctx = _make_context(args=["hello"])
        await cmd.handle(update, ctx)
        update.message.reply_text.assert_awaited_once()
        msg = update.message.reply_text.call_args.args[0]
        assert "disabled" in msg.lower()
        svc.add_episode.assert_not_called()


@pytest.mark.asyncio
async def test_addmemory_with_explicit_text():
    with patch("src.bot.commands.add_memory.config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        svc.add_episode = AsyncMock(return_value="ep-abc")
        cmd = AddMemoryCommand(svc)

        update = _make_update(args=["Cats like fish"])
        ctx = _make_context(args=["Cats like fish"])
        await cmd.handle(update, ctx)

        svc.add_episode.assert_awaited_once()
        call_kwargs = svc.add_episode.call_args.kwargs
        assert call_kwargs["text"] == "Cats like fish"
        # Confirm success message was sent
        msg = update.message.reply_text.call_args.args[0]
        assert "Added to memory" in msg
        assert "ep-abc" in msg


@pytest.mark.asyncio
async def test_addmemory_reply_to_text_message():
    with patch("src.bot.commands.add_memory.config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        svc.add_episode = AsyncMock(return_value="ep-reply")
        cmd = AddMemoryCommand(svc)

        replied = MagicMock()
        replied.text = "Remember this important fact"
        replied.caption = None
        replied.document = None
        replied.photo = None
        update = _make_update(args=[], reply_to=replied)
        ctx = _make_context(args=[])
        await cmd.handle(update, ctx)

        svc.add_episode.assert_awaited_once()
        call_kwargs = svc.add_episode.call_args.kwargs
        assert call_kwargs["text"] == "Remember this important fact"
        assert "ep-reply" in update.message.reply_text.call_args.args[0]


@pytest.mark.asyncio
async def test_addmemory_no_input_shows_usage():
    with patch("src.bot.commands.add_memory.config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        svc.add_episode = AsyncMock(return_value="ep")
        cmd = AddMemoryCommand(svc)

        update = _make_update(args=[])
        ctx = _make_context(args=[])
        await cmd.handle(update, ctx)

        svc.add_episode.assert_not_awaited()
        msg = update.message.reply_text.call_args.args[0]
        assert "Usage" in msg


@pytest.mark.asyncio
async def test_addmemory_failure_replies_error():
    with patch("src.bot.commands.add_memory.config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        svc.add_episode = AsyncMock(return_value=None)
        cmd = AddMemoryCommand(svc)

        update = _make_update(args=["some text"])
        ctx = _make_context(args=["some text"])
        await cmd.handle(update, ctx)

        svc.add_episode.assert_awaited_once()
        msg = update.message.reply_text.call_args.args[0]
        assert "Failed to add" in msg
