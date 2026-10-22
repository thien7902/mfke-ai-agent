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
        msgs = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Added" in m for m in msgs)


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
        msgs = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Added" in m for m in msgs)


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
        # Now that add_episode raises on failure, simulate that behavior.
        svc.add_episode = AsyncMock(side_effect=RuntimeError("boom"))
        cmd = AddMemoryCommand(svc)

        update = _make_update(args=["some text"])
        ctx = _make_context(args=["some text"])
        await cmd.handle(update, ctx)

        svc.add_episode.assert_awaited_once()
        msgs = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Failed to add" in m for m in msgs)
        assert any("RuntimeError: boom" in m for m in msgs)


@pytest.mark.asyncio
async def test_addmemory_not_initialized_surfaces_runtime_error():
    """If MemoryService failed to init, add_episode raises RuntimeError — surface it."""
    with patch("src.bot.commands.add_memory.config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        svc.add_episode = AsyncMock(side_effect=RuntimeError("MemoryService is not initialized"))
        cmd = AddMemoryCommand(svc)

        update = _make_update(args=["some text"])
        ctx = _make_context(args=["some text"])
        await cmd.handle(update, ctx)

        msgs = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("not initialized" in m for m in msgs)


@pytest.mark.asyncio
async def test_addmemory_upload_caption_extracts_and_ingests():
    """File upload with caption '/addmemory' routes through handle_upload."""
    with patch("src.bot.commands.add_memory.config") as cfg, \
         patch("src.bot.commands.add_memory.extract_text_from_file", return_value="FILE BODY"):
        cfg.graphiti_memory_enabled = True
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        svc.add_episode = AsyncMock(return_value="ep-upload")
        cmd = AddMemoryCommand(svc)

        document = MagicMock()
        document.file_id = "f1"
        document.file_name = "notes.md"
        document.mime_type = "text/markdown"
        update = _make_update(args=None, document=document, caption="/addmemory note from Thien", text=None)
        ctx = _make_context(args=None)
        ctx.bot.get_file = AsyncMock()
        tg_file = MagicMock()
        tg_file.download_to_drive = AsyncMock()
        ctx.bot.get_file.return_value = tg_file

        await cmd.handle_upload(update, ctx)

        svc.add_episode.assert_awaited_once()
        kwargs = svc.add_episode.call_args.kwargs
        assert "FILE BODY" in kwargs["text"]
        assert "note from Thien" in kwargs["text"]
        msgs = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Added" in m for m in msgs)


@pytest.mark.asyncio
async def test_addmemory_chunks_large_input_along_headers():
    """Large md input is split into multiple episodes by `^#` header."""
    big_section = "# Section A\n\n" + ("line content\n" * 100)  # ~1.2KB
    body = big_section * 5  # ~6KB total — splits into multiple sections.
    with patch("src.bot.commands.add_memory.config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        svc.add_episode = AsyncMock(return_value=lambda **kw: f"ep-{kw.get('name', '')[-4:]}")
        cmd = AddMemoryCommand(svc)

        update = _make_update(args=[body])
        ctx = _make_context(args=[body])
        await cmd.handle(update, ctx)

        # Should have ingested multiple episodes, not one.
        assert svc.add_episode.await_count >= 2
        # All chunks should be under the soft cap.
        for call in svc.add_episode.call_args_list:
            assert len(call.kwargs["text"]) <= 6000


@pytest.mark.asyncio
async def test_addmemory_small_input_not_chunked():
    """Small input below threshold goes through as one episode."""
    with patch("src.bot.commands.add_memory.config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        svc.add_episode = AsyncMock(return_value="ep1")
        cmd = AddMemoryCommand(svc)

        update = _make_update(args=["short fact"])
        ctx = _make_context(args=["short fact"])
        await cmd.handle(update, ctx)

        svc.add_episode.assert_awaited_once()
        assert svc.add_episode.call_args.kwargs["text"] == "short fact"


@pytest.mark.asyncio
async def test_addmemory_upload_caption_no_extra_text():
    """Caption is just '/addmemory' with no extra args — only file content ingested."""
    with patch("src.bot.commands.add_memory.config") as cfg, \
         patch("src.bot.commands.add_memory.extract_text_from_file", return_value="BODY"):
        cfg.graphiti_memory_enabled = True
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        svc.add_episode = AsyncMock(return_value="ep1")
        cmd = AddMemoryCommand(svc)

        document = MagicMock()
        document.file_id = "f1"
        document.file_name = "doc.txt"
        document.mime_type = "text/plain"
        update = _make_update(args=None, document=document, caption="/addmemory", text=None)
        ctx = _make_context(args=None)
        ctx.bot.get_file = AsyncMock()
        tg_file = MagicMock()
        tg_file.download_to_drive = AsyncMock()
        ctx.bot.get_file.return_value = tg_file

        await cmd.handle_upload(update, ctx)

        kwargs = svc.add_episode.call_args.kwargs
        assert kwargs["text"] == "BODY"
        msgs = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Added" in m for m in msgs)


@pytest.mark.asyncio
async def test_addmemory_upload_extract_failure_replies_error():
    """If the attachment yields no text, reply with an extraction failure message."""
    with patch("src.bot.commands.add_memory.config") as cfg, \
         patch("src.bot.commands.add_memory.extract_text_from_file", return_value=""):
        cfg.graphiti_memory_enabled = True
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        svc.add_episode = AsyncMock(return_value="ep1")
        cmd = AddMemoryCommand(svc)

        document = MagicMock()
        document.file_id = "f1"
        document.file_name = "bad.bin"
        document.mime_type = "application/octet-stream"
        update = _make_update(args=None, document=document, caption="/addmemory", text=None)
        ctx = _make_context(args=None)
        ctx.bot.get_file = AsyncMock()
        tg_file = MagicMock()
        tg_file.download_to_drive = AsyncMock()
        ctx.bot.get_file.return_value = tg_file

        await cmd.handle_upload(update, ctx)

        svc.add_episode.assert_not_awaited()
        msg = update.message.reply_text.call_args.args[0]
        assert "Could not extract" in msg


@pytest.mark.asyncio
async def test_addmemory_upload_disabled_replies_notice():
    with patch("src.bot.commands.add_memory.config") as cfg:
        cfg.graphiti_memory_enabled = False
        cfg.file_upload_tmp_dir = "/tmp/uploads"
        svc = MagicMock(spec=MemoryService)
        cmd = AddMemoryCommand(svc)

        document = MagicMock()
        document.file_id = "f1"
        document.file_name = "x.txt"
        document.mime_type = "text/plain"
        update = _make_update(args=None, document=document, caption="/addmemory", text=None)
        ctx = _make_context(args=None)
        await cmd.handle_upload(update, ctx)

        msg = update.message.reply_text.call_args.args[0]
        assert "disabled" in msg.lower()
        svc.add_episode.assert_not_awaited()
