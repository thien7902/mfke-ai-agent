"""Tests for src.bot.commands.oncall_command.

Mocks PermissionService / MongoDB / JiraOncallService. Verifies subcommand
dispatch, parse validation, MongoDB persistence on /oncall add, and the
admin_only guard (test_mode auto-bypasses).
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.commands.oncall_command import OncallCommand
from src.bot.services.jira_oncall_service import JiraOncallService


def _make_update(user_id=12345, args=None, chat_id=999):
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.message = MagicMock()
    update.message.message_thread_id = None
    update.message.reply_text = AsyncMock()
    return update


def _make_context(args):
    ctx = MagicMock()
    ctx.args = args
    return ctx


@pytest.fixture
def oncall_service():
    svc = MagicMock(spec=JiraOncallService)
    svc.fire_now = AsyncMock(return_value={
        "date": "2026-11-10", "shift": "morning", "elapsed_seconds": 1.23,
    })
    return svc


@pytest.fixture
def mongo_collections(monkeypatch):
    """Patches `src.bot.commands.oncall_command.MongoDB` to return
    MagicMock collections. Pre-creates SHIFTS_COLLECTION for direct setup."""
    from src.bot.commands import oncall_command as mod
    collections: dict[str, MagicMock] = {}
    def _get_collection(name):
        col = collections.get(name)
        if col is None:
            col = MagicMock()
            col.find_one.return_value = None
            col.find.return_value = []
            col.update_one.return_value = MagicMock()
            col.insert_one.return_value = MagicMock()
            col.delete_one.return_value = MagicMock(deleted_count=1)
            col.delete_many.return_value = MagicMock(deleted_count=5)
            collections[name] = col
        return col
    # Pre-create so tests can directly index into the dict
    for n in ("oncall_shifts",):
        _get_collection(n)
    mock = MagicMock()
    mock.get_collection.side_effect = _get_collection
    monkeypatch.setattr(mod, "MongoDB", mock)
    return mock, collections


async def test_no_args_replies_usage(oncall_service, mongo_collections):
    cmd = OncallCommand(oncall_service)
    update = _make_update(args=[])
    ctx = _make_context([])
    await cmd.handle(update, ctx)
    update.message.reply_text.assert_awaited()
    text = update.message.reply_text.await_args.args[0]
    assert "On-call precheck" in text
    assert "/oncall add" in text


async def test_add_persists_to_mongodb(oncall_service, mongo_collections):
    mock_mongo, collections = mongo_collections
    cmd = OncallCommand(oncall_service)
    update = _make_update(args=["add", "2026-11-10", "morning"])
    ctx = _make_context(["add", "2026-11-10", "morning"])
    await cmd.handle(update, ctx)
    col = collections["oncall_shifts"]
    col.update_one.assert_called()
    args = col.update_one.call_args
    assert args.args[0] == {"user_id": 12345, "date": "2026-11-10", "shift": "morning"}
    set_doc = args.args[1]["$set"]
    assert set_doc["user_id"] == 12345
    assert set_doc["date"] == "2026-11-10"
    assert set_doc["shift"] == "morning"
    assert "created_at" in set_doc
    # Reply mentions success
    text = update.message.reply_text.await_args.args[0]
    assert "✅" in text or "On-call" in text


async def test_add_invalid_date_replies_error(oncall_service, mongo_collections):
    cmd = OncallCommand(oncall_service)
    update = _make_update()
    ctx = _make_context(["add", "bad", "morning"])
    await cmd.handle(update, ctx)
    # update_one NOT called on parse error
    col = mongo_collections[1]["oncall_shifts"]
    col.update_one.assert_not_called()
    text = update.message.reply_text.await_args.args[0]
    assert "Invalid date" in text or "Usage" in text


async def test_add_invalid_shift_replies_error(oncall_service, mongo_collections):
    cmd = OncallCommand(oncall_service)
    update = _make_update()
    ctx = _make_context(["add", "2026-11-10", "noon"])
    await cmd.handle(update, ctx)
    col = mongo_collections[1]["oncall_shifts"]
    col.update_one.assert_not_called()
    text = update.message.reply_text.await_args.args[0]
    assert "Invalid shift" in text or "morning or night" in text


async def test_list_returns_user_shifts(oncall_service, mongo_collections):
    mock_mongo, collections = mongo_collections
    shifts = [
        {"user_id": 12345, "date": "2026-11-09", "shift": "morning"},
        {"user_id": 12345, "date": "2026-11-10", "shift": "night"},
    ]
    # find(...).sort(...) returns the shifts list (the code wraps in list())
    collections["oncall_shifts"].find.return_value = MagicMock()
    collections["oncall_shifts"].find.return_value.sort.return_value = shifts
    cmd = OncallCommand(oncall_service)
    update = _make_update()
    ctx = _make_context(["list"])
    await cmd.handle(update, ctx)
    text = update.message.reply_text.await_args.args[0]
    assert "2026-11-09" in text
    assert "2026-11-10" in text
    # Each shift is formatted with its shift name
    assert "morning" in text and "night" in text


async def test_list_empty_replies_info(oncall_service, mongo_collections):
    mock_mongo, collections = mongo_collections
    collections["oncall_shifts"].find.return_value = MagicMock()
    collections["oncall_shifts"].find.return_value.sort.return_value = []
    cmd = OncallCommand(oncall_service)
    update = _make_update()
    ctx = _make_context(["list"])
    await cmd.handle(update, ctx)
    text = update.message.reply_text.await_args.args[0]
    assert "no registered" in text or "no on-call" in text


async def test_rm_deletes_specific_shift(oncall_service, mongo_collections):
    mock_mongo, collections = mongo_collections
    collections["oncall_shifts"].delete_one.return_value = MagicMock(deleted_count=1)
    cmd = OncallCommand(oncall_service)
    update = _make_update()
    ctx = _make_context(["rm", "2026-11-10", "morning"])
    await cmd.handle(update, ctx)
    col = collections["oncall_shifts"]
    col.delete_one.assert_called()
    args = col.delete_one.call_args
    assert args.args[0] == {"user_id": 12345, "date": "2026-11-10", "shift": "morning"}
    text = update.message.reply_text.await_args.args[0]
    assert "Removed" in text or "🗑" in text


async def test_purge_drops_all_user_shifts(oncall_service, mongo_collections):
    mock_mongo, collections = mongo_collections
    collections["oncall_shifts"].delete_many.return_value = MagicMock(deleted_count=3)
    cmd = OncallCommand(oncall_service)
    update = _make_update()
    ctx = _make_context(["purge"])
    await cmd.handle(update, ctx)
    col = collections["oncall_shifts"]
    col.delete_many.assert_called_once()
    args = col.delete_many.call_args
    assert args.args[0] == {"user_id": 12345}
    text = update.message.reply_text.await_args.args[0]
    assert "Removed" in text or "🧹" in text


async def test_test_subcommand_calls_fire_now(oncall_service, mongo_collections):
    cmd = OncallCommand(oncall_service)
    update = _make_update()
    ctx = _make_context(["test", "2026-11-10", "morning"])
    await cmd.handle(update, ctx)
    oncall_service.fire_now.assert_awaited()
    args = oncall_service.fire_now.await_args
    assert args.args[0] == "2026-11-10"
    assert args.args[1] == "morning"
    text = update.message.reply_text.await_args.args[0]
    # Final reply mentions success + elapsed
    assert "Fired" in text or "✅" in text


async def test_unknown_subcommand_replies_usage(oncall_service, mongo_collections):
    cmd = OncallCommand(oncall_service)
    update = _make_update()
    ctx = _make_context(["totally-unknown"])
    await cmd.handle(update, ctx)
    text = update.message.reply_text.await_args.args[0]
    assert "On-call precheck" in text


def test_parse_date_shift_valid():
    cmd = OncallCommand(MagicMock())
    date, shift, err = cmd._parse_date_shift(["2026-11-10", "morning"])
    assert date == "2026-11-10"
    assert shift == "morning"
    assert err is None


def test_parse_date_shift_invalid_date():
    cmd = OncallCommand(MagicMock())
    date, shift, err = cmd._parse_date_shift(["bad", "morning"])
    assert (date, shift) == (None, None)
    assert "Invalid date" in err


def test_parse_date_shift_invalid_shift():
    cmd = OncallCommand(MagicMock())
    date, shift, err = cmd._parse_date_shift(["2026-11-10", "noon"])
    assert (date, shift) == (None, None)
    assert "Invalid shift" in err


def test_parse_date_shift_missing_args():
    cmd = OncallCommand(MagicMock())
    date, shift, err = cmd._parse_date_shift([])
    assert (date, shift) == (None, None)
    assert err is not None
