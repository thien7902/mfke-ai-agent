"""Tests for src.bot.services.jira_oncall_service.

Mocks the MCP client, MongoDB collections, the Telegram bot, the Conversation
service, and HolmesService.chat. No real network, no real Holmes calls, no
real Mongo. Verifies:
- disabled / incomplete config refuses to start the loop
- guard: oncall_bot_user_id in admin_user_ids refuses to start
- shift-boundary tick logic (morning/night, dedup within 5-min window)
- watchdog reset on fire-tick-with-new-tickets
- JQL has no assignee filter
- global dedup (oncall_tickets) — second fire with same key skips
- Holmes.chat called with oncall_bot_user_id (NOT human id), [] permissions,
  and READ_ONLY_RULE via extra_prompt_additions
- topic + header posted even if Holmes chat throws
- ticket marked as fired in oncall_tickets after Holmes success
"""
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.services.jira_oncall_service import JiraOncallService


def _make_issue(key="ISD-1", summary="pod crashloop", priority="P2",
                status="Open", description="pod in crashloop"):
    # Match mcp-atlassian `Issue.to_simplified_dict()` shape: summary,
    # description, priority, status are top-level keys (not nested under
    # `fields` like raw Jira REST v2).
    return {
        "key": key,
        "summary": summary,
        "description": description,
        "priority": {"name": priority},
        "status": {"name": status},
    }


def _make_shift(user_id=12345, date="2026-11-10", shift="morning"):
    return {"user_id": user_id, "date": date, "shift": shift}


class _Bot:
    """Fake python-telegram-bot. create_forum_topic returns a MagicMock with
    .message_thread_id set; send_message is an AsyncMock."""
    def __init__(self):
        self.create_forum_topic = AsyncMock()
        topic = MagicMock()
        topic.message_thread_id = 9991
        self.create_forum_topic.return_value = topic
        self.send_message = AsyncMock()


class _TelegramBot:
    """Fake TelegramBot for the watchdog reset."""
    def __init__(self):
        self.last_update_time = 0.0


@pytest.fixture
def fake_config(monkeypatch):
    """Patch the config object used by JiraOncallService."""
    from src.bot.services import jira_oncall_service as svc_mod
    cfg = MagicMock()
    cfg.oncall_enabled = True
    cfg.oncall_chat_id = -1001234567890
    cfg.oncall_mcp_url = "http://localhost:9001/mcp"
    cfg.oncall_jira_jql_template = (
        'cf[14307] = "Cloud FKE (K8S)" AND statusCategory != Done '
        'AND created >= "{since}"'
    )
    cfg.oncall_poll_interval_seconds = 1  # fast for tests
    cfg.oncall_shift_morning_hour = 8
    cfg.oncall_shift_night_hour = 19
    cfg.oncall_lookback_minutes = 720
    cfg.oncall_topic_icon_color = 0x6FB9F0
    cfg.oncall_bot_user_id = 0
    cfg.admin_user_ids = [111, 222]  # 0 is NOT here
    monkeypatch.setattr(svc_mod, "config", cfg)
    return cfg


@pytest.fixture
def fake_mongo(monkeypatch):
    """Patch MongoDB.get_collection to return a fake per-name collection dict.
    Pre-creates the oncall_shifts and oncall_tickets collections so individual
    tests can directly configure their .find / .find_one return values."""
    from src.bot.services import jira_oncall_service as svc_mod
    collections: dict[str, MagicMock] = {}

    def _get_collection(name):
        if name not in collections:
            col = MagicMock()
            col.find_one.return_value = None
            col.find.return_value = []
            col.update_one.return_value = MagicMock()
            col.insert_one.return_value = MagicMock()
            collections[name] = col
        return collections[name]

    # Pre-create the two collections JiraOncallService uses, so tests can
    # set their return values directly via `collections["oncall_shifts"].find...`
    for n in ("oncall_shifts", "oncall_tickets"):
        _get_collection(n)

    mock = MagicMock()
    mock.get_collection.side_effect = _get_collection
    monkeypatch.setattr(svc_mod, "MongoDB", mock)
    return mock, collections


# ---------------------------------------------------------------------------
# start() / guard
# ---------------------------------------------------------------------------

async def test_disabled_does_not_start_loop(fake_config, fake_mongo):
    fake_config.oncall_enabled = False
    svc = JiraOncallService()
    await svc.start(_Bot(), _TelegramBot())
    assert svc._task is None


async def test_missing_chat_id_does_not_start(fake_config, fake_mongo):
    fake_config.oncall_chat_id = None
    svc = JiraOncallService()
    await svc.start(_Bot(), _TelegramBot())
    assert svc._task is None


async def test_missing_mcp_url_does_not_start(fake_config, fake_mongo):
    fake_config.oncall_mcp_url = None
    svc = JiraOncallService()
    await svc.start(_Bot(), _TelegramBot())
    assert svc._task is None


async def test_refuses_start_when_bot_user_id_is_admin(fake_config, fake_mongo):
    fake_config.oncall_bot_user_id = 111  # in admin_user_ids
    svc = JiraOncallService()
    await svc.start(_Bot(), _TelegramBot())
    assert svc._task is None


# ---------------------------------------------------------------------------
# _tick — continuous-poll worker
# ---------------------------------------------------------------------------

async def test_tick_outside_shift_returns_early(fake_config, fake_mongo):
    """No shift active at the current time → no MongoDB find, no MCP call,
    no _fire_for_shift call. The worker is a true no-op between shifts."""
    # Disable both shifts — 03:00 with morning=08,night=19 is in night
    # shift actually, so pick 06:00 to be in the gap between 19:00→08:00
    # night window. Actually 06:00 IS in night window. The gap is the
    # morning/night boundary is none — they're contiguous. The only "no
    # shift" state is when both shift hours are -1 (disabled).
    fake_config.oncall_shift_morning_hour = -1
    fake_config.oncall_shift_night_hour = -1
    svc = JiraOncallService()
    svc._mcp = MagicMock()
    svc._fire_for_shift = AsyncMock()
    dt = datetime(2026, 11, 10, 14, 0)
    with patch("src.bot.services.jira_oncall_service.datetime") as mock_dt:
        mock_dt.now.return_value = dt
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        await svc._tick()
    svc._fire_for_shift.assert_not_awaited()


async def test_tick_morning_window_fires(fake_config, fake_mongo):
    """14:00 (mid-morning) → shift active, registrant exists →
    _fire_for_shift called with shift_start_dt = today 08:00."""
    mock_mongo, collections = fake_mongo
    collections["oncall_shifts"].find.return_value = [_make_shift()]
    svc = JiraOncallService()
    svc._bot = _Bot()
    svc._telegram_bot = _TelegramBot()
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock(return_value=[])
    svc._fire_for_shift = AsyncMock()
    dt = datetime(2026, 11, 10, 14, 5)
    with patch("src.bot.services.jira_oncall_service.datetime") as mock_dt:
        mock_dt.now.return_value = dt
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        await svc._tick()
    svc._fire_for_shift.assert_awaited_once()
    args = svc._fire_for_shift.await_args
    assert args.args[0] == "2026-11-10"   # date_str
    assert args.args[1] == "morning"     # shift
    assert args.args[2] == datetime(2026, 11, 10, 8, 0, 0)  # shift_start_dt
    # watchdog reset
    assert svc._telegram_bot.last_update_time > 0.0


async def test_tick_night_pre_midnight_fires(fake_config, fake_mongo):
    """22:00 same day → shift active, documents under "today" →
    _fire_for_shift called with shift_start_dt = today 19:00."""
    mock_mongo, collections = fake_mongo
    collections["oncall_shifts"].find.return_value = [_make_shift(shift="night")]
    svc = JiraOncallService()
    svc._bot = _Bot()
    svc._telegram_bot = _TelegramBot()
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock(return_value=[])
    svc._fire_for_shift = AsyncMock()
    dt = datetime(2026, 11, 10, 22, 30)
    with patch("src.bot.services.jira_oncall_service.datetime") as mock_dt:
        mock_dt.now.return_value = dt
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        await svc._tick()
    svc._fire_for_shift.assert_awaited_once()
    args = svc._fire_for_shift.await_args
    assert args.args[0] == "2026-11-10"   # date_str = shift-start day
    assert args.args[1] == "night"
    assert args.args[2] == datetime(2026, 11, 10, 19, 0, 0)


async def test_tick_night_post_midnight_fires_yesterday_doc(fake_config, fake_mongo):
    """03:00 on day N — a night shift registered under day N-1 (when the
    shift started) must still fire. _compute_current_shift returns
    YESTERDAY's date so the MongoDB lookup finds the shift document."""
    mock_mongo, collections = fake_mongo
    # The MongoDB find query in code uses {date: <date>, shift: "night"} —
    # for a post-midnight tick the date is yesterday's. The fixture's
    # mock `find.return_value` doesn't care about the query, so just
    # ensure find was called with the yesterday date.
    collections["oncall_shifts"].find.return_value = [_make_shift(shift="night")]
    svc = JiraOncallService()
    svc._bot = _Bot()
    svc._telegram_bot = _TelegramBot()
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock(return_value=[])
    svc._fire_for_shift = AsyncMock()
    dt = datetime(2026, 11, 12, 3, 0)  # 03:00 day 12
    with patch("src.bot.services.jira_oncall_service.datetime") as mock_dt:
        mock_dt.now.return_value = dt
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        await svc._tick()
    svc._fire_for_shift.assert_awaited_once()
    args = svc._fire_for_shift.await_args
    # date_str is yesterday (2026-11-11) — matches the shift document
    assert args.args[0] == "2026-11-11"
    assert args.args[1] == "night"
    # shift_start_dt is yesterday's 19:00 (when the shift actually began)
    assert args.args[2] == datetime(2026, 11, 11, 19, 0, 0)


async def test_tick_no_registrants_returns_early(fake_config, fake_mongo):
    """Shift active but nobody registered for it → no MCP call, no
    _fire_for_shift. Saves an MCP round-trip."""
    mock_mongo, collections = fake_mongo
    collections["oncall_shifts"].find.return_value = []  # no registrants
    svc = JiraOncallService()
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock()
    svc._fire_for_shift = AsyncMock()
    dt = datetime(2026, 11, 10, 14, 0)  # mid-morning
    with patch("src.bot.services.jira_oncall_service.datetime") as mock_dt:
        mock_dt.now.return_value = dt
        await svc._tick()
    svc._fire_for_shift.assert_not_awaited()
    svc._mcp.search_issues.assert_not_awaited()


async def test_tick_bumps_watchdog_even_with_no_new_tickets(fake_config, fake_mongo):
    """Watchdog reset happens on every tick where a shift is active —
    not just when new tickets are found. A long quiet stretch during a
    shift must not trigger the 90m idle-exit."""
    mock_mongo, collections = fake_mongo
    collections["oncall_shifts"].find.return_value = [_make_shift()]
    # Already-processed ticket — dedup hits, no new issues to process
    collections["oncall_tickets"].find_one.return_value = {"ticket_key": "ISD-1"}
    svc = JiraOncallService()
    bot = _Bot()
    tg_bot = _TelegramBot()
    tg_bot.last_update_time = 1000.0
    svc._bot = bot
    svc._telegram_bot = tg_bot
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock(return_value=[_make_issue(key="ISD-1")])
    svc._process_ticket = AsyncMock()
    dt = datetime(2026, 11, 10, 14, 0)
    with patch("src.bot.services.jira_oncall_service.datetime") as mock_dt:
        mock_dt.now.return_value = dt
        await svc._tick()
    assert tg_bot.last_update_time > 1000.0
    # _process_ticket NOT called — ticket already fired
    svc._process_ticket.assert_not_awaited()


# ---------------------------------------------------------------------------
# _compute_current_shift — boundary cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hour,minute,expected_date,expected_shift", [
    (8, 0, "2026-11-10", "morning"),    # exactly morning start
    (18, 59, "2026-11-10", "morning"),  # last min of morning
    (19, 0, "2026-11-10", "night"),     # exactly night start
    (23, 59, "2026-11-10", "night"),    # last min before midnight
    (0, 0, "2026-11-09", "night"),      # midnight — yesterday doc
    (3, 0, "2026-11-09", "night"),      # 03:00 — yesterday doc
    (7, 59, "2026-11-09", "night"),     # last min of post-midnight night
])
def test_compute_current_shift_boundaries(hour, minute, expected_date, expected_shift, fake_config):
    """Every boundary of morning/night and the midnight-wrap edge case."""
    svc = JiraOncallService()
    now = datetime(2026, 11, 10, hour, minute)
    result = svc._compute_current_shift(now)
    assert result == (expected_date, expected_shift)


def test_compute_current_shift_both_disabled(fake_config):
    """When both shift hours are -1, no shift is ever active."""
    fake_config.oncall_shift_morning_hour = -1
    fake_config.oncall_shift_night_hour = -1
    svc = JiraOncallService()
    assert svc._compute_current_shift(datetime(2026, 11, 10, 14, 0)) is None
    assert svc._compute_current_shift(datetime(2026, 11, 10, 3, 0)) is None


# ---------------------------------------------------------------------------
# _fire_for_shift — JQL + dedup + multi-ticket
# ---------------------------------------------------------------------------

async def test_fire_for_shift_uses_jql_without_assignee_filter(fake_config, fake_mongo):
    mock_mongo, collections = fake_mongo
    collections["oncall_tickets"].find_one.return_value = None
    # Return a single issue with a key to drive one _process_ticket call
    issue = _make_issue(key="ISD-100")
    svc = JiraOncallService()
    bot = _Bot()
    svc._bot = bot
    svc._telegram_bot = _TelegramBot()
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock(return_value=[issue])
    svc._conversation_service = MagicMock()
    svc._conversation_service.get_or_create_for_topic = AsyncMock(
        return_value=MagicMock()
    )
    svc._holmes = MagicMock()
    svc._holmes.chat = AsyncMock(return_value="ok")
    shift_start_dt = datetime(2026, 11, 10, 8, 0, 0)
    await svc._fire_for_shift("2026-11-10", "morning", shift_start_dt, [_make_shift()])
    # Verify JQL passed to MCP has no "assignee"
    args = svc._mcp.search_issues.await_args
    jql = args.args[0]
    assert "assignee" not in jql
    assert 'Cloud FKE (K8S)' in jql
    # Filters on `created`, not `updated` — only newly-created tickets in
    # the shift window, not old tickets with new comments/status.
    assert 'created >= ' in jql
    assert 'updated' not in jql
    # shift_start timestamp is substituted into the JQL
    assert "2026-11-10 08:00" in jql


async def test_fire_for_shift_global_dedup_skips_already_fired_tickets(fake_config, fake_mongo):
    """A ticket already in oncall_tickets should NOT be processed again."""
    mock_mongo, collections = fake_mongo
    # The tickets collection already has ISD-100
    collections["oncall_tickets"].find_one.side_effect = lambda q: (
        {"_id": "x", "ticket_key": q["ticket_key"]}
        if q["ticket_key"] == "ISD-100" else None
    )
    issues = [_make_issue(key="ISD-100"), _make_issue(key="ISD-101")]
    svc = JiraOncallService()
    svc._bot = _Bot()
    svc._telegram_bot = _TelegramBot()
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock(return_value=issues)
    svc._process_ticket = AsyncMock()
    shift_start_dt = datetime(2026, 11, 10, 8, 0, 0)
    await svc._fire_for_shift("2026-11-10", "morning", shift_start_dt, [_make_shift()])
    # Only ISD-101 should reach _process_ticket
    assert svc._process_ticket.await_count == 1
    issue_arg = svc._process_ticket.await_args.args[0]
    assert issue_arg.get("key") == "ISD-101"


async def test_fire_for_shift_no_issues_no_call_to_holmes(fake_config, fake_mongo):
    mock_mongo, collections = fake_mongo
    svc = JiraOncallService()
    svc._bot = _Bot()
    svc._telegram_bot = _TelegramBot()
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock(return_value=[])
    svc._process_ticket = AsyncMock()
    shift_start_dt = datetime(2026, 11, 10, 8, 0, 0)
    await svc._fire_for_shift("2026-11-10", "morning", shift_start_dt, [_make_shift()])
    assert svc._process_ticket.await_count == 0


# ---------------------------------------------------------------------------
# _process_ticket — end-to-end happy path
# ---------------------------------------------------------------------------

async def test_process_ticket_creates_topic_then_calls_holmes_then_replies_then_marks(fake_config, fake_mongo):
    mock_mongo, collections = fake_mongo
    issue = _make_issue(key="ISD-200")
    svc = JiraOncallService()
    bot = _Bot()
    tg_bot = _TelegramBot()
    svc._bot = bot
    svc._telegram_bot = tg_bot
    svc._conversation_service = MagicMock()
    conv = MagicMock()
    svc._conversation_service.get_or_create_for_topic = AsyncMock(return_value=conv)
    svc._conversation_service.save_conversation = AsyncMock()
    svc._holmes = MagicMock()
    svc._holmes.chat = AsyncMock(return_value="all good — only read kubectl")

    await svc._process_ticket(issue, "2026-11-10", "morning", [12345])

    # topic created in the right chat
    bot.create_forum_topic.assert_awaited_once()
    assert bot.create_forum_topic.await_args.kwargs["chat_id"] == -1001234567890
    # header + reply sent
    assert bot.send_message.await_count == 2
    # Holmes.chat called with oncall_bot_user_id (NOT the human id)
    args, kwargs = svc._holmes.chat.await_args
    assert kwargs.get("user_id") == 0  # oncall_bot_user_id
    assert kwargs.get("user_permissions") == []
    # READ-ONLY rule in extra_prompt_additions
    assert "READ-ONLY" in kwargs.get("extra_prompt_additions", "")
    # Conversation persisted so follow-up chats from the on-call human have
    # context of the auto-investigation.
    svc._conversation_service.save_conversation.assert_awaited_once_with(conv)
    # mark fired in oncall_tickets
    collections["oncall_tickets"].update_one.assert_called()
    up_kwargs = collections["oncall_tickets"].update_one.call_args
    assert up_kwargs.args[0] == {"ticket_key": "ISD-200"}


async def test_process_ticket_posts_header_even_when_holmes_fails(fake_config, fake_mongo):
    mock_mongo, collections = fake_mongo
    issue = _make_issue(key="ISD-300")
    svc = JiraOncallService()
    bot = _Bot()
    svc._bot = bot
    svc._telegram_bot = _TelegramBot()
    svc._conversation_service = MagicMock()
    svc._conversation_service.get_or_create_for_topic = AsyncMock(
        return_value=MagicMock()
    )
    svc._holmes = MagicMock()
    svc._holmes.chat = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    await svc._process_ticket(issue, "2026-11-10", "morning", [12345])

    # topic + header + Holmes-failure message = 2 sends
    assert bot.create_forum_topic.await_count == 1
    assert bot.send_message.await_count == 2
    # The second message contains the failure marker
    second_text = bot.send_message.await_args_list[-1].kwargs["text"]
    assert "Auto-investigation failed" in second_text
    assert "https://servicedesk.fci.vn/browse/ISD-300" in second_text


async def test_process_ticket_marks_fired_after_holmes_succeeds(fake_config, fake_mongo):
    """The fired-after-success path calls oncall_tickets.update_one with the
    full doc — but if Holmes fails it returns early and does NOT mark fired."""
    mock_mongo, collections = fake_mongo
    issue = _make_issue(key="ISD-400")
    svc = JiraOncallService()
    svc._bot = _Bot()
    svc._telegram_bot = _TelegramBot()
    svc._conversation_service = MagicMock()
    svc._conversation_service.get_or_create_for_topic = AsyncMock(
        return_value=MagicMock()
    )
    svc._holmes = MagicMock()
    svc._holmes.chat = AsyncMock(side_effect=RuntimeError("boom"))

    await svc._process_ticket(issue, "2026-11-10", "morning", [12345])

    # update_one NOT called when Holmes failed (early return)
    collections["oncall_tickets"].update_one.assert_not_called()


# ---------------------------------------------------------------------------
# _fire_for_shift watchdog bump on new issues
# ---------------------------------------------------------------------------

async def test_tick_resets_watchdog_when_new_tickets_found(fake_config, fake_mongo):
    mock_mongo, collections = fake_mongo
    collections["oncall_shifts"].find.return_value = [_make_shift()]
    collections["oncall_tickets"].find_one.return_value = None
    issue = _make_issue(key="ISD-500")
    svc = JiraOncallService()
    bot = _Bot()
    tg_bot = _TelegramBot()
    tg_bot.last_update_time = 1000.0
    svc._bot = bot
    svc._telegram_bot = tg_bot
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock(return_value=[issue])
    svc._conversation_service = MagicMock()
    svc._conversation_service.get_or_create_for_topic = AsyncMock(
        return_value=MagicMock()
    )
    svc._holmes = MagicMock()
    svc._holmes.chat = AsyncMock(return_value="ok")
    dt = datetime(2026, 11, 10, 14, 0)  # mid-morning, in shift
    with patch("src.bot.services.jira_oncall_service.datetime") as mock_dt:
        mock_dt.now.return_value = dt
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        await svc._tick()
    # last_update_time bumped past the initial 1000.0
    assert tg_bot.last_update_time > 1000.0


async def test_no_new_issues_still_resets_watchdog(fake_config, fake_mongo):
    """All tickets already fired → watchdog STILL reset because a shift
    is active. The worker bumps the watchdog unconditionally per tick
    when a registrant exists, so quiet stretches during a shift don't
    nuke the container via the 90m idle exit."""
    mock_mongo, collections = fake_mongo
    collections["oncall_shifts"].find.return_value = [_make_shift()]
    collections["oncall_tickets"].find_one.return_value = {
        "ticket_key": "ISD-1",
    }
    svc = JiraOncallService()
    bot = _Bot()
    tg_bot = _TelegramBot()
    tg_bot.last_update_time = 1000.0
    svc._bot = bot
    svc._telegram_bot = tg_bot
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock(return_value=[_make_issue(key="ISD-1")])
    dt = datetime(2026, 11, 10, 14, 0)
    with patch("src.bot.services.jira_oncall_service.datetime") as mock_dt:
        mock_dt.now.return_value = dt
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        await svc._tick()
    # Watchdog bumped even though all tickets deduped out.
    assert tg_bot.last_update_time > 1000.0


# ---------------------------------------------------------------------------
# READ_ONLY rule constant
# ---------------------------------------------------------------------------

def test_read_only_rule_forbids_mutations():
    rule = JiraOncallService.READ_ONLY_RULE
    assert "kubectl delete" in rule
    assert "kubectl apply" in rule
    assert "kubectl patch" in rule
    # openstack verbs are listed with `...` placeholder
    assert "openstack ... delete" in rule
    assert "openstack ... set" in rule
    assert "openstack ... update" in rule
    assert "DO NOT MODIFY" in rule
    assert "READ-ONLY" in rule


def test_read_only_rule_mentions_two_layer_enforcement():
    rule = JiraOncallService.READ_ONLY_RULE
    assert "TWO layers" in rule
    assert "admin_gate" in rule


# ---------------------------------------------------------------------------
# fire_now (for /oncall test subcommand)
# ---------------------------------------------------------------------------

async def test_fire_now_with_no_shifts_returns_friendly_error(fake_config, fake_mongo):
    mock_mongo, collections = fake_mongo
    collections["oncall_shifts"].find.return_value = []
    svc = JiraOncallService()
    result = await svc.fire_now("2026-11-10", "morning")
    assert result["error"]
    assert "no oncall shifts" in result["error"]


async def test_fire_now_fires_for_existing_shift(fake_config, fake_mongo):
    mock_mongo, collections = fake_mongo
    collections["oncall_shifts"].find.return_value = [_make_shift()]
    svc = JiraOncallService()
    svc._bot = _Bot()
    svc._telegram_bot = _TelegramBot()
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock(return_value=[])
    svc._process_ticket = AsyncMock()
    result = await svc.fire_now("2026-11-10", "morning")
    assert result.get("date") == "2026-11-10"
    assert result.get("shift") == "morning"
    assert "elapsed_seconds" in result


async def test_fire_now_auto_picks_other_shift_when_first_has_no_registrants(
    fake_config, fake_mongo,
):
    """/oncall test (no args) at 14:00 defaults to morning. If no morning
    registrant exists but night has one, fall back to night. Stops the
    "I registered night but tested mid-morning → no oncall shifts" UX
    trap."""
    mock_mongo, collections = fake_mongo
    # First find() (morning, auto-pick) returns [] — no registrants.
    # Second find() (night, fallback) returns the registered shift.
    collections["oncall_shifts"].find.side_effect = [
        [],                                        # morning lookup
        [_make_shift(date="2026-11-10", shift="night")],  # night fallback
    ]
    svc = JiraOncallService()
    svc._bot = _Bot()
    svc._telegram_bot = _TelegramBot()
    svc._mcp = MagicMock()
    svc._mcp.search_issues = AsyncMock(return_value=[])
    svc._process_ticket = AsyncMock()
    # fire_now with no shift → auto-pick morning, fallback night.
    with patch("src.bot.services.jira_oncall_service.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 11, 10, 14, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = await svc.fire_now("2026-11-10")
    assert result.get("shift") == "night"
    assert result.get("date") == "2026-11-10"
    assert "error" not in result


async def test_fire_now_explicit_shift_no_fallback(fake_config, fake_mongo):
    """If caller explicitly passes a shift, don't fallback to the other
    one — they asked for that specific shift, return the friendly error
    if it's empty."""
    mock_mongo, collections = fake_mongo
    collections["oncall_shifts"].find.return_value = []
    svc = JiraOncallService()
    result = await svc.fire_now("2026-11-10", "morning")
    assert result.get("error")
    assert "no oncall shifts" in result["error"]
    assert result.get("shift") == "morning"
