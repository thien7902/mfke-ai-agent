"""Tests for ConsolidatorService — the idle-triggered background distiller.

Mocks:
- `_call_consolidator_llm` patched at the module path so no network/LLM.
- `ConversationService` mocked via AsyncMock — get_conversation_by_topic returns
  a Conversation with a configurable message list, or None.
- `DistilledFactStore` mocked via MagicMock + AsyncMock for `add` / `record_run`.

Idle-guard semantics tested via the `_last_distilled_count` checkpoint. The
cooldown test ( formerly seconds-based ) is replaced by the count-based
"no new messages" test.
"""
import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.bot.models.conversation import Conversation, Message
from src.bot.services.consolidator_service import ConsolidatorService
from src.bot.utils.config import config as bot_config


def _conversation(messages):
    conv = Conversation(user_id=1, topic_id=10, chat_id=20)
    conv.messages = list(messages)
    return conv


def _msg(role, content):
    return Message(role=role, content=content)


@pytest.fixture
def mock_store():
    """DistilledFactStore mock with AsyncMock for add + record_run."""
    store = MagicMock()
    store.add = AsyncMock(return_value=True)
    store.record_run = AsyncMock()
    return store


@pytest.fixture
def mock_conversations():
    svc = MagicMock()
    svc.get_conversation_by_topic = AsyncMock()
    return svc


@pytest.fixture
def svc(mock_store, mock_conversations):
    return ConsolidatorService(mock_store, mock_conversations)


# --------------------------------------------------------------------- #
# schedule_if_idle + _fire debounce
# --------------------------------------------------------------------- #

def test_schedule_noop_when_disabled(svc, mock_config):
    mock_config.consolidator_enabled = False
    svc.schedule_if_idle(chat_id=1, topic_id=2)
    assert svc._timers == {}  # nothing armed


@pytest.mark.asyncio
async def test_fire_no_conversation(svc, mock_conversations):
    """conversation=None → no LLM call, no store writes."""
    mock_conversations.get_conversation_by_topic.return_value = None
    await svc._fire((20, 10))
    mock_conversations.get_conversation_by_topic.assert_awaited_once()
    mock_store = svc._store
    mock_store.add.assert_not_called()
    mock_store.record_run.assert_not_called()


@pytest.mark.asyncio
async def test_fire_skip_too_few_turns(svc, mock_conversations):
    """Under min_turns → log + no LLM call."""
    mock_conversations.get_conversation_by_topic.return_value = _conversation(
        [_msg("user", "hi"), _msg("assistant", "hello")]  # 2 messages, < min_turns(4)
    )
    with patch("src.bot.services.consolidator_service._call_consolidator_llm") as llm:
        await svc._fire((20, 10))
        llm.assert_not_called()
    svc._store.add.assert_not_called()
    svc._store.record_run.assert_not_called()


@pytest.mark.asyncio
async def test_fire_skip_when_no_new_messages(svc, mock_conversations):
    """Idle-guard: if message count == checkpoint, skip without LLM call."""
    msgs = []
    for i in range(5):
        msgs.append(_msg("user", f"q{i}"))
        msgs.append(_msg("assistant", f"a{i}"))
    mock_conversations.get_conversation_by_topic.return_value = _conversation(msgs)
    svc._last_distilled_count[(20, 10)] = 10  # checkpoint already at 10
    with patch("src.bot.services.consolidator_service._call_consolidator_llm") as llm:
        await svc._fire((20, 10))
        llm.assert_not_called()
    svc._store.add.assert_not_called()
    svc._store.record_run.assert_not_called()


@pytest.mark.asyncio
async def test_fire_skip_when_llm_raises(svc, mock_conversations):
    """LLM exception → log + no store.add, no record_run, no checkpoint update."""
    msgs = [_msg("user", f"q{i}") for i in range(4)] + [_msg("assistant", "a") for i in range(2)]
    mock_conversations.get_conversation_by_topic.return_value = _conversation(msgs)
    with patch("src.bot.services.consolidator_service._call_consolidator_llm",
               side_effect=RuntimeError("llm down")):
        await svc._fire((20, 10))
    svc._store.add.assert_not_called()
    svc._store.record_run.assert_not_called()
    # Checkpoint NOT updated so a retry with same count will still skip, but
    # a future fire with more messages will retry.
    assert (20, 10) not in svc._last_distilled_count


@pytest.mark.asyncio
async def test_fire_empty_facts_still_records_run_and_advances_checkpoint(
    svc, mock_conversations
):
    """LLM returns [] → 0 add calls, but record_run IS called and checkpoint
    advances. This means an empty run replaces the topic's prior injection
    set with nothing — "newer run wins" semantics.
    """
    msgs = [
        _msg("user", "q1"), _msg("assistant", "a1"),
        _msg("user", "q2"), _msg("assistant", "a2"),
    ]
    mock_conversations.get_conversation_by_topic.return_value = _conversation(msgs)
    with patch("src.bot.services.consolidator_service._call_consolidator_llm",
               return_value=[]):
        await svc._fire((20, 10))
    svc._store.add.assert_not_called()
    svc._store.record_run.assert_awaited_once()
    # record_run args
    args, _ = svc._store.record_run.call_args
    assert args[0] == 20  # chat_id
    assert args[1] == 10  # topic_id
    # Only assert non-empty run_id (uuid varies)
    assert isinstance(args[2], str) and len(args[2]) > 0
    # Checkpoint advanced → next fire with 4 messages would skip.
    assert svc._last_distilled_count[(20, 10)] == 4


@pytest.mark.asyncio
async def test_fire_three_new_facts(svc, mock_conversations):
    """LLM returns 3 facts → 3 store.add calls with same run_id, +1 record_run."""
    msgs = [
        _msg("user", "q1"), _msg("assistant", "a1"),
        _msg("user", "q2"), _msg("assistant", "a2"),
    ]
    mock_conversations.get_conversation_by_topic.return_value = _conversation(msgs)
    facts = [{"text": "fact1"}, {"text": "fact2"}, {"text": "fact3"}]
    with patch("src.bot.services.consolidator_service._call_consolidator_llm",
               return_value=facts):
        await svc._fire((20, 10))
    assert svc._store.add.await_count == 3
    # All 3 adds must share the same run_id (consolidator_service.add is
    # called with `source_run_id=...` kwarg).
    run_ids = {call.kwargs["source_run_id"] for call in svc._store.add.await_args_list}
    assert len(run_ids) == 1
    svc._store.record_run.assert_awaited_once()
    assert svc._last_distilled_count[(20, 10)] == 4


@pytest.mark.asyncio
async def test_fire_one_add_raises_continues_with_others(svc, mock_conversations):
    """Per-fact try/except: if add raises on a middle fact, others still run."""
    msgs = [
        _msg("user", "q1"), _msg("assistant", "a1"),
        _msg("user", "q2"), _msg("assistant", "a2"),
    ]
    mock_conversations.get_conversation_by_topic.return_value = _conversation(msgs)
    facts = [{"text": "fact1"}, {"text": "fact2"}, {"text": "fact3"}]
    svc._store.add = AsyncMock(side_effect=[True, RuntimeError("db blip"), True])
    with patch("src.bot.services.consolidator_service._call_consolidator_llm",
               return_value=facts):
        await svc._fire((20, 10))
    assert svc._store.add.await_count == 3
    # record_run still fires — partial success still counts as "the run happened"
    svc._store.record_run.assert_awaited_once()
    assert svc._last_distilled_count[(20, 10)] == 4


@pytest.mark.asyncio
async def test_re_arm_while_in_flight_is_noop(svc, mock_conversations):
    """If the lock is held, a second _fire returns immediately."""
    msgs = [
        _msg("user", "q1"), _msg("assistant", "a1"),
        _msg("user", "q2"), _msg("assistant", "a2"),
    ]
    mock_conversations.get_conversation_by_topic.return_value = _conversation(msgs)
    # Acquire the lock manually before _fire runs.
    key = (20, 10)
    lock = asyncio.Lock()
    await lock.acquire()
    svc._locks[key] = lock
    with patch("src.bot.services.consolidator_service._call_consolidator_llm") as llm:
        await svc._fire(key)
        llm.assert_not_called()
    lock.release()


@pytest.mark.asyncio
async def test_stop_cancels_timers_and_is_idempotent(svc):
    handle = MagicMock()
    svc._timers[(20, 10)] = handle
    svc._locks[(20, 10)] = MagicMock()
    svc._last_distilled_count[(20, 10)] = 5
    await svc.stop()
    handle.cancel.assert_called_once()
    assert svc._timers == {} and svc._locks == {} and svc._last_distilled_count == {}
    # Second stop is a no-op (no exceptions).
    await svc.stop()


@pytest.mark.asyncio
async def test_clear_then_new_chat_grows_past_checkpoint(svc, mock_conversations):
    """Verify /clear + new chat past old checkpoint triggers a new run.

    Sequence:
    1. Checkpoint = 10 from a prior run.
    2. /clear deletes the conversation in MongoDB; get_conversation_by_topic
       returns a conversation with only 2 new messages → `2 <= 10` → skip.
    3. New chat grows to 12 → fire → `12 > 10` → distill, checkpoint → 12.
    """
    # First fire: 2 messages, checkpoint at 10 → skip.
    svc._last_distilled_count[(20, 10)] = 10
    mock_conversations.get_conversation_by_topic.return_value = _conversation(
        [_msg("user", "q1"), _msg("assistant", "a1")]
    )
    with patch("src.bot.services.consolidator_service._call_consolidator_llm") as llm:
        await svc._fire((20, 10))
        llm.assert_not_called()

    # Second fire: conversation grew to 12 → distill.
    new_msgs = [_msg("user", f"q{i}") for i in range(6)] + [
        _msg("assistant", f"a{i}") for i in range(6)
    ]
    mock_conversations.get_conversation_by_topic.return_value = _conversation(new_msgs)
    facts = [{"text": "newfact"}]
    with patch("src.bot.services.consolidator_service._call_consolidator_llm",
               return_value=facts) as llm:
        await svc._fire((20, 10))
        llm.assert_called_once()
    svc._store.record_run.assert_awaited_once()
    assert svc._last_distilled_count[(20, 10)] == 12


# --------------------------------------------------------------------- #
# Cross-topic semaphore + LLM timeout + stop() drain (v2.3.29)
# --------------------------------------------------------------------- #

def _make_distillable_conversation():
    """4 messages ≥ consolidator_min_turns(4) so _consolidate proceeds to the
    LLM call."""
    return _conversation(
        [
            _msg("user", "q1"), _msg("assistant", "a1"),
            _msg("user", "q2"), _msg("assistant", "a2"),
        ]
    )


@pytest.mark.asyncio
async def test_semaphore_limits_concurrent_llm_calls(svc, mock_conversations, monkeypatch):
    """max_concurrent=5: spawn 7 distinct-topic fires simultaneously; only 5
    should be inside the LLM call at once. The other 2 must wait.

    `_call_consolidator_llm` is patched as a SYNC function backing onto a
    `threading.Event`. _consolidate runs the LLM call via
    `asyncio.to_thread(...)`, so the LLM bodies execute in the default
    ThreadPoolExecutor — but the `asyncio.Semaphore(5)` is acquired async
    BEFORE the to_thread dispatch. So even though the sync stubs would have
    worker threads available, only 5 coroutines can be inside the
    semaphore block at once, which is what we're measuring.
    """
    mock_conversations.get_conversation_by_topic.return_value = _make_distillable_conversation()
    monkeypatch.setattr(bot_config, "consolidator_max_concurrent", 5)
    monkeypatch.setattr(bot_config, "consolidator_llm_timeout_seconds", 30)
    svc._semaphore = asyncio.Semaphore(5)

    import threading as _threading
    release = _threading.Event()
    sync_in_flight = {"count": 0, "max": 0, "lock": _threading.Lock()}

    def _sync_slow_llm(transcript):
        with sync_in_flight["lock"]:
            sync_in_flight["count"] += 1
            sync_in_flight["max"] = max(sync_in_flight["max"], sync_in_flight["count"])
        try:
            release.wait(timeout=5)
        finally:
            with sync_in_flight["lock"]:
                sync_in_flight["count"] -= 1
        return []

    with patch(
        "src.bot.services.consolidator_service._call_consolidator_llm",
        side_effect=_sync_slow_llm,
    ):
        tasks = []
        for tid in range(7):
            key = (20, tid)
            tasks.append(asyncio.ensure_future(svc._fire(key)))
        await asyncio.sleep(0.2)
        assert sync_in_flight["max"] <= 5, f"max_concurrent exceeded: {sync_in_flight['max']}"
        assert sync_in_flight["max"] == 5, f"expected 5 concurrent, got {sync_in_flight['max']}"
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_llm_timeout_releases_semaphore_and_returns(svc, mock_conversations, monkeypatch):
    """consolidator_llm_timeout_seconds=60 (test: 0.05) wraps the to_thread LLM
    call. On timeout, the function returns early (no record_run, no
    checkpoint) and the semaphore slot is released.
    """
    monkeypatch.setattr(bot_config, "consolidator_llm_timeout_seconds", 0.05)
    monkeypatch.setattr(bot_config, "consolidator_max_concurrent", 5)
    svc._semaphore = asyncio.Semaphore(5)
    mock_conversations.get_conversation_by_topic.return_value = _make_distillable_conversation()

    # Stub a sync LLM that sleeps much longer than the timeout.
    def _slow_llm(transcript):
        import time as _t
        _t.sleep(2)
        return []

    with patch(
        "src.bot.services.consolidator_service._call_consolidator_llm",
        side_effect=_slow_llm,
    ):
        await svc._fire((20, 10))

    # No record_run, no checkpoint (early-return on timeout).
    svc._store.record_run.assert_not_awaited()
    assert (20, 10) not in svc._last_distilled_count
    # Semaphore released and re-acquirable immediately.
    async with svc._semaphore:
        pass


@pytest.mark.asyncio
async def test_stop_awaits_in_flight_fires(svc, mock_conversations, monkeypatch):
    """stop() drains in-flight _fire tasks (gather with return_exceptions)
    rather than cancelling them. We start a fire on a slow LLM, then call
    stop() while it's still running — stop() should await its completion."""
    monkeypatch.setattr(bot_config, "consolidator_max_concurrent", 5)
    monkeypatch.setattr(bot_config, "consolidator_llm_timeout_seconds", 30)
    svc._semaphore = asyncio.Semaphore(5)
    mock_conversations.get_conversation_by_topic.return_value = _make_distillable_conversation()

    release = asyncio.Event()
    completed = {"value": False}

    async def _slow_async():
        await release.wait()
        completed["value"] = True
        return []

    # Patch the LLM call so _consolidate's `asyncio.wait_for` path hits it
    # via asyncio.to_thread. to_thread wraps a SYNC function — so we use a
    # sync stub that blocks on a threading.Event and check that stop()
    # awaits the in-flight _fire task.
    import threading as _threading
    release_sync = _threading.Event()

    def _sync_llm(transcript):
        release_sync.wait(timeout=5)
        return []

    with patch(
        "src.bot.services.consolidator_service._call_consolidator_llm",
        side_effect=_sync_llm,
    ):
        # Schedule a fire as a tracked task (mirrors how the timer does it).
        task = asyncio.ensure_future(svc._fire_and_track((20, 10)))
        await asyncio.sleep(0.1)  # let it enter the LLM call
        assert len(svc._in_flight) == 1, "expected 1 in-flight fire"
        # Issue stop() — should drain the in-flight fire.
        release_sync.set()
        await asyncio.wait_for(svc.stop(), timeout=5)
        assert task.done()
        assert svc._in_flight == set()
