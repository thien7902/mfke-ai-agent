"""Tests for the whole-graph facts injection into Holmes' system prompt.

Covers the MemoryService fact cache (`_facts_cache`, `get_cached_facts_sync`,
`_fetch_all_facts_on_loop`, `invalidate_facts_cache`) and the HolmesService
facts-block injection in `_build_holmes_messages`.

The Graphiti deps are NOT installed on the test machine — we test the cache
logic (TTL, in-flight guard, invalidation) directly and stub MemoryService
for the HolmesService injection tests.
"""
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.bot.services.memory_service import MemoryService
from src.bot.services.holmes_service import HolmesService


# ---------- MemoryService fact cache ----------

def _make_memory_service_stub() -> MemoryService:
    """Build a MemoryService whose _graphiti is faked (non-None) so
    get_cached_facts_sync passes its graphiti-is-None guard. We bypass
    Graphiti's driver entirely in tests."""
    svc = MemoryService.__new__(MemoryService)
    svc._graphiti = MagicMock()  # non-None so the guard passes
    svc._initialized = True
    svc._facts_cache = None
    svc._facts_refresh_in_flight = False
    svc._loop = None
    svc._loop_thread = None
    svc._loop_lock = __import__("threading").Lock()
    svc._facts_lock = __import__("threading").Lock()
    svc._init_lock = asyncio.Lock()
    return svc


def test_get_cached_facts_sync_cold_returns_empty_and_schedules_refresh():
    """Cold cache (no snapshot yet) → returns []. A background refresh is
    scheduled (we verify via patching _schedule_facts_refresh_unlocked)."""
    svc = _make_memory_service_stub()
    with patch.object(svc, "_schedule_facts_refresh_unlocked") as sched, \
         patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        result = svc.get_cached_facts_sync()
    assert result == []
    sched.assert_called_once()


def test_get_cached_facts_sync_hot_returns_cached_list_no_refresh():
    """Within TTL, returns the cached list and does NOT schedule a refresh."""
    svc = _make_memory_service_stub()
    svc._write_facts_cache(["fact a", "fact b"])
    with patch.object(svc, "_schedule_facts_refresh_unlocked") as sched, \
         patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        result = svc.get_cached_facts_sync()
    assert result == ["fact a", "fact b"]
    sched.assert_not_called()


def test_get_cached_facts_sync_stale_schedules_refresh_returns_old_snapshot():
    """Stale cache → returns the prior snapshot AND schedules a refresh."""
    svc = _make_memory_service_stub()
    svc._write_facts_cache(["old fact"])
    # Force the timestamp to 0 so the cache is definitely stale.
    with svc._facts_lock:
        svc._facts_cache = (0.0, ["old fact"])
    with patch.object(svc, "_schedule_facts_refresh_unlocked") as sched, \
         patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        result = svc.get_cached_facts_sync()
    assert result == ["old fact"], "stale read must return prior snapshot"
    sched.assert_called_once()


def test_get_cached_facts_sync_disabled_returns_empty_no_refresh():
    """When graphiti_memory_enabled=False, no I/O and no refresh scheduled."""
    svc = _make_memory_service_stub()
    with patch.object(svc, "_schedule_facts_refresh_unlocked") as sched, \
         patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = False
        result = svc.get_cached_facts_sync()
    assert result == []
    sched.assert_not_called()


def test_get_cached_facts_sync_refresh_in_flight_does_not_double_schedule():
    """If a refresh is already running, a stale read returns the snapshot
    without scheduling a second refresh."""
    svc = _make_memory_service_stub()
    with svc._facts_lock:
        svc._facts_cache = (0.0, ["prior"])
        svc._facts_refresh_in_flight = True
    with patch.object(svc, "_schedule_facts_refresh_unlocked") as sched, \
         patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        result = svc.get_cached_facts_sync()
    assert result == ["prior"]
    sched.assert_not_called()


@pytest.mark.asyncio
async def test_invalidate_facts_cache_bumps_to_stale_keeps_snapshot():
    """invalidate_facts_cache sets ts=0 but keeps the fact list. The next
    get_cached_facts_sync will trigger a refresh because ts=0 is stale."""
    svc = _make_memory_service_stub()
    svc._write_facts_cache(["fact one"])
    await svc.invalidate_facts_cache()
    with svc._facts_lock:
        ts, facts = svc._facts_cache
    assert ts == 0.0
    assert facts == ["fact one"], "snapshot must survive invalidation"


@pytest.mark.asyncio
async def test_invalidate_facts_cache_no_op_when_cache_is_none():
    """Calling invalidate when there's never been a cache must not crash."""
    svc = _make_memory_service_stub()
    svc._facts_cache = None
    await svc.invalidate_facts_cache()
    assert svc._facts_cache is None


# ---------- _fetch_all_facts_on_loop (Graphiti API surface) ----------

@pytest.mark.asyncio
async def test_fetch_all_facts_on_loop_extracts_fact_from_edges():
    """Stub the driver.execute_query path and confirm we extract .fact.
    v2.3.21 bypasses EntityEdge.get_by_group_ids (which routes to a
    per-group database and misses on default_db-stored edges) and
    queries the driver directly with the same WHERE e.group_id IN [...]
    filter the search path uses."""
    import datetime as _dt
    svc = _make_memory_service_stub()
    svc._graphiti = MagicMock()
    driver = MagicMock()
    driver.provider = "falkordb"

    # Records must round-trip through entity_edge_from_record which
    # builds an EntityEdge (pydantic) — `created_at` must be a datetime.
    now = _dt.datetime.now(_dt.timezone.utc)
    rec1 = {"uuid": "u1", "source_node_uuid": "s1",
            "target_node_uuid": "t1", "group_id": "global",
            "created_at": now, "name": "n1",
            "fact": "acme runs k8s 1.29", "episodes": [],
            "expired_at": None, "valid_at": None,
            "invalid_at": None, "reference_time": None,
            "attributes": {}}
    rec2 = {"uuid": "u2", "source_node_uuid": "s2",
            "target_node_uuid": "t2", "group_id": "global",
            "created_at": now, "name": "n2",
            "fact": "gardener shoots seeds", "episodes": [],
            "expired_at": None, "valid_at": None,
            "invalid_at": None, "reference_time": None,
            "attributes": {}}
    rec3 = dict(rec1, uuid="u3", name="n3", fact="")  # empty fact filtered out
    driver.execute_query = AsyncMock(return_value=([rec1, rec2, rec3], None, None))
    svc._graphiti.driver = driver

    facts = await svc._fetch_all_facts_on_loop()
    assert facts == ["acme runs k8s 1.29", "gardener shoots seeds"]
    # Cache updated as a side-effect.
    assert svc._facts_cache is not None
    assert svc._facts_cache[1] == facts


@pytest.mark.asyncio
async def test_fetch_all_facts_on_loop_failure_preserves_prior_cache():
    """On FalkorDB failure, the prior snapshot is preserved (returns it
    instead of wiping the cache to [])."""
    svc = _make_memory_service_stub()
    svc._write_facts_cache(["previously cached"])
    svc._graphiti = MagicMock()
    driver = MagicMock()
    driver.provider = "falkordb"
    driver.execute_query = AsyncMock(side_effect=RuntimeError("falkordb down"))
    svc._graphiti.driver = driver

    result = await svc._fetch_all_facts_on_loop()
    assert result == ["previously cached"], "failure must return prior snapshot"
    with svc._facts_lock:
        ts, facts = svc._facts_cache
    assert facts == ["previously cached"], "cache must NOT be wiped on failure"


@pytest.mark.asyncio
async def test_fetch_all_facts_on_loop_empty_graph_caches_empty_list():
    """Empty result (no edges in this group on default_db) caches (ts, [])
    so subsequent turns don't retry every tick. Distinct from a real
    failure which preserves the prior snapshot."""
    svc = _make_memory_service_stub()
    svc._write_facts_cache(["stale from before"])  # prior snapshot exists
    svc._graphiti = MagicMock()
    driver = MagicMock()
    driver.provider = "falkordb"
    driver.execute_query = AsyncMock(return_value=([], None, None))
    svc._graphiti.driver = driver

    result = await svc._fetch_all_facts_on_loop()
    assert result == []
    with svc._facts_lock:
        ts, facts = svc._facts_cache
    assert facts == [], "empty graph must overwrite prior snapshot with []"
    assert ts > 0, "timestamp must refresh so TTL window resets"


# ---------- HolmesService._build_holmes_messages injection ----------

def _make_holmes_service_stub(memory_service=None):
    """Build a HolmesService stub — we only call _build_holmes_messages on
    it, so we skip __init__ (which would try to load Holmes config)."""
    svc = HolmesService.__new__(HolmesService)
    svc._memory = memory_service
    svc._tool_executor = MagicMock()
    svc._config = None
    return svc


def _build_messages_text(svc, messages=None):
    """Call _build_holmes_messages and concatenate the resulting message
    contents so we can grep for substrings."""
    out = svc._build_holmes_messages([], 1, is_admin=False,
                                     user_display_name="tester",
                                     conversation=None)
    parts = []
    for m in out:
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content")
        if content:
            parts.append(content if isinstance(content, str) else str(content))
    return "\n".join(parts)


# "KNOWLEDGE MEMORY" alone appears both in the tool-rule text ("see the
# 'KNOWLEDGE MEMORY' block above") and the facts-block header. Use the
# full header marker to detect the actual injection.
_FACTS_BLOCK_MARKER = "KNOWLEDGE MEMORY (whole graph dump)"


def test_build_holmes_messages_includes_facts_block_when_memory_enabled_and_non_empty():
    """When memory is enabled and the cache has facts, the system prompt
    contains the KNOWLEDGE MEMORY header and each fact as a bullet."""
    memory = _make_memory_service_stub()
    memory._write_facts_cache(["acme runs k8s 1.29", "gardener uses seeds v2"])

    svc = _make_holmes_service_stub(memory_service=memory)
    with patch("src.bot.services.holmes_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.holmes_system_prompt_additions = ""
        with patch.object(memory, "get_cached_facts_sync",
                          return_value=["acme runs k8s 1.29", "gardener uses seeds v2"]):
            text = _build_messages_text(svc)
    assert _FACTS_BLOCK_MARKER in text
    assert "acme runs k8s 1.29" in text
    assert "gardener uses seeds v2" in text


def test_build_holmes_messages_skips_facts_block_when_cache_empty():
    """Cold cache → no facts-block header in the prompt. The LLM must not
    see an empty 'KNOWLEDGE MEMORY' section."""
    memory = _make_memory_service_stub()

    svc = _make_holmes_service_stub(memory_service=memory)
    with patch("src.bot.services.holmes_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.holmes_system_prompt_additions = ""
        with patch.object(memory, "get_cached_facts_sync", return_value=[]):
            text = _build_messages_text(svc)
    assert _FACTS_BLOCK_MARKER not in text


def test_build_holmes_messages_skips_facts_block_when_memory_disabled():
    """When graphiti_memory_enabled=False, neither the tool rule nor the
    facts block is appended."""
    memory = _make_memory_service_stub()

    svc = _make_holmes_service_stub(memory_service=memory)
    with patch("src.bot.services.holmes_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = False
        cfg.holmes_system_prompt_additions = ""
        with patch.object(memory, "get_cached_facts_sync",
                          return_value=["should not appear"]) as get_call:
            text = _build_messages_text(svc)
    assert _FACTS_BLOCK_MARKER not in text
    assert "should not appear" not in text
    assert "search_memory TOOL USE POLICY" not in text
    get_call.assert_not_called()


def test_build_holmes_messages_skips_facts_block_when_no_memory_service():
    """If HolmesService was constructed without a MemoryService (the
    default for deployments without memory), the facts block is skipped
    even when graphiti_memory_enabled is True — no NoneType crash."""
    svc = _make_holmes_service_stub(memory_service=None)
    with patch("src.bot.services.holmes_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.holmes_system_prompt_additions = ""
        # Must not raise AttributeError.
        text = _build_messages_text(svc)
    assert _FACTS_BLOCK_MARKER not in text
