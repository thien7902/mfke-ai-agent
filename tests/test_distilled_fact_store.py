"""Tests for DistilledFactStore — the MongoDB-backed store of distilled facts.

Patches MongoDB via the autouse `mock_mongodb` fixture in conftest.py. To
bypass the lazy collection accessors, each test injects a MagicMock
directly into `_facts` / `_runs` so the property `facts`/`runs` returns
it without touching the real MongoDB wiring.
"""
import datetime

import pytest
from pymongo.errors import DuplicateKeyError
from unittest.mock import MagicMock, patch

from src.bot.services import distilled_fact_store as dfs_mod
from src.bot.services.distilled_fact_store import DistilledFactStore


def _store_with(facts=None, runs=None):
    """Construct a DistilledFactStore and pre-populate its private
    collection attributes so the lazy properties never call Mongo."""
    s = DistilledFactStore()
    s._facts = facts if facts is not None else MagicMock()
    s._runs = runs if runs is not None else MagicMock()
    return s


@pytest.mark.asyncio
async def test_add_returns_true_on_new_fact():
    facts = MagicMock()
    facts.insert_one.return_value = MagicMock(inserted_id="oid")
    store = _store_with(facts=facts)
    added = await store.add("MFKE api-gw runs on port 8080", 1, 2, "run-1")
    assert added is True
    facts.insert_one.assert_called_once()
    doc = facts.insert_one.call_args[0][0]
    assert doc["text"] == "MFKE api-gw runs on port 8080"
    assert doc["source_chat_id"] == 1
    assert doc["source_topic_id"] == 2
    assert doc["source_run_id"] == "run-1"
    assert isinstance(doc["created_at"], datetime.datetime)


@pytest.mark.asyncio
async def test_add_returns_false_on_duplicate():
    facts = MagicMock()
    facts.insert_one.side_effect = DuplicateKeyError("dup")
    store = _store_with(facts=facts)
    added = await store.add("dup text", 1, 2, "run-1")
    assert added is False


@pytest.mark.asyncio
async def test_add_returns_false_on_empty_text():
    facts = MagicMock()
    store = _store_with(facts=facts)
    assert await store.add("   ", 1, 2, "run-1") is False
    assert await store.add("", 1, 2, "run-1") is False
    facts.insert_one.assert_not_called()


@pytest.mark.asyncio
async def test_record_run_upserts_latest_run():
    runs = MagicMock()
    store = _store_with(runs=runs)
    await store.record_run(chat_id=1, topic_id=2, run_id="run-xyz")
    runs.update_one.assert_called_once()
    args, kwargs = runs.update_one.call_args
    assert args[0] == {"chat_id": 1, "topic_id": 2}
    assert args[1]["$set"]["latest_run_id"] == "run-xyz"
    assert args[1]["$set"]["chat_id"] == 1
    assert args[1]["$set"]["topic_id"] == 2
    assert kwargs.get("upsert") is True


def test_get_injectable_facts_empty_when_no_runs():
    runs = MagicMock()
    runs.find.return_value = []
    store = _store_with(runs=runs)
    assert store.get_injectable_facts() == []


def test_get_injectable_facts_returns_only_latest_run_per_topic():
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [
        {"latest_run_id": "runA2", "topic_id": 10},
        {"latest_run_id": "runB1", "topic_id": 20},
    ]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        {"text": "fact A2a", "source_run_id": "runA2", "source_topic_id": 10},
        {"text": "fact A2b", "source_run_id": "runA2", "source_topic_id": 10},
        {"text": "fact B1", "source_run_id": "runB1", "source_topic_id": 20},
    ])
    facts.find.return_value = cursor

    store = _store_with(facts=facts, runs=runs)
    out = store.get_injectable_facts()

    find_filter = facts.find.call_args[0][0]
    assert find_filter == {"source_run_id": {"$in": ["runA2", "runB1"]}}
    assert len(out) == 3
    assert all(f["source_run_id"] in ("runA2", "runB1") for f in out)


def test_get_injectable_facts_swallows_db_error():
    runs = MagicMock()
    runs.find.side_effect = Exception("mongo down")
    store = _store_with(runs=runs)
    assert store.get_injectable_facts() == []


def test_get_injectable_facts_skips_dedup_when_threshold_zero():
    """Threshold 0 → dedup off, full list returned even though embeddings
    would collapse identical-vector facts if dedup ran."""
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        {"text": "alpha", "source_run_id": "r1", "source_topic_id": 5, "created_at": 1},
        {"text": "alpha dup", "source_run_id": "r1", "source_topic_id": 5, "created_at": 2},
    ])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)
    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.0):
        out = store.get_injectable_facts()
    assert len(out) == 2


def test_get_injectable_facts_dedups_near_duplicate_facts():
    """Two facts above the cosine threshold get collapsed; newest (per
    created_at) wins; dissimilar facts pass through."""
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        # Ascending created_at — oldest first.
        {"text": "old OOM rule", "source_run_id": "r1", "source_topic_id": 5, "created_at": 1},
        {"text": "newest OOM rule", "source_run_id": "r1", "source_topic_id": 5, "created_at": 3},
        {"text": "totally unrelated disk fact", "source_run_id": "r1", "source_topic_id": 5, "created_at": 2},
    ])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)

    # Embeddings: dup pair shares the same vector; unrelated fact orthogonal.
    # embed_chunks is called in input order — same as the list order above.
    vectors = [
        [1.0, 0.0, 0.0],   # old OOM rule
        [1.0, 0.0, 0.0],   # newest OOM rule — identical vector → cosine 1.0 ≥ threshold
        [0.0, 1.0, 0.0],   # disk fact — orthogonal to both
    ]
    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.92), \
         patch("src.bot.services.embedder.embed_chunks", return_value=vectors):
        out = store.get_injectable_facts()

    # Newest wins → "newest OOM rule" kept, "old OOM rule" dropped.
    texts = [f["text"] for f in out]
    assert "newest OOM rule" in texts
    assert "old OOM rule" not in texts
    assert "totally unrelated disk fact" in texts
    assert len(out) == 2


def test_get_injectable_facts_dedup_failure_returns_undismissed_facts():
    """If the embed API is down, dedup is skipped and the original list is
    returned — never break the prompt."""
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        {"text": "fact A", "source_run_id": "r1", "source_topic_id": 5, "created_at": 1},
        {"text": "fact B", "source_run_id": "r1", "source_topic_id": 5, "created_at": 2},
    ])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)

    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.92), \
         patch("src.bot.services.embedder.embed_chunks", return_value=[]):
        out = store.get_injectable_facts()
    # Embed returned nothing → dedup skipped → both facts returned.
    assert len(out) == 2


def test_get_injectable_facts_dedup_uses_persisted_embedding_skips_api():
    """Facts whose docs already carry an ``embedding`` vector are deduped
    WITHOUT calling embed_chunks — pure-local cosine. This is the path
    every inject hits after add() persisted the embedding at insert time.
    """
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        {"text": "old OOM rule",
         "embedding": [1.0, 0.0, 0.0],
         "source_run_id": "r1", "source_topic_id": 5, "created_at": 1},
        {"text": "newest OOM rule",
         "embedding": [1.0, 0.0, 0.0],
         "source_run_id": "r1", "source_topic_id": 5, "created_at": 3},
        {"text": "unrelated disk fact",
         "embedding": [0.0, 1.0, 0.0],
         "source_run_id": "r1", "source_topic_id": 5, "created_at": 2},
    ])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)

    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.92), \
         patch("src.bot.services.embedder.embed_chunks") as mock_embed:
        out = store.get_injectable_facts()

    mock_embed.assert_not_called()  # zero API calls — vectors came from Mongo.
    texts = [f["text"] for f in out]
    assert "newest OOM rule" in texts
    assert "old OOM rule" not in texts
    assert "unrelated disk fact" in texts
    assert len(out) == 2


@pytest.mark.asyncio
async def test_add_persists_embedding_in_doc():
    """`add()` embeds the fact text once and stores the vector in the
    fact doc so dedup at inject time skips the API call."""
    facts = MagicMock()
    facts.insert_one.return_value = MagicMock(inserted_id="oid")
    store = _store_with(facts=facts)

    with patch("src.bot.services.embedder.embed_text", return_value=[0.1, 0.2, 0.3]):
        added = await store.add(
            text="OOM pod needs more memory",
            source_chat_id=20,
            source_topic_id=10,
            source_run_id="r1",
        )
    assert added is True
    inserted = facts.insert_one.call_args[0][0]
    assert inserted["embedding"] == [0.1, 0.2, 0.3]
    assert inserted["text"] == "OOM pod needs more memory"


@pytest.mark.asyncio
async def test_add_continues_without_embedding_on_embed_failure():
    """If the embed API is down at insert time, the fact is still inserted
    with embedding=None — dedup will recompute on demand and the next
    backfill_embeddings() pass can fill it in."""
    facts = MagicMock()
    facts.insert_one.return_value = MagicMock(inserted_id="oid")
    store = _store_with(facts=facts)

    with patch("src.bot.services.embedder.embed_text", side_effect=RuntimeError("api down")):
        added = await store.add(text="x", source_chat_id=20, source_topic_id=10, source_run_id="r1")
    assert added is True
    inserted = facts.insert_one.call_args[0][0]
    assert inserted["embedding"] is None


def test_get_injectable_facts_serves_cached_snapshot_skips_dedup():
    """When background refresh has populated the cache, get_injectable_facts
    reads O(1) from memory — must NOT touch Mongo's `find` or call
    _dedup_facts_by_cosine again."""
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = iter([])  # would yield empty list
    store = _store_with(facts=facts, runs=runs)
    cached = [{"text": "cached fact alpha", "source_run_id": "r1"}]
    store._injectable_cache = list(cached)

    out = store.get_injectable_facts()

    assert out == cached
    # Mongo projections weren't touched this turn — the cache served fast path.
    runs.find.assert_not_called()
    facts.find.assert_not_called()


def test_record_run_pokes_refresh_event():
    """record_run must set the refresh event after a successful update_one
    so the background thread rebuilds the cache within seconds rather
    than waiting for the next tick."""
    runs = MagicMock()
    runs.update_one.return_value = MagicMock(modified_count=1)
    store = _store_with(runs=runs)
    assert not store._refresh_event.is_set()
    import asyncio
    asyncio.get_event_loop().run_until_complete(
        store.record_run(chat_id=20, topic_id=10, run_id="r1", last_distilled_count=12)
    )
    assert store._refresh_event.is_set()


def test_refresh_snapshot_rebuilds_cache():
    """_refresh_snapshot calls full pipeline (dedup + score + slice) and
    stores the result into _injectable_cache under the lock."""
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    fact_doc = {
        "text": "fact A",
        "embedding": [1.0, 0.0],
        "source_run_id": "r1",
        "created_at": 1,
    }
    cursor.sort.return_value = iter([fact_doc])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)

    # Patch _load_matched_kb_rules to return [] (no kb matches → all facts
    # get score 0, greedy slice includes them all under any positive budget).
    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.0), \
         patch.object(store, "_load_matched_kb_rules", return_value=[]), \
         patch.object(store, "_compute_token_budget", return_value=8000), \
         patch.object(store, "_estimate_tokens", return_value=100):
        store._refresh_snapshot()

    assert store._injectable_cache == [fact_doc]


def test_score_facts_against_kb_rules_zero_kb_returns_all_zero_scores():
    """Empty matched_kb list → all facts score 0, ordering stable (input order)."""
    store = _store_with()
    facts = [
        {"text": "f1", "embedding": [1.0, 0.0]},
        {"text": "f2", "embedding": [0.0, 1.0]},
    ]
    scored = store._score_facts_against_kb_rules(facts, [])
    assert len(scored) == 2
    assert all(s == 0.0 for _, s in scored)
    # Order preserved
    assert scored[0][0]["text"] == "f1"
    assert scored[1][0]["text"] == "f2"


def test_score_facts_against_kb_rules_cosine_max_picks_nearest():
    """Each fact takes its max cosine over all matched kb_rules."""
    store = _store_with()
    facts = [
        {"text": "f1 aligned with kb1", "embedding": [1.0, 0.0]},
        {"text": "f2 aligned with kb2", "embedding": [0.0, 1.0]},
        {"text": "f3 orthogonal", "embedding": [-1.0, -1.0]},
    ]
    matched_kb = [
        {"rule_id": "kb1", "match_count": 0, "issue_title": "",
         "embedding": __import__("numpy").array([1.0, 0.0], dtype="float32")},
        {"rule_id": "kb2", "match_count": 0, "issue_title": "",
         "embedding": __import__("numpy").array([0.0, 1.0], dtype="float32")},
    ]
    scored = store._score_facts_against_kb_rules(facts, matched_kb)
    scored_dict = {f["text"]: s for f, s in scored}
    # f1 cosine = 1.0 vs kb1 → 1.0 + cross_topic_bonus
    # f2 cosine = 1.0 vs kb2 → 1.0 + cross_topic_bonus
    # f3 cosine < 1.0 (orthogonal-ish but normalized → cos = -1)
    assert scored_dict["f1 aligned with kb1"] >= 0.99
    assert scored_dict["f2 aligned with kb2"] >= 0.99
    assert scored_dict["f3 orthogonal"] < scored_dict["f1 aligned with kb1"]


def test_score_facts_freq_bonus_applies_when_nearest_kb_match_high():
    """Facts whose nearest kb_rule has match_count >= 10 get +freq_bonus."""
    store = _store_with()
    import numpy as np
    facts = [{"text": "high-cos", "embedding": [1.0, 0.0]}]
    matched_kb = [
        {"rule_id": "kb-frequent", "match_count": 15,
         "issue_title": "", "embedding": np.array([1.0, 0.0], dtype="float32")},
    ]
    with patch.object(dfs_mod.bot_config, "distilled_inject_freq_bonus", 0.10), \
         patch.object(dfs_mod.bot_config, "distilled_inject_cross_topic_bonus", 0.0):
        scored = store._score_facts_against_kb_rules(facts, matched_kb)
    # cosine = 1.0 + freq_bonus = 0.10 → 1.10
    assert abs(scored[0][1] - 1.10) < 0.01


def test_score_facts_freq_bonus_skips_when_match_count_below_threshold():
    """Facts whose nearest kb_rule has match_count < threshold get no freq bonus."""
    store = _store_with()
    import numpy as np
    facts = [{"text": "high-cos", "embedding": [1.0, 0.0]}]
    matched_kb = [
        {"rule_id": "kb-rare", "match_count": 3,
         "issue_title": "", "embedding": np.array([1.0, 0.0], dtype="float32")},
    ]
    with patch.object(dfs_mod.bot_config, "distilled_inject_freq_bonus", 0.10), \
         patch.object(dfs_mod.bot_config, "distilled_inject_frequent_match_count", 10), \
         patch.object(dfs_mod.bot_config, "distilled_inject_cross_topic_bonus", 0.0):
        scored = store._score_facts_against_kb_rules(facts, matched_kb)
    assert abs(scored[0][1] - 1.0) < 0.01  # cosine only, no freq bonus


def test_ranked_slice_within_budget_enforces_token_cap():
    """Greedy fill stops when next fact would exceed the token budget."""
    store = _store_with()
    facts = [
        ({"text": "f1", "_scored_tokens": 100}, 1.0),
        ({"text": "f2", "_scored_tokens": 100}, 0.9),
        ({"text": "f3", "_scored_tokens": 100}, 0.8),
    ]
    out = store._ranked_slice_within_budget(facts, token_budget=250)
    # 100 + 100 + 100 > 250 → only first two fit
    assert len(out) == 2
    assert [f["text"] for f in out] == ["f1", "f2"]


def test_ranked_slice_within_budget_skips_oversized_fact():
    """A single fact larger than budget is skipped (not included)."""
    store = _store_with()
    facts = [
        ({"text": "huge", "_scored_tokens": 5000}, 1.0),
        ({"text": "small", "_scored_tokens": 100}, 0.5),
    ]
    out = store._ranked_slice_within_budget(facts, token_budget=1000)
    assert len(out) == 1
    assert out[0]["text"] == "small"


def test_compute_token_budget_at_watermarks():
    """Budget follows min → mid → max as matched_kb_count grows."""
    store = _store_with()
    with patch.object(dfs_mod.bot_config, "distilled_inject_min_tokens", 3000), \
         patch.object(dfs_mod.bot_config, "distilled_inject_max_tokens", 8000), \
         patch.object(dfs_mod.bot_config, "distilled_inject_high_watermark_kb_match", 30), \
         patch.object(dfs_mod.bot_config, "distilled_inject_max_watermark_kb_match", 60):
        assert store._compute_token_budget(0) == 3000
        assert store._compute_token_budget(60) == 8000
        assert store._compute_token_budget(100) == 8000  # caps at max
        # Below high watermark: 15/30 → halfway between min and mid (5500)
        assert 3000 < store._compute_token_budget(15) < (3000 + 8000) // 2
        # Above high: between mid and max
        mid_to_max = store._compute_token_budget(45)
        assert (3000 + 8000) // 2 < mid_to_max < 8000


def test_search_returns_empty_for_empty_query():
    store = _store_with()
    assert store.search("", k=10) == []
    assert store.search("   ", k=10) == []


def test_search_returns_empty_when_embedder_fails():
    store = _store_with()
    with patch("src.bot.services.embedder.embed_text",
               side_effect=RuntimeError("api down")):
        assert store.search("OOM", k=10) == []


def test_search_returns_topk_ordered_by_score_desc():
    """Cosine ranking returns facts sorted by similarity to query (desc)."""
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        {"text": "OOM fact", "embedding": [1.0, 0.0],
         "source_run_id": "r1", "source_topic_id": 5, "source_chat_id": 1},
        {"text": "disk fact", "embedding": [0.0, 1.0],
         "source_run_id": "r1", "source_topic_id": 5, "source_chat_id": 1},
    ])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)
    # Query aligned with OOM fact → should rank first.
    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.0), \
         patch("src.bot.services.embedder.embed_text",
               return_value=[1.0, 0.0]):
        out = store.search("OOM k8s", k=10)
    assert len(out) == 2
    assert out[0]["text"] == "OOM fact"
    assert out[1]["text"] == "disk fact"
    assert out[0]["score"] > out[1]["score"]
    assert out[0]["source_run_id"] == "r1"
    assert out[0]["source_chat_id"] == 1
    assert out[0]["source_topic_id"] == 5


def test_search_respects_k_limit():
    """Top-k limit returns at most k facts."""
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        {"text": f"fact-{i}", "embedding": [float(i), 0.0],
         "source_run_id": "r1", "source_topic_id": 5, "source_chat_id": 1}
        for i in range(5)
    ])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)
    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.0), \
         patch("src.bot.services.embedder.embed_text", return_value=[1.0, 0.0]):
        out = store.search("query", k=2)
    assert len(out) == 2


def _days_ago(days: int):
    import datetime as _dt
    return _dt.datetime.utcnow() - _dt.timedelta(days=days)


def test_time_decay_no_penalty_for_recent_fact():
    """Fact referenced 10 days ago → no time penalty."""
    store = _store_with()
    fact = {"text": "fresh", "created_at": _days_ago(10),
            "last_referenced_at": _days_ago(10)}
    assert store._time_decay_penalty(fact) == 0.0


def test_time_decay_90d_penalized():
    """Fact referenced 95 days ago → -0.05."""
    store = _store_with()
    fact = {"text": "stale", "created_at": _days_ago(200),
            "last_referenced_at": _days_ago(95)}
    assert abs(store._time_decay_penalty(fact) - 0.05) < 1e-6


def test_time_decay_180d_penalized_cumulative():
    """Fact 200 days old → -0.05 (90d) + -0.10 (180d) = -0.15."""
    store = _store_with()
    fact = {"text": "older", "created_at": _days_ago(200)}
    assert abs(store._time_decay_penalty(fact) - 0.15) < 1e-6


def test_time_decay_365d_penalized_cumulative():
    """Fact 400 days old → -0.05 + -0.10 + -0.15 = -0.30."""
    store = _store_with()
    fact = {"text": "ancient", "created_at": _days_ago(400)}
    assert abs(store._time_decay_penalty(fact) - 0.30) < 1e-6


def test_time_decay_last_referenced_overrides_old_created():
    """Last_referenced_at newer than created_at means fact is fresh even
    if originally created long ago. Anchor = max of the two."""
    store = _store_with()
    fact = {"text": "revived", "created_at": _days_ago(400),
            "last_referenced_at": _days_ago(10)}
    assert store._time_decay_penalty(fact) == 0.0


def test_time_decay_falls_back_to_created_when_no_last_referenced():
    """Facts inserted before v2.5.2 don't have last_referenced_at yet.
    Scoring must still work using created_at as anchor."""
    store = _store_with()
    fact = {"text": "old-school", "created_at": _days_ago(200)}
    assert abs(store._time_decay_penalty(fact) - 0.15) < 1e-6


def test_time_decay_no_anchor_returns_zero():
    """Fact with neither last_referenced_at nor created_at → no penalty."""
    store = _store_with()
    fact = {"text": "ghost"}
    assert store._time_decay_penalty(fact) == 0.0


def test_scoring_with_decay_reorders_recent_above_stale():
    """Two facts with identical cosine and kb_match — the recent one should
    rank above the stale one because of time-decay."""
    store = _store_with()
    import numpy as np
    facts = [
        {"text": "stale", "embedding": [1.0, 0.0],
         "created_at": _days_ago(200)},
        {"text": "recent", "embedding": [1.0, 0.0],
         "created_at": _days_ago(10)},
    ]
    matched_kb = [
        {"rule_id": "kb1", "match_count": 0, "issue_title": "",
         "embedding": np.array([1.0, 0.0], dtype="float32")},
    ]
    with patch.object(dfs_mod.bot_config, "distilled_inject_cross_topic_bonus", 0.0):
        scored = store._score_facts_against_kb_rules(facts, matched_kb)
    # Recent first (no penalty); stale last (penalty -0.15)
    assert scored[0][0]["text"] == "recent"
    assert scored[1][0]["text"] == "stale"
    assert scored[0][1] > scored[1][1]


def test_search_touches_last_referenced_on_returned_facts():
    """When search() returns facts, it must bump last_referenced_at on
    those fact docs so future scoring sees fresh activity."""
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        {"text": "fact-1", "embedding": [1.0, 0.0],
         "source_run_id": "r1", "source_topic_id": 5, "source_chat_id": 1},
        {"text": "fact-2", "embedding": [0.0, 1.0],
         "source_run_id": "r1", "source_topic_id": 5, "source_chat_id": 1},
    ])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)
    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.0), \
         patch("src.bot.services.embedder.embed_text", return_value=[1.0, 0.0]):
        out = store.search("query", k=2)
    # update_many called once with text matches
    facts.update_many.assert_called_once()
    args, _ = facts.update_many.call_args
    assert args[0] == {"text": {"$in": ["fact-1", "fact-2"]}}
    assert "last_referenced_at" in args[1]["$set"]


def test_add_seeds_last_referenced_at_equal_created_at():
    """Newly inserted fact should have last_referenced_at = created_at so
    scoring has an anchor even before search() ever returns the fact."""
    facts = MagicMock()
    facts.insert_one.return_value = MagicMock(inserted_id="oid")
    store = _store_with(facts=facts)
    import asyncio
    asyncio.get_event_loop().run_until_complete(
        store.add("x", source_chat_id=1, source_topic_id=2, source_run_id="r1")
    )
    inserted = facts.insert_one.call_args[0][0]
    assert "last_referenced_at" in inserted
    assert inserted["last_referenced_at"] == inserted["created_at"]

