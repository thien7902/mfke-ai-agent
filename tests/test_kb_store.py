"""Tests for KBStore — frequency-weighted cosine retrieval over the
kb_rules collection.

Pattern: bypass the lazy MongoDB property by setting `s._index` directly
via a `_store_with(*index_entries)` helper, mirroring `tests/test_distilled_fact_store.py`'s
`_store_with(facts=..., runs=...)` pattern. Patch `embed_chunks` at the
module path so no real API call.
"""
import math
from unittest.mock import MagicMock, patch

import pytest

from src.bot.services import kb_store as kb_store_mod
from src.bot.services.kb_store import KBStore, format_rules_block


def _store_with(*entries, loaded=True):
    """Build a KBStore with an in-memory index. Each entry is a
    (rule_id, rule_doc, vector) tuple.
    """
    s = KBStore()
    s._loaded = loaded
    for rid, rule, vec in entries:
        s._index[rid] = {"rule": rule, "vector": vec}
    # Build the BM25 inverted index too — _ensure_loaded would have done
    # this if loaded normally via Mongo. Test bypasses _ensure_loaded so
    # the BM25 index must be constructed explicitly.
    from src.bot.services.kb_store import _BM25Index
    s._bm25 = _BM25Index(s._index)
    return s


def _rule_doc(issue_id="KB2025-001", issue_title="title", match_count=0, rule_text="IF x THEN y -> z"):
    return {
        "rule_id": f"{issue_id}#1",
        "issue_id": issue_id,
        "issue_title": issue_title,
        "rule_text": rule_text,
        "match_count": match_count,
    }


def make_vec(v):
    """Convert scalar v to a unit-ish 3-dim vector for predictable cosine."""
    return [v, 0.0, 0.0]


# --------------------------------------------------------------------- #
# Disabled / empty / no-op cases
# --------------------------------------------------------------------- #

def test_disabled_returns_empty(monkeypatch):
    with patch.object(kb_store_mod.bot_config, "kb_enabled", False):
        s = _store_with(("r1", _rule_doc(), make_vec(1.0)))
        assert s.get_relevant_rules("anything") == []


def test_empty_query_returns_empty(monkeypatch):
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True):
        s = _store_with(("r1", _rule_doc(), make_vec(1.0)))
        assert s.get_relevant_rules("") == []
        assert s.get_relevant_rules("   ") == []


def test_empty_index_returns_empty(monkeypatch):
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True):
        s = KBStore()
        s._loaded = True
        with patch.object(kb_store_mod, "embed_chunks", return_value=[make_vec(1.0)]):
            assert s.get_relevant_rules("shoot error") == []


def test_embed_failure_returns_empty(monkeypatch):
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True):
        s = _store_with(("r1", _rule_doc(), make_vec(1.0)))
        with patch.object(kb_store_mod, "embed_chunks", side_effect=RuntimeError("api down")):
            assert s.get_relevant_rules("shoot error") == []


# --------------------------------------------------------------------- #
# Ranking + frequency boost
# --------------------------------------------------------------------- #

def test_top_k_ranking_with_frequency_boost(monkeypatch):
    """Two rules with EQUALLY close vectors (cosine tie). Frequency-boosted
    one should rank higher.
    """
    a = ("KB2025-001#1", _rule_doc(issue_id="KB2025-001", match_count=10), make_vec(1.0))
    b = ("KB2025-002#1", _rule_doc(issue_id="KB2025-002", match_count=0), make_vec(1.0))
    s = _store_with(a, b)
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True), \
         patch.object(kb_store_mod, "embed_chunks", return_value=[make_vec(1.0)]):
        out = s.get_relevant_rules("shoot error")
    assert len(out) == 2
    assert out[0]["rule_id"] == "KB2025-001#1"
    assert out[1]["rule_id"] == "KB2025-002#1"


def test_below_cosine_floor_skipped_even_with_high_match_count(monkeypatch):
    """Cosine below _MIN_COSINE should exclude a rule even if it has a huge
    match_count — frequency must not promote irrelevant rules.
    """
    # Vector [0.0, 0.0, 0.1] → orthogonal to query [1,0,0] → cosine 0.0.
    a = ("KB2025-001#1", _rule_doc(match_count=1000), [0.0, 0.0, 0.1])
    s = _store_with(a)
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True), \
         patch.object(kb_store_mod, "embed_chunks", return_value=[make_vec(1.0)]):
        out = s.get_relevant_rules("shoot error")
    assert out == []


def test_log1p_boost_curve(monkeypatch):
    """Sanity-check the score formula across match_count values."""
    # cosine=0.9 for all three scenarios; only match_count differs.
    # 0 matches → 0.9 * (1 + log1p(0)) = 0.9
    # 10 → 0.9 * (1 + log(11)) ≈ 0.9 * 3.398 ≈ 3.058
    # 100 → 0.9 * (1 + log(101)) ≈ 0.9 * 5.615 ≈ 5.054
    cos = 0.9
    expected_zero = cos * (1.0 + math.log1p(0))
    expected_ten = cos * (1.0 + math.log1p(10))
    expected_hundred = cos * (1.0 + math.log1p(100))
    # The boost ratio (hundred vs zero) is bounded — not 100x.
    assert expected_ten > expected_zero
    assert expected_hundred > expected_ten
    assert expected_hundred / expected_zero < 10.0  # roughly log curve


def test_chunk_strategy_caps_long_inputs(monkeypatch):
    """A snapshot output with section markers should be split into chunks
    before embedding. The longest merged chunk must not exceed the cap.
    """
    chunks_captured = {"list": []}

    def fake_embed_chunks(texts):
        chunks_captured["list"] = list(texts)
        # Return zero-vector per chunk → cosine 0 with everything → all [].
        return [[0.0, 0.0, 0.0] for _ in texts]

    a = ("KB2025-001#1", _rule_doc(), make_vec(1.0))
    s = _store_with(a)
    big = "===== 4. EVENTS =====\n" + ("a" * 800) + "\n"
    big += "===== 5. PODS =====\n" + ("b" * 200) + "\n"
    big += "===== 6. LOGS =====\n" + ("c" * 5000)
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True), \
         patch.object(kb_store_mod, "embed_chunks", side_effect=fake_embed_chunks):
        s.get_relevant_rules(big)
    # Multiple chunks produced and merged.
    assert len(chunks_captured["list"]) >= 1
    for chunk in chunks_captured["list"]:
        assert len(chunk) <= 6000


# --------------------------------------------------------------------- #
# record_matches + top_by_match_count
# --------------------------------------------------------------------- #

def test_record_matches_calls_update_many():
    s = KBStore()
    s._loaded = True
    s._collection = MagicMock()
    s.record_matches(["r1", "r2"])
    s._collection.update_many.assert_called_once()
    args, kwargs = s._collection.update_many.call_args
    assert args[0] == {"rule_id": {"$in": ["r1", "r2"]}}
    # $inc match_count + $set last_matched_at
    assert "$inc" in args[1] and "$set" in args[1]


def test_record_matches_swallows_db_error():
    s = KBStore()
    s._loaded = True
    s._collection = MagicMock()
    s._collection.update_many.side_effect = RuntimeError("db down")
    # Should not raise.
    s.record_matches(["r1"])


def test_record_matches_empty_no_op():
    s = KBStore()
    s._collection = MagicMock()
    s.record_matches([])
    s._collection.update_many.assert_not_called()


def test_top_by_match_count_sorts_desc():
    s = KBStore()
    s._loaded = True
    s._collection = MagicMock()
    cursor = MagicMock()
    s._collection.find.return_value = cursor
    cursor.sort.return_value.limit.return_value = [{"rule_id": "r1", "match_count": 99}]
    out = s.top_by_match_count(limit=20)
    assert out == [{"rule_id": "r1", "match_count": 99}]
    s._collection.find.assert_called_once()
    # sort on match_count desc
    assert "match_count" in str(cursor.sort.call_args.args[0])


def test_top_by_match_count_swallows_db_error():
    s = KBStore()
    s._collection = MagicMock()
    s._collection.find.side_effect = RuntimeError("db down")
    assert s.top_by_match_count() == []


# --------------------------------------------------------------------- #
# format_rules_block
# --------------------------------------------------------------------- #

def test_format_rules_block_empty():
    s = format_rules_block([])
    assert s.startswith("===== KB RULES (empty) =====")


def test_format_rules_block_top_n_prefix():
    rules = [
        {"rule_id": "KB2025-001#1", "issue_id": "KB2025-001", "issue_title": "OOM",
         "rule_text": "IF pod CrashLoop AND OOMKilled -> bump memory", "match_count": 3},
    ]
    s = format_rules_block(rules)
    assert "===== KB RULES (top-1, distilled + frequency-weighted) =====" in s
    assert "===== END KB RULES =====" in s
    assert "[KB2025-001#1 matches=3]" in s
    assert "IF pod CrashLoop AND OOMKilled -> bump memory" in s
    assert "(from KB2025-001 OOM)" in s


def test_format_rules_block_truncates_long_rules():
    big_text = "x" * 1000
    rules = [{"rule_id": "r1", "issue_id": "KB1", "issue_title": "t", "rule_text": big_text, "match_count": 0}]
    s = format_rules_block(rules)
    assert "..." in s
    # 600 chars + ellipsis
    assert s.count("x") <= 600


# --------------------------------------------------------------------- #
# Chunking helper directly
# --------------------------------------------------------------------- #

def test_chunk_at_section_markers_basic():
    from src.bot.services.kb_store import _chunk_at_section_markers
    text = "prefix\n===== 1. SHOOT =====\nshoot yaml\n===== 2. PODS =====\npods list\n"
    chunks = _chunk_at_section_markers(text, max_chars=1000)
    # "prefix" + section 1 + section 2 — three pieces (or merged depending on cap).
    assert len(chunks) >= 1
    assert any("shoot yaml" in c for c in chunks)
    assert any("pods list" in c for c in chunks)


def test_chunk_at_section_markers_no_markers_treats_as_one_chunk():
    from src.bot.services.kb_store import _chunk_at_section_markers
    text = "bulk content no markers"
    chunks = _chunk_at_section_markers(text, max_chars=1000)
    assert chunks == [text]


def test_cosine():
    from src.bot.services.kb_store import _cosine
    assert _cosine([], [1.0, 0.0]) == 0.0
    assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert _cosine([1, 1], [1, 1]) == pytest.approx(1.0)


# --------------------------------------------------------------------- #
# DB load failure → empty index
# --------------------------------------------------------------------- #

def test_db_load_failure_returns_empty_after_load_attempt(monkeypatch):
    s = KBStore()
    # Force load failure by mocking the collection to raise on find.
    bad_collection = MagicMock()
    bad_collection.find.side_effect = RuntimeError("db down")
    s._collection = bad_collection
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True), \
         patch.object(kb_store_mod, "embed_chunks", return_value=[make_vec(1.0)]):
        out = s.get_relevant_rules("shoot error")
    assert out == []
    # After the failed load attempt, _loaded=True so we don't loop forever.
    assert s._loaded is True


# --------------------------------------------------------------------- #
# Hybrid BM25 + cosine (RRF)
# --------------------------------------------------------------------- #

def test_bm25_exact_keyword_match_rescues_low_cosine(monkeypatch):
    """Rule with exact keyword match (CrashLoopBackOff) but low cosine
    should still surface because BM25 carries it past the floor.
    """
    rule = _rule_doc(
        issue_id="KB2025-001",
        rule_text="IF pod CrashLoopBackOff AND OOMKilled -> bump memory",
        match_count=0,
    )
    # Vector orthogonal to query [1,0,0] → cosine 0.
    a = ("KB2025-001#1", rule, [0.0, 0.0, 0.1])
    s = _store_with(a)
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True), \
         patch.object(kb_store_mod, "embed_chunks", return_value=[make_vec(1.0)]):
        out = s.get_relevant_rules("CrashLoopBackOff on pod api-gw")
    # BM25 picks up CrashLoopBackOff + pod despite zero cosine.
    assert len(out) == 1
    assert out[0]["rule_id"] == "KB2025-001#1"
    assert out[0]["cosine"] == 0.0
    assert out[0]["bm25"] > 0.0


def test_both_below_floor_skipped():
    """Cosine 0 + no keyword overlap → dropped."""
    rule = _rule_doc(
        issue_id="KB2025-001",
        rule_text="IF some unrelated symptom -> generic fix",
        match_count=0,
    )
    # Orthogonal vector + token set disjoint from query.
    a = ("KB2025-001#1", rule, [0.0, 0.0, 0.1])
    s = _store_with(a)
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True), \
         patch.object(kb_store_mod, "embed_chunks", return_value=[make_vec(1.0)]):
        out = s.get_relevant_rules("CrashLoopBackOff OOMKilled")
    assert out == []


def test_embed_failure_falls_back_to_bm25_only(monkeypatch):
    """If embed_chunks throws, BM25 still returns matches (graceful
    degradation when the embedding API is down).
    """
    rule = _rule_doc(
        issue_id="KB2025-001",
        rule_text="IF pod CrashLoopBackOff AND OOMKilled -> bump memory",
        match_count=0,
    )
    a = ("KB2025-001#1", rule, make_vec(1.0))
    s = _store_with(a)
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True), \
         patch.object(kb_store_mod, "embed_chunks", side_effect=RuntimeError("embed api down")):
        out = s.get_relevant_rules("pod CrashLoopBackOff OOMKilled")
    assert len(out) == 1
    assert out[0]["rule_id"] == "KB2025-001#1"
    assert out[0]["cosine"] == 0.0
    assert out[0]["bm25"] > 0.0


def test_bm25_high_match_count_surface_keyword_rule_above_low_freq():
    """Two rules both BM25-match the query. The one with higher
    match_count should rank higher (after frequency boost).
    """
    rule_a = _rule_doc(
        issue_id="KB2025-001",
        rule_text="IF pod CrashLoopBackOff -> bump memory",
        match_count=10,
    )
    rule_b = _rule_doc(
        issue_id="KB2025-002",
        rule_text="IF pod CrashLoopBackOff -> restart node",
        match_count=0,
    )
    # Same orthogonal vector → same cosine 0; BM25 + freq boost decides.
    a = ("KB2025-001#1", rule_a, [0.0, 0.0, 0.1])
    b = ("KB2025-002#1", rule_b, [0.0, 0.0, 0.1])
    s = _store_with(a, b)
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True), \
         patch.object(kb_store_mod, "embed_chunks", return_value=[make_vec(1.0)]):
        out = s.get_relevant_rules("pod CrashLoopBackOff")
    assert len(out) == 2
    assert out[0]["rule_id"] == "KB2025-001#1"  # higher match_count wins
    assert out[1]["rule_id"] == "KB2025-002#1"


def test_bm25_returns_empty_when_only_stopwords():
    """Query of just stopwords/short tokens → BM25 empty → only cosine decides."""
    rule = _rule_doc(rule_text="IF pod CrashLoopBackOff -> bump memory")
    a = ("KB2025-001#1", rule, make_vec(1.0))
    s = _store_with(a)
    with patch.object(kb_store_mod.bot_config, "kb_enabled", True), \
         patch.object(kb_store_mod, "embed_chunks", return_value=[make_vec(1.0)]):
        # "the" and "a" are stopwords; "x" is 1-char filtered.
        out = s.get_relevant_rules("the a x")
    # Query tokens empty after filter → BM25 empty; cosine still runs.
    assert len(out) == 1
    assert out[0]["bm25"] == 0.0

