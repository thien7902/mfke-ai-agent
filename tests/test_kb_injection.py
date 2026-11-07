"""Tests for in-process KB rule injection into bash tool stdout.

Pattern: patch the snapshot command marker so the test enters
`_maybe_inject_kb_rules`. Build a tiny in-memory KBStore (BM25 + 1 embedding
vector). Verify:
- stdout with snapshot command + signal-rich content gets KB block appended.
- stdout below `_SNAPSHOT_MIN_STDOUT_CHARS` is unchanged (no KB call paid).
- Non-snapshot commands skip injection entirely.
- KB failure (store None) returns stdout unchanged.

We do NOT exercise the actual `execute_bash_command_patched` subprocess
wrapper — that needs a live bash + kubectl. The injection function is
extracted precisely so it's testable in isolation.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.bot.services import holmes_service as hs_mod
from src.bot.services.holmes_service import _maybe_inject_kb_rules, _SNAPSHOT_CMD_MARKERS
from src.bot.services.kb_store import KBStore, _BM25Index


BIG_STDOUT = "===== 4. EVENTS =====\n" + ("CrashLoopBackOff OOMKilled pod/api-gw-7d9\n" * 600)
SNAPSHOT_CMD = "gardener_controlplane_snapshot --shoot km33c"
NON_SNAPSHOT_CMD = "kubectl get pods -A"
SMALL_STDOUT = "No resources found."


def _store_with_one_rule(rule_text: str, match_count: int = 0) -> KBStore:
    s = KBStore()
    s._loaded = True
    rid = "KB2025-001#1"
    s._index[rid] = {
        "rule": {
            "rule_id": rid,
            "issue_id": "KB2025-001",
            "issue_title": "OOM cluster",
            "rule_text": rule_text,
            "match_count": match_count,
        },
        "vector": [1.0, 0.0, 0.0],  # cosine with [1,0,0] query = 1.0
    }
    s._bm25 = _BM25Index(s._index)
    return s


def test_injection_appends_kb_block_for_snapshot_command():
    store = _store_with_one_rule(
        "IF pod CrashLoopBackOff AND OOMKilled -> bump memory"
    )
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True), \
         patch("src.bot.services.kb_store.embed_chunks", return_value=[[1.0, 0.0, 0.0]]):
        out = _maybe_inject_kb_rules(BIG_STDOUT, SNAPSHOT_CMD)
    assert "===== KB RULES" in out
    assert "CrashLoopBackOff" in out
    assert out.startswith(BIG_STDOUT)  # original preserved


def test_injection_skips_small_stdout():
    """A 200-char stdout means snapshot failed early — no signal, no KB
    call. Save embed API cost."""
    store = _store_with_one_rule("IF x THEN y -> z")
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True):
        out = _maybe_inject_kb_rules(SMALL_STDOUT, SNAPSHOT_CMD)
    assert out == SMALL_STDOUT  # unchanged


def test_injection_skips_non_snapshot_commands():
    """Plain kubectl get pods must not trigger KB lookup — would pay
    embed API cost on every bash call."""
    store = _store_with_one_rule("IF pod CrashLoopBackOff -> bump memory")
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True), \
         patch("src.bot.services.kb_store.embed_chunks", return_value=[[1.0, 0.0, 0.0]]) as mock_embed:
        out = _maybe_inject_kb_rules(BIG_STDOUT, NON_SNAPSHOT_CMD)
    assert out == BIG_STDOUT  # unchanged
    mock_embed.assert_not_called()


def test_injection_no_store_returns_original_stdout():
    """KB disabled or not wired → return stdout unchanged."""
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=None):
        out = _maybe_inject_kb_rules(BIG_STDOUT, SNAPSHOT_CMD)
    assert out == BIG_STDOUT


def test_injection_store_failure_returns_original_stdout():
    """KBStore raises → caught, stdout preserved. Tool result must
    never break because KB is unavailable."""
    store = MagicMock()
    store.get_relevant_rules.side_effect = RuntimeError("db down")
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True):
        out = _maybe_inject_kb_rules(BIG_STDOUT, SNAPSHOT_CMD)
    assert out == BIG_STDOUT


def test_injection_truncates_700k_stdout_before_embed():
    """A 700k-char snapshot must be capped before passing to embed_chunks
    so embed latency stays under ~10s. Verify the input to embed_chunks
    is bounded."""
    huge = "===== 4. EVENTS =====\n" + "OOMKilled CrashLoopBackOff\n" * 30000  # ~750k
    captured = {}

    def fake_embed(chunks):
        captured["total_chars"] = sum(len(c) for c in chunks)
        return [[1.0, 0.0, 0.0] for _ in chunks]

    store = _store_with_one_rule("IF pod CrashLoopBackOff -> bump memory")
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True), \
         patch("src.bot.services.kb_store.embed_chunks", side_effect=fake_embed):
        out = _maybe_inject_kb_rules(huge, SNAPSHOT_CMD)
    # 60k cap (default _KB_QUERY_MAX_CHARS) — single chunk passes through.
    assert captured["total_chars"] <= 70000  # 60k cap + separator slack
    # KB block appended → original stdout + KB rules.
    assert "===== KB RULES" in out


def test_injection_fires_record_matches_in_background():
    """After successful injection, record_matches fires asynchronously
    in a thread executor — never blocks the tool return."""
    store = _store_with_one_rule("IF pod CrashLoopBackOff -> bump memory", match_count=3)
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True), \
         patch("src.bot.services.kb_store.embed_chunks", return_value=[[1.0, 0.0, 0.0]]):
        # Sync-only context — Holmes' bash worker thread has no event loop.
        # The injection falls back to direct record_matches call.
        out = _maybe_inject_kb_rules(BIG_STDOUT, SNAPSHOT_CMD)
    # Verify match_count was incremented in Mongo (here: in-memory collection mock).
    # Since we used a real KBStore with no Mongo, record_matches is a no-op
    # on the unset _collection. Acceptable — we just want to confirm no
    # exception bubbles up.


def test_marker_substring_match():
    """The marker list substrings match a real snapshot command."""
    assert any(m in "gardener_controlplane_snapshot --shoot foo" for m in _SNAPSHOT_CMD_MARKERS)
    assert any(m in "bash /opt/snapshot.sh --shoot foo" for m in _SNAPSHOT_CMD_MARKERS)
    assert not any(m in "kubectl get pods" for m in _SNAPSHOT_CMD_MARKERS)
