"""Tests for in-process KB rule injection into Holmes tool results.

Pattern: build a tiny in-memory KBStore (BM25 + 1 embedding vector) and
call ``_maybe_inject_kb_rules_into_tool_result`` with a fake
``StructuredToolResult`` (a plain object exposing ``data`` + ``invocation``
attributes — duck-typed). Verify:
- tool_name in marker list (custom Holmes tool path) AND signal-rich data →
  KB block appended to data.
- bash-toolset path: tool_name="Bash", invocation carries snapshot.sh →
  KB block appended.
- tool_name NOT in markers + invocation NOT in markers → no injection.
- stdout below ``_SNAPSHOT_MIN_STDOUT_CHARS`` is unchanged (no KB call paid).
- Failure modes (store None, store raises) leave tool_result untouched.

We do NOT exercise the actual ``execute_bash_command_patched`` subprocess
wrapper nor the real ``format_tool_result_data`` Holmes serializer — the
injection function is extracted precisely so it's testable in isolation.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.bot.services import holmes_service as hs_mod
from src.bot.services.holmes_service import (
    _maybe_inject_kb_rules_into_tool_result,
    _SNAPSHOT_TOOL_MARKERS,
)
from src.bot.services.kb_store import KBStore, _BM25Index


BIG_STDOUT = "===== 4. EVENTS =====\n" + ("CrashLoopBackOff OOMKilled pod/api-gw-7d9\n" * 600)
SNAPSHOT_TOOL_NAME = "gardener_controlplane_snapshot"
BASH_SNAPSHOT_INVOCATION = "bash /opt/snapshot.sh --shoot foo"
NON_SNAPSHOT_TOOL_NAME = "kubectl_get_pods"
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


def _tool_result(data, invocation=None):
    """Duck-typed Stub for holmes.core.tools.StructuredToolResult — only
    the fields the injection function reads/sets need to exist."""
    return SimpleNamespace(data=data, invocation=invocation)


def test_injection_appends_kb_block_for_native_snapshot_tool():
    """Custom Holmes tool name matches a marker → block appended to
    tool_result.data (str case)."""
    store = _store_with_one_rule(
        "IF pod CrashLoopBackOff AND OOMKilled -> bump memory"
    )
    tr = _tool_result(BIG_STDOUT, invocation=None)
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True), \
         patch("src.bot.services.kb_store.embed_chunks", return_value=[[1.0, 0.0, 0.0]]):
        _maybe_inject_kb_rules_into_tool_result(tr, SNAPSHOT_TOOL_NAME)
    assert "===== KB RULES" in tr.data
    assert "CrashLoopBackOff" in tr.data
    assert tr.data.startswith(BIG_STDOUT)  # original preserved


def test_injection_appends_kb_block_for_bash_wrapped_snapshot():
    """Bash toolset path: tool_name is generic, but invocation carries
    snapshot.sh — should still match and inject."""
    store = _store_with_one_rule(
        "IF pod CrashLoopBackOff AND OOMKilled -> bump memory"
    )
    tr = _tool_result(BIG_STDOUT, invocation=BASH_SNAPSHOT_INVOCATION)
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True), \
         patch("src.bot.services.kb_store.embed_chunks", return_value=[[1.0, 0.0, 0.0]]):
        _maybe_inject_kb_rules_into_tool_result(tr, "Bash")
    assert "===== KB RULES" in tr.data


def test_injection_skips_small_stdout():
    """A 30-char stdout means snapshot failed early — no signal, no KB call."""
    store = _store_with_one_rule("IF x THEN y -> z")
    tr = _tool_result(SMALL_STDOUT, invocation=BASH_SNAPSHOT_INVOCATION)
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True):
        _maybe_inject_kb_rules_into_tool_result(tr, SNAPSHOT_TOOL_NAME)
    assert tr.data == SMALL_STDOUT  # unchanged


def test_injection_skips_non_snapshot_tools():
    """Plain kubectl_get_pods must not trigger KB lookup — would pay embed
    API cost on every tool call."""
    store = _store_with_one_rule("IF pod CrashLoopBackOff -> bump memory")
    tr = _tool_result(BIG_STDOUT, invocation=None)
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True), \
         patch("src.bot.services.kb_store.embed_chunks", return_value=[[1.0, 0.0, 0.0]]) as mock_embed:
        _maybe_inject_kb_rules_into_tool_result(tr, NON_SNAPSHOT_TOOL_NAME)
    assert tr.data == BIG_STDOUT  # unchanged
    mock_embed.assert_not_called()


def test_injection_no_store_returns_original_data():
    """KB disabled or not wired → tool_result untouched."""
    tr = _tool_result(BIG_STDOUT, invocation=BASH_SNAPSHOT_INVOCATION)
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=None):
        _maybe_inject_kb_rules_into_tool_result(tr, SNAPSHOT_TOOL_NAME)
    assert tr.data == BIG_STDOUT


def test_injection_store_failure_leaves_tool_result_untouched():
    """KBStore raises → caught, tool_result.data preserved. Tool result must
    never break because KB is unavailable."""
    store = MagicMock()
    store.get_relevant_rules.side_effect = RuntimeError("db down")
    tr = _tool_result(BIG_STDOUT, invocation=BASH_SNAPSHOT_INVOCATION)
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True):
        _maybe_inject_kb_rules_into_tool_result(tr, SNAPSHOT_TOOL_NAME)
    assert tr.data == BIG_STDOUT


def test_injection_truncates_huge_stdout_before_embed():
    """A 700k-char snapshot must be capped before passing to embed_chunks
    so embed latency stays under ~10s. Verify the input to embed_chunks
    is bounded."""
    huge = "===== 4. EVENTS =====\n" + "OOMKilled CrashLoopBackOff\n" * 30000  # ~750k
    captured = {}

    def fake_embed(chunks):
        captured["total_chars"] = sum(len(c) for c in chunks)
        return [[1.0, 0.0, 0.0] for _ in chunks]

    store = _store_with_one_rule("IF pod CrashLoopBackOff -> bump memory")
    tr = _tool_result(huge, invocation=BASH_SNAPSHOT_INVOCATION)
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True), \
         patch("src.bot.services.kb_store.embed_chunks", side_effect=fake_embed):
        _maybe_inject_kb_rules_into_tool_result(tr, SNAPSHOT_TOOL_NAME)
    # 60k cap (default _KB_QUERY_MAX_CHARS) — single chunk passes through.
    assert captured["total_chars"] <= 70000  # 60k cap + separator slack
    # KB block appended to original stdout.
    assert "===== KB RULES" in tr.data


def test_injection_handles_dict_data_with_output_key():
    """Custom Holmes tools often return data as a dict like {'output': ...}.
    The inject block should be appended to the nested field, not the dict."""
    store = _store_with_one_rule(
        "IF pod CrashLoopBackOff AND OOMKilled -> bump memory"
    )
    tr = _tool_result({"output": BIG_STDOUT}, invocation=None)
    with patch.object(hs_mod, "_get_kb_store_singleton", return_value=store), \
         patch.object(hs_mod.bot_config, "kb_enabled", True), \
         patch("src.bot.services.kb_store.embed_chunks", return_value=[[1.0, 0.0, 0.0]]):
        _maybe_inject_kb_rules_into_tool_result(tr, SNAPSHOT_TOOL_NAME)
    assert isinstance(tr.data, dict)
    assert "===== KB RULES" in tr.data["output"]
    # Other keys untouched
    assert "output" in tr.data


def test_marker_substring_match():
    """The marker list substrings match real tool names / commands."""
    assert any(m in "gardener_controlplane_snapshot" for m in _SNAPSHOT_TOOL_MARKERS)
    assert any(m in "bash /opt/snapshot.sh --shoot foo" for m in _SNAPSHOT_TOOL_MARKERS)
    assert not any(m in "kubectl get pods" for m in _SNAPSHOT_TOOL_MARKERS)


def test_patch_format_tool_result_data_calls_injection_then_original():
    """The chokepoint patch wraps the original format_tool_result_data —
    every call should invoke the KB injector first, then hand off to the
    original Holmes serializer. We exercise it by patching the inject
    helper and confirming it was called with the right args."""
    from holmes.core.tools import StructuredToolResult, StructuredToolResultStatus
    import holmes.core.models as _m

    # Reset patch state so we can re-apply it (idempotent guard).
    if getattr(_m, "_mfke_kb_patched", False):
        _m.format_tool_result_data = _m.format_tool_result_data._mfke_orig
        _m._mfke_kb_patched = False

    tr = StructuredToolResult(
        status=StructuredToolResultStatus.SUCCESS,
        data=BIG_STDOUT,
        invocation=None,
    )
    captured = {"called_with": None}

    def fake_inject(tr_arg, name_arg):
        captured["called_with"] = (tr_arg, name_arg)

    with patch.object(hs_mod, "_maybe_inject_kb_rules_into_tool_result", side_effect=fake_inject):
        hs_mod._patch_format_tool_result_data()
        out = _m.format_tool_result_data(tr, "tool_call_id_1", "gardener_controlplane_snapshot", None)

    assert captured["called_with"] is not None
    assert captured["called_with"][1] == "gardener_controlplane_snapshot"
    assert captured["called_with"][0] is tr
    # The original serializer must still run — output is non-empty and
    # carries the Holmes tool_call_metadata header.
    assert "tool_call_metadata" in out
    assert BIG_STDOUT in out

    # Restore original to keep module state clean for other tests.
    if getattr(_m, "_mfke_kb_patched", False):
        _m.format_tool_result_data = _m.format_tool_result_data._mfke_orig
        _m._mfke_kb_patched = False
