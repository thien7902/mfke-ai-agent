"""Tests for the search_memory Holmes tool wrapper.

Multi-source: Graphiti + Distilled Facts + KB Rules, returned as a tagged list.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.tools.search_memory_tool import SearchMemoryTool


def _make_context():
    from holmes.core.tools import ToolInvokeContext
    ctx = MagicMock(spec=ToolInvokeContext)
    ctx.tool_number = None
    ctx.user_approved = False
    ctx.tool_call_id = "test-id"
    ctx.tool_name = "search_memory"
    return ctx


def _make_stores(memory=None, distilled=None, kb=None):
    """Construct a SearchMemoryTool with the three source stores."""
    return SearchMemoryTool(
        memory_service=memory,
        distilled_store=distilled,
        kb_store=kb,
    )


def test_tool_metadata():
    """Tool exposes name, description, parameters in OpenAI format."""
    svc = MagicMock()
    tool = _make_stores(memory=svc)
    fmt = tool.get_openai_format()
    assert fmt["function"]["name"] == "search_memory"
    props = fmt["function"]["parameters"]["properties"]
    assert "query" in props
    assert "k" in props
    assert "sources" in props
    desc = fmt["function"]["description"]
    assert "AFTER" in desc or "after" in desc
    assert "tool" in desc.lower()
    # Description should mention the multi-source nature.
    assert "graphiti" in desc.lower()
    assert "distilled" in desc.lower()
    assert "kb_rule" in desc.lower() or "kb rule" in desc.lower()


def test_invoke_with_graphiti_facts_returns_them_in_section():
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=["fact one", "fact two"])
    distilled = MagicMock()
    distilled.search = MagicMock(return_value=[])
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock(return_value=[])
    tool = _make_stores(memory=svc, distilled=distilled, kb=kb)
    result = tool.invoke({"query": "k8s crash", "k": 5}, context=_make_context())
    assert "fact one" in result.data
    assert "fact two" in result.data
    assert "Graphiti" in result.data
    # Sanity: query reached retrieve_relevant
    call_kwargs = svc.retrieve_relevant.call_args
    assert call_kwargs.args[0] == "k8s crash"
    assert call_kwargs.kwargs.get("k") == 5


def test_invoke_no_facts_any_source_returns_no_results_lines():
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=[])
    distilled = MagicMock()
    distilled.search = MagicMock(return_value=[])
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock(return_value=[])
    tool = _make_stores(memory=svc, distilled=distilled, kb=kb)
    result = tool.invoke({"query": "nothing matches"}, context=_make_context())
    # Sections still appear with "(no results)" rather than being omitted.
    assert "Graphiti" in result.data
    assert "Distilled Facts" in result.data
    assert "KB Rules" in result.data
    assert "(no results)" in result.data


def test_invoke_empty_query_returns_nothing_to_retrieve():
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock()
    distilled = MagicMock()
    distilled.search = MagicMock()
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock()
    tool = _make_stores(memory=svc, distilled=distilled, kb=kb)
    result = tool.invoke({"query": "   "}, context=_make_context())
    assert "Empty query" in result.data
    svc.retrieve_relevant.assert_not_called()
    distilled.search.assert_not_called()
    kb.get_relevant_rules.assert_not_called()


def test_invoke_disabled_memory_returns_no_results_for_graphiti_only():
    """Tool is still callable when memory is disabled — Graphiti section
    shows (no results), other sections still queried."""
    distilled = MagicMock()
    distilled.search = MagicMock(return_value=[{"text": "x", "score": 0.5}])
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock(return_value=[])
    tool = _make_stores(memory=None, distilled=distilled, kb=kb)
    result = tool.invoke({"query": "anything"}, context=_make_context())
    assert "Distilled Facts" in result.data
    assert "Graphiti" in result.data
    assert "KB Rules" in result.data


def test_invoke_handles_retrieve_error_gracefully():
    """Any source raising → tool still succeeds, that source shows (no results)."""
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(side_effect=RuntimeError("graphiti down"))
    distilled = MagicMock()
    distilled.search = MagicMock(return_value=[])
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock(return_value=[])
    tool = _make_stores(memory=svc, distilled=distilled, kb=kb)
    result = tool.invoke({"query": "anything"}, context=_make_context())
    assert result.status.value == "success"


def test_default_k_when_not_provided():
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=["x"])
    distilled = MagicMock()
    distilled.search = MagicMock(return_value=[])
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock(return_value=[])
    tool = _make_stores(memory=svc, distilled=distilled, kb=kb)
    tool.invoke({"query": "hello"}, context=_make_context())
    call_kwargs = svc.retrieve_relevant.call_args
    assert call_kwargs.kwargs.get("k") == 10
    distilled.search.assert_called_once_with("hello", k=10)
    kb.get_relevant_rules.assert_called_once_with("hello", top_k=10)


def test_default_k_when_invalid():
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=["x"])
    distilled = MagicMock()
    distilled.search = MagicMock(return_value=[])
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock(return_value=[])
    tool = _make_stores(memory=svc, distilled=distilled, kb=kb)
    # Non-integer k → falls back to 10
    tool.invoke({"query": "hello", "k": "many"}, context=_make_context())
    call_kwargs = svc.retrieve_relevant.call_args
    assert call_kwargs.kwargs.get("k") == 10


def test_sources_param_restricts_to_kb_rule_only():
    """`sources="kb_rule"` skips Graphiti and Distilled queries."""
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=["g"])
    distilled = MagicMock()
    distilled.search = MagicMock(return_value=[{"text": "d"}])
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock(return_value=[{
        "rule_id": "KB001#1",
        "rule_text": "kb-rule-text",
        "match_count": 14,
        "issue_title": "issue title",
        "issue_id": "ISSUE-1",
    }])
    tool = _make_stores(memory=svc, distilled=distilled, kb=kb)
    result = tool.invoke(
        {"query": "k8s", "sources": "kb_rule"},
        context=_make_context(),
    )
    assert "KB Rules" in result.data
    assert "rule_id=KB001#1" in result.data
    assert "match_count=14" in result.data
    assert "kb-rule-text" in result.data
    # Other sources should NOT have sections present — only their queries are skipped.
    svc.retrieve_relevant.assert_not_called()
    distilled.search.assert_not_called()
    kb.get_relevant_rules.assert_called_once()


def test_sources_param_restricts_to_distilled_only():
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=["g"])
    distilled = MagicMock()
    distilled.search = MagicMock(return_value=[{
        "text": "chat-derived fact",
        "score": 0.42,
        "source_run_id": "r-abc",
    }])
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock(return_value=[{"rule_id": "X"}])
    tool = _make_stores(memory=svc, distilled=distilled, kb=kb)
    result = tool.invoke(
        {"query": "k8s", "sources": "distilled"},
        context=_make_context(),
    )
    assert "Distilled Facts" in result.data
    assert "chat-derived fact" in result.data
    # run_id should appear in the tag prefix.
    assert "run_id=r-abc" in result.data
    assert "score=0.420" in result.data
    svc.retrieve_relevant.assert_not_called()
    kb.get_relevant_rules.assert_not_called()


def test_invalid_sources_falls_back_to_all():
    """`sources="invalid,foo"` (no valid tokens) falls back to all 3 sources."""
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=["g"])
    distilled = MagicMock()
    distilled.search = MagicMock(return_value=[])
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock(return_value=[])
    tool = _make_stores(memory=svc, distilled=distilled, kb=kb)
    result = tool.invoke(
        {"query": "k8s", "sources": "invalid,foo"},
        context=_make_context(),
    )
    svc.retrieve_relevant.assert_called_once()
    distilled.search.assert_called_once()
    kb.get_relevant_rules.assert_called_once()
    assert "Graphiti" in result.data
    assert "Distilled Facts" in result.data
    assert "KB Rules" in result.data


def test_multi_source_output_has_section_headers_and_total_footer():
    """End-to-end: all 3 sources return a fact → tagged-list output with footer."""
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=["graphiti-fact"])
    distilled = MagicMock()
    distilled.search = MagicMock(return_value=[{
        "text": "distilled-fact",
        "score": 0.77,
        "source_run_id": "r-1",
    }])
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock(return_value=[{
        "rule_id": "KB20260309-004#1",
        "rule_text": "kb-rule",
        "match_count": 14,
        "issue_title": "OOM runbook",
        "issue_id": "ISSUE-1",
    }])
    tool = _make_stores(memory=svc, distilled=distilled, kb=kb)
    result = tool.invoke({"query": "OOM"}, context=_make_context())
    assert "=== SEARCH_MEMORY RESULTS" in result.data
    assert "--- Graphiti" in result.data
    assert "--- Distilled Facts" in result.data
    assert "--- KB Rules" in result.data
    # Tagged prefixes
    assert "- graphiti-fact" in result.data
    assert "[run_id=r-1" in result.data
    assert "[rule_id=KB20260309-004#1, match_count=14" in result.data
    # Footer with N total
    assert "(N=3" in result.data


def test_kb_rule_text_truncated_when_long():
    """Rule text over 600 chars gets truncated in the prefixed output."""
    long_text = "x" * 700
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=[])
    distilled = MagicMock()
    distilled.search = MagicMock(return_value=[])
    kb = MagicMock()
    kb.get_relevant_rules = MagicMock(return_value=[{
        "rule_id": "KB-LONG#1",
        "rule_text": long_text,
        "match_count": 1,
        "issue_title": "",
        "issue_id": "I",
    }])
    tool = _make_stores(memory=svc, distilled=distilled, kb=kb)
    result = tool.invoke({"query": "q"}, context=_make_context())
    assert "..." in result.data
    assert long_text not in result.data  # full 700 chars not present
