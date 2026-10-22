"""Tests for the search_memory Holmes tool wrapper."""
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


def test_tool_metadata():
    """Tool exposes name, description, parameters in OpenAI format."""
    svc = MagicMock()
    tool = SearchMemoryTool(svc)
    fmt = tool.get_openai_format()
    assert fmt["function"]["name"] == "search_memory"
    props = fmt["function"]["parameters"]["properties"]
    assert "query" in props
    assert "k" in props
    # Description should mention calling after other tools (the rule)
    desc = fmt["function"]["description"]
    assert "AFTER" in desc or "after" in desc
    assert "tool" in desc.lower()


def test_invoke_with_facts_returns_them_as_data():
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=["fact one", "fact two"])
    tool = SearchMemoryTool(svc)
    result = tool.invoke({"query": "k8s crash", "k": 5}, context=_make_context())
    assert "fact one" in result.data
    assert "fact two" in result.data
    # Sanity: query reached retrieve_relevant
    call_kwargs = svc.retrieve_relevant.call_args
    assert call_kwargs.args[0] == "k8s crash"
    assert call_kwargs.kwargs.get("k") == 5


def test_invoke_no_facts_returns_no_data_message():
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=[])
    tool = SearchMemoryTool(svc)
    result = tool.invoke({"query": "nothing matches"}, context=_make_context())
    assert "No relevant" in result.data


def test_invoke_empty_query_returns_nothing_to_retrieve():
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock()
    tool = SearchMemoryTool(svc)
    result = tool.invoke({"query": "   "}, context=_make_context())
    assert "Empty query" in result.data
    svc.retrieve_relevant.assert_not_called()


def test_invoke_disabled_memory_returns_disabled_message():
    """Tool is still callable when memory is disabled — returns a friendly message."""
    svc = None  # memory service is None when bot disabled memory
    tool = SearchMemoryTool(svc)
    result = tool.invoke({"query": "anything"}, context=_make_context())
    assert "disabled" in result.data.lower() or "not" in result.data.lower()


def test_invoke_handles_retrieve_error_gracefully():
    """retrieve_relevant raising → tool returns empty/error, doesn't propagate."""
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(side_effect=RuntimeError("graphiti down"))
    tool = SearchMemoryTool(svc)
    # _run_async swallows exceptions and returns []
    result = tool.invoke({"query": "anything"}, context=_make_context())
    # Either "No relevant" (from []) or an error message — both are fine
    assert result.status.value == "success"


def test_default_k_when_not_provided():
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=["x"])
    tool = SearchMemoryTool(svc)
    tool.invoke({"query": "hello"}, context=_make_context())
    call_kwargs = svc.retrieve_relevant.call_args
    assert call_kwargs.kwargs.get("k") == 5


def test_default_k_when_invalid():
    svc = MagicMock()
    svc.retrieve_relevant = AsyncMock(return_value=["x"])
    tool = SearchMemoryTool(svc)
    # Non-integer k → falls back to 5
    tool.invoke({"query": "hello", "k": "many"}, context=_make_context())
    call_kwargs = svc.retrieve_relevant.call_args
    assert call_kwargs.kwargs.get("k") == 5
