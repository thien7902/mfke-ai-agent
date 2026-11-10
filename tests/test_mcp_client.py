"""Tests for src.bot.utils.mcp_client.

Pattern: mock the `mcp` SDK's `streamablehttp_client` + `ClientSession` so no
real HTTP call is made. Verify our wrappers correctly parse the JSON issue
list returned by mcp-atlassian, and gracefully degrade when the SDK is missing
or the call fails.
"""
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _content_with_text(text: str):
    """Build the CallToolResult-shaped object the mcp SDK returns from
    call_tool — a MagicMock with a `.content` list of objects each having
    .type and .text attributes."""
    item = MagicMock()
    item.type = "text"
    item.text = text
    result = MagicMock()
    result.content = [item]
    return result


def _install_fake_mcp_sdk(monkeypatch):
    """Inject fake `mcp.client.streamablehttp` and `mcp.client.session` modules
    into sys.modules so the lazy import in `_call` finds them."""

    class _Ctx:
        def __init__(self, read, write, _):
            self.read = read
            self.write = write
            self._ = _

        async def __aenter__(self):
            return (self.read, self.write, self._)

        async def __aexit__(self, *exc):
            return False

    def fake_streamablehttp_client(url, timeout=None):
        return _Ctx(MagicMock(), MagicMock(), MagicMock())

    fake_session = MagicMock()
    fake_session.ClientSession = MagicMock()
    fake_streamablehttp = types.ModuleType("mcp.client.streamablehttp")
    fake_streamablehttp.streamablehttp_client = fake_streamablehttp_client
    fake_session_mod = types.ModuleType("mcp.client.session")
    fake_session_mod.ClientSession = MagicMock()
    fake_mcp = types.ModuleType("mcp")
    fake_client = types.ModuleType("mcp.client")
    fake_client_mod = types.ModuleType("mcp.client")
    fake_client_mod.streamablehttp = fake_streamablehttp
    fake_client_mod.session = fake_session_mod

    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", fake_client_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamablehttp", fake_streamablehttp)
    monkeypatch.setitem(sys.modules, "mcp.client.session", fake_session_mod)
    return fake_streamablehttp, fake_session_mod


def test_search_issues_parses_jira_response(monkeypatch):
    """search_issues receives a JSON {"issues": [...]} string in the
    CallToolResult, parses and returns the list."""
    fake_streamablehttp, fake_session_mod = _install_fake_mcp_sdk(monkeypatch)
    session = MagicMock()
    session.initialize = AsyncMock()
    session.call_tool = AsyncMock(return_value=_content_with_text(
        json.dumps({"issues": [{"key": "ISD-1"}, {"key": "ISD-2"}]})
    ))
    fake_session_mod.ClientSession.return_value.__aenter__ = AsyncMock(return_value=session)
    fake_session_mod.ClientSession.return_value.__aexit__ = AsyncMock(return_value=False)

    from src.bot.utils.mcp_client import McpJiraClient
    client = McpJiraClient("http://localhost:9001/mcp")
    issues = pytest.run_async_loopsafe(client.search_issues("category = X")) if hasattr(pytest, "run_async_loopsafe") else None

    # Fallback: use asyncio.run directly since pytest-asyncio auto mode might not
    # pick up this non-async test. Use anyio/asyncio.
    import asyncio
    issues = asyncio.get_event_loop().run_until_complete(client.search_issues("category = X"))
    assert [i["key"] for i in issues] == ["ISD-1", "ISD-2"]


def test_search_issues_handles_empty_issues_list(monkeypatch):
    fake_streamablehttp, fake_session_mod = _install_fake_mcp_sdk(monkeypatch)
    session = MagicMock()
    session.initialize = AsyncMock()
    session.call_tool = AsyncMock(return_value=_content_with_text(
        json.dumps({"issues": []})
    ))
    fake_session_mod.ClientSession.return_value.__aenter__ = AsyncMock(return_value=session)
    fake_session_mod.ClientSession.return_value.__aexit__ = AsyncMock(return_value=False)

    from src.bot.utils.mcp_client import McpJiraClient
    client = McpJiraClient("http://localhost:9001/mcp")
    import asyncio
    issues = asyncio.get_event_loop().run_until_complete(client.search_issues("category = X"))
    assert issues == []


def test_search_issues_returns_empty_on_mcp_call_failure(monkeypatch):
    """A network/server failure surfaces as a None result → [] returned."""
    fake_streamablehttp, fake_session_mod = _install_fake_mcp_sdk(monkeypatch)

    # The streamablehttp_client context manager raises.
    def raising_client(url, timeout=None):
        class _R:
            async def __aenter__(self): raise ConnectionError("server down")
            async def __aexit__(self, *exc): return False
        return _R()
    fake_streamablehttp.streamablehttp_client = raising_client

    from src.bot.utils.mcp_client import McpJiraClient
    client = McpJiraClient("http://localhost:9001/mcp")
    import asyncio
    issues = asyncio.get_event_loop().run_until_complete(client.search_issues("category = X"))
    assert issues == []


def test_search_issues_returns_empty_when_sdk_not_installed(monkeypatch):
    """If `mcp` SDK is missing entirely, all calls return []. Lazy import
    swallows ImportError as a structured warning."""
    for mod in ("mcp", "mcp.client", "mcp.client.streamablehttp", "mcp.client.session"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    real_import = __import__

    def blocking_import(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("mcp SDK not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", blocking_import):
        from src.bot.utils.mcp_client import McpJiraClient
        client = McpJiraClient("http://localhost:9001/mcp")
        import asyncio
        issues = asyncio.get_event_loop().run_until_complete(client.search_issues("X"))
        assert issues == []


def test_get_issue_returns_dict(monkeypatch):
    fake_streamablehttp, fake_session_mod = _install_fake_mcp_sdk(monkeypatch)
    session = MagicMock()
    session.initialize = AsyncMock()
    session.call_tool = AsyncMock(return_value=_content_with_text(
        json.dumps({"key": "ISD-1", "fields": {"summary": "x"}})
    ))
    fake_session_mod.ClientSession.return_value.__aenter__ = AsyncMock(return_value=session)
    fake_session_mod.ClientSession.return_value.__aexit__ = AsyncMock(return_value=False)

    from src.bot.utils.mcp_client import McpJiraClient
    client = McpJiraClient("http://localhost:9001/mcp")
    import asyncio
    issue = asyncio.get_event_loop().run_until_complete(client.get_issue("ISD-1"))
    assert issue == {"key": "ISD-1", "fields": {"summary": "x"}}


def test_get_issue_returns_none_on_invalid_json(monkeypatch):
    fake_streamablehttp, fake_session_mod = _install_fake_mcp_sdk(monkeypatch)
    session = MagicMock()
    session.initialize = AsyncMock()
    session.call_tool = AsyncMock(return_value=_content_with_text("not json"))
    fake_session_mod.ClientSession.return_value.__aenter__ = AsyncMock(return_value=session)
    fake_session_mod.ClientSession.return_value.__aexit__ = AsyncMock(return_value=False)

    from src.bot.utils.mcp_client import McpJiraClient
    client = McpJiraClient("http://localhost:9001/mcp")
    import asyncio
    issue = asyncio.get_event_loop().run_until_complete(client.get_issue("ISD-1"))
    assert issue is None


def test_list_tools_returns_tools(monkeypatch):
    fake_streamablehttp, fake_session_mod = _install_fake_mcp_sdk(monkeypatch)
    session = MagicMock()
    session.initialize = AsyncMock()
    tools_result = MagicMock()
    tool = MagicMock()
    tool.name = "jira_search_issues"
    tool.description = "Search Jira issues via JQL"
    tools_result.tools = [tool]
    session.list_tools = AsyncMock(return_value=tools_result)
    fake_session_mod.ClientSession.return_value.__aenter__ = AsyncMock(return_value=session)
    fake_session_mod.ClientSession.return_value.__aexit__ = AsyncMock(return_value=False)

    from src.bot.utils.mcp_client import McpJiraClient
    client = McpJiraClient("http://localhost:9001/mcp")
    import asyncio
    tools = asyncio.get_event_loop().run_until_complete(client.list_tools())
    assert any(t["name"] == "jira_search_issues" for t in tools)


def test_extract_content_normalizes_call_tool_result():
    from src.bot.utils.mcp_client import McpJiraClient
    item = MagicMock()
    item.type = "text"
    item.text = "hello"
    result = MagicMock()
    result.content = [item]
    out = McpJiraClient._extract_content(result)
    assert out == [{"type": "text", "text": "hello"}]


def test_extract_content_empty_for_none():
    from src.bot.utils.mcp_client import McpJiraClient
    assert McpJiraClient._extract_content(None) == []
