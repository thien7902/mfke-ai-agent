"""Async MCP HTTP/SSE client wrapper for mcp-atlassian (or any MCP server
exposing Jira tools). Uses the `mcp` Python SDK's streamable HTTP transport.

The `mcp` SDK is imported lazily inside `_call` so a missing dependency only
fails at call time (with a structured warning), not at module import. This
matches the bot's existing failure-isolation pattern — see MemoryService init
at main.py:158-171 and DistilledFactStore.get_injectable_facts.

All public methods return [] / None on any failure so callers don't have to
wrap every call in try/except.
"""

import json
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class McpJiraClient:
    """Async client for mcp-atlassian (or any MCP server exposing Jira tools)
    over HTTP/SSE. The MCP SDK handles the SSE handshake + JSON-RPC 2.0 framing.

    This is the first async HTTP client pattern in this codebase — pattern
    mirrors existing service-style: lazy client, structured logging via stdlib
    `logging` (kept stdlib here because structlog is not used in utils/).
    """

    DEFAULT_TIMEOUT_SECONDS = 30

    def __init__(self, url: str, request_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._url = url
        self._timeout = request_timeout_seconds

    async def list_tools(self) -> list[dict]:
        """Returns the tools exposed by the MCP server. Useful for /oncall test
        and for verifying the server is reachable. Returns [] on any failure."""
        result = await self._call("tools/list", {})
        if not result:
            return []
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "tools" in result:
            return result["tools"]
        return []

    async def search_issues(self, jql: str, fields: Optional[list[str]] = None,
                            max_results: int = 100) -> list[dict]:
        """Calls mcp-atlassian's `jira_search_issues` tool with the given JQL.

        Returns a list of issue dicts (the Jira issue shape: {key, fields: {...}, ...}).
        Returns [] on any failure (server down, SDK missing, parse error).
        """
        args: dict[str, Any] = {"jql": jql, "maxResults": max_results}
        if fields:
            args["fields"] = fields
        result = await self._call_tool("jira_search_issues", args)
        return self._parse_issue_list(result)

    async def get_issue(self, key: str) -> Optional[dict]:
        """Fetches a single issue's full metadata via mcp-atlassian's
        `jira_get_issue` tool. Returns None on any failure."""
        result = await self._call_tool("jira_get_issue", {"issue_key": key})
        if not result:
            return None
        text = self._first_text(result)
        if text is None:
            return None
        data = self._parse_json_lenient(text)
        if data is None:
            logger.warning("mcp_jira_get_issue_parse_failed", text_preview=text[:200])
            return None
        return data if isinstance(data, dict) else None

    async def _call_tool(self, name: str, args: dict) -> Any:
        return await self._call("tools/call", {"name": name, "arguments": args})

    async def _call(self, method: str, params: dict) -> Any:
        try:
            # Module name is `streamable_http` (with underscore) on the
            # shipped `mcp` SDK. The function is `streamablehttp_client`
            # (no underscore). Don't confuse the two — both forms exist
            # in online references but only the underscore module name
            # matches what `mcp>=1.2.0` exposes.
            from mcp.client.streamable_http import streamablehttp_client
            from mcp.client.session import ClientSession
        except ImportError as e:
            logger.warning("mcp_sdk_not_installed", error=str(e))
            return None
        try:
            async with streamablehttp_client(self._url, timeout=self._timeout) as (
                read, write, _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    if method == "tools/call":
                        tool_name = params.get("name")
                        tool_args = params.get("arguments", {}) or {}
                        result = await session.call_tool(tool_name, tool_args)
                        return self._extract_content(result)
                    elif method == "tools/list":
                        result = await session.list_tools()
                        return [{"name": t.name, "description": getattr(t, "description", "")}
                                for t in result.tools]
                    return None
        except Exception as e:
            logger.warning("mcp_call_failed", method=method, error=str(e))
            return None

    @staticmethod
    def _extract_content(result: Any) -> list[dict]:
        """Normalize mcp SDK's CallToolResult to a plain list of {type, text}
        dicts. Returns [] for None/unknown shapes."""
        if result is None:
            return []
        if hasattr(result, "content"):
            return [{"type": getattr(c, "type", "text"),
                     "text": getattr(c, "text", "")}
                    for c in result.content]
        if isinstance(result, list):
            return result
        return []

    @staticmethod
    def _first_text(content: list[dict]) -> Optional[str]:
        if not content:
            return None
        first = content[0]
        if isinstance(first, dict):
            return first.get("text") or None
        return None

    @classmethod
    def _parse_issue_list(cls, content: list[dict]) -> list[dict]:
        if not content:
            return []
        text = cls._first_text(content)
        if text is None:
            return []
        data = cls._parse_json_lenient(text)
        if data is None:
            logger.warning("mcp_jira_search_parse_failed", text_preview=text[:200])
            return []
        if isinstance(data, dict) and "issues" in data:
            return data["issues"] or []
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _parse_json_lenient(text: str) -> Any:
        """Tolerant JSON extraction. mcp-atlassian sometimes wraps tool
        output in markdown fences (```json ... ```) or prefixed prose.
        Strip fences, then try direct json.loads; on failure, grab the
        first balanced {...} or [...] substring and retry."""
        if not text:
            return None
        s = text.strip()
        # Strip a single surrounding markdown code fence if present.
        if s.startswith("```"):
            # Drop the opening fence line (``` or ```json).
            nl = s.find("\n")
            if nl != -1:
                s = s[nl + 1:]
            if s.rstrip().endswith("```"):
                s = s.rstrip()[:-3]
            s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        # Fall back: find the first balanced JSON object/array.
        for opener, closer in (("{", "}"), ("[", "]")):
            start = s.find(opener)
            if start == -1:
                continue
            end = s.rfind(closer)
            if end == -1 or end <= start:
                continue
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                continue
        return None
