"""Holmes tool wrapper around multi-source memory retrieval.

Registers `search_memory` as a Holmes tool so Holmes can call it during
investigations. The tool queries 3 sources in parallel and returns a
tagged list so the LLM can reason about provenance:

  1. Graphiti knowledge graph (via MemoryService.retrieve_relevant)
  2. Distilled Facts (auto-extracted from prior bot conversations; from
     the `distilled_facts` MongoDB collection)
  3. KB Rules (FPT Container published KB; authoritative — from the
     `kb_rules` MongoDB collection)

The tool-use policy (when to call, what query text) is governed by a
rule in HOLMES_SYSTEM_PROMPT_ADDITIONS — not by Holmes deciding freely.

MemoryService owns its dedicated event loop and handles cross-loop
bridging internally via `MemoryService.run_sync(coro)`. We call that from
the sync `_invoke`.
"""
import logging
from typing import Dict, List, Optional

import structlog
from holmes.core.tools import Tool, ToolParameter, StructuredToolResult, StructuredToolResultStatus, ToolInvokeContext

logger = structlog.get_logger(__name__)


_DEFAULT_SOURCES = "graphiti,distilled,kb_rule"
_VALID_SOURCES = {"graphiti", "distilled", "kb_rule"}
_DEFAULT_K = 10


class SearchMemoryTool(Tool):
    """Tool exposing multi-source memory retrieval to Holmes.

    Description is intentionally specific so the LLM, combined with the
    rule in HOLMES_SYSTEM_PROMPT_ADDITIONS, calls this only when it has
    concrete signals from prior tool results — not on raw user messages.
    """

    name: str = "search_memory"
    description: str = (
        "Retrieve facts from multiple memory sources: Graphiti knowledge graph, "
        "Distilled Facts (auto-extracted from prior bot conversations), and "
        "KB Rules (FPT Container published KB; authoritative). Returns tagged "
        "results per source so you can reason about provenance. Call ONLY AFTER "
        "you have run at least one investigative tool and have concrete signals "
        "(pod names, error messages, service names, metric anomalies). Pass a "
        "concise query distilled from those signals, NOT the raw user message. "
        "Use `sources` param to restrict lookups (e.g. 'kb_rule' for authoritative "
        "diagnosis, 'distilled' for lived-experience observations)."
    )
    parameters: Dict[str, ToolParameter] = {
        "query": ToolParameter(
            description=(
                "Concise search terms distilled from concrete tool results "
                "(e.g. 'api-gw-7 CrashLoopBackOff', 'mysql connection refused prod'). "
                "Avoid full sentences."
            ),
            type="string",
            required=True,
        ),
        "k": ToolParameter(
            description="Max number of results to return PER source. Default 10.",
            type="integer",
            required=False,
        ),
        "sources": ToolParameter(
            description=(
                "Comma-separated list of sources to query. Default "
                "'graphiti,distilled,kb_rule'. Restrict for targeted lookups: "
                "'kb_rule' for authoritative KB, 'distilled' for chat-derived "
                "observations, 'graphiti' for the curated knowledge graph."
            ),
            type="string",
            required=False,
        ),
    }
    user_description: Optional[str] = (
        "Searched memory for: {{query}}"
    )

    def __init__(self, memory_service=None, distilled_store=None, kb_store=None):
        super().__init__()
        self._memory = memory_service
        self._distilled_store = distilled_store
        self._kb_store = kb_store

    def get_parameterized_one_liner(self, params: Dict) -> str:
        q = (params or {}).get("query", "")
        return f"search_memory(query={q!r})"

    def _invoke(self, params: dict, context: ToolInvokeContext) -> StructuredToolResult:
        query = (params or {}).get("query", "").strip()
        if not query:
            return StructuredToolResult(
                status=StructuredToolResultStatus.SUCCESS,
                data="Empty query — nothing to retrieve.",
                params=params,
                invocation=self.get_parameterized_one_liner(params),
            )
        try:
            k = int((params or {}).get("k") or _DEFAULT_K)
        except (TypeError, ValueError):
            k = _DEFAULT_K
        if k <= 0:
            k = _DEFAULT_K
        sources_raw = (params or {}).get("sources") or _DEFAULT_SOURCES
        sources = self._parse_sources(sources_raw)
        if not sources:
            sources = list(_VALID_SOURCES)

        sections: List[str] = []
        counts: Dict[str, int] = {}

        if "graphiti" in sources:
            facts_g = self._query_graphiti(query, k)
            counts["graphiti"] = len(facts_g)
            sections.append(self._format_graphiti_section(facts_g))
        if "distilled" in sources:
            facts_d = self._query_distilled(query, k)
            counts["distilled"] = len(facts_d)
            sections.append(self._format_distilled_section(facts_d))
        if "kb_rule" in sources:
            facts_kb = self._query_kb_rules(query, k)
            counts["kb_rule"] = len(facts_kb)
            sections.append(self._format_kb_section(facts_kb))

        total = sum(counts.values())
        header = (
            f"=== SEARCH_MEMORY RESULTS ({len(counts)} source(s) queried) ==="
        )
        body = "\n\n".join(sections)
        footer = f"\n(N={total} results total; query: {query!r}; sources: {','.join(sorted(counts.keys()))})"
        data = f"{header}\n\n{body}\n{footer}" if sections else "No configured sources returned results."
        return StructuredToolResult(
            status=StructuredToolResultStatus.SUCCESS,
            data=data,
            params=params,
            invocation=self.get_parameterized_one_liner(params),
        )

    @staticmethod
    def _parse_sources(raw: str) -> List[str]:
        out: List[str] = []
        for tok in (raw or "").split(","):
            s = tok.strip().lower()
            if s in _VALID_SOURCES and s not in out:
                out.append(s)
        return out

    def _query_graphiti(self, query: str, k: int) -> List[str]:
        if self._memory is None:
            return []
        try:
            facts = self._run_retrieve_graphiti(query, k)
        except Exception as e:
            logger.warning("search_memory_graphiti_failed",
                           error=str(e), error_type=type(e).__name__)
            return []
        return [str(f) for f in (facts or [])]

    def _query_distilled(self, query: str, k: int) -> List[dict]:
        if self._distilled_store is None:
            return []
        try:
            return self._distilled_store.search(query, k=k)
        except Exception as e:
            logger.warning("search_memory_distilled_failed",
                           error=str(e), error_type=type(e).__name__)
            return []

    def _query_kb_rules(self, query: str, k: int) -> List[dict]:
        if self._kb_store is None:
            return []
        try:
            rules = self._kb_store.get_relevant_rules(query, top_k=k)
            return rules or []
        except Exception as e:
            logger.warning("search_memory_kb_failed",
                           error=str(e), error_type=type(e).__name__)
            return []

    @staticmethod
    def _format_graphiti_section(facts: List[str]) -> str:
        lines = ["--- Graphiti (extracted knowledge graph) ---"]
        if not facts:
            lines.append("(no results)")
        else:
            for f in facts:
                lines.append(f"- {f}")
        return "\n".join(lines)

    @staticmethod
    def _format_distilled_section(facts: List[dict]) -> str:
        lines = [
            "--- Distilled Facts (chat-derived, from distilled_facts MongoDB collection) ---"
        ]
        if not facts:
            lines.append("(no results)")
        else:
            for f in facts:
                text = (f.get("text") or "").strip()
                run_id = f.get("source_run_id")
                score = f.get("score")
                prefix_parts = []
                if run_id:
                    prefix_parts.append(f"run_id={run_id}")
                if score is not None:
                    try:
                        prefix_parts.append(f"score={float(score):.3f}")
                    except (TypeError, ValueError):
                        pass
                prefix = f"[{', '.join(prefix_parts)}] " if prefix_parts else ""
                lines.append(f"- {prefix}{text}")
        return "\n".join(lines)

    @staticmethod
    def _format_kb_section(rules: List[dict]) -> str:
        lines = [
            "--- KB Rules (FPT Container KB; authoritative) ---"
        ]
        if not rules:
            lines.append("(no results)")
        else:
            for r in rules:
                rid = r.get("rule_id") or "?"
                mc = r.get("match_count", 0)
                title = (r.get("issue_title") or "").strip()
                text = (r.get("rule_text") or "").strip()
                if len(text) > 600:
                    text = text[:600].rstrip() + " ..."
                prefix_parts = [f"rule_id={rid}", f"match_count={mc}"]
                if title:
                    prefix_parts.append(f"title={title!r}")
                prefix = f"[{', '.join(prefix_parts)}] "
                lines.append(f"- {prefix}{text}")
        return "\n".join(lines)

    def _run_retrieve_graphiti(self, query: str, k: int):
        """Run retrieve_relevant on the memory service's loop and block.

        Tests typically stub `memory_service.retrieve_relevant` with an
        AsyncMock. The real service exposes `run_sync(coro, timeout)` which
        drives the coroutine via the service's dedicated loop. For test
        mocks (MagicMock) we fall back to asyncio.run in this thread — works
        because AsyncMock returns a real coroutine.
        """
        from src.bot.services.memory_service import MemoryService
        coro = self._memory.retrieve_relevant(query, k=k)
        if isinstance(self._memory, MemoryService):
            return self._memory.run_sync(coro, timeout=30.0)
        import asyncio as _asyncio
        return _asyncio.run(coro)
