"""Holmes tool wrapper around MemoryService.retrieve_relevant.

Registers `search_memory` as a Holmes tool so Holmes can call it during
investigations. The tool-use policy (when to call, what query text) is
governed by a rule in HOLMES_SYSTEM_PROMPT_ADDITIONS — not by Holmes
deciding freely.

MemoryService owns its dedicated event loop and handles cross-loop
bridging internally via `MemoryService.run_sync(coro)`. We call that from
the sync `_invoke`.
"""
import logging
from typing import Dict, Optional

import structlog
from holmes.core.tools import Tool, ToolParameter, StructuredToolResult, StructuredToolResultStatus, ToolInvokeContext

logger = structlog.get_logger(__name__)


class SearchMemoryTool(Tool):
    """Tool exposing memory retrieval to Holmes.

    Description is intentionally specific so the LLM, combined with the
    rule in HOLMES_SYSTEM_PROMPT_ADDITIONS, calls this only when it has
    concrete signals from prior tool results — not on raw user messages.
    """

    name: str = "search_memory"
    description: str = (
        "Retrieve relevant facts from the bot's knowledge memory. "
        "Call ONLY AFTER you have run at least one investigative tool and "
        "have concrete signals (pod names, error messages, service names, "
        "metric anomalies). Pass a concise query distilled from those "
        "signals, NOT the raw user message. Returns a list of facts. "
        "Use the facts if relevant; ignore them if not. Optional — skip if "
        "the question does not need prior knowledge."
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
            description="Max number of facts to return. Default 20.",
            type="integer",
            required=False,
        ),
    }
    user_description: Optional[str] = (
        "Searched memory for: {{query}}"
    )

    def __init__(self, memory_service):
        super().__init__()
        self._memory = memory_service

    def get_parameterized_one_liner(self, params: Dict) -> str:
        q = (params or {}).get("query", "")
        return f"search_memory(query={q!r})"

    def _invoke(self, params: dict, context: ToolInvokeContext) -> StructuredToolResult:
        if self._memory is None:
            return StructuredToolResult(
                status=StructuredToolResultStatus.SUCCESS,
                data="Memory service is disabled or not initialized.",
                params=params,
                invocation=self.get_parameterized_one_liner(params),
            )
        query = (params or {}).get("query", "").strip()
        if not query:
            return StructuredToolResult(
                status=StructuredToolResultStatus.SUCCESS,
                data="Empty query — nothing to retrieve.",
                params=params,
                invocation=self.get_parameterized_one_liner(params),
            )
        try:
            k = int((params or {}).get("k") or 20)
        except (TypeError, ValueError):
            k = 20

        try:
            facts = self._run_retrieve(query, k)
        except Exception as e:
            logger.warning("search_memory_failed", error=str(e),
                           error_type=type(e).__name__)
            facts = []

        if not facts:
            data = "No relevant facts found in memory."
        else:
            data = "\n".join(f"- {f}" for f in facts)
        return StructuredToolResult(
            status=StructuredToolResultStatus.SUCCESS,
            data=data,
            params=params,
            invocation=f"search_memory(query={query!r}, k={k})",
        )

    def _run_retrieve(self, query: str, k: int):
        """Run retrieve_relevant on the memory service's loop and block.

        Tests typically stub `memory_service.retrieve_relevant` with an
        AsyncMock. The real service exposes `run_sync(coro, timeout)` which
        drives the coroutine via the service's dedicated loop. For test
        mocks (MagicMock) we fall back to asyncio.run in this thread — works
        because AsyncMock returns a real coroutine.

        Don't try/catch inside — outer caller handles exceptions.
        """
        from src.bot.services.memory_service import MemoryService
        coro = self._memory.retrieve_relevant(query, k=k)
        if isinstance(self._memory, MemoryService):
            return self._memory.run_sync(coro, timeout=30.0)
        import asyncio as _asyncio
        return _asyncio.run(coro)

        if not facts:
            data = "No relevant facts found in memory."
        else:
            data = "\n".join(f"- {f}" for f in facts)
        return StructuredToolResult(
            status=StructuredToolResultStatus.SUCCESS,
            data=data,
            params=params,
            invocation=f"search_memory(query={query!r}, k={k})",
        )
