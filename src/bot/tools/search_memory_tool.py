"""Holmes tool wrapper around MemoryService.retrieve_relevant.

Registers `search_memory` as a Holmes tool so Holmes can call it during
investigations. The tool-use policy (when to call, what query text) is
governed by a rule in HOLMES_SYSTEM_PROMPT_ADDITIONS — not by Holmes
deciding freely.

MemoryService owns its dedicated event loop and handles cross-loop
bridging internally, so this tool is a trivial sync call into the service.
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
            description="Max number of facts to return. Default 5.",
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
            k = int((params or {}).get("k") or 5)
        except (TypeError, ValueError):
            k = 5

        # MemoryService.retrieve_relevant is async. Holmes calls _invoke
        # from a sync context (its own thread pool), so we need to await
        # the coroutine. The synchronous call we want is: block on the
        # event loop that owns graphiti's connections and return the result.
        # MemoryService owns that loop and exposes `_run` for sync callers.
        try:
            import asyncio as _asyncio
            # Is there a running loop in this thread? If yes (rare for Holmes
            # tools), we can't block on a coroutine — use the service's loop
            # via run_coroutine_threadsafe.
            try:
                _asyncio.get_running_loop()
                running = True
            except RuntimeError:
                running = False

            coro = self._memory.retrieve_relevant(query, k=k)
            if running:
                # Use the service's dedicated loop to avoid blocking our own.
                loop = self._memory._ensure_loop()
                future = _asyncio.run_coroutine_threadsafe(coro, loop)
                facts = future.result(timeout=30.0)
            else:
                # No loop running in this thread — block on the coroutine
                # via the service's loop.
                facts = self._memory._run(coro, timeout=30.0)
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
