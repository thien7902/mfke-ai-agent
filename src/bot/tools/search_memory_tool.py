"""Holmes tool wrapper around MemoryService.retrieve_relevant.

Registers `search_memory` as a Holmes tool so Holmes can call it during
investigations. The tool-use policy (when to call, what query text) is
governed by a rule in HOLMES_SYSTEM_PROMPT_ADDITIONS — not by Holmes
deciding freely.

Tool.invoke is sync, but MemoryService.retrieve_relevant is async. We
bridge via a dedicated event loop running in a worker thread so we don't
stomp on Telegram's main loop and don't deadlock when called from inside
Holmes' own thread pool.
"""
import asyncio
import logging
import threading
from typing import Dict, Optional

import structlog
from holmes.core.tools import Tool, ToolParameter, StructuredToolResult, StructuredToolResultStatus, ToolInvokeContext

logger = structlog.get_logger(__name__)


# Module-level loop + thread shared by all search_memory invocations.
# Created lazily on first invoke — avoids spinning a thread in tests that
# never call the tool.
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()
_loop_thread: Optional[threading.Thread] = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    if _loop is not None and not _loop.is_closed():
        return _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(
                target=_loop.run_forever, name="search-memory-loop", daemon=True
            )
            _loop_thread.start()
    return _loop


def _run_async(coro, timeout: float = 30.0):
    """Run an async coroutine from sync context on the dedicated loop."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except Exception as e:
        logger.warning("search_memory_async_failed", error=str(e))
        return []


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
        # Pydantic BaseModel __init__ — pass fields explicitly then stash svc
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

        # MemoryService.retrieve_relevant is async — bridge to sync.
        facts = _run_async(self._memory.retrieve_relevant(query, k=k))
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
