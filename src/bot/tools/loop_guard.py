"""Per-tool loop guard that blocks repeated identical tool calls.

Wraps every tool's `invoke` on a Holmes ToolExecutor. The wrapper keeps a
sliding window of recent successful calls per tool (normalized params) and
returns a StructuredToolResult ERROR when the same normalized params would
hit the threshold within the window.

This exists because Holmes 0.36.0's built-in `prevent_overly_repeated_tool_call`
safeguard (in holmes.core.safeguards) only catches byte-identical params and
goes blind after compaction. PR #2433 in HolmesGPT upstream adds a proper
sliding-window detector but as of 2026-10 it is still unmerged. This local
wrapper is the app-level backport — applies to every tool (bash, kubectl,
curl, search_memory, ...) regardless of Holmes version.

Configuration is module-level so tests can monkey-patch before wrapping:
- LOOP_GUARD_WINDOW:        number of recent successful calls kept per tool
- LOOP_GUARD_THRESHOLD:     count of identical-normalized calls within window
                             that triggers a block (call N+1 is blocked)
- LOOP_GUARD_SKIP_TOOLS:    tool names exempt from the guard (e.g. todo_write,
                             which legitimately writes multiple times)
"""
import json
from collections import deque
from threading import Lock
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


LOOP_GUARD_WINDOW = 5
LOOP_GUARD_THRESHOLD = 4
LOOP_GUARD_SKIP_TOOLS = frozenset({"todo_write"})


def _normalize_value(v: Any) -> Any:
    """Recursively normalize string values: strip + collapse whitespace + lower.

    Whitespace inside JSON-quoted values (e.g. `"  Hello   World  "`) is
    significant semantically but cosmetic for loop detection — LLM "retry
    with extra space" loops should still match. Done at the value level,
    not the JSON-string level, so quote characters don't break tokenization.
    """
    if isinstance(v, str):
        return " ".join(v.split()).lower()
    if isinstance(v, dict):
        return {str(k).lower(): _normalize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_normalize_value(x) for x in v]
    return v


def _normalize_params(params: Any) -> str:
    """Stable string fingerprint of a tool-call params dict.

    Normalizes whitespace inside string values (collapse + strip), lowercases
    all string values and dict keys, sorts dict keys. Two calls with cosmetic
    differences map to the same fingerprint — catches LLM "retry with extra
    space" loops. Lazy JSON dump keeps the fingerprint stable acrossruns.
    """
    if params is None:
        return ""
    if isinstance(params, str):
        return " ".join(params.split()).lower()
    normalized = _normalize_value(params)
    try:
        return json.dumps(normalized, sort_keys=True, default=str).lower()
    except (TypeError, ValueError):
        return str(normalized).lower()


def _make_key(tool_name: str, params: Any) -> str:
    return f"{tool_name}|{_normalize_params(params)}"


def wrap_tools_with_loop_guard(tool_executor) -> None:
    """Monkey-patch invoke on every tool on the executor.

    Idempotent: a tool already wrapped is skipped (flag set on the wrapper).
    Run after all toolsets are loaded — typically right after
    ToolExecutor construction. Mirrors the pattern in
    HolmesService._wrap_bash_tool_timeout / _wrap_todo_write_tool.
    """
    if tool_executor is None:
        return

    for toolset in getattr(tool_executor, "toolsets", []) or []:
        for tool in getattr(toolset, "tools", []) or []:
            if tool.name in LOOP_GUARD_SKIP_TOOLS:
                continue
            if getattr(tool, "_loop_guard_wrapped", False):
                continue
            _wrap_one(tool)


def _wrap_one(tool) -> None:
    original_invoke = tool.invoke
    recent: Deque[str] = deque(maxlen=LOOP_GUARD_WINDOW)
    lock = Lock()

    def wrapped_invoke(params, context=None):
        key = _make_key(tool.name, params)
        with lock:
            count = sum(1 for k in recent if k == key)
            # count = identical successes already tracked before this call.
            # If count+1 >= threshold, this call would push us over -> block.
            would_block = (count + 1) >= LOOP_GUARD_THRESHOLD
        if would_block:
            logger.warning(
                "loop_guard_blocked",
                tool=tool.name,
                threshold=LOOP_GUARD_THRESHOLD,
                window=LOOP_GUARD_WINDOW,
                recent_matches=count,
                params_preview=str(params)[:200],
            )
            return _build_block_result(tool.name, params)
        result = original_invoke(params, context)
        if _is_success(result):
            with lock:
                recent.append(key)
        return result

    object.__setattr__(tool, "invoke", wrapped_invoke)
    object.__setattr__(tool, "_loop_guard_wrapped", True)


def _is_success(result: Any) -> bool:
    """True if the tool returned a success StructuredToolResult.

    Failures and errors are NOT tracked — LLM-retry on transient errors is
    legitimate, and recording them would block retried-after-error loops
    incorrectly. Only successful identical calls count toward the threshold.
    """
    status = getattr(result, "status", None)
    status_val = getattr(status, "value", None) or str(status).lower()
    return "success" in status_val.lower()


def _build_block_result(tool_name: str, params: Any):
    """Return a StructuredToolResult ERROR with a clear LLM-facing message.

    Lazy import so this module is importable without holmes installed
    (e.g. for unit tests of _normalize_params).
    """
    from holmes.core.tools import StructuredToolResult, StructuredToolResultStatus
    return StructuredToolResult(
        status=StructuredToolResultStatus.ERROR,
        error=(
            f"Loop guard: refusing to call {tool_name} with the same parameters "
            f"as {LOOP_GUARD_THRESHOLD} recent successful calls. Reuse the previous "
            f"result, change the parameters, or move to a different tool."
        ),
        params=params if isinstance(params, dict) else {"raw": str(params)},
        invocation=f"{tool_name}(blocked-by-loop-guard)",
    )
