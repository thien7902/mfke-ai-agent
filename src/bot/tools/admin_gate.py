"""Admin-gate that blocks non-admins from running mutating K8s/Helm commands.

Wraps the Holmes `bash` tool's `invoke`. When the LLM tries to run a mutating
kubectl/helm command (apply/delete/edit/patch/scale/replace/rollout/exec/cp/
create/label/annotate/taint/cordon/uncordon/drain, helm install/upgrade/uninstall/
rollback/delete/repo add/remove), the wrapper checks the calling user's admin
status. Admins skip Holmes' built-in allow-list (context.user_approved=True) for
1-prompt execution; non-admins get a StructuredToolResult ERROR that tells the
LLM to pivot to read-only investigation in the same turn.

UserId reaches the wrapper via `context.request_context["user_id"]` (already
set in holmes_service._request_context at chat() entry). No new plumbing.

This is the app-level enforcement layer. Holmes' built-in safeguard
(`prevent_overly_repeated_tool_call`) does byte-identical detection only and is
disabled when user_approved=True — this gate is what makes admin status matter
for mutations.
"""
import asyncio
import re
from typing import Any, Optional

import structlog

from src.bot.utils.config import config

logger = structlog.get_logger(__name__)


_ADMIN_LOOP: Optional[asyncio.AbstractEventLoop] = None
_ADMIN_LOOKUP_TIMEOUT_S = 1.5


# Regex patterns for mutating kubectl / helm subcommands. Anchored at start
# (re.match). env-prefix stripping is done before matching, so leading
# `KUBECONFIG=...` is already gone. Whitespace prefix is tolerated by re.match
# implicitly (it matches at position 0 only — we strip first).
#
# `kubectl_run_image` (separate Holmes tool, not bash) is OUT OF SCOPE here —
# it's disabled by default and we don't wrap it.
MUTATING_PATTERNS: tuple = (
    # kubectl mutating subcommands. Allow optional resource-type positional
    # between `kubectl` and the verb (e.g. `kubectl rollout undo`, `kubectl
    # scale deploy`, `kubectl apply -f`). Match the verb right after kubectl.
    re.compile(r"^kubectl\s+(apply|delete|edit|patch|scale|replace|rollout|exec|cp|create|label|annotate|taint|cordon|uncordon|drain|rolling-update|expose|set\s+env|set\s+image|set\s+resources|set\s+serviceaccount|set\s+selector|auth\s+reconcile|auth\s+can-i\s+--delete|taint\s+)\b"),
    # helm mutating subcommands. install/upgrade/uninstall/rollback/delete pull
    # repo add/remove are the ones that change cluster or repo state.
    re.compile(r"^helm\s+(install|upgrade|uninstall|rollback|delete|repo\s+add|repo\s+remove|push|pull|cm-push|registry\s+login)\b"),
)


_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*\s+")


def _strip_env_prefix(cmd: str) -> str:
    """Strip leading VAR=value assignments and whitespace so detection works.

    The LLM sometimes prepends KUBECONFIG=/tmp/x or similar env vars to a
    bash command. The mutation keyword is still in there, just past the
    env prefix. Strip iteratively (multiple env vars can stack).
    """
    if not cmd:
        return ""
    s = cmd.strip()
    # Strip leading env assignments one at a time (handles
    # `A=1 B=2 kubectl apply ...`).
    while True:
        m = _ENV_ASSIGN_RE.match(s)
        if not m:
            break
        s = s[m.end():]
    return s.strip()


def _is_mutation(cmd: Any) -> bool:
    """True if `cmd` is a mutating kubectl/helm command.

    Returns False on empty / non-string / parse error — never raises.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        return False
    try:
        stripped = _strip_env_prefix(cmd)
    except Exception:
        return False
    if not stripped:
        return False
    for pat in MUTATING_PATTERNS:
        if pat.match(stripped):
            return True
    return False


def set_admin_check_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Register the bot's event loop for sync admin lookups.

    The bash tool's `invoke` runs sync inside Holmes' ToolCallingLLM worker
    thread. When `is_admin_sync` can't resolve from the static admin list, it
    needs to schedule an async PermissionService.get_user call onto the bot
    loop. `set_admin_check_loop` is called once per chat() turn with the
    running loop.
    """
    global _ADMIN_LOOP
    _ADMIN_LOOP = loop


async def _async_is_admin(telegram_id: int) -> bool:
    """Re-export of `_is_admin` from decorators, kept local to avoid import
    cycle (decorators.py imports from services which imports tools)."""
    from src.bot.utils.decorators import _is_admin
    return await _is_admin(telegram_id)


def is_admin_sync(telegram_id_str: Optional[str]) -> bool:
    """Synchronous admin check callable from Holmes worker thread.

    Fast path: telegram_id is in config.admin_user_ids (no DB, no loop).
    Slow path: schedule `_is_admin(int(id))` on the bot loop captured by
    `set_admin_check_loop` and wait up to _ADMIN_LOOKUP_TIMEOUT_S seconds.

    Returns False on None/parse error / loop not set / timeout. Treats
    unknown as non-admin (safer default — the wrapper will block).
    """
    if telegram_id_str is None:
        return False
    try:
        tid = int(telegram_id_str)
    except (TypeError, ValueError):
        return False

    # Fast path: static admin list. O(1), no DB hit, no loop hop.
    try:
        if tid in config.admin_user_ids:
            return True
    except Exception:
        return False

    # Slow path: hop to bot loop for PermissionService.get_user (Mongo).
    loop = _ADMIN_LOOP
    if loop is None or loop.is_closed():
        return False
    try:
        future = asyncio.run_coroutine_threadsafe(_async_is_admin(tid), loop)
        try:
            return future.result(timeout=_ADMIN_LOOKUP_TIMEOUT_S)
        except TimeoutError:
            # Don't leave the coroutine dangling — cancel so the loop can clean
            # it up on its next tick instead of leaking a pending task.
            future.cancel()
            return False
    except Exception:
        return False


def wrap_tools_with_admin_gate(tool_executor) -> None:
    """Wrap the `bash` tool's invoke with the admin-gate. Idempotent.

    Wraps AFTER `_wrap_bash_tool_timeout` and `wrap_tools_with_loop_guard` so
    admin-gate is the OUTERMOST wrapper — a blocked mutating call short-
    circuits before timeout clamping and loop tracking run (correct: blocked
    calls shouldn't count toward the loop threshold).
    """
    if tool_executor is None:
        return

    for toolset in getattr(tool_executor, "toolsets", []) or []:
        for tool in getattr(toolset, "tools", []) or []:
            if tool.name != "bash":
                continue
            if getattr(tool, "_admin_gate_wrapped", False):
                continue
            _wrap_one(tool)


def _wrap_one(tool) -> None:
    original_invoke = tool.invoke

    def wrapped_invoke(params, context=None):
        cmd = (params or {}).get("command") if isinstance(params, dict) else None
        if not _is_mutation(cmd):
            return original_invoke(params, context)

        request_ctx = getattr(context, "request_context", None) or {}
        user_id_str = request_ctx.get("user_id")
        is_admin = is_admin_sync(user_id_str)

        if is_admin:
            # Skip Holmes' APPROVAL_REQUIRED prompt for admins — 1-prompt
            # mutation execution. bash_toolset.py:234 bypasses validation
            # entirely when user_approved=True.
            try:
                # ToolInvokeContext is a dataclass — set field directly.
                object.__setattr__(context, "user_approved", True)
            except Exception:
                pass
            logger.info(
                "admin_gate_admin_execute",
                command_preview=str(cmd)[:120],
                user_id=user_id_str,
            )
            return original_invoke(params, context)

        logger.warning(
            "admin_gate_blocked",
            command_preview=str(cmd)[:120],
            user_id=user_id_str,
            reason="mutation_prefix_non_admin",
        )
        return _build_block_result(cmd)

    object.__setattr__(tool, "invoke", wrapped_invoke)
    object.__setattr__(tool, "_admin_gate_wrapped", True)
    logger.info("Wrapped bash tool with admin_gate")


def _build_block_result(cmd: Any):
    """Return StructuredToolResult ERROR with an LLM-facing pivot hint.

    Lazy import so this module is importable without holmes installed (for
    unit-test of _is_mutation / is_admin_sync slow path).
    """
    from holmes.core.tools import StructuredToolResult, StructuredToolResultStatus

    cmd_preview = str(cmd)[:200] if cmd is not None else ""
    return StructuredToolResult(
        status=StructuredToolResultStatus.ERROR,
        error=(
            "Admin access required: this is a mutating Kubernetes/Helm command "
            "and the current user is not an admin. Do NOT retry this command or "
            "variants of it. Pivot now to read-only investigation: run "
            "`kubectl get`/`describe`/`logs`/`explain` for the resource, or "
            "`helm list`/`helm get manifest`, then report findings and the exact "
            "change an admin should apply. Tell the user an admin is needed."
        ),
        params={"command": cmd_preview},
        invocation="bash(blocked-by-admin-gate)",
    )
