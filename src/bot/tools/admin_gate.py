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


# Healthcare for cluster state: these patterns MUST be blocked regardless of
# admin status — they destroy native K8s/Gardener resources in a way that
# cannot be recovered (deleting a node loses the volume调度, editing a
# protected-namespace configMap can break the control plane, apply -f can
# re-create arbitrary state). Hard-block wins over admin-gate: even root
# admins cannot accidentally wipe out seed cluster control plane.
#
# Pattern A — `kubectl apply` is globally hard-blocked for ALL arguments
# (no -f filter): apply mutates cluster state declaratively and the file
# may contain any resource type. Read-only alternatives (get, describe)
# are NOT in this list.
#
# Pattern B — `kubectl delete <critical_resource>` is blocked regardless
# of namespace. Critical resources: node, pv, pvc, namespace, crd,
# clusterrole, service — singular + plural + short forms, case-insensitive.
# Deleting pod/deploy/sts/etc is allowed (recoverable from spec) — only
# the "unrecoverable" targets are protected here.
#
# Pattern C — any mutating verb (delete/edit/scale/patch/replace/drain)
# against a protected namespace: kube-system, kube-public, kube-node-lease,
# garden, gardener-system, extensions, or any namespace prefixed
# `garden-` (Gardener project/secrets namespaces that hold shoot
# control-plane wiring). Shoot namespaces (shoot--<project>--<name>) are
# NOT blocked by user choice — only the garden-* family is.

_CRITICAL_RESOURCE_RE = re.compile(
    r"^(?:[A-Za-z][\w-]*/)?("
    r"node|nodes|no|"
    r"persistentvolume|persistentvolumes|pv|"
    r"persistentvolumeclaim|persistentvolumeclaims|pvc|"
    r"namespace|namespaces|ns|"
    r"customresourcedefinition|customresourcedefinitions|crd|"
    r"clusterrole|clusterroles|"
    r"service|services|svc"
    r")(?:/[A-Za-z][\w-]*)?$",
    re.IGNORECASE,
)

# Protected namespace pattern. Accepts both `-n NAME` and `--namespace=NAME`
# (with optional `=` or whitespace separator). Names: kube-system,
# kube-public, kube-node-lease, garden, gardener-system, extensions, plus
# anything starting with `garden-`. `shoot--...` deliberately NOT matched
# per user choice.
_PROTECTED_NS_NAME_RE = re.compile(
    r"\b("
    r"kube-system|kube-public|kube-node-lease|"
    r"garden|gardener-system|extensions|"
    r"garden-[\w-]+"
    r")\b",
    re.IGNORECASE,
)

# Match `-n NAME` or `--namespace NAME` or `--namespace=NAME`. Allow the
# namespace value to appear in the next arg (split on space) or via `=`.
_NS_FLAG_RE = re.compile(
    r"(?:-n\s+|--namespace=?)\s*([^\s]+)",
    re.IGNORECASE,
)

# Pattern A — kubectl apply (any args, any ns).
_HARD_BLOCK_APPLY_RE = re.compile(r"^kubectl\s+apply\b", re.IGNORECASE)

# Pattern B — kubectl delete <critical_resource>. The verb comes right
# after `kubectl`, then resource-type positional. The critical-resource
# search is anchored to the FIRST positional token after `delete` (using
# `_extract_positional_resource`) so flag values like `--namespace=kube-node-lease`
# (which contains "node") and `--namespace=...` (which contains "namespace") do
# NOT false-positive.
_HARD_BLOCK_DELETE_RE = re.compile(r"^kubectl\s+delete\b", re.IGNORECASE)

# Resource-type positional token extraction. Skips flags (tokens starting
# with `-`). Returns the first non-flag token after the verb, or "" if
# none found. Handles `/`-segregated forms like `pod/foo` and `service/bar`.
_RESOURCE_TYPE_TOKEN_RE = re.compile(r"^[A-Za-z][\w./-]*$")

# Pattern C — kubectl + mutating verb + protected namespace. Verbs are
# the destructive verbs that touch cluster state across namespaces:
# delete/edit/scale/patch/replace/drain. `apply` is handled by Pattern A
# alone since it is globally blocked.
_HARD_BLOCK_MUTATING_VERBS_RE = re.compile(
    r"^kubectl\s+(delete|edit|scale|patch|replace|drain)\b",
    re.IGNORECASE,
)


_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*\s+")
# Shell command separators we tolerate before the kubectl/helm mutation.
# `&&`, `||`, `;` — any of these can sit between `export KUBECONFIG=...`
# and `kubectl delete ...`. We split on each occurrence so blocked segments
# past a separator still trigger. Aided by `_strip_env_prefix`.
_SEPARATOR_RE = re.compile(r"(?:&&|\|\||;)")


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


def _extract_segments(cmd: str) -> list:
    """Split ``cmd`` on shell separators (``&&``, ``||``, ``;``) into
    individual sub-commands, each with leading env vars stripped.

    `export KUBECONFIG=/tmp/x && kubectl delete node worker-1` becomes:
    `["export KUBECONFIG=/tmp/x", "kubectl delete node worker-1"]` — the
    first segment evaluates as benign (no kubectl/helm verb), the second
    triggers the mutation pattern.
    """
    if not cmd:
        return []
    parts = _SEPARATOR_RE.split(cmd)
    return [_strip_env_prefix(p) for p in parts]


def _extract_positional_resources(stripped: str, verb: str) -> list:
    """Return all positional resource-type tokens after ``kubectl <verb>``.

    Skips tokens starting with ``-`` (flags). For ``type/name`` slashed forms
    (e.g. ``pod/foo``) returns the ``type`` part only. Does NOT skip flag
    values — but ``_CRITICAL_RESOURCE_RE.fullmatch`` rejects hyphenated
    namespace values like ``kube-node-lease`` (where ``node`` is a substring
    but the full string does not match any critical resource name).

    Examples:
        ``kubectl delete node worker-1`` → ``["node", "worker-1"]``
        ``kubectl delete -n kube-system secret foo`` → ``["kube-system", "secret", "foo"]``
        ``kubectl delete pod/foo service/bar`` → ``["pod", "service"]``
    """
    tokens = stripped.split()
    if len(tokens) < 3:
        return []
    out = []
    for tok in tokens[2:]:  # tokens[0]="kubectl", tokens[1]=verb
        if tok.startswith("-"):
            continue
        if "/" in tok:
            tok = tok.split("/", 1)[0]
        out.append(tok)
    return out


def _is_mutation(cmd: Any) -> bool:
    """True if `cmd` is a mutating kubectl/helm command.

    Splits on shell separators (``&&``, ``||``, ``;``) and tests each
    sub-command separately, so chained mutations like
    ``kubectl get pods && kubectl delete svc my-svc`` are detected even
    when the FIRST segment is benign.

    Returns False on empty / non-string / parse error — never raises.
    """
    reason = _mutation_reason(cmd)
    return reason is not None


def _mutation_reason(cmd: Any, *, hard_block_only: bool = False) -> Optional[str]:
    """Return a short reason string if ``cmd`` should be blocked; ``None``
    if it should pass through.

    - ``hard_block_only=False`` (default): returns the FIRST reason found,
      whether hard-block OR admin-gate mutation. Used by `_is_mutation` for
      legacy behavior compatibility.
    - ``hard_block_only=True``: only check hard-block patterns A/B/C —
      used by `_wrap_one` BEFORE the admin check so hard-block wins over
      admin-gate. An admin issuing `kubectl apply -f x.yaml` gets the
      hard-block message, not the admin-gate pass-through.

    Reason strings: ``"hard_block_apply"``, ``"hard_block_delete_critical"``,
    ``"hard_block_protected_ns"``, or ``"mutation_prefix"`` (the
    admin-gate mutation match without a hard-block trigger).
    """
    if not isinstance(cmd, str) or not cmd.strip():
        return None
    try:
        segments = _extract_segments(cmd) or [_strip_env_prefix(cmd)]
    except Exception:
        return None
    if not segments:
        return None
    for stripped in segments:
        if not stripped:
            continue
        # --- Hard-block rules first (apply to admins too) -----------
        # Pattern A: any `kubectl apply ...`. Globally blocked because
        # apply mutates cluster state declaratively and the file may
        # contain any resource type — too risky to auto-approve even for
        # admins without an out-of-band confirmation.
        if _HARD_BLOCK_APPLY_RE.match(stripped):
            return "hard_block_apply"
        # Pattern B: `kubectl delete <critical_resource>` regardless of
        # namespace. Critical resource matched only on the FIRST positional
        # token after `delete` (or the prefix of `type/name` form) — never on
        # flag values like `--namespace=kube-node-lease` that contain
        # critical-resource words.
        if _HARD_BLOCK_DELETE_RE.match(stripped):
            rtypes = _extract_positional_resources(stripped, verb="delete")
            if rtypes and any(_CRITICAL_RESOURCE_RE.fullmatch(t) for t in rtypes):
                return "hard_block_delete_critical"
        # Pattern C: `kubectl <mutating-verb> ... -n <protected-ns>`.
        # The mutating verbs are the ones that touch cluster state
        # across namespaces: delete/edit/scale/patch/replace/drain.
        # `apply` already handled by Pattern A.
        if _HARD_BLOCK_MUTATING_VERBS_RE.match(stripped):
            ns_match = _NS_FLAG_RE.search(stripped)
            if ns_match:
                ns_value = ns_match.group(1).strip()
                if _PROTECTED_NS_NAME_RE.fullmatch(ns_value):
                    return "hard_block_protected_ns"
        if hard_block_only:
            continue
        # --- Admin-gate mutation rules (skip for admins) -----------
        for pat in MUTATING_PATTERNS:
            if pat.match(stripped):
                return "mutation_prefix"
    return None


def _is_hard_blocked(cmd: Any) -> Optional[str]:
    """Convenience wrapper returning the hard-block reason string or
    ``None``. Used by `_wrap_one` to short-circuit before the admin check.
    """
    return _mutation_reason(cmd, hard_block_only=True)


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
        # Hard-block fires BEFORE admin-gate: applies to admins too. The four
        # patterns (kubectl apply anywhere, kubectl delete <critical>, and
        # any mutating verb in a protected namespace) are unrecoverable and
        # must never auto-execute on behalf of the LLM — even a senior admin
        # fat-fingering them would lose cluster state.
        hard_block_reason = _is_hard_blocked(cmd)
        if hard_block_reason is not None:
            logger.warning(
                "admin_gate_hard_blocked",
                command_preview=str(cmd)[:200],
                reason=hard_block_reason,
            )
            return _build_hard_block_result(cmd, hard_block_reason)

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


_HARD_BLOCK_MESSAGES = {
    "hard_block_apply": (
        "kubectl apply is globally blocked by the bot's safety policy — "
        "declarative apply can mutate any resource type, which is too "
        "risky to auto-execute. Do NOT retry this command or `apply` "
        "variants. If you need to apply a manifest, instruct the user to "
        "run it via /shell manually after review."
    ),
    "hard_block_delete_critical": (
        "kubectl delete on a critical resource (node/pv/pvc/namespace/crd/"
        "clusterrole/service) is blocked by the bot's safety policy — "
        "such deletions are unrecoverable and may cause permanent data "
        "loss. Do NOT retry or attempt alternate resource aliases (no/ns/"
        "pvc/svc/etc). Pivot to read-only investigation: `kubectl get` / "
        "describe` / `logs` to gather context, then instruct the user to "
        "run the delete via /shell manually."
    ),
    "hard_block_protected_ns": (
        "kubectl <verb> in a protected namespace (kube-system / kube-public "
        "/ kube-node-lease / garden / gardener-system / extensions / "
        "garden-*) is blocked by the bot's safety policy — these namespaces "
        "hold K8s/Gardener control-plane state. Do NOT retry or target a "
        "different protected namespace. Pivot to read-only investigation: "
        "`kubectl get` / `describe` / `logs` to gather context, then "
        "instruct the user to run the change via /shell manually."
    ),
}


def _build_hard_block_result(cmd: Any, reason: str):
    """Return StructuredToolResult ERROR for a hard-blocked command. Lazy
    import (same shape as `_build_block_result`)."""
    from holmes.core.tools import StructuredToolResult, StructuredToolResultStatus

    cmd_preview = str(cmd)[:300] if cmd is not None else ""
    msg_body = _HARD_BLOCK_MESSAGES.get(reason, _HARD_BLOCK_MESSAGES["hard_block_apply"])
    return StructuredToolResult(
        status=StructuredToolResultStatus.ERROR,
        error=f"HARD-BLOCKED ({reason}): {msg_body}",
        params={"command": cmd_preview},
        invocation="bash(blocked-by-hard-policy)",
    )


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
