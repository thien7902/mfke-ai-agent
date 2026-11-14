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


# Block-observer registry. Each observer is a sync callable
# `(topic_id: str, kind: str, reason: str, cmd_preview: str) -> bool`.
# `kind` is "hard" or "soft". Returns True to signal "abort the
# investigation": observers are expected to set the `cancel_event`
# found in `request_context["_oncall_cancel_event"]` themselves, but
# we also reach into the context right here to do it.
#
# Used by JiraOncallService to count policy violations per ticket and
# abort Holmes when the bot keeps trying mutating commands after a
# first block. Sync (runs from inside the Holmes worker thread) so the
# observer must NOT await async work directly — `asyncio.run_coroutine_threadsafe`
# is the supported bridge back to the bot loop.
_BLOCK_OBSERVERS: list = []


def register_block_observer(fn) -> None:
    """Register a sync callable to fire on every admin_gate hard/soft block.

    Idempotent per-fn; registering the same bound method twice is a no-op.
    """
    if fn in _BLOCK_OBSERVERS:
        return
    _BLOCK_OBSERVERS.append(fn)


def unregister_block_observer(fn) -> None:
    try:
        _BLOCK_OBSERVERS.remove(fn)
    except ValueError:
        pass


def _dispatch_block(request_ctx: dict, kind: str, reason: str,
                     cmd_preview: str) -> None:
    """Fire all registered block observers for the given context, then
    best-effort set the cancel_event if any observer returned True.

    `request_ctx` is the Holmes ToolInvokeContext.request_context dict.
    On regular Telegram chats (no oncall investigation running) the
    observers look up by `topic_id` and ESI-no-op when unknown.
    """
    if not _BLOCK_OBSERVERS:
        return
    topic_id = request_ctx.get("topic_id")
    if topic_id is None:
        return
    abort = False
    for fn in _BLOCK_OBSERVERS:
        try:
            if fn(topic_id, kind, reason, cmd_preview):
                abort = True
        except Exception as e:
            logger.warning("block_observer_failed", error=str(e))
    if abort:
        cancel_event = request_ctx.get("_oncall_cancel_event")
        if cancel_event is not None and hasattr(cancel_event, "set"):
            try:
                cancel_event.set()
            except Exception as e:
                logger.warning("block_observer_cancel_set_failed", error=str(e))


# Kubectl / Helm global flag prefix. Matches zero-or-more flag tokens
# between `kubectl`/`helm` and the verb. Each flag is `-X` or `--xxx`,
# optionally followed by `=value` in the SAME token AND a separate value
# token when the flag takes one without `=` (e.g. `-n default` or
# `--kubeconfig /tmp/x`). Value tokens can't start with `-` (would be a
# different flag) — bounded so the loop can't swallow the verb itself.
#
# Used by MUTATING_PATTERNS and the hard-block regexes below so the LLM
# bypass `kubectl --kubeconfig=/tmp/x.yaml apply -f -` and the variant
# `kubectl -n default --kubeconfig=... apply -f -` both still trigger.
_KUBECTL_GLOBAL_FLAGS = (
    r"(?:"
    r"--?[A-Za-z][\w./=:-]*"               # the flag token (-x or --xxx[=y])
    r"(?:=\S+)?"                            # optional =value in same token
    r"(?:\s+(?!-)[^\s]+)?"                  # optional separate value token (not `-`)
    r"\s+"
    r")*"
)


# Regex patterns for mutating kubectl / helm subcommands. Anchored at start
# (re.match). env-prefix stripping is done before matching, so leading
# `KUBECONFIG=...` is already gone. Whitespace prefix is tolerated by re.match
# implicitly (it matches at position 0 only — we strip first).
#
# `kubectl_run_image` (separate Holmes tool, not bash) is OUT OF SCOPE here —
# it's disabled by default and we don't wrap it.
MUTATING_PATTERNS: tuple = (
    # kubectl mutating subcommands. Tolerate kubectl global flags (e.g.
    # `--kubeconfig=...`, `-n`) between `kubectl` and the verb — see
    # `_KUBECTL_GLOBAL_FLAGS` above.
    re.compile(
        rf"^kubectl\s+{_KUBECTL_GLOBAL_FLAGS}(?:apply|delete|edit|patch|scale|replace|rollout|exec|cp|create|label|annotate|taint|cordon|uncordon|drain|rolling-update|expose|set\s+env|set\s+image|set\s+resources|set\s+serviceaccount|set\s+selector|auth\s+reconcile|auth\s+can-i\s+--delete|taint\s+)\b"
    ),
    # helm mutating subcommands. install/upgrade/uninstall/rollback/delete
    # pull repo add/remove are the ones that change cluster or repo state.
    # Tolerate helm global flags before the verb the same way.
    re.compile(
        rf"^helm\s+{_KUBECTL_GLOBAL_FLAGS}(?:install|upgrade|uninstall|rollback|delete|repo\s+add|repo\s+remove|push|pull|cm-push|registry\s+login)\b"
    ),
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
# per user choice. `extension*` (gardener extension namespaces, not a literal
# `extensions` namespace) — matched via prefix only per user note that no
# namespace is exactly named `extensions`.
_PROTECTED_NS_NAME_RE = re.compile(
    r"\b("
    r"kube-system|kube-public|kube-node-lease|"
    r"garden|gardener-system|"
    r"garden-[\w-]+|"
    r"extension[\w-]*"
    r")\b",
    re.IGNORECASE,
)

# Match `-n NAME` or `--namespace NAME` or `--namespace=NAME`. Allow the
# namespace value to appear in the next arg (split on space) or via `=`.
_NS_FLAG_RE = re.compile(
    r"(?:-n\s+|--namespace=?)\s*([^\s]+)",
    re.IGNORECASE,
)

# Pattern A — kubectl apply (any args, any ns). Tolerate kubectl global
# flags (e.g. `--kubeconfig=...`, `-n`) between `kubectl` and the verb. The
# LLM bypass `kubectl --kubeconfig=/tmp/x.yaml apply -f -` (and the
# variant `kubectl -n default --kubeconfig=... apply ...`) must still
# trigger.
#
# `_KUBECTL_GLOBAL_FLAGS` is defined above `MUTATING_PATTERNS` and reused
# here. It matches zero-or-more flag tokens (each with optional separate
# value token) so the anchored verb at the end of the group is reachable
# even when the LLM inserts `--kubeconfig=...` or `-n default` between
# `kubectl` and the verb.
_HARD_BLOCK_APPLY_RE = re.compile(rf"^kubectl\s+{_KUBECTL_GLOBAL_FLAGS}apply\b", re.IGNORECASE)

# Pattern B — kubectl delete <critical_resource>. The verb comes right
# after `kubectl`, then resource-type positional. The critical-resource
# search is anchored to the FIRST positional token after `delete` (using
# `_extract_positional_resource`) so flag values like `--namespace=kube-node-lease`
# (which contains "node") and `--namespace=...` (which contains "namespace") do
# NOT false-positive.
_HARD_BLOCK_DELETE_RE = re.compile(rf"^kubectl\s+{_KUBECTL_GLOBAL_FLAGS}delete\b", re.IGNORECASE)

# Resource-type positional token extraction. Skips flags (tokens starting
# with `-`). Returns the first non-flag token after the verb, or "" if
# none found. Handles `/`-segregated forms like `pod/foo` and `service/bar`.
_RESOURCE_TYPE_TOKEN_RE = re.compile(r"^[A-Za-z][\w./-]*$")

# Pattern C — kubectl + mutating verb + protected namespace. Verbs are
# the destructive verbs that touch cluster state across namespaces:
# delete/edit/scale/patch/replace/drain. `apply` is handled by Pattern A
# alone since it is globally blocked.
# Pattern C — kubectl + mutating verb + protected namespace. Verbs are
# the destructive verbs that touch cluster state across namespaces:
# delete/edit/scale/patch/replace/drain plus the metadata mutators
# (annotate/label/taint) and node lifecycle verbs (cordon/uncordon). Also
# rollout/create/exec/cp/etc. can harm control-plane state. `apply` is
# handled by Pattern A alone since it is globally blocked.
_HARD_BLOCK_MUTATING_VERBS_RE = re.compile(
    rf"^kubectl\s+{_KUBECTL_GLOBAL_FLAGS}(?:delete|edit|scale|patch|replace|drain|"
    rf"annotate|label|taint|cordon|uncordon|rollout|create|exec|cp|"
    rf"set\s+env|set\s+image|set\s+resources|set\s+serviceaccount|set\s+selector|"
    rf"expose|rolling-update)\b",
    re.IGNORECASE,
)

# Pattern D — OpenStack mutating verbs on resources whose name does NOT
# contain "fke". OpenStack CLI verbs that change state: delete (destroy),
# set/unset (mutate properties), update (Heat stack update), remove (role/
# aggregate membership).
#
# User policy: only allow these verbs against resources whose name contains
# `fke` (case-insensitive substring). Any non-`fke` target is blocked —
# admins too — because the bot might be inspecting prod/customer cloud
# resources shared with other teams. Operators run mutations via /shell
# manually after review.
#
# Detection: any token in the command past `openstack <resource-type> <verb>`
# that contains "fke" (case-insensitive) marks the command as fke-targeted.
# Tokens starting with `-` are flags (their values can still be resource
# names — `--server fke-1` — those are scanned too). A bare `openstack
# server delete` with no name argument is blocked on grounds of ambiguity.
#
# Examples blocked:
#   openstack server delete prod-web-1               (no "fke")
#   openstack volume set --bootable prod-vol         (no "fke")
#   openstack stack update my-stack                 (no "fke")
# Examples allowed:
#   openstack server delete mfke-api-gw-1            ("fke" substring)
#   openstack volume set --bootable fke-data-vol    ("fke" substring)
_OPENSTACK_MUTATING_VERB_RE = re.compile(
    r"^openstack\s+(?:\S+\s+){0,3}(?:delete|set|unset|update|remove)\b",
    re.IGNORECASE,
)
_OPENSTACK_FKE_RE = re.compile(r"fke", re.IGNORECASE)


# Pattern E — pip / pipx / apt · package management is HARD-BLOCKED
# regardless of admin status. Installing a wheel from inside a chat is
# how supply-chain compromise happens on a perm-running bot, and apt
# can replace system packages (incl. kubectl / helm themselves).
# Covers: `pip`, `pip3`, `python3 -m pip`, `pipx`; apt, apt-get,
# dpkg (reconfigure/purge), snap. Directions: install/uninstall/
# upgrade/remove/purge/download. `pip list` / `pip show` are NOT
# blocked — they are read-only.
_PACKAGE_MANAGER_INSTALL_RE = re.compile(
    r"^(?:"
    r"pip3?\s+(?:install|uninstall|download|upgrade|wheel|build)\b"
    r"|python3?\s+-m\s+pip\s+(?:install|uninstall|download|upgrade|wheel|build)\b"
    r"|pipx\s+(?:install|uninstall|upgrade|inject|upgrade-all|install-all)\b"
    r"|apt(?:-get)?\s+(?:install|remove|purge|update|upgrade|dist-upgrade|full-upgrade|autoremove|hold|unhold|mark|reinstall|source|build-dep|change|modify)\b"
    r"|dpkg\s+(?:-i|--install|-r|--remove|-P|--purge|--configure|--update-avail|--clear-avail)\b"
    r"|snap\s+(?:install|remove|refresh|rollback|switch|channel)\b"
    r")",
    re.IGNORECASE,
)


# Pattern F — curl hitting the Kubernetes API server with a mutating
# HTTP method. The LLM can bypass kubectl by curl'ing the apiserver
# directly with a service-account token it found in a secret.
# Detection: `curl` followed by `-X POST/PUT/DELETE/PATCH` (case-
# insensitive) OR `--request POST/.../` OR a URL containing the
# apiserver path signature (`/api/` or `/apis/`) combined with a POST/
# PUT/DELETE/PATCH method (default GET is safe). We don't try to be
# clever about it — if BOTH a K8s API URL and a mutating method are
# present, block. Plain `curl https://kubernetes/api/v1/pods` (GET) is
# fine.
_CURL_MUTATING_METHOD_RE = re.compile(
    r"\bcurl\b.*?(?:-X\s+|--request\s+)(POST|PUT|DELETE|PATCH|MERGE|APPLY)\b",
    re.IGNORECASE,
)
_K8S_API_URL_RE = re.compile(
    # K8s API URLs come in two shapes:
    #   /api/v1/...              (core group, single version path)
    #   /apis/<group>/<vN>/...   (named group, MUST be followed by /v\d+ to
    #                            avoid matching /apis/webhook/foo which is
    #                            not a K8s group).
    r"/api/v\d+/|/apis/[A-Za-z][\w.]*/v\d+/",
    re.IGNORECASE,
)
# Method may also be in `--data` with `--request` omitted when curl is
# hit by `-d`. Match the `-X <METHOD>` flag OR `(POST|PUT|DELETE|PATCH)`
# verb at start of method-flag value. (See _CURL_MUTATING_METHOD_RE
# above.)


# Pattern G — Python one-liner / script invocation that imports a
# kubernetes client or builds a requests/httpx/aiohttp call with a
# mutating HTTP verb against a K8s API URL. This is the LLM's bypass
# when kubectl and curl are both blocked.
#
# We block on TWO signals together to keep false positives low:
#   1. The command invokes a Python interpreter with an arbitrary
#      inline script: `python3 -c "..."`. (Running a .py file the LLM
#      wrote is treated the same — `python3 path/to/x.py` is blocked
#      because Holmes could have written it first.)
#   2. The script content contains BOTH a K8s API URL pattern
#      (`/api/v1/`, `/apis/...`) AND a mutating-verb token from the
#      kubernetes-client method family (`create_`, `delete_`,
#      `patch_`, `replace_`) OR a raw `requests.post/put/delete/patch`
#      / `httpx.post/put/delete/patch`. Read-only patterns
#      (`requests.get`, `list_`) are NOT blocked.

# Recognize `python3 <script.py>` or `python3 -c "..."` or `python3 -m pkg`
# — anything starting with a python interpreter. This is broader than the
# previous version that only matched `-c`. We use it for two purposes:
# blocking K8s mutations (Pattern G — see _PYTHON_K8S_MUTATION_RE) and
# blocking any python invocation that doesn't target the outbox dir
# (Pattern H).
_PYTHON_INVOCATION_RE = re.compile(
    r"^python3?\s+",
    re.IGNORECASE,
)
# `python3 <args>` is allowed ONLY in inline `-c "..."` form. Running
# a script file — whether `python3 outbox/x.py`, `python3 /tmp/x` (no
# extension), `python3 -m http.server`, or `./run.sh`-style launcher —
# makes the script's source invisible to Pattern G (K8s mutation scan on
# `python3 -c "..."` source). The LLM could write a malicious script
# into the outbox then run it, bypassing every regex pattern. Closing
# this gap: only the inline form, where source IS in the command and
# Pattern G can scan it, passes Pattern I/H. Everything else must run
# via `/shell` after operator review.
#
# Match shape: `python3` (or `python`), optional non-`-c` flags, `-c`.
# `-c` must be a flag — not a value for another flag, not a substring
# of a path token — so we anchor on `\-c\b` after whitespace.
_PYTHON_INLINE_ONLY_RE = re.compile(
    r"^python3?\s+(?:-(?!c\b)[A-Za-z]+\s+)*-c\b",
    re.IGNORECASE,
)
# Recognize `pip <query-verb>` and `python3 -m pip <query-verb>` forms
# that are READ-ONLY and should NOT trip Pattern H (python-without-
# outbox block). The Holmes LLM evaluating this is allowed to query
# "what Python packages are available" without first being told to
# route through the outbox. Used to PRE-EMPT Pattern H — when the verb
# is install/uninstall/etc Pattern E blocks; when the verb is list/
# show/freeze/... this exception lets it pass through Pattern H.
_PYTHON_PIP_READONLY_RE = re.compile(
    r"^python3?\s+-m\s+pip\s+(?:list|show|freeze|search|index|inspect|debug|config|cache|verify|hash)\b"
    r"|^pip3?\s+(?:list|show|freeze|search|index|inspect|debug|config|cache|verify|hash)\b",
    re.IGNORECASE,
)
_PYTHON_K8S_MUTATION_RE = re.compile(
    r"(?:"
    # Branch 1 — kubernetes-client SDK method call. The verb prefix
    # plus at least one snake_case segment identifies a mutating
    # client method (create_namespaced_pod, delete_namespaced_service,
    # patch_cluster_role_binding, replace_namespaced_config_map,
    # update_namespaced_pod, ...). The SDK builds the URL from
    # kubeconfig so we DON'T also require a /api/v1/ URL here.
    r"\b(?:create|delete|patch|replace|update)_[a-z]+(?:_[a-z]+)+\b"
    r")"
    r"|(?:"
    # Branch 2 — raw HTTP mutating verb in requests/httpx/aiohttp/
    # urllib3 calls AND a K8s API URL pattern. Both signals together
    # keep false positives low (a generic `requests.post` to a
    # webhook is NOT blocked, a `requests.get` to /api/v1/pods is
    # NOT blocked).
    r"\b(?:requests|httpx|aiohttp|urllib3)\.(?:post|put|delete|patch|request)\b"
    r".*?(?:/api/v\d+/|/apis/[A-Za-z][\w.]*/v\d+/)"
    r")",
    re.IGNORECASE | re.DOTALL,
)


_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*\s+")
# Shell command separators we tolerate before the kubectl/helm mutation.
# `&&`, `||`, `;`, and single `|` (pipe). The LLM pipes manifests like
# `echo 'apiVersion: ...' | kubectl --kubeconfig=... apply -f -` to bypass
# detection — single pipe must split too. Regex alternation tries
# left-to-right, so `||` must come before `|` so it matches as a single
# separator (not `|` twice with empty middle).
_SEPARATOR_RE = re.compile(r"(?:&&|\|\||;|\|)")


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
    # Pattern G: Python script + K8s mutation is checked on the WHOLE
    # raw command, not on per-segment stripped forms. The shell
    # separator splitter is naive about quoting and would split a
    # `python3 -c "from kubernetes import client; ..."` script in
    # the middle of a quoted string — the segment that lands on
    # `python3 -c "from kubernetes import client` would lack the
    # mutation token, and the segment that lands on `client.create_...
    # body={})` would not be recognized as a Python invocation. So we
    # apply the Python pattern against the raw command up-front.
    if _PYTHON_INVOCATION_RE.match(_strip_env_prefix(cmd)) and _PYTHON_K8S_MUTATION_RE.search(cmd):
        return "hard_block_python_k8s_mutation"
    # Pattern E: package-manager install. Checked BEFORE Pattern I/H
    # because `python3 -m pip install kubernetes` is also a "python3
    # invocation that isn't `-c`" — Pattern I would catch it, but the
    # user-facing message would point at "don't run scripts" instead
    # of the more accurate "don't install packages". Read-only
    # `pip list` / `pip show` is NOT a Pattern E match and falls
    # through to the pip-readonly exception in Pattern I/H below.
    if _PACKAGE_MANAGER_INSTALL_RE.match(_strip_env_prefix(cmd)):
        return "hard_block_package_manager"
    # Pattern F: curl with a mutating HTTP method AND a K8s-shaped URL.
    # On raw command (not per-segment) so it survives `&&`-chained shell
    # commands too — both halves of `curl -X POST ... && echo done`
    # contain enough URL info for the pattern, and the splitter is
    # quote-naive like Pattern G's case.
    if _CURL_MUTATING_METHOD_RE.search(cmd) and _K8S_API_URL_RE.search(cmd):
        return "hard_block_curl_k8s_api"
    # Pattern I/H together: restrict ALL python3 invocations to the
    # inline `-c "..."` form. Pattern G only sees source that's
    # literally in the bash command; running `python3 outbox/x.py` (or
    # `/tmp/x` with no extension, or `python3 -m http.server`) hides
    # the source from Pattern G — the LLM could write a malicious
    # script into the outbox then run it, bypassing every regex. So
    # anything that ISN'T `python3 -c "..."` is hard-blocked, admins
    # included. The inline form is further gated by Pattern H (must
    # reference the outbox — only exception is read-only `pip list` /
    # `pip show` etc, which Pattern E doesn't catch but we want to
    # allow for introspection). Pattern E was already checked above
    # so install/uninstall verbs are routed to the package-manager
    # message, not the inline-only message.
    if _PYTHON_INVOCATION_RE.match(_strip_env_prefix(cmd)):
        if not _PYTHON_PIP_READONLY_RE.match(_strip_env_prefix(cmd)):
            if not _PYTHON_INLINE_ONLY_RE.match(_strip_env_prefix(cmd)):
                return "hard_block_python_file_run"
            if not _cmd_references_outbox(cmd):
                return "hard_block_python_no_outbox"
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
        # Pattern D: `openstack <resource> <delete|set|unset|update|remove>`
        # targeting a resource whose name does NOT contain "fke". The bot
        # may inspect shared prod/customer cloud resources — mutations on
        # non-fke targets are blocked regardless of admin status; operators
        # run them via /shell manually after review.
        if _OPENSTACK_MUTATING_VERB_RE.match(stripped) and not _OPENSTACK_FKE_RE.search(stripped):
            return "hard_block_openstack_non_fke"
        # Pattern E (package manager) and Pattern F (curl K8s API) are
        # checked UP-FRONT on the raw command (above the per-segment
        # loop) so the priority over Pattern H is correct. Skipping
        # them here avoids duplicate matches and keeps the loop body
        # focused on per-segment patterns (kubectl/helm/openstack).
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


def _cmd_references_outbox(cmd: str) -> bool:
    """True if ``cmd`` contains a reference to the configured outbox dir.

    Used by Pattern H to allowlist `python3 ...` calls that target the
    outbox — that's where the bot's file-output policy says deliverables
    must land. Anything else is unreviewed shell access.

    Lazy imports keep admin_gate importable without file_sender's
    transitive deps. Resolves the outbox dir at call time.

    Returns False on any error (config missing, path cannot resolve) —
    we fail CLOSED, treating "couldn't resolve outbox" as "no outbox
    reference" so a config bug never silently opens python up.
    """
    if not isinstance(cmd, str) or not cmd:
        return False
    try:
        from src.bot.utils import file_sender
        outbox = file_sender.outbox_dir()
    except Exception:
        return False
    try:
        candidates = {str(outbox), str(outbox).rstrip("/"), outbox.as_posix()}
        # Relative `outbox/` prefix — the chat_send_file tool itself
        # validates the resolved path against the outbox, so we accept
        # the LLM's relative reference here.
        try:
            candidates.add(outbox.name + "/")
        except Exception:
            pass
    except Exception:
        return False
    return any(c and c in cmd for c in candidates)


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
            request_ctx = getattr(context, "request_context", None) or {}
            _dispatch_block(request_ctx, "hard", hard_block_reason, str(cmd)[:200])
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
        request_ctx = getattr(context, "request_context", None) or {}
        _dispatch_block(request_ctx, "soft", "mutation_prefix_non_admin", str(cmd)[:120])
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
        "/ kube-node-lease / garden / gardener-system / extension* / "
        "garden-*) is blocked by the bot's safety policy — these namespaces "
        "hold K8s/Gardener control-plane state. Do NOT retry or target a "
        "different protected namespace. Pivot to read-only investigation: "
        "`kubectl get` / `describe` / `logs` to gather context, then "
        "instruct the user to run the change via /shell manually."
    ),
    "hard_block_openstack_non_fke": (
        "openstack <delete|set|unset|update|remove> on a resource whose "
        "name does NOT contain 'fke' is blocked by the bot's safety policy — "
        "the bot may inspect shared prod/customer OpenStack cloud resources "
        "and mutations on non-fke targets are not auto-approved. Do NOT "
        "retry or alternative-target a different non-fke resource. Pivot "
        "to read-only investigation: `openstack <resource> show` / `list` "
        "to gather context, then instruct the user to run the change via "
        "/shell manually. FKE-targeted resources (name contains 'fke') "
        "pass through."
    ),
    "hard_block_package_manager": (
        "Package installation / removal via pip / pipx / apt / dpkg / snap "
        "is blocked by the bot's safety policy — installing a wheel or "
        "replacing a system package from inside a chat is a supply-chain "
        "vector and risks replacing kubectl/helm themselves. Do NOT retry "
        "or attempt alternate package managers. If the user genuinely needs "
        "a Python package or apt dependency, instruct them to install it "
        "manually via /shell after review."
    ),
    "hard_block_curl_k8s_api": (
        "curl with a mutating HTTP method (POST/PUT/DELETE/PATCH/...) "
        "against a Kubernetes API URL (/api/v1/ or /apis/...) is blocked "
        "by the bot's safety policy — a direct apiserver call bypasses "
        "kubectl/helm and the admin-gate that protects them. Do NOT "
        "retry with a different method flag or a different URL shape. "
        "Pivot to read-only investigation: `kubectl get` / `describe` / "
        "`logs`, then instruct the user to apply any change via /shell "
        "manually."
    ),
    "hard_block_python_k8s_mutation": (
        "Running a Python script that creates / deletes / patches / "
        "replaces a Kubernetes resource via the kubernetes-client SDK "
        "(create_namespaced_*/delete_*/patch_*/replace_*) or via "
        "requests/httpx/aiohttp POST/PUT/DELETE/PATCH against a K8s API "
        "URL is blocked by the bot's safety policy — calling the "
        "apiserver from Python bypasses kubectl and the admin-gate that "
        "protects it. Do NOT retry with a different SDK verb or a "
        "different URL shape. Pivot to read-only investigation: "
        "`kubectl get` / `describe` / `logs`, then instruct the user "
        "to apply any change via /shell manually."
    ),
    "hard_block_python_no_outbox": (
        "Running `python3 -c \"...\"` from inside a chat is blocked "
        "by the bot's safety policy when it doesn't target the bot's "
        "configured outbox directory. Python is only permitted for "
        "producing deliverable FILES (xlsx, json, csv, txt, yaml, ...) "
        "that MUST be written under the outbox (same directory "
        "`chat_send_file` accepts). Do NOT retry with a `python3 -c` "
        "that writes outside the outbox. To produce a deliverable: "
        "write the file to the outbox dir in the inline `-c` one-liner "
        "then call `chat_send_file` with the path. For any other use "
        "(debug inspect, scope-state check, ad-hoc computation), "
        "instruct the user to run it via /shell manually."
    ),
    "hard_block_python_file_run": (
        "Running `python3 <anything>` in non-inline form from inside a "
        "chat is blocked by the bot's safety policy — only `python3 "
        "-c \"<source>\"` is allowed because the inline source IS "
        "visible in the bash command and the bot can scan it for "
        "kubernetes mutations and apiserver calls. File runs "
        "(`python3 x.py`, `python3 outbox/x` without an extension), "
        "module runs (`python3 -m http.server`), and any other form "
        "hide the source from the bot. Do NOT retry with `.py` files, "
        "do NOT write a script to the outbox and then run it, and do "
        "NOT use `python3 -m <pkg>`. To execute Python code, inline the "
        "source with `python3 -c \"...\"` and write any deliverable "
        "file straight to the outbox in that one-liner, then call "
        "`chat_send_file` with the path. For longer scripts the user "
        "must run them via /shell manually."
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
