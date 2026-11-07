"""Tests for src.bot.tools.admin_gate.

Covers mutation-prefix detection, env-prefix stripping, the sync admin-check
fast and slow paths, and the bash-tool wrapper (admin vs non-admin branch).
Lazy StructuredToolResult import is used (same pattern as test_loop_guard) so
the pure-logic tests run without holmes installed.
"""
import asyncio
import importlib

import pytest

from src.bot.tools import admin_gate


# ---------- _strip_env_prefix ----------

def test_strip_env_prefix_single():
    assert admin_gate._strip_env_prefix("KUBECONFIG=/tmp/x kubectl apply -f f.yaml") == "kubectl apply -f f.yaml"


def test_strip_env_prefix_multiple():
    assert admin_gate._strip_env_prefix("A=1 B=2 kubectl delete pod foo") == "kubectl delete pod foo"


def test_strip_env_prefix_whitespace_only():
    assert admin_gate._strip_env_prefix("   kubectl get pods   ") == "kubectl get pods"


def test_strip_env_prefix_empty():
    assert admin_gate._strip_env_prefix("") == ""
    assert admin_gate._strip_env_prefix(None) == ""


# ---------- _is_mutation ----------

@pytest.mark.parametrize("cmd", [
    "kubectl apply -f foo.yaml",
    "kubectl delete pod foo",
    "kubectl edit deployment api",
    "kubectl patch svc x -p '{...}'",
    "kubectl scale deploy/x --replicas=4",
    "kubectl replace -f x.yaml --force",
    "kubectl rollout undo deployment/api",
    "kubectl exec pod-x -- ls",
    "kubectl cp foo bar",
    "kubectl create namespace x",
    "kubectl label pod foo x=1",
    "kubectl annotate pod foo k=v",
    "kubectl taint node x k=v:NoSchedule",
    "kubectl cordon node-1",
    "kubectl uncordon node-1",
    "kubectl drain node-1",
    "helm install my-release ./chart",
    "helm upgrade my-release ./chart",
    "helm uninstall my-release",
    "helm rollback my-release 1",
    "helm delete my-release",
    "helm repo add stable https://...",
    "helm repo remove stable",
])
def test_is_mutation_true(cmd):
    assert admin_gate._is_mutation(cmd) is True


@pytest.mark.parametrize("cmd", [
    "kubectl get pods",
    "kubectl describe pod foo",
    "kubectl logs api-pod",
    "kubectl top nodes",
    "kubectl explain pod.spec",
    "kubectl api-resources",
    "kubectl config view",
    "kubectl cluster-info",
    "kubectl auth can-i list pods",
    "helm list",
    "helm get manifest my-release",
    "helm get values my-release",
    "helm search repo nginx",
    "helm status my-release",
    "echo hello",
    "ls -la",
    "",
    None,
    "   ",
])
def test_is_mutation_false(cmd):
    assert admin_gate._is_mutation(cmd) is False


def test_is_mutation_env_prefix_tolerated():
    assert admin_gate._is_mutation("KUBECONFIG=/tmp/x kubectl delete pod foo") is True
    assert admin_gate._is_mutation("ENV=prod A=1 helm upgrade r ./chart") is True


def test_is_mutation_leading_whitespace_tolerated():
    assert admin_gate._is_mutation("   kubectl scale deploy/x --replicas=2") is True


def test_is_mutation_separator_amp_amp():
    """`export KUBECONFIG=... && kubectl delete ...` must be detected."""
    assert admin_gate._is_mutation("export KUBECONFIG=/tmp/x && kubectl delete pod foo") is True
    assert admin_gate._is_mutation("kubectl get pods && kubectl delete svc my-svc") is True


def test_is_mutation_separator_semicolon():
    """Semicolon-separated mutations are detected even if the first
    segment is benign."""
    assert admin_gate._is_mutation("kubectl get pods; kubectl delete pod foo") is True


def test_is_mutation_separator_pipe_pipe():
    assert admin_gate._is_mutation("kubectl get pods || kubectl delete pod foo") is True


def test_is_mutation_multiple_separators_with_env():
    """Multiple env prefixes + separators stacked — each segment is
    stripped before pattern matching, so detection still fires."""
    cmd = (
        "export KUBECONFIG=/tmp/x && cd /tmp && kubectl delete node worker-1"
    )
    assert admin_gate._is_mutation(cmd) is True


def test_is_mutation_benign_chained_segments_not_flagged():
    """All segments benign → no false positive."""
    assert admin_gate._is_mutation("kubectl get pods && kubectl get svc && kubectl get nodes") is False
    assert admin_gate._is_mutation("echo hi; ls -la") is False


# ---------- is_admin_sync fast path ----------

def test_is_admin_sync_none_returns_false():
    assert admin_gate.is_admin_sync(None) is False


def test_is_admin_sync_non_numeric_returns_false():
    assert admin_gate.is_admin_sync("not-a-number") is False


def test_is_admin_sync_fast_path_in_admin_list(monkeypatch):
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    # Loop is not set — fast path should not need it.
    admin_gate._ADMIN_LOOP = None
    assert admin_gate.is_admin_sync("12345") is True


def test_is_admin_sync_fast_path_excluded_returns_false_without_loop(monkeypatch):
    """Non-admin id without a loop returns False (slow path unavailable)."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [1, 2, 3])
    admin_gate._ADMIN_LOOP = None
    assert admin_gate.is_admin_sync("99999") is False


# ---------- is_admin_sync slow path ----------

def test_is_admin_sync_slow_path_returns_true(monkeypatch):
    """Non-admin id with a running loop + permissive _is_admin returns True."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [])

    async def fake_is_admin(tid):
        return True

    monkeypatch.setattr(admin_gate, "_async_is_admin", fake_is_admin)

    loop = asyncio.new_event_loop()
    import threading
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        admin_gate.set_admin_check_loop(loop)
        assert admin_gate.is_admin_sync("99999") is True
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)
        loop.close()


def test_is_admin_sync_slow_path_timeout_returns_false(monkeypatch):
    """Slow path that exceeds timeout returns False (safer default)."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [])
    monkeypatch.setattr(admin_gate, "_ADMIN_LOOKUP_TIMEOUT_S", 0.2)

    async def slow_is_admin(tid):
        await asyncio.sleep(2)
        return True

    monkeypatch.setattr(admin_gate, "_async_is_admin", slow_is_admin)

    loop = asyncio.new_event_loop()
    import threading
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        admin_gate.set_admin_check_loop(loop)
        assert admin_gate.is_admin_sync("99999") is False
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)
        loop.close()


def test_is_admin_sync_slow_path_exception_returns_false(monkeypatch):
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [])

    async def exploding_is_admin(tid):
        raise RuntimeError("db down")

    monkeypatch.setattr(admin_gate, "_async_is_admin", exploding_is_admin)

    loop = asyncio.new_event_loop()
    import threading
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        admin_gate.set_admin_check_loop(loop)
        assert admin_gate.is_admin_sync("99999") is False
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)
        loop.close()


# ---------- wrap_tools_with_admin_gate ----------

class FakeStatus:
    def __init__(self, value):
        self.value = value


class FakeResult:
    def __init__(self, status_value="success", data="ok", error=None):
        self.status = FakeStatus(status_value)
        self.data = data
        self.error = error


class FakeBashTool:
    name = "bash"

    def __init__(self):
        self.invoke = self._real_invoke
        self.last_params = None
        self.last_context = None

    def _real_invoke(self, params, context=None):
        self.last_params = params
        self.last_context = context
        return FakeResult("success", "ran")


class FakeNonBashTool:
    name = "kubectl_run_image"

    def __init__(self):
        self.invoke = self._real_invoke

    def _real_invoke(self, params, context=None):
        return FakeResult("success", "ran")


class FakeToolset:
    def __init__(self, tools):
        self.tools = tools


class FakeExecutor:
    def __init__(self, toolsets):
        self.toolsets = toolsets


class FakeCtx:
    def __init__(self, request_context=None, user_approved=False):
        self.request_context = request_context or {}
        self.user_approved = user_approved


def test_wrap_skips_non_bash_tools():
    ex = FakeExecutor([FakeToolset([FakeNonBashTool()])])
    admin_gate.wrap_tools_with_admin_gate(ex)
    tool = ex.toolsets[0].tools[0]
    assert not getattr(tool, "_admin_gate_wrapped", False)


def test_wrap_marks_bash_wrapped():
    ex = FakeExecutor([FakeToolset([FakeBashTool()])])
    admin_gate.wrap_tools_with_admin_gate(ex)
    assert ex.toolsets[0].tools[0]._admin_gate_wrapped is True


def test_wrap_idempotent():
    ex = FakeExecutor([FakeToolset([FakeBashTool()])])
    admin_gate.wrap_tools_with_admin_gate(ex)
    first = ex.toolsets[0].tools[0].invoke
    admin_gate.wrap_tools_with_admin_gate(ex)
    second = ex.toolsets[0].tools[0].invoke
    assert first is second


def test_non_admin_mutation_blocked(monkeypatch):
    # `kubectl create` is a mutation but NOT a hard-block pattern — admin-gate
    # path. (kubectl apply, delete node, etc. would also fire hard-block, which
    # is covered by separate tests below.)
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [])  # not admin
    monkeypatch.setattr(admin_gate, "_ADMIN_LOOP", None)  # slow path unavailable
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "99999"})
    r = tool.invoke({"command": "kubectl create namespace test"}, ctx)
    assert r.status.value == "error"
    assert "Admin access required" in r.error
    assert "Pivot now to read-only" in r.error
    # Underlying invoke NOT called
    assert tool.last_params is None


def test_admin_mutation_pass_through_with_user_approved(monkeypatch):
    # `kubectl create` is admin-gate-mutation but NOT hard-block — admin can
    # still pass through.
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"}, user_approved=False)
    r = tool.invoke({"command": "kubectl create namespace test"}, ctx)
    assert r.status.value == "success"
    assert tool.last_params == {"command": "kubectl create namespace test"}
    assert tool.last_context.user_approved is True


def test_non_admin_non_mutation_passes_through(monkeypatch):
    """Read-only commands from non-admins are NOT blocked — investigation stays open."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [])
    monkeypatch.setattr(admin_gate, "_ADMIN_LOOP", None)
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "99999"})
    r = tool.invoke({"command": "kubectl get pods"}, ctx)
    assert r.status.value == "success"
    assert tool.last_params == {"command": "kubectl get pods"}


def test_admin_non_mutation_preserves_user_approved_state(monkeypatch):
    """Read-only commands from admins don't flip user_approved — only mutations do."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"}, user_approved=False)
    tool.invoke({"command": "kubectl get pods"}, ctx)
    assert tool.last_context.user_approved is False  # unchanged for non-mutations


def test_block_result_is_structured_tool_result(monkeypatch):
    """Sanity: StructuredToolResult import path works (lazy import)."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [])
    monkeypatch.setattr(admin_gate, "_ADMIN_LOOP", None)
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "99999"})
    r = tool.invoke({"command": "kubectl delete pod foo"}, ctx)
    from holmes.core.tools import StructuredToolResult
    assert isinstance(r, StructuredToolResult)


def test_no_user_id_blocks_mutation(monkeypatch):
    """When request_context has no user_id (e.g. CLI path), mutations still blocked."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [])
    monkeypatch.setattr(admin_gate, "_ADMIN_LOOP", None)
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={})
    r = tool.invoke({"command": "kubectl apply -f x.yaml"}, ctx)
    assert r.status.value == "error"


# ---------- Hard-block rules (Pattern A: kubectl apply) ----------

@pytest.mark.parametrize("cmd", [
    "kubectl apply -f x.yaml",
    "kubectl apply -f -",
    "kubectl apply -k dir/",
    "kubectl apply --prune -f x.yaml",
    "kubectl apply -n default -f x.yaml",
    "kubectl apply -n kube-system -f x.yaml",  # ns-irrelevant — apply globally blocked
    "KUBECONFIG=/tmp/x kubectl apply -f y.yaml",
    "kubectl get pods && kubectl apply -f z.yaml",  # blocked via second segment
])
def test_hard_block_apply_blocks_all(monkeypatch, cmd):
    """Pattern A: kubectl apply (any args, any ns) is globally hard-blocked —
    admins and non-admins both blocked."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"})
    r = tool.invoke({"command": cmd}, ctx)
    assert r.status.value == "error"
    assert "HARD-BLOCKED" in r.error
    assert "hard_block_apply" in r.error
    assert tool.last_params is None  # underlying invoke NOT called


def test_hard_block_apply_beats_admin_gate(monkeypatch):
    """Hard-block fires BEFORE admin check — admin issuing apply still blocked."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"}, user_approved=False)
    r = tool.invoke({"command": "kubectl apply -f critical.yaml"}, ctx)
    assert r.status.value == "error"
    assert "hard_block_apply" in r.error
    # user_approved MUST NOT be flipped to True for hard-blocked commands
    assert ctx.user_approved is False


# ---------- Hard-block rules (Pattern B: kubectl delete critical) ----------

@pytest.mark.parametrize("cmd", [
    "kubectl delete node worker-1",
    "kubectl delete nodes worker-1",
    "kubectl delete no worker-1",
    "kubectl delete pv my-pv",
    "kubectl delete persistentvolume my-pv",
    "kubectl delete persistentvolumes my-pv",
    "kubectl delete pvc my-pvc",
    "kubectl delete persistentvolumeclaim my-pvc",
    "kubectl delete persistentvolumeclaims my-pvc",
    "kubectl delete namespace my-ns",
    "kubectl delete namespaces my-ns",
    "kubectl delete ns my-ns",
    "kubectl delete crd my-crd",
    "kubectl delete customresourcedefinition my-crd",
    "kubectl delete customresourcedefinitions my-crd",
    "kubectl delete clusterrole my-role",
    "kubectl delete clusterroles my-role",
    "kubectl delete service my-svc",
    "kubectl delete services my-svc",
    "kubectl delete svc my-svc",
    "kubectl delete svc/my-svc",
    "kubectl delete pod/foo service/bar",  # critical resource anywhere in cmd
    "kubectl delete -n some-ns service my-svc",  # any ns
])
def test_hard_block_delete_critical_blocks(monkeypatch, cmd):
    """Pattern B: kubectl delete <critical_resource> blocked regardless of ns."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"})
    r = tool.invoke({"command": cmd}, ctx)
    assert r.status.value == "error"
    assert "hard_block_delete_critical" in r.error
    assert tool.last_params is None


@pytest.mark.parametrize("cmd", [
    "kubectl delete pod foo",           # pod NOT critical — recoverable
    "kubectl delete deploy foo",
    "kubectl delete deployment foo",
    "kubectl delete sts foo",
    "kubectl delete configmap foo",
    "kubectl delete secret foo",
    "kubectl delete job foo",
    "kubectl delete cronjob foo",
    "kubectl delete rs foo",
    "kubectl delete ds foo",
])
def test_hard_block_delete_non_critical_passes_through_to_admin_gate(monkeypatch, cmd):
    """Non-critical deletes go through the admin-gate (mutation_prefix) path,
    NOT the hard-block path — admins still pass through."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"}, user_approved=False)
    r = tool.invoke({"command": cmd}, ctx)
    # Admin path → success
    assert r.status.value == "success"
    assert ctx.user_approved is True


# ---------- Hard-block rules (Pattern C: mutating verb in protected ns) ----------

@pytest.mark.parametrize("verb", ["delete", "edit", "scale", "patch", "replace", "drain"])
@pytest.mark.parametrize("ns_flag", ["-n kube-system", "--namespace=kube-system",
                                      "--namespace kube-public", "-n kube-node-lease",
                                      "-n garden", "-n gardener-system", "-n extensions",
                                      "-n garden-foo", "-n garden-some-project"])
def test_hard_block_mutating_verb_in_protected_ns(monkeypatch, verb, ns_flag):
    """Pattern C: kubectl <mutating-verb> -n <protected-ns> blocked for admins too."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    cmd = f"kubectl {verb} {ns_flag} some-resource"
    ctx = FakeCtx(request_context={"user_id": "12345"})
    r = tool.invoke({"command": cmd}, ctx)
    assert r.status.value == "error"
    assert "hard_block_protected_ns" in r.error
    assert tool.last_params is None


@pytest.mark.parametrize("cmd", [
    # shoot-- namespaces deliberately NOT blocked (per user choice)
    "kubectl delete -n shoot--proj--name pod foo",
    "kubectl edit -n shoot--proj--name deploy bar",
    # Non-protected ns unaffected — admin-gate path
    "kubectl delete -n default pod foo",
    "kubectl edit -n my-ns deploy bar",
    "kubectl patch -n some-app svc x",
    # Read-only verbs in protected ns still pass through
    "kubectl get -n kube-system pods",
    "kubectl describe -n kube-system pod foo",
    "kubectl logs -n garden pod-x",
    "kubectl top -n kube-public pods",
])
def test_hard_block_protected_ns_not_triggered(monkeypatch, cmd):
    """shoot--*, default, and read-only verbs in protected ns are not hard-blocked."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"}, user_approved=False)
    r = tool.invoke({"command": cmd}, ctx)
    # Should NOT hard-block — passes through (admin may pass through for
    # mutating verbs in unprotected ns; read-only passes through for any user).
    assert "hard_block_protected_ns" not in (r.error or "")


def test_hard_block_protected_ns_with_env_prefix(monkeypatch):
    """env-prefix-stripped command still triggers hard-block."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    cmd = "KUBECONFIG=/tmp/x kubectl delete -n kube-system configmap foo"
    ctx = FakeCtx(request_context={"user_id": "12345"})
    r = tool.invoke({"command": cmd}, ctx)
    assert r.status.value == "error"
    assert "hard_block_protected_ns" in r.error


def test_hard_block_chained_with_separator_triggers(monkeypatch):
    """Mutating verb in protected ns via && still triggers hard-block."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    cmd = "kubectl get pods && kubectl delete -n kube-system secret foo"
    ctx = FakeCtx(request_context={"user_id": "12345"})
    r = tool.invoke({"command": cmd}, ctx)
    assert r.status.value == "error"
    # Hard-block wins; the first benign segment doesn't matter
    assert "hard_block_protected_ns" in r.error or "hard_block_delete_critical" in r.error


# ---------- Hard-block rules (Pattern D: openstack non-fke mutations) ----------

@pytest.mark.parametrize("cmd", [
    # delete operations on non-fke resources
    "openstack server delete prod-web-1",
    "openstack volume delete prod-vol-1",
    "openstack network delete prod-net",
    "openstack subnet delete prod-subnet",
    "openstack stack delete my-stack",
    "openstack image delete prod-img",
    "openstack router delete prod-rtr",
    "openstack port delete prod-port",
    # set/unset — property mutations
    "openstack volume set --bootable prod-vol",
    "openstack server set --property foo=bar prod-web-1",
    "openstack server unset --property foo prod-web-1",
    # update (e.g. Heat stack update)
    "openstack stack update my-stack -t template.yaml",
    # remove (role/aggregate membership)
    "openstack role remove --user alice --project prod Member",
    # positional name flag forms
    "openstack server delete --server prod-web-1",
    # case-insensitive: verb in caps, name in caps
    "OPENSTACK SERVER DELETE PROD-WEB-1",
    # env-prefix tolerated
    "OS_AUTH_URL=https://x openstack server delete prod-web-1",
    # bare verb with no name token — blocked on ambiguity (no fke anywhere)
    "openstack server delete",
])
def test_hard_block_openstack_non_fke_blocks(monkeypatch, cmd):
    """Pattern D: openstack <verb> with no fke token in command blocked."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"})
    r = tool.invoke({"command": cmd}, ctx)
    assert r.status.value == "error"
    assert "hard_block_openstack_non_fke" in r.error
    assert tool.last_params is None


@pytest.mark.parametrize("cmd", [
    # fke in positional name → allowed
    "openstack server delete mfke-api-gw-1",
    "openstack volume delete fke-data-vol",
    "openstack stack update fke-stack -t t.yaml",
    "openstack network delete fke-net",
    # fke as substring of a larger name → still allowed (case-insensitive)
    "openstack server delete MFKE-PROD-DB-1",
    "openstack volume delete somethinfke-thing",
    # fke in flag value (--server fke-x) → allowed
    "openstack server delete --server fke-web-1",
    # fke in a flag value with =
    "openstack server delete --server=fke-web-1",
])
def test_hard_block_openstack_fke_passes_through(monkeypatch, cmd):
    """fke-named openstack resources pass through (admin-gate may still
    block non-admins, but the hard-block D pattern does not fire)."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"}, user_approved=False)
    r = tool.invoke({"command": cmd}, ctx)
    # Not hard-blocked — admin path passes through
    assert "hard_block_openstack_non_fke" not in (r.error or "")


@pytest.mark.parametrize("cmd", [
    # read-only verbs — not mutations
    "openstack server show prod-web-1",
    "openstack server list",
    "openstack volume list --all-projects",
    "openstack network show prod-net",
    "openstack stack list",
    "openstack image list",
    # mutating verbs against fke resources — allowed (no fke check fails)
    "openstack server delete fke-x",
])
def test_hard_block_openstack_read_only_not_affected(monkeypatch, cmd):
    """Read-only openstack commands do not trigger any hard-block."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"}, user_approved=False)
    r = tool.invoke({"command": cmd}, ctx)
    assert "hard_block_openstack_non_fke" not in (r.error or "")


def test_hard_block_openstack_non_fke_beats_admin(monkeypatch):
    """Hard-block D fires BEFORE admin check — admin issuing openstack
    server delete prod-web-1 still blocked."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"}, user_approved=False)
    r = tool.invoke({"command": "openstack server delete prod-web-1"}, ctx)
    assert r.status.value == "error"
    assert "hard_block_openstack_non_fke" in r.error
    # user_approved MUST NOT be flipped
    assert ctx.user_approved is False


def test_hard_block_openstack_chained_with_separator_triggers(monkeypatch):
    """Mutating openstack verb via && still triggers hard-block when no
    fke token in any segment that has the verb."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    cmd = "openstack server list && openstack volume delete prod-vol"
    ctx = FakeCtx(request_context={"user_id": "12345"})
    r = tool.invoke({"command": cmd}, ctx)
    assert r.status.value == "error"
    assert "hard_block_openstack_non_fke" in r.error


def test_hard_block_openstack_chained_with_fke_in_first_segment_does_not_save(monkeypatch):
    """A fke token in the FIRST (benign) segment does not save a non-fke
    mutation in the SECOND segment — each segment is scanned independently."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    cmd = "openstack server show fke-x && openstack server delete prod-web"
    ctx = FakeCtx(request_context={"user_id": "12345"})
    r = tool.invoke({"command": cmd}, ctx)
    assert r.status.value == "error"
    assert "hard_block_openstack_non_fke" in r.error


# ---------- Extension* namespace fix ----------

@pytest.mark.parametrize("cmd", [
    "kubectl delete -n extensions pod foo",         # exact "extensions" no longer blocked
    "kubectl delete -n extension pod foo",
    "kubectl delete -n extensions-foo pod foo",
    "kubectl delete -n extension-gardener-foo pod foo",
    "kubectl edit -n extensions svc bar",
    "kubectl scale -n extension deploy/x --replicas=2",
])
def test_extension_namespace_pattern_blocks(monkeypatch, cmd):
    """Post-fix: `extension*` (prefix match) blocks; pure `extensions`
    alone is NOT a hard-block anymore — only `extension[\\w-]*` matches,
    so `extensions` and `extensions-foo` both match as prefix forms.
    However `extension` alone also matches."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"})
    r = tool.invoke({"command": cmd}, ctx)
    assert r.status.value == "error"
    assert "hard_block_protected_ns" in r.error


def test_extension_namespace_not_false_positive_on_unrelated_ns(monkeypatch):
    """Namespaces like `my-extension-data` would still match (prefix). User
    decided extension* is protected — that's the policy. Verify a name
    that does NOT start with `extension` does not false-positive."""
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    # unrelated namespace → not blocked (admin pass-through succeeds).
    ctx = FakeCtx(request_context={"user_id": "12345"}, user_approved=False)
    r = tool.invoke({"command": "kubectl delete -n my-ns pod foo"}, ctx)
    assert "hard_block_protected_ns" not in (r.error or "")
