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
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [])  # not admin
    monkeypatch.setattr(admin_gate, "_ADMIN_LOOP", None)  # slow path unavailable
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "99999"})
    r = tool.invoke({"command": "kubectl apply -f x.yaml"}, ctx)
    assert r.status.value == "error"
    assert "Admin access required" in r.error
    assert "Pivot now to read-only" in r.error
    # Underlying invoke NOT called
    assert tool.last_params is None


def test_admin_mutation_pass_through_with_user_approved(monkeypatch):
    monkeypatch.setattr(admin_gate.config, "admin_user_ids", [12345])
    tool = FakeBashTool()
    ex = FakeExecutor([FakeToolset([tool])])
    admin_gate.wrap_tools_with_admin_gate(ex)

    ctx = FakeCtx(request_context={"user_id": "12345"}, user_approved=False)
    r = tool.invoke({"command": "kubectl apply -f x.yaml"}, ctx)
    assert r.status.value == "success"
    # Underlying invoke was called
    assert tool.last_params == {"command": "kubectl apply -f x.yaml"}
    # Wrapper set user_approved=True so Holmes' bash tool skips validation
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
