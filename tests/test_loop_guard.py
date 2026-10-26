"""Tests for src.bot.tools.loop_guard.

Covers normalization, threshold behavior, success-only tracking, skip-list,
and idempotent wrapping. Uses a lightweight fake tool/exec to avoid importing
holmes (loop_guard lazily imports StructuredToolResult only on block).
"""
import importlib
import types

from src.bot.tools import loop_guard


class FakeStatus:
    def __init__(self, value):
        self.value = value


class FakeResult:
    def __init__(self, status_value="success", data="ok"):
        self.status = FakeStatus(status_value)
        self.data = data


class FakeTool:
    def __init__(self, name):
        self.name = name
        self.invoke = self._real_invoke

    def _real_invoke(self, params, context=None):
        return FakeResult("success", f"ran {self.name} with {params}")


class FakeToolset:
    def __init__(self, tools):
        self.tools = tools


class FakeExecutor:
    def __init__(self, toolsets):
        self.toolsets = toolsets


def _make_executor_with(*tool_names):
    tools = [FakeTool(n) for n in tool_names]
    return FakeExecutor([FakeToolset(tools)])


# ---------- normalization ----------

def test_normalize_strips_whitespace_and_lowercases():
    s1 = loop_guard._normalize_params({"q": "  Hello   World  "})
    s2 = loop_guard._normalize_params({"q": "hello world"})
    assert s1 == s2


def test_normalize_key_order_independent():
    a = loop_guard._normalize_params({"a": 1, "b": 2})
    b = loop_guard._normalize_params({"b": 2, "a": 1})
    assert a == b


def test_normalize_handles_none():
    assert loop_guard._normalize_params(None) == ""


def test_make_key_includes_tool_name():
    k1 = loop_guard._make_key("bash", {"cmd": "ls"})
    k2 = loop_guard._make_key("kubectl", {"cmd": "ls"})
    assert k1 != k2


# ---------- threshold behavior ----------

def test_blocks_on_fourth_identical_success_within_window(monkeypatch):
    monkeypatch.setattr(loop_guard, "LOOP_GUARD_THRESHOLD", 4)
    monkeypatch.setattr(loop_guard, "LOOP_GUARD_WINDOW", 5)
    ex = _make_executor_with("bash")
    loop_guard.wrap_tools_with_loop_guard(ex)
    tool = ex.toolsets[0].tools[0]

    # 3 success calls allowed
    for _ in range(3):
        r = tool.invoke({"cmd": "kubectl get pods"})
        assert "ran" in r.data
    # 4th call -> block
    r = tool.invoke({"cmd": "kubectl get pods"})
    assert r.status.value == "error"
    assert "Loop guard" in r.error


def test_allows_first_three_identical_calls(monkeypatch):
    monkeypatch.setattr(loop_guard, "LOOP_GUARD_THRESHOLD", 4)
    ex = _make_executor_with("bash")
    loop_guard.wrap_tools_with_loop_guard(ex)
    tool = ex.toolsets[0].tools[0]
    for _ in range(3):
        r = tool.invoke({"cmd": "kubectl get pods"})
        assert r.status.value == "success"


def test_different_params_do_not_count_toward_threshold(monkeypatch):
    monkeypatch.setattr(loop_guard, "LOOP_GUARD_THRESHOLD", 4)
    ex = _make_executor_with("bash")
    loop_guard.wrap_tools_with_loop_guard(ex)
    tool = ex.toolsets[0].tools[0]
    tool.invoke({"cmd": "kubectl get pods"})
    tool.invoke({"cmd": "kubectl get nodes"})
    tool.invoke({"cmd": "kubectl get svc"})
    # 4th distinct call should still succeed
    r = tool.invoke({"cmd": "kubectl get deployments"})
    assert r.status.value == "success"


# ---------- success-only tracking ----------

def test_failed_calls_not_counted(monkeypatch):
    """Two successes + one failure with same params should not block retry.

    Failure on call 3 is NOT tracked, so calls 4+ should still be allowed
    until 4 SUCCESSES with same params have been recorded.
    """
    monkeypatch.setattr(loop_guard, "LOOP_GUARD_THRESHOLD", 4)
    tool = FakeTool("bash")
    real_invoke = tool._real_invoke

    call_count = [0]

    def flaky_invoke(params, context=None):
        call_count[0] += 1
        if call_count[0] == 3:
            return FakeResult("error", "transient")
        return real_invoke(params, context)

    # Wrap FIRST, then swap the underlying invoke to the flaky one via the
    # closure-captured original. We do this by replacing tool.invoke after
    # wrap and pointing at flaky_invoke — but wrap replaced tool.invoke, so
    # we patch the wrapped one's underlying original instead. Easiest path:
    # set tool.invoke = flaky_invoke BEFORE wrap, and the wrap closure calls
    # the (now flaky) original through `original_invoke` captured at wrap.
    tool.invoke = flaky_invoke
    ex = FakeExecutor([FakeToolset([tool])])
    loop_guard.wrap_tools_with_loop_guard(ex)

    r1 = tool.invoke({"cmd": "kubectl get pods"})  # success -> tracked (1)
    assert r1.status.value == "success"
    r2 = tool.invoke({"cmd": "kubectl get pods"})  # success -> tracked (2)
    assert r2.status.value == "success"
    r3 = tool.invoke({"cmd": "kubectl get pods"})  # error -> NOT tracked
    assert r3.status.value == "error"
    # After 2 successes + 1 failure, 3rd success is r4. With threshold=4
    # (block when count+1 >= 4), r5 (4th success) is the one blocked,
    # because count=[1,2,4]=3 before r5 -> 3+1=4 -> block.
    r4 = tool.invoke({"cmd": "kubectl get pods"})  # success -> tracked (3)
    assert r4.status.value == "success"
    r5 = tool.invoke({"cmd": "kubectl get pods"})  # 4th would-be success -> BLOCK
    assert r5.status.value == "error"
    assert "Loop guard" in r5.error


# ---------- window sliding ----------

def test_window_evicts_oldest(monkeypatch):
    """A burst of identical calls then distinct calls should age out the burst."""
    monkeypatch.setattr(loop_guard, "LOOP_GUARD_THRESHOLD", 4)
    monkeypatch.setattr(loop_guard, "LOOP_GUARD_WINDOW", 5)
    ex = _make_executor_with("bash")
    loop_guard.wrap_tools_with_loop_guard(ex)
    tool = ex.toolsets[0].tools[0]

    # 3 identical (would block 4th)
    for _ in range(3):
        tool.invoke({"cmd": "kubectl get pods"})
    # 5 distinct calls -> evict 3 oldest "kubectl get pods" entries
    for i in range(5):
        tool.invoke({"cmd": f"kubectl get pod pod-{i}"})
    # Now window has 5 distinct entries, none match old "kubectl get pods"
    r = tool.invoke({"cmd": "kubectl get pods"})
    assert r.status.value == "success"


# ---------- skip list ----------

def test_skip_tools_not_wrapped(monkeypatch):
    monkeypatch.setattr(loop_guard, "LOOP_GUARD_SKIP_TOOLS", frozenset({"todo_write"}))
    ex = _make_executor_with("todo_write", "bash")
    loop_guard.wrap_tools_with_loop_guard(ex)
    todo = ex.toolsets[0].tools[0]
    bash = ex.toolsets[0].tools[1]
    assert not getattr(todo, "_loop_guard_wrapped", False)
    assert getattr(bash, "_loop_guard_wrapped", False)


# ---------- idempotency ----------

def test_wrapping_twice_does_not_double_wrap():
    ex = _make_executor_with("bash")
    loop_guard.wrap_tools_with_loop_guard(ex)
    first = ex.toolsets[0].tools[0].invoke
    loop_guard.wrap_tools_with_loop_guard(ex)
    second = ex.toolsets[0].tools[0].invoke
    assert first is second


# ---------- block result ----------

def test_block_result_includes_tool_name(monkeypatch):
    monkeypatch.setattr(loop_guard, "LOOP_GUARD_THRESHOLD", 2)
    ex = _make_executor_with("kshell")
    loop_guard.wrap_tools_with_loop_guard(ex)
    tool = ex.toolsets[0].tools[0]
    tool.invoke({"cmd": "ls"})
    r = tool.invoke({"cmd": "ls"})
    assert r.status.value == "error"
    assert "kshell" in r.error


def test_block_result_pytest_can_import_holmes(monkeypatch):
    """Sanity: StructuredToolResult import path works (used at block time)."""
    # Force a block and confirm the result has expected holmes shape.
    monkeypatch.setattr(loop_guard, "LOOP_GUARD_THRESHOLD", 2)
    ex = _make_executor_with("bash")
    loop_guard.wrap_tools_with_loop_guard(ex)
    tool = ex.toolsets[0].tools[0]
    tool.invoke({"cmd": "ls"})
    r = tool.invoke({"cmd": "ls"})
    # Must be a StructuredToolResult instance
    from holmes.core.tools import StructuredToolResult
    assert isinstance(r, StructuredToolResult)
