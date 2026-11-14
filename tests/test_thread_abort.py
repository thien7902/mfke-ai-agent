"""Tests for src/bot/utils/thread_abort.py.

PyThreadState_SetAsyncExc-based force abort of worker threads. Used by
/stop to break a hung litellm/httpx.recv call when the cooperative
cancel_event never gets polled. Smoke-tests the registry shape, the
basic inject-into-busy-loop path, and the no-op-when-unknown-key branch.

Real-world httpx behavior is not unit-tested here — that requires a
live socket and is fragile in CI. The Python-bytecode interrupt path
is the same mechanism httpx ultimately takes (its recv wrapper yields
back to Python on each chunk, so the next bytecode boundary will see
the pending exception).
"""
import threading
import time

import pytest

from src.bot.utils import thread_abort as ta


def test_register_then_unregister_removes_entry():
    ta.register(("k", 1))
    assert ("k", 1) in ta._THREADS
    ta.unregister(("k", 1))
    assert ("k", 1) not in ta._THREADS


def test_unregister_unknown_key_is_noop():
    # Should not raise.
    ta.unregister(("never-registered", 99))


def test_abort_unknown_key_returns_false():
    assert ta.abort(("no-such-key", 0)) is False


def test_abort_injects_keyboardinterrupt_into_busy_loop():
    """Pure-Python busy loop: SetAsyncExc interrupts at the next bytecode
    boundary."""
    result = {}

    def busy_loop():
        ta.register("busy")
        try:
            i = 0
            while True:
                i += 1
        except KeyboardInterrupt:
            result["status"] = "aborted"
        finally:
            ta.unregister("busy")

    t = threading.Thread(target=busy_loop)
    t.start()
    time.sleep(0.1)  # let the thread get into the loop
    ok = ta.abort("busy")
    assert ok is True
    t.join(timeout=2)
    assert not t.is_alive()
    assert result.get("status") == "aborted"


def test_abort_after_thread_exits_returns_false(monkeypatch):
    """If the worker thread already finished and unregister'd itself, abort
    is a safe no-op."""
    done = threading.Event()

    def short_lived():
        ta.register("short")
        time.sleep(0.05)
        ta.unregister("short")
        done.set()

    t = threading.Thread(target=short_lived)
    t.start()
    done.wait(timeout=1)
    # Thread has unregister'd and exited; abort should be False.
    assert ta.abort("short") is False


def test_re_register_replaces_previous_entry():
    """Idempotent overwrite — a re-register doesn't accumulate stale handles."""
    ta.register("dup", threading.current_thread())
    t1 = ta._THREADS.get("dup")
    # Re-register: should overwrite, not append.
    ta.register("dup", threading.current_thread())
    t2 = ta._THREADS.get("dup")
    assert t1 is t2  # same Thread object — overwrite
    ta.unregister("dup")


def test_force_close_litellm_clients_handles_missing_cache(monkeypatch):
    """_force_close_litellm_clients is safe when litellm / cache / dict is
    absent. Returns 0, doesn't raise."""
    # Force litellm module attribute to None.
    import litellm
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", None, raising=False)
    assert ta._force_close_litellm_clients() == 0


def test_force_close_litellm_clients_closes_sync_handlers_only(monkeypatch):
    """force-close iterates in_memory_llm_clients_cache and calls close()
    on every entry whose close() is a sync (non-coroutine) function. Async
    handlers (close() is `async def`) are skipped — can't await from sync
    context.

    IMPORTANT: detection MUST be by close() signature, NOT by cache-key
    substring. The real sync OpenAI SDK client's cache key is
    "hashed_api_key=...,is_async=False,..." — it contains the substring
    "async_" from "is_async=False". A substring-based skip would
    accidentally skip the very client Holmes is using, leaving recv()
    blocked forever — that was the v2.4.6 regression.
    """
    import asyncio
    import litellm

    class _SyncHandler:
        def __init__(self, name):
            self.name = name
            self.closed = False

        def close(self):
            self.closed = True

    class _AsyncHandler:
        def __init__(self, name):
            self.name = name
            self.closed = False

        async def close(self):
            self.closed = True

    sync_h = _SyncHandler("sync_httpx")
    openai_sdk = _SyncHandler("openai-sdk-real-cache-key")
    async_h = _AsyncHandler("async_httpx")

    class _FakeCache:
        # The OpenAI SDK key naturally contains "is_async=False" — must
        # NOT be filtered out by an "async_" substring check.
        cache_dict = {
            "hashed_api_key=abc,is_async=False,timeout=600": openai_sdk,
            "httpx_clienttimeout_600.0vertex_ai": sync_h,
            "async_httpx_clientvertex_ai": async_h,  # genuinely async
        }

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", _FakeCache(), raising=False)
    closed = ta._force_close_litellm_clients()
    # Both sync handlers closed — including the OpenAI SDK one whose key
    # contains "is_async=False". Async handler skipped.
    assert closed == 2
    assert sync_h.closed
    assert openai_sdk.closed
    assert not async_h.closed  # coroutine close — skipped


def test_force_close_litellm_clients_handles_inner_client_close(monkeypatch):
    """Some cached wrappers expose .client.close (raw HTTPHandler holding
    an httpx.Client) rather than a top-level close(). Fallback path
    covers both shapes."""
    import litellm

    class _InnerClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _HTTPHandler:
        def __init__(self):
            self.client = _InnerClient()

    handler = _HTTPHandler()

    class _FakeCache:
        cache_dict = {"httpx_clienttimeout_600": handler}

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", _FakeCache(), raising=False)
    closed = ta._force_close_litellm_clients()
    assert closed == 1
    assert handler.client.closed
