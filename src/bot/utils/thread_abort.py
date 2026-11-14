"""Force-kill a worker thread running an in-flight Holmes LLM call.

`/stop` today sets a `threading.Event` and waits for Holmes'
`ToolCallingLLM` to check it between iterations. That check never runs
while `litellm.completion()` is blocked inside `httpx.Client.recv()`
on a slow / hung LLM HTTP call — the user has to wait up to the
litellm timeout (often 600s) before the event gets polled.

This module exposes a `register`/`unregister`/`abort` triple keyed by
an opaque `key` (the bot uses `(chat_id, topic_id)` tuples). When a
worker thread enters the section of code that calls Holmes, it
`register(key, threading.current_thread())`. `/stop` then has the
option to `abort(key)` — which uses `PyThreadState_SetAsyncExc` to
inject `KeyboardInterrupt` directly into the worker thread's
Python-level execution.

`PyThreadState_SetAsyncExc` is the documented CPython API for "raise
an exception in another thread from outside". httpx's `recv()`
wraps socket reads in Python, so even mid-syscall it will receive the
`KeyboardInterrupt` on the next bytecode boundary and propagate it
up. litellm surfaces it as a `litellm.exceptions.APIError` /
`Timeout` / similar, which Holmes unwinds via the regular exception
path.

This is invasive — the worker thread's stack will unwind through
Holmes, litellm, httpx, all of which need to clean up properly.
Empty `except Exception:` blocks upstream could swallow it badly;
Holmes' own `ToolCallingLLM.call` doesn't catch `KeyboardInterrupt`
so it propagates all the way back to `loop.run_in_executor`'s future
where the bot's chat() wrapper translates it to a friendly response.

There is no equivalent for the streaming path — the worker thread
running `next(stream_iter)` blocks on `recv()` similarly, so the
same injection works there.
"""
import ctypes
import threading
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# Key → threading.Thread. Threads unregister themselves on cleanup
# so a stale entry means the worker has already exited (no-op abort).
# Protected by a single module-level lock because /stop and the worker
# both touch it from different threads.
_LOCK = threading.Lock()
_THREADS: dict = {}


def register(key, thread: Optional[threading.Thread] = None) -> None:
    """Register `thread` (defaults to current thread) for `key`.

    Idempotent: a re-register replaces the previous handle. The bot
    dispatches one batch at a time per topic (per-topic asyncio lock in
    MessageHandler), so in practice the key is unique per in-flight
    request.
    """
    if thread is None:
        thread = threading.current_thread()
    with _LOCK:
        _THREADS[key] = thread


def unregister(key) -> None:
    """Remove `key` from the registry. No-op if missing."""
    with _LOCK:
        _THREADS.pop(key, None)


def abort(key) -> bool:
    """Inject `KeyboardInterrupt` into the worker thread registered for
    `key`, if any. Returns True on successful injection.

    `PyThreadState_SetAsyncExc(tid, exc_type)` raises `exc_type` in the
    thread identified by `tid`. The API leaves the exception "pending"
    on the target thread — it'll surface at the next bytecode boundary.
    Passing `None` as the second arg CLEARs a pending async exception
    (used as a safety belt after a failed raise), followed by re-trying
    the raise; this is the recommended pattern from CPython's docs.

    Multiple concurrent calls are safe in the sense that the worker
    thread will receive at most one `KeyboardInterrupt` (the second
    call's pending exc overrides the first), but /stop should be
    idempotent at the caller level anyway.
    """
    with _LOCK:
        thread = _THREADS.get(key)
    if thread is None or not thread.is_alive():
        return False
    tid = thread.ident
    if tid is None:
        return False
    try:
        # `ident` is the OS-level thread id. `PyThreadState_SetAsyncExc`
        # expects this — NOT `native_id`.
        ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = [
            ctypes.c_ulong, ctypes.py_object,
        ]
        ctypes.pythonapi.PyThreadState_SetAsyncExc.restype = ctypes.c_int
        rc = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid), KeyboardInterrupt
        )
    except Exception as e:
        logger.warning(
            "thread_abort_set_async_exc_failed",
            key=str(key), tid=tid, error=str(e),
        )
        return False
    if rc == 0:
        # Invalid thread id — thread exited concurrently. Not an error.
        return False
    if rc > 1:
        # rc > 1 means multiple thread states were affected — a sign of
        # corrupt state. The CPython docs say to clear the pending
        # exception with `PyThreadState_SetAsyncExc(tid, None)` and bail.
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid), None
            )
        except Exception:
            pass
        logger.warning(
            "thread_abort_unexpected_thread_count",
            key=str(key), tid=tid, rc=rc,
        )
        return False
    logger.info(
        "thread_abort_injected",
        key=str(key), tid=tid, name=thread.name,
    )
    return True
