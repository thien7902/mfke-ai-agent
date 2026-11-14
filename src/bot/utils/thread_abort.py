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
import inspect
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
    `key`, if any, AND force-close every cached litellm httpx.Client so
    the worker's blocked recv() syscall returns immediately.

    Returns True on successful thread injection. The litellm client
    force-close runs unconditionally — even if the worker key is gone,
    a hung litellm call on any worker benefits from the socket
    shutdown, and `close()` is a no-op on already-closed handlers.

    Why both: `PyThreadState_SetAsyncExc` only fires at the next Python
    bytecode boundary — there is NO bytecode boundary inside
    `socket.recv()`, which is a C-level blocking syscall. So the
    injected `KeyboardInterrupt` can sit pending for up to the litellm
    HTTP_TIMEOUT (default 600s) while the worker is stuck in recv().
    Force-closing the cached httpx.Client (close() shuts the socket)
    makes the syscall return immediately with ConnectionRefused, which
    httpx translates into an exception and propagates up through
    litellm → Holmes → asyncio.run_in_executor — before the pending
    KeyboardInterrupt ever needs to fire.
    """
    closed = _force_close_litellm_clients()
    if closed:
        logger.info("litellm_clients_force_closed", count=closed, key=str(key))

    with _LOCK:
        thread = _THREADS.get(key)
    if thread is None or not thread.is_alive():
        # The litellm close may still have freed the worker — return True
        # if we closed anything so the caller knows an abort signal was sent.
        return closed > 0
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
        return closed > 0
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


def _force_close_litellm_clients() -> int:
    """Close every cached httpx.Client in litellm's module-level cache.
    Returns count of clients closed (diagnostic; not load-bearing).

    httpx.Client.close() synchronously shuts underlying sockets: the
    blocked recv() in the worker thread wakes up almost immediately
    and raises ConnectionClosed -> httpx raises -> litellm surfaces ->
    Holmes unwinds. This is the ONLY mechanism that breaches the C
    syscall boundary — `PyThreadState_SetAsyncExc` won't fire until
    the next Python bytecode boundary, and there's none inside
    `socket.recv()`.

    Side effect: every concurrent litellm call in the bot loses its
    httpx.Client connection. litellm's `_get_httpx_client` lazily
    reconstructs on the next call, so subsequent chats re-establish
    TCP/TLS without impact.
    """
    closed = 0
    try:
        import litellm  # local import — keeps top-level import-time light
    except Exception as e:
        logger.warning("litellm_unavailable_for_force_close", error=str(e))
        return 0
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is None:
        return 0
    cache_dict = getattr(cache, "cache_dict", None)
    if not cache_dict:
        return 0
    # Iterate over a snapshot — close() may mutate cache (TTL eviction).
    #
    # Detection by cache_key substring ("async_") is WRONG: the sync OpenAI
    # SDK client's cache key is "hashed_api_key=...,is_async=False,..." which
    # naturally contains the substring "async_" — so a substring filter
    # accidentally skips the very client Holmes is actively using. Skip by
    # inspecting the value's close() method instead: async clients have a
    # coroutine close() (can't await from sync context); sync clients
    # (OpenAI / AzureOpenAI / HTTPHandler) have a regular function close().
    for cache_key, handler in list(cache_dict.items()):
        close = getattr(handler, "close", None)
        if close is None or not callable(close):
            # Fall back to inner .client.close (used by raw HTTPHandler wrappers).
            inner = getattr(handler, "client", None)
            close = getattr(inner, "close", None)
        if close is None or not callable(close):
            continue
        if inspect.iscoroutinefunction(close):
            continue
        try:
            close()
            closed += 1
        except Exception as e:
            logger.warning(
                "litellm_client_close_failed", cache_key=cache_key, error=str(e),
            )
    return closed
