"""Tests for src.bot.services.cluster_probe_service.

Covers probe semantics (ok / timeout / non-zero exit / unexpected output /
encrypted kubeconfig), the GFM table renderer, the send-then-delete flow,
and task-cancellation lifecycle. Uses fake bots and patched subprocesses —
no network, no real kubectl. pytest-asyncio auto mode (pytest.ini).
"""
import asyncio
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from src.bot.services import cluster_probe_service as cps
from src.bot.services.cluster_probe_service import ClusterProbeService


# ---------- _probe_one ----------

class _FakeProc:
    """Fake asyncio.subprocess.Process — records communicate/returncode."""
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int,
                 hang_communicate: bool = False):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.pid = 12345
        self._hang = hang_communicate

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(100)
        return self._stdout, self._stderr


def _patch_subprocess(fake_proc, hang: bool = False):
    async def _fake_create(*args, **kwargs):
        # If hang=True, mark the proc so communicate() hangs (the wait_for
        # around communicate is what should time out, not the create itself).
        fake_proc._hang = hang
        return fake_proc
    return patch("asyncio.create_subprocess_shell", new=_fake_create)


async def test_probe_one_happy_path():
    svc = ClusterProbeService()
    fake = _FakeProc(b"[+]ping ok\n[+]etcd ok\nready", b"", 0)
    with _patch_subprocess(fake):
        result = await svc._probe_one(Path("prod-eu"))
    assert result["ok"] is True
    assert result["name"] == "prod-eu"
    assert "ready" in result["reason"]


async def test_probe_one_timeout_kills_process_group():
    svc = ClusterProbeService()
    fake = _FakeProc(b"", b"", 0)
    killed = {"called": False}

    def _fake_killpg(pg, sig):
        killed["called"] = True

    # os.killpg / os.getpgid don't exist on Windows. The production container
    # is Linux where both exist natively; inject them for the test so the
    # service code can run cross-platform.
    sentinel = object()
    prev_killpg = getattr(cps.os, "killpg", sentinel)
    prev_getpgid = getattr(cps.os, "getpgid", sentinel)
    cps.os.killpg = _fake_killpg
    cps.os.getpgid = lambda pid: 0
    try:
        with _patch_subprocess(fake, hang=True):
            result = await svc._probe_one(Path("slow-cluster"))
    finally:
        if prev_killpg is sentinel:
            try: del cps.os.killpg
            except AttributeError: pass
        else:
            cps.os.killpg = prev_killpg
        if prev_getpgid is sentinel:
            try: del cps.os.getpgid
            except AttributeError: pass
        else:
            cps.os.getpgid = prev_getpgid

    assert result["ok"] is False
    assert "timeout" in result["reason"]
    assert killed["called"] is True, "os.killpg must be called on timeout"


async def test_probe_one_non_zero_exit():
    svc = ClusterProbeService()
    fake = _FakeProc(b"", b"connection refused", 1)
    with _patch_subprocess(fake):
        result = await svc._probe_one(Path("prod-us"))
    assert result["ok"] is False
    assert result["reason"].startswith("exit 1:")
    assert "connection refused" in result["reason"]


async def test_probe_one_unexpected_output():
    svc = ClusterProbeService()
    fake = _FakeProc(b"garbage output", b"", 0)
    with _patch_subprocess(fake):
        result = await svc._probe_one(Path("weird-cluster"))
    assert result["ok"] is False
    assert result["reason"] == "unexpected output"


async def test_probe_one_encrypted_kubeconfig_cleans_up_tempfile():
    svc = ClusterProbeService()
    fake = _FakeProc(b"[+]ping ok\nready", b"", 0)
    cleanup = {"called_with": None}

    def _fake_decrypt(path):
        return "/tmp/fake-kc-123"

    def _fake_cleanup(path):
        cleanup["called_with"] = path

    with patch.object(cps, "is_encrypted", return_value=True), \
         patch.object(cps, "decrypt_to_tempfile", _fake_decrypt), \
         patch.object(cps, "cleanup_tempfile", _fake_cleanup), \
         _patch_subprocess(fake):
        result = await svc._probe_one(Path("secret.enc"))

    assert result["ok"] is True
    assert cleanup["called_with"] == "/tmp/fake-kc-123", \
        "cleanup_tempfile must be called in finally even on success"


async def test_probe_one_decrypt_failure_returns_fail():
    svc = ClusterProbeService()
    cleanup = {"called": False}

    def _fake_decrypt(path):
        return None

    def _fake_cleanup(path):
        cleanup["called"] = True

    with patch.object(cps, "is_encrypted", return_value=True), \
         patch.object(cps, "decrypt_to_tempfile", _fake_decrypt), \
         patch.object(cps, "cleanup_tempfile", _fake_cleanup):
        result = await svc._probe_one(Path("corrupt.enc"))

    assert result["ok"] is False
    assert "decrypt failed" in result["reason"]
    assert cleanup["called"] is False, "no tempfile returned → nothing to clean"


# ---------- _render ----------

def test_render_empty_results_shows_no_kubeconfigs_message():
    svc = ClusterProbeService()
    out = svc._render([])
    assert "No kubeconfigs" in out
    assert "| Cluster |" not in out


def test_render_includes_agent_down_warning():
    """Footer warning must appear in both empty and populated renders so
    a stale probe always tells the user who to contact."""
    svc = ClusterProbeService()
    empty = svc._render([])
    populated = svc._render([
        {"name": "prod", "ok": True, "reason": "ready"},
    ])
    warning = "Contact Le Minh Thien"
    assert warning in empty, "empty render must show agent-down warning"
    assert warning in populated, "populated render must show agent-down warning"
    assert "older than 1 minute" in empty
    assert "older than 1 minute" in populated


def test_render_sorts_failures_first():
    svc = ClusterProbeService()
    results = [
        {"name": "zzz-ok", "ok": True, "reason": "ready"},
        {"name": "aaa-fail", "ok": False, "reason": "timeout"},
        {"name": "mmm-fail", "ok": False, "reason": "exit 1"},
    ]
    out = svc._render(results)
    data_rows = [
        l for l in out.splitlines()
        if l.startswith("| ") and "---" not in l and "Cluster" not in l
    ]
    names_in_order = [l.split("|")[1].strip() for l in data_rows]
    assert names_in_order == ["aaa-fail", "mmm-fail", "zzz-ok"]


def test_render_escapes_pipe_in_reason():
    svc = ClusterProbeService()
    results = [{"name": "x", "ok": False, "reason": "a|b|c"}]
    out = svc._render(results)
    assert "a\\|b\\|c" in out


# ---------- _send_and_delete_old ----------

class _FakeSentMessage:
    def __init__(self, message_id: int):
        self.message_id = message_id


class _FakeBot:
    def __init__(self):
        self.deletes: List[dict] = []

    async def delete_message(self, **kwargs):
        self.deletes.append(kwargs)
        return True


def _patch_send_rich_message(message_id: int):
    async def _fake_send(*args, **kwargs):
        return [_FakeSentMessage(message_id)]
    return patch.object(cps, "send_rich_message", new=_fake_send)


async def test_send_first_call_does_not_delete():
    svc = ClusterProbeService()
    bot = _FakeBot()
    svc._bot = bot
    with _patch_send_rich_message(message_id=100):
        await svc._send_and_delete_old("markdown")
    assert svc._prev_message_id == 100
    assert bot.deletes == [], "no delete on first send (no previous)"


async def test_send_second_call_deletes_previous_message_id():
    svc = ClusterProbeService()
    svc._prev_message_id = 100
    bot = _FakeBot()
    svc._bot = bot
    with _patch_send_rich_message(message_id=200):
        await svc._send_and_delete_old("markdown")
    assert svc._prev_message_id == 200
    assert len(bot.deletes) == 1
    assert bot.deletes[0]["message_id"] == 100


async def test_send_delete_failure_does_not_raise():
    """If delete_message throws (user deleted it manually), the tick survives."""
    svc = ClusterProbeService()
    svc._prev_message_id = 100
    bot = _FakeBot()

    async def _failing_delete(**kwargs):
        raise RuntimeError("message to delete not found")

    bot.delete_message = _failing_delete
    svc._bot = bot

    with _patch_send_rich_message(message_id=200):
        await svc._send_and_delete_old("markdown")
    assert svc._prev_message_id == 200


# ---------- _recover_prev_message_id on start ----------

class _FakeChat:
    def __init__(self, pinned_message=None):
        self.pinned_message = pinned_message


class _FakePinnedMessage:
    def __init__(self, message_id: int, text: str = "🛰 Cluster Probe — ..."):
        self.message_id = message_id
        self.text = text


async def test_recover_prev_message_id_from_pinned_message():
    svc = ClusterProbeService()
    bot = _FakeBot()
    bot.get_chat = AsyncMock(return_value=_FakeChat(
        pinned_message=_FakePinnedMessage(message_id=999)
    ))
    svc._bot = bot
    mid = await svc._recover_prev_message_id()
    assert mid == 999


async def test_recover_prev_message_id_none_when_no_pin():
    svc = ClusterProbeService()
    bot = _FakeBot()
    bot.get_chat = AsyncMock(return_value=_FakeChat(pinned_message=None))
    svc._bot = bot
    assert await svc._recover_prev_message_id() is None


async def test_recover_prev_message_id_rejects_unrelated_pin():
    """A pin by an admin that isn't one of our probes must NOT be adopted —
    otherwise the next tick would delete the admin's pin."""
    svc = ClusterProbeService()
    bot = _FakeBot()
    bot.get_chat = AsyncMock(return_value=_FakeChat(
        pinned_message=_FakePinnedMessage(
            message_id=555, text="Important admin announcement"
        )
    ))
    svc._bot = bot
    assert await svc._recover_prev_message_id() is None


async def test_recover_prev_message_id_swallows_get_chat_failure():
    """If get_chat raises (chat not found, network error), start() must
    still proceed — _probe_loop runs without a recovered prev."""
    svc = ClusterProbeService()
    bot = _FakeBot()

    async def _failing_get_chat(**kwargs):
        raise RuntimeError("chat not found")

    bot.get_chat = _failing_get_chat
    svc._bot = bot
    assert await svc._recover_prev_message_id() is None


async def test_send_pins_new_message_each_tick():
    """Each tick pins the freshly sent probe so restart can recover it."""
    svc = ClusterProbeService()
    svc._prev_message_id = 100
    bot = _FakeBot()
    pinned = {"calls": []}

    async def _fake_pin(**kwargs):
        pinned["calls"].append(kwargs)

    bot.pin_chat_message = _fake_pin
    bot.delete_message = AsyncMock(return_value=True)
    svc._bot = bot

    with _patch_send_rich_message(message_id=200):
        await svc._send_and_delete_old("🛰 Cluster Probe — ...\n| Col | ... |")

    assert len(pinned["calls"]) == 1
    assert pinned["calls"][0]["message_id"] == 200
    assert pinned["calls"][0]["disable_notification"] is True


# ---------- _probe_loop lifecycle ----------

async def test_probe_loop_cancels_cleanly_during_sleep():
    svc = ClusterProbeService()
    svc._bot = _FakeBot()

    async def _fake_probe_all():
        return []

    async def _fake_send(*args, **kwargs):
        return None

    async def _cancelling_sleep(seconds):
        raise asyncio.CancelledError()

    with patch.object(svc, "_probe_all", _fake_probe_all), \
         patch.object(svc, "_send_and_delete_old", _fake_send), \
         patch.object(cps.asyncio, "sleep", _cancelling_sleep):
        task = asyncio.create_task(svc._probe_loop())
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert task.cancelled() or task.done()


async def test_probe_loop_swallows_tick_exceptions():
    """If a probe run raises (non-cancel), the loop logs and continues to
    the next tick. Verify by raising on tick 2 and cancelling on tick 3."""
    svc = ClusterProbeService()
    svc._bot = _FakeBot()

    call_count = {"probe": 0}

    async def _flaky_probe_all():
        call_count["probe"] += 1
        if call_count["probe"] == 2:
            raise RuntimeError("transient kubectl error")
        if call_count["probe"] == 3:
            raise asyncio.CancelledError()
        return []

    async def _fake_send(*args, **kwargs):
        return None

    async def _cancelling_sleep(seconds):
        # Don't sleep at all — let the loop tick immediately.
        return

    with patch.object(svc, "_probe_all", _flaky_probe_all), \
         patch.object(svc, "_send_and_delete_old", _fake_send), \
         patch.object(cps.asyncio, "sleep", _cancelling_sleep):
        task = asyncio.create_task(svc._probe_loop())
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Tick 1 (ok) → tick 2 (RuntimeError swallowed) → tick 3 (CancelledError
    # propagated). Reaching count==3 proves the loop survived the failure.
    assert call_count["probe"] == 3
