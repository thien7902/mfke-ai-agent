"""Tests for src.bot.services.cluster_probe_service.

Covers probe semantics (ok / timeout / non-zero exit / unexpected output /
encrypted kubeconfig), the GFM table renderer, the send-then-delete flow,
and task-cancellation lifecycle. Uses fake bots and patched subprocesses —
no network, no real kubectl. pytest-asyncio auto mode (pytest.ini).
"""
import asyncio
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

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
    html, _plain = svc._render([])
    assert "No kubeconfigs" in html


def test_render_includes_agent_down_warning():
    """Footer warning must appear in both empty and populated renders so
    a stale probe always tells the user who to contact."""
    svc = ClusterProbeService()
    empty_html, _ = svc._render([])
    populated_html, _ = svc._render([
        {"name": "prod", "ok": True, "reason": "ready"},
    ])
    warning = "Contact Le Minh Thien"
    assert warning in empty_html
    assert warning in populated_html
    assert "older than 1 minute" in empty_html
    assert "older than 1 minute" in populated_html


def test_render_html_uses_pre_block():
    """Populated render wraps status table in <pre> for monospace alignment."""
    svc = ClusterProbeService()
    html, _ = svc._render([
        {"name": "prod", "ok": True, "reason": "ready"},
    ])
    assert html.startswith("<pre>")
    assert html.endswith("</pre>")


def test_render_html_escapes_lt_gt_in_reason():
    """kubectl errors with `<` or `>` (rare) must be HTML-escaped so they
    don't break Telegram's HTML parsing."""
    svc = ClusterProbeService()
    html, _ = svc._render([
        {"name": "bad", "ok": False, "reason": "got <tag>"},
    ])
    assert "&lt;tag&gt;" in html


def test_render_uses_emoji_status_icons():
    """✅ for ok, ❌ for fail — visually obvious status."""
    svc = ClusterProbeService()
    html, _ = svc._render([
        {"name": "good", "ok": True, "reason": "ready"},
        {"name": "bad", "ok": False, "reason": "timeout"},
    ])
    assert "✅" in html
    assert "❌" in html


def test_render_sorts_failures_first():
    svc = ClusterProbeService()
    results = [
        {"name": "zzz-ok", "ok": True, "reason": "ready"},
        {"name": "aaa-fail", "ok": False, "reason": "timeout"},
        {"name": "mmm-fail", "ok": False, "reason": "exit 1"},
    ]
    html, _ = svc._render(results)
    body_lines = [
        l for l in html.splitlines()
        if l.startswith(("✅", "❌"))
    ]
    assert len(body_lines) == 3
    assert body_lines[0].startswith("❌") and "aaa-fail" in body_lines[0]
    assert body_lines[1].startswith("❌") and "mmm-fail" in body_lines[1]
    assert body_lines[2].startswith("✅") and "zzz-ok" in body_lines[2]


def test_render_aligns_columns_by_padding_names():
    """Short names padded so the REASON column lines up in <pre> monospace."""
    svc = ClusterProbeService()
    html, _ = svc._render([
        {"name": "short", "ok": True, "reason": "ready"},
        {"name": "very-long-cluster-name", "ok": True, "reason": "ready"},
    ])
    body_lines = [l for l in html.splitlines() if l.startswith("✅")]
    assert len(body_lines) == 2
    # "ready" appears at the same column on both lines.
    pos0 = body_lines[0].find(" ready")
    pos1 = body_lines[1].find(" ready")
    assert pos0 == pos1, f"reason column misaligned: short={pos0} long={pos1}"


# ---------- _send_and_delete_old ----------

class _FakeSentMessage:
    def __init__(self, message_id: int):
        self.message_id = message_id


class _FakeBot:
    def __init__(self):
        self.deletes: List[dict] = []
        self.sends: List[dict] = []
        self._send_id_seq = 100

    async def send_message(self, **kwargs):
        self.sends.append(kwargs)
        msg = _FakeSentMessage(self._send_id_seq)
        self._send_id_seq += 100
        return msg

    async def delete_message(self, **kwargs):
        self.deletes.append(kwargs)
        return True


def _make_bot_with_send(message_id: int):
    bot = _FakeBot()
    async def _send(**kwargs):
        bot.sends.append(kwargs)
        return _FakeSentMessage(message_id)
    bot.send_message = _send
    return bot


async def test_send_first_call_does_not_delete():
    svc = ClusterProbeService()
    bot = _make_bot_with_send(message_id=100)
    svc._bot = bot
    await svc._send_and_delete_old("text")
    assert svc._prev_message_id == 100
    assert bot.deletes == [], "no delete on first send (no previous)"


async def test_send_second_call_deletes_previous_message_id():
    svc = ClusterProbeService()
    svc._prev_message_id = 100
    bot = _make_bot_with_send(message_id=200)
    svc._bot = bot
    await svc._send_and_delete_old("text")
    assert svc._prev_message_id == 200
    assert len(bot.deletes) == 1
    assert bot.deletes[0]["message_id"] == 100


async def test_send_delete_failure_does_not_raise():
    """If delete_message throws (user deleted it manually), the tick survives."""
    svc = ClusterProbeService()
    svc._prev_message_id = 100
    bot = _make_bot_with_send(message_id=200)

    async def _failing_delete(**kwargs):
        raise RuntimeError("message to delete not found")

    bot.delete_message = _failing_delete
    svc._bot = bot
    await svc._send_and_delete_old("text")
    assert svc._prev_message_id == 200


async def test_send_send_failure_does_not_raise_or_set_prev():
    """If send_message throws (rate-limit, chat not found), tick must survive
    and must NOT update prev_message_id (otherwise the next tick would
    delete a Wrong/None message)."""
    svc = ClusterProbeService()
    svc._prev_message_id = 100
    bot = _FakeBot()

    async def _failing_send(**kwargs):
        raise RuntimeError("timed out")

    bot.send_message = _failing_send
    svc._bot = bot
    # Suppress persistence so it doesn't pollute other tests.
    with patch.object(svc, "_persist_prev_message_id"):
        await svc._send_and_delete_old("text")
    # Prev message_id unchanged — we did NOT receive a new message.
    assert svc._prev_message_id == 100
    # Did not delete (because send failed first).
    assert bot.deletes == []


async def test_send_passes_thread_id_when_set():
    """When CLUSTER_PROBE_TOPIC_ID != 0, send_message includes message_thread_id."""
    svc = ClusterProbeService()
    bot = _make_bot_with_send(message_id=100)
    svc._bot = bot
    with patch.object(cps.config, "cluster_probe_topic_id", 42):
        await svc._send_and_delete_old("<pre>x</pre>")
    assert bot.sends[0]["message_thread_id"] == 42
    assert bot.sends[0]["parse_mode"] == "HTML"


async def test_send_omits_thread_id_when_zero():
    """When CLUSTER_PROBE_TOPIC_ID == 0 (general topic), send_message omits
    message_thread_id so Telegram routes to General topic."""
    svc = ClusterProbeService()
    bot = _make_bot_with_send(message_id=100)
    svc._bot = bot
    with patch.object(cps.config, "cluster_probe_topic_id", 0):
        await svc._send_and_delete_old("<pre>x</pre>")
    assert "message_thread_id" not in bot.sends[0]


async def test_send_html_failure_retries_as_plain_text():
    """If send_message with parse_mode=HTML throws (Telegram rejected HTML),
    retry without parse_mode, stripping <pre>/<i> tags from the body."""
    svc = ClusterProbeService()
    bot = _FakeBot()
    sends = {"calls": []}

    async def _flaking_send(**kwargs):
        sends["calls"].append(kwargs)
        if len(sends["calls"]) == 1:
            raise RuntimeError("can't parse entities")
        return _FakeSentMessage(200)

    bot.send_message = _flaking_send
    svc._bot = bot
    await svc._send_and_delete_old("<pre>hello <i>world</i></pre>")
    # Second call: no parse_mode, tags stripped from text.
    assert len(sends["calls"]) == 2
    assert "parse_mode" not in sends["calls"][1]
    assert "<pre>" not in sends["calls"][1]["text"]
    assert "<i>" not in sends["calls"][1]["text"]
    assert "hello world" in sends["calls"][1]["text"]
    assert svc._prev_message_id == 200


# ---------- _recover_prev_message_id on start (MongoDB-backed) ----------


async def test_recover_prev_message_id_from_mongodb():
    svc = ClusterProbeService()
    fake_col = MagicMock()
    fake_col.find_one = MagicMock(return_value={"_id": "prev_message_id",
                                                "message_id": 999})
    with patch.object(cps.MongoDB, "get_collection", return_value=fake_col):
        mid = svc._recover_prev_message_id()
    assert mid == 999


async def test_recover_prev_message_id_none_when_no_doc():
    svc = ClusterProbeService()
    fake_col = MagicMock()
    fake_col.find_one = MagicMock(return_value=None)
    with patch.object(cps.MongoDB, "get_collection", return_value=fake_col):
        assert svc._recover_prev_message_id() is None


async def test_recover_prev_message_id_none_when_field_missing():
    svc = ClusterProbeService()
    fake_col = MagicMock()
    fake_col.find_one = MagicMock(return_value={"_id": "prev_message_id"})
    with patch.object(cps.MongoDB, "get_collection", return_value=fake_col):
        assert svc._recover_prev_message_id() is None


async def test_recover_prev_message_id_swallows_mongo_failure():
    svc = ClusterProbeService()
    fake_col = MagicMock()
    fake_col.find_one = MagicMock(side_effect=RuntimeError("mongo down"))
    with patch.object(cps.MongoDB, "get_collection", return_value=fake_col):
        assert svc._recover_prev_message_id() is None


async def test_persist_prev_message_id_upserts_state_doc():
    svc = ClusterProbeService()
    fake_col = MagicMock()
    fake_col.update_one = MagicMock()
    with patch.object(cps.MongoDB, "get_collection", return_value=fake_col):
        svc._persist_prev_message_id(123)
    assert fake_col.update_one.called
    call_args = fake_col.update_one.call_args
    assert call_args.args[0] == {"_id": cps.STATE_DOC_ID}
    assert call_args.args[1]["$set"]["message_id"] == 123
    assert call_args.kwargs.get("upsert") is True


async def test_persist_prev_message_id_swallows_mongo_failure():
    svc = ClusterProbeService()
    fake_col = MagicMock()
    fake_col.update_one = MagicMock(side_effect=RuntimeError("mongo down"))
    with patch.object(cps.MongoDB, "get_collection", return_value=fake_col):
        # Must not raise.
        svc._persist_prev_message_id(123)


async def test_send_persists_new_message_id_to_mongodb():
    """Each tick persists new message_id so a crash mid-send still lets the
    next process delete the orphan."""
    svc = ClusterProbeService()
    svc._prev_message_id = 100
    bot = _make_bot_with_send(message_id=200)
    svc._bot = bot
    fake_col = MagicMock()
    fake_col.update_one = MagicMock()
    with patch.object(cps.MongoDB, "get_collection", return_value=fake_col):
        await svc._send_and_delete_old("text")
    assert svc._prev_message_id == 200
    # Persisted with new id (not old).
    set_doc = fake_col.update_one.call_args.args[1]["$set"]
    assert set_doc["message_id"] == 200


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
