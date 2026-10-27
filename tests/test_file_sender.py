"""Tests for src/bot/utils/file_sender — outbound file policy + delivery bridge."""
import asyncio
import importlib
import threading

import pytest

from src.bot.utils import file_sender

# `src/bot/utils/__init__.py` re-exports the `config` instance, shadowing the
# submodule of the same name — import the module explicitly to patch it.
config_module = importlib.import_module("src.bot.utils.config")


@pytest.fixture(autouse=True)
def outbox(tmp_path, monkeypatch):
    """Point the outbox at a per-test temp dir and clear the sender registration."""
    d = tmp_path / "outbox"
    d.mkdir()
    monkeypatch.setattr(config_module.config, "file_send_outbox_dir", str(d))
    monkeypatch.setattr(config_module.config, "file_send_max_bytes", 50 * 1024 * 1024)
    monkeypatch.setattr(config_module.config, "file_send_outbox_ttl_hours", 24)
    file_sender.reset_file_sender()
    yield d
    file_sender.reset_file_sender()


# ---------- path policy ----------

def test_relative_path_resolves_against_outbox(outbox):
    (outbox / "report.txt").write_text("hi")
    assert file_sender.resolve_sendable_path("report.txt") == (outbox / "report.txt").resolve()


def test_absolute_path_inside_outbox_allowed(outbox):
    f = outbox / "sub_dir.csv"
    f.write_text("a,b")
    assert file_sender.resolve_sendable_path(str(f)) == f.resolve()


def test_path_outside_outbox_rejected(tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    with pytest.raises(file_sender.SendPolicyError, match="outside the sendable directory"):
        file_sender.resolve_sendable_path(str(outside))


def test_parent_traversal_rejected(outbox, tmp_path):
    (tmp_path / "escape.txt").write_text("nope")
    with pytest.raises(file_sender.SendPolicyError, match="outside the sendable directory"):
        file_sender.resolve_sendable_path("../escape.txt")


def test_symlink_escaping_outbox_rejected(outbox, tmp_path):
    """A symlink inside the outbox must not grant access to its target."""
    secret = tmp_path / "secret.txt"
    secret.write_text("do not leak")
    link = outbox / "innocent.txt"
    link.symlink_to(secret)

    with pytest.raises(file_sender.SendPolicyError, match="outside the sendable directory"):
        file_sender.resolve_sendable_path(str(link))


def test_missing_file_rejected(outbox):
    with pytest.raises(file_sender.SendPolicyError, match="File not found"):
        file_sender.resolve_sendable_path("ghost.txt")


def test_directory_rejected(outbox):
    (outbox / "adir").mkdir()
    with pytest.raises(file_sender.SendPolicyError, match="not a regular file"):
        file_sender.resolve_sendable_path("adir")


def test_empty_file_rejected(outbox):
    (outbox / "empty.txt").write_text("")
    with pytest.raises(file_sender.SendPolicyError, match="empty"):
        file_sender.resolve_sendable_path("empty.txt")


def test_blank_path_rejected():
    with pytest.raises(file_sender.SendPolicyError, match="No file path"):
        file_sender.resolve_sendable_path("   ")


@pytest.mark.parametrize("name", [
    "kubeconfig.enc", "server.pem", "tls.key", "id_rsa", "id_ed25519.pub",
    ".env", ".env.production", "identity.p12", "store.jks",
])
def test_credential_like_names_rejected(outbox, name):
    (outbox / name).write_text("secret")
    with pytest.raises(file_sender.SendPolicyError, match="credential-like"):
        file_sender.resolve_sendable_path(name)


def test_oversize_file_rejected(outbox, monkeypatch):
    monkeypatch.setattr(config_module.config, "file_send_max_bytes", 10)
    (outbox / "big.bin").write_bytes(b"x" * 100)
    with pytest.raises(file_sender.SendPolicyError, match="send limit"):
        file_sender.resolve_sendable_path("big.bin")


def test_max_bytes_clamped_to_telegram_limit(outbox, monkeypatch):
    monkeypatch.setattr(config_module.config, "file_send_max_bytes", 0)
    assert file_sender._max_bytes() == file_sender.TELEGRAM_UPLOAD_LIMIT_BYTES
    monkeypatch.setattr(config_module.config, "file_send_max_bytes", 500 * 1024 * 1024)
    assert file_sender._max_bytes() == file_sender.TELEGRAM_UPLOAD_LIMIT_BYTES


# ---------- size formatting ----------

def test_format_size():
    assert file_sender.format_size(512) == "512 B"
    assert file_sender.format_size(2048) == "2.0 KB"
    assert file_sender.format_size(5 * 1024 * 1024) == "5.0 MB"


# ---------- outbox helpers ----------

def test_ensure_outbox_dir_creates_missing(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "outbox"
    monkeypatch.setattr(config_module.config, "file_send_outbox_dir", str(target))
    assert file_sender.ensure_outbox_dir() == target
    assert target.is_dir()


def test_list_outbox_newest_first_and_skips_symlinks(outbox):
    import os
    a = outbox / "old.txt"
    b = outbox / "new.txt"
    a.write_text("1")
    b.write_text("22")
    os.utime(a, (1000, 1000))
    os.utime(b, (2000, 2000))
    (outbox / "link.txt").symlink_to(a)

    entries = file_sender.list_outbox()
    names = [e[0].name for e in entries]
    assert names == ["new.txt", "old.txt"]
    assert entries[0][1] == 2


def test_prune_outbox_removes_stale_files(outbox, monkeypatch):
    import os
    stale = outbox / "stale.txt"
    fresh = outbox / "fresh.txt"
    stale.write_text("old")
    fresh.write_text("new")
    os.utime(stale, (1000, 1000))

    monkeypatch.setattr(config_module.config, "file_send_outbox_ttl_hours", 1)
    file_sender.prune_outbox()

    assert not stale.exists()
    assert fresh.exists()


def test_prune_outbox_ttl_zero_disables(outbox, monkeypatch):
    import os
    stale = outbox / "stale.txt"
    stale.write_text("old")
    os.utime(stale, (1000, 1000))

    monkeypatch.setattr(config_module.config, "file_send_outbox_ttl_hours", 0)
    file_sender.prune_outbox()
    assert stale.exists()


def test_list_outbox_includes_received_subdir(outbox):
    received = outbox / "received"
    received.mkdir()
    (received / "uploaded.xlsx").write_text("data")
    (outbox / "generated.txt").write_text("x")

    names = {e[0].name for e in file_sender.list_outbox()}
    assert names == {"uploaded.xlsx", "generated.txt"}


# ---------- stashing user uploads ----------

def test_stash_upload_copies_into_received(outbox, tmp_path):
    src = tmp_path / "download"
    src.write_text("hello world")

    stored = file_sender.stash_upload(src, "note.txt")

    assert stored == outbox / "received" / "note.txt"
    assert stored.read_text() == "hello world"
    # And it is now sendable, which is the whole point.
    assert file_sender.resolve_sendable_path("received/note.txt") == stored.resolve()


def test_stash_upload_strips_path_components_from_name(outbox, tmp_path):
    src = tmp_path / "download"
    src.write_text("x")
    stored = file_sender.stash_upload(src, "../../etc/passwd")
    assert stored.parent == outbox / "received"
    assert stored.name == "passwd"


def test_stash_upload_same_name_does_not_overwrite(outbox, tmp_path):
    a = tmp_path / "a"
    a.write_text("first")
    b = tmp_path / "b"
    b.write_text("second")

    first = file_sender.stash_upload(a, "report.txt")
    second = file_sender.stash_upload(b, "report.txt")

    assert first != second
    assert first.read_text() == "first"
    assert second.read_text() == "second"


@pytest.mark.parametrize("name", ["kubeconfig.enc", "id_rsa", ".env", ".hidden"])
def test_stash_upload_refuses_credential_names(outbox, tmp_path, name):
    src = tmp_path / "download"
    src.write_text("secret")
    assert file_sender.stash_upload(src, name) is None
    assert not (outbox / "received").exists()


def test_stash_upload_respects_disabled_flag(outbox, tmp_path, monkeypatch):
    monkeypatch.setattr(config_module.config, "file_send_keep_uploads", False)
    src = tmp_path / "download"
    src.write_text("x")
    assert file_sender.stash_upload(src, "note.txt") is None


def test_stash_upload_skips_empty_and_oversize(outbox, tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.write_text("")
    assert file_sender.stash_upload(empty, "empty.txt") is None

    monkeypatch.setattr(config_module.config, "file_send_max_bytes", 5)
    big = tmp_path / "big"
    big.write_text("way more than five bytes")
    assert file_sender.stash_upload(big, "big.txt") is None


def test_stash_upload_missing_source_returns_none(outbox, tmp_path):
    assert file_sender.stash_upload(tmp_path / "nope", "nope.txt") is None


# ---------- delivery bridge ----------

def test_send_without_registered_sender_raises(outbox):
    (outbox / "x.txt").write_text("data")
    with pytest.raises(file_sender.SenderNotConfiguredError):
        file_sender.send_document_sync(1, str(outbox / "x.txt"))


def test_send_sync_bridges_to_event_loop(outbox):
    """send_document_sync must schedule onto the bot loop from a worker thread."""
    sent = {}

    class FakeBot:
        async def send_document(self, **kwargs):
            sent.update(kwargs)
            return "message"

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        file_sender.configure_file_sender(FakeBot(), loop)
        target = outbox / "report.csv"
        target.write_text("a,b\n1,2\n")

        result = file_sender.send_document_sync(
            42, str(target), topic_id=7, caption="here you go",
        )
        assert result == "message"
        assert sent["chat_id"] == 42
        assert sent["message_thread_id"] == 7
        assert sent["caption"] == "here you go"
        assert sent["filename"] == "report.csv"
        assert sent["document"] == str(target)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def test_send_sync_rejects_call_from_bot_loop(outbox):
    """Calling the sync bridge on the loop thread would deadlock — refuse it."""
    (outbox / "x.txt").write_text("data")

    async def scenario():
        class FakeBot:
            async def send_document(self, **kwargs):
                return "message"

        loop = asyncio.get_running_loop()
        file_sender.configure_file_sender(FakeBot(), loop)
        with pytest.raises(RuntimeError, match="use send_document_async"):
            file_sender.send_document_sync(1, str(outbox / "x.txt"))

    asyncio.run(scenario())


def test_send_sync_reports_closed_loop(outbox):
    (outbox / "x.txt").write_text("data")
    loop = asyncio.new_event_loop()
    loop.close()
    file_sender.configure_file_sender(object(), loop)
    with pytest.raises(file_sender.SenderNotConfiguredError, match="closed"):
        file_sender.send_document_sync(1, str(outbox / "x.txt"))


def test_send_async_truncates_long_caption(outbox):
    captured = {}

    class FakeBot:
        async def send_document(self, **kwargs):
            captured.update(kwargs)

    async def scenario():
        await file_sender.send_document_async(
            FakeBot(), 1, str(outbox / "x.txt"),
            caption="z" * 5000,
        )

    asyncio.run(scenario())
    assert len(captured["caption"]) <= file_sender.CAPTION_MAX_CHARS


def test_is_configured_lifecycle():
    assert file_sender.is_configured() is False
    loop = asyncio.new_event_loop()
    try:
        file_sender.configure_file_sender(object(), loop)
        assert file_sender.is_configured() is True
    finally:
        file_sender.reset_file_sender()
        loop.close()
    assert file_sender.is_configured() is False
