"""Outbound file delivery to Telegram (bot → user).

Two callers use this module:

* `SendFileTool` — a Holmes tool. Holmes invokes tools synchronously on a
  worker thread, but `Bot.send_document` is a coroutine that must run on the
  bot's own event loop. `send_document_sync` bridges the two with
  `asyncio.run_coroutine_threadsafe`, using the (bot, loop) pair registered by
  `configure_file_sender()` at startup.
* `/sendfile` — an async command handler, which calls `send_document_async`
  directly.

**Send policy.** Only files under `config.file_send_outbox_dir` may be sent.
The bot runs as root inside the container, next to the encrypted kubeconfigs
in /root/.holmes and the API keys in the environment; an unrestricted "send
this path" tool would let the LLM — or a log line that prompt-injects it —
exfiltrate those to the chat. The outbox is the model's workspace: it writes
the deliverable there (bash, python, write_file, …) and then asks for it to be
sent. Symlinks are resolved before the containment check, so a link inside the
outbox cannot point outside it.

Known trade-off: one outbox is shared by every topic in every chat the bot
serves, so a user in topic A could ask for a file another topic generated. The
bot already runs as one shared assistant inside one group, so this adds no
access a user does not already have via the shell-tool-capable LLM, but it is
worth remembering if the bot is ever handed to mutually-untrusted groups.

Path lookups go through the `src.bot.utils.config` *module* rather than a
bound `config` object, so `patch("src.bot.utils.config.config")` in tests
reaches this module too. The indirection is needed because
`src/bot/utils/__init__.py` re-exports the `config` instance, which shadows
the submodule of the same name — `from src.bot.utils import config` yields the
Config instance, not the module. Hence `importlib`.
"""
import asyncio
import concurrent.futures
import importlib
import shutil
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import structlog

_config_module = importlib.import_module("src.bot.utils.config")

logger = structlog.get_logger(__name__)

# Telegram's bot API caps bot uploads at 50 MB unless a local Bot API server is
# used. `config.file_send_max_bytes` can lower this, never raise it.
TELEGRAM_UPLOAD_LIMIT_BYTES = 50 * 1024 * 1024

# Telegram caption limit.
CAPTION_MAX_CHARS = 1024

# How long to wait for Telegram to accept the upload. Generous: a 50 MB
# document over a slow link can take minutes, and the Holmes worker thread
# blocking here is cheaper than a silently dropped deliverable.
SEND_TIMEOUT_SECONDS = 180

# Read/write timeouts for the underlying HTTP request. The Application-level
# default is 60s, which is not enough for large uploads.
_UPLOAD_HTTP_TIMEOUT_SECONDS = 180

# Filenames refused even inside the outbox. No legitimate deliverable needs
# these, and refusing them means a model that copies a secret into its
# workspace ("cat the kubeconfig into report.txt" aside — that one defeats any
# name check) cannot then hand the usual credential file straight back.
DENIED_NAMES = frozenset({".env"})
DENIED_SUFFIXES = (".enc", ".pem", ".key", ".p12", ".pfx", ".jks")
DENIED_PREFIXES = ("id_rsa", "id_ed25519", "kubeconfig", ".env.")


class SendPolicyError(Exception):
    """The requested path may not be sent (outside the outbox, missing, ...)."""


class SenderNotConfiguredError(Exception):
    """No Telegram bot/loop has been registered yet, or it is gone."""


# (bot, loop) registered at startup. Tests call configure_file_sender() with
# fakes; production calls it from TelegramBot.initialize().
_sender: Tuple[Any, Any] = (None, None)


def configure_file_sender(bot: Any, loop: asyncio.AbstractEventLoop) -> None:
    """Register the live bot and its event loop. Called once at startup."""
    global _sender
    _sender = (bot, loop)
    logger.info("file_sender_configured", loop=id(loop))


def reset_file_sender() -> None:
    """Forget the registered bot/loop (tests, or shutdown)."""
    global _sender
    _sender = (None, None)


def is_configured() -> bool:
    bot, loop = _sender
    return bot is not None and loop is not None and not loop.is_closed()


def outbox_dir() -> Path:
    """Configured outbox directory (not created — see `ensure_outbox_dir`)."""
    return Path(_config_module.config.file_send_outbox_dir).expanduser()


def ensure_outbox_dir() -> Path:
    d = outbox_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _max_bytes() -> int:
    configured = int(getattr(_config_module.config, "file_send_max_bytes", 0) or 0)
    if configured <= 0:
        return TELEGRAM_UPLOAD_LIMIT_BYTES
    return min(configured, TELEGRAM_UPLOAD_LIMIT_BYTES)


def format_size(num_bytes: int) -> str:
    """Human-readable size for tool output and command replies."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _check_name(path: Path) -> None:
    name = path.name.lower()
    if name in DENIED_NAMES or name.startswith(DENIED_PREFIXES) or name.endswith(DENIED_SUFFIXES):
        raise SendPolicyError(
            f"Refusing to send '{path.name}': credential-like filenames are never sent."
        )


def resolve_sendable_path(raw_path: str) -> Path:
    """Validate `raw_path` against the outbox policy and return the real path.

    Relative paths are resolved against the outbox (so the model can pass
    "report.xlsx"). Raises `SendPolicyError` with an LLM-actionable message.
    """
    if not raw_path or not str(raw_path).strip():
        raise SendPolicyError("No file path given.")

    candidate = Path(str(raw_path).strip()).expanduser()
    outbox = ensure_outbox_dir()
    if not candidate.is_absolute():
        candidate = outbox / candidate

    try:
        resolved = candidate.resolve(strict=False)
        outbox_resolved = outbox.resolve(strict=False)
    except OSError as e:
        raise SendPolicyError(f"Could not resolve path '{raw_path}': {e}")

    # Resolve first, compare second: a symlink in the outbox pointing at
    # /root/.holmes would otherwise pass a naive startswith() check.
    if not resolved.is_relative_to(outbox_resolved):
        raise SendPolicyError(
            f"'{raw_path}' is outside the sendable directory. Files must be written "
            f"to {outbox_resolved} before they can be sent."
        )

    _check_name(resolved)

    if not resolved.exists():
        raise SendPolicyError(
            f"File not found: {resolved}. Write it into {outbox_resolved} first."
        )
    if not resolved.is_file():
        raise SendPolicyError(f"'{resolved}' is not a regular file.")

    size = resolved.stat().st_size
    limit = _max_bytes()
    if size > limit:
        raise SendPolicyError(
            f"File is {format_size(size)}, over the {format_size(limit)} send limit."
        )
    if size == 0:
        raise SendPolicyError(f"'{resolved.name}' is empty — nothing to send.")

    return resolved


def prune_outbox() -> None:
    """Best-effort removal of stale deliverables so the outbox stays bounded."""
    ttl_hours = int(getattr(_config_module.config, "file_send_outbox_ttl_hours", 24) or 0)
    if ttl_hours <= 0:
        return
    cutoff = time.time() - ttl_hours * 3600
    try:
        for entry in _iter_outbox_files(ensure_outbox_dir()):
            try:
                if entry.stat().st_mtime < cutoff:
                    entry.unlink()
            except OSError:
                pass
        # Drop the received/ dir once emptied so the outbox listing stays tidy.
        received = outbox_dir() / RECEIVED_DIRNAME
        if received.is_dir() and not any(received.iterdir()):
            received.rmdir()
    except OSError as e:
        logger.warning("outbox_prune_failed", error=str(e))


async def send_document_async(
    bot: Any,
    chat_id: int,
    path: str,
    *,
    topic_id: Optional[int] = 0,
    caption: Optional[str] = None,
    filename: Optional[str] = None,
):
    """Send `path` as a Telegram document. Caller has already validated it."""
    p = Path(str(path))
    if caption and len(caption) > CAPTION_MAX_CHARS:
        caption = caption[: CAPTION_MAX_CHARS - 1] + "…"
    return await bot.send_document(
        chat_id=chat_id,
        document=str(p),
        filename=filename or p.name,
        caption=caption or None,
        message_thread_id=topic_id or None,
        read_timeout=_UPLOAD_HTTP_TIMEOUT_SECONDS,
        write_timeout=_UPLOAD_HTTP_TIMEOUT_SECONDS,
        connect_timeout=30,
    )


def send_document_sync(
    chat_id: int,
    path: str,
    *,
    topic_id: Optional[int] = 0,
    caption: Optional[str] = None,
    filename: Optional[str] = None,
):
    """Blocking wrapper for sync callers (Holmes tool invocations).

    Must NOT be called from the bot's event-loop thread — `future.result()`
    would deadlock waiting on the loop that is blocked right here.
    """
    bot, loop = _sender
    if bot is None or loop is None:
        raise SenderNotConfiguredError(
            "Telegram sender is not configured; the bot cannot deliver files right now."
        )
    if loop.is_closed():
        raise SenderNotConfiguredError("The bot's event loop is closed; cannot deliver files.")

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        raise RuntimeError(
            "send_document_sync called on the bot event loop thread; "
            "use send_document_async instead."
        )

    future = asyncio.run_coroutine_threadsafe(
        send_document_async(
            bot, chat_id, path,
            topic_id=topic_id, caption=caption, filename=filename,
        ),
        loop,
    )
    try:
        return future.result(timeout=SEND_TIMEOUT_SECONDS)
    except (TimeoutError, concurrent.futures.TimeoutError):
        future.cancel()
        raise TimeoutError(
            f"Telegram did not accept the file within {SEND_TIMEOUT_SECONDS}s."
        )


def list_outbox() -> list:
    """(path, size, modified) for every file in the outbox, newest first."""
    entries = []
    try:
        for entry in _iter_outbox_files(ensure_outbox_dir()):
            try:
                st = entry.stat()
            except OSError:
                continue
            entries.append((entry, st.st_size, st.st_mtime))
    except OSError as e:
        logger.warning("outbox_list_failed", error=str(e))
        return []
    entries.sort(key=lambda item: item[2], reverse=True)
    return entries


def _iter_outbox_files(root: Path):
    """Yield regular, non-symlink files in the outbox (one level into subdirs)."""
    for entry in root.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            for nested in entry.iterdir():
                if nested.is_file() and not nested.is_symlink():
                    yield nested
        elif entry.is_file() and not entry.is_symlink():
            yield entry


# ---------- uploaded files kept for re-sending ----------

RECEIVED_DIRNAME = "received"


def stash_upload(source_path, original_name: str) -> Optional[Path]:
    """Copy a just-downloaded user upload into the outbox so it can be re-sent.

    Telegram hands us a file once; `handle_file` deletes its temp copy after
    extraction, so without this an uploaded xlsx could never be delivered back
    ("send me that file again"). Returns the stored path, or None when the
    upload is not stashed (disabled, credential-shaped name, over the cap, or
    a copy error) — callers treat None as "not re-sendable" and continue.

    Stored under `<outbox>/received/`, which `resolve_sendable_path` accepts,
    so `chat_send_file` can send it back later in the conversation.
    """
    if not getattr(_config_module.config, "file_send_keep_uploads", True):
        return None

    name = Path(original_name or "upload").name  # strip any path components
    if not name or name.startswith("."):
        return None
    try:
        _check_name(Path(name))
    except SendPolicyError:
        # Never persist credential-shaped uploads — the at-rest copy would
        # outlive the message and become another thing on disk to protect.
        logger.info("upload_not_stashed_denied_name", filename=name)
        return None

    try:
        src = Path(str(source_path))
        size = src.stat().st_size
        if size == 0 or size > _max_bytes():
            return None

        received = ensure_outbox_dir() / RECEIVED_DIRNAME
        received.mkdir(parents=True, exist_ok=True)
        dest = received / name
        if dest.exists():
            # Same filename uploaded twice: keep both rather than overwrite, so
            # a re-send request always refers to the file the user just sent.
            dest = received / f"{Path(name).stem}-{int(time.time())}{Path(name).suffix}"
        shutil.copyfile(src, dest)
        logger.info("upload_stashed", path=str(dest), size=size, original=original_name)
        return dest
    except OSError as e:
        logger.warning("upload_stash_failed", filename=original_name, error=str(e))
        return None
