"""Background probe of every kubeconfig in cluster_probe_kubeconfig_dir.

Each minute: decrypt-if-needed, run `kubectl get --raw=/readyz` with 5s
timeout per cluster (parallel via asyncio.gather), render a GFM table, send
a fresh message at the bottom of the forum's "general" topic, then delete
the previous probe message. Runs as a single asyncio task cancelled on bot
shutdown.

Send-then-delete (not delete-then-send) keeps the previous probe visible
until the new one is confirmed delivered — no gap if the send fails.
"""
import asyncio
import os
import shlex
import time
from pathlib import Path
from typing import List, Optional

import structlog

from src.bot.utils.config import config
from src.bot.utils.kubeconfig_crypto import (
    cleanup_tempfile,
    decrypt_to_tempfile,
    is_encrypted,
)
from src.bot.utils.rich_message import send_rich_message

logger = structlog.get_logger(__name__)

PROBE_INTERVAL_SECONDS = 60
PROBE_TIMEOUT_SECONDS = 5


class ClusterProbeService:
    """Per-minute probe of every kubeconfig in `cluster_probe_kubeconfig_dir`."""

    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._bot = None
        # message_id of the previous probe — deleted once the new one is sent.
        self._prev_message_id: Optional[int] = None

    async def start(self, bot) -> None:
        self._bot = bot
        # Recover the previous probe's message_id from the forum topic's
        # pinned message so a bot restart doesn't orphan the last probe.
        # If there's no pinned message (or it's not from us), the first
        # tick just sends fresh without deleting — same as the no-prior case.
        self._prev_message_id = await self._recover_prev_message_id()
        self._task = asyncio.create_task(self._probe_loop())
        logger.info(
            "cluster_probe_started",
            dir=config.cluster_probe_kubeconfig_dir,
            chat_id=config.cluster_probe_chat_id,
            topic_id=config.cluster_probe_topic_id,
            recovered_prev_message_id=self._prev_message_id,
        )

    async def _recover_prev_message_id(self) -> Optional[int]:
        """Read the forum chat's pinned message — that's where we left the
        last probe before restart. Returns its message_id, or None on any
        failure (no pin, wrong chat, network error — treat all as "no prior").
        """
        try:
            chat = await self._bot.get_chat(chat_id=config.cluster_probe_chat_id)
            pinned = getattr(chat, "pinned_message", None)
            if pinned is None:
                return None
            # Sanity: only adopt the pin if the message text looks like one
            # of our probes — otherwise it's an unrelated pin by an admin.
            text = getattr(pinned, "text", None) or ""
            if "Cluster Probe" not in text:
                return None
            return getattr(pinned, "message_id", None)
        except Exception as e:
            logger.warning("cluster_probe_recover_prev_failed", error=str(e))
            return None

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("cluster_probe_stop_error", error=str(e))
        self._task = None

    async def _probe_loop(self) -> None:
        # First tick runs immediately — the message appears at start, not 60s in.
        while True:
            try:
                results = await self._probe_all()
                markdown = self._render(results)
                await self._send_and_delete_old(markdown)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("cluster_probe_tick_failed", error=str(e))
            await asyncio.sleep(PROBE_INTERVAL_SECONDS)

    async def _probe_all(self) -> List[dict]:
        kc_dir = Path(config.cluster_probe_kubeconfig_dir)
        if not kc_dir.is_dir():
            return [{"name": "(dir missing)", "ok": False,
                     "reason": f"{kc_dir} not found"}]
        files = sorted(
            p for p in kc_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
        )
        if not files:
            return []
        return await asyncio.gather(*[self._probe_one(p) for p in files])

    async def _probe_one(self, path: Path) -> dict:
        tmp_kc: Optional[str] = None
        try:
            if is_encrypted(str(path)):
                tmp_kc = decrypt_to_tempfile(str(path))
                if tmp_kc is None:
                    return {"name": path.name, "ok": False,
                            "reason": "decrypt failed"}
                kc_env = tmp_kc
            else:
                kc_env = str(path)
            cmd = f"KUBECONFIG={shlex.quote(kc_env)} kubectl get --raw=/readyz"
            proc = await asyncio.create_subprocess_shell(
                cmd,
                executable="/bin/bash",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=PROBE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                # Kill the whole process group so any kubectl child (curl under
                # the hood) doesn't keep the stdout pipe open. Same pattern as
                # /kshell timeout handling (command_handler.py:587).
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except (ProcessLookupError, PermissionError):
                    pass
                return {"name": path.name, "ok": False,
                        "reason": f"timeout after {PROBE_TIMEOUT_SECONDS}s"}
            out = stdout.decode(errors="replace").strip() if stdout else ""
            err = stderr.decode(errors="replace").strip() if stderr else ""
            if proc.returncode != 0:
                return {"name": path.name, "ok": False,
                        "reason": f"exit {proc.returncode}: {err[:80]}"}
            if not any(tok in out.lower() for tok in ("ok", "ready")):
                return {"name": path.name, "ok": False,
                        "reason": "unexpected output"}
            last_line = out.splitlines()[-1] if out else "ready"
            return {"name": path.name, "ok": True, "reason": last_line[:80]}
        except Exception as e:
            return {"name": path.name, "ok": False,
                    "reason": f"{type(e).__name__}: {str(e)[:80]}"}
        finally:
            if tmp_kc:
                try:
                    cleanup_tempfile(tmp_kc)
                except Exception:
                    pass

    def _render(self, results: List[dict]) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        if not results:
            return (
                f"🛰 Cluster Probe — {ts}\n\n"
                f"_No kubeconfigs in {config.cluster_probe_kubeconfig_dir}._\n\n"
                f"_If this message is older than 1 minute, the agent is down. "
                f"Contact Le Minh Thien._"
            )
        rows = ["| Cluster | Status | Reason |", "| --- | --- | --- |"]
        for r in sorted(results, key=lambda x: (x["ok"], x["name"])):
            status = "✅ ok" if r["ok"] else "❌ fail"
            reason = r["reason"].replace("|", "\\|").replace("\n", " ")[:120]
            rows.append(f"| {r['name']} | {status} | {reason} |")
        return (
            f"🛰 Cluster Probe — {ts}\n\n"
            + "\n".join(rows)
            + "\n\nUpdated every 60s."
            + "\n_If this message is older than 1 minute, the agent is down. "
            "Contact Le Minh Thien._"
        )

    async def _send_and_delete_old(self, markdown: str) -> None:
        # Send fresh message FIRST so there's always one visible (no gap).
        sent = await send_rich_message(
            self._bot,
            chat_id=config.cluster_probe_chat_id,
            text=markdown,
            message_thread_id=config.cluster_probe_topic_id,
            prefer_rich=False,
        )
        new_message_id: Optional[int] = None
        if sent:
            first = sent[0] if isinstance(sent, list) else sent
            new_message_id = getattr(first, "message_id", None)

        # Pin the new probe so restart can recover it via get_chat. Pinning
        # is silent (disable_notification=True) — purely a state-marshalling
        # side-channel, not a user-visible alert.
        if new_message_id is not None:
            try:
                await self._bot.pin_chat_message(
                    chat_id=config.cluster_probe_chat_id,
                    message_id=new_message_id,
                    disable_notification=True,
                )
            except Exception as e:
                logger.debug("cluster_probe_pin_failed", error=str(e))

        # Delete the previous probe now that the new one is up.
        if self._prev_message_id is not None:
            try:
                await self._bot.delete_message(
                    chat_id=config.cluster_probe_chat_id,
                    message_id=self._prev_message_id,
                )
            except Exception as e:
                # User may have deleted it manually, or it's already gone.
                logger.debug("cluster_probe_delete_old_failed", error=str(e))
        self._prev_message_id = new_message_id
