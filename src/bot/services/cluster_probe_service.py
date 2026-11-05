"""Background probe of every kubeconfig in cluster_probe_kubeconfig_dir.

Each minute: decrypt-if-needed, run `kubectl get --raw=/readyz` with 5s
timeout per cluster (parallel via asyncio.gather), render a GFM table, send
a fresh message at the bottom of the forum's "general" topic, then delete
the previous probe message. Runs as a single asyncio task cancelled on bot
shutdown.

Send-then-delete (not delete-then-send) keeps the previous probe visible
until the new one is confirmed delivered — no gap if the send fails.

Previous probe's message_id is persisted to MongoDB (collection
`cluster_probe_state`) so a bot restart can delete the last probe from
the prior run instead of orphaning it in the topic forever.
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
from src.bot.utils.mongodb import MongoDB

logger = structlog.get_logger(__name__)

PROBE_INTERVAL_SECONDS = 60
PROBE_TIMEOUT_SECONDS = 5
STATE_COLLECTION = "cluster_probe_state"
STATE_DOC_ID = "prev_message_id"  # single-doc collection, _id is fixed


class ClusterProbeService:
    """Per-minute probe of every kubeconfig in `cluster_probe_kubeconfig_dir`."""

    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._bot = None
        # message_id of the previous probe — deleted once the new one is sent.
        self._prev_message_id: Optional[int] = None

    async def start(self, bot) -> None:
        self._bot = bot
        # Recover the previous probe's message_id from MongoDB so a bot
        # restart doesn't orphan the last probe in the topic. If MongoDB
        # has no record (first ever start, or mongo unavailable), treat as
        # no-prior — the first tick just sends fresh without deleting.
        self._prev_message_id = self._recover_prev_message_id()
        self._task = asyncio.create_task(self._probe_loop())
        logger.info(
            "cluster_probe_started",
            dir=config.cluster_probe_kubeconfig_dir,
            chat_id=config.cluster_probe_chat_id,
            topic_id=config.cluster_probe_topic_id,
            recovered_prev_message_id=self._prev_message_id,
        )

    def _recover_prev_message_id(self) -> Optional[int]:
        """Read the previous probe's message_id from MongoDB state collection."""
        try:
            doc = MongoDB.get_collection(STATE_COLLECTION).find_one(
                {"_id": STATE_DOC_ID}
            )
            if not doc:
                return None
            mid = doc.get("message_id")
            return int(mid) if mid is not None else None
        except Exception as e:
            logger.warning("cluster_probe_recover_prev_failed", error=str(e))
            return None

    def _persist_prev_message_id(self, message_id: Optional[int]) -> None:
        """Upsert the previous probe's message_id into MongoDB state collection."""
        try:
            MongoDB.get_collection(STATE_COLLECTION).update_one(
                {"_id": STATE_DOC_ID},
                {"$set": {"message_id": message_id,
                          "updated_at": time.time()}},
                upsert=True,
            )
        except Exception as e:
            logger.warning("cluster_probe_persist_prev_failed", error=str(e))

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
        footer = (
            "\n_If this message is older than 1 minute, the agent is down. "
            "Contact Le Minh Thien._"
        )
        if not results:
            return (
                f"🛰 Cluster Probe — {ts}\n\n"
                f"No kubeconfigs in {config.cluster_probe_kubeconfig_dir}."
                + footer
            )
        # Plain-text list view (not GFM table). Sent via sendMessage without
        # parse_mode — Telegram renders raw markdown chars literally, so we
        # avoid "*", "|", and "_" that would otherwise show as garbage if the
        # client doesn't interpret them.
        lines = [f"🛰 Cluster Probe — {ts}", ""]
        for r in sorted(results, key=lambda x: (x["ok"], x["name"])):
            status = "OK" if r["ok"] else "FAIL"
            reason = r["reason"].replace("\n", " ")[:120]
            lines.append(f"{status}  {r['name']}  —  {reason}")
        lines.append("")
        lines.append("Updated every 60s." + footer)
        return "\n".join(lines)

    async def _send_and_delete_old(self, text: str) -> None:
        # Plain-text send. No parse_mode (so Telegram renders the body
        # literally — no markdown interpretation, no HTML). Probe is short
        # and pure-ASCII enough that formatting is unnecessary; avoids the
        # sendRichMessage timeouts and HTML-tag-bleed issues.
        kwargs = {
            "chat_id": config.cluster_probe_chat_id,
            "text": text,
            "disable_notification": True,
        }
        if config.cluster_probe_topic_id:
            kwargs["message_thread_id"] = config.cluster_probe_topic_id
        try:
            sent = await self._bot.send_message(**kwargs)
        except Exception as e:
            logger.warning("cluster_probe_send_failed", error=str(e))
            # Send failed — keep _prev_message_id as-is so the next tick
            # can still try to delete the prior probe (it's still in the
            # topic, assuming Telegram didn't lose it).
            return
        new_message_id: Optional[int] = getattr(sent, "message_id", None)

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
        # Persist so the next bot process can delete THIS one if we crash.
        self._persist_prev_message_id(new_message_id)
