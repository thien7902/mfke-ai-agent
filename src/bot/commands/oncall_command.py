"""`/oncall` admin-only slash command for registering on-call shifts and
manually firing the precheck scheduler.

Subcommands:
- `/oncall add <YYYY-MM-DD> <morning|night>`
- `/oncall list`                  — lists caller's upcoming shifts
- `/oncall rm <YYYY-MM-DD> <morning|night>`
- `/oncall purge`                  — drops all caller's future shifts
- `/oncall test`                   — fires the scheduler immediately for
                                      today's current shift (debugging/E2E)

The slash command is subcommand-dispatched via `context.args[0]`. There is
no existing subcommand precedent in this repo (every other bot command is a
flat handler), so this file establishes the dispatcher pattern.

Storage: MongoDB `oncall_shifts` collection — one document per
(user_id, date, shift). See src.bot.services.jira_oncall_service.
"""
import re
from datetime import datetime
from typing import Optional

import structlog
from telegram import Update
from telegram.ext import ContextTypes

from src.bot.services.jira_oncall_service import JiraOncallService
from src.bot.utils.decorators import admin_only
from src.bot.utils.mongodb import MongoDB

logger = structlog.get_logger(__name__)

SHIFTS_COLLECTION = "oncall_shifts"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VALID_SHIFTS = ("morning", "night")


class OncallCommand:
    """`/oncall <subcommand>` dispatcher."""

    def __init__(self, oncall_service: JiraOncallService):
        self._oncall = oncall_service

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    @admin_only
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Dispatches on the first argument."""
        args = context.args or []
        if not args:
            await self._reply_usage(update)
            return
        sub = args[0].lower()
        rest = args[1:]
        if sub == "add":
            await self._handle_add(update, rest)
        elif sub == "list":
            await self._handle_list(update, rest)
        elif sub == "rm":
            await self._handle_rm(update, rest)
        elif sub == "purge":
            await self._handle_purge(update, rest)
        elif sub == "test":
            await self._handle_test(update, rest)
        else:
            await self._reply_usage(update)

    # ------------------------------------------------------------------
    # Subcommands
    # ------------------------------------------------------------------
    async def _handle_add(self, update: Update, rest: list[str]) -> None:
        date_str, shift, error = self._parse_date_shift(rest)
        if error:
            await update.message.reply_text(error)
            return
        user_id = update.effective_user.id
        col = MongoDB.get_collection(SHIFTS_COLLECTION)
        try:
            col.update_one(
                {"user_id": user_id, "date": date_str, "shift": shift},
                {"$set": {
                    "user_id": user_id,
                    "date": date_str,
                    "shift": shift,
                    "created_at": datetime.utcnow(),
                }},
                upsert=True,
            )
        except Exception as e:
            logger.error("oncall_add_failed", error=str(e))
            await update.message.reply_text(
                f"❌ Failed to register shift: {e}"
            )
            return
        await update.message.reply_text(
            f"✅ On-call shift added: {date_str} ({shift}). The bot will "
            f"fetch all k8s Jira tickets at the shift start time and "
            f"create one precheck topic per NEW ticket."
        )

    async def _handle_list(self, update: Update, rest: list[str]) -> None:
        user_id = update.effective_user.id
        col = MongoDB.get_collection(SHIFTS_COLLECTION)
        try:
            docs = list(col.find({"user_id": user_id}).sort("date", 1))
        except Exception as e:
            logger.error("oncall_list_failed", error=str(e))
            await update.message.reply_text(f"❌ Failed to list shifts: {e}")
            return
        if not docs:
            await update.message.reply_text(
                "ℹ️ You have no registered on-call shifts.\n"
                "Usage: `/oncall add <YYYY-MM-DD> <morning|night>`"
            )
            return
        lines = []
        today = datetime.utcnow().strftime("%Y-%m-%d")
        for doc in docs:
            date_str = doc.get("date", "?")
            shift = doc.get("shift", "?")
            marker = " *" if date_str >= today else ""
            lines.append(f"- `{date_str}` ({shift}){marker}")
        await update.message.reply_text(
            "📋 Your on-call shifts:\n" + "\n".join(lines),
            parse_mode="Markdown",
        )

    async def _handle_rm(self, update: Update, rest: list[str]) -> None:
        date_str, shift, error = self._parse_date_shift(rest)
        if error:
            await update.message.reply_text(error)
            return
        user_id = update.effective_user.id
        col = MongoDB.get_collection(SHIFTS_COLLECTION)
        try:
            result = col.delete_one(
                {"user_id": user_id, "date": date_str, "shift": shift}
            )
        except Exception as e:
            logger.error("oncall_rm_failed", error=str(e))
            await update.message.reply_text(f"❌ Failed: {e}")
            return
        if result.deleted_count:
            await update.message.reply_text(
                f"🗑️ Removed: {date_str} ({shift})"
            )
        else:
            await update.message.reply_text(
                f"ℹ️ No shift found for {date_str} ({shift})"
            )

    async def _handle_purge(self, update: Update, rest: list[str]) -> None:
        user_id = update.effective_user.id
        col = MongoDB.get_collection(SHIFTS_COLLECTION)
        try:
            result = col.delete_many({"user_id": user_id})
        except Exception as e:
            logger.error("oncall_purge_failed", error=str(e))
            await update.message.reply_text(f"❌ Failed: {e}")
            return
        await update.message.reply_text(
            f"🧹 Removed {result.deleted_count} shifts"
        )

    async def _handle_test(self, update: Update, rest: list[str]) -> None:
        """Manual fire for End-to-end verification. Optionally takes
        `<YYYY-MM-DD> <morning|night>`; defaults to today/current shift."""
        date_str = ""
        shift = ""
        if len(rest) >= 2:
            date_str, shift, error = self._parse_date_shift(rest)
            if error:
                await update.message.reply_text(error)
                return
        await update.message.reply_text(
            "🧪 Firing on-call precheck now (override)..."
        )
        try:
            result = await self._oncall.fire_now(date_str, shift)
        except Exception as e:
            await update.message.reply_text(f"❌ Test failed: {e}")
            return
        if result.get("error"):
            await update.message.reply_text(
                f"ℹ️ {result.get('error')}"
            )
            return
        await update.message.reply_text(
            f"✅ Fired: {result.get('date')} {result.get('shift')} "
            f"in {result.get('elapsed_seconds', '?')}s. See forum topics."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_date_shift(rest: list[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Returns (date_str, shift, error_message). On success, error is None."""
        if len(rest) < 2:
            return (None, None,
                    "ℹ️ Usage: `/oncall add <YYYY-MM-DD> <morning|night>`")
        date_str, shift = rest[0], rest[1].lower()
        if not _DATE_RE.match(date_str):
            return (None, None,
                    f"❌ Invalid date `{date_str}`. Format: YYYY-MM-DD")
        if shift not in _VALID_SHIFTS:
            return (None, None,
                    f"❌ Invalid shift `{shift}`. Must be morning or night.")
        return (date_str, shift, None)

    @staticmethod
    async def _reply_usage(update: Update) -> None:
        await update.message.reply_text(
            "ℹ️ *On-call precheck* usage:\n"
            "• `/oncall add 2026-11-10 morning` — register your shift\n"
            "• `/oncall list` — see your upcoming shifts\n"
            "• `/oncall rm 2026-11-10 morning` — remove a shift\n"
            "• `/oncall purge` — remove all your shifts\n"
            "• `/oncall test` — fire precheck now (debug)\n\n"
            "At the configured shift-start hour (08:00 morning / 19:00 "
            "night), the bot lists all k8s Jira tickets in category 'Cloud "
            "FKE (K8S)' updated since the lookback window, creates one "
            "forum topic per NEW ticket, and auto-investigates each with "
            "Holmes on the main model in READ-ONLY mode (no mutations).",
            parse_mode="Markdown",
        )
