"""Jira Oncall Precheck service.

Background asyncio task that polls Jira at shift-start (08:00 morning /
19:00 night local time), lists all k8s-category tickets fetched from
mcp-atlassian (HTTP/SSE), creates one Telegram forum topic per NEW ticket,
and auto-investigates each using HolmesService.chat on the MAIN model under
a READ-ONLY system-prompt rule + a non-admin user_id (so admin_gate's hard
block on every mutating kubectl/openstack verb stays active).

Lifecycle mirrors ClusterProbeService:
- TelegramBot constructs JiraOncallService() in __init__ (ungated).
- TelegramBot.initialize wires dependencies via set_*() calls.
- TelegramBot.start calls `await self.jira_oncall_service.start(self.application.bot, self)`
  gated by config.oncall_enabled and config.oncall_chat_id.
- TelegramBot.stop calls `await self.jira_oncall_service.stop()`.

Failure isolation: every public method swallows exceptions. MCP / Holmes /
Telegram-BS / MongoDB failures log structured warnings and skip the tick or
the ticket — never break the bot.
"""

import asyncio
import html
import time
from datetime import datetime, timedelta
from typing import Optional

import structlog

from src.bot.utils.config import config
from src.bot.utils.mcp_client import McpJiraClient
from src.bot.utils.mongodb import MongoDB
from src.bot.utils.rich_message import send_rich_reply

logger = structlog.get_logger(__name__)

SHIFTS_COLLECTION = "oncall_shifts"
TICKETS_COLLECTION = "oncall_tickets"

_READ_ONLY_RULE = """⚠️ READ-ONLY ON-CALL PRECHECK — DO NOT MODIFY ANY RESOURCE.

You are investigating this Jira ticket as an automated on-call precheck. The
on-call engineer has not yet reviewed the ticket. Your job is to:

- Read the ticket summary + description.
- Run ONLY inspection commands: kubectl get/describe/logs/top/events,
  kubectl explain, garden shoot snapshot read tools, opensafe inspection.
- Identify the affected cluster / shoot / namespace / resource from the
  ticket text and your diagnostic output.
- Propose root-cause candidates and the next diagnostic step.

NEVER call any mutating command. This includes, but is not limited to,
ALL of the following verbs (regardless of flags, namespaces, or context):

  kubectl delete            kubectl apply
  kubectl edit               kubectl replace
  kubectl patch              kubectl create
  kubectl scale              kubectl rollout
  kubectl annotate           kubectl label
  kubectl taint              kubectl drain
  kubectl cordon             kubectl uncordon
  kubectl expose             kubectl rolling-update
  kubectl exec               kubectl cp
  kubectl set (env / image / resources / serviceaccount / selector)
  openstack ... delete       openstack ... set
  openstack ... unset        openstack ... update
  openstack ... remove

Also any tool that writes, restarts, scales, or destroys resources.

If a remediation step would require a mutation, output the proposed command
as a fenced ```bash code block for the on-call engineer to review and run
manually. DO NOT invoke it as a tool call.

This is enforced at TWO layers: the prompt you are reading now, AND a
hard tool-executor block (admin_gate). Tool calls that match the mutation
list above will fail with a "hard-blocked: ask an admin" message. That is
expected — the on-call engineer must run them themselves.

Do NOT mark the Jira ticket as resolved — this is a precheck only. Reply
with:
1) What's happening (one paragraph)
2) Likely root cause candidates
3) Suggested next diagnostic commands or actions, as fenced ```bash
   blocks when they are mutations, or as tool-call results when read-only.
"""


class JiraOncallService:
    """Background scheduler for on-call precheck.

    Mirrors ClusterProbeService lifecycle (start(bot) / _loop / stop). All
    failure points are wrapped — a failed tick or ticket does not crash the
    bot.
    """

    SHIFTS_COLLECTION = SHIFTS_COLLECTION
    TICKETS_COLLECTION = TICKETS_COLLECTION
    READ_ONLY_RULE = _READ_ONLY_RULE

    def __init__(self) -> None:
        self._bot = None
        self._telegram_bot = None
        self._holmes = None
        self._permissions = None
        self._conversation_service = None
        self._task: Optional[asyncio.Task] = None
        self._mcp: Optional[McpJiraClient] = None

    def set_holmes(self, holmes_service) -> None:
        self._holmes = holmes_service

    def set_permissions(self, permission_service) -> None:
        self._permissions = permission_service

    def set_conversation_service(self, conv) -> None:
        self._conversation_service = conv

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self, bot, telegram_bot) -> None:
        self._bot = bot
        self._telegram_bot = telegram_bot
        if not config.oncall_enabled or not config.oncall_chat_id \
                or not config.oncall_mcp_url:
            logger.info("oncall_disabled_or_incomplete_config")
            return
        # Guard: the auto-investigation user MUST NOT be an admin. Otherwise
        # admin_gate's hard-block patterns would silently bypass and the bot
        # could mutate production clusters from a Jira ticket.
        if config.oncall_bot_user_id in (config.admin_user_ids or []):
            logger.error(
                "oncall_refuse_start_bot_user_id_is_admin",
                bot_user_id=config.oncall_bot_user_id,
                admin_ids=config.admin_user_ids,
            )
            return
        self._mcp = McpJiraClient(config.oncall_mcp_url)
        self._task = asyncio.create_task(self._oncall_loop())
        logger.info(
            "oncall_started",
            chat_id=config.oncall_chat_id,
            mcp_url=config.oncall_mcp_url,
            bot_user_id=config.oncall_bot_user_id,
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    async def _oncall_loop(self) -> None:
        # Continuous worker: every `oncall_poll_interval_seconds` ticks,
        # checks if `now` falls inside a registered shift, fetches Jira
        # for that shift's start-to-now window, dedups against
        # oncall_tickets, processes new ones. No "fire-once at shift-start"
        # gate — tickets created mid-shift are caught within ~120s.
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("oncall_tick_failed_bot_continues", error=str(e))
            await asyncio.sleep(config.oncall_poll_interval_seconds)

    async def _tick(self) -> None:
        now = datetime.now()
        current = self._compute_current_shift(now)
        if current is None:
            return
        date_str, shift = current
        shift_start_dt = self._shift_start_datetime(now, shift)
        # Load registrants — no one on call means no poll (saves MCP
        # calls and keeps the worker a true no-op outside shifts).
        col = MongoDB.get_collection(SHIFTS_COLLECTION)
        try:
            shifts = list(col.find({"date": date_str, "shift": shift}))
        except Exception as e:
            logger.warning("oncall_shifts_load_failed", error=str(e))
            return
        if not shifts:
            return
        logger.info(
            "oncall_shift_active",
            date=date_str, shift=shift, user_count=len(shifts),
            shift_start=shift_start_dt.isoformat(),
        )
        try:
            await self._fire_for_shift(date_str, shift, shift_start_dt, shifts)
        except Exception as e:
            logger.error(
                "oncall_fire_failed_shift",
                date=date_str, shift=shift, error=str(e),
            )
        # Bump watchdog whenever an active shift is running, not just on
        # new tickets — a long quiet stretch during a shift shouldn't
        # nuke the container via the 90m idle exit.
        if self._telegram_bot is not None:
            self._telegram_bot.last_update_time = time.time()

    @staticmethod
    def _compute_current_shift(now: datetime) -> Optional[tuple[str, str]]:
        """Return (date_str, shift) if `now` falls inside an enabled shift,
        else None. Morning = [08:00, 19:00), Night = [19:00, 08:00 next day).
        For post-midnight night ticks the date is YESTERDAY's — matches the
        shift document's `date` field set by `/oncall add <date> night`.

        A shift hour < 0 disables that shift (e.g. set morning_hour=-1 to
        run night-only).
        """
        m_h = config.oncall_shift_morning_hour
        n_h = config.oncall_shift_night_hour
        if m_h < 0 and n_h < 0:
            return None
        today = now.date()
        if m_h >= 0 and m_h <= now.hour < n_h:
            return (today.strftime("%Y-%m-%d"), "morning")
        if n_h >= 0 and now.hour >= n_h:
            return (today.strftime("%Y-%m-%d"), "night")
        if n_h >= 0 and now.hour < m_h:
            yesterday = today - timedelta(days=1)
            return (yesterday.strftime("%Y-%m-%d"), "night")
        return None

    @staticmethod
    def _shift_start_datetime(now: datetime, shift: str) -> datetime:
        """For morning: today at morning_hour. For night pre-midnight: today
        at night_hour. For night post-midnight: yesterday at night_hour —
        matches the (date, shift) document key so JQL `created >=` covers
        the entire ongoing night shift.
        """
        d = now.date()
        if shift == "morning":
            return datetime(d.year, d.month, d.day,
                            config.oncall_shift_morning_hour, 0, 0)
        if now.hour >= config.oncall_shift_night_hour:
            return datetime(d.year, d.month, d.day,
                            config.oncall_shift_night_hour, 0, 0)
        yesterday = d - timedelta(days=1)
        return datetime(yesterday.year, yesterday.month, yesterday.day,
                        config.oncall_shift_night_hour, 0, 0)

    async def _fire_for_shift(self, date_str: str, shift: str,
                              shift_start_dt: datetime,
                              shifts: list[dict]) -> None:
        # 1) Build JQL with window starting at shift_start. No assignee
        # filter — bot fetches ALL k8s tickets in the window.
        try:
            jql = config.oncall_jira_jql_template.format(
                since=shift_start_dt.strftime("%Y-%m-%d %H:%M"),
            )
        except Exception as e:
            logger.error("oncall_jql_format_failed", error=str(e))
            return
        issues = await self._mcp.search_issues(jql) if self._mcp else []
        if not issues:
            logger.info("oncall_no_issues", date=date_str, shift=shift)
            return
        # 2) Dedup against oncall_tickets (GLOBAL dedup — ticket scope is
        # shift-wide, so two humans registered for the same shift must not
        # double-create topics).
        tickets_col = MongoDB.get_collection(TICKETS_COLLECTION)
        oncall_user_ids = [
            doc.get("user_id") for doc in shifts if doc.get("user_id")
        ]
        new_issues = []
        try:
            for issue in issues:
                key = issue.get("key")
                if not key:
                    continue
                if tickets_col.find_one({"ticket_key": key}):
                    continue
                new_issues.append(issue)
        except Exception as e:
            logger.warning(
                "oncall_dedup_failed_processing_all",
                error=str(e), total=len(issues),
            )
            new_issues = [i for i in issues if i.get("key")]
        if not new_issues:
            logger.info(
                "oncall_no_new_issues", total=len(issues),
                date=date_str, shift=shift,
            )
            return
        if len(new_issues) > 50:
            logger.warning(
                "oncall_many_new_issues_capping",
                count=len(new_issues), cap=50,
            )
            new_issues = new_issues[:50]
        # 3) For each new ticket: create topic → header → Holmes.chat →
        # reply → mark fired. The bot.last_update_time bump below keeps
        # the watchdog quiet between tickets.
        for issue in new_issues:
            try:
                await self._process_ticket(
                    issue, date_str, shift, oncall_user_ids,
                )
                if self._telegram_bot is not None:
                    self._telegram_bot.last_update_time = time.time()
            except Exception as e:
                logger.error(
                    "oncall_ticket_failed",
                    key=issue.get("key"), error=str(e),
                )

    async def _process_ticket(self, issue: dict, date_str: str, shift: str,
                              oncall_user_ids: list[int]) -> None:
        key = issue.get("key", "ISD-UNKNOWN")
        # mcp-atlassian `jira_search` returns issues in the
        # `Issue.to_simplified_dict()` shape — summary, description, status,
        # priority, issuetype are TOP-LEVEL fields on each issue dict,
        # NOT nested under `fields` like raw Jira REST API v2. See upstream
        # src/mcp_atlassian/models/jira/issue.py:to_simplified_dict.
        summary = issue.get("summary") or "(no summary)"
        priority_obj = issue.get("priority", {}) or {}
        priority = priority_obj.get("name", "?")
        status_obj = issue.get("status", {}) or {}
        status = status_obj.get("name", "?")
        url = f"https://servicedesk.fci.vn/browse/{key}"
        # a) create topic.
        topic_name = f"{key} {summary}"[:100]
        try:
            forum_topic = await self._bot.create_forum_topic(
                chat_id=config.oncall_chat_id,
                name=topic_name,
                icon_color=config.oncall_topic_icon_color,
            )
            topic_id = forum_topic.message_thread_id
        except Exception as e:
            logger.error("oncall_topic_create_failed", key=key, error=str(e))
            return
        # b) post ticket header (always — even if Holmes fails). The
        # human mention is a list of Telegram user ids, not @handles (the
        # bot doesn't know handles here).
        oncall_mentions = ", ".join(str(uid) for uid in oncall_user_ids[:3])
        if len(oncall_user_ids) > 3:
            oncall_mentions += f" (+{len(oncall_user_ids) - 3} more)"
        header = (
            f"🚨 On-call precheck ({shift} shift, {date_str})\n"
            f"On-call: {html.escape(oncall_mentions) or '(no registrants)'}\n"
            f"Ticket: {key}\n"
            f"URL: {url}\n"
            f"Summary: {summary}\n"
            f"Priority: {priority}  Status: {status}\n"
            f"Auto-investigating with Holmes (main model, READ-ONLY)..."
        )
        try:
            await self._bot.send_message(
                chat_id=config.oncall_chat_id,
                text=header,
                message_thread_id=topic_id,
                disable_web_page_preview=True,
            )
        except Exception as e:
            logger.warning("oncall_header_send_failed", key=key, error=str(e))
        # c) build a Conversation under the virtual bot user (NOT the
        # on-call human) so the human's distilled history is not polluted.
        try:
            conversation = await (
                self._conversation_service.get_or_create_for_topic(
                    chat_id=config.oncall_chat_id, topic_id=topic_id,
                    user_id=config.oncall_bot_user_id,
                )
            )
        except Exception as e:
            logger.error(
                "oncall_conversation_init_failed",
                key=key, error=str(e),
            )
            return
        # d) ask Holmes with full tools — as a NON-admin user so
        # admin_gate hard-blocks mutations. user_permissions=[] (no
        # permission flags). READ-ONLY rule is injected via
        # extra_prompt_additions so it sits at the TOP of the system prompt
        # additions stack (highest priority).
        prompt = self._build_investigation_prompt(issue)
        try:
            response = await self._holmes.chat(
                user_id=config.oncall_bot_user_id,
                user_message=prompt,
                conversation=conversation,
                user_permissions=[],
                stream=False,
                telegram_context=None,
                chat_id=config.oncall_chat_id,
                topic_id=topic_id,
                user_display_name="oncall-precheck (auto)",
                extra_prompt_additions=_READ_ONLY_RULE,
            )
            # Persist conversation so on-call humans can chat follow-ups
            # in this topic and Holmes sees the prior investigation. Without
            # this, HolmesService.chat only mutates the in-memory Conversation
            # object — the actual save happens in MessageHandler, which the
            # oncall flow bypasses. ConversationService uses (chat_id,
            # topic_id) as the unique key for forum topics, so the same
            # conversation is fetched when a real user replies in the topic.
            try:
                await self._conversation_service.save_conversation(conversation)
            except Exception as save_err:
                logger.warning(
                    "oncall_conversation_save_failed",
                    key=key, error=str(save_err),
                )
            # Rich text render — Holmes responses frequently contain GFM
            # tables (kubectl output, yaml diffs, ticket field listings) so
            # route through send_rich_reply which auto-detects tables and
            # renders HTML/Markdown. Plain bot.send_message would dump raw
            # markdown into Telegram.
            try:
                await send_rich_reply(
                    self._bot,
                    chat_id=config.oncall_chat_id,
                    text=str(response)[:4000] or "(empty response)",
                    message_thread_id=topic_id,
                )
            except Exception as send_err:
                logger.warning(
                    "oncall_rich_reply_failed_falling_back",
                    key=key, error=str(send_err),
                )
                await self._bot.send_message(
                    chat_id=config.oncall_chat_id,
                    text=str(response)[:4000] or "(empty response)",
                    message_thread_id=topic_id,
                )
        except Exception as e:
            logger.warning(
                "oncall_holmes_failed",
                key=key, error=str(e),
            )
            try:
                await self._bot.send_message(
                    chat_id=config.oncall_chat_id,
                    text=(
                        f"⚠️ Auto-investigation failed: {e}\n"
                        f"See ticket: {url}"
                    ),
                    message_thread_id=topic_id,
                )
            except Exception:
                pass
            return
        # e) mark ticket as fired so future shift-starts don't re-process
        # (GLOBAL dedup — key is the only unique identifier).
        try:
            MongoDB.get_collection(TICKETS_COLLECTION).update_one(
                {"ticket_key": key},
                {"$set": {
                    "ticket_key": key,
                    "fired_at": datetime.utcnow(),
                    "shift_date": date_str,
                    "topic_id": topic_id,
                    "oncall_user_ids": oncall_user_ids,
                }},
                upsert=True,
            )
        except Exception as e:
            logger.warning("oncall_ticket_mark_failed", key=key, error=str(e))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_investigation_prompt(issue: dict) -> str:
        key = issue.get("key", "ISD-UNKNOWN")
        url = f"https://servicedesk.fci.vn/browse/{key}"
        summary = issue.get("summary") or ""
        description = issue.get("description") or ""
        return (
            f"This is an automated on-call precheck for Jira ticket "
            f"[{key}]({url}).\n"
            f"Category: Cloud FKE (K8S)\n"
            f"Summary: {summary}\n"
            f"Description:\n{description[:4000]}\n\n"
            f"Please investigate this ticket as a Kubernetes/Gardener/"
            f"OpenStack issue. Identify the affected cluster/resources from "
            f"the summary and any text that references a shoot name, a "
            f"kubeconfig, a namespace, or an error string. Run the "
            f"appropriate read-only tools (snapshot, kubectl get, kubectl "
            f"describe, kubectl logs, etc.) and propose next-step "
            f"remediation.\n"
            f"Remember: do NOT run any mutating command. Output mutation "
            f"proposals as fenced ```bash blocks for the on-call engineer.\n"
            f"Reply with:\n"
            f"1) What's happening (one paragraph)\n"
            f"2) Likely root cause candidates\n"
            f"3) Suggested next diagnostic commands or actions\n"
        )

    # ------------------------------------------------------------------
    # Admin slash-command entry points (called by OncallCommand)
    # ------------------------------------------------------------------
    async def fire_now(self, date_str: str = "", shift: str = "") -> dict:
        """Manual fire for `/oncall test` admin slash command. Returns a
        short summary dict for the reply. Bypasses the time-check."""
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")
        auto_pick_shift = not shift
        if auto_pick_shift:
            now = datetime.now()
            shift = "morning" if now.hour < 19 else "night"
        col = MongoDB.get_collection(SHIFTS_COLLECTION)
        try:
            shifts = list(col.find({"date": date_str, "shift": shift}))
        except Exception as e:
            return {"date": date_str, "shift": shift, "error": str(e),
                    "fired": 0}
        # Auto-pick fallback: if the time-based shift guess has no
        # registrants, try the OTHER shift for the same date. Helps
        # "I registered night but tested mid-morning" not error out.
        if not shifts and auto_pick_shift:
            other = "night" if shift == "morning" else "morning"
            try:
                shifts = list(col.find({"date": date_str, "shift": other}))
            except Exception as e:
                return {"date": date_str, "shift": other, "error": str(e),
                        "fired": 0}
            if shifts:
                shift = other
        if not shifts:
            return {"date": date_str, "shift": shift,
                    "error": "no oncall shifts registered for this date/shift",
                    "fired": 0}
        before = time.time()
        # /oncall test may be invoked outside any configured shift, so use
        # the lookback window instead of shift_start — the operator's
        # intent is "investigate what's been happening recently", not
        # "investigate the current shift".
        shift_start_dt = datetime.now() - timedelta(
            minutes=config.oncall_lookback_minutes,
        )
        try:
            await self._fire_for_shift(date_str, shift, shift_start_dt, shifts)
        except Exception as e:
            return {"date": date_str, "shift": shift, "error": str(e),
                    "fired": 0}
        # Reset watchdog on manual fire too — long investigations need
        # protection.
        if self._telegram_bot is not None:
            self._telegram_bot.last_update_time = time.time()
        elapsed = round(time.time() - before, 2)
        return {"date": date_str, "shift": shift, "elapsed_seconds": elapsed}
