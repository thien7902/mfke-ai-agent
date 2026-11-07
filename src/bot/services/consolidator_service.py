"""ConsolidatorService — background knowledge distiller for chat topics.

After a forum topic has been idle for ``consolidator_idle_seconds`` after
the last assistant response, this service fires an LLM call on a separate
cheap model (``consolidator_model``) over the recent transcript, extracts
reusable facts, and stores them via ``DistilledFactStore.add``.

Design notes:

- **Idle trigger**: per-``(chat_id, topic_id)`` ``asyncio.TimerHandle``
  armed by ``schedule_if_idle()``. The MessageHandler calls it after each
  ``save_conversation``. Re-arming cancels any prior timer (debounce-style).

- **Idle guard (no LLM storm)**: per-topic checkpoint
  ``_last_distilled_count[key]``. If the reloaded conversation has no more
  messages than the previous run, skip without calling the LLM. Idle
  topics are free after the first distillation.

- **Concurrency**: per-topic ``asyncio.Lock``. If a new ``schedule_if_idle``
  re-arms while a prior ``_fire`` is in flight, the new fire is a no-op
  (lock held → skip). No queueing — protects against cascading LLM calls
  when a fast back-and-forth chat re-arms repeatedly.

- **Cross-topic bound**: a single ``asyncio.Semaphore(consolidator_max_concurrent)``
  wraps only the LLM call (not the whole consolidate path). Per-topic locks
  still prevent same-topic overlap; the semaphore bounds how many topics
  hit the model endpoint at once so a burst of idle topics doesn't hammer
  the API.

- **LLM timeout**: ``asyncio.wait_for`` wraps the ``to_thread(_call_consolidator_llm)``
  call with ``consolidator_llm_timeout_seconds``. A hung model call would
  otherwise hold the cross-topic semaphore indefinitely.

- **Failure isolation**: every step inside ``_fire`` is wrapped in
  try/except and logged. A consolidator failure must never surface to the
  user or block the bot — timer fires fire-and-forget.

- **No skill creation / no transcript dump**: only distilled facts are
  stored (no raw text). Graphiti + ``/addmemory`` + ``search_memory`` are
  untouched for admin-curated knowledge.
"""
import asyncio
import json
import os
import uuid
from typing import Optional

import openai
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
)

from src.bot.services.conversation_service import ConversationService
from src.bot.services.distilled_fact_store import DistilledFactStore
from src.bot.utils.config import config as bot_config

logger = structlog.get_logger(__name__)


_SYSTEM_PROMPT = (
    "You are a knowledge extractor for an MFKE/Gardener operations Telegram bot.\n"
    "Read the chat transcript below and extract GENERAL, reusable operational "
    "knowledge that will help in future conversations on ANY cluster — not "
    "pinned to one specific shoot, namespace, or account.\n\n"
    "## CRITICAL CONSTRAINTS\n"
    "1. Each fact must be ONE RICH PARAGRAPH of about 5 sentences. Combine "
    "related observations about the same pattern, component, or behavior into "
    "one paragraph. Do NOT split one insight into multiple atomic one-liners, "
    "and do NOT emit a single terse sentence either.\n"
    "2. Generalize away from the specific cluster name in the transcript. "
    "Replace 'In the fmon-agent-1-yant1n2u Shoot' with 'In MFKE Shoots running "
    "<component>' or 'For OTEL agents on MFKE shoots'. Keep technical specifics "
    "that are general (image names, namespace conventions, controller names, "
    "error signatures, OpenStack flavor/gardener region mappings) — drop "
    "qualifiers that only identify a single cluster.\n"
    "3. Scope = MFKE, Gardener, OpenStack, VMware/vSphere platforms from a "
    "operator's lens. Extract:\n"
    "   - Architecture patterns (OTEL agents deployed as DaemonSet/StatefulSet "
    "     on MFKE shoots, control plane pod topology in seeds, MCM boot flow, "
    "     etcd backup-restore sidecar behavior)\n"
    "   - Runbook patterns proven in this conversation (how a senior engineer "
    "     diagnosed an issue, what they checked, in what order)\n"
    "   - Cross-component gotchas (CSI driver ↔ Cinder, MCM ↔ OpenStack "
    "     image_id, Gardenlet ↔ seed throttling)\n"
    "   - Common pitfalls + their fixes observed here\n"
    "   - Account/tenant/project naming conventions that hold across the "
    "     platform\n"
    "4. SKIP:\n"
    "   - Conversation-ephemeral state: 'currently debugging X', 'patch in "
    "     progress', a specific transient log line, a one-off scaling event\n"
    "   - Generic Kubernetes textbook truths with no MFKE/Gardener flavor "
    "     ('StatefulSet pods are ordered' — until you tie it to a Gardener\n"
    "     control plane pattern)\n"
    "   - Greetings, status updates, ad-hoc kubectl commands with no "
    "     reusable lesson\n"
    "5. The fact must read as a self-contained knowledge paragraph that a "
    "   senior engineer would write in a runbook — not as a transcript "
    "   excerpt with shoot names still in it.\n\n"
    "## OUTPUT FORMAT\n"
    "Strict JSON only — no prose, no markdown fences:\n"
    '{"facts":[{"text":"<5-sentence general-knowledge paragraph>"}]}\n'
    'Return {"facts":[]} when nothing qualifies — a conversation with zero '
    'reusable lessons should produce an empty array, not filler.'
)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_consolidator_llm(transcript: str) -> list:
    """Run the consolidator LLM on a transcript. Returns the parsed
    ``facts`` list (may be empty). Raises on persistent LLM failure — the
    caller wraps it in try/except and logs ``consolidator_llm_failed``.

    Reuses ``OPENAI_API_KEY`` / ``OPENAI_API_BASE`` from the environment
    (same as Holmes/vision_service). The model is ``consolidator_model``
    (env ``CONSOLIDATOR_MODEL``); if unset, returns ``[]`` so a missing
    config disables the consolidator gracefully instead of crashing.
    """
    if not bot_config.consolidator_model:
        return []
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []
    api_base = os.environ.get("OPENAI_API_BASE")
    client = (
        openai.OpenAI(api_key=api_key, base_url=api_base)
        if api_base
        else openai.OpenAI(api_key=api_key)
    )
    resp = client.chat.completions.create(
        model=bot_config.consolidator_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        # Strip ```json...``` / ```...``` fences defensively. Some chat
        # models wrap JSON output in fences regardless of the prompt.
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            logger.warning("consolidator_invalid_json", raw_preview=raw[:200])
            return []
        facts = data.get("facts", [])
        if not isinstance(facts, list):
            logger.warning("consolidator_invalid_facts_shape", raw_preview=raw[:200])
            return []
        return facts
    except json.JSONDecodeError as e:
        logger.warning("consolidator_invalid_json", error=str(e), raw_preview=raw[:200])
        return []


class ConsolidatorService:
    """Schedules + runs per-topic knowledge distillation on idle."""

    def __init__(
        self,
        distilled_store: DistilledFactStore,
        conversation_service: ConversationService,
    ):
        self._store = distilled_store
        self._conversations = conversation_service
        # Per-(chat_id, topic_id) pending idle timer (debounce-style).
        self._timers: dict = {}
        # Per-topic lock — if held, a consolidation is in flight; new _fire is a no-op.
        self._locks: dict = {}
        # Per-topic checkpoint: the message count inspected on the last
        # successful consolidation. Used to skip idle topics without LLM cost.
        self._last_distilled_count: dict = {}
        # Cross-topic bound on concurrent consolidator LLM calls. Acquired
        # only around the LLM call, not the whole consolidate path, so DB
        # writes / conversation reload don't count against the limit.
        self._semaphore: Optional[asyncio.Semaphore] = None
        # Tasks for in-flight _fire coroutines. Tracked so stop() can drain
        # them instead of cancelling mid-LLM-call (per user choice B).
        self._in_flight: set = set()

    # ------------------------------------------------------------------ #
    # Public API — called by MessageHandler after each save_conversation
    # ------------------------------------------------------------------ #

    def schedule_if_idle(self, chat_id: int, topic_id: Optional[int]) -> None:
        """Arm (or re-arm) the idle timer for this topic. No-op when the
        consolidator is disabled. Always cancels any prior pending timer
        for the same topic so rapid chat re-arms the timer rather than
        stacking multiple fires."""
        if not bot_config.consolidator_enabled:
            return
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(
                max(1, bot_config.consolidator_max_concurrent)
            )
        key = (chat_id, topic_id or 0)
        prior = self._timers.get(key)
        if prior is not None:
            try:
                prior.cancel()
            except Exception:
                pass
            self._timers.pop(key, None)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        self._timers[key] = loop.call_later(
            bot_config.consolidator_idle_seconds,
            lambda: asyncio.ensure_future(self._fire_and_track(key)),
        )

    # ------------------------------------------------------------------ #
    # Fire path — runs on the bot event loop after the idle timer elapses
    # ------------------------------------------------------------------ #

    async def _fire_and_track(self, key) -> None:
        """Task body scheduled by the idle timer. Wraps _fire so stop()
        can await all in-flight fires (drain) rather than leave a fire
        half-cut. The set is the source of truth for stop()'s drain — any
        fire that completes discards its task from the set.
        """
        task = asyncio.current_task()
        if task is not None:
            self._in_flight.add(task)
        try:
            await self._fire(key)
        finally:
            if task is not None:
                self._in_flight.discard(task)

    async def _fire(self, key) -> None:
        # Remove the spent timer handle so the next schedule_if_idle starts clean.
        self._timers.pop(key, None)
        lock = self._locks.setdefault(key, asyncio.Lock())
        if lock.locked():
            # A consolidation is already running for this topic. Don't queue
            # another — the in-flight run will pick up any newer messages on
            # its OWN reload window. The next message after that run completes
            # will re-arm and pick up whatever is left.
            logger.info("consolidator_skipped_in_flight", key=key)
            return
        async with lock:
            try:
                await self._consolidate(key)
            except Exception as e:
                logger.warning(
                    "memory_consolidated_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    key=key,
                    exc_info=True,
                )

    async def _consolidate(self, key) -> None:
        chat_id, topic_id = key
        conversation = await self._conversations.get_conversation_by_topic(
            topic_id, chat_id
        )
        if conversation is None:
            # Conversation was cleared/deleted between arm and fire — nothing to distill.
            logger.info("consolidator_no_conversation", chat_id=chat_id, topic_id=topic_id)
            return
        messages = conversation.get_recent_messages(bot_config.consolidator_max_turns)

        # Idle guard — skip if no new messages since the last consolidation
        # for this topic. This makes idle topics free after the first run.
        last_count = self._last_distilled_count.get(key, 0)
        if len(messages) <= last_count:
            logger.info(
                "consolidator_no_new_messages",
                chat_id=chat_id,
                topic_id=topic_id,
                current=len(messages),
                last=last_count,
            )
            return
        if len(messages) < bot_config.consolidator_min_turns:
            logger.info(
                "consolidator_skipped_too_few_turns",
                chat_id=chat_id,
                topic_id=topic_id,
                num_messages=len(messages),
            )
            return

        transcript = self._format_transcript(messages)
        if not transcript.strip():
            logger.info(
                "consolidator_empty_transcript",
                chat_id=chat_id,
                topic_id=topic_id,
            )
            return

        run_id = uuid.uuid4().hex
        try:
            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(
                    max(1, bot_config.consolidator_max_concurrent)
                )
            async with self._semaphore:
                try:
                    facts = await asyncio.wait_for(
                        asyncio.to_thread(_call_consolidator_llm, transcript),
                        timeout=bot_config.consolidator_llm_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "consolidator_llm_timeout",
                        chat_id=chat_id,
                        topic_id=topic_id,
                        timeout_seconds=bot_config.consolidator_llm_timeout_seconds,
                    )
                    return
        except Exception as e:
            logger.warning(
                "consolidator_llm_failed",
                error=str(e),
                error_type=type(e).__name__,
                chat_id=chat_id,
                topic_id=topic_id,
            )
            return

        new_count = 0
        dup_count = 0
        # Clean-slate per topic: retire all facts from prior runs of this
        # topic before inserting the new run's facts. The user's intent is
        # "1 fact per entity, fully formed, no temporal accumulation" —
        # if a new run produces richer facts they overwrite (not append to)
        # the old ones. Same-text-upsert is insufficient because the prior
        # facts would still linger and confuse the injection layer.
        try:
            self._store.retire_topic_facts(chat_id, topic_id)
        except Exception as e:
            logger.warning(
                "distilled_facts_retire_failed",
                error=str(e),
                error_type=type(e).__name__,
                chat_id=chat_id,
                topic_id=topic_id,
            )
            # Continue anyway — best-effort. Worst case the new facts
            # coexist with old ones for this topic; injection still picks
            # the latest run_id (the new one) via distilled_runs.
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            text = (fact.get("text") or "").strip()
            if not text:
                continue
            try:
                added = await self._store.add(
                    text=text,
                    source_chat_id=chat_id,
                    source_topic_id=topic_id,
                    source_run_id=run_id,
                )
                if added:
                    new_count += 1
                else:
                    dup_count += 1
            except Exception as e:
                logger.warning(
                    "distilled_fact_add_failed",
                    error=str(e),
                    fact_preview=text[:80],
                )

        # Record this run as the latest for the topic so the injection
        # swaps to the new facts. Even runs with 0 new facts advance the
        # pointer — newer run wins per the user's "fact sau ưu tiên hơn
        # fact trước" intent. (See Known follow-ups in the plan for the
        # empty-run wipe caveat.)
        await self._store.record_run(chat_id, topic_id, run_id)

        # Update the per-topic checkpoint so the next idle fire on the
        # same message count is skipped.
        self._last_distilled_count[key] = len(messages)

        logger.info(
            "memory_consolidated",
            chat_id=chat_id,
            topic_id=topic_id,
            run_id=run_id,
            num_new=new_count,
            num_dup=dup_count,
            num_llm_facts=len(facts),
            checkpoint=len(messages),
        )

    @staticmethod
    def _format_transcript(messages) -> str:
        """Flatten Holmes-style messages into a `role: content` transcript.
        Only user/assistant roles are kept — system messages and tool calls
        are noise for the knowledge extractor.
        """
        lines = []
        for m in messages:
            role = getattr(m, "role", None) or ""
            if role not in ("user", "assistant"):
                continue
            content = (getattr(m, "content", "") or "").strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def stop(self) -> None:
        """Cancel pending timers + drain in-flight fires + clear state.

        Per user choice B, in-flight fires are AWAITED (with the per-fire
        LLM timeout already in place) rather than cancelled — so a half-done
        consolidator can finish its DB writes and checkpoint update instead
        of leaving MongoDB in a half-consolidated state. Idempotent.
        """
        for handle in list(self._timers.values()):
            try:
                handle.cancel()
            except Exception:
                pass
        self._timers.clear()

        # Drain in-flight fires (with the LLM timeout already bounding each
        # one's tail). Copy the set so concurrent discard doesn't trip
        # iteration mid-await.
        pending = list(self._in_flight)
        if pending:
            logger.info("consolidator_stop_draining", in_flight=len(pending))
            await asyncio.gather(*pending, return_exceptions=True)

        self._locks.clear()
        self._last_distilled_count.clear()
        self._in_flight.clear()
