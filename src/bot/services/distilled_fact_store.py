"""DistilledFactStore — MongoDB backing for the background consolidator.

Two collections:

- ``distilled_facts`` — every fact ever extracted by a consolidator run.
  Each document carries ``source_run_id`` so we can filter by run.
  Unique index on ``text`` gives race-safe global dedup — a DuplicateKeyError
  means another run (concurrent or past) already extracted the same sentence.

- ``distilled_runs`` — one document per ``(chat_id, topic_id)`` pointing at
  the latest run id for that topic, plus ``updated_at``. Used at inject time
  to pick "the most recent distillation run per topic" — older runs' facts
  stay in ``distilled_facts`` for future /memory queries but are not injected
  into the system prompt (newer run wins, per the user's
  "fact sau ưu tiên hơn fact trước" intent).

Sync pymongo inside ``async def`` mirrors ConversationService — no motor
anywhere in the repo (verified). Pymongo calls are blocking but cheap
(indexed lookup, small result sets) and the bot relies on per-topic locks
to avoid stacking them.

Failure isolation: every public method swallows DB errors and returns a
safe empty/false value so the bot keeps working when Mongo blips. The
consolidator log surface gets the warning; user-visible behavior is
"no facts this turn" rather than a crashed turn.
"""
import datetime
from typing import List, Optional

import structlog
from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from src.bot.utils.mongodb import MongoDB

logger = structlog.get_logger(__name__)


class DistilledFactStore:
    """Owns the two distilled-facts collections and the inject-time lookup."""

    FACTS_COLLECTION = "distilled_facts"
    RUNS_COLLECTION = "distilled_runs"

    def __init__(self):
        self._facts: Optional[Collection] = None
        self._runs: Optional[Collection] = None

    # ------------------------------------------------------------------ #
    # Lazy collection accessors (mirror ConversationService pattern)
    # ------------------------------------------------------------------ #

    @property
    def facts(self) -> Collection:
        if self._facts is None:
            self._facts = MongoDB.get_collection(self.FACTS_COLLECTION)
            try:
                self._facts.create_index([("text", ASCENDING)], unique=True)
                self._facts.create_index(
                    [("source_chat_id", ASCENDING), ("source_topic_id", ASCENDING)]
                )
                self._facts.create_index(
                    [("source_topic_id", ASCENDING), ("source_run_id", ASCENDING)]
                )
            except Exception as e:
                logger.warning("distilled_facts_index_failed", error=str(e))
        return self._facts

    @property
    def runs(self) -> Collection:
        if self._runs is None:
            self._runs = MongoDB.get_collection(self.RUNS_COLLECTION)
            try:
                self._runs.create_index(
                    [("chat_id", ASCENDING), ("topic_id", ASCENDING)],
                    unique=True,
                )
            except Exception as e:
                logger.warning("distilled_runs_index_failed", error=str(e))
        return self._runs

    # ------------------------------------------------------------------ #
    # Writes (called by ConsolidatorService)
    # ------------------------------------------------------------------ #

    async def add(
        self,
        text: str,
        source_chat_id: int,
        source_topic_id: int,
        source_run_id: str,
    ) -> bool:
        """Insert one fact. Returns True if newly inserted, False on duplicate
        (same ``text`` already exists). All other errors are swallowed and
        return False so a Mongo blip doesn't crash the consolidator.
        """
        if not text or not text.strip():
            return False
        try:
            self.facts.insert_one(
                {
                    "text": text.strip(),
                    "source_chat_id": source_chat_id,
                    "source_topic_id": source_topic_id,
                    "source_run_id": source_run_id,
                    "created_at": datetime.datetime.utcnow(),
                }
            )
            return True
        except DuplicateKeyError:
            return False
        except Exception as e:
            logger.warning(
                "distilled_fact_add_failed",
                error=str(e),
                error_type=type(e).__name__,
                fact_preview=text[:80],
            )
            return False

    async def record_run(
        self,
        chat_id: int,
        topic_id: int,
        run_id: str,
        last_distilled_count: Optional[int] = None,
    ) -> None:
        """Mark ``run_id`` as the latest distillation run for this topic.
        Optionally also persist ``last_distilled_count`` (the message-count
        checkpoint the consolidator inspected) so it survives process
        restart — without this, the consolidator re-runs every old topic
        after a restart because the in-memory checkpoint is empty.

        Best-effort: a Mongo failure here means the topic's injection set
        keeps the previous run's facts for now — the next successful
        ``record_run`` will catch up.
        """
        update = {
            "$set": {
                "chat_id": chat_id,
                "topic_id": topic_id,
                "latest_run_id": run_id,
                "updated_at": datetime.datetime.utcnow(),
            }
        }
        if last_distilled_count is not None:
            update["$set"]["last_distilled_count"] = int(last_distilled_count)
        try:
            self.runs.update_one(
                {"chat_id": chat_id, "topic_id": topic_id},
                update,
                upsert=True,
            )
        except Exception as e:
            logger.warning(
                "distilled_run_record_failed",
                error=str(e),
                error_type=type(e).__name__,
                chat_id=chat_id,
                topic_id=topic_id,
                run_id=run_id,
            )

    def get_last_distilled_count(self, chat_id: int, topic_id: int) -> int:
        """Return the persisted message-count checkpoint for this topic,
        or 0 when no record exists. Sync — called from the consolidator's
        _consolidate path before the LLM call to short-circuit topics whose
        message count hasn't changed since the last run (including across
        bot restarts)."""
        try:
            doc = self.runs.find_one(
                {"chat_id": chat_id, "topic_id": topic_id},
                {"last_distilled_count": 1, "_id": 0},
            )
            return int((doc or {}).get("last_distilled_count") or 0)
        except Exception as e:
            logger.warning(
                "distilled_run_get_count_failed",
                error=str(e),
                error_type=type(e).__name__,
                chat_id=chat_id,
                topic_id=topic_id,
            )
            return 0

    def retire_topic_facts(self, chat_id: int, topic_id: int) -> int:
        """Delete every fact previously stored for this topic, regardless of
        which run produced it. Called by ConsolidatorService before inserting
        a new run's facts — clean-slate semantics.

        Why delete (not soft-mark): the user wants "1 fully-formed fact per
        entity, no temporal accumulation". Old facts lingering with
        ``superseded=True`` would still consume space + index slots, and
        ``get_injectable_facts`` already filters by latest_run_id — soft-
        mark adds work for no benefit.

        Sync — called from the consolidator's async _consolidate path
        (not awaited, runs in the same thread context as the consolidator's
        other store calls). Best-effort: errors bubble to the caller to log.

        Returns the number of deleted documents.
        """
        try:
            result = self.facts.delete_many(
                {"source_chat_id": chat_id, "source_topic_id": topic_id}
            )
            return int(getattr(result, "deleted_count", 0) or 0)
        except Exception as e:
            logger.warning(
                "distilled_facts_retire_failed",
                error=str(e),
                error_type=type(e).__name__,
                chat_id=chat_id,
                topic_id=topic_id,
            )
            return 0

    # ------------------------------------------------------------------ #
    # Reads (called from sync _build_holmes_messages in HolmesService)
    # ------------------------------------------------------------------ #

    def get_injectable_facts(self) -> List[dict]:
        """Return facts from the latest run per topic, across all topics.

        Sync — called from the sync ``_build_holmes_messages`` path so the
        prompt is assembled without an extra thread hop. Returns ``[]`` on
        any error so the bot keeps working without injected facts this
        turn. Documents are projected to ``{text, source_topic_id,
        source_run_id, created_at}`` for the caller.

        Algorithm: read all latest-run pointers from ``distilled_runs``
        (one doc per topic), then fetch facts whose ``source_run_id`` is in
        that set. Older runs' facts stay in ``distilled_facts`` but are
        filtered out here — this is the "newer run wins per topic" filter.
        """
        try:
            latest_runs = list(
                self.runs.find({}, {"_id": 0, "latest_run_id": 1, "topic_id": 1})
            )
            if not latest_runs:
                return []
            run_ids = [r["latest_run_id"] for r in latest_runs if r.get("latest_run_id")]
            if not run_ids:
                return []
            cursor = (
                self.facts.find(
                    {"source_run_id": {"$in": run_ids}},
                    {
                        "_id": 0,
                        "text": 1,
                        "source_topic_id": 1,
                        "source_run_id": 1,
                        "created_at": 1,
                    },
                )
                .sort("created_at", ASCENDING)
            )
            return list(cursor)
        except Exception as e:
            logger.warning(
                "distilled_facts_get_injectable_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return []
