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
import math
from typing import List, Optional

import structlog
from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from src.bot.utils.config import config as bot_config
from src.bot.utils.mongodb import MongoDB

logger = structlog.get_logger(__name__)


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _dedup_facts_by_cosine(facts: List[dict], threshold: float) -> List[dict]:
    """Drop near-duplicate facts (cosine ≥ ``threshold``) by greedy
    pairwise comparison. Newest fact wins — when two facts collide, the
    one with the LATER ``created_at`` is kept in the output list.

    Inputs are sorted ascending by ``created_at`` (the caller's sort order),
    so iterating from the end toward the start, building a kept-list, gives
    the newest facts priority: a fact is kept iff none of the already-kept
    (newer) facts is similar to it.

    Embeddings are read from each fact's ``embedding`` field — populated at
    ``add()`` time and/or by ``backfill_embeddings()``. Facts missing a
    vector get re-embedded via ``embed_chunks`` (batched) and the result is
    not persisted here — ``add()`` is the only writer. On any embed failure
    the caller falls back to the un-deduplicated list — we surface that as
    an exception so the wrapper can log + return the original ``facts``
    untouched.
    """
    if not facts:
        return facts

    texts = [(f.get("text") or "").strip() for f in facts]
    if not any(texts):
        return facts
    # Index: where each fact's vector came from. Pre-computed vectors (from
    # the doc's `embedding` field) cost 0 API calls; recomputed vectors
    # cost one batched call covering all missing slots.
    vectors = [list(f.get("embedding") or []) for f in facts]
    missing_idx = [i for i, v in enumerate(vectors) if not v]
    if missing_idx:
        # Lazy import to keep distilled_fact_store importable in tests that
        # don't exercise the embed path (avoids the OPENAI_API_KEY requirement
        # at module load).
        from src.bot.services.embedder import embed_chunks
        try:
            recalced = embed_chunks([texts[i] for i in missing_idx])
        except Exception as e:
            logger.warning(
                "distilled_facts_dedup_embed_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError("embed_chunks raised") from e
        if not recalced or all(not v for v in recalced):
            raise RuntimeError("embed_chunks returned no vectors")
        for idx, vec in zip(missing_idx, recalced):
            if vec:
                vectors[idx] = vec
    # Greedy from newest (end of ascending list) to oldest.
    kept: List[dict] = []
    kept_vecs: List[List[float]] = []
    for i in range(len(facts) - 1, -1, -1):
        v = vectors[i] if i < len(vectors) else []
        if not v:
            # No embedding for this fact — keep it (can't compare), do not
            # block the dedup pass for the rest.
            kept.append(facts[i])
            kept_vecs.append([])
            continue
        dup = False
        for kv in kept_vecs:
            if not kv:
                continue
            if _cosine(v, kv) >= threshold:
                dup = True
                break
        if not dup:
            kept.append(facts[i])
            kept_vecs.append(v)
    # Reverse to preserve original ascending created_at order in the output.
    kept.reverse()
    return kept


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
        """Insert one fact with its BGE-m3 embedding embedded in the doc
        (so the injector can dedup at zero API cost per turn). Returns True
        if newly inserted, False on duplicate (same ``text`` already exists).
        All other errors are swallowed and return False so a Mongo blip
        doesn't crash the consolidator — embedding failures still allow
        the insert (with an empty ``embedding`` field) so dedup falls back
        to recompute-then-cache on the first inject that touches that fact.
        """
        if not text or not text.strip():
            return False
        clean = text.strip()
        embedding = None
        try:
            # Lazy import — keeps test imports free of the OPENAI_API_KEY
            # requirement at module load.
            from src.bot.services.embedder import embed_text
            embedding = embed_text(clean) or None
        except Exception as e:
            logger.warning(
                "distilled_fact_embed_failed",
                error=str(e),
                error_type=type(e).__name__,
                fact_preview=clean[:80],
            )
        try:
            self.facts.insert_one(
                {
                    "text": clean,
                    "embedding": embedding,
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

    def backfill_embeddings(self, batch_size: int = 64) -> dict:
        """One-shot backfill: for every fact doc missing an ``embedding``,
        embed its text and write the vector back. Idempotent — facts that
        already have an ``embedding`` (truthy non-empty list) are skipped.

        Run this ONCE after deploying v2.3.44 (e.g. from a one-off Python
        shell inside the container). Returns ``{"scanned", "embedded",
        "errors"}``.

        The unique index on ``text`` means same-text facts collapse at
        insert time, so most fact docs are unique — but a fact inserted
        before embeddings existed will have ``embedding: None`` and needs
        this pass to get one. Subsequent ``add()`` calls always embed.
        """
        if batch_size < 1:
            batch_size = 1
        from src.bot.services.embedder import embed_chunks
        scanned = 0
        embedded = 0
        errors = 0
        cursor = self.facts.find(
            {"$or": [{"embedding": None}, {"embedding": {"$exists": False}}]},
            {"_id": 1, "text": 1},
        )
        batch: list = []
        for doc in cursor:
            scanned += 1
            batch.append(doc)
            if len(batch) >= batch_size:
                e = self._embed_and_update_batch(batch)
                embedded += e
                errors += len(batch) - e
                batch.clear()
        if batch:
            e = self._embed_and_update_batch(batch)
            embedded += e
            errors += len(batch) - e
        return {"scanned": scanned, "embedded": embedded, "errors": errors}

    def _embed_and_update_batch(self, batch: list) -> int:
        """Embed one batch of fact docs and write embeddings back. Returns
        number of docs successfully written — the difference between
        ``len(batch)`` and the return value is the error count. Used by
        ``backfill_embeddings``."""
        try:
            texts = [d.get("text") or "" for d in batch]
            vecs = embed_chunks(texts)
        except Exception as e:
            logger.warning(
                "distilled_facts_backfill_embed_failed",
                error=str(e),
                error_type=type(e).__name__,
                batch_size=len(batch),
            )
            return 0
        written = 0
        for doc, vec in zip(batch, vecs):
            if not vec:
                continue
            try:
                self.facts.update_one({"_id": doc["_id"]}, {"$set": {"embedding": vec}})
                written += 1
            except Exception as e:
                logger.warning(
                    "distilled_facts_backfill_write_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    doc_id=str(doc.get("_id")),
                )
        return written

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
        """Return facts from the latest run per topic, across all topics,
        with near-duplicate facts deduplicated by cosine similarity.

        Sync — called from the sync ``_build_holmes_messages`` path so the
        prompt is assembled without an extra thread hop. Returns ``[]`` on
        any error so the bot keeps working without injected facts this
        turn. Documents are projected to ``{text, source_topic_id,
        source_run_id, created_at}`` for the caller.

        Algorithm:
          1. Read all latest-run pointers from ``distilled_runs`` (one doc
             per topic), then fetch facts whose ``source_run_id`` is in
             that set. Older runs' facts stay in ``distilled_facts`` but
             are filtered out here — this is the "newer run wins per topic"
             filter.
          2. Optionally dedup the surviving facts via cosine similarity on
             their text embeddings (``distilled_dedup_cosine`` threshold).
             Two facts above the threshold are treated as duplicates; the
             newest one (per ``created_at``) wins. Dedup failures (embed
             API down, threshold disabled) fall back to returning the
             un-deduplicated list — never break the prompt.
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
                        "embedding": 1,
                        "source_topic_id": 1,
                        "source_run_id": 1,
                        "created_at": 1,
                    },
                )
                .sort("created_at", ASCENDING)
            )
            facts = list(cursor)
        except Exception as e:
            logger.warning(
                "distilled_facts_get_injectable_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

        # Dedup pass — fails open (return un-deduplicated list) if the embed
        # API is unavailable. Failing open means the prompt may carry near-
        # duplicates for one turn; never breaks the bot.
        threshold = getattr(bot_config, "distilled_dedup_cosine", 0.0) or 0.0
        if not facts or threshold <= 0.0:
            return facts
        try:
            deduped = _dedup_facts_by_cosine(facts, threshold)
            if len(deduped) < len(facts):
                logger.info(
                    "distilled_facts_deduped",
                    before=len(facts),
                    after=len(deduped),
                    threshold=threshold,
                )
            return deduped
        except Exception as e:
            logger.warning(
                "distilled_facts_dedup_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return facts
