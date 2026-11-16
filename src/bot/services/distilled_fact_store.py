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
import threading
import time
from typing import Dict, List, Optional, Tuple

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
        # In-memory snapshot of the deduplicated injectable facts —
        # refreshed by a background thread (see `_refresh_loop`). Each
        # `get_injectable_facts()` call reads this snapshot in O(1) so the
        # per-request prompt build never pays the cosine dedup cost
        # (was O(N²×1024) Python pure — multi-second for ~1k facts).
        self._injectable_cache: List[dict] = []
        self._cache_lock = threading.Lock()
        # Signaling refresh thread — `record_run` pokes this to skip the
        # wait and rebuild the snapshot immediately after a new run lands.
        self._refresh_event = threading.Event()
        # Refresh interval — defaults to 60s; 0 disables the background
        # loop and falls back to "recompute per call" (legacy behavior).
        self._refresh_seconds: float = float(
            getattr(bot_config, "distilled_dedup_cache_seconds", 60) or 0
        )
        self._refresh_thread: Optional[threading.Thread] = None
        self._started = False

    def start_background_refresh(self) -> None:
        """Launch the background refresh thread. Idempotent. Called from
        `main.py` after the DistilledFactStore is constructed — no-op when
        `_refresh_seconds` is 0 (config disables background refresh).
        First call does an immediate sync rebuild so the first user turn
        after bot startup doesn't stall waiting on the cache."""
        if self._started or self._refresh_seconds <= 0:
            return
        self._started = True
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            name="distilled-facts-refresh",
            daemon=True,
        )
        self._refresh_thread.start()
        logger.info(
            "distilled_facts_refresh_started",
            refresh_seconds=self._refresh_seconds,
        )

    def stop_background_refresh(self) -> None:
        """Signal the background thread to stop on next loop tick. Used
        by tests / graceful shutdown — the thread is daemon so a hard
        process exit will kill it anyway. Idempotent."""
        self._started = False
        self._refresh_event.set()

    def _refresh_loop(self) -> None:
        """Background dedup pass: rebuild the injectable snapshot,
        sleep `_refresh_seconds`, repeat. Early-wake on
        `_refresh_event.set()` triggered by `record_run` so a new
        distillation is reflected within seconds rather than the full
        interval. Failures during a single refresh are logged and waited
        out — the loop must keep trying so a blip doesn't permanently
        leave the bot with stale or empty facts.
        """
        # First pass — synchronous-feel so the first user turn doesn't
        # hit an empty cache.
        try:
            self._refresh_snapshot()
        except Exception as e:
            logger.warning(
                "distilled_facts_refresh_initial_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
        while self._started:
            # Reset BEFORE sleeping so a record_run poke arriving during
            # sleep wakes us promptly.
            self._refresh_event.clear()
            # Sleep in small slices so stop_background_refresh takes effect
            # quickly rather than waiting for the full interval.
            slept = 0.0
            while self._started and slept < self._refresh_seconds:
                if self._refresh_event.wait(timeout=1.0):
                    break
                slept += 1.0
            try:
                self._refresh_snapshot()
            except Exception as e:
                logger.warning(
                    "distilled_facts_refresh_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )

    def _refresh_snapshot(self) -> None:
        """Rebuild `_injectable_cache` from MongoDB + cosine dedup + scoring.

        Pipeline:
          1. latest-run-per-topic filter (existing)
          2. cosine dedup at `distilled_dedup_cosine` (existing)
          3. score facts against matched kb_rules (kb_rules with match_count > 0)
          4. greedy slice to fit a dynamic 3K-8K token budget

        Writes the resulting top-K subset atomically under `_cache_lock`.
        """
        full = self._fetch_injectable_facts_raw(force_recompute=True)
        # Score against matched kb_rules.
        matched_kb = self._load_matched_kb_rules()
        scored = self._score_facts_against_kb_rules(full, matched_kb)
        budget = self._compute_token_budget(len(matched_kb))
        sliced = self._ranked_slice_within_budget(scored, budget)

        if len(sliced) < len(full):
            logger.info(
                "distilled_facts_ranked_injectable",
                orig_count=len(full),
                inject_count=len(sliced),
                matched_kb_rules=len(matched_kb),
                token_budget=budget,
            )
        with self._cache_lock:
            self._injectable_cache = sliced

    def _fetch_injectable_facts_raw(self, force_recompute: bool = False) -> List[dict]:
        """The original inject path — read latest-run facts, dedup by
        cosine, return the list. Called either from `_refresh_snapshot`
        (background thread, every N seconds) or from
        `get_injectable_facts` (fallback path when background refresh
        is disabled OR cache is empty on first call)."""
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

    # ------------------------------------------------------------------ #
    # Ranked slicing — top-K facts fitting a token budget, scored against
    # kb_rules with match_count > 0 as a trust anchor.
    # ------------------------------------------------------------------ #

    # Cache of matched-kb-rules' embeddings: list of (rule_id, match_count,
    # embedding_np). Refreshed on demand — checked against current matched
    # count each call and re-pulled when new rules crossed match_count > 0.
    _matched_kb_cache: Optional[List[dict]] = None
    _matched_kb_cache_count: int = -1
    _kb_cache_lock = threading.Lock()

    def _load_matched_kb_rules(self) -> List[dict]:
        """Return all kb_rules with match_count > 0, with their embeddings.
        Cached on the store instance; re-pulled when the count changes.

        Each dict shape: {rule_id, match_count, issue_title, embedding_np}.
        We pull embedding_np as float32 numpy array for vectorized cosine.
        """
        try:
            kb_col = MongoDB.get_collection("kb_rules")
            count = kb_col.count_documents({"match_count": {"$gt": 0}})
        except Exception as e:
            logger.warning(
                "distilled_facts_kb_count_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

        with self._kb_cache_lock:
            if self._matched_kb_cache is not None and self._matched_kb_cache_count == count:
                return self._matched_kb_cache
        try:
            cursor = kb_col.find(
                {"match_count": {"$gt": 0}},
                {"_id": 0, "rule_id": 1, "match_count": 1, "issue_title": 1, "embedding": 1},
            )
            rules = list(cursor)
        except Exception as e:
            logger.warning(
                "distilled_facts_kb_load_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return []
        prepared: List[dict] = []
        for r in rules:
            emb = r.get("embedding") or []
            if not emb:
                continue
            try:
                import numpy as np
                vec = np.asarray(emb, dtype="float32")
                # L2 normalize for cosine via dot product.
                norm = float(np.linalg.norm(vec))
                if norm <= 0:
                    continue
                vec = vec / norm
            except Exception:
                continue
            prepared.append({
                "rule_id": r.get("rule_id"),
                "match_count": int(r.get("match_count") or 0),
                "issue_title": r.get("issue_title") or "",
                "embedding": vec,
            })
        with self._kb_cache_lock:
            self._matched_kb_cache = prepared
            self._matched_kb_cache_count = count
        return prepared

    @staticmethod
    def _fact_embedding_np(fact: dict):
        """Return L2-normalized numpy float32 embedding of `fact`, or None
        if missing / zero. Used for vectorized cosine scoring."""
        emb = fact.get("embedding") or []
        if not emb:
            return None
        try:
            import numpy as np
            vec = np.asarray(emb, dtype="float32")
            norm = float(np.linalg.norm(vec))
            if norm <= 0:
                return None
            return vec / norm
        except Exception:
            return None

    def _time_decay_penalty(self, fact: dict) -> float:
        """Return cumulative time-decay penalty for `fact` (negative score
        contribution). Anchor is max(last_referenced_at, created_at) — the
        fact's most recent activity. Defaults: -0.05 if age>90d, -0.10 if
        >180d, -0.15 if >365d (cumulative steps). Returns 0 if no anchor
        field can be parsed. Cosine still dominates: max penalty 0.30 <
        cosine range 0-1.
        """
        decay_90d = float(getattr(bot_config, "distilled_inject_decay_90d", 0.05))
        decay_180d = float(getattr(bot_config, "distilled_inject_decay_180d", 0.10))
        decay_365d = float(getattr(bot_config, "distilled_inject_decay_365d", 0.15))
        if not fact:
            return 0.0
        lr = fact.get("last_referenced_at")
        ca = fact.get("created_at")
        # Use whichever is later — a fact referenced recently stays fresh
        # even if created long ago.
        candidates = [v for v in (lr, ca) if v is not None]
        if not candidates:
            return 0.0
        try:
            anchors = []
            for v in candidates:
                if isinstance(v, datetime.datetime):
                    anchors.append(v)
                else:
                    import datetime as _dt
                    anchors.append(_dt.datetime.fromisoformat(str(v)))
            anchor = max(anchors)
        except Exception:
            return 0.0
        now = datetime.datetime.utcnow()
        age_days = (now - anchor).total_seconds() / 86400.0
        penalty = 0.0
        if age_days > 365:
            penalty += decay_365d
        if age_days > 180:
            penalty += decay_180d
        if age_days > 90:
            penalty += decay_90d
        return penalty

    def _score_facts_against_kb_rules(
        self, facts: List[dict], matched_kb: List[dict],
    ) -> List[Tuple[dict, float]]:
        """Score each fact by overlap with matched kb_rules.

        Returns a list of (fact, score) tuples sorted desc by score. Empty
        `matched_kb` → all scores 0 (returned in original order, no crash).

        Formula (per plan):
            score = cosine_max_vs_matched_kb_rules
                  + freq_bonus * 1_if_nearest_kb_match_count_>=_frequent_threshold
                  + cross_topic_bonus * min(distinct_run_count, cross_topic_cap)
                  - time_decay_penalty (additive steps, anchored on
                    max(last_referenced_at, created_at))

        Vectorized via numpy single matmul (facts_matrix @ kb_matrix.T).
        Penalty steps (defaults): -0.05 if age>90d, -0.10 if >180d,
        -0.15 if >365d. Cumulative — a 400d-old fact gets all three.
        Cosine still dominates (max penalty 0.15 << cosine range 0-1).
        """
        if not facts:
            return []
        import numpy as np

        # Filter facts with valid embeddings; keep index map to original list.
        scored: List[Tuple[dict, float]] = []
        if not matched_kb:
            for f in facts:
                # Even with no kb anchor, apply time-decay so stale facts
                # fall behind fresh ones in the no-trust-anchor case. Cosine
                # term is 0 here, so penalty is the entire score (negative).
                scored.append((f, -self._time_decay_penalty(f)))
            # Sort desc so newer (penalty closer to 0) comes first.
            scored.sort(key=lambda x: -x[1])
            return scored

        # Build fact matrix (n_facts x dim).
        fact_vecs: List = []
        fact_idx_to_orig: List[int] = []
        for i, f in enumerate(facts):
            v = self._fact_embedding_np(f)
            if v is None:
                # No embedding — score 0, keep stable ordering.
                scored.append((f, 0.0))
                continue
            fact_vecs.append(v)
            fact_idx_to_orig.append(i)

        if not fact_vecs:
            return scored

        # Build kb matrix (n_kb x dim).
        kb_dim = matched_kb[0]["embedding"].shape[0]
        fact_mat = np.vstack(fact_vecs).astype("float32")  # (F, D)
        kb_mat = np.vstack([r["embedding"] for r in matched_kb]).astype("float32")
        # Ensure dims match. If they don't, treat as 0 score (model mismatch).
        if fact_mat.shape[1] != kb_mat.shape[1]:
            logger.warning(
                "distilled_facts_kb_dim_mismatch",
                fact_dim=int(fact_mat.shape[1]),
                kb_dim=int(kb_mat.shape[1]),
            )
            for i in fact_idx_to_orig:
                scored.append((facts[i], 0.0))
            return scored

        # Cosine via dot product (both matrices L2-normalized).
        sim = fact_mat @ kb_mat.T  # (F, K)
        # Track best sim per fact AND the match_count of the closest kb_rule.
        max_idx = np.argmax(sim, axis=1)
        max_sim = sim[np.arange(sim.shape[0]), max_idx]

        freq_bonus = float(getattr(bot_config, "distilled_inject_freq_bonus", 0.10))
        cross_topic_bonus = float(getattr(bot_config, "distilled_inject_cross_topic_bonus", 0.05))
        cross_topic_cap = int(getattr(bot_config, "distilled_inject_cross_topic_cap", 5))
        frequent_threshold = int(getattr(bot_config, "distilled_inject_frequent_match_count", 10))

        # Append scored facts in original facts list order.
        scored_by_idx: Dict[int, float] = {}
        for j, orig_i in enumerate(fact_idx_to_orig):
            mc = int(matched_kb[int(max_idx[j])].get("match_count") or 0)
            score = float(max_sim[j])
            if mc >= frequent_threshold:
                score += freq_bonus
            # distinct_run_count ≈ 1 (single fact = single source). Min with cap.
            score += cross_topic_bonus * min(1, cross_topic_cap)
            score -= self._time_decay_penalty(facts[orig_i])
            scored_by_idx[orig_i] = score

        for i, f in enumerate(facts):
            scored.append((f, scored_by_idx.get(i, 0.0)))

        # Sort desc; ties break by original order (stable sort).
        scored.sort(key=lambda x: -x[1])
        return scored

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for `text` via litellm. Cached on the doc
        via a private sentinel — re-calling on same `text` returns without
        re-counting. Falls back to len(text) // 4 if litellm unavailable."""
        if not text:
            return 0
        # Lightweight heuristic ~chars/4. Real litellm token count is more
        # accurate but adds ~5-10ms per call — acceptable inside the
        # background thread, not on the hot read path. The cap is enforced
        # by the slice call, so eps here is fine; we resort to litellm if
        # caller asks for a precise count (search() path doesn't, since
        # it returns ranked fact list, not a token-budgeted block).
        try:
            import litellm
            return int(litellm.token_counter(
                model="gpt-4o",
                messages=[{"role": "system", "content": text}],
            ))
        except Exception:
            return max(1, len(text) // 4)

    def _ranked_slice_within_budget(
        self,
        scored: List[Tuple[dict, float]],
        token_budget: int,
    ) -> List[dict]:
        """Greedy fill by score desc until cumulative tokens exceed budget.

        Returns the ordered list of fact dicts. Token count per fact is
        memoized on the dict under a private key to avoid recounting across
        calls. Empty / oversized facts are skipped (a single fact > budget
        is skipped to keep behavior sane)."""
        if not scored or token_budget <= 0:
            return []
        out: List[dict] = []
        running = 0
        for fact, score in scored:
            text = fact.get("text", "") or ""
            # Memoize on dict — sets a private key; harmless if the doc
            # later gets re-used by callers trusting the same key.
            tok = fact.get("_scored_tokens")
            if tok is None:
                tok = self._estimate_tokens(text)
                try:
                    fact["_scored_tokens"] = int(tok)
                except Exception:
                    pass
            tok = int(tok)
            if running + tok > token_budget:
                continue
            out.append(fact)
            running += tok
        return out

    def _compute_token_budget(self, matched_kb_count: int) -> int:
        """Dynamic budget 3K-8K based on matched-kb-rules count.

        - 0 matched → min_tokens (always-fill floor; still inject a curated
          set, ranked purely on cross-topic bonus when kb is empty)
        - 1..high_watermark → linear interp min→mid
        - high_watermark..max_watermark → linear interp mid→max
        - >= max_watermark → max_tokens
        """
        mn = int(getattr(bot_config, "distilled_inject_min_tokens", 3000))
        mx = int(getattr(bot_config, "distilled_inject_max_tokens", 8000))
        hi_wm = int(getattr(bot_config, "distilled_inject_high_watermark_kb_match", 30))
        max_wm = int(getattr(bot_config, "distilled_inject_max_watermark_kb_match", 60))
        if hi_wm < 1:
            hi_wm = 1
        if max_wm <= hi_wm:
            max_wm = hi_wm + 1
        if matched_kb_count <= 0:
            return mn
        mid = (mn + mx) // 2
        if matched_kb_count >= max_wm:
            return mx
        if matched_kb_count >= hi_wm:
            # Linear from mid to max between hi_wm and max_wm.
            ratio = (matched_kb_count - hi_wm) / max(1, (max_wm - hi_wm))
            return int(mid + (mx - mid) * ratio)
        # Below hi_wm: linear from min to mid.
        ratio = matched_kb_count / max(1, hi_wm)
        return int(mn + (mid - mn) * ratio)


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
            now = datetime.datetime.utcnow()
            self.facts.insert_one(
                {
                    "text": clean,
                    "embedding": embedding,
                    "source_chat_id": source_chat_id,
                    "source_topic_id": source_topic_id,
                    "source_run_id": source_run_id,
                    "created_at": now,
                    # Seed last_referenced_at = created_at so time-decay
                    # penalty in _score_facts_against_kb_rules has an
                    # anchor before the fact is ever returned by search().
                    # Updated by search() each time the fact is returned.
                    "last_referenced_at": now,
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
            from src.bot.services.embedder import embed_chunks
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
            return
        # Poke the background refresh thread so a fresh distillation is
        # reflected in the injectable cache within seconds rather than
        # waiting for the next `_refresh_seconds` tick. No-op when the
        # thread isn't running (the cache will refresh on its next tick
        # OR via the cold-start path on the next request).
        self._refresh_event.set()

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
        cosine-dedupeed and *scored* (top-K subset fitting a dynamic 3K-8K
        token budget).

        Sync — called from the sync ``_build_holmes_messages`` path per
        Holmes turn. **Hot path**: reads straight from an in-memory cache
        populated by ``_refresh_snapshot()`` (background thread every
        ``distilled_dedup_cache_seconds``, invalidated on ``record_run()``).
        O(1) for the caller — no DB read, no cosine dedup, no scoring. When
        the cache is empty (first call after bot startup before the refresh
        thread has run, OR background refresh disabled) it falls back to a
        synchronous pipeline so the prompt is never empty if Mongo works.
        """
        # Fast path: serve cached snapshot.
        with self._cache_lock:
            snapshot = self._injectable_cache
        if snapshot:
            return snapshot
        # Cold start — run the full pipeline synchronously ONCE so the
        # first user turn isn't empty AND gets the scoring + budget cap
        # applied (not just the dedup pass). If background refresh is
        # enabled, this is the same work the refresh thread is about to
        # do; the lock-guarded write below ensures we don't race.
        if self._refresh_seconds <= 0:
            # Background refresh disabled — always compute synchronously.
            return self._compute_ranked_injectable()
        # Background refresh IS enabled but cache is empty — likely the
        # very first turn before the refresh thread has finished its
        # initial snapshot. Compute once and seed the cache so subsequent
        # turns hit the fast path.
        computed = self._compute_ranked_injectable()
        with self._cache_lock:
            if not self._injectable_cache:
                self._injectable_cache = computed
        return computed

    def _compute_ranked_injectable(self) -> List[dict]:
        """Fresh synchronous pipeline: dedup + score vs matched kb_rules +
        slice within token budget. Returns the top-K subset to inject."""
        full = self._fetch_injectable_facts_raw(force_recompute=False)
        matched_kb = self._load_matched_kb_rules()
        scored = self._score_facts_against_kb_rules(full, matched_kb)
        budget = self._compute_token_budget(len(matched_kb))
        return self._ranked_slice_within_budget(scored, budget)

    def search(self, query: str, k: int = 50) -> List[dict]:
        """Cosine-rank distilled facts against `query` and return top-k.

        Iterates over the dedup-filtered facts (same set as injectable, but
        IGNORES scoring vs kb_rules — query is the trust signal here). Each
        result dict carries: text, score (cosine to query), and source
        metadata (source_run_id, source_chat_id, source_topic_id).

        Used by the search_memory tool's on-demand retrieval. Sync — runs
        in Holmes worker thread (caller is search_memory tool's sync _invoke).

        Falls back to injectable cache if a fresh dedup is too expensive
        (no Mongo, no embedder) — returns empty list on any failure to
        keep search_memory resilient.
        """
        if not query or not query.strip():
            return []
        if k <= 0:
            k = 50
        try:
            from src.bot.services.embedder import embed_text
            q_vec = embed_text(query)
        except Exception as e:
            logger.warning(
                "distilled_facts_search_embed_failed",
                error=str(e),
                error_type=type(e).__name__,
                query_preview=query[:60],
            )
            return []
        if not q_vec:
            return []
        try:
            import numpy as np
            q = np.asarray(q_vec, dtype="float32")
            norm = float(np.linalg.norm(q))
            if norm <= 0:
                return []
            q = q / norm
        except Exception:
            return []

        # Use the cached injectable facts if available; otherwise compute
        # a fresh dedup snapshot (no scoring, no slicing).
        with self._cache_lock:
            cached: List[dict] = self._injectable_cache
        # NOTE: cached slice may have been pruned to top-K; for search we
        # want ALL dedup'd facts so the LLM can find facts not in the top-K.
        # Compute fresh — _fetch_injectable_facts_raw is cheap (DB read +
        # cosine dedup, runs <50ms for ~1k facts cached in-memory).
        try:
            full = self._fetch_injectable_facts_raw(force_recompute=False)
        except Exception as e:
            logger.warning(
                "distilled_facts_search_list_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

        scored: List[Tuple[dict, float]] = []
        for f in full:
            v = self._fact_embedding_np(f)
            if v is None:
                continue
            sim = float(q @ v)
            scored.append((f, sim))
        scored.sort(key=lambda x: -x[1])

        out: List[dict] = []
        for f, score in scored[:k]:
            out.append({
                "text": f.get("text", "") or "",
                "score": float(score),
                "source_run_id": f.get("source_run_id"),
                "source_chat_id": f.get("source_chat_id"),
                "source_topic_id": f.get("source_topic_id"),
            })
        # Fire-and-forget: refresh last_referenced_at on the returned facts.
        # Time-decay scoring in _score_facts_against_kb_rules reads this
        # field to down-rank stale facts. Bumping only the top-K returned
        # (not all 282) keeps write amplification low.
        if out:
            texts = [r["text"] for r in out if r.get("text")]
            self._touch_last_referenced(texts)
        return out

    def _touch_last_referenced(self, texts: List[str]) -> None:
        """Bump last_referenced_at=now on the docs matching `texts`. Fire-
        and-forget — swallows DB errors. Called from search() after
        ranking+limiting so write cost scales with k, not with corpus size.
        """
        if not texts:
            return
        try:
            now = datetime.datetime.utcnow()
            self.facts.update_many(
                {"text": {"$in": texts}},
                {"$set": {"last_referenced_at": now}},
            )
        except Exception as e:
            logger.warning(
                "distilled_facts_touch_last_referenced_failed",
                error=str(e),
                error_type=type(e).__name__,
                count=len(texts),
            )

