"""KBStore — in-memory rule index + frequency-weighted cosine retrieval.

Backed by MongoDB collection `kb_rules` (one doc per distilled if-then
rule, populated by `scripts/ingest_kb_docx.py`). On first use, all
rule docs are loaded into memory as `{rule_id: {rule, vector}}` so query-
time cosine is pure Python without per-call DB hits. Rules themselves
are short (one line each) so the index fits comfortably in memory for
~1.3k rules × 1024 floats per vector (~5 MB).

Score formula:
    score = cosine_sim * (1 + log(1 + match_count))

The log1p caps the frequency boost so 0 matches → no boost, 10 → ~2.4x,
100 → ~4.6x. Without it, a single very-frequent rule would dominate
regardless of cosine relevance.

The match_count increment is **fire-and-forget** from the caller's
perspective — `/kb_match` returns the rules block immediately, then
schedules `record_matches` as a background task so DB write latency
never blocks the snapshot tool's curl.
"""
import math
import re
from datetime import datetime
from typing import List, Optional

import structlog
from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection

from src.bot.services.embedder import embed_chunks
from src.bot.utils.config import config as bot_config
from src.bot.utils.mongodb import MongoDB

logger = structlog.get_logger(__name__)

# Cosine below this floor → skip regardless of match_count. Tune via the
# KB_MIN_COSINE env follow-up. 0.15 is a conservative default for BGE-m3
# on Vietnamese text (which tends to score higher baseline cosine).
_MIN_COSINE = 0.15

# Per-chunk char cap for the snapshot query. 6k chars ≈ 2k tokens for
# BGE-m3's 8192-token cap — leaves 4x headroom since most snapshot
# sections (events, failed pods) are well under this even with verbose
# Kubernetes event messages.
_MAX_CHARS_PER_CHUNK = 6000

# Section markers used by the snapshot tool. They look like
# "===== 4. EVENTS =====" — keep the regex loose to match any number.
_SECTION_RE = re.compile(r"^=====\s*\d+\..*?=====\s*$", re.MULTILINE)


class KBStore:
    KB_COLLECTION = "kb_rules"

    def __init__(self) -> None:
        self._collection: Optional[Collection] = None
        self._loaded: bool = False
        # {rule_id: {"rule": doc, "vector": list[float]}}
        self._index: dict = {}

    # ------------------------------------------------------------------ #
    # Lazy MongoDB collection (mirror DistilledFactStore pattern)
    # ------------------------------------------------------------------ #

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self._collection = MongoDB.get_collection(self.KB_COLLECTION)
            try:
                self._collection.create_index([("rule_id", ASCENDING)], unique=True)
                self._collection.create_index([("issue_id", ASCENDING)])
                self._collection.create_index([("match_count", DESCENDING)])
            except Exception as e:
                logger.warning("kb_rules_index_failed", error=str(e))
        return self._collection

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            for doc in self.collection.find({}, {"_id": 0}):
                emb = doc.get("embedding")
                if not emb:
                    continue
                rid = doc.get("rule_id")
                if not rid:
                    continue
                self._index[rid] = {"rule": doc, "vector": emb}
            self._loaded = True
            logger.info("kb_rules_loaded", count=len(self._index))
        except Exception as e:
            logger.warning("kb_rules_load_failed", error=str(e))
            # Mark loaded even on failure so we don't retry forever — the
            # bot will see an empty index this turn but next restart (or
            # a manual reload) re-populates if Mongo recovers.
            self._loaded = True

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def get_relevant_rules(self, query_text: str, top_k: int = 0) -> List[dict]:
        """Return top-K rules for `query_text`. Score = cosine * (1 + log1p(match_count)).

        Returns [] on any failure (disabled, empty index, embed error).
        Does NOT mutate match_count — the caller (kb_match_handler) does
        that fire-and-forget after sending the response.
        """
        if not bot_config.kb_enabled or not query_text or not query_text.strip():
            return []
        self._ensure_loaded()
        if not self._index:
            return []
        k = top_k if top_k and top_k > 0 else bot_config.kb_top_k
        chunks = _chunk_at_section_markers(query_text, _MAX_CHARS_PER_CHUNK)
        try:
            q_vecs = embed_chunks(chunks)
        except Exception as e:
            logger.warning("kb_query_embed_failed", error=str(e))
            return []
        # Filter out empty vecs from dropped-empty-string inputs.
        q_vecs = [v for v in q_vecs if v]
        if not q_vecs:
            return []
        scored = []
        for rule_id, entry in self._index.items():
            v = entry["vector"]
            best_cos = 0.0
            for qv in q_vecs:
                s = _cosine(qv, v)
                if s > best_cos:
                    best_cos = s
            match_count = int(entry["rule"].get("match_count", 0) or 0)
            score = best_cos * (1.0 + math.log1p(match_count))
            scored.append((score, best_cos, match_count, rule_id, entry["rule"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, cos, mc, rule_id, rule in scored[:k]:
            if cos < _MIN_COSINE:
                continue
            out.append({
                "rule_id": rule_id,
                "issue_id": rule.get("issue_id"),
                "issue_title": rule.get("issue_title"),
                "rule_text": rule.get("rule_text"),
                "match_count": mc,
                "score": float(score),
            })
        return out

    def record_matches(self, rule_ids: List[str]) -> None:
        """Sync fire-and-forget increment. Swallows DB errors — stats
        tracking must never break the bot.
        """
        if not rule_ids:
            return
        try:
            now = datetime.utcnow()
            self.collection.update_many(
                {"rule_id": {"$in": rule_ids}},
                {"$inc": {"match_count": 1}, "$set": {"last_matched_at": now}},
            )
        except Exception as e:
            logger.warning("kb_rules_match_count_failed", error=str(e))

    def top_by_match_count(self, limit: int = 20) -> List[dict]:
        """For /kb_stats admin endpoint. Top-N rules by match_count desc."""
        try:
            return list(self.collection.find(
                {}, {"_id": 0, "rule_id": 1, "issue_id": 1, "issue_title": 1,
                     "rule_text": 1, "match_count": 1, "last_matched_at": 1}
            ).sort("match_count", DESCENDING).limit(limit))
        except Exception as e:
            logger.warning("kb_rules_top_failed", error=str(e))
            return []


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _chunk_at_section_markers(text: str, max_chars: int) -> List[str]:
    """Split the snapshot output on `===== N. ... =====` section markers,
    then merge short chunks up to `max_chars` so we don't pay per-chunk
    embed cost for tiny trailing sections like StatefulSets list.
    """
    if not text:
        return []
    parts = _SECTION_RE.split(text)
    chunks: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if chunks and len(chunks[-1]) + len(part) + 1 <= max_chars:
            chunks[-1] = chunks[-1] + "\n" + part
        else:
            chunks.append(part[:max_chars])
    # Edge case: no section markers at all → treat the entire text as one
    # chunk (capped by max_chars upstream in the embedder).
    if not chunks:
        chunks = [text[:max_chars]]
    return chunks


def _cosine(a: List[float], b: List[float]) -> float:
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


def format_rules_block(rules: List[dict]) -> str:
    """Format the top-K rules as the block Holmes sees at the end of the
    snapshot tool output. Compact (one line per rule) so the entire block
    is at most ~20 lines for default top_k=20.
    """
    n = len(rules)
    if n == 0:
        return "===== KB RULES (empty) =====\n===== END KB RULES ====="
    lines = [f"===== KB RULES (top-{n}, distilled + frequency-weighted) ====="]
    for r in rules:
        rid = r.get("rule_id", "?")
        mc = r.get("match_count", 0)
        issue = r.get("issue_id", "")
        title = r.get("issue_title", "")
        text = r.get("rule_text", "").strip()
        # Truncate a runaway rule line. 600 chars ≈ ~200 tokens, well
        # under any per-line cap even if an LLM distilled a verbose rule.
        if len(text) > 600:
            text = text[:600].rstrip() + " ..."
        lines.append(f"- [{rid} matches={mc}] {text} (from {issue} {title})")
    lines.append("===== END KB RULES =====")
    return "\n".join(lines)
