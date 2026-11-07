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
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Optional, Dict

import structlog
from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection

from src.bot.services.embedder import embed_chunks
from src.bot.utils.config import config as bot_config
from src.bot.utils.mongodb import MongoDB

logger = structlog.get_logger(__name__)

# Cosine below this floor → skip UNLESS BM25 carries the rule. Tune via
# KB_MIN_COSINE env follow-up. 0.15 is conservative for BGE-m3 on Vietnamese.
_MIN_COSINE = 0.15
# BM25 below this floor → ignore. With Okapi IDF on a ~1300-rule corpus,
# scoring real keyword matches clears 1.0 easily; 0.1 just filters zero-
# overlap noise while letting the small-corpus test fixtures match.
_MIN_BM25 = 0.1
# Per-chunk char cap for the snapshot query. 6k chars ≈ 2k tokens for
# BGE-m3's 8192-token cap — leaves 4x headroom.
_MAX_CHARS_PER_CHUNK = 6000
# RRF constant from the IR literature (Cormack et al. 2009). Standard 60 —
# damps the contribution of top-ranked items so a #1 cosine with #50 bm25
# doesn't dominate a #2 cosine with #1 bm25.
_K_RRF = 60

# Section markers used by the snapshot tool. They look like
# "===== 4. EVENTS =====" — keep the regex loose to match any number.
_SECTION_RE = re.compile(r"^=====\s*\d+\..*?=====\s*$", re.MULTILINE)

# Tokenizer: keep hyphens (etcd-main-0, kube-apiserver), underscore, and
# Latin-ext range so Vietnamese diacritics stay attached. Single chars + 2-letter
# rule syntax (IF, x, y, z) get filtered to avoid noise.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_À-ɏ][A-Za-z0-9_À-ɏ-]*", re.UNICODE)

# Minimal stopword set — Vietnamese + English. Don't be aggressive: in
# technical KB matching, "error" / "fail" carry signal, "and"/"the" don't.
_STOPWORDS = frozenset({
    # English
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with", "this", "these", "those",
    # Vietnamese (light)
    "và", "của", "là", "có", "trong", "với", "được", "không", "cho", "để",
    "những", "các", "một", "này", "khi", "thì", "về", "đã", "sẽ",
    # Rule syntax — distilled rules start with IF/THEN. Drop them so BM25
    # doesn't over-weight "if" across all rules (every rule has "if").
    "if", "then", "and", "or", "the", "to",
})


def _tokenize(text: str) -> List[str]:
    """Tokenize for BM25. Lowercase, drop stopwords + 1-char tokens.
    Hyphenated terms survive as single tokens (etcd-main-0, kube-apiserver)."""
    return [
        t.lower() for t in _TOKEN_RE.findall(text or "")
        if t.lower() not in _STOPWORDS and len(t) > 1
    ]


class _BM25Index:
    """Lightweight in-memory BM25 (Okapi variant) over rule_text tokens.

    Built once when KBStore loads its embedding index — same data, same
    lifecycle. Query tokens are looked up against an inverted index so
    only rules containing the term are visited. Typical scale: 1300 rules,
    ~5 hits per term → ~25k ops for a 5k-token query (< 50ms in pure Python).

    Why not `rank_bm25` library: BM25 is ~50 lines and we want the inverted
    index exposed for tight memory accounting on a long-running bot.
    """
    def __init__(self, rules: Dict[str, dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # term -> list of (rule_id, term_freq, doc_len)
        self.postings: Dict[str, List[tuple]] = defaultdict(list)
        self.n = 0
        self.avgdl = 0.0
        total_len = 0
        for rid, entry in rules.items():
            tokens = _tokenize(entry["rule"].get("rule_text", ""))
            dl = len(tokens)
            total_len += dl
            self.n += 1
            for term, freq in Counter(tokens).items():
                self.postings[term].append((rid, freq, dl))
        self.avgdl = (total_len / self.n) if self.n else 0.0

    def score(self, query_tokens: List[str]) -> Dict[str, float]:
        """Return {rule_id: bm25_score}. Rules with no token overlap are
        absent from the result (score 0 by omission)."""
        if not query_tokens or self.n == 0:
            return {}
        scores: Dict[str, float] = defaultdict(float)
        for q_t in set(query_tokens):  # dedupe — IDF is query-term, not query-freq
            postings = self.postings.get(q_t)
            if not postings:
                continue
            df = len(postings)
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)
            for rid, freq, dl in postings:
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[rid] += idf * (freq * (self.k1 + 1)) / denom
        return dict(scores)


class KBStore:
    KB_COLLECTION = "kb_rules"

    def __init__(self) -> None:
        self._collection: Optional[Collection] = None
        self._loaded: bool = False
        # {rule_id: {"rule": doc, "vector": list[float]}}
        self._index: dict = {}
        self._bm25: Optional[_BM25Index] = None

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
            # Build BM25 over the same rules we just loaded. If this throws,
            # we still mark _loaded — embeddings alone are enough to serve.
            self._bm25 = _BM25Index(self._index)
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
        """Return top-K rules for `query_text`. Hybrid BM25 + cosine via
        Reciprocal Rank Fusion, then frequency-weighted by match_count.

        Returns [] on any failure (disabled, empty index, both embed AND
        BM25 empty). Does NOT mutate match_count — the caller (kb_match_handler)
        does that fire-and-forget after sending the response.
        """
        if not bot_config.kb_enabled or not query_text or not query_text.strip():
            return []
        self._ensure_loaded()
        if not self._index:
            return []
        k = top_k if top_k and top_k > 0 else bot_config.kb_top_k

        chunks = _chunk_at_section_markers(query_text, _MAX_CHARS_PER_CHUNK)
        query_tokens = _tokenize(query_text)

        # --- Cosine track -------------------------------------------------
        # Best cosine per rule across all chunks. Empty if embed fails —
        # BM25 alone still works (graceful degradation when API is down).
        cos_scores: Dict[str, float] = {}
        try:
            q_vecs = [v for v in embed_chunks(chunks) if v]
            if q_vecs:
                for rule_id, entry in self._index.items():
                    v = entry["vector"]
                    best_cos = 0.0
                    for qv in q_vecs:
                        s = _cosine(qv, v)
                        if s > best_cos:
                            best_cos = s
                    cos_scores[rule_id] = best_cos
        except Exception as e:
            logger.warning("kb_query_embed_failed_bm25_only", error=str(e))

        # --- BM25 track ---------------------------------------------------
        bm25_scores: Dict[str, float] = {}
        if self._bm25 is not None and query_tokens:
            bm25_scores = self._bm25.score(query_tokens)

        if not cos_scores and not bm25_scores:
            return []

        # --- Reciprocal Rank Fusion --------------------------------------
        # Rank each track separately, then combine as 1/(k+rank). RRF is
        # robust to scale differences between cosine ∈ [0,1] and BM25's
        # unbounded IDF-weighted scores — fused rank is scale-free.
        cos_ranked = sorted(cos_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        cos_rank = {rid: i + 1 for i, (rid, _) in enumerate(cos_ranked)}
        bm25_rank = {rid: i + 1 for i, (rid, _) in enumerate(bm25_ranked)}
        rrf_scores: Dict[str, float] = defaultdict(float)
        for rid, r in cos_rank.items():
            rrf_scores[rid] += 1.0 / (_K_RRF + r)
        for rid, r in bm25_rank.items():
            rrf_scores[rid] += 1.0 / (_K_RRF + r)

        # --- Frequency boost + floor -------------------------------------
        # Apply match_count boost on the FUSED rank score so a rule that's
        # both semantically and lexically close AND matched many times
        # bubbles up. Floor: a rule must clear EITHER cosine OR BM25 —
        # exact keyword match shouldn't be discarded just because cosine
        # is below 0.15, and vice versa.
        scored = []
        for rid, fused in rrf_scores.items():
            cos = cos_scores.get(rid, 0.0)
            bm = bm25_scores.get(rid, 0.0)
            if cos < _MIN_COSINE and bm < _MIN_BM25:
                continue
            match_count = int(self._index[rid]["rule"].get("match_count", 0) or 0)
            boosted = fused * (1.0 + math.log1p(match_count))
            scored.append((boosted, cos, bm, match_count, rid))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for boosted, cos, bm, mc, rid in scored[:k]:
            rule = self._index[rid]["rule"]
            out.append({
                "rule_id": rid,
                "issue_id": rule.get("issue_id"),
                "issue_title": rule.get("issue_title"),
                "rule_text": rule.get("rule_text"),
                "match_count": mc,
                "score": float(boosted),
                "cosine": float(cos),
                "bm25": float(bm),
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
