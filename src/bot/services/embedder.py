"""Sync OpenAI-compatible embeddings wrapper shared by the KB ingest script
and KBStore.

Why a separate module (instead of reusing graphiti's OpenAIEmbedder):
graphiti's embedder is built for entity-extraction pipelines and returns
vectors wrapped in graphiti's own data classes. For KB we just need raw
`list[float]` per text — a thin sync wrapper around the OpenAI client
matches the usage pattern in `consolidator_service._call_consolidator_llm`
and `vision_service._call_openai`. The same `OPENAI_API_KEY` /
`OPENAI_API_BASE` env vars are reused so there's nothing extra to configure.

All functions return `[]` on any failure (missing key, API error, empty
input) so callers can no-op without raising — the bot must never break
because KB embedding is unavailable.
"""
import os
from typing import List

import openai
import structlog

from src.bot.utils.config import config as bot_config

logger = structlog.get_logger(__name__)

# BGE-m3 (Vietnamese_Embedding) accepts up to 8192 tokens. Conservative
# char-based heuristic: 6k chars ≈ 2k tokens, leaving comfortable headroom.
# Used for `embed_text` (single-text path); `embed_chunks` is fed pre-cut
# chunks by callers so its own cap is the per-text size, not a sum.
_MAX_CHARS_PER_TEXT = 6000


def _client() -> "openai.OpenAI":
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key, base_url=api_base) if api_base else openai.OpenAI(api_key=api_key)


def embed_text(text: str) -> List[float]:
    """Embed a single text. Truncates to `_MAX_CHARS_PER_TEXT` chars as a
    token-budget heuristic. Returns [] on any failure or empty input.
    """
    if not text or not text.strip():
        return []
    truncated = text[:_MAX_CHARS_PER_TEXT]
    try:
        client = _client()
    except Exception as e:
        logger.warning("kb_embed_no_api_key", error=str(e))
        return []
    try:
        resp = client.embeddings.create(
            model=bot_config.graphiti_embedding_model,
            input=truncated,
        )
        if not resp.data:
            return []
        return list(resp.data[0].embedding or [])
    except Exception as e:
        logger.warning("kb_embed_failed", error=str(e), error_type=type(e).__name__)
        return []


def embed_chunks(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts in a single API call. OpenAI's embeddings
    endpoint accepts `input=[...]` and returns parallel `data[i].embedding`.
    Returns [] on any failure or empty input. The caller (KBStore) iterates
    the returned list by index — empty list means "no vectors, skip
    retrieval entirely".
    """
    if not texts:
        return []
    truncated = [t[:_MAX_CHARS_PER_TEXT] if t else "" for t in texts]
    # Drop empty strings entirely — OpenAI rejects empty inputs.
    pairs = [(i, t) for i, t in enumerate(truncated) if t.strip()]
    if not pairs:
        return []
    try:
        client = _client()
    except Exception as e:
        logger.warning("kb_embed_no_api_key", error=str(e))
        return []
    try:
        resp = client.embeddings.create(
            model=bot_config.graphiti_embedding_model,
            input=[t for _, t in pairs],
        )
        # Re-key back into the caller's original positions. Returned
        # `data` order matches input order, so we map back via `pairs`.
        out: List[List[float]] = [[] for _ in texts]
        for (orig_idx, _text), datum in zip(pairs, resp.data):
            out[orig_idx] = list(datum.embedding or [])
        return out
    except Exception as e:
        logger.warning("kb_embed_chunks_failed", error=str(e), error_type=type(e).__name__)
        return []
