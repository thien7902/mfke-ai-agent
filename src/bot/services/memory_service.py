"""Memory Service - Graphiti-backed knowledge retrieval.

Stores product knowledge in a graph (FalkorDB, embedded or server) and
retrieves relevant facts at query time. The LLM extracts search keywords
from the user's message, then Graphiti's hybrid search returns matching
facts which are injected into the system prompt.

Disabled by default (Config.graphiti_memory_enabled=False) — when off,
every method is a no-op so the bot keeps working without a graph DB.
"""
import asyncio
import datetime
import logging
import os
import uuid
from typing import List, Optional

import anthropic
import structlog

from src.bot.utils.config import config as bot_config

logger = structlog.get_logger(__name__)

# Graphiti imports are lazy so the bot can import this module on machines
# without graphiti_core installed (e.g. a fresh dev checkout that hasn't
# pip installed -r requirements.txt yet).
_graphiti = None


def _import_graphiti():
    global _graphiti
    if _graphiti is not None:
        return _graphiti
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.llm_client.anthropic_client import AnthropicClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.nodes import EpisodeType

    _graphiti = {
        "Graphiti": Graphiti,
        "FalkorDriver": FalkorDriver,
        "AnthropicClient": AnthropicClient,
        "OpenAIEmbedder": OpenAIEmbedder,
        "OpenAIEmbedderConfig": OpenAIEmbedderConfig,
        "EpisodeType": EpisodeType,
    }
    return _graphiti


class MemoryService:
    """Singleton-ish service wrapping Graphiti.

    Lifecycle: instantiate once at app start, call `await initialize()` before
    first use, call `await close()` on shutdown. When `graphiti_memory_enabled`
    is False, every method becomes a no-op.
    """

    def __init__(self):
        self._graphiti = None  # graphiti_core.Graphiti instance
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._anthropic = None  # lazy AsyncAnthropic client for keyword extraction

    async def initialize(self) -> None:
        if not bot_config.graphiti_memory_enabled:
            return
        async with self._init_lock:
            if self._initialized:
                return
            mods = _import_graphiti()
            Graphiti = mods["Graphiti"]
            FalkorDriver = mods["FalkorDriver"]
            AnthropicClient = mods["AnthropicClient"]

            driver = self._build_driver(FalkorDriver)
            llm_client = AnthropicClient()
            embedder = self._build_embedder(mods["OpenAIEmbedder"], mods["OpenAIEmbedderConfig"])

            self._graphiti = Graphiti(
                graph_driver=driver,
                llm_client=llm_client,
                embedder=embedder,
            )
            await self._graphiti.build_indices_and_constraints()

            self._anthropic = anthropic.AsyncAnthropic()

            self._initialized = True
            logger.info("graphiti_memory_initialized",
                        mode=bot_config.graphiti_falkor_mode,
                        group_id=bot_config.graphiti_group_id)

    def _build_embedder(self, OpenAIEmbedder, OpenAIEmbedderConfig):
        """Build the OpenAI-compatible embedder from config + env.

        Pulls model name + dim from Config (default: multilingual-e5-large /
        1024-dim) and API credentials from OPENAI_API_KEY / OPENAI_API_BASE.
        """
        embedder_config = OpenAIEmbedderConfig(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE"),
            embedding_model=bot_config.graphiti_embedding_model,
            embedding_dim=bot_config.graphiti_embedding_dim,
        )
        return OpenAIEmbedder(config=embedder_config)

    def _build_driver(self, FalkorDriver):
        if bot_config.graphiti_falkor_mode == "embedded":
            from redislite.async_falkordb_client import AsyncFalkorDB
            falkordb_client = AsyncFalkorDB(dbfilename=bot_config.graphiti_db_path)
            return FalkorDriver(
                falkor_db=falkordb_client,
                database=bot_config.graphiti_falkor_database,
            )
        else:
            return FalkorDriver(
                host=bot_config.graphiti_falkor_host,
                port=bot_config.graphiti_falkor_port,
                database=bot_config.graphiti_falkor_database,
            )

    async def retrieve_relevant(self, query: str, k: int = 5) -> List[str]:
        """Return up to k facts relevant to the query.

        LLM-driven: the user message is first converted to a concise search
        query (keyword extraction), then Graphiti's hybrid search returns
        matching facts. Returns an empty list when memory is disabled.
        """
        if not bot_config.graphiti_memory_enabled or not self._initialized:
            return []
        if not query or not query.strip():
            return []

        # Keyword extraction is best-effort: if it fails, fall back to the raw
        # user query so we still try to retrieve. Failures here are logged but
        # not fatal — the search step below is what matters.
        try:
            search_query = await self._extract_keywords(query)
            if not search_query:
                search_query = query
        except Exception as e:
            logger.warning("keyword_extraction_failed", error=str(e))
            search_query = query

        try:
            edges = await self._graphiti.search(
                query=search_query,
                num_results=k,
            )
            facts = [e.fact for e in edges if getattr(e, "fact", None)]
            logger.info("memory_retrieved",
                        query_preview=query[:60],
                        search_query=search_query,
                        num_facts=len(facts))
            return facts
        except Exception as e:
            logger.warning("memory_retrieval_failed", error=str(e))
            return []

    async def _extract_keywords(self, query: str) -> str:
        """Ask the LLM to compress the user message into 1-3 search terms.

        Cheaper + more precise than embedding the whole user message: long
        conversational messages often retrieve nothing on plain semantic
        similarity, but a distilled keyword query hits the right facts.
        """
        if self._anthropic is None:
            return query
        try:
            resp = await self._anthropic.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                messages=[{
                    "role": "user",
                    "content": (
                        "Extract 1-3 concise search terms from this user message "
                        "for retrieving relevant knowledge from a fact database. "
                        "Return only the terms as a single short phrase, comma-separated. "
                        "No preamble.\n\nMessage: " + query
                    ),
                }],
            )
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )
            return text.strip() or query
        except Exception as e:
            logger.warning("keyword_extraction_failed", error=str(e))
            return query

    async def add_episode(
        self,
        text: str,
        source_description: str = "",
        reference_time: Optional[datetime.datetime] = None,
        name: Optional[str] = None,
    ) -> Optional[str]:
        """Ingest a text blob as an episode. Returns the episode uuid, or None
        when memory is disabled or ingest fails.
        """
        if not bot_config.graphiti_memory_enabled:
            logger.warning("add_episode_called_but_disabled")
            return None
        if not self._initialized:
            # Initialize failed at startup — main.py:120 logged it once with
            # memory_init_failed_bot_continues, but every add_episode call
            # silently returns None which makes /addmemory failures invisible.
            # Log here too so the user sees the init failure linked to their
            # add request (not just a generic "failed to add" from the command).
            logger.error(
                "add_episode_called_but_not_initialized",
                hint="MemoryService.initialize() failed at startup — "
                     "check earlier 'memory_init_failed_bot_continues' log "
                     "for the cause (FalkorDB unreachable, embedder env "
                     "missing, graphiti_core import error, etc.)",
            )
            raise RuntimeError("MemoryService is not initialized")
        if not text or not text.strip():
            return None

        mods = _import_graphiti()
        EpisodeType = mods["EpisodeType"]
        ref_time = reference_time or datetime.datetime.now(datetime.timezone.utc)
        ep_name = name or f"mem-{uuid.uuid4().hex[:8]}"

        try:
            result = await self._graphiti.add_episode(
                name=ep_name,
                episode_body=text,
                source_description=source_description or "telegram-bot",
                reference_time=ref_time,
                source=EpisodeType.text,
                group_id=bot_config.graphiti_group_id,
            )
            logger.info("memory_episode_added",
                        name=ep_name,
                        source=source_description,
                        text_len=len(text))
            return getattr(result, "uuid", None) or ep_name
        except Exception as e:
            # Log with full traceback + the first/last 200 chars of the
            # episode body so we can diagnose whether it was a size limit,
            # LLM error, embedder error, or FalkorDB write failure.
            logger.error(
                "memory_episode_failed",
                error=str(e),
                error_type=type(e).__name__,
                text_len=len(text),
                text_preview_head=text[:200],
                text_preview_tail=text[-200:] if len(text) > 200 else "",
                exc_info=True,
                source=source_description,
            )
            raise

    async def close(self) -> None:
        if self._graphiti is not None:
            try:
                await self._graphiti.close()
            except Exception as e:
                logger.warning("graphiti_close_failed", error=str(e))
        self._graphiti = None
        self._initialized = False
