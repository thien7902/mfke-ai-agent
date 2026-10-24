"""Memory Service - Graphiti-backed knowledge retrieval.

Stores product knowledge in a graph (FalkorDB, embedded or server) and
retrieves relevant facts at query time. The LLM extracts search keywords
from the user's message, then Graphiti's hybrid search returns matching
facts which are injected into the system prompt.

Disabled by default (Config.graphiti_memory_enabled=False) — when off,
every method is a no-op so the bot keeps working without a graph DB.

Threading model: MemoryService owns a dedicated event loop running in a
daemon thread ("memory-loop"). All graphiti calls (initialize, add_episode,
search, close) run on that loop. The Telegram main loop and the Holmes
worker thread both reach the service via run_coroutine_threadsafe — graphiti
holds redis/FalkorDB connections bound to a specific loop, so every access
must go through the same loop, otherwise we hit "Future attached to a
different loop" RuntimeError.
"""
import asyncio
import datetime
import logging
import os
import threading
import uuid
from typing import List, Optional

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
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.nodes import EpisodeType

    _graphiti = {
        "Graphiti": Graphiti,
        "FalkorDriver": FalkorDriver,
        "OpenAIGenericClient": OpenAIGenericClient,
        "LLMConfig": LLMConfig,
        "OpenAIEmbedder": OpenAIEmbedder,
        "OpenAIEmbedderConfig": OpenAIEmbedderConfig,
        "EpisodeType": EpisodeType,
    }
    return _graphiti


class MemoryService:
    """Singleton-ish service wrapping Graphiti.

    Lifecycle: instantiate once at app start. The dedicated loop is started
    lazily on first use (initialize or any public method). Call
    `await initialize()` before first use, call `await close()` on shutdown.
    When `graphiti_memory_enabled` is False, every method becomes a no-op.
    """

    def __init__(self):
        self._graphiti = None  # graphiti_core.Graphiti instance
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._anthropic = None  # reserved for legacy keyword-extraction path

        # Dedicated loop + worker thread for all graphiti calls. Created
        # lazily so tests that never touch memory don't spin a thread.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Loop management
    # ------------------------------------------------------------------ #

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Return the dedicated loop, starting it in a daemon thread if needed."""
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        with self._loop_lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                self._loop_thread = threading.Thread(
                    target=self._loop.run_forever,
                    name="memory-loop",
                    daemon=True,
                )
                self._loop_thread.start()
        return self._loop

    def _run(self, coro, timeout: Optional[float] = None):
        """Schedule a coroutine on the dedicated loop and block until done.

        Use this from sync contexts (e.g. SearchMemoryTool._invoke running in
        Holmes' ThreadPoolExecutor). From async contexts, prefer
        `await self._run_async(coro)` which uses run_coroutine_threadsafe and
        awaits the threaded future via asyncio.wrap_future — that avoids
        blocking the caller's loop and stays cooperative.
        """
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    async def _run_async(self, coro):
        """Schedule a coroutine on the dedicated loop and await it from
        the caller's loop without blocking."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return await asyncio.wrap_future(future)

    # ------------------------------------------------------------------ #
    # Public API — all async methods schedule onto the dedicated loop
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        if not bot_config.graphiti_memory_enabled:
            return
        # Initialize is itself async and runs on the caller's loop. We need
        # it to run on the dedicated loop so graphiti's connections bind to
        # the dedicated loop (not the caller's loop). Schedule + await.
        await self._run_async(self._initialize_on_loop())

    async def _initialize_on_loop(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            try:
                logger.info("memory_init_step", step="import_graphiti")
                mods = _import_graphiti()
                Graphiti = mods["Graphiti"]
                FalkorDriver = mods["FalkorDriver"]
                OpenAIGenericClient = mods["OpenAIGenericClient"]
                LLMConfig = mods["LLMConfig"]

                logger.info("memory_init_step",
                            step="build_driver",
                            mode=bot_config.graphiti_falkor_mode,
                            db_path=bot_config.graphiti_db_path,
                            host=bot_config.graphiti_falkor_host,
                            port=bot_config.graphiti_falkor_port)
                driver = self._build_driver(FalkorDriver)

                llm_model = os.environ.get("OPENAI_MODEL")
                logger.info("memory_init_step",
                            step="build_llm",
                            model=llm_model or "OpenAI default",
                            has_openai_key=bool(os.environ.get("OPENAI_API_KEY")),
                            openai_base=os.environ.get("OPENAI_API_BASE", "<unset>"))
                llm_config = LLMConfig(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=os.environ.get("OPENAI_API_BASE"),
                    model=llm_model or None,
                )
                llm_client = OpenAIGenericClient(config=llm_config)

                logger.info("memory_init_step",
                            step="build_embedder",
                            model=bot_config.graphiti_embedding_model,
                            dim=bot_config.graphiti_embedding_dim,
                            has_openai_key=bool(os.environ.get("OPENAI_API_KEY")),
                            openai_base=os.environ.get("OPENAI_API_BASE", "<unset>"))
                embedder = self._build_embedder(mods["OpenAIEmbedder"], mods["OpenAIEmbedderConfig"])

                self._graphiti = Graphiti(
                    graph_driver=driver,
                    llm_client=llm_client,
                    embedder=embedder,
                )

                logger.info("memory_init_step", step="build_indices")
                await self._graphiti.build_indices_and_constraints()
            except Exception:
                logger.exception("memory_init_failed_at_step")
                raise

            self._initialized = True
            logger.info("graphiti_memory_initialized",
                        mode=bot_config.graphiti_falkor_mode,
                        group_id=bot_config.graphiti_group_id)

    def _build_embedder(self, OpenAIEmbedder, OpenAIEmbedderConfig):
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
        if not bot_config.graphiti_memory_enabled:
            return []
        if not query or not query.strip():
            return []
        if not self._initialized:
            # Auto-init on first use. If init failed before, this will raise
            # from _initialize_on_loop and retrieve returns [] via the except
            # below.
            try:
                await self.initialize()
            except Exception as e:
                logger.warning("retrieve_relevant_init_failed",
                               error=str(e), error_type=type(e).__name__)
                return []
        return await self._run_async(self._retrieve_on_loop(query, k))

    async def _retrieve_on_loop(self, query: str, k: int) -> List[str]:
        if not bot_config.graphiti_memory_enabled or not self._initialized:
            return []
        if not query or not query.strip():
            return []

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
                group_ids=[bot_config.graphiti_group_id],
            )
            facts = [e.fact for e in edges if getattr(e, "fact", None)]
            logger.info("memory_retrieved",
                        query_preview=query[:60],
                        search_query=search_query,
                        group_id=bot_config.graphiti_group_id,
                        num_edges=len(edges),
                        num_facts=len(facts),
                        initialized=self._initialized)
            return facts
        except Exception as e:
            logger.error(
                "memory_retrieval_failed",
                error=str(e),
                error_type=type(e).__name__,
                search_query=search_query,
                exc_info=True,
            )
            return []

    async def _extract_keywords(self, query: str) -> str:
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
        return await self._run_async(
            self._add_episode_on_loop(text, source_description, reference_time, name)
        )

    async def _add_episode_on_loop(
        self,
        text: str,
        source_description: str,
        reference_time: Optional[datetime.datetime],
        name: Optional[str],
    ) -> Optional[str]:
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
                await self._run_async(self._graphiti.close())
            except Exception as e:
                logger.warning("graphiti_close_failed", error=str(e))
        self._graphiti = None
        self._initialized = False
        # Stop the dedicated loop
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        self._loop = None
        self._loop_thread = None
