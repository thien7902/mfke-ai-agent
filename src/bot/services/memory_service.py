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
import time
import uuid
from typing import List, Optional, Tuple

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

        # Whole-graph facts cache: (timestamp, facts). Refreshed on a 5-min
        # TTL, served to HolmesService._build_holmes_messages as a system
        # prompt block. Refresh runs in a background thread (memory loop) so
        # the sync caller never blocks on FalkorDB.
        self._facts_cache: Optional[Tuple[float, List[str]]] = None
        self._facts_refresh_in_flight: bool = False
        self._facts_lock = threading.Lock()

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

    # Public alias for tests/tools that want to invoke via a stable name.
    run_sync = _run

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

            # Prime the whole-graph facts cache so the first LLM turn after
            # startup has facts in the system prompt instead of an empty
            # block. Failures here are logged inside _fetch_all_facts_on_loop
            # and don't fail initialization — the cache just stays empty
            # and the first turn's facts block is skipped.
            try:
                await self._fetch_all_facts_on_loop()
            except Exception:
                pass

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
            # Enable AOF + frequent RDB save so data persists across hard
            # container kills (docker stop's 10s SIGTERM grace often isn't
            # enough for redislite's graceful-shutdown RDB dump).
            # `save 60 1` = RDB snapshot every 60s if ≥1 key changed.
            # `appendonly yes` = every write appended to .aof, replayed on
            # startup. AOF + RDB together = data survives crashes.
            serverconfig = {
                "save": "60 1",
                "appendonly": "yes",
                "appendfsync": "everysec",
            }
            falkordb_client = AsyncFalkorDB(
                dbfilename=bot_config.graphiti_db_path,
                serverconfig=serverconfig,
            )
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

    async def retrieve_relevant(self, query: str, k: int = 20) -> List[str]:
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

    # ------------------------------------------------------------------ #
    # Whole-graph facts cache (system-prompt injection path)
    # ------------------------------------------------------------------ #
    #
    # `HolmesService._build_holmes_messages` is sync and runs on the main
    # loop thread. It needs the current fact set without blocking on a
    # FalkorDB round-trip. `get_cached_facts_sync` is the sync entry: it
    # returns the cached snapshot immediately and, when stale, schedules
    # a background refresh on the dedicated memory loop. The refresh
    # thread writes the new snapshot via `_write_facts_cache`; the next
    # sync caller sees it.
    #
    # Async callers (tests, explicit re-query) use `dump_all_facts` which
    # bypasses the cache and fetches on the dedicated loop.

    async def dump_all_facts(self) -> List[str]:
        """Bypass the cache and fetch all facts from the graph now.

        Used by tests and explicit re-query paths. Runs the fetch on the
        dedicated memory loop via `_run_async` (Graphiti's connections are
        bound to that loop). Auto-initializes if needed; returns [] when
        memory is disabled or the fetch fails.
        """
        if not bot_config.graphiti_memory_enabled:
            return []
        if not self._initialized:
            try:
                await self.initialize()
            except Exception as e:
                logger.warning("dump_all_facts_init_failed",
                               error=str(e), error_type=type(e).__name__)
                return []
        return await self._run_async(self._fetch_all_facts_on_loop())

    async def _fetch_all_facts_on_loop(self) -> List[str]:
        """Run on the dedicated memory loop. Loads every EntityEdge for the
        bot's group_id and returns the `fact` text of each. Updates the
        cache as a side-effect. Treats 'no edges for group' as an empty
        graph (cache = []) so the facts-block is skipped cleanly until
        /addmemory adds something — not as a fetch failure."""
        if not bot_config.graphiti_memory_enabled or not self._graphiti:
            return []
        try:
            from graphiti_core.edges import EntityEdge
            from graphiti_core.errors import GroupsEdgesNotFoundError
        except Exception as e:
            logger.warning("memory_facts_import_failed",
                           error=str(e), error_type=type(e).__name__)
            return self._facts_cache[1] if self._facts_cache else []
        try:
            # EntityEdge.get_by_group_ids on FalkorDB routes to a per-group
            # database (clone(database=group_ids[0])). Our deployment stores
            # all edges on the configured default database with group_id set
            # to bot_config.graphiti_group_id — so the routed query lands on
            # an empty "global" database and raises GroupsEdgesNotFoundError,
            # even though search_memory (which doesn't route) sees edges.
            # Bypass the routing by querying the driver directly with the
            # same WHERE e.group_id IN [...] filter the search path uses.
            from graphiti_core.driver.driver import GraphProvider
            from graphiti_core.models.edges.edge_db_queries import (
                get_entity_edge_return_query,
            )

            provider = getattr(self._graphiti.driver, "provider", None)
            match_query = (
                "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) "
                "WHERE e.group_id IN $group_ids "
                "RETURN "
                + get_entity_edge_return_query(provider)
                + " ORDER BY e.uuid DESC"
            )
            records, _, _ = await self._graphiti.driver.execute_query(
                match_query,
                group_ids=[bot_config.graphiti_group_id],
                routing_="r",
            )
            from graphiti_core.driver.record_parsers import entity_edge_from_record
            edges = [entity_edge_from_record(r) for r in records]
            facts = [e.fact for e in edges if getattr(e, "fact", None)]
            self._write_facts_cache(facts)
            logger.info("memory_facts_dumped", num_facts=len(facts))
            return facts
        except Exception as e:
            logger.warning("memory_facts_dump_failed",
                           error=str(e), error_type=type(e).__name__)
            # Leave the previous snapshot intact so a transient FalkorDB
            # blip doesn't wipe the prompt's facts block.
            return self._facts_cache[1] if self._facts_cache else []

    def get_cached_facts_sync(self, max_age_seconds: int = 300) -> List[str]:
        """Return the cached fact snapshot if within TTL; otherwise kick off
        a background refresh and return the prior snapshot (or []). Never
        blocks on FalkorDB — the LLM turn is never delayed by the refresh.
        """
        if not bot_config.graphiti_memory_enabled:
            return []
        # If graphiti is not initialized yet (or the priming fetch hasn't
        # completed), there's no snapshot to serve. Don't schedule a refresh
        # here — `initialize()` primes the cache during startup.
        if self._graphiti is None:
            return []
        with self._facts_lock:
            ts: Optional[float] = None
            facts: Optional[List[str]] = None
            if self._facts_cache is not None:
                ts, facts = self._facts_cache
            refresh_due = ts is None or (time.time() - ts) > max_age_seconds
            if refresh_due and not self._facts_refresh_in_flight:
                self._facts_refresh_in_flight = True
                self._schedule_facts_refresh_unlocked()
            return list(facts) if facts else []

    def _write_facts_cache(self, facts: List[str]) -> None:
        with self._facts_lock:
            self._facts_cache = (time.time(), list(facts))

    async def invalidate_facts_cache(self) -> None:
        """Mark the facts cache as stale without wiping the snapshot.

        Called by `/addmemory` after a successful episode ingest. The next
        `get_cached_facts_sync` will trigger a background refresh and serve
        the previous snapshot until the new one is ready.
        """
        with self._facts_lock:
            if self._facts_cache is not None:
                self._facts_cache = (0.0, self._facts_cache[1])

    def _schedule_facts_refresh_unlocked(self) -> None:
        """Fire-and-forget refresh. Caller MUST hold `_facts_lock` and have
        already set `_facts_refresh_in_flight=True`. Spawns a daemon thread
        that runs the fetch on the memory loop and clears the in-flight
        flag in finally."""
        def _runner():
            try:
                self._run(self._fetch_all_facts_on_loop(), timeout=30)
            except Exception as e:
                logger.warning("memory_facts_refresh_failed",
                               error=str(e), error_type=type(e).__name__)
            finally:
                with self._facts_lock:
                    self._facts_refresh_in_flight = False
        threading.Thread(
            target=_runner,
            name="memory-facts-refresh",
            daemon=True,
        ).start()

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
            # Force a background save so the .aof/.rdb file gets the new
            # episode promptly — without this, a hard container kill between
            # the auto-save intervals (60s) would lose the just-added data.
            await self._bgwriteaof()
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

    async def _bgwriteaof(self) -> None:
        """Kick redislite to flush the AOF buffer to disk after a write.

        With `appendonly yes` + `appendfsync everysec`, redis flushes every
        second anyway, but explicit BGREWRITEAOF on every add_episode makes
        the on-disk file small + guarantees the new episode survives even a
        hard kill happening within the 1-second window. Best-effort — won't
        block ingest on failure.
        """
        if self._graphiti is None:
            return
        try:
            driver = self._graphiti.driver
            client = getattr(driver, "client", None)
            if client is None:
                return
            # FalkorDB/redis `client` exposes execute_command via the
            # underlying async redis client.
            exec_cmd = getattr(client, "execute_command", None)
            if exec_cmd is None:
                # Some versions wrap the redis client under .connection
                conn = getattr(client, "connection", None)
                exec_cmd = getattr(conn, "execute_command", None)
            if exec_cmd is None:
                return
            await exec_cmd("BGSAVE")
        except Exception as e:
            logger.debug("memory_bgwriteaof_failed",
                          error=str(e), error_type=type(e).__name__)

    async def close(self) -> None:
        if self._graphiti is not None:
            try:
                await self._run_async(self._graphiti.close())
            except Exception as e:
                logger.warning("graphiti_close_failed", error=str(e))
        self._graphiti = None
        self._initialized = False
        # Drop the facts cache snapshot so the next init primes fresh.
        with self._facts_lock:
            self._facts_cache = None
            self._facts_refresh_in_flight = False
        # Stop the dedicated loop
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        self._loop = None
        self._loop_thread = None
