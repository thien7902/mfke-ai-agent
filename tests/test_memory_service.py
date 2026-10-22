"""Tests for MemoryService — the Graphiti-backed knowledge retrieval layer.

Disabled-mode behaviour is tested with the real (non-mocked) MemoryService
constructor. Enabled-mode behaviour mocks Graphiti to avoid needing FalkorDB.
"""
import asyncio
import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.services.memory_service import MemoryService


@pytest.mark.asyncio
async def test_disabled_service_is_noop():
    """When config.graphiti_memory_enabled is False, every method short-circuits."""
    with patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = False
        svc = MemoryService()
        await svc.initialize()
        assert svc._initialized is False
        assert await svc.retrieve_relevant("anything") == []
        assert await svc.add_episode("some text") is None
        await svc.close()


@pytest.mark.asyncio
async def test_retrieve_relevant_returns_empty_when_not_initialized():
    """Even if config is enabled, retrieve before initialize() returns []."""
    with patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        svc = MemoryService()
        # Don't call initialize() — should not blow up, just return []
        assert await svc.retrieve_relevant("hello") == []


@pytest.mark.asyncio
async def test_retrieve_relevant_extracts_keywords_then_searches():
    """Keyword extraction runs first, then graphiti.search is called with the result."""
    with patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.graphiti_falkor_mode = "server"

        svc = MemoryService()
        svc._initialized = True
        svc._anthropic = MagicMock()
        # Keyword extraction returns a distilled query
        svc._extract_keywords = AsyncMock(return_value="deploy kubernetes")

        # Mock graphiti.search returns EntityEdge-like objects with .fact
        edge1 = MagicMock()
        edge1.fact = "deploys use kubectl apply"
        edge2 = MagicMock()
        edge2.fact = "pods restart on crash"
        svc._graphiti = MagicMock()
        svc._graphiti.search = AsyncMock(return_value=[edge1, edge2])

        facts = await svc.retrieve_relevant("How do I deploy to k8s?", k=5)

        assert facts == ["deploys use kubectl apply", "pods restart on crash"]
        svc._graphiti.search.assert_awaited_once()
        # The search query passed to Graphiti should be the extracted keywords
        call_kwargs = svc._graphiti.search.call_args.kwargs
        assert call_kwargs["query"] == "deploy kubernetes"
        assert call_kwargs["num_results"] == 5


@pytest.mark.asyncio
async def test_retrieve_relevant_falls_back_to_raw_query_on_extraction_failure():
    """If keyword extraction raises, retrieve_relevant still searches with the raw query."""
    with patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.graphiti_falkor_mode = "server"

        svc = MemoryService()
        svc._initialized = True
        svc._extract_keywords = AsyncMock(side_effect=Exception("boom"))
        svc._graphiti = MagicMock()
        svc._graphiti.search = AsyncMock(return_value=[])

        facts = await svc.retrieve_relevant("raw user query")
        assert facts == []
        call_kwargs = svc._graphiti.search.call_args.kwargs
        # Falls back to the raw user message
        assert call_kwargs["query"] == "raw user query"


@pytest.mark.asyncio
async def test_retrieve_relevant_empty_query_returns_empty():
    with patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        svc = MemoryService()
        svc._initialized = True
        svc._graphiti = MagicMock()
        svc._graphiti.search = AsyncMock()
        assert await svc.retrieve_relevant("") == []
        assert await svc.retrieve_relevant("   ") == []
        svc._graphiti.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_relevant_swallows_search_errors():
    """Graphiti search failures should never break the bot — return [] instead."""
    with patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        svc = MemoryService()
        svc._initialized = True
        svc._extract_keywords = AsyncMock(return_value="kw")
        svc._graphiti = MagicMock()
        svc._graphiti.search = AsyncMock(side_effect=RuntimeError("graph db down"))
        assert await svc.retrieve_relevant("q") == []


@pytest.mark.asyncio
async def test_add_episode_disabled_returns_none():
    with patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = False
        svc = MemoryService()
        assert await svc.add_episode("hello world") is None


@pytest.mark.asyncio
async def test_add_episode_passes_through_to_graphiti():
    with patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        cfg.graphiti_group_id = "global"

        svc = MemoryService()
        svc._initialized = True

        # Mock the graphiti instance and the EpisodeType import inside add_episode
        fake_result = MagicMock()
        fake_result.uuid = "ep-123"
        svc._graphiti = MagicMock()
        svc._graphiti.add_episode = AsyncMock(return_value=fake_result)

        ref_time = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        uuid_out = await svc.add_episode(
            text="some knowledge blob",
            source_description="telegram-addmemory",
            reference_time=ref_time,
            name="test-ep",
        )

        assert uuid_out == "ep-123"
        svc._graphiti.add_episode.assert_awaited_once()
        call_kwargs = svc._graphiti.add_episode.call_args.kwargs
        assert call_kwargs["name"] == "test-ep"
        assert call_kwargs["episode_body"] == "some knowledge blob"
        assert call_kwargs["source_description"] == "telegram-addmemory"
        assert call_kwargs["reference_time"] == ref_time
        assert call_kwargs["group_id"] == "global"


@pytest.mark.asyncio
async def test_add_episode_empty_text_returns_none():
    with patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        svc = MemoryService()
        svc._initialized = True
        svc._graphiti = MagicMock()
        svc._graphiti.add_episode = AsyncMock()
        assert await svc.add_episode("") is None
        assert await svc.add_episode("   ") is None
        svc._graphiti.add_episode.assert_not_awaited()


@pytest.mark.asyncio
async def test_add_episode_raises_on_ingest_failure():
    """add_episode now re-raises so callers see the actual exception."""
    with patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        svc = MemoryService()
        svc._initialized = True
        svc._graphiti = MagicMock()
        svc._graphiti.add_episode = AsyncMock(side_effect=RuntimeError("ingest failed"))
        with pytest.raises(RuntimeError, match="ingest failed"):
            await svc.add_episode("valid text")


@pytest.mark.asyncio
async def test_add_episode_raises_when_not_initialized():
    """When MemoryService init failed at startup, add_episode surfaces RuntimeError."""
    with patch("src.bot.services.memory_service.bot_config") as cfg:
        cfg.graphiti_memory_enabled = True
        svc = MemoryService()
        svc._initialized = False
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.add_episode("valid text")


def test_build_embedder_uses_config_defaults():
    """_build_embedder pulls model + dim from config and creds from env."""
    import os as _os
    with patch("src.bot.services.memory_service.bot_config") as cfg, \
         patch.dict(_os.environ, {
             "OPENAI_API_KEY": "sk-test",
             "OPENAI_API_BASE": "https://embed.test/v1",
         }):
        cfg.graphiti_embedding_model = "multilingual-e5-large"
        cfg.graphiti_embedding_dim = 1024

        # Fake OpenAIEmbedder/OpenAIEmbedderConfig to capture the kwargs passed
        seen = {}

        class FakeConfig:
            def __init__(self, **kwargs):
                seen["config"] = kwargs

        class FakeEmbedder:
            def __init__(self, config=None):
                seen["embedder"] = config

        svc = MemoryService()
        out = svc._build_embedder(FakeEmbedder, FakeConfig)

        # Returns the embedder instance
        assert out is not None
        # Embedder received the config object
        assert seen["embedder"] is not None
        # Config was built with the correct model + dim from graphiti config
        assert seen["config"]["embedding_model"] == "multilingual-e5-large"
        assert seen["config"]["embedding_dim"] == 1024
        assert seen["config"]["api_key"] == "sk-test"
        assert seen["config"]["base_url"] == "https://embed.test/v1"
