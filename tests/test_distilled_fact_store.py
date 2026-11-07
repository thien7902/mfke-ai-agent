"""Tests for DistilledFactStore — the MongoDB-backed store of distilled facts.

Patches MongoDB via the autouse `mock_mongodb` fixture in conftest.py. To
bypass the lazy collection accessors, each test injects a MagicMock
directly into `_facts` / `_runs` so the property `facts`/`runs` returns
it without touching the real MongoDB wiring.
"""
import datetime

import pytest
from pymongo.errors import DuplicateKeyError
from unittest.mock import MagicMock, patch

from src.bot.services import distilled_fact_store as dfs_mod
from src.bot.services.distilled_fact_store import DistilledFactStore


def _store_with(facts=None, runs=None):
    """Construct a DistilledFactStore and pre-populate its private
    collection attributes so the lazy properties never call Mongo."""
    s = DistilledFactStore()
    s._facts = facts if facts is not None else MagicMock()
    s._runs = runs if runs is not None else MagicMock()
    return s


@pytest.mark.asyncio
async def test_add_returns_true_on_new_fact():
    facts = MagicMock()
    facts.insert_one.return_value = MagicMock(inserted_id="oid")
    store = _store_with(facts=facts)
    added = await store.add("MFKE api-gw runs on port 8080", 1, 2, "run-1")
    assert added is True
    facts.insert_one.assert_called_once()
    doc = facts.insert_one.call_args[0][0]
    assert doc["text"] == "MFKE api-gw runs on port 8080"
    assert doc["source_chat_id"] == 1
    assert doc["source_topic_id"] == 2
    assert doc["source_run_id"] == "run-1"
    assert isinstance(doc["created_at"], datetime.datetime)


@pytest.mark.asyncio
async def test_add_returns_false_on_duplicate():
    facts = MagicMock()
    facts.insert_one.side_effect = DuplicateKeyError("dup")
    store = _store_with(facts=facts)
    added = await store.add("dup text", 1, 2, "run-1")
    assert added is False


@pytest.mark.asyncio
async def test_add_returns_false_on_empty_text():
    facts = MagicMock()
    store = _store_with(facts=facts)
    assert await store.add("   ", 1, 2, "run-1") is False
    assert await store.add("", 1, 2, "run-1") is False
    facts.insert_one.assert_not_called()


@pytest.mark.asyncio
async def test_record_run_upserts_latest_run():
    runs = MagicMock()
    store = _store_with(runs=runs)
    await store.record_run(chat_id=1, topic_id=2, run_id="run-xyz")
    runs.update_one.assert_called_once()
    args, kwargs = runs.update_one.call_args
    assert args[0] == {"chat_id": 1, "topic_id": 2}
    assert args[1]["$set"]["latest_run_id"] == "run-xyz"
    assert args[1]["$set"]["chat_id"] == 1
    assert args[1]["$set"]["topic_id"] == 2
    assert kwargs.get("upsert") is True


def test_get_injectable_facts_empty_when_no_runs():
    runs = MagicMock()
    runs.find.return_value = []
    store = _store_with(runs=runs)
    assert store.get_injectable_facts() == []


def test_get_injectable_facts_returns_only_latest_run_per_topic():
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [
        {"latest_run_id": "runA2", "topic_id": 10},
        {"latest_run_id": "runB1", "topic_id": 20},
    ]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        {"text": "fact A2a", "source_run_id": "runA2", "source_topic_id": 10},
        {"text": "fact A2b", "source_run_id": "runA2", "source_topic_id": 10},
        {"text": "fact B1", "source_run_id": "runB1", "source_topic_id": 20},
    ])
    facts.find.return_value = cursor

    store = _store_with(facts=facts, runs=runs)
    out = store.get_injectable_facts()

    find_filter = facts.find.call_args[0][0]
    assert find_filter == {"source_run_id": {"$in": ["runA2", "runB1"]}}
    assert len(out) == 3
    assert all(f["source_run_id"] in ("runA2", "runB1") for f in out)


def test_get_injectable_facts_swallows_db_error():
    runs = MagicMock()
    runs.find.side_effect = Exception("mongo down")
    store = _store_with(runs=runs)
    assert store.get_injectable_facts() == []


def test_get_injectable_facts_skips_dedup_when_threshold_zero():
    """Threshold 0 → dedup off, full list returned even though embeddings
    would collapse identical-vector facts if dedup ran."""
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        {"text": "alpha", "source_run_id": "r1", "source_topic_id": 5, "created_at": 1},
        {"text": "alpha dup", "source_run_id": "r1", "source_topic_id": 5, "created_at": 2},
    ])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)
    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.0):
        out = store.get_injectable_facts()
    assert len(out) == 2


def test_get_injectable_facts_dedups_near_duplicate_facts():
    """Two facts above the cosine threshold get collapsed; newest (per
    created_at) wins; dissimilar facts pass through."""
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        # Ascending created_at — oldest first.
        {"text": "old OOM rule", "source_run_id": "r1", "source_topic_id": 5, "created_at": 1},
        {"text": "newest OOM rule", "source_run_id": "r1", "source_topic_id": 5, "created_at": 3},
        {"text": "totally unrelated disk fact", "source_run_id": "r1", "source_topic_id": 5, "created_at": 2},
    ])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)

    # Embeddings: dup pair shares the same vector; unrelated fact orthogonal.
    # embed_chunks is called in input order — same as the list order above.
    vectors = [
        [1.0, 0.0, 0.0],   # old OOM rule
        [1.0, 0.0, 0.0],   # newest OOM rule — identical vector → cosine 1.0 ≥ threshold
        [0.0, 1.0, 0.0],   # disk fact — orthogonal to both
    ]
    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.92), \
         patch("src.bot.services.embedder.embed_chunks", return_value=vectors):
        out = store.get_injectable_facts()

    # Newest wins → "newest OOM rule" kept, "old OOM rule" dropped.
    texts = [f["text"] for f in out]
    assert "newest OOM rule" in texts
    assert "old OOM rule" not in texts
    assert "totally unrelated disk fact" in texts
    assert len(out) == 2


def test_get_injectable_facts_dedup_failure_returns_undismissed_facts():
    """If the embed API is down, dedup is skipped and the original list is
    returned — never break the prompt."""
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        {"text": "fact A", "source_run_id": "r1", "source_topic_id": 5, "created_at": 1},
        {"text": "fact B", "source_run_id": "r1", "source_topic_id": 5, "created_at": 2},
    ])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)

    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.92), \
         patch("src.bot.services.embedder.embed_chunks", return_value=[]):
        out = store.get_injectable_facts()
    # Embed returned nothing → dedup skipped → both facts returned.
    assert len(out) == 2


def test_get_injectable_facts_dedup_uses_persisted_embedding_skips_api():
    """Facts whose docs already carry an ``embedding`` vector are deduped
    WITHOUT calling embed_chunks — pure-local cosine. This is the path
    every inject hits after add() persisted the embedding at insert time.
    """
    facts = MagicMock()
    runs = MagicMock()
    runs.find.return_value = [{"latest_run_id": "r1", "topic_id": 5}]
    cursor = MagicMock()
    cursor.sort.return_value = iter([
        {"text": "old OOM rule",
         "embedding": [1.0, 0.0, 0.0],
         "source_run_id": "r1", "source_topic_id": 5, "created_at": 1},
        {"text": "newest OOM rule",
         "embedding": [1.0, 0.0, 0.0],
         "source_run_id": "r1", "source_topic_id": 5, "created_at": 3},
        {"text": "unrelated disk fact",
         "embedding": [0.0, 1.0, 0.0],
         "source_run_id": "r1", "source_topic_id": 5, "created_at": 2},
    ])
    facts.find.return_value = cursor
    store = _store_with(facts=facts, runs=runs)

    with patch.object(dfs_mod.bot_config, "distilled_dedup_cosine", 0.92), \
         patch("src.bot.services.embedder.embed_chunks") as mock_embed:
        out = store.get_injectable_facts()

    mock_embed.assert_not_called()  # zero API calls — vectors came from Mongo.
    texts = [f["text"] for f in out]
    assert "newest OOM rule" in texts
    assert "old OOM rule" not in texts
    assert "unrelated disk fact" in texts
    assert len(out) == 2


@pytest.mark.asyncio
async def test_add_persists_embedding_in_doc():
    """`add()` embeds the fact text once and stores the vector in the
    fact doc so dedup at inject time skips the API call."""
    facts = MagicMock()
    facts.insert_one.return_value = MagicMock(inserted_id="oid")
    store = _store_with(facts=facts)

    with patch("src.bot.services.embedder.embed_text", return_value=[0.1, 0.2, 0.3]):
        added = await store.add(
            text="OOM pod needs more memory",
            source_chat_id=20,
            source_topic_id=10,
            source_run_id="r1",
        )
    assert added is True
    inserted = facts.insert_one.call_args[0][0]
    assert inserted["embedding"] == [0.1, 0.2, 0.3]
    assert inserted["text"] == "OOM pod needs more memory"


@pytest.mark.asyncio
async def test_add_continues_without_embedding_on_embed_failure():
    """If the embed API is down at insert time, the fact is still inserted
    with embedding=None — dedup will recompute on demand and the next
    backfill_embeddings() pass can fill it in."""
    facts = MagicMock()
    facts.insert_one.return_value = MagicMock(inserted_id="oid")
    store = _store_with(facts=facts)

    with patch("src.bot.services.embedder.embed_text", side_effect=RuntimeError("api down")):
        added = await store.add(text="x", source_chat_id=20, source_topic_id=10, source_run_id="r1")
    assert added is True
    inserted = facts.insert_one.call_args[0][0]
    assert inserted["embedding"] is None
