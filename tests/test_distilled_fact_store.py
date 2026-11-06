"""Tests for DistilledFactStore — the MongoDB-backed store of distilled facts.

Patches MongoDB via the autouse `mock_mongodb` fixture in conftest.py. To
bypass the lazy collection accessors, each test injects a MagicMock
directly into `_facts` / `_runs` so the property `facts`/`runs` returns
it without touching the real MongoDB wiring.
"""
import datetime

import pytest
from pymongo.errors import DuplicateKeyError
from unittest.mock import MagicMock

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
