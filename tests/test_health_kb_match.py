"""Tests for /kb_match + /kb_stats endpoints on src.bot.utils.health.

Pattern: aiohttp.test_utils.TestClient drives the actual aiohttp app
(in-process, real HTTP). Mock the KBStore per-test.
"""
import asyncio
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.bot.utils.health import HealthChecker


async def _make_client():
    """Start a HealthChecker on an ephemeral port, return (client, checker).
    Caller must `await client.close()` at end.
    """
    checker = HealthChecker(host="127.0.0.1", port=0)
    app = checker.create_app()
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client, checker


SAMPLE_RULES = [
    {
        "rule_id": "KB2025-001#1",
        "issue_id": "KB2025-001",
        "issue_title": "OOM cluster",
        "rule_text": "IF pod CrashLoopBackOff AND OOMKilled -> bump memory",
        "match_count": 5,
        "score": 0.87,
    },
    {
        "rule_id": "KB2025-003#2",
        "issue_id": "KB2025-003",
        "issue_title": "PVC stuck",
        "rule_text": "IF PVC Pending AND csi-driver log timeout -> check csi-driver pod",
        "match_count": 1,
        "score": 0.72,
    },
]


@pytest.mark.asyncio
async def test_kb_match_disabled_returns_empty_block():
    client, checker = await _make_client()
    try:
        resp = await client.post("/kb_match", data="CrashLoopBackOff on api-gw")
        assert resp.status == 200
        body = await resp.text()
        assert "===== KB RULES (empty) =====" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kb_match_returns_top_k_rules_block():
    client, checker = await _make_client()
    try:
        store = MagicMock()
        store.get_relevant_rules = MagicMock(return_value=SAMPLE_RULES)
        store.record_matches = MagicMock()
        checker.set_kb_store(store)
        resp = await client.post("/kb_match", data="CrashLoopBackOff on api-gw")
        assert resp.status == 200
        body = await resp.text()
        assert "===== KB RULES (top-2, distilled + frequency-weighted) =====" in body
        assert "[KB2025-001#1 matches=5]" in body
        assert "IF pod CrashLoopBackOff AND OOMKilled -> bump memory" in body
        assert "[KB2025-003#2 matches=1]" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kb_match_increments_match_count_fire_and_forget():
    client, checker = await _make_client()
    try:
        store = MagicMock()
        store.get_relevant_rules = MagicMock(return_value=SAMPLE_RULES)
        store.record_matches = MagicMock()
        checker.set_kb_store(store)
        await client.post("/kb_match", data="CrashLoopBackOff")
        await asyncio.sleep(0.05)
        store.record_matches.assert_called_once_with(
            ["KB2025-001#1", "KB2025-003#2"]
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kb_match_does_not_block_response_on_record_failure():
    client, checker = await _make_client()
    try:
        store = MagicMock()
        store.get_relevant_rules = MagicMock(return_value=SAMPLE_RULES)
        store.record_matches = MagicMock(side_effect=RuntimeError("db down"))
        checker.set_kb_store(store)
        resp = await client.post("/kb_match", data="CrashLoopBackOff")
        assert resp.status == 200
        body = await resp.text()
        assert "KB RULES" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kb_match_empty_body_returns_400():
    client, checker = await _make_client()
    try:
        resp = await client.post("/kb_match", data="")
        assert resp.status == 400
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kb_match_kb_store_failure_returns_safe_empty():
    client, checker = await _make_client()
    try:
        store = MagicMock()
        store.get_relevant_rules = MagicMock(side_effect=RuntimeError("db down"))
        checker.set_kb_store(store)
        resp = await client.post("/kb_match", data="CrashLoopBackOff")
        assert resp.status == 200
        body = await resp.text()
        assert "KB RULES" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kb_stats_returns_top_n_by_match_count():
    client, checker = await _make_client()
    try:
        store = MagicMock()
        store.top_by_match_count = MagicMock(return_value=[
            {"rule_id": "r1", "issue_title": "OOM", "rule_text": "x",
             "match_count": 12, "last_matched_at": None},
        ])
        checker.set_kb_store(store)
        resp = await client.get("/kb_stats?limit=10")
        assert resp.status == 200
        data = await resp.json()
        assert "rules" in data
        assert len(data["rules"]) == 1
        assert data["rules"][0]["match_count"] == 12
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kb_stats_limit_invalid_falls_back_to_default():
    client, checker = await _make_client()
    try:
        store = MagicMock()
        store.top_by_match_count = MagicMock(return_value=[])
        checker.set_kb_store(store)
        resp = await client.get("/kb_stats?limit=not-a-number")
        assert resp.status == 200
        args, _ = store.top_by_match_count.call_args
        assert args[0] == 20
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_kb_stats_no_store_returns_empty_list():
    client, checker = await _make_client()
    try:
        resp = await client.get("/kb_stats")
        assert resp.status == 200
        data = await resp.json()
        assert data == {"rules": []}
    finally:
        await client.close()
