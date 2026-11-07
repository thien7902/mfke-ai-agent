"""Tests for the docx parser + distill step inside scripts.ingest_kb_docx.

We build a tiny synthetic .docx in-memory using stdlib (zipfile + hand-
rolled XML) so we don't need python-docx. The fixture exercises:
- Heading1 KB article start (`^[KB\\d{8}-\\d{3}\\]`)
- `## Symptoms` / `## Resolution Steps` markdown section markers
- One code block paragraph (kubectl-suffixed)
- One "Nhờ KT..." chatter line (must be skipped)
- Two distinct KB IDs → two articles
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from xml.sax.saxutils import escape

import pytest

from scripts.ingest_kb_docx import (
    KBArticle,
    parse_docx,
    _parse_kb_heading,
    _looks_like_code,
    _call_distill_llm,
    ingest,
)


# --------------------------------------------------------------------- #
# Synthetic docx builder
# --------------------------------------------------------------------- #

_DOCX_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
<w:body>
{paragraphs}
</w:body>
</w:document>
"""

_RELS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{rels}
</Relationships>
"""


def _para(text: str) -> str:
    return f"<w:p><w:r><w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r></w:p>"


def _make_docx(paragraphs, image_rels=None, image_bytes=None) -> bytes:
    body = "\n".join(_para(p) for p in paragraphs)
    doc_xml = _DOCX_TEMPLATE.format(paragraphs=body)
    rels_xml = _RELS_TEMPLATE.format(
        rels="".join(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>'
            for rid, target in (image_rels or {}).items()
        )
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", doc_xml)
        z.writestr("word/_rels/document.xml.rels", rels_xml)
        for name, data in (image_bytes or {}).items():
            z.writestr(name, data)
    return buf.getvalue()


# globals image_bytes injected lazily inside _make_docx via closure


def test_parse_two_kb_articles(tmp_path):
    paras = [
        "[KB20250929-001] FKE [Known-Issue] Cluster K8S không truy cập được",
        "## 📌 Overview / Purpose",
        "Cluster không phản hồi API.",
        "## 🚨 Symptoms / Problem Description",
        "kubectl get pods trả về connection refused.",
        "## 🛠 Resolution Steps",
        "1. Kiểm tra kube-apiserver pod",
        "kubectl -n kube-system get pods -l component=kube-apiserver",
        "[KB20250929-002] FKE [Known-Issue] Service LB bị treo",
        "## 🚨 Symptoms / Problem Description",
        "Service type LoadBalancer ở trạng thái Pending.",
        "## 🛠 Resolution Steps",
        "Kiểm tra cloud-controller-manager logs.",
    ]
    p = tmp_path / "test.docx"
    p.write_bytes(_make_docx(paras))
    articles = parse_docx(p)
    assert len(articles) == 2
    assert articles[0].issue_id == "KB20250929-001"
    assert "Cluster K8S không truy cập" in articles[0].title
    assert "kubectl get pods" in articles[0].symptoms
    assert "kube-apiserver pod" in articles[0].resolution
    # Code block identified by kubectl keyword.
    assert len(articles[0].code_blocks) >= 1
    assert articles[1].issue_id == "KB20250929-002"
    assert "LoadBalancer" in articles[1].symptoms


def test_skip_nho_kt_chatter(tmp_path):
    paras = [
        "[KB20250930-001] FKE [Known-Issue] Test",
        "## 🚨 Symptoms / Problem Description",
        "Symptom line A",
        "+ Nhờ team hạ tầng xóa LB racy",
        "Symptom line B (must be kept, chatter above must vanish).",
    ]
    p = tmp_path / "test.docx"
    p.write_bytes(_make_docx(paras))
    articles = parse_docx(p)
    assert len(articles) == 1
    all_text = articles[0].symptoms
    assert "Nhờ" not in all_text
    assert "Symptom line A" in all_text
    assert "Symptom line B" in all_text


def test_duplicate_id_different_title_yields_two_records(tmp_path):
    """Explore agent verified some KB IDs are reused across distinct articles.
    The parser must NOT dedupe on issue_id alone — (issue_id, title) is the
    real identity.
    """
    paras = [
        "[KB20251224-001] FKE first article",
        "## 🚨 Symptoms / Problem Description",
        "Symptom one.",
        "[KB20251224-001] NCP GPU Container second article (same ID, different title)",
        "## 🚨 Symptoms / Problem Description",
        "Symptom two.",
    ]
    p = tmp_path / "test.docx"
    p.write_bytes(_make_docx(paras))
    articles = parse_docx(p)
    assert len(articles) == 2
    assert articles[0].title != articles[1].title


def test_parse_kb_heading():
    rid, title = _parse_kb_heading("[KB20250929-002] FKE [Known-Issue] something")
    assert rid == "KB20250929-002"
    assert title == "[KB20250929-002] FKE [Known-Issue] something"


def test_looks_like_code_keyword_match():
    assert _looks_like_code("kubectl get pods") is True
    assert _looks_like_code("apiVersion: v1") is True
    assert _looks_like_code("Plain Vietnamese description.") is False


# --------------------------------------------------------------------- #
# Distill
# --------------------------------------------------------------------- #

def test_distill_happy_path():
    """Mock the OpenAI client so _call_distill_llm parses strict JSON."""
    import scripts.ingest_kb_docx as mod
    rules_text = [
        "IF pod CrashLoopBackOff AND OOMKilled -> bump memory",
        "IF node DiskPressure -> drain node and check ephemeral storage",
    ]
    payload = {"rules": [{"text": r} for r in rules_text]}
    with patch.object(mod, "bot_config") as mock_cfg, \
         patch.object(mod, "openai") as mock_openai_module, \
         patch.dict("os.environ", {"OPENAI_API_KEY": "k"}):
        mock_cfg.kb_distill_model = "test-model"
        mock_cfg.consolidator_model = None
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(payload)))]
        )
        mock_openai_module.OpenAI.return_value = client
        out = mod._call_distill_llm("dummy article text")
    assert out == rules_text


def test_distill_invalid_json_returns_empty():
    import scripts.ingest_kb_docx as mod
    with patch.object(mod, "bot_config") as mock_cfg, \
         patch.object(mod, "openai") as mock_openai_module, \
         patch.dict("os.environ", {"OPENAI_API_KEY": "k"}):
        mock_cfg.kb_distill_model = "test-model"
        mock_cfg.consolidator_model = None
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not json"))]
        )
        mock_openai_module.OpenAI.return_value = client
        out = mod._call_distill_llm("dummy article text")
    assert out == []


def test_distill_strips_markdown_fences():
    import scripts.ingest_kb_docx as mod
    payload = {"rules": [{"text": "IF x THEN y -> z"}]}
    raw = "```json\n" + json.dumps(payload) + "\n```"
    with patch.object(mod, "bot_config") as mock_cfg, \
         patch.object(mod, "openai") as mock_openai_module, \
         patch.dict("os.environ", {"OPENAI_API_KEY": "k"}):
        mock_cfg.kb_distill_model = "test-model"
        mock_cfg.consolidator_model = None
        client = MagicMock()
        client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=raw))]
        )
        mock_openai_module.OpenAI.return_value = client
        out = mod._call_distill_llm("dummy")
    assert out == ["IF x THEN y -> z"]


def test_distill_no_model_returns_empty():
    import scripts.ingest_kb_docx as mod
    with patch.object(mod, "bot_config") as mock_cfg:
        mock_cfg.kb_distill_model = None
        mock_cfg.consolidator_model = None
        out = mod._call_distill_llm("dummy")
    assert out == []


def test_distill_empty_rules_skipped_at_ingest():
    """Article whose LLM distill returns [] should not produce any rule docs."""
    articles = [KBArticle(issue_id="KB2025-001", title="t", symptoms="s")]
    with patch("scripts.ingest_kb_docx._call_distill_llm", return_value=[]):
        # Avoid real Mongo: replace get_collection with a MagicMock.
        with patch("scripts.ingest_kb_docx.MongoDB") as mock_mongo:
            coll = MagicMock()
            mock_mongo.get_collection.return_value = coll
            upserted, skipped, errors = ingest(articles, purge=False)
    assert upserted == 0
    assert skipped == 1
    assert errors == 0
    coll.update_one.assert_not_called()
    coll.delete_many.assert_not_called()


def test_distill_three_rules_upserts_with_indexed_rule_ids():
    """Three rules from one article → three upsert calls with rule_ids
    'KB2025-001#1', '#2', '#3'.
    """
    rules = [
        "IF pod CrashLoopBackOff AND OOMKilled -> bump memory",
        "IF Shoot ReconciliationFailed AND mcm log timeout -> restart mcm",
        "IF etcd-main-0 DiskPressure -> check etcd corruption from backup-restore",
    ]
    articles = [KBArticle(issue_id="KB2025-001", title="t", symptoms="s", resolution="r")]
    with patch("scripts.ingest_kb_docx._call_distill_llm", return_value=rules), \
         patch("scripts.ingest_kb_docx.embed_text", return_value=[0.1] * 1024), \
         patch("scripts.ingest_kb_docx.MongoDB") as mock_mongo:
        coll = MagicMock()
        mock_mongo.get_collection.return_value = coll
        upserted, _skipped, errors = ingest(articles, purge=False)
    assert upserted == 3
    assert errors == 0
    rule_ids = [
        call.args[0]["rule_id"]
        for call in coll.update_one.call_args_list
    ]
    assert rule_ids == ["KB2025-001#1", "KB2025-001#2", "KB2025-001#3"]


def test_distill_purge_drops_collection_first():
    articles = []
    with patch("scripts.ingest_kb_docx.MongoDB") as mock_mongo:
        coll = MagicMock()
        mock_mongo.get_collection.return_value = coll
        ingest(articles, purge=True)
    coll.delete_many.assert_called_once_with({})


# Need json import for the patch.content test setups above.

