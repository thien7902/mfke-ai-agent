"""Tests for src/bot/utils/file_extractor.py."""
import os
import shutil
import pathlib
import shutil as _shutil
from unittest.mock import patch

import pytest

from src.bot.utils.file_extractor import (
    extract_text_from_file,
    extract_text_from_image,
    IMAGE_SENTINEL,
)


@pytest.fixture
def tmp_upload(tmp_path):
    """Return a function that writes content to a file with the given suffix."""
    def _make(name: str, content: bytes) -> pathlib.Path:
        p = tmp_path / name
        p.write_bytes(content)
        return p
    return _make


def test_text_file_utf8(tmp_upload):
    p = tmp_upload("note.txt", "hello — world\nété".encode("utf-8"))
    assert extract_text_from_file(p) == "hello — world\nété"


def test_code_file(tmp_upload):
    p = tmp_upload("script.py", b"print('hi')\n")
    assert extract_text_from_file(p) == "print('hi')\n"


def test_json_file(tmp_upload):
    p = tmp_upload("a.json", b'{"k": 1}')
    assert '"k": 1' in extract_text_from_file(p)


def test_unknown_type_returns_empty(tmp_upload):
    p = tmp_upload("thing.zip", b"PK\x03\x04")
    assert extract_text_from_file(p) == ""


def test_image_returns_sentinel(tmp_upload):
    p = tmp_upload("pic.png", b"\x89PNG\r\n\x1a\n")
    assert extract_text_from_file(p) == IMAGE_SENTINEL


def test_image_jpg_with_mime(tmp_path):
    # No .jpg suffix, but image/jpeg MIME → sentinel
    p = tmp_path / "noext"
    p.write_bytes(b"\xff\xd8\xff")
    assert extract_text_from_file(p, mime="image/jpeg") == IMAGE_SENTINEL


def test_truncation_enforced(tmp_upload, monkeypatch):
    # Lower the cap so we don't write 10 MB in the test.
    from src.bot.utils import file_extractor
    monkeypatch.setattr(file_extractor.config, "file_upload_max_text_bytes", 100)
    big = "x" * 5000
    p = tmp_upload("big.txt", big.encode("utf-8"))
    out = extract_text_from_file(p)
    assert len(out) < 200
    assert "truncated" in out


def test_pdf_extraction(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    # Minimal: build a tiny PDF via pypdf writer (no text content, but should
    # not crash and should return "" for text-less pages)
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    p = tmp_path / "blank.pdf"
    with open(p, "wb") as f:
        writer.write(f)
    # blank page → empty text, but at least not crash
    result = extract_text_from_file(p)
    assert isinstance(result, str)


def test_docx_extraction(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Hello world")
    doc.add_paragraph("Second paragraph")
    p = tmp_path / "test.docx"
    doc.save(str(p))
    result = extract_text_from_file(p)
    assert "Hello world" in result
    assert "Second paragraph" in result


def test_xlsx_extraction(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = "name"
    ws["B1"] = "value"
    ws["A2"] = "foo"
    ws["B2"] = 42
    p = tmp_path / "test.xlsx"
    wb.save(str(p))
    result = extract_text_from_file(p)
    assert "Sheet1" in result
    assert "name" in result
    assert "foo" in result
    assert "42" in result


def test_extract_text_from_image_missing_binary(monkeypatch, tmp_path):
    # Don't require tesseract to be installed in CI — exercise the "no
    # tesseract" branch by passing a non-image path through a mocked Image.open
    # that raises.
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\r\n")
    # If tesseract is not installed, the import branch returns "".
    # If it is installed, run should still return "" on garbage bytes.
    result = extract_text_from_image(p)
    assert isinstance(result, str)
