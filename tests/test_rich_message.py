"""Tests for src.bot.utils.rich_message.

Covers GFM table detection, markdown→HTML conversion, the fallback
send path, and the auto-router. Uses fake bots — no network.
"""
import asyncio
from typing import Any, List

import pytest

from src.bot.utils import rich_message


# ---------- has_rich_table ----------

def test_has_table_basic():
    text = "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |"
    assert rich_message.has_rich_table(text) is True


def test_has_table_without_leading_pipe():
    text = "Name | Age\n--- | ---\nAlice | 30\nBob | 25"
    assert rich_message.has_rich_table(text) is True


def test_has_table_with_alignment_colons():
    text = "| Name | Age |\n| :--- | ---: |\n| Alice | 30 |"
    assert rich_message.has_rich_table(text) is True


def test_has_table_prose_no_false_positive():
    text = "I have a pipe | here but no table.\nAnother line without pipes."
    assert rich_message.has_rich_table(text) is False


def test_has_table_separator_only_no_header():
    text = "--- | ---\nfoo | bar"
    # First line is separator-only, no header — not a GFM table.
    # Our detector checks header first; if first line is separator, we
    # look for a separator after a non-separator line. This text starts
    # with a separator row, so there's no header detected.
    assert rich_message.has_rich_table(text) is False


def test_has_table_setext_heading_not_table():
    text = "Title\n-----\n\nSome paragraph below."
    assert rich_message.has_rich_table(text) is False


def test_has_table_empty_text():
    assert rich_message.has_rich_table("") is False
    assert rich_message.has_rich_table(None) is False


# ---------- markdown_to_html: inline ----------

def test_md_bold_italic():
    assert rich_message.markdown_to_html("**bold**") == "<b>bold</b>"
    assert rich_message.markdown_to_html("*italic*") == "<i>italic</i>"


def test_md_inline_code():
    out = rich_message.markdown_to_html("`code`")
    assert "<code>code</code>" in out


def test_md_link():
    out = rich_message.markdown_to_html("[text](https://example.com)")
    assert '<a href="https://example.com">text</a>' in out


def test_md_heading_to_bold():
    out = rich_message.markdown_to_html("## Heading 2\n\nbody")
    assert "<b>Heading 2</b>" in out


def test_md_strike():
    assert rich_message.markdown_to_html("~~strike~~") == "<s>strike</s>"


def test_md_spoiler():
    out = rich_message.markdown_to_html("||secret||")
    assert '<span class="tg-spoiler">secret</span>' in out


def test_md_fenced_code_escapes_html():
    out = rich_message.markdown_to_html("```python\nprint('<script>')\n```")
    assert "<pre><code" in out
    assert "&lt;script&gt;" in out
    # Ensure the <script> wasn't double-escaped or left raw.
    assert "<script>" not in out.replace("&lt;script&gt;", "")


def test_md_inline_code_in_paragraph():
    out = rich_message.markdown_to_html("Use `kubectl get pods` to list.")
    assert "<code>kubectl get pods</code>" in out


# ---------- markdown_to_html: tables ----------

def test_md_table_to_pre_grid():
    text = "| Name | Age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |"
    out = rich_message.markdown_to_html(text)
    assert "<pre>" in out
    assert "Name" in out
    assert "Alice" in out
    # Header row padded aligned; separator dashed line below header.
    assert "-+-" in out  # column separator


def test_md_table_html_in_cell_escaped():
    # The `<` in a cell must be escaped so it doesn't break HTML parsing.
    text = "| Cmd |\n| --- |\n| `<script>` |"
    out = rich_message.markdown_to_html(text)
    assert "&lt;script&gt;" in out


def test_md_table_ragged_falls_to_plain_pre():
    # Ragged rows (varying column counts) → can't align → plain escaped <pre>.
    text = "| A | B |\n| --- | --- |\n| 1 |\n| 2 | 3 |"
    out = rich_message.markdown_to_html(text)
    assert out.startswith("<pre>") and out.endswith("</pre>")
    # No alignment dashes.
    assert "-+-" not in out


# ---------- markdown_to_html: misc ----------

def test_md_escape_html_in_text():
    out = rich_message.markdown_to_html("a < b > c & d")
    assert "&lt;" in out
    assert "&gt;" in out
    assert "&amp;" in out


def test_md_combined_bold_link():
    out = rich_message.markdown_to_html("**see [docs](https://x.com)**")
    assert "<b>see <a href=\"https://x.com\">docs</a></b>" in out


# ---------- send_rich_message fallback contract ----------

class FakeBot:
    """Minimal bot fake. Records calls to send_message and _post."""
    def __init__(self, rich_should_fail: bool = False):
        self.rich_should_fail = rich_should_fail
        self.calls: List[Any] = []

    async def _post(self, method, data=None):
        self.calls.append((method, data))
        if self.rich_should_fail:
            raise RuntimeError("sendRichMessage not supported")
        return {"result": {"message_id": 1, "chat": {"id": 1}, "text": "ok"}}

    async def send_message(self, **kwargs):
        self.calls.append(("send_message", kwargs))
        return {"message_id": 2, "text": kwargs.get("text", "")}


def test_send_rich_falls_back_to_html_when_endpoint_fails():
    bot = FakeBot(rich_should_fail=True)
    text = "| H |\n| --- |\n| data |"
    sent = asyncio.run(rich_message.send_rich_message(
        bot, chat_id=1, text=text, message_thread_id=None
    ))
    # Rich attempted, then HTML fallback ran.
    methods = [c[0] for c in bot.calls]
    assert "sendRichMessage" in methods
    assert "send_message" in methods
    # The fallback chunk should be HTML.
    send_call = [c for c in bot.calls if c[0] == "send_message"][0]
    assert send_call[1].get("parse_mode") == "HTML"
    assert "<pre>" in send_call[1]["text"]


def test_send_rich_reply_auto_prose_uses_html_path():
    bot = FakeBot(rich_should_fail=False)
    asyncio.run(rich_message.send_rich_reply(
        bot, chat_id=1, text="Just prose, no table here."
    ))
    methods = [c[0] for c in bot.calls]
    # Prose and prefer_rich=None → no rich attempt; HTML path used.
    assert "sendRichMessage" not in methods
    assert "send_message" in methods


def test_send_rich_reply_auto_table_uses_rich_path():
    bot = FakeBot(rich_should_fail=False)
    text = "| H |\n| --- |\n| data |"
    asyncio.run(rich_message.send_rich_reply(
        bot, chat_id=1, text=text
    ))
    methods = [c[0] for c in bot.calls]
    assert "sendRichMessage" in methods
    assert "send_message" not in methods


def test_send_rich_reply_force_prefer_rich_false():
    bot = FakeBot(rich_should_fail=False)
    text = "| H |\n| --- |\n| data |"
    asyncio.run(rich_message.send_rich_reply(
        bot, chat_id=1, text=text, prefer_rich=False
    ))
    methods = [c[0] for c in bot.calls]
    assert "sendRichMessage" not in methods
    assert "send_message" in methods


def test_send_rich_message_empty_text():
    bot = FakeBot()
    sent = asyncio.run(rich_message.send_rich_message(bot, chat_id=1, text=""))
    assert sent == []
    assert bot.calls == []


# ---------- edit_or_send_rich (dedup contract) ----------

class FakeMessage:
    """Fake python-telegram-bot Message. Records edit_text/delete calls."""
    def __init__(self, edit_should_fail: bool = False):
        self.edit_should_fail = edit_should_fail
        self.edits: List[Any] = []
        self.deleted = False

    async def edit_text(self, text, parse_mode=None, **kw):
        self.edits.append({"text": text, "parse_mode": parse_mode})
        if self.edit_should_fail:
            raise RuntimeError("message is too long")
        return self

    async def delete(self, **kw):
        self.deleted = True


def test_edit_or_send_rich_edits_in_place_on_success():
    """Happy path: edit_text succeeds → message edited in place, no new
    message sent. parse_mode is HTML (formatted)."""
    bot = FakeBot(rich_should_fail=False)
    msg = FakeMessage(edit_should_fail=False)
    asyncio.run(rich_message.edit_or_send_rich(
        bot, chat_id=1, message=msg,
        text="**bold** and *italic*", message_thread_id=None,
    ))
    assert len(msg.edits) == 1
    assert msg.edits[0]["parse_mode"] == "HTML"
    assert "<b>bold</b>" in msg.edits[0]["text"]
    assert msg.deleted is False
    assert bot.calls == [], "no new message sent on happy path"


def test_edit_or_send_rich_deletes_and_resends_on_edit_failure():
    """Bug: previously, edit_text threw and the code sent a NEW rich
    message WITHOUT deleting the original, so user saw two copies of
    the same content (raw markdown on the edited one, formatted on
    the new one). Fix: on edit failure, delete the streaming message
    THEN send the rich reply."""
    bot = FakeBot(rich_should_fail=False)
    msg = FakeMessage(edit_should_fail=True)
    asyncio.run(rich_message.edit_or_send_rich(
        bot, chat_id=1, message=msg,
        text="**bold** reply", message_thread_id=None,
    ))
    assert msg.deleted is True, "streaming message must be deleted on edit failure"
    methods = [c[0] for c in bot.calls]
    assert "send_message" in methods, "a fresh rich message must be sent after delete"
    # The new message must be HTML-formatted (not raw markdown).
    send_call = [c for c in bot.calls if c[0] == "send_message"][0]
    assert send_call[1].get("parse_mode") == "HTML"
    assert "<b>bold</b>" in send_call[1]["text"]


def test_edit_or_send_rich_empty_text_noop():
    bot = FakeBot()
    msg = FakeMessage()
    asyncio.run(rich_message.edit_or_send_rich(
        bot, chat_id=1, message=msg, text="", message_thread_id=None,
    ))
    assert msg.edits == []
    assert msg.deleted is False
    assert bot.calls == []
