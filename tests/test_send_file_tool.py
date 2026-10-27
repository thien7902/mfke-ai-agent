"""Tests for the chat_send_file / list_chat_files Holmes tools."""
import importlib
from unittest.mock import MagicMock, patch

import pytest

from src.bot.tools.send_file_tool import ListChatFilesTool, SendFileTool
from src.bot.utils import file_sender

# `src/bot/utils/__init__.py` re-exports the `config` instance, shadowing the
# submodule of the same name — import the module explicitly to patch it.
config_module = importlib.import_module("src.bot.utils.config")


def _context(chat_id="12345", topic_id="99"):
    ctx = MagicMock()
    ctx.request_context = {"user_id": "cli", "chat_id": chat_id, "topic_id": topic_id}
    return ctx


def _context_without_chat():
    ctx = MagicMock()
    ctx.request_context = {"user_id": "cli"}
    return ctx


@pytest.fixture(autouse=True)
def outbox(tmp_path, monkeypatch):
    d = tmp_path / "outbox"
    d.mkdir()
    monkeypatch.setattr(config_module.config, "file_send_outbox_dir", str(d))
    monkeypatch.setattr(config_module.config, "file_send_max_bytes", 50 * 1024 * 1024)
    file_sender.reset_file_sender()
    yield d
    file_sender.reset_file_sender()


def test_tool_metadata():
    tool = SendFileTool()
    fmt = tool.get_openai_format()
    assert fmt["function"]["name"] == "chat_send_file"
    props = fmt["function"]["parameters"]["properties"]
    assert "path" in props
    assert "caption" in props
    # Holmes runs tools in OpenAI strict mode, where every declared parameter
    # is listed as required and optional ones admit null — `caption` must
    # therefore still be callable with null.
    assert "path" in fmt["function"]["parameters"]["required"]
    assert any(
        branch.get("type") == "null" for branch in props["caption"].get("anyOf", [])
    ), "caption must accept null so the model can omit it"

    listing = ListChatFilesTool()
    assert listing.get_openai_format()["function"]["name"] == "list_chat_files"


def test_invoke_sends_file(outbox):
    target = outbox / "report.txt"
    target.write_text("hello")

    with patch.object(file_sender, "send_document_sync", return_value="msg") as send:
        result = SendFileTool().invoke(
            {"path": str(target), "caption": "your report"}, context=_context()
        )

    assert result.status.value == "success"
    assert "Sent 'report.txt'" in result.data
    assert "5 B" in result.data
    send.assert_called_once()
    args, kwargs = send.call_args
    assert args[0] == 12345
    assert args[1] == str(target.resolve())
    assert kwargs["topic_id"] == 99
    assert kwargs["caption"] == "your report"


def test_invoke_resolves_relative_path(outbox):
    (outbox / "data.csv").write_text("a,b")
    with patch.object(file_sender, "send_document_sync") as send:
        result = SendFileTool().invoke({"path": "data.csv"}, context=_context())
    assert result.status.value == "success"
    assert send.call_args[0][1] == str((outbox / "data.csv").resolve())


def test_invoke_without_chat_context_errors(outbox):
    (outbox / "x.txt").write_text("data")
    with patch.object(file_sender, "send_document_sync") as send:
        result = SendFileTool().invoke({"path": "x.txt"}, context=_context_without_chat())
    assert result.status.value == "error"
    assert "No chat is associated" in result.error
    send.assert_not_called()


def test_invoke_missing_path_errors():
    result = SendFileTool().invoke({}, context=_context())
    assert result.status.value == "error"
    assert "Missing 'path'" in result.error


def test_invoke_outside_outbox_errors():
    result = SendFileTool().invoke({"path": "/etc/passwd"}, context=_context())
    assert result.status.value == "error"
    assert "outside the sendable directory" in result.error


def test_invoke_credential_name_errors(outbox):
    (outbox / "kubeconfig.enc").write_text("secret")
    result = SendFileTool().invoke({"path": "kubeconfig.enc"}, context=_context())
    assert result.status.value == "error"
    assert "credential-like" in result.error


def test_invoke_unconfigured_sender_errors(outbox):
    (outbox / "x.txt").write_text("data")
    result = SendFileTool().invoke({"path": "x.txt"}, context=_context())
    assert result.status.value == "error"
    assert "not configured" in result.error


def test_invoke_telegram_failure_becomes_tool_error(outbox):
    (outbox / "x.txt").write_text("data")
    with patch.object(file_sender, "send_document_sync", side_effect=RuntimeError("bad request")):
        result = SendFileTool().invoke({"path": "x.txt"}, context=_context())
    assert result.status.value == "error"
    assert "Telegram rejected the upload" in result.error


def test_invoke_timeout_becomes_tool_error(outbox):
    (outbox / "x.txt").write_text("data")
    with patch.object(file_sender, "send_document_sync", side_effect=TimeoutError("too slow")):
        result = SendFileTool().invoke({"path": "x.txt"}, context=_context())
    assert result.status.value == "error"
    assert "too slow" in result.error


def test_list_tool_empty_outbox_names_directory(outbox):
    result = ListChatFilesTool().invoke({}, context=_context())
    assert result.status.value == "success"
    assert str(outbox) in result.data
    assert "empty" in result.data.lower()


def test_list_tool_lists_files(outbox):
    (outbox / "a.txt").write_text("1")
    (outbox / "b.csv").write_text("2,3")
    result = ListChatFilesTool().invoke({}, context=_context())
    assert "a.txt" in result.data
    assert "b.csv" in result.data
