"""Tests for partial tool-call persistence on /stop and the prior-stop
system-prompt note injected on the next turn.

Covers:
- `_stream_chat` `finally` block persists tool_calls + tool_results accumulated
  before an `LLMInterruptedError` mid-stream.
- The persisted assistant message is stamped with `metadata.stopped_by_user`
  and `metadata.original_user_request`.
- `_build_holmes_messages` injects the `--- PRIOR REQUEST INTERRUPTED ---`
  note when the most recent assistant message has `stopped_by_user=True`.
- No note when no such marker exists, or when the marker is too old.
"""
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from holmes.core.tool_calling_llm import LLMInterruptedError

from src.bot.models.conversation import Conversation, Message
from src.bot.services.holmes_service import HolmesService


def _make_service_stub() -> HolmesService:
    """Build a HolmesService that's NOT initialized — we call private
    methods only. We monkey-patch `_run_holmes_stream` per test."""
    return HolmesService.__new__(HolmesService)


# ------------ _prior_stop_note ------------

def test_prior_stop_note_injects_when_stopped_recent():
    svc = _make_service_stub()
    conv = Conversation(chat_id=1, user_id=100)
    conv.add_message(Message(role="user", content="investigate pod X crash"))
    asst = Message(role="assistant", content="partial...",
                   tool_calls=[{"id": "x", "type": "function",
                                "function": {"name": "bash", "arguments": "{}"}}])
    asst.metadata["stopped_by_user"] = True
    asst.metadata["original_user_request"] = "investigate pod X crash"
    conv.add_message(asst)
    note = svc._prior_stop_note(conv)
    assert "--- PRIOR REQUEST INTERRUPTED ---" in note
    assert "investigate pod X crash" in note
    assert "deliberate retry" in note.lower() or "deliberately" in note.lower() or "retry" in note.lower()


def test_prior_stop_note_no_stopped_message_returns_empty():
    svc = _make_service_stub()
    conv = Conversation(chat_id=1, user_id=100)
    conv.add_message(Message(role="user", content="hi"))
    conv.add_message(Message(role="assistant", content="hello"))
    assert svc._prior_stop_note(conv) == ""


def test_prior_stop_note_no_conversation_returns_empty():
    svc = _make_service_stub()
    assert svc._prior_stop_note(None) == ""


def test_prior_stop_note_empty_original_returns_empty():
    svc = _make_service_stub()
    conv = Conversation(chat_id=1, user_id=100)
    asst = Message(role="assistant", content="partial")
    asst.metadata["stopped_by_user"] = True
    asst.metadata["original_user_request"] = ""  # empty
    conv.add_message(asst)
    assert svc._prior_stop_note(conv) == ""


def test_prior_stop_note_too_old_returns_empty():
    """An old stopped message (more than 5 messages back) should NOT trigger
    a note — the user has moved on."""
    svc = _make_service_stub()
    conv = Conversation(chat_id=1, user_id=100)
    asst_stopped = Message(role="assistant", content="part")
    asst_stopped.metadata["stopped_by_user"] = True
    asst_stopped.metadata["original_user_request"] = "old request"
    conv.add_message(asst_stopped)
    # 5 fresh messages pushing the stopped one out of the window.
    for i in range(5):
        conv.add_message(Message(role="user", content=f"new request {i}"))
        conv.add_message(Message(role="assistant", content=f"new reply {i}"))
    assert svc._prior_stop_note(conv) == ""


# ------------ _stream_chat partial-persistence on interrupt ------------

@pytest.mark.asyncio
async def test_stream_chat_persists_partial_on_interrupt():
    """When _run_holmes_stream yields tool_call + tool_result events then
    raises LLMInterruptedError, _stream_chat should still persist the
    accumulated tool_calls + tool_results to the conversation."""
    svc = _make_service_stub()

    conv = Conversation(chat_id=1, user_id=100)
    conv.add_message(Message(role="user", content="investigate pod X"))

    # Event stream: 1 tool_call (with id) → 1 tool_result → raise.
    async def fake_stream(*args, **kwargs):
        yield {"type": "tool_call", "tool_name": "bash",
                "tool_call_id": "call-a", "params": {"command": "kubectl get pods"}}
        yield {"type": "tool_result", "tool_name": "bash",
                "tool_call_id": "call-a",
                "result": {"data": "pod-name  Ready"}}
        raise LLMInterruptedError("user stopped")

    svc._run_holmes_stream = fake_stream

    yielded = []
    try:
        async for evt in svc._stream_chat(
            messages=[], features={"tool_approval": False}, user_id=100,
            conversation=conv, telegram_context=None, chat_id=1, topic_id=0,
            cancel_event=None,
        ):
            yielded.append(evt)
    except LLMInterruptedError:
        pass

    types = [e.get("type") for e in yielded]
    assert "tool_call" in types
    assert "tool_result" in types

    msgs = conv.messages
    roles = [m.role for m in msgs]
    assert "assistant" in roles
    assert "tool" in roles
    asst = next(m for m in msgs if m.role == "assistant")
    assert asst.tool_calls, "expected tool_calls persisted"
    assert asst.tool_calls[0]["id"] == "call-a"
    assert asst.metadata.get("stopped_by_user") is True
    assert asst.metadata.get("original_user_request") == "investigate pod X"
    tool_msgs = [m for m in msgs if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call-a"
    assert "pod-name" in tool_msgs[0].content


@pytest.mark.asyncio
async def test_stream_chat_no_persist_when_nothing_accumulated():
    """If the stream raises immediately without yielding anything, _stream_chat
    should NOT add an empty assistant message."""
    svc = _make_service_stub()
    conv = Conversation(chat_id=1, user_id=100)
    conv.add_message(Message(role="user", content="hello"))

    async def empty_stream(*args, **kwargs):
        raise LLMInterruptedError("instant stop")
        yield  # unreachable; makes this a generator

    svc._run_holmes_stream = empty_stream

    try:
        async for _ in svc._stream_chat(
            messages=[], features={"tool_approval": False}, user_id=100,
            conversation=conv, cancel_event=None,
        ):
            pass
    except LLMInterruptedError:
        pass

    assert len(conv.messages) == 1
    assert conv.messages[0].role == "user"


@pytest.mark.asyncio
async def test_stream_chat_success_path_still_persists():
    """Without interrupt, _stream_chat should persist assistant content and
    tool calls exactly as before (no regression on the happy path)."""
    svc = _make_service_stub()
    conv = Conversation(chat_id=1, user_id=100)
    conv.add_message(Message(role="user", content="ask"))

    async def good_stream(*args, **kwargs):
        yield {"type": "content", "content": "Answer: "}
        yield {"type": "content", "content": "yes"}
        yield {"type": "tool_call", "tool_name": "bash",
                "tool_call_id": "call-b", "params": {"command": "ls"}}
        yield {"type": "tool_result", "tool_name": "bash",
                "tool_call_id": "call-b", "result": {"data": "file1\nfile2"}}

    svc._run_holmes_stream = good_stream

    async for _ in svc._stream_chat(
        messages=[], features={"tool_approval": False}, user_id=100,
        conversation=conv, cancel_event=None,
    ):
        pass

    asst = next(m for m in conv.messages if m.role == "assistant")
    assert asst.content == "Answer: yes"
    assert asst.tool_calls and asst.tool_calls[0]["id"] == "call-b"
    assert not asst.metadata.get("stopped_by_user")
