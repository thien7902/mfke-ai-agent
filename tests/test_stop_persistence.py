"""Tests for partial tool-call persistence on /stop and the inline
`role="system"` note appended to conversation history at the interruption
point (Claude-Code-style inline marker, not a system-prompt injection).

Covers:
- `_stream_chat` `finally` block persists tool_calls + tool_results
  accumulated before an `LLMInterruptedError` mid-stream.
- The persisted assistant message is stamped with
  `metadata.stopped_by_user=True` and `metadata.original_user_request`.
- `_stream_chat` `finally` block appends an inline
  `Message(role="system", content="[Previous turn was interrupted...]")`
  to `conversation.messages` when stopped AND `original_request` is
  non-empty.
- No inline note on the success path.
- No inline note when `original_request` is missing (empty/None).
- `_build_holmes_messages` does NOT inject a prior-stop note into
  system_prompt_additions (the v2.3.8 system-prompt approach was
  removed in v2.3.9).
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


# ------------ _build_holmes_messages: no prior-stop injection ------------

def test_build_holmes_messages_no_prior_stop_injection():
    """v2.3.9: prior-stop signal is an inline `role="system"` Message in
    conversation.history, NOT a system_prompt_additions injection.

    We assert this at the source level — `_build_holmes_messages` (and
    helpers it calls) must NOT emit a `--- PRIOR REQUEST INTERRUPTED ---`
    header. The v2.3.8 `_prior_stop_note` method was removed in v2.3.9.
    """
    import inspect
    from src.bot.services import holmes_service as mod

    # The helper from v2.3.8 must no longer exist.
    assert not hasattr(HolmesService, "_prior_stop_note"), (
        "_prior_stop_note was removed in v2.3.9 — inline system Message "
        "in conversation.history replaces it."
    )
    # And no PRIOR REQUEST INTERRUPTED header anywhere in the module.
    src = inspect.getsource(mod)
    assert "--- PRIOR REQUEST INTERRUPTED ---" not in src
    assert "PRIOR REQUEST INTERRUPTED" not in src


# ------------ _stream_chat partial-persistence on interrupt ------------

@pytest.mark.asyncio
async def test_stream_chat_persists_partial_on_interrupt():
    """When _run_holmes_stream yields tool_call + tool_result events then
    raises LLMInterruptedError, _stream_chat should still persist the
    accumulated tool_calls + tool_results to the conversation and stamp
    the assistant message with stopped_by_user metadata."""
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
async def test_stream_chat_appends_inline_system_note_on_stop():
    """v2.3.9: On interrupt with `original_request` non-empty, _stream_chat
    should append a `role="system"` Message describing the interrupted
    turn to the conversation history (Claude-Code-style inline marker)."""
    svc = _make_service_stub()
    conv = Conversation(chat_id=1, user_id=100)
    conv.add_message(Message(role="user", content="investigate pod X"))

    original_request = "investigate pod X"

    async def fake_stream(*args, **kwargs):
        yield {"type": "tool_call", "tool_name": "bash",
                "tool_call_id": "call-a", "params": {"command": "kubectl get pods"}}
        raise LLMInterruptedError("user stopped")

    svc._run_holmes_stream = fake_stream

    try:
        async for _ in svc._stream_chat(
            messages=[], features={"tool_approval": False}, user_id=100,
            conversation=conv, telegram_context=None, chat_id=1, topic_id=0,
            cancel_event=None, original_request=original_request,
        ):
            pass
    except LLMInterruptedError:
        pass

    system_msgs = [m for m in conv.messages if m.role == "system"]
    assert len(system_msgs) >= 1
    last_sys = system_msgs[-1]
    assert "interrupted" in last_sys.content.lower()
    assert original_request in last_sys.content


@pytest.mark.asyncio
async def test_stream_chat_no_note_when_original_request_missing():
    """If `original_request` is empty/None, no inline system note is
    appended even on interrupt — we have nothing meaningful to put in
    the note."""
    svc = _make_service_stub()
    conv = Conversation(chat_id=1, user_id=100)
    conv.add_message(Message(role="user", content="hi"))

    async def fake_stream(*args, **kwargs):
        raise LLMInterruptedError("instant stop")
        yield  # unreachable; makes this a generator

    svc._run_holmes_stream = fake_stream

    try:
        async for _ in svc._stream_chat(
            messages=[], features={"tool_approval": False}, user_id=100,
            conversation=conv, cancel_event=None,
        ):
            pass
    except LLMInterruptedError:
        pass

    system_msgs = [m for m in conv.messages if m.role == "system"]
    assert system_msgs == []


@pytest.mark.asyncio
async def test_stream_chat_no_persist_when_nothing_accumulated():
    """If the stream raises immediately without yielding anything,
    _stream_chat should NOT add an empty assistant message."""
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
async def test_stream_chat_success_path_no_note():
    """On the happy path (no interrupt), no inline `role="system"` note
    should be appended and `stopped_by_user` should be False/absent."""
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
    # No inline system note on success path.
    system_msgs = [m for m in conv.messages if m.role == "system"]
    assert system_msgs == []
