"""Tests for the distilled-facts block injection in HolmesService's
`_build_holmes_messages`.

We patch `build_initial_ask_messages` (Holmes' downstream function) so the
test doesn't need a real ToolExecutor — _build_holmes_messages' behavior of
interest is the system_prompt_additions string it passes through, not the
Holmes-specific prompt assembly. The patched function returns a list with
a single "system" message echoing the additions, so we can assert on it.
"""
from unittest.mock import MagicMock, patch


def _patched_build_messages():
    """Patch build_initial_ask_messages to echo its `system_prompt_additions`
    kwarg as a system message — this is what _build_holmes_messages passes
    through to it, and what we want to assert on."""
    def _fake_build_initial_ask_messages(
        initial_user_prompt=None,
        file_paths=None,
        tool_executor=None,
        skills=None,
        system_prompt_additions=None,
        global_instructions=None,
        cluster_name=None,
        ask_user_enabled=True,
        file_paths_arg=None,
        include_todowrite_reminder=True,
        images=None,
        prompt_component_overrides=None,
        **kwargs,
    ):
        return [
            {"role": "system", "content": system_prompt_additions or ""},
            {"role": "user", "content": initial_user_prompt or ""},
        ]
    return patch(
        "src.bot.services.holmes_service.build_initial_ask_messages",
        side_effect=_fake_build_initial_ask_messages,
    )


def _make_holmes_service(distilled_store=None):
    """Instantiate HolmesService without running initialize() — we only
    test the pure _build_holmes_messages path."""
    from src.bot.services.holmes_service import HolmesService
    return HolmesService(memory_service=None, distilled_fact_store=distilled_store)


def _messages_with_user_text(text="Hello"):
    return [{"role": "user", "content": text}]


def _system_text(messages):
    for m in messages:
        if m.get("role") == "system":
            return m.get("content", "")
    return ""


def test_distilled_facts_injected_when_nonempty():
    store = MagicMock()
    store.get_injectable_facts.return_value = [
        {"text": "MFKE api-gw runs on port 8080", "source_topic_id": 10},
        {"text": "Shoot kubeconfigs live under /root/kubeconfig", "source_topic_id": 20},
    ]
    hs = _make_holmes_service(distilled_store=store)
    with _patched_build_messages():
        messages = hs._build_holmes_messages(
            messages=_messages_with_user_text(),
            user_id=1,
            is_admin=False,
        )
    sys_text = _system_text(messages)
    assert "DISTILLED KNOWLEDGE" in sys_text
    assert "MFKE api-gw runs on port 8080" in sys_text
    assert "Shoot kubeconfigs live under /root/kubeconfig" in sys_text


def test_distilled_facts_skipped_when_empty():
    store = MagicMock()
    store.get_injectable_facts.return_value = []
    hs = _make_holmes_service(distilled_store=store)
    with _patched_build_messages():
        messages = hs._build_holmes_messages(
            messages=_messages_with_user_text(),
            user_id=1,
            is_admin=False,
        )
    sys_text = _system_text(messages)
    assert "DISTILLED KNOWLEDGE" not in sys_text


def test_distilled_block_precedes_search_memory_rule(mock_config):
    """When both `distilled_fact_store` is wired AND `graphiti_memory_enabled`
    is True, both blocks must appear with distilled first.
    """
    mock_config.graphiti_memory_enabled = True
    store = MagicMock()
    store.get_injectable_facts.return_value = [{"text": "fact-one"}]
    hs = _make_holmes_service(distilled_store=store)
    with _patched_build_messages():
        messages = hs._build_holmes_messages(
            messages=_messages_with_user_text(),
            user_id=1,
            is_admin=False,
        )
    sys_text = _system_text(messages)
    distilled_idx = sys_text.find("DISTILLED KNOWLEDGE")
    search_idx = sys_text.find("search_memory TOOL USE POLICY")
    assert distilled_idx >= 0, "distilled block missing"
    assert search_idx >= 0, "search_memory rule missing"
    assert distilled_idx < search_idx, "distilled block must precede the search_memory rule"
