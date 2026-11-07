"""Tests for HolmesService._admin_gate_rule — the per-turn system-prompt block
teaching the LLM about admin-gate + hard-block policy.

Verifies that BOTH admin and non-admin branches mention every hard-block
category (kubectl apply, delete critical, protected ns, openstack non-fke)
and the /shell manual fallback, so the LLM pivots correctly when a tool
returns a HARD-BLOCKED error.
"""
from src.bot.services.holmes_service import HolmesService


def _rule_text(is_admin: bool, name: str = "alice") -> str:
    svc = HolmesService()
    return svc._admin_gate_rule(is_admin=is_admin, user_display_name=name)


# ---------- Common content (admin + non-admin) ----------

def test_admin_rule_includes_user_line():
    text = _rule_text(is_admin=True, name="bob")
    assert "Current user: bob (admin: yes)" in text


def test_non_admin_rule_includes_user_line():
    text = _rule_text(is_admin=False, name="carol")
    assert "Current user: carol (admin: no)" in text


def test_admin_rule_mentions_hard_block_apply():
    text = _rule_text(is_admin=True)
    assert "kubectl apply" in text
    assert "HARD-BLOCKED" in text or "HARD-BLOCK" in text


def test_admin_rule_mentions_critical_resource_delete():
    text = _rule_text(is_admin=True)
    assert "critical" in text.lower()
    assert "node" in text
    assert "pv" in text
    assert "pvc" in text
    assert "namespace" in text
    assert "crd" in text
    assert "clusterrole" in text
    assert "service" in text


def test_admin_rule_mentions_protected_namespaces():
    text = _rule_text(is_admin=True)
    for ns in ("kube-system", "kube-public", "kube-node-lease", "garden",
               "gardener-system", "extension*", "garden-*"):
        assert ns in text, f"missing protected-ns token: {ns}"


def test_admin_rule_mentions_openstack_fke_policy():
    text = _rule_text(is_admin=True)
    assert "openstack" in text.lower()
    assert "fke" in text
    assert "delete" in text
    assert "shell" in text.lower() or "/shell" in text


def test_admin_rule_tells_llm_not_to_retry_hard_block():
    text = _rule_text(is_admin=True)
    assert "do NOT retry" in text or "DO NOT retry" in text
    assert "Pivot to read-only" in text or "read-only" in text


def test_admin_rule_clarifies_other_mutations_pass_through():
    text = _rule_text(is_admin=True)
    assert "kubectl create" in text
    assert "fke-*" in text  # openstack pass-through example


# ---------- Non-admin branch ----------

def test_non_admin_rule_keeps_three_mutating_rules():
    """The original 3 rules (pure mutation / mixed inv+mut / follow-up
    selection) must remain — only the hard-block appendix is added."""
    text = _rule_text(is_admin=False)
    assert "Three rules" in text
    assert "scale X" in text
    assert "check why" in text or "check why pod X is crashing" in text
    assert "Follow-up selection" in text or "option 2 đi" in text or "dùng option A" in text


def test_non_admin_rule_includes_hard_block_apply():
    text = _rule_text(is_admin=False)
    assert "kubectl apply" in text
    assert "HARD-BLOCKED" in text or "HARD-BLOCK" in text


def test_non_admin_rule_includes_critical_resource_delete():
    text = _rule_text(is_admin=False)
    assert "node" in text and "pv" in text and "pvc" in text
    assert "namespace" in text and "crd" in text and "clusterrole" in text
    assert "service" in text


def test_non_admin_rule_includes_protected_namespaces():
    text = _rule_text(is_admin=False)
    for ns in ("kube-system", "kube-public", "kube-node-lease", "garden",
               "gardener-system", "extension*", "garden-*"):
        assert ns in text, f"missing protected-ns token in non-admin rule: {ns}"


def test_non_admin_rule_includes_openstack_fke_policy():
    text = _rule_text(is_admin=False)
    assert "openstack" in text.lower()
    assert "fke" in text


def test_non_admin_rule_advises_against_calling_hard_block_tool():
    """Non-admin path explicitly tells the LLM NOT to invoke the bash tool
    for hard-block actions — reply directly instead."""
    text = _rule_text(is_admin=False)
    assert "DO NOT call the tool" in text or "DO NOT call any tool" in text
    assert "shell" in text.lower() or "/shell" in text


# ---------- Regression: existing admin-gate behavior preserved ----------

def test_non_admin_rule_still_mentions_admin_access_required_error():
    """The legacy 'Admin access required' error message instruction still
    present — non-admins who somehow hit the gate still get the right pivot."""
    text = _rule_text(is_admin=False)
    assert "Admin access required" in text


def test_admin_rule_does_not_say_admin_only():
    """Admin rule should NOT say 'admin is needed' (user IS admin)."""
    text = _rule_text(is_admin=True)
    # The phrase "ask an admin" should not appear in the admin path.
    assert "ask an admin" not in text.lower()
