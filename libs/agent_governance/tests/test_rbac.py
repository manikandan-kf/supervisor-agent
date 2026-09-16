"""Authorization: who may reach which agent, and what the agent's own database role may do.

The policy half is pure and offline; the privilege half checks that the grants the
deployment guide asks for were actually applied — the difference between a tamper-evident
table and a table anyone can rewrite.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from agent_governance import rbac, trust
from agent_governance.rbac import RbacPolicy


def test_mapped_role_is_allowed():
    policy = RbacPolicy({"BA": ["requirement-agent"]})
    assert policy.check("BA", "requirement-agent").allowed


def test_unmapped_agent_is_denied():
    policy = RbacPolicy({"BA": ["requirement-agent"]})
    decision = policy.check("BA", "deployment-agent")
    assert not decision.allowed


def test_unknown_role_is_denied():
    policy = RbacPolicy({"BA": ["requirement-agent"]})
    assert not policy.check("Intern", "requirement-agent").allowed


def test_missing_role_or_agent_is_denied():
    policy = RbacPolicy({"BA": ["requirement-agent"]})
    assert not policy.check("", "requirement-agent").allowed
    assert not policy.check("BA", "").allowed


# ═══ Database privileges ═════════════════════════════════════════════════════════
#
# The control that notices a skipped post-deploy step: report excess, stay silent on a
# correct posture, and — the property that makes it safe to call at startup — never raise.

TABLES = {
    "config": "supervisor_config",
    "audit": "supervisor_audit_log",
    "reviews": "supervisor_review_queue",
}

# The posture DEPLOYMENT.md §7b's REVOKEs produce.
RESTRICTED = [
    ("supervisor_config", "SELECT"),
    ("supervisor_audit_log", "INSERT"),
    ("supervisor_audit_log", "SELECT"),
    ("supervisor_review_queue", "INSERT"),
    ("supervisor_review_queue", "SELECT"),
    ("supervisor_review_queue", "UPDATE"),
]

# What the runtime holds before anyone runs them — it created the tables.
UNRESTRICTED = [
    (table, privilege)
    for table in TABLES.values()
    for privilege in ("INSERT", "SELECT", "UPDATE", "DELETE", "TRUNCATE", "TRIGGER", "REFERENCES")
]


class FakeCursor:
    def __init__(self, grants, role, raises):
        self._grants, self._role, self._raises = grants, role, raises
        self._last = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if self._raises:
            raise RuntimeError("connection reset by peer")
        self._last = sql

    def fetchone(self):
        return {"role": self._role}

    def fetchall(self):
        return [{"table_name": t, "privilege_type": p} for t, p in self._grants]


class FakeConnection:
    def __init__(self, grants, role, raises):
        self._args = (grants, role, raises)

    def cursor(self):
        return FakeCursor(*self._args)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def source(grants, role="0a1b-uuid-role", raises=False):
    def _source():
        return FakeConnection(grants, role, raises)

    return _source


# ── the two postures ────────────────────────────────────────────────────────


def test_a_correctly_restricted_identity_produces_no_findings():
    assert rbac.check_privileges(source(RESTRICTED), TABLES) == []


def test_an_unrestricted_identity_is_reported_on_every_table():
    findings = rbac.check_privileges(source(UNRESTRICTED), TABLES)
    assert {f.table for f in findings} == set(TABLES.values())


def test_the_audit_table_reports_exactly_the_privileges_that_defeat_it():
    """UPDATE and DELETE on the audit table are the whole point of the check."""
    findings = rbac.check_privileges(source(UNRESTRICTED), TABLES)
    audit = next(f for f in findings if f.table == "supervisor_audit_log")
    assert set(audit.excess) == {"UPDATE", "DELETE", "TRUNCATE", "TRIGGER", "REFERENCES"}
    assert "INSERT" not in audit.excess and "SELECT" not in audit.excess


def test_trigger_alone_is_reported():
    """TRIGGER is the privilege-escalation route, and it is easy to miss."""
    grants = RESTRICTED + [("supervisor_audit_log", "TRIGGER")]
    findings = rbac.check_privileges(source(grants), TABLES)
    assert [f.excess for f in findings] == [("TRIGGER",)]


def test_the_review_queue_keeps_update_but_not_delete():
    grants = RESTRICTED + [("supervisor_review_queue", "DELETE")]
    findings = rbac.check_privileges(source(grants), TABLES)
    assert [f.excess for f in findings] == [("DELETE",)]


def test_config_may_only_be_read():
    grants = RESTRICTED + [("supervisor_config", "UPDATE")]
    findings = rbac.check_privileges(source(grants), TABLES)
    assert [f.excess for f in findings] == [("UPDATE",)]


# ── it has to be safe to call at startup ────────────────────────────────────


def test_a_database_failure_is_not_fatal():
    """A control that cannot run must not take the endpoint down with it."""
    assert rbac.check_privileges(source(RESTRICTED, raises=True), TABLES) == []


def test_a_table_that_does_not_exist_yet_is_not_a_finding():
    """First boot: the runtime has not created its tables."""
    assert rbac.check_privileges(source([]), TABLES) == []


def test_no_configured_tables_short_circuits():
    assert rbac.check_privileges(source(UNRESTRICTED), {}) == []


# ── what an operator is told ────────────────────────────────────────────────


def test_a_finding_names_the_role_and_the_remedy():
    findings = rbac.check_privileges(source(UNRESTRICTED, role="abc-123"), TABLES)
    text = str(findings[0])
    assert "REVOKE" in text and "abc-123" in text and "§7b" in text


def test_report_logs_each_finding_at_error(caplog):
    findings = rbac.check_privileges(source(UNRESTRICTED), TABLES)
    with caplog.at_level("ERROR"):
        returned = rbac.report_privileges(findings)
    assert len(returned) == len(findings)
    assert len(caplog.records) == len(findings)


def test_report_says_so_when_the_posture_is_clean(caplog):
    with caplog.at_level("INFO"):
        rbac.report_privileges([])
    assert any("verified" in r.message for r in caplog.records)


@pytest.mark.parametrize("role_key", list(rbac.ALLOWED))
def test_every_declared_role_forbids_delete_and_trigger(role_key):
    """Whatever else a table allows, these two are never granted."""
    allowed = rbac.ALLOWED[role_key]
    assert "DELETE" not in allowed
    assert "TRIGGER" not in allowed


# ── The entitlement HMAC — what makes a caller-supplied permitted set safe ──
#
# The Governance Front Door keeps its own copy of the signer, so the exact bytes signed
# are a wire contract — compact sorted JSON of `trust.SIGNED_FIELDS`, HMAC-SHA256, hex.
# A change here would break a live deployment mid-rollout.

SECRET = "0123456789abcdef0123456789abcdef"


def _block() -> dict:
    return {
        "user_role": "BA",
        "agent_id": "requirement-agent",
        "permitted_agents": ["requirement-agent", "coding-agent"],
        "approvable_agents": [],
        "user_id": "usr_ab12",
        "correlation_id": "corr-1",
        "environment": "dev",
    }


def test_entitlement_signature_is_hmac_over_compact_sorted_json_of_signed_fields():
    block = _block()
    payload = {k: block[k] for k in trust.SIGNED_FIELDS if k in block}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    expected = hmac.new(SECRET.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    assert trust.sign_entitlements(block, SECRET) == expected


def test_a_signed_block_verifies_and_a_tampered_one_does_not():
    block = _block()
    block[trust.SIGNATURE_FIELD] = trust.sign_entitlements(block, SECRET)
    assert trust.verify_entitlements(block, SECRET) is True

    field = trust.SIGNED_FIELDS[0]
    tampered = {**block, field: "something-else"}
    assert trust.verify_entitlements(tampered, SECRET) is False


def test_an_unsigned_block_is_refused_only_when_a_secret_is_configured():
    block = _block()
    # Secret configured: unsigned means it did not come through the gateway.
    assert trust.verify_entitlements(block, SECRET) is False
    # Control dark: nothing is checked and every block passes.
    assert trust.verify_entitlements(block, "") is True


def test_signing_with_an_empty_secret_is_an_error_not_a_weak_signature():
    with pytest.raises(ValueError):
        trust.sign_entitlements(_block(), "")
    with pytest.raises(ValueError):
        trust.sign_dispatch(_block(), "")


def test_the_secret_is_read_from_the_environment_per_call(monkeypatch):
    monkeypatch.delenv("SUPERVISOR_TRUST_SECRET", raising=False)
    assert trust.trust_secret() == ""
    monkeypatch.setenv("SUPERVISOR_TRUST_SECRET", SECRET)
    assert trust.trust_secret() == SECRET


def test_dispatch_signature_round_trips_and_a_replayed_nonce_change_is_caught():
    dispatch = {
        "conversation_id": "thr-1",
        "agent_id": "coding-agent",
        "request_id": "req-1",
        "correlation_id": "corr-1",
        "pseudonymous_user_reference": "usr_ab12",
        "nonce": trust.new_nonce(),
    }
    dispatch[trust.DISPATCH_SIGNATURE_FIELD] = trust.sign_dispatch(dispatch, SECRET)
    assert trust.verify_dispatch(dispatch, SECRET) is True
    assert trust.verify_dispatch({**dispatch, "nonce": trust.new_nonce()}, SECRET) is False
    assert (
        trust.verify_dispatch(
            {k: v for k, v in dispatch.items() if k != trust.DISPATCH_SIGNATURE_FIELD}, SECRET
        )
        is False
    )
    # Ships dark, like the entitlement check.
    assert trust.verify_dispatch({"agent_id": "x"}, "") is True


def test_nonces_are_32_hex_chars_and_never_repeat():
    nonces = {trust.new_nonce() for _ in range(64)}
    assert len(nonces) == 64
    assert all(len(n) == 32 and int(n, 16) >= 0 for n in nonces)
