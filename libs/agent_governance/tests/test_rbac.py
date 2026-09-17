"""Authorization: who may reach which agent, and what the agent's own database role may do.

The policy half is pure and offline; the privilege half checks that the grants the
deployment guide asks for were actually applied — the difference between a tamper-evident
table and a table anyone can rewrite.
"""

from __future__ import annotations

import pytest
from agent_governance import rbac
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
}

# The posture DEPLOYMENT.md §7b's REVOKEs produce.
RESTRICTED = [
    ("supervisor_config", "SELECT"),
    ("supervisor_audit_log", "INSERT"),
    ("supervisor_audit_log", "SELECT"),
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
