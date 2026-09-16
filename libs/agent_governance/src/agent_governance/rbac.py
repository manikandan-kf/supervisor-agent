"""Access control, in both directions.

`RbacPolicy` is the request-path gate; a denial never reveals which agents exist or what the
role can reach. `check_privileges` is the deployment-time posture check: the governance tables
are append-only or read-only *from the data plane*, but that posture is applied by the table
owner after the runtime creates its tables (DEPLOYMENT.md §7b), and an audit trail writable by
the component it audits is not an audit trail. It warns rather than refuses, because the runtime
must hold DDL on first boot to create those tables; the finding is logged at ERROR, with remedy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

# Generic denial — never reveal which agents exist or what this role can access.
DENIED_MESSAGE = (
    "You don't have access to the requested agent. "
    "If you believe this is an error, please contact your administrator."
)


@dataclass(frozen=True)
class RbacDecision:
    allowed: bool
    reason: str


class RbacPolicy:
    def __init__(self, role_to_agents: dict[str, list[str]]):
        self._map = {role: set(agents or []) for role, agents in (role_to_agents or {}).items()}

    @classmethod
    def from_mapping(cls, data: dict) -> "RbacPolicy":
        """Build from the parsed rbac document (bundled seed or governed table)."""
        return cls((data or {}).get("roles", {}))

    def check(self, role: str, agent_id: str) -> RbacDecision:
        if not role:
            return RbacDecision(False, "missing user role")
        if not agent_id:
            return RbacDecision(False, "missing target agent id")
        if agent_id in self._map.get(role, set()):
            return RbacDecision(True, f"role '{role}' is mapped to agent '{agent_id}'")
        return RbacDecision(False, f"role '{role}' is not mapped to agent '{agent_id}'")


# ---------------------------------------------------------------------------
# Privileges the runtime itself still holds
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Privilege names as `information_schema` spells them.
INSERT, SELECT, UPDATE, DELETE = "INSERT", "SELECT", "UPDATE", "DELETE"
TRUNCATE, TRIGGER, REFERENCES = "TRUNCATE", "TRIGGER", "REFERENCES"

#: What the data-plane identity may hold on each governance table, keyed by the table's role (the
#: caller supplies deployment-specific names). `TRIGGER` matters most: it lets the serving identity
#: attach a trigger that runs as the config-publish identity — the escalation these grants close.
ALLOWED: Mapping[str, frozenset[str]] = {
    # Policy is read-only from the data plane.
    "config": frozenset({SELECT}),
    # The decision trail is append-only.
    "audit": frozenset({INSERT, SELECT}),
    # A review is opened by the supervisor and resolved by the gateway, so the
    # runtime keeps UPDATE. DELETE is nobody's business.
    "reviews": frozenset({INSERT, SELECT, UPDATE}),
}


@dataclass(frozen=True)
class PrivilegeFinding:
    """One table on which this identity holds more than it should."""

    table: str
    role: str
    excess: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"the serving identity holds {', '.join(self.excess)} on {self.table}; "
            f"revoke with: REVOKE {', '.join(self.excess)} ON {self.table} "
            f'FROM "{self.role}"  (DEPLOYMENT.md §7b)'
        )


_GRANTS_SQL = """
SELECT table_name, privilege_type
  FROM information_schema.role_table_grants
 WHERE grantee = current_user
   AND table_schema = current_schema()
   AND table_name = ANY(%s)
"""


def check_privileges(
    connection_source,
    tables: Mapping[str, str],
    allowed: Optional[Mapping[str, frozenset[str]]] = None,
) -> list[PrivilegeFinding]:
    """Privileges held beyond `allowed`, one finding per table.

    `tables` maps a role key in `ALLOWED` to the deployed table name; a table not yet created
    produces no finding. Queries `role_table_grants` because a `REVOKE` reports success whether or
    not it removed anything. Never raises: a control that cannot run must not sink the endpoint.
    """
    allowed = allowed or ALLOWED
    wanted = {name: key for key, name in tables.items() if name}
    if not wanted:
        return []

    held: dict[str, set[str]] = {}
    role = "<role>"
    try:
        with connection_source() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_user AS role")
                row = cur.fetchone()
                if row is not None:
                    role = str(row["role"] if isinstance(row, dict) else row[0])
                cur.execute(_GRANTS_SQL, (list(wanted),))
                for raw in cur.fetchall() or []:
                    if isinstance(raw, dict):
                        table, privilege = raw["table_name"], raw["privilege_type"]
                    else:
                        table, privilege = raw[0], raw[1]
                    held.setdefault(str(table), set()).add(str(privilege).upper())
    except Exception:
        logger.warning(
            "least-privilege check could not run — the governance-table grants were NOT verified",
            exc_info=True,
        )
        return []

    findings = []
    for table, privileges in sorted(held.items()):
        permitted = allowed.get(wanted.get(table, ""), frozenset())
        excess = tuple(sorted(privileges - permitted))
        if excess:
            findings.append(PrivilegeFinding(table=table, role=role, excess=excess))
    return findings


def report_privileges(findings: Iterable[PrivilegeFinding]) -> list[PrivilegeFinding]:
    """Log each finding at ERROR and hand them back.

    ERROR on purpose: each finding means a documented post-deploy step was skipped and the audit
    trail and governed policy are writable by the process they constrain. An alert, not an FYI.
    """
    findings = list(findings)
    for finding in findings:
        logger.error("governance-table privileges: %s", finding)
    if not findings:
        logger.info("governance-table privileges: least privilege verified")
    return findings
