"""RBAC gate: the requested agent is checked against the user's
role-to-agent mapping; requests outside that set are denied without
revealing other agents.
"""

from __future__ import annotations

from dataclasses import dataclass

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
