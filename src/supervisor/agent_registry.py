"""Worker-agent registry.

New agents are onboarded with a registry entry (agents.yaml) — no supervisor
code change.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerAgent:
    id: str
    name: str
    description: str
    endpoint: str
    domain_scope: str
    required_context: tuple[str, ...] = ()
    deny_patterns: tuple[str, ...] = ()
    # Endpoint for this agent's *governance* calls when MULTI_MODEL_ENABLED; empty
    # means the routing LLM. Never the worker's own model: that is the worker's business.
    model: str = ""
    # ── Supervisor-enforced human approval ──────────────────────────────────
    # Solution §04: irreversible actions sit behind the approval gate regardless of
    # whether the worker volunteers `hitl.status`. Matched against the *request*,
    # naming the action not its subject — see `agents.yaml`. Shape: `{"pattern", "reason"}`.
    approval_patterns: tuple[dict, ...] = ()
    # Classification only (low|medium|high|critical), recorded in the trail. Not wired
    # to behaviour: a risk *label* driving hidden policy is unexplainable governance.
    risk_level: str = "low"


class AgentRegistry:
    def __init__(self, agents: list[WorkerAgent]):
        self._agents = {a.id: a for a in agents}

    @classmethod
    def from_mapping(cls, data: dict) -> "AgentRegistry":
        """Build from the parsed agents document.

        The one parsing seam for both the bundled YAML and the governed table.
        """
        data = data or {}
        agents = [
            WorkerAgent(
                id=row["id"],
                name=row.get("name", row["id"]),
                description=row.get("description", ""),
                endpoint=row["endpoint"],
                domain_scope=row.get("domain_scope", row.get("description", "")),
                required_context=tuple(row.get("required_context", []) or []),
                deny_patterns=tuple(row.get("deny_patterns", []) or []),
                model=str(row.get("model", "") or "").strip(),
                approval_patterns=tuple(
                    rule
                    for rule in (row.get("approval_patterns", []) or [])
                    if isinstance(rule, dict)
                ),
                risk_level=str(row.get("risk_level", "low") or "low").strip().lower(),
            )
            for row in data.get("agents", [])
        ]
        return cls(agents)

    def get(self, agent_id: str) -> WorkerAgent | None:
        return self._agents.get(agent_id)

    @staticmethod
    def approval_reason(agent: "WorkerAgent | None", query: str) -> str:
        """Why this request to `agent` needs human sign-off, or "" if it does not.

        A match is a *requirement*: the response is staged regardless of the worker's
        reply. Compiled per call — `re` caches, and `governed_config_store` validated at publish.
        """
        if agent is None or not query:
            return ""
        for rule in getattr(agent, "approval_patterns", ()) or ():
            pattern = (rule or {}).get("pattern") or ""
            if not pattern:
                continue
            try:
                if re.search(pattern, query, flags=re.IGNORECASE):
                    return str(rule.get("reason") or "the request matches an approval rule")
            except re.error:
                # A pattern that bypassed publish-time validation must not take the
                # turn down; skipping it is the only safe direction.
                logger.warning(
                    "agent %s has an invalid approval pattern %r — ignored", agent.id, pattern
                )
        return ""

    def ids(self) -> list[str]:
        return list(self._agents)

    def context_keys(self) -> frozenset[str]:
        """Every context key any agent declares it needs.

        The long-term memory write allowlist: persisting anything else is the
        memory-poisoning path §04 closes.
        """
        return frozenset(key for agent in self._agents.values() for key in agent.required_context)
