"""Worker-agent registry.

New agents are onboarded with a registry entry (agents.yaml) — no supervisor
code change.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

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
    # Multi-model support: the serving endpoint this agent's
    # *governance* calls — semantic screen, context resolution, worker
    # simulation — should use instead of the global routing LLM. Empty means
    # "use the global endpoint", which is every entry today. Honoured only
    # when MULTI_MODEL_ENABLED is true; model_provider.get_agent_model owns
    # that gate. This is NOT the worker's own internal model — that stays the
    # worker's concern (ASM-03); `endpoint` above is still where a dispatch
    # actually goes.
    model: str = ""
    # ── Supervisor-enforced human approval ──────────────────────────────────
    # Requests to *this* agent that match one of these patterns have their
    # response staged for human sign-off, whether or not the worker asks for a
    # gate. Each entry is `{"pattern": <regex>, "reason": <shown to the
    # approver>}`, the same shape as the guardrails document's deny rules.
    #
    # This exists because the approval gate used to depend entirely on a worker
    # volunteering `custom_outputs.hitl.status == "pending_approval"`. Solution
    # §04 puts "irreversible actions (deployment steps) behind the
    # human-in-the-loop approval gate" — a promise that rested on worker
    # cooperation, so a worker that forgot the flag, was misconfigured or was
    # compromised simply returned an answer and no gate opened. Declaring the
    # requirement here moves it into governed configuration the supervisor
    # enforces, on the same publish path as every other rule.
    #
    # Matched against the user's request, not the worker's response: what makes
    # an action reviewable is what was asked for, and the response is the thing
    # being reviewed. Patterns should name the *action*, not its subject — see
    # `agents.yaml` for why "the rollback procedure" must not trip a gate that
    # "roll back prod" does.
    approval_patterns: tuple[dict, ...] = ()
    # Classification only — low | medium | high | critical. Recorded in the
    # decision trail so a reviewer sees what class of agent produced an
    # artifact, and so a report can separate high-risk traffic. Deliberately
    # not wired to any automatic behaviour: a risk *label* driving a hidden
    # policy is how a governance decision becomes unexplainable. What gates is
    # `approval_patterns` above, which is explicit.
    risk_level: str = "low"
    # `supports_hitl` and `stages` used to sit here. Nothing read them: the
    # staged order a UI shows is that UI's own concern, and a *sequence* is
    # still something the supervisor never enforces. Declaring
    # them here implied otherwise, which is worse than not saying it.


class AgentRegistry:
    def __init__(self, agents: list[WorkerAgent]):
        self._agents = {a.id: a for a in agents}

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentRegistry":
        return cls.from_mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    @classmethod
    def from_mapping(cls, data: dict) -> "AgentRegistry":
        """Build from an already-parsed document.

        The parsing seam: `from_yaml` reads a file, `config_store` reads a row
        in the governed table, and both end up here so the two sources cannot
        drift into interpreting the same document differently.
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
                    rule for rule in (row.get("approval_patterns", []) or []) if isinstance(rule, dict)
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

        Deterministic and case-insensitive, evaluated on the user's request. A
        match is a *requirement*, never a suggestion: the supervisor stages the
        response regardless of what the worker returned.

        Patterns are compiled per call rather than at construction, unlike the
        guardrail engine's. This runs once per dispatch against a handful of
        short patterns, `re` keeps its own compiled cache, and the registry is
        rebuilt whenever the governed document changes — pre-compiling would
        add a build step to buy nothing measurable. An unusable pattern cannot
        reach here anyway: `config_store` compiles every one at publish time.
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
                # Belt and braces: a pattern that somehow bypassed publish-time
                # validation must not take the turn down. Skipping it is the
                # only safe direction — the alternative is a failed dispatch on
                # a request that may not have needed a gate at all.
                logger.warning(
                    "agent %s has an invalid approval pattern %r — ignored", agent.id, pattern
                )
        return ""

    def ids(self) -> list[str]:
        return list(self._agents)

    def context_keys(self) -> frozenset[str]:
        """Every context key any agent declares it needs.

        The allowlist for long-term memory writes: nothing outside this set was
        asked for by an agent, so nothing outside it is worth persisting — and
        persisting it anyway is the memory-poisoning path §04 closes.
        """
        return frozenset(key for agent in self._agents.values() for key in agent.required_context)
