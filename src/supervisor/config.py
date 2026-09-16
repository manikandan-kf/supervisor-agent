"""The supervisor's governed documents, and how the graph reads them.

Three documents live in the governed table, matching the seeds in `config/`:
agents (registry.AgentRegistry), rbac (rbac.RbacPolicy), guardrails (deny
patterns, output policy, canaries, kill switch). Store, checksum, TTL cache and
fallback rules are `agent_governance.config_store`; this module owns the
supervisor-specific validators and the consumer objects built from all three.
"""

from __future__ import annotations

import re
from typing import Any

from agent_governance.config_store import (
    PROMPT_FIELD_CONTROL,
    PROMPT_FIELD_MARKERS,
    ConfigError,
    ConfigProvider,
    ConfigStore,
    compile_pattern,
    require,
    validate_guardrails,
    validate_prompt_field,
)

from .memory import audit_connection_source
from .settings import Settings

CONFIG_NAMES = ("agents", "rbac", "guardrails")

# Agent ids reach the invocation path, derived role names and audit rows; an id
# that would break any of those must never be published.
_AGENT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_CONTEXT_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})
# These fields are interpolated into system prompts as trusted instruction text;
# an unbounded scope is prompt real estate for smuggled instructions.
_PROMPT_FIELD_LIMITS = {"name": 120, "description": 2000, "domain_scope": 4000}


def _validate_agents(payload: Any) -> None:
    require(isinstance(payload, dict), "agents config must be a mapping")
    agents = payload.get("agents")
    require(isinstance(agents, list) and agents, "agents config needs a non-empty 'agents' list")

    seen: set[str] = set()
    for index, row in enumerate(agents):
        where = f"agents[{index}]"
        require(isinstance(row, dict), f"{where} must be a mapping")

        agent_id = row.get("id")
        require(
            isinstance(agent_id, str) and bool(_AGENT_ID.match(agent_id)),
            f"{where}.id must be a lowercase slug, got {agent_id!r}",
        )
        require(agent_id not in seen, f"duplicate agent id {agent_id!r}")
        seen.add(agent_id)

        require(
            isinstance(row.get("endpoint"), str) and row["endpoint"].strip() != "",
            f"{where}.endpoint must be a non-empty string",
        )
        for field, limit in _PROMPT_FIELD_LIMITS.items():
            if field in row:
                require(isinstance(row[field], str), f"{where}.{field} must be a string")
                validate_prompt_field(row[field], where, field, limit)

        # Validated whenever present, even while multi-model is dark; present-but-empty
        # is an unfinished edit.
        if "model" in row and row["model"] is not None:
            require(
                isinstance(row["model"], str) and row["model"].strip() != "",
                f"{where}.model must be a non-empty serving-endpoint name when present",
            )

        required = row.get("required_context", []) or []
        require(isinstance(required, list), f"{where}.required_context must be a list")
        for key in required:
            require(
                isinstance(key, str) and bool(_CONTEXT_KEY.match(key)),
                f"{where}.required_context contains an invalid key {key!r}",
            )

        patterns = row.get("deny_patterns", []) or []
        require(isinstance(patterns, list), f"{where}.deny_patterns must be a list")
        for pattern in patterns:
            require(isinstance(pattern, str), f"{where}.deny_patterns must contain strings")
            compile_pattern(pattern, f"{where}.deny_patterns")

        # Supervisor-enforced approval (registry.approval_reason). The `reason`
        # is shown to the human approver, so it is bounded and marker-free.
        approvals = row.get("approval_patterns", []) or []
        require(isinstance(approvals, list), f"{where}.approval_patterns must be a list")
        for position, rule in enumerate(approvals):
            at = f"{where}.approval_patterns[{position}]"
            require(isinstance(rule, dict), f"{at} must be a mapping")
            pattern = rule.get("pattern")
            require(isinstance(pattern, str) and pattern != "", f"{at}.pattern must be a string")
            compile_pattern(pattern, at)
            reason = rule.get("reason")
            require(
                isinstance(reason, str) and 0 < len(reason) <= 500,
                f"{at}.reason must be a non-empty string of at most 500 characters — it is "
                "shown to the human being asked to approve",
            )
            require(
                not PROMPT_FIELD_CONTROL.search(reason) and not PROMPT_FIELD_MARKERS.search(reason),
                f"{at}.reason contains control characters or turn-boundary markers",
            )

        # Classification only, but a typo must not become a class of its own and
        # under-count a report grouped by risk level.
        if "risk_level" in row and row["risk_level"] is not None:
            require(
                isinstance(row["risk_level"], str)
                and row["risk_level"].strip().lower() in _RISK_LEVELS,
                f"{where}.risk_level must be one of {', '.join(sorted(_RISK_LEVELS))}",
            )


def _validate_rbac(payload: Any) -> None:
    require(isinstance(payload, dict), "rbac config must be a mapping")
    roles = payload.get("roles")
    require(isinstance(roles, dict), "rbac config needs a 'roles' mapping")
    for role, agents in roles.items():
        require(isinstance(role, str) and role.strip() != "", f"invalid role name {role!r}")
        require(isinstance(agents, list), f"roles[{role!r}] must be a list of agent ids")
        for agent_id in agents:
            require(
                isinstance(agent_id, str) and bool(_AGENT_ID.match(agent_id)),
                f"roles[{role!r}] contains an invalid agent id {agent_id!r}",
            )


VALIDATORS = {
    "agents": _validate_agents,
    "rbac": _validate_rbac,
    "guardrails": validate_guardrails,
}


def validate(name: str, payload: Any) -> None:
    """Raise `ConfigError` unless `payload` is a usable `name` document."""
    validator = VALIDATORS.get(name)
    if validator is None:
        raise ConfigError(f"unknown configuration document {name!r}")
    validator(payload)


def config_store(settings: Settings) -> ConfigStore:
    """The store the publish path writes to. Raises if no Postgres is configured."""
    source = audit_connection_source()
    if source is None:
        raise RuntimeError(
            "No Lakebase instance is configured, so there is no configuration table to "
            "publish to. Set LAKEBASE_INSTANCE (and LAKEBASE_SCHEMA for the environment)."
        )
    return ConfigStore(source, settings.config_table, VALIDATORS)


class SupervisorConfig:
    """The three documents as the objects the graph consumes.

    Accessors are wrapped in `config_store.Reloading` by `services.build_services`,
    so a published change reaches a running endpoint within the cache TTL.
    """

    def __init__(self, settings: Settings, store: ConfigStore | None = None):
        self.provider = ConfigProvider(
            validators=VALIDATORS,
            bundled_dir=settings.config_dir,
            table=settings.config_table,
            # The same shared Postgres the audit sink writes to. Reads are
            # TTL-cached, so this adds a few statements a minute per process.
            connection_source=audit_connection_source,
            ttl_seconds=settings.config_cache_seconds,
            source=settings.config_source,
            store=store,
        )

    def registry(self):
        from .registry import AgentRegistry

        return self.provider.built("agents", AgentRegistry.from_mapping)

    def rbac(self):
        from agent_governance.rbac import RbacPolicy

        return self.provider.built("rbac", RbacPolicy.from_mapping)

    def guardrails(
        self,
        llm,
        confidence_threshold: float,
        model_for=None,
        decisive_threshold: float = 0.9,
        contested_margin: float = 0.15,
    ):
        from .guardrail_engine import GuardrailEngine

        return self.provider.built(
            "guardrails",
            lambda payload: GuardrailEngine.from_mapping(
                llm,
                payload,
                confidence_threshold,
                model_for=model_for,
                decisive_threshold=decisive_threshold,
                contested_margin=contested_margin,
            ),
        )

    def output_guard(self, mask_pii: bool = True):
        """Layer-7 output screen, built from the same governed guardrails doc.

        Separate object cache. Governance prompt lines are protected text: a reply
        reproducing one has reproduced the supervisor's instructions.
        """
        from agent_governance.output_guard import OutputGuard

        from .prompt_provider import protected_lines

        return self.provider.built(
            "guardrails",
            lambda payload: OutputGuard.from_mapping(
                payload, mask_pii=mask_pii, protected_texts=protected_lines()
            ),
            cache_key="output_guard",
        )
