"""Runtime configuration for the Supervisor Agent."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from agent_governance.environment import (  # noqa: F401 — re-exported for the package
    catalog,
    is_local_environment,
    resource_environment,
)
from agent_governance.environment import environment_schema as _environment_schema

logger = logging.getLogger(__name__)

_PACKAGE_CONFIG = Path(__file__).resolve().parent / "config"

# `ENVIRONMENT` picks the deployment's resources and whether a control may degrade.
_SCHEMA_PREFIX = "supervisor"


def environment_schema(environment: str | None = None) -> str:
    """`supervisor_dev`, `supervisor_prod`, … — this agent's per-environment schema.

    Used for both the Lakebase and Unity Catalog schema so the two cannot drift.
    """
    return _environment_schema(_SCHEMA_PREFIX, environment)


@dataclass(frozen=True)
class Settings:
    # From the approved model list (solution §06); wired via DAB target vars.
    routing_llm_endpoint: str = field(
        default_factory=lambda: os.getenv("ROUTING_LLM_ENDPOINT", "databricks-claude-sonnet-4-5")
    )
    # Near-deterministic so governance decisions are reproducible.
    routing_temperature: float = field(
        default_factory=lambda: float(os.getenv("ROUTING_TEMPERATURE", "0.0"))
    )

    # ── Multi-model support ─────────────────────────────────────────────────
    # Off by default so a `model:` published early cannot silently redirect spend.
    multi_model_enabled: bool = field(
        default_factory=lambda: os.getenv("MULTI_MODEL_ENABLED", "").lower() == "true"
    )

    # ── Outbound call bounds ────────────────────────────────────────────────
    # Per-call half of the execution timeout; the library default is effectively 600s.
    routing_llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("ROUTING_LLM_TIMEOUT_SECONDS", "30"))
    )
    # **Zero on purpose**: `retry_and_deadline.invoke_with_retries` is the one retry authority.
    routing_llm_max_retries: int = field(
        default_factory=lambda: int(os.getenv("ROUTING_LLM_MAX_RETRIES", "0"))
    )
    # Attempts per governance call before the node's fail-closed hold takes over.
    governance_llm_attempts: int = field(
        default_factory=lambda: int(os.getenv("GOVERNANCE_LLM_ATTEMPTS", "3"))
    )
    # Bounds one dispatch attempt; worst case ~ attempts x timeout + backoff.
    worker_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("WORKER_TIMEOUT_SECONDS", "45"))
    )
    # ── The turn's total time budget ────────────────────────────────────────
    # Per-call bounds don't compose across the fan-out. Guarantee: budget + longest
    # call = 120 + 45 = 165s, under a 180s caller timeout. **Re-check the sum if either bound moves.**
    turn_budget_seconds: float = field(
        default_factory=lambda: float(os.getenv("TURN_BUDGET_SECONDS", "120"))
    )
    # Which deployment this is (dev / prod); selects the promoted prompt alias.
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "local"))
    config_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SUPERVISOR_CONFIG_DIR", str(_PACKAGE_CONFIG)))
    )

    # ── Governance configuration source ─────────────────────────────────────
    # Solution §02 puts configuration in governed tables; the SP cannot hold
    # `databricks-sql-access`, so the table is the Lakebase one (governed_config_store.py). auto = table if Postgres
    # is configured else bundled YAML; table = require it; files = bundled YAML only.
    config_source: str = field(
        default_factory=lambda: os.getenv("SUPERVISOR_CONFIG_SOURCE", "auto").strip().lower()
    )
    config_table: str = field(
        default_factory=lambda: os.getenv("SUPERVISOR_CONFIG_TABLE", "supervisor_config")
    )
    # 60s matches MLflow's prompt-alias cache so the two governed things move together.
    config_cache_seconds: float = field(
        default_factory=lambda: float(os.getenv("CONFIG_CACHE_TTL_SECONDS", "60"))
    )

    # ── Audit sink ──────────────────────────────────────────────────────────
    # Postgres when a DSN/Lakebase resolves, else the process log (never lost, not
    # queryable). No SQL-warehouse sink: the SP cannot hold `databricks-sql-access`.
    audit_pg_table: str = field(
        default_factory=lambda: os.getenv("AUDIT_PG_TABLE", "supervisor_audit_log")
    )

    # Clarification loops before escalating to a human.
    max_clarifications: int = field(
        default_factory=lambda: int(os.getenv("MAX_CLARIFICATIONS", "2"))
    )

    # ── Session lifetime ────────────────────────────────────────────────────
    # GDPR Art. 5(1)(e), SOC 2 CC6.1: measured from the last turn, not the first;
    # evaluated lazily on the next turn, so no sweeper is needed.
    session_max_age_seconds: float = field(
        default_factory=lambda: float(os.getenv("SESSION_MAX_AGE_SECONDS", str(7 * 24 * 3600)))
    )
    # Longer idle bound for a staged artifact: a multi-stage approval can outlast a week.
    approval_max_age_seconds: float = field(
        default_factory=lambda: float(os.getenv("APPROVAL_MAX_AGE_SECONDS", str(30 * 24 * 3600)))
    )

    # ── Long-term memory retention (guardrail layer 3) ──────────────────────
    # Without a ceiling a value answered once drives routing forever. Zero keeps
    # everything, so a deployed environment refuses to serve with it.
    long_term_memory_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.getenv("LONG_TERM_MEMORY_TTL_SECONDS", str(90 * 24 * 3600))
        )
    )

    # ── Inbound message bounds (guardrail layer 1, defence in depth) ────────
    # Twin of the calling application's payload size check (solution §05), which
    # protects only callers who came through it. Keep >= that cap.
    input_max_chars: int = field(default_factory=lambda: int(os.getenv("INPUT_MAX_CHARS", "8000")))

    # ── Guardrail-block anomaly threshold (guardrail layer 6) ───────────────
    # N consecutive blocks is probing or a stuck user — both are reviewer work.
    guardrail_block_streak_limit: int = field(
        default_factory=lambda: int(os.getenv("GUARDRAIL_BLOCK_STREAK_LIMIT", "5"))
    )

    # ── Output guard (guardrail layer 7) ────────────────────────────────────
    # Governs the PII tier only (`sensitive.PII_TIER`); the high tier is always acted
    # on. Switchable: a stakeholder's name in a drafted artifact may be content.
    output_pii_masking: bool = field(
        default_factory=lambda: os.getenv("OUTPUT_PII_MASKING", "true").lower() != "false"
    )
    # ── Worker output handling ──────────────────────────────────────────────
    # Worker output is untrusted, so its size is bounded. 24000 chars (~6000
    # tokens) is more than one artifact and a ceiling on a worker that loops.
    worker_output_max_chars: int = field(
        default_factory=lambda: int(os.getenv("WORKER_OUTPUT_MAX_CHARS", "24000"))
    )

    # ── Checkpoint durability ───────────────────────────────────────────────
    # "sync" per LangGraph HITL guidance: a lost approval checkpoint drops the artifact.
    durability: str = field(default_factory=lambda: os.getenv("SUPERVISOR_DURABILITY", "sync"))

    # ── Conversation window budgets ─────────────────────────────────────────
    # Replay budget per turn, in tokens; the worker's is larger since it builds the artifact.
    history_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("SUPERVISOR_HISTORY_MAX_TOKENS", "4000"))
    )
    worker_history_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("WORKER_HISTORY_MAX_TOKENS", "16000"))
    )

    # Off-domain verdicts below this fall through to route/clarify instead of blocking.
    guardrail_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("GUARDRAIL_CONFIDENCE_THRESHOLD", "0.7"))
    )
    # ── Which agent owns a request the caller can reach several agents for ──
    # Bar for stopping early; below it every candidate is asked and the strongest
    # claim wins. Equal to the threshold restores first-past-the-post (order decides).
    routing_decisive_confidence: float = field(
        default_factory=lambda: float(os.getenv("ROUTING_DECISIVE_CONFIDENCE", "0.9"))
    )
    # A runner-up within this margin is noise: the user is asked which they meant.
    routing_contested_margin: float = field(
        default_factory=lambda: float(os.getenv("ROUTING_CONTESTED_MARGIN", "0.15"))
    )

    # ── Worker dispatch failsafe ────────────────────────────────────────────
    # Backoff retries, then a per-agent circuit breaker so a failing worker is not hammered.
    worker_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("WORKER_MAX_ATTEMPTS", "3"))
    )
    worker_backoff_seconds: float = field(
        default_factory=lambda: float(os.getenv("WORKER_BACKOFF_SECONDS", "1.0"))
    )
    worker_circuit_threshold: int = field(
        default_factory=lambda: int(os.getenv("WORKER_CIRCUIT_THRESHOLD", "3"))
    )
    worker_circuit_cooldown_seconds: float = field(
        default_factory=lambda: float(os.getenv("WORKER_CIRCUIT_COOLDOWN_SECONDS", "60"))
    )

    # ── Graph step ceiling ──────────────────────────────────────────────────
    # LangGraph's default 10007 is no bound for a six-node acyclic graph. 12 is
    # double the real path: an accidental cycle stops here, not in an incident.
    graph_recursion_limit: int = field(
        default_factory=lambda: int(os.getenv("GRAPH_RECURSION_LIMIT", "12"))
    )

    # Local development: replace worker Model Serving calls with an echo mock.
    mock_workers: bool = field(
        default_factory=lambda: os.getenv("SUPERVISOR_MOCK_WORKERS", "").lower() == "true"
    )

    def validate(self) -> list[str]:
        """Findings about disabled or nonsensical safety-critical settings.

        Deployed environments raise on a non-empty list (`build_services`); a local
        run logs each finding at ERROR. None should be reachable by typo when deployed.
        """
        findings: list[str] = []
        if self.turn_budget_seconds <= 0:
            findings.append("TURN_BUDGET_SECONDS<=0 disables the turn time budget (solution §05)")
        if self.max_clarifications < 1:
            findings.append("MAX_CLARIFICATIONS<1 escalates on the first ambiguous turn")
        if not (0.0 <= self.routing_contested_margin <= 1.0):
            findings.append("ROUTING_CONTESTED_MARGIN must be between 0.0 and 1.0")
        if not (0.0 < self.routing_decisive_confidence <= 1.0):
            findings.append("ROUTING_DECISIVE_CONFIDENCE must be between 0.0 and 1.0")
        if self.routing_decisive_confidence < self.guardrail_confidence_threshold:
            findings.append(
                "ROUTING_DECISIVE_CONFIDENCE below GUARDRAIL_CONFIDENCE_THRESHOLD means the "
                "screen stops at the first candidate above the acting threshold, so which "
                "agent owns a contested request is decided by candidate order"
            )
        if not (0.0 < self.guardrail_confidence_threshold <= 1.0):
            findings.append(
                "GUARDRAIL_CONFIDENCE_THRESHOLD outside (0, 1] weakens or disables the "
                "semantic guardrail"
            )
        if self.session_max_age_seconds <= 0:
            findings.append("SESSION_MAX_AGE_SECONDS<=0 disables the session lifetime bound")
        if self.approval_max_age_seconds < self.session_max_age_seconds:
            findings.append(
                "APPROVAL_MAX_AGE_SECONDS below SESSION_MAX_AGE_SECONDS expires a pending "
                "approval sooner than an ordinary conversation"
            )
        if self.long_term_memory_ttl_seconds <= 0:
            findings.append(
                "LONG_TERM_MEMORY_TTL_SECONDS<=0 keeps remembered context forever "
                "(guardrail layer 3: memory expiry)"
            )
        if self.worker_output_max_chars <= 0:
            findings.append("WORKER_OUTPUT_MAX_CHARS<=0 removes the worker output size bound")
        if self.input_max_chars <= 0:
            findings.append("INPUT_MAX_CHARS<=0 removes the inbound message size bound")
        if self.guardrail_block_streak_limit < 1:
            findings.append(
                "GUARDRAIL_BLOCK_STREAK_LIMIT<1 disables the repeated-block escalation "
                "(guardrail probing goes unflagged)"
            )
        if self.routing_llm_timeout_seconds <= 0 or self.worker_timeout_seconds <= 0:
            findings.append("a per-call timeout <= 0 leaves an outbound call unbounded")
        if self.worker_max_attempts < 1 or self.governance_llm_attempts < 1:
            findings.append("retry attempt counts below 1 make every call fail before it starts")
        if self.graph_recursion_limit < 6:
            # Six is the graph's longest legitimate path; below it an ordinary
            # approval turn fails as a recursion error.
            findings.append(
                "GRAPH_RECURSION_LIMIT below 6 is shorter than the graph's longest "
                "legitimate path and will fail normal turns"
            )
        return findings

    def enforce(self) -> None:
        """Apply `validate()`: raise when deployed, log ERROR when local."""
        findings = self.validate()
        if not findings:
            return
        if not is_local_environment(self.environment):
            raise RuntimeError(
                "refusing to serve with safety-critical settings disabled: " + "; ".join(findings)
            )
        for finding in findings:
            logger.error("settings: %s (tolerated only outside deployed environments)", finding)
