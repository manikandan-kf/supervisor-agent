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
from agent_governance.trust import trust_secret

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
    # Client-selected from the approved model list (ASM-04); wired via DAB target vars.
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
    # **Zero on purpose**: `resilience.invoke_with_retries` is the one retry authority.
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
    # call = 120 + 45 = 165s < 180s gateway. **Re-check the sum if either bound moves.**
    turn_budget_seconds: float = field(
        default_factory=lambda: float(os.getenv("TURN_BUDGET_SECONDS", "120"))
    )
    # ── The turn's spend envelope (cost control) ────────────────────────────
    # Spend ceiling (spend.py), supervisor's own calls only. **Zero disables each
    # cap independently.** 8 = screen (one per candidate, 4) + route, with headroom.
    turn_max_model_calls: int = field(
        default_factory=lambda: int(os.getenv("TURN_MAX_MODEL_CALLS", "8"))
    )
    # ~6k tokens x 8 calls; an estimate, so the call ceiling is the primary control.
    turn_max_tokens: int = field(default_factory=lambda: int(os.getenv("TURN_MAX_TOKENS", "60000")))
    # Per-subject rolling allowance, per replica (spend.py). Off by default: it
    # wants a value from observed spend, and refusing real users is worse than none.
    subject_max_model_calls: int = field(
        default_factory=lambda: int(os.getenv("SUBJECT_MAX_MODEL_CALLS", "0"))
    )
    subject_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("SUBJECT_MAX_TOKENS", "0"))
    )
    subject_spend_window_seconds: float = field(
        default_factory=lambda: float(os.getenv("SUBJECT_SPEND_WINDOW_SECONDS", "3600"))
    )

    # Mirrored from the platform (§2.2); selects the promoted prompt alias.
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "local"))
    config_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SUPERVISOR_CONFIG_DIR", str(_PACKAGE_CONFIG)))
    )

    # ── Governance configuration source ─────────────────────────────────────
    # R1 wants config in governed tables; the SP cannot hold `databricks-sql-access`,
    # so the table is the Lakebase one (config_store.py). auto = table if Postgres
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

    # ── Thread execution serialization (§4.4) ───────────────────────────────
    # §4.4 "only one execution updates a thread at a time"; Model Serving has no affinity.
    thread_lock_enabled: bool = field(
        default_factory=lambda: os.getenv("THREAD_LOCK_ENABLED", "true").lower() != "false"
    )
    # Absorbs a normal turn queued ahead (P95 ~8s); past this the turn is refused.
    thread_lock_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("THREAD_LOCK_TIMEOUT_SECONDS", "15"))
    )
    thread_lock_poll_seconds: float = field(
        default_factory=lambda: float(os.getenv("THREAD_LOCK_POLL_SECONDS", "0.25"))
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

    # ── Session notes (short-term memory) ───────────────────────────────────
    # "Keep this in mind" notes held for the conversation (session_notes.py). Bounded
    # because the text is replayed into a later prompt. **Zero disables the feature.**
    session_notes_max: int = field(
        default_factory=lambda: int(os.getenv("SESSION_NOTES_MAX", "20"))
    )
    session_note_max_chars: int = field(
        default_factory=lambda: int(os.getenv("SESSION_NOTE_MAX_CHARS", "500"))
    )

    # ── Appeal / escalation review queue (§05 Stage 03, Stage 04) ───────────
    # Queryable review state (review_queue.py): a log entry nothing reads back is not a control.
    review_queue_table: str = field(
        default_factory=lambda: os.getenv("REVIEW_QUEUE_TABLE", "supervisor_review_queue")
    )

    # ── Session lifetime (§05 Stage 04) ─────────────────────────────────────
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
    # Twin of the gateway's 8000-char cap, which protects only callers who came
    # through it. Keep >= the gateway's cap.
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
    # Live relay runs through `output_guard.StreamGuard`; off closes the residual
    # entirely and the closing item carries the whole guarded answer.
    output_stream_worker_tokens: bool = field(
        default_factory=lambda: os.getenv("OUTPUT_STREAM_WORKER_TOKENS", "true").lower() != "false"
    )
    # Longer than any contiguous credential shape, shorter than a sentence. Zero
    # lets a value split across tokens escape — `validate()` flags it when deployed.
    output_stream_holdback_chars: int = field(
        default_factory=lambda: int(os.getenv("OUTPUT_STREAM_HOLDBACK_CHARS", "160"))
    )
    # Grounding footnote for unevidenced claims (grounding.py); never a rewrite.
    output_provenance_notes: bool = field(
        default_factory=lambda: os.getenv("OUTPUT_PROVENANCE_NOTES", "true").lower() != "false"
    )

    # ── Worker output handling (§05 Stage 06) ───────────────────────────────
    # "Treat worker output as untrusted — bound its size." 24000 chars (~6000
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

    # ── Worker output budget ────────────────────────────────────────────────
    # The prompt's length rule is advisory; this bounds spend when shaping fails.
    # 1024 tokens ~ 750 words.
    worker_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("WORKER_MAX_TOKENS", "1024"))
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
        if not self.thread_lock_enabled:
            findings.append("THREAD_LOCK_ENABLED=false disables execution serialization (§4.4)")
        if self.turn_budget_seconds <= 0:
            findings.append("TURN_BUDGET_SECONDS<=0 disables the turn time budget (§05 Stage 05)")
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
        if self.output_stream_worker_tokens and self.output_stream_holdback_chars < 40:
            findings.append(
                "OUTPUT_STREAM_HOLDBACK_CHARS below 40 lets a sensitive value split across "
                "streamed tokens reach the client before the output guard sees it whole"
            )
        if self.guardrail_block_streak_limit < 1:
            findings.append(
                "GUARDRAIL_BLOCK_STREAK_LIMIT<1 disables the repeated-block escalation "
                "(guardrail probing goes unflagged)"
            )
        if self.routing_llm_timeout_seconds <= 0 or self.worker_timeout_seconds <= 0:
            findings.append("a per-call timeout <= 0 leaves an outbound call unbounded")
        if self.worker_max_attempts < 1 or self.governance_llm_attempts < 1:
            findings.append("retry attempt counts below 1 make every call fail before it starts")
        if self.turn_max_model_calls <= 0 and self.turn_max_tokens <= 0:
            findings.append(
                "TURN_MAX_MODEL_CALLS and TURN_MAX_TOKENS are both <=0, so a turn has no "
                "spend ceiling — only a time bound (see spend.py)"
            )
        elif 0 < self.turn_max_model_calls < 2:
            # One call is not a working configuration: an ordinary turn spends
            # a screen verdict *and* a context resolution.
            findings.append(
                "TURN_MAX_MODEL_CALLS=1 refuses context resolution on every turn that "
                "passes the screen — a normal turn needs at least 2"
            )
        if self.graph_recursion_limit < 6:
            # Six is the graph's longest legitimate path; below it an ordinary
            # approval turn fails as a recursion error.
            findings.append(
                "GRAPH_RECURSION_LIMIT below 6 is shorter than the graph's longest "
                "legitimate path and will fail normal turns"
            )
        if (
            self.subject_max_model_calls > 0 or self.subject_max_tokens > 0
        ) and self.subject_spend_window_seconds <= 0:
            findings.append(
                "a subject spend cap is set with SUBJECT_SPEND_WINDOW_SECONDS<=0, which "
                "disables the window and therefore the cap"
            )
        # ── The entitlement signature (§2.7) ────────────────────────────────
        # Without the shared secret nothing is checked, `context.verified` stays
        # True, and any CAN QUERY principal can name its own permitted set.
        if not trust_secret():
            findings.append(
                "SUPERVISOR_TRUST_SECRET is unset, so entitlement signatures cannot be "
                "verified and a caller's own permitted_agents set is taken at face value "
                "(§2.7 — see DEPLOYMENT.md §7e)"
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

    @property
    def guardrails_config(self) -> Path:
        """The bundled guardrails seed — what the tier-1 pattern tests read."""
        return self.config_dir / "guardrails.yaml"
