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

# `ENVIRONMENT` answers two questions — which deployment's resources to use,
# and whether a control may degrade — and the shared library owns both answers
# (`agent_governance.environment`). What is the supervisor's alone is the
# prefix of its per-environment schema.
_SCHEMA_PREFIX = "supervisor"


def environment_schema(environment: str | None = None) -> str:
    """`supervisor_dev`, `supervisor_prod`, … — this agent's per-environment schema.

    Used unqualified as the Lakebase Postgres schema and qualified with the
    catalog as the Unity Catalog schema, so the two cannot drift.
    """
    return _environment_schema(_SCHEMA_PREFIX, environment)


@dataclass(frozen=True)
class Settings:
    # The routing LLM is client-selected (approved model list, ASM-04). Dev runs
    # a medium-tier endpoint, prod a high-tier one — wired via DAB target vars.
    routing_llm_endpoint: str = field(
        default_factory=lambda: os.getenv("ROUTING_LLM_ENDPOINT", "databricks-claude-sonnet-4-5")
    )
    # Governance decisions should be reproducible, so the routing model runs
    # near-deterministic unless deliberately overridden.
    routing_temperature: float = field(
        default_factory=lambda: float(os.getenv("ROUTING_TEMPERATURE", "0.0"))
    )

    # ── Multi-model support ─────────────────────────────────────────────────
    # When true, a registry entry's optional `model` field selects the serving
    # endpoint that agent's governance calls use — semantic screen, context
    # resolution, worker simulation — instead of `routing_llm_endpoint`. Off by
    # default: the feature ships switched off, so
    # the supporting code ships dark and enabling it later is configuration —
    # this flag plus `model:` entries in the published agents document — not a
    # deploy. With the flag off a declared `model` is ignored and logged once,
    # so a config published early cannot silently change which endpoint spends
    # tokens. See model_provider.get_agent_model for the resolution rules.
    multi_model_enabled: bool = field(
        default_factory=lambda: os.getenv("MULTI_MODEL_ENABLED", "").lower() == "true"
    )

    # ── Outbound call bounds ────────────────────────────────────────────────
    # Framework "Execution timeout": every invocation needs a configurable
    # overall timeout, and it assigns the *overall* bound to the framework while
    # leaving "tool-specific timeouts, retries, loop limits and other execution
    # controls" to the agent team. These are that half. The overall bound is the
    # gateway's `INVOCATION_TIMEOUT_SECONDS`.
    #
    # Neither library defaults to a usable value: `ChatDatabricks.timeout` is None (so the underlying
    # OpenAI client's 600-second read timeout applied) and the Databricks SDK's
    # `http_timeout_seconds` is None. A routing model that accepted the
    # connection and then stalled held a serving worker for ten minutes per
    # attempt.
    routing_llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("ROUTING_LLM_TIMEOUT_SECONDS", "30"))
    )
    # **Zero on purpose.** The chat client's own retry layer stays off — the
    # OpenAI client's default is 2 retries, invisible to the node span the
    # latency KPI is computed from. One retry authority, and it is
    # `resilience.invoke_with_retries`, wrapped around each individual
    # governance model call: declared, deadline-aware, and it retries only the
    # verdict that failed rather than re-screening every candidate agent.
    # (The graph-level RetryPolicy that used to claim this job never fired —
    # the nodes catch every exception and return a fail-closed Command, so the
    # policy saw nothing. It has been removed rather than left as a comforting
    # comment.) If a client-level budget is ever preferred, set it here and
    # drop the resilience wrapper — but do not run both.
    routing_llm_max_retries: int = field(
        default_factory=lambda: int(os.getenv("ROUTING_LLM_MAX_RETRIES", "0"))
    )
    # How many attempts `resilience.invoke_with_retries` gives one governance
    # model call (screen verdict, context resolution) before the node's
    # fail-closed hold takes over. Transient failures only; bounded by the turn
    # budget either way.
    governance_llm_attempts: int = field(
        default_factory=lambda: int(os.getenv("GOVERNANCE_LLM_ATTEMPTS", "3"))
    )
    # Bounds one dispatch attempt. `WORKER_MAX_ATTEMPTS` (below) then bounds how
    # many attempts there are, so the worst case is roughly
    # attempts x timeout + backoff.
    worker_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("WORKER_TIMEOUT_SECONDS", "45"))
    )
    # ── The turn's total time budget ────────────────────────────────────────
    # Governance Blueprint §05 Stage 05 asks for "per-attempt timeout, **a total
    # time budget**, bounded retries with backoff, then circuit-break". The
    # per-call bounds above are necessary and not sufficient, because they
    # multiply by two things invisible from any single call site: the guardrail
    # screen fans out one semantic call per candidate agent, and `resilience`
    # gives each call up to `governance_llm_attempts` tries. 4 agents x 30s x 3
    # attempts is 361s for one node — against a gateway that gives up at 180s.
    #
    # The budget is enforced by refusing to *start* a call that cannot finish
    # in time, not by cancelling one in flight — a per-call timeout rebind would
    # defeat `model_provider`'s client cache and rebuild an HTTP client per
    # verdict. So the guarantee is:
    #
    #     worst case  =  turn_budget_seconds + the longest single call
    #                 =  120 + 45 (a worker dispatch attempt)
    #                 =  165s,  against the gateway's 180s
    #
    # leaving ~15s for `respond` and the audit write. That headroom is the whole
    # point: exhausting this budget raises *inside* the graph, so `respond`
    # still produces a governed message and still writes the audit row.
    # Overrunning the gateway instead produces neither — the compliance trail is
    # lost exactly on the failure path. **If either bound is retuned, re-check
    # that sum**; see `deadline.py`.
    turn_budget_seconds: float = field(
        default_factory=lambda: float(os.getenv("TURN_BUDGET_SECONDS", "120"))
    )
    # ── The turn's spend envelope (cost control) ────────────────────────────
    # `turn_budget_seconds` above bounds how *long* a turn may take; these
    # bound how much it may *spend*. The two are not the same control — a turn
    # can sit well inside 120 seconds while making one governance call per
    # candidate agent, each carrying a 4000-token window and an 8000-character
    # query, and the gateway's rate limiter counts requests rather than spend.
    #
    # The quantity being bounded is the tokens the Supervisor spends on its own
    # governance calls per turn — routing, guardrails, clarification — tracked
    # separately from any worker agent's consumption. See spend.py for what is
    # charged (the supervisor's own calls, never a worker's) and why calls are
    # the enforced unit while tokens are the estimated one.
    #
    # **Zero disables each cap independently**, so an existing deployment is
    # unchanged until an operator sets one. The defaults are sized from the
    # real worst case rather than picked round: the screen makes at most one
    # call per candidate agent (4 in `agents.yaml`) and route makes one, so 8
    # calls is comfortable headroom over a legitimate turn and still an order
    # of magnitude below a fan-out gone wrong.
    turn_max_model_calls: int = field(
        default_factory=lambda: int(os.getenv("TURN_MAX_MODEL_CALLS", "8"))
    )
    # ~6k tokens per governance call (4000-token window + 8000-char query +
    # the system prompt) x 8 calls. An estimate bounding an estimate, which is
    # why the call ceiling above is the primary control.
    turn_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("TURN_MAX_TOKENS", "60000"))
    )
    # One subject's rolling governance allowance — the backstop against a
    # sustained, authenticated, entirely in-policy way to drive cost that a
    # per-request rate limit does not bound. Per replica (see spend.py); the
    # platform-level equivalent is Unity AI Gateway's hard budget limits, and
    # the durable per-turn figures in the audit table are what a cross-replica
    # report aggregates. Off by default: a ceiling that refuses real users is
    # worse than none, so it wants a value chosen from this deployment's own
    # observed spend rather than a guess shipped in code.
    subject_max_model_calls: int = field(
        default_factory=lambda: int(os.getenv("SUBJECT_MAX_MODEL_CALLS", "0"))
    )
    subject_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("SUBJECT_MAX_TOKENS", "0"))
    )
    subject_spend_window_seconds: float = field(
        default_factory=lambda: float(os.getenv("SUBJECT_SPEND_WINDOW_SECONDS", "3600"))
    )

    # Deployment environment, mirrored from the platform (§2.2). Names *this*
    # deployment — `dev`, `prod`, or `local` off Databricks — and selects the
    # promoted prompt alias. `is_local_environment` above is the other half:
    # this value says which resources to use, that one says whether a control
    # may degrade. The deploy stamps it onto the endpoint explicitly, so the
    # default below only ever applies to a workstation run.
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "local"))
    config_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SUPERVISOR_CONFIG_DIR", str(_PACKAGE_CONFIG)))
    )

    # ── Governance configuration source ─────────────────────────────────────
    # R1 puts the supervisor's configuration "directly in Unity Catalog tables
    # (pre-wrapper-API)". `config_store.py` explains which table and why that
    # one; in short, the endpoint's service principal can never hold
    # `databricks-sql-access`, so the governed table it *can* read is the
    # Lakebase one registered in Unity Catalog as `supervisor_memory`.
    #
    #   auto   use the table when Postgres is configured, else the bundled YAML
    #   table  require the table; log an error and fall back if it is missing
    #   files  bundled YAML only — offline tests and local iteration
    config_source: str = field(
        default_factory=lambda: os.getenv("SUPERVISOR_CONFIG_SOURCE", "auto").strip().lower()
    )
    config_table: str = field(
        default_factory=lambda: os.getenv("SUPERVISOR_CONFIG_TABLE", "supervisor_config")
    )
    # How long a loaded document is trusted before the table is consulted again.
    # 60s matches MLflow's prompt-alias cache, for the same reason: it is the
    # difference between "published" and "live" that an operator has to reason
    # about, so the two governed things that can change under a running endpoint
    # should not have different answers.
    config_cache_seconds: float = field(
        default_factory=lambda: float(os.getenv("CONFIG_CACHE_TTL_SECONDS", "60"))
    )

    # ── Thread execution serialization (§4.4) ───────────────────────────────
    # "Only one execution updates a thread at a time." Model Serving does not
    # do this for us — several replicas, several worker processes, no affinity
    # by conversation — so `agent.py` takes `agent_governance.locking`'s
    # advisory lock around every graph run. See that module for where it
    # degrades.
    thread_lock_enabled: bool = field(
        default_factory=lambda: os.getenv("THREAD_LOCK_ENABLED", "true").lower() != "false"
    )
    # Long enough to absorb a normal governed turn ahead of it in the queue
    # (P95 is ~8s), short enough that a user is told what is happening rather
    # than watching a spinner. Past this the second turn is refused, not run.
    thread_lock_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("THREAD_LOCK_TIMEOUT_SECONDS", "15"))
    )
    thread_lock_poll_seconds: float = field(
        default_factory=lambda: float(os.getenv("THREAD_LOCK_POLL_SECONDS", "0.25"))
    )

    # ── Audit sink ──────────────────────────────────────────────────────────
    # Postgres when a DSN or Lakebase instance resolves — one INSERT on the
    # connection the checkpointer already holds, no extra entitlement, and it
    # works from inside Model Serving. Otherwise the process log: never lost,
    # but not queryable. There is no SQL-warehouse sink: the endpoint's system
    # service principal can never hold `databricks-sql-access`.
    # Postgres table for the decision trail. Unqualified names land in the
    # connection's default schema.
    audit_pg_table: str = field(
        default_factory=lambda: os.getenv("AUDIT_PG_TABLE", "supervisor_audit_log")
    )

    # Clarification loops before escalating to a human.
    max_clarifications: int = field(default_factory=lambda: int(os.getenv("MAX_CLARIFICATIONS", "2")))

    # ── Session notes (short-term memory) ───────────────────────────────────
    # "…, keep this in mind for later" is acknowledged and held for the rest of
    # the conversation instead of being dispatched as work. See
    # session_notes.py — in particular why a note reaches the worker and never
    # a governance model.
    #
    # Both bounds exist because the note is user-written text replayed into a
    # later prompt: without them one conversation could pin an unbounded amount
    # of it. 20 x 500 characters is far more than a working conversation
    # accumulates and still an order of magnitude below the dispatch window.
    #
    # **Zero disables the feature**, and the fallback is the previous behaviour
    # (the turn is routed and dispatched like any other), not a broken one — so
    # this is an operator preference rather than a safety control, and
    # `validate()` deliberately says nothing about it.
    session_notes_max: int = field(
        default_factory=lambda: int(os.getenv("SESSION_NOTES_MAX", "20"))
    )
    session_note_max_chars: int = field(
        default_factory=lambda: int(os.getenv("SESSION_NOTE_MAX_CHARS", "500"))
    )

    # ── Appeal / escalation review queue (§05 Stage 03, Stage 04) ───────────
    # Queryable state with an authorized reviewer action, because the blueprint
    # is explicit that "a fire-and-forget log entry that nothing reads back is
    # not a control". See review_queue.py.
    review_queue_table: str = field(
        default_factory=lambda: os.getenv("REVIEW_QUEUE_TABLE", "supervisor_review_queue")
    )

    # ── Session lifetime (§05 Stage 04) ─────────────────────────────────────
    # "Bound session lifetime and record duration, so an abandoned session
    # cannot hold an approval or context open indefinitely" — GDPR Art. 5(1)(e),
    # SOC 2 CC6.1. Seven days of **inactivity**: measured from the last turn,
    # not from the conversation's first turn, because a conversation someone is
    # actively working in is not abandoned however old it is — expiring it
    # mid-use wipes context the user just established, which is data loss
    # dressed up as hygiene. Expiry is evaluated lazily on the next turn rather
    # than by a sweeper, so it needs no scheduled job to be correct; a sweeper
    # would only make the *storage* expire on time, which is what the retention
    # sweep is for.
    session_max_age_seconds: float = field(
        default_factory=lambda: float(os.getenv("SESSION_MAX_AGE_SECONDS", str(7 * 24 * 3600)))
    )
    # A conversation holding a staged artifact awaiting sign-off gets a longer
    # idle bound before expiry destroys the draft. A multi-stage approval flow
    # (HLD -> LLD -> Epic) legitimately spans more than a week — a reviewer on
    # leave, a public holiday — and silently discarding in-progress work on a
    # timer is the wrong answer to it. Thirty days keeps the Stage 04 bound (an
    # abandoned approval still cannot hang open forever) while making room for
    # ordinary absence.
    approval_max_age_seconds: float = field(
        default_factory=lambda: float(
            os.getenv("APPROVAL_MAX_AGE_SECONDS", str(30 * 24 * 3600))
        )
    )

    # ── Long-term memory retention (guardrail layer 3) ──────────────────────
    # How long a remembered `required_context` value keeps feeding later turns.
    # Unlike a session, this survives the conversation that established it, so
    # without a ceiling a product line answered once drives routing prompts
    # indefinitely — including after the thing it named was reorganised away.
    # Evaluated lazily on read (see `memory.LongTermMemory`). Zero or less keeps
    # everything forever — a retention control switched off, so a deployed
    # environment refuses to serve with it, exactly as it does for the session
    # bound above.
    long_term_memory_ttl_seconds: float = field(
        default_factory=lambda: float(
            os.getenv("LONG_TERM_MEMORY_TTL_SECONDS", str(90 * 24 * 3600))
        )
    )

    # ── Inbound message bounds (guardrail layer 1, defence in depth) ────────
    # The gateway's InvocationRequest already caps `input` at 8000 characters,
    # but that bound lives in a different deployable and protects only callers
    # who came through it. A direct endpoint invocation had
    # no size ceiling at all inside the graph, and every downstream consumer —
    # the token-window trimmer, the checkpointer, the audit excerpts — was
    # sized on the assumption the gateway's cap held. Mirrored here so the
    # assumption is enforced where it is relied on. Keep >= the gateway's cap,
    # or legitimate front-door traffic gets refused in-graph.
    input_max_chars: int = field(
        default_factory=lambda: int(os.getenv("INPUT_MAX_CHARS", "8000"))
    )

    # ── Guardrail-block anomaly threshold (guardrail layer 6) ───────────────
    # Consecutive blocked turns in one conversation before the supervisor stops
    # answering each probe individually and escalates the conversation to a
    # human reviewer. A user hitting the screen once or twice is a user
    # exploring scope; the same conversation blocked five times in a row is
    # either someone probing the guardrails or someone genuinely stuck — both
    # are review-queue work, not more refusals. Uses the same terminal
    # review-hold machinery as the clarification cap.
    guardrail_block_streak_limit: int = field(
        default_factory=lambda: int(os.getenv("GUARDRAIL_BLOCK_STREAK_LIMIT", "5"))
    )

    # ── Output guard (guardrail layer 7) ────────────────────────────────────
    # The high tier — credentials, government identifiers, payment, bank and
    # health identifiers — is always acted on (see output_guard.py). This flag
    # governs the *PII tier* only: names, contact details, dates of birth and
    # health conditions (`sensitive.PII_TIER`). On by default, switchable
    # because a stakeholder's name inside a drafted artifact may be the content
    # rather than a leak. It never relaxes the high tier.
    output_pii_masking: bool = field(
        default_factory=lambda: os.getenv("OUTPUT_PII_MASKING", "true").lower() != "false"
    )
    # Whether a worker's answer tokens are relayed live at all. On, they pass
    # through `output_guard.StreamGuard` — masked behind a hold-back window
    # and stopped at the first withhold-tier finding. Off closes the residual
    # entirely at the cost of the live stream: the closing item carries the
    # whole guarded answer. Only the simulated worker streams tokens today; a
    # Model Serving worker returns its response whole.
    output_stream_worker_tokens: bool = field(
        default_factory=lambda: os.getenv("OUTPUT_STREAM_WORKER_TOKENS", "true").lower() != "false"
    )
    # How many characters of streamed text wait behind the emission point.
    # Longer than every contiguous credential or identifier shape the
    # catalogue recognises (a private key has its own hold rule), shorter than
    # a sentence of a drafted artifact. Zero relays each token as it arrives —
    # a masked token is still masked, but a value split across two tokens is
    # not — so `validate()` flags it in deployed environments.
    output_stream_holdback_chars: int = field(
        default_factory=lambda: int(os.getenv("OUTPUT_STREAM_HOLDBACK_CHARS", "160"))
    )
    # Append the grounding cue when a reply asserts an executed action or
    # cites a source the worker never declared (grounding.py). A footnote,
    # never a rewrite — switchable because a worker fleet that ships its own
    # execution evidence on the wire makes the cue redundant.
    output_provenance_notes: bool = field(
        default_factory=lambda: os.getenv("OUTPUT_PROVENANCE_NOTES", "true").lower() != "false"
    )

    # ── Worker output handling (§05 Stage 06) ───────────────────────────────
    # "Treat worker output as untrusted — bound its size, sanitize before it
    # reaches downstream tooling". `worker_max_tokens` bounds what a *simulated*
    # worker generates; this bounds what the supervisor will relay from any
    # worker, real or mocked, and applies to output the supervisor did not
    # produce and cannot constrain at the source. 24000 characters is roughly
    # 6000 tokens — comfortably more than one legitimate artifact, and a hard
    # ceiling on a worker that streams a loop. See sanitize.py.
    worker_output_max_chars: int = field(
        default_factory=lambda: int(os.getenv("WORKER_OUTPUT_MAX_CHARS", "24000"))
    )

    # ── Checkpoint durability ───────────────────────────────────────────────
    # How eagerly LangGraph persists each step's checkpoint: "sync" writes it
    # before the next step runs, "async" writes it while the next step runs,
    # "exit" writes only when the graph finishes or interrupts. LangGraph's own
    # guidance is "sync" for production human-in-the-loop flows, and this graph
    # is one: an approval gate whose checkpoint was lost to a crash would drop
    # the staged artifact and the pause with it. The latency cost is one
    # Lakebase round-trip per stage, which the relay pattern absorbs.
    durability: str = field(
        default_factory=lambda: os.getenv("SUPERVISOR_DURABILITY", "sync")
    )

    # ── Conversation window budgets ─────────────────────────────────────────
    # The checkpointer keeps the whole conversation; these bound how much of it
    # is *replayed* per turn, counted in tokens rather than messages so ten
    # one-line turns are not treated the same as ten pasted documents.
    # `history_max_tokens` feeds the governance models (screen, route);
    # `worker_history_max_tokens` bounds what a dispatch relays to the worker,
    # larger because the worker is the one producing the artifact.
    history_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("SUPERVISOR_HISTORY_MAX_TOKENS", "4000"))
    )
    worker_history_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("WORKER_HISTORY_MAX_TOKENS", "16000"))
    )

    # Semantic guardrail: off-domain verdicts below this confidence fall through
    # to route/clarify instead of hard-blocking.
    guardrail_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("GUARDRAIL_CONFIDENCE_THRESHOLD", "0.7"))
    )
    # ── Which agent owns a request the caller can reach several agents for ──
    #
    # The screen asks each permitted agent in turn and used to stop at the first
    # verdict above the threshold above. Candidates are ordered target-first, so
    # that made ownership a property of which chat widget the user opened rather
    # than of the request: "write a unit test for this" is the Coding Agent's by
    # one reading and the Test Case Agent's by another, and whoever was asked
    # first won without the other ever being consulted.
    #
    # `decisive` is the bar for stopping early. Below it the remaining
    # candidates are asked and the strongest claim wins — a comparison rather
    # than an ordering. Raising it buys accuracy with model calls: a decisive
    # verdict still costs one call, a borderline one costs up to one per
    # reachable agent, bounded by the turn deadline and charged to the turn's
    # spend ledger. 1.0 asks every candidate on every turn; setting it equal to
    # the threshold above restores the old first-past-the-post behaviour.
    routing_decisive_confidence: float = field(
        default_factory=lambda: float(os.getenv("ROUTING_DECISIVE_CONFIDENCE", "0.9"))
    )
    # How close the runner-up has to be before the difference is treated as
    # noise rather than a decision. Within this margin the user is asked which
    # deliverable they meant, because a 0.72-to-0.71 split is not a judgement —
    # it is a coin toss with a decimal point, and the user can settle it in one
    # word. 0 disables the question and always takes the top score.
    routing_contested_margin: float = field(
        default_factory=lambda: float(os.getenv("ROUTING_CONTESTED_MARGIN", "0.15"))
    )

    # ── Worker dispatch failsafe ────────────────────────────────────────────
    # Transient worker failures are retried with exponential backoff; repeated
    # failure opens a per-agent circuit breaker so a failing worker is not
    # hammered. The user then gets a clear "temporarily unavailable" message.
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
    # Hard cap on the tokens a simulated worker may generate per turn. The
    # prompt's own length rule ("under 400 words") is advisory — a model given
    # a broad ask will happily emit HLD + LLD + user stories in one response
    # and blow through it. max_tokens is the only guaranteed ceiling, so both
    # exist: the prompt shapes the answer (one deliverable, answer-first), the
    # cap bounds the spend when shaping fails. 1024 tokens ≈ 750 words — room
    # for one full artifact, not for three.
    worker_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("WORKER_MAX_TOKENS", "1024"))
    )

    # ── Graph step ceiling ──────────────────────────────────────────────────
    # LangGraph's own default is 25 steps. The graph is acyclic and its longest
    # path is six nodes (rbac_gate -> guardrails -> route -> dispatch ->
    # approval -> respond), so the default is unreachable by construction — and
    # "unreachable by construction" is an argument in a comment, not a bound in
    # code. Setting it turns the argument into an assertion: if a future edge
    # ever makes the graph cyclic, the ceiling stops it in this process rather
    # than in an incident. 12 is double the real path, which leaves room for a
    # legitimately longer flow to be *noticed* rather than silently absorbed.
    graph_recursion_limit: int = field(
        default_factory=lambda: int(os.getenv("GRAPH_RECURSION_LIMIT", "12"))
    )

    # Local development: replace worker Model Serving calls with an echo mock.
    mock_workers: bool = field(
        default_factory=lambda: os.getenv("SUPERVISOR_MOCK_WORKERS", "").lower() == "true"
    )

    def validate(self) -> list[str]:
        """Findings about disabled or nonsensical safety-critical settings.

        Every control parameter here is a single environment variable, and
        several values silently disable a control outright:
        `THREAD_LOCK_ENABLED=false` drops execution serialization,
        `TURN_BUDGET_SECONDS=0` removes the time budget,
        `MAX_CLARIFICATIONS=0` escalates on the first ambiguous turn, and
        `GUARDRAIL_CONFIDENCE_THRESHOLD=0` turns every off-domain verdict into
        a pass-through. None of that should be reachable by typo in a deployed
        environment.

        Returns the list of findings. In any deployed environment — `dev` on
        Databricks as much as `prod` — the caller (`build_services`) raises on a
        non-empty list, refusing to boot rather than serving with a control off.
        Only a local run logs each finding at ERROR instead, so experimentation
        on a workstation stays possible but never silent.
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
            # one screen verdict *and* one context resolution.
            findings.append(
                "TURN_MAX_MODEL_CALLS=1 refuses context resolution on every turn that "
                "passes the screen — a normal turn needs at least 2"
            )
        if self.graph_recursion_limit < 6:
            # Six is the graph's longest legitimate path. Below it, an ordinary
            # approval turn fails as a recursion error — a control tuned into
            # an outage.
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
        return findings

    def enforce(self) -> None:
        """Apply `validate()`: raise when deployed, log ERROR when local."""
        findings = self.validate()
        if not findings:
            return
        if not is_local_environment(self.environment):
            raise RuntimeError(
                "refusing to serve with safety-critical settings disabled: "
                + "; ".join(findings)
            )
        for finding in findings:
            logger.error("settings: %s (tolerated only outside deployed environments)", finding)

    @property
    def guardrails_config(self) -> Path:
        """The bundled guardrails seed — what the tier-1 pattern tests read."""
        return self.config_dir / "guardrails.yaml"
