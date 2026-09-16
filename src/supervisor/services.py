"""Dependency container for the supervisor graph.

Nodes only touch these interfaces, so tests (and future backends) can swap
any of them without changing the graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from agent_governance import audit, rbac
from agent_governance.config_store import Reloading
from agent_governance.output_guard import OutputGuard
from agent_governance.review_queue import NullReviewQueue, ReviewQueue
from agent_governance.spend import SubjectWindow, build_spend_window

from .config import SupervisorConfig
from .memory import LongTermMemory, audit_connection_source, build_store
from .routing import Router
from .settings import Settings
from .worker_client import (
    CircuitBreaker,
    ModelServingWorkerClient,
    SimulatedWorkerClient,
    WorkerClient,
)

logger = logging.getLogger(__name__)


# ── What the container requires of each collaborator ─────────────────────────
#
# Protocols, not concrete classes, so the tests' stand-ins satisfy them structurally.
# `registry`, `rbac` and `guardrails` stay `Any`: in production they are `Reloading` proxies
# resolving every attribute at call time, which no structural type can describe.


class ContextResolver(Protocol):
    """The route stage's collaborator (`routing.Router`)."""

    def resolve(
        self,
        agent,
        history: list[str],
        prior_context: dict[str, str],
        deadline=None,
        carried_over: Optional[dict[str, str]] = None,
    ) -> Any: ...


class AuditSink(Protocol):
    """Where a turn's decision record lands (`agent_governance.audit`)."""

    def log(self, record: dict) -> None: ...


class ReviewSink(Protocol):
    """The appeal / escalation queue (§05 Stage 03, Stage 04).

    Structural on purpose: `NullReviewQueue` is not a subclass of the concrete `ReviewQueue`.
    """

    def open_review(self, **kwargs) -> Any: ...

    def claim_allowance(self, conversation_id: str) -> Any: ...

    def get(self, ref: str) -> Any: ...


class ResponseGuard(Protocol):
    """Layer 7. Four surfaces, one policy — see the field comment below."""

    def screen(self, text: str) -> Any: ...

    def relay(self, text: str) -> tuple[str, tuple]: ...

    def scrub(self, text: str, fallback: str = "") -> str: ...

    def stream_should_hold(self, buffer: str) -> bool: ...

    def stream_findings(self, buffer: str) -> tuple: ...


class SpendWindow(Protocol):
    """One subject's rolling governance-spend allowance (`spend.py`)."""

    def check(self, subject: str) -> str: ...

    def charge(self, subject: str, *, model_calls: int = 0, tokens: int = 0) -> None: ...


@dataclass
class Services:
    settings: Settings
    # In production these three are `config_store.Reloading` proxies over the governed table,
    # so a published change reaches a running endpoint without a redeploy.
    registry: Any  # AgentRegistry, behind a Reloading proxy
    rbac: Any  # RbacPolicy, behind a Reloading proxy
    # The graph calls .screen(query, candidates, history=None) -> ScreenResult;
    # .evaluate(query, agent, history=None) is the single-agent form for direct callers.
    guardrails: Any  # GuardrailEngine, behind a Reloading proxy
    router: ContextResolver
    workers: WorkerClient
    audit: AuditSink
    memory: LongTermMemory
    # Appeal / escalation queue (§05 Stage 03, 04). The `NullReviewQueue` default raises on
    # every write on purpose: an appeal that cannot be recorded must not be reported as flagged.
    reviews: ReviewSink = field(default_factory=lambda: NullReviewQueue())
    # Layer-7 screen (output_guard.py). Four surfaces, one policy: `.screen()` for a worker
    # reply, `.relay()` for the conversation bound *to* a worker, `.scrub()` for model-written
    # governance text, and `stream_*` for live tokens. A test stand-in must provide all four.
    output_guard: ResponseGuard = field(default_factory=OutputGuard)
    # One subject's rolling governance-spend allowance (spend.py). Process-wide rather than
    # per-turn, which is why it lives beside the other long-lived collaborators.
    spend_window: SpendWindow = field(default_factory=SubjectWindow)


def build_audit_logger(settings: Settings):
    """The decision-trail sink over the same Postgres the checkpointer uses."""
    return audit.build_audit_logger(audit_connection_source(), settings.audit_pg_table)


def _report_privileges(settings: Settings) -> None:
    """Check the governance-table grants once, at startup. Never raises."""
    source = audit_connection_source()
    if source is None:
        return
    rbac.report_privileges(
        rbac.check_privileges(
            source,
            {
                "config": settings.config_table,
                "audit": settings.audit_pg_table,
                "reviews": settings.review_queue_table,
            },
        )
    )


def build_review_queue(settings: Settings):
    """The appeal queue in the same Postgres as the audit sink, or a null one.

    Same database on purpose: an appeal is a governance record, and a reviewer should not
    query one store for the decision and another for the appeal against it.
    """
    source = audit_connection_source()
    if source is None:
        logger.warning(
            "review queue: none configured — appeals and escalations cannot be "
            "recorded as queryable state, so Stage 03's appeal path is not met. "
            "Configure Lakebase (LAKEBASE_INSTANCE)."
        )
        return NullReviewQueue()
    logger.info("review queue: Postgres table %s", settings.review_queue_table)
    return ReviewQueue(source, settings.review_queue_table)


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or Settings()
    # A deployed process with a safety-critical control disabled by environment variable
    # refuses to build rather than serve without it. See Settings.enforce.
    settings.enforce()

    # §4.1: model access is isolated in model_provider; no node or tool names a
    # model endpoint directly.
    from .model_provider import get_agent_model, get_routing_model

    llm = get_routing_model(settings)

    # Every per-agent governance call resolves its model through this one seam. Until
    # MULTI_MODEL_ENABLED is true it returns `llm`, so the enable stays a config change.
    def model_for(agent):
        return get_agent_model(settings, agent)

    workers = (
        SimulatedWorkerClient(llm, max_tokens=settings.worker_max_tokens, model_for=model_for)
        if settings.mock_workers
        else ModelServingWorkerClient(
            max_attempts=settings.worker_max_attempts,
            backoff_seconds=settings.worker_backoff_seconds,
            breaker=CircuitBreaker(
                threshold=settings.worker_circuit_threshold,
                cooldown_seconds=settings.worker_circuit_cooldown_seconds,
            ),
            timeout_seconds=settings.worker_timeout_seconds,
        )
    )

    # `config` owns the governed-table-or-bundled-YAML choice, validation and fallback.
    # Wrapped in `Reloading` so a publish is picked up on the next request, not the next deploy.
    config = SupervisorConfig(settings)

    # Did the post-deploy REVOKEs run? (DEPLOYMENT.md §7b.) Until they do, this identity can
    # rewrite the audit trail and the policy that governs it. Non-fatal by design: the bootstrap
    # has the runtime create its own tables, so a new environment holds DDL on first boot.
    _report_privileges(settings)

    return Services(
        settings=settings,
        registry=Reloading(config.registry),
        rbac=Reloading(config.rbac),
        guardrails=Reloading(
            lambda: config.guardrails(
                llm,
                settings.guardrail_confidence_threshold,
                model_for=model_for,
                decisive_threshold=settings.routing_decisive_confidence,
                contested_margin=settings.routing_contested_margin,
            )
        ),
        router=Router(llm, model_for=model_for),
        workers=workers,
        # Built after the store, so both share one Postgres connection.
        audit=build_audit_logger(settings),
        # The allowlist derives from the registry: the only context worth persisting is
        # context some agent declared it needs (§04). A callable, for the same reason the
        # registry is a proxy — frozen here it would refuse a key added a minute ago.
        memory=LongTermMemory(
            build_store(),
            allowed_keys=lambda: config.registry().context_keys(),
            ttl_seconds=settings.long_term_memory_ttl_seconds,
        ),
        reviews=build_review_queue(settings),
        # Same governed guardrails document and Reloading proxy as the input engine, so
        # publishing an output rule needs no redeploy either.
        output_guard=Reloading(lambda: config.output_guard(mask_pii=settings.output_pii_masking)),
        # Built once per process: the window it keeps *is* the cross-turn state. Not `Reloading`
        # — rebuilding on a publish would hand an exhausted subject a fresh allowance.
        spend_window=build_spend_window(settings),
    )
