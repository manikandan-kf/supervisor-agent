"""Dependency container for the supervisor graph.

Nodes only touch these interfaces, so tests (and future backends) can swap
any of them without changing the graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from agent_governance import audit_trail, rbac
from agent_governance.governed_config_store import Reloading
from agent_governance.output_guard import OutputGuard

from .context_resolver import ContextResolver
from .governed_config import SupervisorConfig
from .memory import LongTermMemory, audit_connection_source, build_store
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


class ResolvesContext(Protocol):
    """The route stage's collaborator (`context_resolver.ContextResolver`)."""

    def resolve(
        self,
        agent,
        history: list[str],
        prior_context: dict[str, str],
        deadline=None,
        carried_over: Optional[dict[str, str]] = None,
    ) -> Any: ...


class AuditSink(Protocol):
    """Where a turn's decision record lands (`agent_governance.audit_trail`)."""

    def log(self, record: dict) -> None: ...


class ResponseGuard(Protocol):
    """Layer 7. Three surfaces, one policy — see the field comment below."""

    def screen(self, text: str) -> Any: ...

    def relay(self, text: str) -> tuple[str, tuple]: ...

    def scrub(self, text: str, fallback: str = "") -> str: ...


@dataclass
class Services:
    settings: Settings
    # In production these three are `governed_config_store.Reloading` proxies over the governed table,
    # so a published change reaches a running endpoint without a redeploy.
    registry: Any  # AgentRegistry, behind a Reloading proxy
    rbac: Any  # RbacPolicy, behind a Reloading proxy
    # The graph calls .screen(query, candidates, history=None) -> ScreenResult;
    # .evaluate(query, agent, history=None) is the single-agent form for direct callers.
    guardrails: Any  # GuardrailEngine, behind a Reloading proxy
    router: ResolvesContext
    workers: WorkerClient
    audit: AuditSink
    memory: LongTermMemory
    # Layer-7 screen (output_guard.py). Three surfaces, one policy: `.screen()` for a worker
    # reply, `.relay()` for the conversation bound *to* a worker, and `.scrub()` for
    # model-written governance text. A test stand-in must provide all three.
    output_guard: ResponseGuard = field(default_factory=OutputGuard)


def build_audit_logger(settings: Settings):
    """The decision-trail sink over the same Postgres the checkpointer uses."""
    return audit_trail.build_audit_logger(audit_connection_source(), settings.audit_pg_table)


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
            },
        )
    )


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or Settings()
    # A deployed process with a safety-critical control disabled by environment variable
    # refuses to build rather than serve without it. See Settings.enforce.
    settings.enforce()

    # Model access is isolated in llm_provider; no node or tool names a model
    # endpoint directly (solution §06: the model is configuration).
    from .llm_provider import get_agent_model, get_routing_model

    llm = get_routing_model(settings)

    # Every per-agent governance call resolves its model through this one seam. Until
    # MULTI_MODEL_ENABLED is true it returns `llm`, so the enable stays a config change.
    def model_for(agent):
        return get_agent_model(settings, agent)

    workers = (
        SimulatedWorkerClient(llm, model_for=model_for)
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
        router=ContextResolver(llm, model_for=model_for),
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
        # Same governed guardrails document and Reloading proxy as the input engine, so
        # publishing an output rule needs no redeploy either.
        output_guard=Reloading(lambda: config.output_guard(mask_pii=settings.output_pii_masking)),
    )
