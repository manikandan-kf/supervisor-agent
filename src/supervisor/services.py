"""Dependency container for the supervisor graph.

Nodes only touch these interfaces, so tests (and future backends) can swap
any of them without changing the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .audit import build_audit_logger
from .config_store import ConfigProvider, Reloading
from .dispatch import CircuitBreaker, ModelServingWorkerClient, SimulatedWorkerClient
from .memory import LongTermMemory, build_store
from .output_guard import OutputGuard
from .rbac import RbacPolicy
from .registry import AgentRegistry
from .review_queue import NullReviewQueue, build_review_queue
from .routing import Router
from .settings import Settings
from .spend import SubjectWindow, build_spend_window


@dataclass
class Services:
    settings: Settings
    # In production these three are `config_store.Reloading` proxies over the
    # governed configuration table, so a published change reaches a running
    # endpoint without a redeploy. Nodes cannot tell the difference, and tests
    # pass the concrete objects.
    registry: AgentRegistry
    rbac: RbacPolicy
    # The graph calls .screen(query, candidates, history=None) -> ScreenResult.
    # .evaluate(query, agent, history=None) -> GuardrailResult is the single-agent
    # form, kept for direct callers and tests; no node uses it.
    guardrails: object
    router: object  # exposes .resolve(agent, history, prior_context) -> RouteResult
    workers: object  # WorkerClient
    audit: object  # exposes .log(record)
    memory: LongTermMemory
    # The appeal / escalation queue (§05 Stage 03, Stage 04). Exposes
    # .open_review(...), .resolve(...), .claim_allowance(...), .get(...),
    # .list_open(...) and .open_for_conversation(...). Defaults to a
    # `NullReviewQueue` so a hand-built `Services` in a test needs no argument;
    # that stand-in raises on every write, which is the correct behaviour rather
    # than a convenience — an appeal that cannot be recorded must not be
    # reported to the user as flagged.
    reviews: object = field(default_factory=lambda: NullReviewQueue())
    # Layer-7 screen (output_guard.py). Four surfaces, one policy: `.screen()`
    # for a worker reply (allow / mask / block / escalate per finding
    # category, plus the governed `output_deny_patterns` and the canary and
    # prompt-leak check), `.relay()` for the conversation bound *to* a worker,
    # `.scrub()` for model-written governance text shown to the user, and the
    # `stream_*` half that `agent.StreamGuard` drives over live tokens.
    #
    # Defaults to a bare guard — the shipped category tiers, no extra policy
    # rules — so a hand-built `Services` in a test gets the production
    # behaviour without any configuration. A stand-in passed by a test must
    # provide all four; `screen` alone is not the interface any more.
    output_guard: object = field(default_factory=OutputGuard)
    # One subject's rolling governance-spend allowance (spend.py). Process-wide
    # rather than per-turn, so it lives here beside the other long-lived
    # collaborators. Defaults to a window with no caps configured — disabled,
    # which is what a hand-built `Services` in a test and an unconfigured
    # deployment both want.
    spend_window: object = field(default_factory=SubjectWindow)


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or Settings()
    # A deployed process with a safety-critical control disabled by environment
    # variable refuses to build rather than serving without it; dev logs each
    # finding at ERROR instead. See Settings.enforce.
    settings.enforce()

    # §4.1: model access is isolated in model_provider; no node or tool names a
    # model endpoint directly.
    from .model_provider import get_agent_model, get_routing_model

    llm = get_routing_model(settings)

    # Multi-model support: every per-agent governance call —
    # semantic screen, context resolution, worker simulation — resolves its
    # model through this one seam. Until MULTI_MODEL_ENABLED is true (and a
    # registry entry declares a `model`), it returns `llm` for every agent, so
    # wiring it unconditionally costs nothing and keeps the enable a pure
    # configuration change.
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

    # Governance configuration comes from the UC-registered table when one is
    # reachable, and from the bundled YAML when it is not — `config_store` owns
    # that choice, the validation and the fallback. Wrapped in `Reloading` so a
    # config published while this process is alive is picked up on the next
    # request rather than at the next deploy.
    config = ConfigProvider(settings)

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
        # The allowlist is derived from the registry rather than configured
        # separately: the only context worth persisting is context some agent
        # actually declared it needs, and deriving it means onboarding an agent
        # with a new required_context key does not also need a second edit
        # somewhere else to make that key storable (§04, memory poisoning).
        #
        # Passed as a callable for the same reason the registry is a proxy: the
        # registry can now change under a live process, and an allowlist frozen
        # here would refuse a context key added to the table a minute ago.
        memory=LongTermMemory(build_store(), allowed_keys=lambda: config.registry().context_keys()),
        # Same Postgres as the audit sink and the config table — an appeal is a
        # governance record, and putting it anywhere else would mean a reviewer
        # queries one store for the decision and another for the appeal against
        # it.
        reviews=build_review_queue(settings),
        # Built from the same governed guardrails document as the input engine,
        # through the same Reloading proxy, so publishing an output rule needs
        # no redeploy either.
        output_guard=Reloading(
            lambda: config.output_guard(mask_pii=settings.output_pii_masking)
        ),
        # Built once per process, not per turn: the window it keeps *is* the
        # cross-turn state. Not a `Reloading` proxy — rebuilding it on a config
        # publish would discard every counter and hand a subject who had just
        # exhausted their allowance a fresh one.
        spend_window=build_spend_window(settings),
    )
