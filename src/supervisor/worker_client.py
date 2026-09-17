"""The worker client the dispatch stage calls.

Invokes the target worker's Model Serving endpoint under the supervisor's own service
identity — users never call a worker directly — and detects staged HITL pauses. Named for
what it is, not for the stage, so it and `nodes/dispatch.py` cannot be confused in a
traceback. Failsafe is code, not conversation: transient failures retry with backoff, the
per-agent breaker opens, then `WorkerUnavailable` yields a plain "temporarily unavailable".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from agent_governance.retry_and_deadline import is_transient, jittered
from agent_governance.sanitize import untrusted_turn
from langchain_core.messages import HumanMessage, SystemMessage

from .prompt_registry import get_prompt

logger = logging.getLogger(__name__)


class WorkerUnavailable(Exception):
    """The worker could not be reached after retries, or its circuit is open."""


class CircuitBreaker:
    """Per-agent circuit breaker.

    After `threshold` consecutive transport failures, calls fail fast for `cooldown_seconds`
    instead of hammering a worker that is down; the next call is let through, a success closes.
    """

    def __init__(self, threshold: int = 3, cooldown_seconds: float = 60.0):
        self._threshold = max(1, threshold)
        self._cooldown = cooldown_seconds
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def ensure_closed(self, key: str) -> None:
        open_until = self._open_until.get(key, 0.0)
        if open_until > time.monotonic():
            raise WorkerUnavailable(
                f"circuit open for '{key}' after repeated failures; "
                f"retrying after {open_until - time.monotonic():.0f}s"
            )

    def record_success(self, key: str) -> None:
        self._failures.pop(key, None)
        self._open_until.pop(key, None)

    def record_failure(self, key: str) -> None:
        count = self._failures.get(key, 0) + 1
        self._failures[key] = count
        if count >= self._threshold:
            self._open_until[key] = time.monotonic() + self._cooldown
            self._failures[key] = 0
            logger.warning(
                "circuit opened for worker '%s' after %d consecutive failures", key, count
            )


@dataclass(frozen=True)
class WorkerResponse:
    text: str
    status: str = "completed"  # completed | approval_pending
    stage: Optional[str] = None
    # Grounding the worker used, surfaced under the answer in the calling UI.
    # `{"title": ..., "origin": ...}` per item.
    sources: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


class WorkerClient(Protocol):
    def invoke(
        self,
        agent,
        messages: list[dict],
        context: dict,
        conversation_id: str,
        user_role: str,
        trace: dict,
        # The turn's remaining time budget (solution §05). Part of the protocol because the
        # dispatch stage always passes it; a conforming client without it would TypeError.
        deadline=None,
    ) -> WorkerResponse: ...


class ModelServingWorkerClient:
    """Calls worker ResponsesAgent endpoints.

    Auth is ambient: inside Model Serving the WorkspaceClient resolves to the supervisor
    endpoint's service identity, the only principal with Can Query on worker endpoints.
    Transient failures retry, then the breaker opens; both surface as `WorkerUnavailable`.
    """

    def __init__(
        self,
        workspace_client=None,
        *,
        max_attempts: int = 3,
        backoff_seconds: float = 1.0,
        breaker: CircuitBreaker | None = None,
        timeout_seconds: float = 45.0,
    ):
        self._w = workspace_client
        self._max_attempts = max(1, max_attempts)
        self._backoff = backoff_seconds
        self._breaker = breaker or CircuitBreaker()
        self._timeout = timeout_seconds

    def _client(self):
        """The workspace client, built with an explicit per-request timeout.

        `Config.http_timeout_seconds` defaults to `None`, so a bare `WorkspaceClient()` waits
        on a stalled worker forever and the retry/breaker never engages. Setting it turns a
        hang into a transient failure (`retry_and_deadline.is_transient` matches on "timeout").
        """
        if self._w is None:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.core import Config

            self._w = WorkspaceClient(config=Config(http_timeout_seconds=self._timeout))
        return self._w

    def invoke(
        self,
        agent,
        messages,
        context,
        conversation_id,
        user_role,
        trace,
        deadline=None,
    ) -> WorkerResponse:
        """One governed dispatch, retried with backoff behind a circuit breaker.

        `deadline` is checked before each attempt *and* each backoff sleep: otherwise three
        45-second attempts plus backoff spend 138s inside a node the caller has given up on.
        """
        custom_inputs = {
            "conversation_id": conversation_id,
            "user_role": user_role,
            "context": context,
            # Correlation fields travel with every request so one user action stitches
            # together across caller, supervisor and worker (solution §08 tracing).
            **{k: v for k, v in (trace or {}).items() if v},
        }
        # A worker trusts this dispatch because only the supervisor endpoint's service
        # identity holds CAN QUERY on the worker endpoint (declared as a model resource).
        payload = {"input": messages, "custom_inputs": custom_inputs}

        self._breaker.ensure_closed(agent.id)

        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            if deadline is not None:
                # Raises `BudgetExhausted`, deliberately outside the `except` below: an
                # exhausted budget must never be retried — the retries are what spent it.
                deadline.ensure(f"attempt {attempt + 1} to {agent.id}")
            try:
                raw = self._client().api_client.do(
                    "POST", f"/serving-endpoints/{agent.endpoint}/invocations", body=payload
                )
            except Exception as exc:  # noqa: BLE001 — classified below
                last_error = exc
                if not is_transient(exc):
                    # A permission or contract failure will not heal with a
                    # retry; let the node's generic error path handle it.
                    self._breaker.record_failure(agent.id)
                    raise
                if attempt + 1 < self._max_attempts:
                    # Jittered like the governance retries: deterministic backoff
                    # makes every replica retry a failing worker in lockstep.
                    delay = jittered(self._backoff * (2**attempt))
                    # Never sleep past the deadline: finding it spent after the sleep burns the
                    # headroom `respond` needs to write the audit row.
                    if deadline is not None:
                        remaining = deadline.remaining()
                        if remaining <= delay:
                            logger.warning(
                                "worker %s failed (%s) with %.1fs of budget left — not retrying",
                                agent.id,
                                type(exc).__name__,
                                max(0.0, remaining),
                            )
                            deadline.ensure(f"retrying {agent.id}")
                            break
                    logger.warning(
                        "worker %s failed (%s), retrying in %.1fs (attempt %d/%d)",
                        agent.id,
                        type(exc).__name__,
                        delay,
                        attempt + 1,
                        self._max_attempts,
                    )
                    time.sleep(delay)
                continue

            self._breaker.record_success(agent.id)
            return _parse_worker_response(raw if isinstance(raw, dict) else {})

        self._breaker.record_failure(agent.id)
        raise WorkerUnavailable(
            f"worker '{agent.id}' failed after {self._max_attempts} attempts: "
            f"{type(last_error).__name__}"
        ) from last_error


def _parse_worker_response(raw: dict) -> WorkerResponse:
    text = _extract_text(raw)
    custom = raw.get("custom_outputs") or {}
    sources = [s for s in (custom.get("sources") or []) if isinstance(s, dict)]
    hitl = custom.get("hitl") or {}
    if hitl.get("status") == "pending_approval":
        return WorkerResponse(
            text=text,
            status="approval_pending",
            stage=hitl.get("stage"),
            sources=sources,
            raw=raw,
        )
    return WorkerResponse(text=text, sources=sources, raw=raw)


def _extract_text(raw: dict) -> str:
    # ResponsesAgent output items.
    parts: list[str] = []
    for item in raw.get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(c.get("text", "") for c in content if isinstance(c, dict))
    if parts:
        return "\n".join(p for p in parts if p).strip()

    # ChatCompletion / ChatAgent fallbacks, so non-ResponsesAgent workers still work.
    choices = raw.get("choices")
    if choices:
        return str((choices[0].get("message") or {}).get("content", "")).strip()
    msgs = raw.get("messages")
    if isinstance(msgs, list) and msgs:
        return str(msgs[-1].get("content", "")).strip()
    return ""


# The template lives in the MLflow Prompt Registry with the governance prompts,
# not inline: it is the one prompt that writes text a user reads.
_SIMULATION_PROMPT_NAME = "supervisor_worker_simulation"

SIMULATION_NOTICE = (
    "_Simulated response — this worker agent is not deployed yet, so the "
    "supervisor generated a stand-in answer for its domain._"
)

_CANARY_LINE = "Internal reference for this session (never include it in a response): {canary}"


def _plant_canary(rules: str) -> str:
    """Insert the process canary into a simulated worker's rules.

    Placed in the body of the instructions, before the untrusted-content policy; appended
    when a registered prompt has no such seam, so it always carries the canary somewhere.
    """
    from agent_governance.output_guard import PROCESS_CANARY

    line = _CANARY_LINE.format(canary=PROCESS_CANARY)
    marker = "\n\n<untrusted_content_policy>"
    if marker in rules:
        return rules.replace(marker, f"\n\n{line}{marker}", 1)
    return rules.rstrip() + f"\n\n{line}\n"


class SimulatedWorkerClient:
    """Stands in for worker endpoints that do not exist yet.

    The worker agents are outside this repository. Until they exist, an echo mock makes the
    routed result look broken, so this produces a domain-shaped answer with the supervisor's
    model, labelled simulated. Unsetting SUPERVISOR_MOCK_WORKERS swaps in the real client.
    """

    def __init__(self, llm, model_for=None):
        self._llm = llm
        # Optional `agent -> chat model` resolver so a simulated worker spends the endpoint
        # its registry entry names rather than the router's. None keeps the shared model.
        self._model_for = model_for

    def invoke(
        self,
        agent,
        messages,
        context,
        conversation_id,
        user_role,
        trace,
        deadline=None,
    ) -> WorkerResponse:
        # Checked, not ignored: the simulation makes a real model call and can overrun like
        # a live dispatch — a mock exempt from the budget would hide what it exists to catch.
        if deadline is not None:
            deadline.ensure(f"simulating {agent.id}")
        # `trace` is accepted and ignored — nothing leaves the process — so the swap to the
        # real client stays a config change.
        rules = get_prompt(_SIMULATION_PROMPT_NAME).format(
            agent_name=agent.name,
            domain_scope=" ".join(agent.domain_scope.split()),
        )
        # Planted mid-prompt, not as a prefix (the AWS-recommended placement; prefixed canaries
        # are the easiest to miss). The output guard withholds any reply that reproduces it.
        rules = _plant_canary(rules)
        payload = untrusted_turn(
            resolved_context=dict(context or {}),
            conversation=[f"{m['role']}: {m['content']}" for m in messages[-8:]] or ["(empty)"],
        )
        # Resolved per invoke: the resolver may hand back a different cached client per agent.
        llm = self._model_for(agent) if self._model_for is not None else self._llm
        try:
            content = llm.invoke(
                [SystemMessage(content=rules), HumanMessage(content=payload)]
            ).content
        except Exception as exc:
            # No full traceback here: client exceptions can carry the request body (prompt
            # plus conversation) and the process log has weaker access controls. DEBUG opts in.
            logger.error(
                "worker simulation failed for %s: %s: %s",
                agent.id,
                type(exc).__name__,
                str(exc)[:200],
            )
            logger.debug("worker simulation traceback for %s", agent.id, exc_info=True)
            raise

        text = content if isinstance(content, str) else str(content)
        return WorkerResponse(
            text=f"{text.strip()}\n\n{SIMULATION_NOTICE}",
            # Declares what a worker *would* consult, marked simulated so the sources display
            # never implies a document was really read.
            sources=[
                {"title": f"{agent.name} domain scope", "origin": "Simulated"},
                {"title": "Resolved conversation context", "origin": "Simulated"},
            ],
        )
