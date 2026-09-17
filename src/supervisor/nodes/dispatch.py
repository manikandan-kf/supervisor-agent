"""Stage 4 — dispatch.

Calls the worker under the supervisor's own identity, screens what comes
back, and hands a staged artifact to the approval gate.
"""

from __future__ import annotations

import logging
from typing import Literal

from agent_governance.retry_and_deadline import BudgetExhausted, deadline_for
from agent_governance.sanitize import (
    clean_worker_output,
)
from langgraph.runtime import Runtime
from langgraph.types import Command

from ..state import SupervisorContext
from ..user_facing_text import (
    CARRIED_CONTEXT_NOTICE,
    DEFERRED_REQUEST_OFFER,
    DEFERRED_REQUEST_REFUSED,
    GENERIC_ERROR,
    OUTPUT_ESCALATED_MESSAGE,
    OUTPUT_WITHHELD_MESSAGE,
    SENSITIVE_INPUT_NOTICE,
    WORKER_UNAVAILABLE_MESSAGE,
    context_prose,
    sensitive_prose,
)
from ..worker_client import WorkerUnavailable
from .base import (
    RelayScreen,
    _clean_sources,
    _entry,
    _excerpt,
    _latest_user_text,
    _trail,
    _window,
    _worker_messages,
)
from .failsafes import FailsafesMixin

logger = logging.getLogger(__name__)


class DispatchMixin(FailsafesMixin):
    def dispatch(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["respond", "approval"]]:
        context = runtime.context or SupervisorContext()
        deadline = deadline_for(context, self.s.settings)
        agent = self.s.registry.get(state["target_agent_id"])
        resolved = state.get("route", {}).get("resolved_context", {})

        # ── The relay boundary (guardrail layer 7, inbound half) ─────────────
        # What the worker receives is screened by the same policy as what it returns.
        relay = _worker_messages(
            _window(state.get("messages"), self.s.settings.worker_history_max_tokens),
            guard=self.s.output_guard,
        )
        relay_masked = list(relay.masked)

        # Resolved context is model-lifted from the conversation, so it is a second route to
        # the worker for a value the relay just masked; `memory._validate` only bounds storage.
        screened_context = {}
        for key, value in (resolved or {}).items():
            if isinstance(value, str):
                value, findings = self.s.output_guard.relay(value)
                relay_masked.extend(f.label for f in findings)
            screened_context[key] = value
        resolved = screened_context

        try:
            deadline.ensure(f"dispatch to {agent.id}")
            resp = self.s.workers.invoke(
                agent,
                relay.messages,
                resolved,
                state.get("conversation_id", ""),
                context.user_role,
                # The correlation set, so the worker's traces join this turn's (solution §08).
                # `user_key` is pseudonymous; no raw subject or token reaches a worker.
                {
                    "correlation_id": context.correlation_id,
                    "request_id": state.get("request_id", ""),
                    "agent_id": agent.id,
                    "environment": context.environment,
                    "pseudonymous_user_reference": context.user_key,
                },
                deadline=deadline,
            )
        except BudgetExhausted as exc:
            return self._budget_exhausted(state, "dispatch", exc)
        except WorkerUnavailable as exc:
            # The client already retried and the breaker has spoken — a clear "unavailable"
            # message, never another LLM-generated prompt.
            logger.warning("worker unavailable for %s: %s", agent.id, exc)
            return Command(
                goto="respond",
                update={
                    "outcome": "error",
                    "final_text": WORKER_UNAVAILABLE_MESSAGE,
                    "audit_trail": _trail(
                        state, "dispatch", "unavailable", f"worker unavailable: {exc}"
                    ),
                },
            )
        except Exception as exc:
            # Scrubbed like the governance paths — a worker-client exception can
            # carry the relayed conversation in its request body.
            logger.error(
                "worker dispatch failed for %s: %s: %s",
                agent.id,
                type(exc).__name__,
                str(exc)[:200],
            )
            logger.debug("worker dispatch traceback", exc_info=True)
            return Command(
                goto="respond",
                update={
                    "outcome": "error",
                    "final_text": GENERIC_ERROR,
                    "audit_trail": _trail(
                        state, "dispatch", "error", f"{type(exc).__name__}: {exc}"
                    ),
                },
            )

        # Sources are worker strings a UI renders, so screened like other output — and emitted
        # only after the guard: a withheld response must not have leaked its citation titles.
        sources = _clean_sources(resp.sources, self.s.output_guard)

        # ── Worker output is untrusted (solution §02: "the worker's response is validated") ──
        # Bounded and sanitized *before* the text reaches state: as an AIMessage it becomes
        # history replayed into governance prompts, where a fake turn boundary would pass as real.
        cleaned = clean_worker_output(
            resp.text or "", max_chars=self.s.settings.worker_output_max_chars
        )
        text = cleaned.text
        output_trail = []
        if relay.modified or relay_masked:
            # On the dispatch stage because it describes what the worker was given: "did the
            # worker ever see the pasted token?" needs an answer in the trail.
            output_trail.append(
                _entry(
                    "dispatch",
                    "relay_screened",
                    RelayScreen(
                        relay.messages, tuple(relay_masked), relay.directives
                    ).audit_detail(),
                )
            )
            if relay.directives:
                logger.warning(
                    "conversation relayed to %s carried %d embedded directive marker(s) — "
                    "neutralised before dispatch",
                    agent.id,
                    relay.directives,
                )
        if cleaned.modified or cleaned.code_blocks:
            output_trail.append(_entry("dispatch", "output_sanitized", cleaned.audit_detail()))
        if cleaned.role_markers or cleaned.template_markers:
            # Something in the text was shaped like a turn boundary — the coercion channel
            # NIST SP 800-207 §5.7 describes, hence warning level.
            logger.warning(
                "worker %s returned %d role marker(s) and %d template marker(s) — "
                "neutralised before entering conversation history",
                agent.id,
                cleaned.role_markers,
                cleaned.template_markers,
            )

        # ── Output guard (guardrail layer 7) ─────────────────────────────────
        # The last check before worker text becomes user-visible state. Before the approval
        # branch on purpose: a staged artifact is screened exactly like a delivered answer.
        guarded = self.s.output_guard.screen(text)
        # getattr: a stub predating the escalate tier must degrade to "block", never "deliver".
        if getattr(guarded, "escalate", False):
            # Withheld, and recorded as an escalation rather than a block: a leak or a data
            # dump is a finding a reviewer must see apart from an ordinary policy match.
            logger.warning(
                "worker %s response escalated by the output guard: %s", agent.id, guarded.reason
            )
            return Command(
                goto="respond",
                update={
                    "pending_approval": None,
                    "worker_response": {"status": resp.status, "stage": resp.stage},
                    "outcome": "escalated",
                    "final_text": OUTPUT_ESCALATED_MESSAGE,
                    "audit_trail": _trail(
                        state,
                        "output_guard",
                        "escalated",
                        f"{guarded.audit_detail()} — recorded for a human reviewer",
                    )
                    + output_trail,
                },
            )
        if guarded.blocked:
            logger.warning(
                "worker %s response withheld by the output guard: %s",
                agent.id,
                guarded.reason,
            )
            return Command(
                goto="respond",
                update={
                    "pending_approval": None,
                    "worker_response": {"status": resp.status, "stage": resp.stage},
                    "outcome": "blocked",
                    # Final, like an input-tier block: the event is in the trail for review,
                    # and there is no appeal path from the conversation.
                    "final_text": OUTPUT_WITHHELD_MESSAGE,
                    "audit_trail": _trail(state, "output_guard", "withheld", guarded.audit_detail())
                    + output_trail,
                },
            )
        text = guarded.text
        if guarded.modified:
            output_trail.append(_entry("output_guard", "masked", guarded.audit_detail()))

        # ── Supervisor-enforced approval (Solution §04) ──────────────────────
        # A registry `approval_patterns` match stages the response on the supervisor's own
        # authority, so the gate does not rest on worker cooperation. Evaluated on the user's
        # request — what was asked for is what makes it reviewable — and after the output guard.
        mandated = self.s.registry.approval_reason(agent, _latest_user_text(state.get("messages")))

        if resp.status == "approval_pending" or mandated:
            # The pause lives in its own node: everything before an `interrupt()` re-runs on
            # resume, so one node would call the worker twice per approval. `stage` falls back
            # to a generic label; inventing "HLD" would put a claim into a record a human signs.
            stage = resp.stage or ("review" if mandated else None)
            if mandated and resp.status != "approval_pending":
                logger.info(
                    "staging %s response for sign-off: %s (worker did not request a gate)",
                    agent.id,
                    mandated,
                )
            trail_detail = f"stage: {stage}"
            if mandated:
                trail_detail = (
                    f"stage: {stage} — required by the supervisor "
                    f"({agent.risk_level} risk): {mandated}"
                    + ("" if resp.status == "approval_pending" else "; worker did not request one")
                )
            return Command(
                goto="approval",
                update={
                    "pending_approval": {
                        "agent_id": agent.id,
                        "agent_name": agent.name,
                        "stage": stage,
                        "artifact": text,
                        "context": resolved,
                        # Shown to the approver so the reason for the gate travels with the
                        # artifact rather than living only in the audit row.
                        "required_because": mandated,
                        "risk_level": agent.risk_level,
                    },
                    "worker_response": {"status": "approval_pending", "stage": stage},
                    "sources": sources,
                    "audit_trail": _trail(state, "dispatch", "approval_pending", trail_detail)
                    + output_trail,
                },
            )

        if not text:
            return Command(
                goto="respond",
                update={
                    "pending_approval": None,
                    "outcome": "error",
                    "final_text": GENERIC_ERROR,
                    "audit_trail": _trail(
                        state, "dispatch", "invalid_response", "worker returned an empty response"
                    ),
                },
            )

        # ── Tell the user what was masked on the way in ──────────────────────
        # After the approval branch, deliberately: a message *to the user about their own
        # message*, not part of the artifact a human signs off.
        if relay_masked:
            text = (
                text.rstrip()
                + "\n\n"
                + SENSITIVE_INPUT_NOTICE.format(what=sensitive_prose(relay_masked))
            )

        # Context this conversation never stated, carried in from another one.
        # After the approval branch for the same reason as the notice above.
        carried_over = (state.get("route") or {}).get("carried_over") or {}
        if carried_over:
            text = text.rstrip() + CARRIED_CONTEXT_NOTICE.format(what=context_prose(carried_over))

        # ── The second task in the same message ──────────────────────────────
        # The first point at which the first task has produced something. After the approval
        # branch: a second request is dropped rather than held across a human review.
        deferred_update: dict = {}
        additional = ((state.get("guardrail") or {}).get("additional_request") or "").strip()
        if additional:
            # Pre-screened deterministically: offering what the tier-1 rules refuse, then refusing
            # on "yes", is worse than a straight answer now. The full screen and RBAC still run.
            blocked = ""
            try:
                blocked = self.s.guardrails.deterministic_block(additional)
            except Exception:  # noqa: BLE001 — an offer is never worth a failed turn
                logger.warning("could not pre-screen a held request", exc_info=True)
                blocked = "it could not be screened"
            if blocked:
                text = text.rstrip() + DEFERRED_REQUEST_REFUSED.format(
                    what=_excerpt(additional), reason=blocked
                )
                output_trail.append(
                    _entry(
                        "guardrails",
                        "second_request_refused",
                        f"held request not offered — {blocked}: {_excerpt(additional)}",
                    )
                )
            else:
                text = text.rstrip() + DEFERRED_REQUEST_OFFER.format(what=_excerpt(additional))
                deferred_update = {
                    "deferred_request": {
                        "text": additional,
                        "asked_at": state.get("request_id", ""),
                    }
                }
                output_trail.append(
                    _entry(
                        "guardrails",
                        "second_request_offered",
                        f"held for the user's confirmation: {_excerpt(additional)}",
                    )
                )

        return Command(
            goto="respond",
            update={
                "pending_approval": None,
                **deferred_update,
                "worker_response": {"status": resp.status, "stage": resp.stage},
                "outcome": "answer",
                "final_text": text,
                "sources": sources,
                "audit_trail": _trail(
                    state, "dispatch", "completed", f"{len(text)} chars from {agent.id}"
                )
                + output_trail,
            },
        )
