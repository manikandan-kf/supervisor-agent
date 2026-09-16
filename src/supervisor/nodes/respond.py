"""Stage 5 — respond and audit.

The single exit. Writes the turn's audit record before the answer leaves,
synchronously for the outcomes that are themselves a governance decision.
"""

from __future__ import annotations

import logging

from agent_governance.resilience import deadline_for
from agent_governance.sensitive import redact_structure
from agent_governance.spend import (
    TurnSpend,
)
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..messages import (
    AUDIT_UNAVAILABLE_MESSAGE,
    GENERIC_ERROR,
    progress,
)
from ..state import SupervisorContext
from .turn import (
    NodeBase,
    _entry,
    _is_governance_decision,
    _narrates,
    _provenance,
    _session_age_seconds,
)

logger = logging.getLogger(__name__)


class RespondMixin(NodeBase):
    def respond(self, state: dict, runtime: Runtime[SupervisorContext]) -> dict:
        context = runtime.context or SupervisorContext()
        text = state.get("final_text") or GENERIC_ERROR
        routed = self.s.registry.get(state.get("target_agent_id") or "")

        deadline = deadline_for(context, self.s.settings)
        signoff = state.get("signoff") or {}

        # ── Redaction at the persistence boundary (Blueprint §06) ────────────
        # Query-derived trail text is scrubbed once here before both sinks (Postgres
        # and MLflow trace); the live conversation is untouched. Redactions are recorded.
        trail, redacted_count = redact_structure(state.get("audit_trail", []))
        if redacted_count:
            trail = list(trail) + [
                _entry(
                    "respond",
                    "redacted",
                    f"{redacted_count} secret/PII value(s) redacted from the decision "
                    "trail before persistence",
                )
            ]

        # ── What this turn spent, and what decided it ────────────────────────
        # Deliberately not trail entries: §05 Stage 06 defines the trail as the ordered
        # governance decisions, and readers select by stage. Metadata gets a column.
        spend = TurnSpend.from_state(state)
        if spend.model_calls or spend.tokens:
            # Charged once with the turn's final figures: charging mid-turn would let
            # a turn refuse itself halfway, spending the calls and delivering nothing.
            try:
                self.s.spend_window.charge(
                    context.user_key, model_calls=spend.model_calls, tokens=spend.tokens
                )
            except Exception:
                logger.warning("subject spend window charge failed", exc_info=True)

        record = {
            "request_id": state.get("request_id", ""),
            "conversation_id": state.get("conversation_id", ""),
            "correlation_id": context.correlation_id,
            "user_role": context.user_role,
            "user_key": context.user_key,
            "target_agent_id": state.get("target_agent_id", ""),
            "outcome": state.get("outcome", ""),
            "decision_trail": trail,
            # §05 Stage 06 lists **latency** among what the trail persists; MLflow
            # spans alone have shorter retention than the audit table.
            "latency_ms": round(deadline.spent() * 1000) if deadline.enabled else None,
            # §05 Stage 04's session bound needs a Duration KPI to be evidence
            # rather than a setting. This is that column.
            "session_age_seconds": _session_age_seconds(state),
            # Who signed off what, when — present only on an approval turn.
            "signoff": signoff or None,
            # v1.1 §08 token KPI as durable columns. `model_calls` is exact;
            # `tokens_estimated` is named for what it is (spend.py).
            "model_calls": spend.model_calls or None,
            "tokens_estimated": spend.tokens or None,
            # Which model, prompt versions and per-stage spend decided the turn.
            # JSONB like `signoff`, so the shape can grow without a migration.
            "provenance": _provenance(self.s.settings, spend) or None,
        }

        # ── Synchronous audit for governance decisions (§05 Stage 06, §08) ───
        # "Fail the operation rather than complete it unaudited" — applied to
        # decisions, not answers: a block whose record vanished cannot be proved to
        # have happened; an unaudited answer restricted nobody (Blueprint §01).
        governed = _is_governance_decision(state)
        try:
            self.s.audit.log(record)
        except Exception:
            logger.warning("audit write failed", exc_info=True)
            # The row itself at ERROR, same shape as `LoggingAuditLogger`, so a sink
            # hiccup degrades to "recover rows from the log", not under-counting BR-006.
            try:
                import json as _json

                logger.error("supervisor-audit-fallback %s", _json.dumps(record, default=str))
            except Exception:
                logger.error("supervisor-audit-fallback could not serialise the record")
            if governed:
                progress("respond", "error", "decision not recorded")
                return {
                    "messages": [AIMessage(content=AUDIT_UNAVAILABLE_MESSAGE)],
                    "outcome": "error",
                    "final_text": AUDIT_UNAVAILABLE_MESSAGE,
                    "routed_agent_name": "",
                }

        # A directly-answered turn shows no plan, so no closing tick; a refusal
        # always narrates, and "Finishing up" completes that plan.
        if _narrates(state) or state.get("outcome") != "answer":
            progress("respond", "done", state.get("outcome", ""))
        return {
            "messages": [AIMessage(content=text)],
            # Attribute to a worker only when one produced the answer; a denial,
            # block or clarification reaches here without any worker call.
            "routed_agent_name": (
                (routed.name if routed else "") if state.get("worker_response") else ""
            ),
        }
