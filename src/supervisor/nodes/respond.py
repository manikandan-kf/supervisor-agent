"""Stage 5 — respond and audit.

The single exit. Writes the turn's audit record before the answer leaves,
synchronously for the outcomes that are themselves a governance decision.
"""

from __future__ import annotations

import logging

from agent_governance.retry_and_deadline import deadline_for
from agent_governance.sensitive_data import redact_structure
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from ..state import SupervisorContext
from ..user_facing_text import (
    AUDIT_UNAVAILABLE_MESSAGE,
    GENERIC_ERROR,
)
from .base import (
    NodeBase,
    _entry,
    _is_governance_decision,
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

        # ── Redaction at the persistence boundary ────────────────────────────
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

        # ── What decided this turn ───────────────────────────────────────────
        # Deliberately not trail entries: the trail is the ordered list of governance
        # decisions, and readers select by stage. Metadata gets a column.
        record = {
            "request_id": state.get("request_id", ""),
            "conversation_id": state.get("conversation_id", ""),
            "correlation_id": context.correlation_id,
            "user_role": context.user_role,
            "user_key": context.user_key,
            "target_agent_id": state.get("target_agent_id", ""),
            "outcome": state.get("outcome", ""),
            "decision_trail": trail,
            # Latency (solution §07 KPI) is persisted here as well as on the trace; MLflow
            # spans alone have shorter retention than the audit table.
            "latency_ms": round(deadline.spent() * 1000) if deadline.enabled else None,
            # The session-duration KPI (solution §07) needs a column, not a setting.
            "session_age_seconds": _session_age_seconds(state),
            # Who signed off what, when — present only on an approval turn.
            "signoff": signoff or None,
            # Which model and prompt versions decided the turn. JSONB like `signoff`, so the
            # shape can grow without a migration. Token usage is on the MLflow trace (§07 KPI).
            "provenance": _provenance(self.s.settings) or None,
        }

        # ── Synchronous audit for governance decisions (solution §02) ────────
        # Fail the operation rather than complete it unaudited — applied to decisions,
        # not answers: a block whose record vanished cannot be proved to have happened;
        # an unaudited answer restricted nobody.
        governed = _is_governance_decision(state)
        try:
            self.s.audit.log(record)
        except Exception:
            logger.warning("audit write failed", exc_info=True)
            # The row itself at ERROR, same shape as `LoggingAuditLogger`, so a sink
            # hiccup degrades to "recover rows from the log", not an under-counted trail.
            try:
                import json as _json

                logger.error("supervisor-audit-fallback %s", _json.dumps(record, default=str))
            except Exception:
                logger.error("supervisor-audit-fallback could not serialise the record")
            if governed:
                return {
                    "messages": [AIMessage(content=AUDIT_UNAVAILABLE_MESSAGE)],
                    "outcome": "error",
                    "final_text": AUDIT_UNAVAILABLE_MESSAGE,
                    "routed_agent_name": "",
                }

        return {
            "messages": [AIMessage(content=text)],
            # Attribute to a worker only when one produced the answer; a denial,
            # block or clarification reaches here without any worker call.
            "routed_agent_name": (
                (routed.name if routed else "") if state.get("worker_response") else ""
            ),
        }
