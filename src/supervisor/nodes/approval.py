"""Stage 4b — human approval.

Suspends on `interrupt()` in a node of its own, so resuming re-runs the
pause rather than the worker call that produced the artifact.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from ..state import SupervisorContext
from .base import (
    NodeBase,
    _now_iso,
    _seconds_since,
    _trail,
)

logger = logging.getLogger(__name__)


class ApprovalMixin(NodeBase):
    def _may_approve(self, context: SupervisorContext, agent_id: str) -> bool:
        """May this caller sign off work produced by `agent_id`?

        None and () differ: () means this caller may approve nothing; None means the field was
        not supplied — only possible off the deployed path — so it defers to the access decision.
        """
        if not agent_id:
            return False
        if context.approvable_agents is not None:
            return agent_id in context.approvable_agents

        logger.warning(
            "no approvable_agents supplied — approving '%s' on the access decision alone. "
            "The calling application always supplies this field; seeing it here means the "
            "graph was invoked some other way.",
            agent_id,
        )
        permitted = context.permitted_agents
        if permitted is not None:
            return agent_id in permitted
        return self.s.rbac.check((context.user_role or "").strip(), agent_id).allowed

    def approval(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["respond"]]:
        """Suspend until a human accepts or rejects the staged artifact.

        Side-effect free before the `interrupt()`: LangGraph re-runs a node from the top on
        resume, which is why the worker call lives in `dispatch` and this node only reads state.
        """
        context = runtime.context or SupervisorContext()
        pending = state.get("pending_approval") or {}
        stage = pending.get("stage") or "staged"
        owner_id = pending.get("agent_id", "")
        owner_name = pending.get("agent_name", "") or owner_id

        # A decision the RBAC gate read out of the conversation settles the gate without
        # suspending: `interrupt()` here would re-open the gate the reviewer just answered.
        decision = state.get("approval_reply") or interrupt(
            {
                "kind": "worker_approval",
                "agent_id": owner_id,
                "agent_name": pending.get("agent_name", ""),
                "stage": pending.get("stage"),
                "artifact": pending.get("artifact", ""),
                # Why the gate opened when the supervisor required it (empty for a worker stage).
                # ASI09: the approver sees the real reason, not a summary the producer chose.
                "required_because": pending.get("required_because", ""),
                "risk_level": pending.get("risk_level", ""),
            }
        )

        # Both spellings are accepted: callers send "approved", people type "approve".
        # Anything else — including a malformed payload — is a rejection, which widens nothing.
        stated = (
            str(decision.get("decision", "")).strip().lower() if isinstance(decision, dict) else ""
        )
        approved = decision is True or stated in ("approved", "approve", "accept", "accepted")
        comment = decision.get("comment", "") if isinstance(decision, dict) else ""

        # ── Who approved what, and when ───────────────────────────────────────
        # Solution §02: a staged artifact waits for explicit human approval, and the trail
        # records who gave it. `user_key` is the strongest attributable identity here; the raw
        # subject never reaches the supervisor.
        approver = context.user_key or "unattributed"
        decided_at = _now_iso()
        # Solution §07 KPI, "HITL approval turnaround": how long the artifact waited at the
        # gate. Measured from `staged_at`, stamped by dispatch — a reviewer may answer days
        # later, so the turn's own latency says nothing about it.
        staged_at = pending.get("staged_at", "")
        waited = _seconds_since(staged_at, "pending_approval.staged_at")
        waited_note = f" after {waited:.0f}s at the gate" if waited is not None else ""
        signoff = {
            "stage": stage,
            "agent_id": owner_id,
            "approver": approver,
            "approver_role": context.user_role,
            "decided_at": decided_at,
            "staged_at": staged_at,
            "waited_seconds": round(waited, 3) if waited is not None else None,
            "comment": comment[:2000],
        }
        if approved and approver == "unattributed":
            # Refused, not merely logged: a compensating control that records "unattributed"
            # does not compensate; the artifact stays staged for an attributable caller.
            # A rejection below still goes through: declining to act widens nothing.
            logger.warning(
                "approval decision for '%s' stage '%s' carries no approver identity — "
                "refused. The calling application always supplies one, so the graph was "
                "invoked some other way.",
                owner_id,
                stage,
            )
            return Command(
                goto="respond",
                update={
                    # The gate stays open: nothing was decided.
                    "approval_reply": None,
                    "outcome": "blocked",
                    "final_text": (
                        f"I can't record who is approving this, so the {stage} draft "
                        "has been left unapproved. Please approve it from the application, "
                        "where your identity travels with the decision."
                    ),
                    "audit_trail": _trail(
                        state,
                        "approval",
                        "deny",
                        f"approval of '{owner_id}' stage {stage} refused: no attributable "
                        f"approver identity was supplied at {decided_at}",
                    ),
                },
            )

        # An approval is bound to the agent that produced the artifact (from the checkpoint),
        # not the id the request was addressed with. Rejecting needs no permission.
        if approved and not self._may_approve(context, owner_id):
            return Command(
                goto="respond",
                update={
                    "pending_approval": None,
                    "approval_reply": None,
                    "outcome": "blocked",
                    "final_text": (
                        f"You don't have permission to approve the {owner_name}'s work. "
                        f"The {stage} draft has been left unapproved — ask someone with "
                        f"sign-off rights for this agent to review it."
                    ),
                    "audit_trail": _trail(
                        state,
                        "approval",
                        "deny",
                        f"caller lacks approve rights for '{owner_id}' (stage {stage}); "
                        f"attempted by {approver} at {decided_at}",
                    ),
                },
            )

        if not approved:
            return Command(
                goto="respond",
                update={
                    "pending_approval": None,
                    "approval_reply": None,
                    # Relayed to the worker on the next dispatch: its own workflow gated this
                    # stage, and a rejection is what tells it to rework rather than continue.
                    "last_signoff": {**signoff, "decision": "rejected"},
                    # Stays "answer" on purpose: a rejection is audited via `signoff`, not a new
                    # outcome value — the routing-completion KPI is computed from
                    # `outcome == "answer"`, and a signed-off artifact *is* a completed routing.
                    "outcome": "answer",
                    "signoff": {**signoff, "decision": "rejected"},
                    "final_text": (
                        f"Understood — I've discarded the {stage} draft. "
                        + (f"Note: {comment}" if comment else "Tell me what to change.")
                    ),
                    "audit_trail": _trail(
                        state,
                        "approval",
                        "rejected",
                        f"stage {stage} of '{owner_id}' rejected by {approver} "
                        f"(role {context.user_role or 'unknown'}) at {decided_at}{waited_note}"
                        + (f"; note: {comment}" if comment else ""),
                    ),
                },
            )

        return Command(
            goto="respond",
            update={
                "pending_approval": None,
                "approval_reply": None,
                # Relayed to the worker on the next dispatch, so a workflow that gates
                # HLD → LLD → Epic can continue from the stage a human just signed off.
                "last_signoff": {**signoff, "decision": "approved"},
                "worker_response": {"status": "approved", "stage": pending.get("stage")},
                "outcome": "answer",
                "signoff": {**signoff, "decision": "approved"},
                "final_text": pending.get("artifact", ""),
                "audit_trail": _trail(
                    state,
                    "approval",
                    "approved",
                    f"stage {stage} of '{owner_id}' approved by {approver} "
                    f"(role {context.user_role or 'unknown'}) at {decided_at}{waited_note}"
                    + (f"; note: {comment}" if comment else ""),
                ),
            },
        )
