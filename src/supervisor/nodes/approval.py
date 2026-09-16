"""Stage 4b — human approval.

Suspends on `interrupt()` in a node of its own, so resuming re-runs the
pause rather than the worker call that produced the artifact.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from ..messages import progress
from ..state import SupervisorContext
from .turn import (
    NodeBase,
    _now_iso,
    _trail,
)

logger = logging.getLogger(__name__)


class ApprovalMixin(NodeBase):
    def _may_approve(self, context: SupervisorContext, agent_id: str) -> bool:
        """May this caller sign off work produced by `agent_id`? (§2.11)

        None and () differ: () means this caller may approve nothing; None means the field was
        not supplied — only possible off the deployed path — so it defers to the access decision.
        """
        if not agent_id:
            return False
        if context.approvable_agents is not None:
            return agent_id in context.approvable_agents

        logger.warning(
            "no approvable_agents supplied — approving '%s' on the access decision alone. "
            "The Platform API always supplies this field; seeing it here means the graph "
            "was invoked outside the gateway.",
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

        decision = interrupt(
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

        approved = decision is True or (
            isinstance(decision, dict) and decision.get("decision") == "approved"
        )
        comment = decision.get("comment", "") if isinstance(decision, dict) else ""

        # ── Who approved what, and when (§05 Stage 04 callout) ───────────────
        # §05 Stage 04: "record who approved what and when". §09 allows self-approval only
        # *"with a recorded approver identity"* — the trail is the compensating control.
        # `user_key` is the strongest attributable identity here; §1.10 keeps the raw subject out.
        approver = context.user_key or "unattributed"
        decided_at = _now_iso()
        signoff = {
            "stage": stage,
            "agent_id": owner_id,
            "approver": approver,
            "approver_role": context.user_role,
            "decided_at": decided_at,
            "comment": comment[:2000],
        }
        if approved and approver == "unattributed":
            # Refused, not merely logged: a compensating control that records "unattributed"
            # does not compensate (§09); the artifact stays staged for an attributable caller.
            # A rejection below still goes through: declining to act widens nothing.
            logger.warning(
                "approval decision for '%s' stage '%s' carries no approver identity — "
                "refused. The Platform API always supplies one, so the graph was "
                "invoked outside the gateway.",
                owner_id,
                stage,
            )
            progress("dispatch", "blocked", "approval not attributable")
            return Command(
                goto="respond",
                update={
                    # The gate stays open: nothing was decided.
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
            progress("dispatch", "blocked", "approval not permitted")
            return Command(
                goto="respond",
                update={
                    "pending_approval": None,
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
            progress("dispatch", "done", "rejected")
            return Command(
                goto="respond",
                update={
                    "pending_approval": None,
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
                        f"(role {context.user_role or 'unknown'}) at {decided_at}"
                        + (f"; note: {comment}" if comment else ""),
                    ),
                },
            )

        progress("dispatch", "done", "approved")
        return Command(
            goto="respond",
            update={
                "pending_approval": None,
                "worker_response": {"status": "approved", "stage": pending.get("stage")},
                "outcome": "answer",
                "signoff": {**signoff, "decision": "approved"},
                "final_text": pending.get("artifact", ""),
                "audit_trail": _trail(
                    state,
                    "approval",
                    "approved",
                    f"stage {stage} of '{owner_id}' approved by {approver} "
                    f"(role {context.user_role or 'unknown'}) at {decided_at}"
                    + (f"; note: {comment}" if comment else ""),
                ),
            },
        )
