"""What a stage does when the turn runs out of time, budget or patience.

Shared by guardrails, route and dispatch, which is why these are here and
not in any one stage's module. Every path routes to `respond` rather than
raising: the user gets a governed answer and the turn is still audited.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent_governance import review_queue
from agent_governance.resilience import BudgetExhausted
from agent_governance.review_queue import ReviewQueueError
from agent_governance.spend import (
    SpendExhausted,
    TurnSpend,
)
from langgraph.types import Command

from ..messages import (
    APPEAL_UNAVAILABLE_MESSAGE,
    BUDGET_EXHAUSTED_MESSAGE,
    CLARIFY_FINAL_PREFIX,
    CLARIFY_RETRY_PREFIX,
    ESCALATION_MESSAGE,
    SPEND_EXHAUSTED_MESSAGE,
    progress,
)
from ..state import SupervisorContext
from .turn import (
    NodeBase,
    _excerpt,
    _latest_user_text,
    _trail,
)

logger = logging.getLogger(__name__)


class LimitsMixin(NodeBase):
    def _budget_exhausted(self, state: dict, stage: str, exc: BudgetExhausted) -> Command:
        """The turn ran out of its total time budget (§05 Stage 05).

        Routed to `respond`, not raised, so the user gets a governed message and the trail a
        row. `outcome: "error"`, not a new value: the KPI report already treats it as controlled.
        """
        logger.warning("turn budget exhausted at %s: %s", stage, exc)
        progress(stage, "error", "time budget exhausted")
        return Command(
            goto="respond",
            update={
                "outcome": "error",
                "final_text": BUDGET_EXHAUSTED_MESSAGE,
                "audit_trail": _trail(
                    state,
                    stage,
                    "budget_exhausted",
                    f"turn budget of {exc.budget:.0f}s exhausted after {exc.spent:.1f}s "
                    f"before {exc.what or stage} — request held, nothing dispatched",
                ),
            },
        )

    def _spend_exhausted(
        self, state: dict, stage: str, exc: SpendExhausted, spend: TurnSpend
    ) -> Command:
        """The turn reached its governance spend ceiling (spend.py).

        Its own method and exception, not shared with `_budget_exhausted`: "ran long" and
        "spent its allowance" point an operator at different settings; the trail must say which.
        """
        logger.warning("turn spend ceiling reached at %s: %s", stage, exc)
        progress(stage, "error", "spend ceiling reached")
        return Command(
            goto="respond",
            update={
                "spend": spend.record(),
                "outcome": "error",
                "final_text": SPEND_EXHAUSTED_MESSAGE,
                "audit_trail": _trail(
                    state,
                    stage,
                    "spend_exhausted",
                    f"{exc.scope} ceiling of {exc.limit:g} {exc.unit} reached after "
                    f"{exc.spent:g} before {exc.what or stage} — request held, "
                    "nothing dispatched",
                ),
            },
        )

    def _clarify_or_escalate(
        self,
        state: dict,
        stage: str,
        question: str,
        update: dict,
        context: Optional[SupervisorContext] = None,
    ) -> Command:
        """Ask one clarifying question, or escalate once the limit is reached.

        The screen and the router share one counter deliberately: a limit each would let two
        stages take turns asking and never reach it.
        """
        count = state.get("clarification_count", 0) + 1
        if count > self.s.settings.max_clarifications:
            context = context or SupervisorContext()
            # The escalation has to become queryable state, or "escalate to a human" is a
            # message rather than a control (§05 Stage 04). Unrecorded, the user is not told.
            try:
                review = self.s.reviews.open_review(
                    kind=review_queue.ESCALATION,
                    conversation_id=state.get("conversation_id", ""),
                    reason=(
                        f"clarification limit reached at {stage} after "
                        f"{count - 1} unresolved attempts"
                    ),
                    user_key=context.user_key,
                    user_role=context.user_role,
                    target_agent_id=state.get("target_agent_id", ""),
                    request_id=state.get("request_id", ""),
                    correlation_id=context.correlation_id,
                    query_excerpt=_excerpt(_latest_user_text(state.get("messages"))),
                )
            except (ReviewQueueError, ValueError) as exc:
                logger.warning("could not open an escalation: %s", exc)
                progress(stage, "error", "escalation not recorded")
                return Command(
                    goto="respond",
                    update={
                        **update,
                        "outcome": "error",
                        "final_text": APPEAL_UNAVAILABLE_MESSAGE,
                        # The counter is *not* reset: the limit has not been honoured yet, and
                        # resetting would reward a storage failure with two more loops.
                        "audit_trail": _trail(
                            state,
                            stage,
                            "escalate_failed",
                            f"escalation could not be recorded ({type(exc).__name__}) — "
                            "not reported to the user as flagged",
                        ),
                    },
                )

            progress(stage, "error", "escalated")
            return Command(
                goto="respond",
                update={
                    **update,
                    "outcome": "escalated",
                    "final_text": ESCALATION_MESSAGE,
                    "clarification_count": 0,
                    "pending_clarification": None,
                    # Terminal until a reviewer resolves it — the next turn is refused by
                    # `_hold_for_review` rather than restarting the pipeline.
                    "open_review": {"ref": review.ref, "kind": review.kind},
                    "audit_trail": _trail(
                        state,
                        stage,
                        "escalate",
                        f"clarification limit reached after {count - 1} attempts — "
                        f"escalation {review.ref} opened in the review queue",
                    ),
                },
            )

        text = question
        if question and state.get("pending_clarification"):
            # A question was already open, so what the user just said did not settle it — this
            # is at least the second ask in a row, not a fresh one.
            final_attempt = count >= self.s.settings.max_clarifications
            text = (CLARIFY_FINAL_PREFIX if final_attempt else CLARIFY_RETRY_PREFIX) + question

        progress(stage, "clarify", question or "")
        return Command(
            goto="respond",
            update={
                **update,
                "outcome": "clarify",
                "final_text": text,
                "clarification_count": count,
                "pending_clarification": question,
                "audit_trail": _trail(state, stage, "clarify", question or ""),
            },
        )
