"""What a stage does when the turn runs out of time, budget or patience.

Shared by guardrails, route and dispatch, which is why these are here and
not in any one stage's module. Every path routes to `respond` rather than
raising: the user gets a governed answer and the turn is still audited.
"""

from __future__ import annotations

import logging

from agent_governance.retry_and_deadline import BudgetExhausted
from langgraph.types import Command

from ..user_facing_text import (
    BUDGET_EXHAUSTED_MESSAGE,
    CLARIFY_FINAL_PREFIX,
    CLARIFY_RETRY_PREFIX,
    ESCALATION_MESSAGE,
)
from .base import (
    NodeBase,
    _trail,
)

logger = logging.getLogger(__name__)


class FailsafesMixin(NodeBase):
    def _budget_exhausted(self, state: dict, stage: str, exc: BudgetExhausted) -> Command:
        """The turn ran out of its total time budget (solution §05).

        Routed to `respond`, not raised, so the user gets a governed message and the trail a
        row. `outcome: "error"`, not a new value: the KPI report already treats it as controlled.
        """
        logger.warning("turn budget exhausted at %s: %s", stage, exc)
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

    def _clarify_or_escalate(
        self,
        state: dict,
        stage: str,
        question: str,
        update: dict,
    ) -> Command:
        """Ask one clarifying question, or escalate once the limit is reached.

        The screen and the router share one counter deliberately: a limit each would let two
        stages take turns asking and never reach it.
        """
        count = state.get("clarification_count", 0) + 1
        if count > self.s.settings.max_clarifications:
            # §05: "escalating to a human after two loops". The escalation is the `escalated`
            # outcome in the audit trail, written before the user is told;
            # a reviewer works from that trail. The conversation is not held.
            return Command(
                goto="respond",
                update={
                    **update,
                    "outcome": "escalated",
                    "final_text": ESCALATION_MESSAGE,
                    "clarification_count": 0,
                    "pending_clarification": None,
                    "audit_trail": _trail(
                        state,
                        stage,
                        "escalate",
                        f"clarification limit reached after {count - 1} attempts — "
                        "recorded for a human reviewer",
                    ),
                },
            )

        text = question
        if question and state.get("pending_clarification"):
            # A question was already open, so what the user just said did not settle it — this
            # is at least the second ask in a row, not a fresh one.
            final_attempt = count >= self.s.settings.max_clarifications
            text = (CLARIFY_FINAL_PREFIX if final_attempt else CLARIFY_RETRY_PREFIX) + question

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
