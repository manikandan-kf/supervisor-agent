"""Stage 2 — guardrails.

Both tiers of the screen, plus everything the supervisor answers itself:
small talk, follow-ups, and the deferred second request a message sometimes
carries.
"""

from __future__ import annotations

import logging
from typing import Literal

from agent_governance.retry_and_deadline import BudgetExhausted, deadline_for
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from ..guardrail_engine import followup_answer
from ..state import SupervisorContext
from ..user_facing_text import (
    BLOCK_STREAK_ESCALATION_MESSAGE,
    DEFERRED_REQUEST_DROPPED,
    DETERMINISTIC_ESCALATION_MESSAGE,
    GOVERNANCE_UNAVAILABLE_MESSAGE,
    join_names,
    sentence,
    small_talk_reply,
)
from .base import (
    _entry,
    _excerpt,
    _history_lines,
    _latest_user_text,
    _small_talk_seen,
    _trail,
    _window,
)
from .failsafes import FailsafesMixin

logger = logging.getLogger(__name__)


class GuardrailsMixin(FailsafesMixin):
    def guardrails(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["route", "respond"]]:
        context = runtime.context or SupervisorContext()

        # ── Emergency stop (kill switch) ─────────────────────────────────────
        # A `kill_switch` in the governed document refuses every turn before any model
        # call or any dispatch; it rides the config publish path. Staged
        # approvals already interrupted are not swept up: that would turn a hold into data loss.
        kill_message = getattr(self.s.guardrails, "kill_switch_message", "") or ""
        if kill_message:
            return Command(
                goto="respond",
                update={
                    "guardrail": {
                        "passed": False,
                        "tier": "kill_switch",
                        "reason": "the operations team has paused the assistant",
                    },
                    "outcome": "blocked",
                    "final_text": kill_message,
                    "audit_trail": _trail(
                        state,
                        "guardrails",
                        "kill_switch",
                        "the emergency stop in the governed guardrails document is "
                        "engaged — turn refused before any model call",
                    ),
                },
            )

        deadline = deadline_for(context, self.s.settings)
        addressed = self.s.registry.get(state["target_agent_id"])
        query = _latest_user_text(state.get("messages"))

        # The agents this caller may reach, the addressed one first so the usual case costs
        # one model call. Nothing outside the permitted set is ever a candidate.
        permitted = context.permitted_agents
        if permitted is None:
            role = (context.user_role or "").strip()
            permitted = tuple(
                a for a in self.s.registry.ids() if self.s.rbac.check(role, a).allowed
            )
        reachable = [found for found in (self.s.registry.get(a) for a in permitted) if found]
        candidates = ([addressed] if addressed else []) + [
            a for a in reachable if not addressed or a.id != addressed.id
        ]

        # ── Answering the held-over-task offer ───────────────────────────────
        # Checked before the screen, because "yes" screened as a request is off-domain and
        # refused. An accepted offer skips nothing: the held text goes through the full
        # screen below. Holding a request is a pause, not a pre-authorisation.
        deferred = state.get("deferred_request") or {}
        deferred_resumed = ""
        if deferred.get("text"):
            answer = followup_answer(query or "")
            if answer == "accept":
                deferred_resumed = str(deferred["text"])
                query = deferred_resumed
            elif answer == "decline":
                return Command(
                    goto="respond",
                    update={
                        "deferred_request": None,
                        "outcome": "answer",
                        "routed_agent_name": "",
                        "final_text": DEFERRED_REQUEST_DROPPED,
                        "guardrail": {
                            "passed": True,
                            "tier": "small_talk",
                            "reason": "the user declined the held-over request",
                        },
                        "audit_trail": _trail(
                            state,
                            "guardrails",
                            "deferred_dropped",
                            f"held request declined by the user: {_excerpt(deferred['text'])}",
                        ),
                    },
                )
            # Anything else is a new request and the offer expires: "yes" three messages
            # later would resume something the user no longer has in front of them.

        window = _window(state.get("messages"), self.s.settings.history_max_tokens)

        try:
            deadline.ensure("the guardrail screen")
            # `[:-1]` drops the current turn, which the screen already receives as `query`.
            screened = self.s.guardrails.screen(
                query, candidates, history=_history_lines(window)[:-1], deadline=deadline
            )
        except BudgetExhausted as exc:
            return self._budget_exhausted(state, "guardrails", exc)
        except Exception as exc:
            # A screen failure is a *transport* failure, not a verdict, so the query is held
            # (solution §05: fail closed, never an implicit pass).
            # Type and a bounded message only: client exceptions can carry the request body.
            logger.error(
                "guardrail screening failed for %s: %s: %s",
                state["target_agent_id"],
                type(exc).__name__,
                str(exc)[:200],
            )
            logger.debug("guardrail screening traceback", exc_info=True)
            return Command(
                goto="respond",
                update={
                    "guardrail": {"passed": False, "tier": "unavailable", "reason": str(exc)[:200]},
                    "outcome": "error",
                    "final_text": GOVERNANCE_UNAVAILABLE_MESSAGE,
                    "audit_trail": _trail(
                        state,
                        "guardrails",
                        "fail_closed",
                        f"screening unavailable ({type(exc).__name__}) — request held, not dispatched",
                    ),
                },
            )
        result = screened.result
        agent = screened.agent or addressed

        if not result.passed and getattr(result, "escalate", False):
            # ── A tier-1 rule published with `action: escalate` ─────────────
            # Refused exactly as a block is, and recorded as an escalation rather than a
            # block, for requests a refusal alone under-reports.
            return Command(
                goto="respond",
                update={
                    "guardrail": {
                        "passed": False,
                        "tier": result.tier,
                        "reason": result.reason,
                        "escalated": True,
                    },
                    "outcome": "escalated",
                    "final_text": DETERMINISTIC_ESCALATION_MESSAGE,
                    "guardrail_block_streak": state.get("guardrail_block_streak", 0) + 1,
                    "audit_trail": _trail(
                        state,
                        "guardrails",
                        "escalate",
                        f"{result.tier}: {result.reason} — recorded for a human reviewer",
                    ),
                },
            )

        if not result.passed:
            # ── Repeated-block anomaly (guardrail layer 6) ───────────────────
            # One block is a verdict; a streak is a signal. After the limit the conversation
            # goes to a human. Per-conversation and deterministic, so rephrasing cannot dilute it.
            streak = state.get("guardrail_block_streak", 0) + 1
            limit = self.s.settings.guardrail_block_streak_limit
            if limit > 0 and streak >= limit:
                return Command(
                    goto="respond",
                    update={
                        "guardrail": {
                            "passed": False,
                            "tier": result.tier,
                            "reason": result.reason,
                            "block_streak": streak,
                        },
                        "outcome": "escalated",
                        "final_text": BLOCK_STREAK_ESCALATION_MESSAGE,
                        "guardrail_block_streak": 0,
                        "audit_trail": _trail(
                            state,
                            "guardrails",
                            "anomaly_escalate",
                            f"{streak} consecutive blocks reached the streak limit — "
                            "recorded for a human reviewer",
                        ),
                    },
                )

            covered = join_names([a.name for a in reachable])

            # A block is final. There is no appeal path: a refused request is never re-screened
            # on the user's say-so and cannot be overturned from the conversation. The scope
            # limitation and what this role can reach are stated, and the turn ends.
            return Command(
                goto="respond",
                update={
                    "guardrail": {
                        "passed": False,
                        "tier": result.tier,
                        "reason": result.reason,
                        "considered": list(screened.considered),
                    },
                    "outcome": "blocked",
                    "guardrail_block_streak": streak,
                    # One voice with the small-talk replies. The reason is model-written under
                    # a prompt untrusted content can steer and is shown word for word, so it
                    # passes the output guard's scrub.
                    "final_text": " ".join(
                        part
                        for part in (
                            "I can't route this request. "
                            + sentence(
                                self.s.output_guard.scrub(
                                    result.reason, "it isn't something I can pass to an agent"
                                )
                            ),
                            f"For your role I can reach {covered}." if covered else "",
                        )
                        if part
                    ),
                    "audit_trail": _trail(
                        state,
                        "guardrails",
                        "block",
                        f"{result.tier}: {result.reason}"
                        + (
                            f" (considered {', '.join(screened.considered)})"
                            if screened.considered
                            else ""
                        ),
                    ),
                },
            )

        if result.small_talk:
            # Answered by the supervisor: a greeting sent to the worker yields a wall of
            # capabilities and spends a call. All reachable agents: a greeting has no topic.
            return Command(
                goto="respond",
                update={
                    "guardrail": {
                        "passed": True,
                        "tier": result.tier,
                        "reason": result.reason,
                        # Which kind, not just that it was small talk: a greeting
                        # and a thank-you get different replies.
                        "small_talk": result.small_talk,
                    },
                    "outcome": "answer",
                    "final_text": small_talk_reply(
                        result.small_talk,
                        [a.name for a in reachable],
                        _small_talk_seen(state.get("messages"), result.small_talk),
                    ),
                    # No worker ran, so nothing is attributed to one and no
                    # sources are claimed.
                    "routed_agent_name": "",
                    "audit_trail": _trail(
                        state,
                        "guardrails",
                        "answered_directly",
                        f"{result.small_talk} (x{_small_talk_seen(state.get('messages'), result.small_talk) + 1}): "
                        "answered by the supervisor, no worker call",
                    ),
                },
            )

        if result.clarification:
            # In-domain subject, no deliverable named. Asked here rather than at `route`
            # because the questions differ ("what do you want produced?" vs "which product
            # line?"). `target_agent_id` moves too, so the answer lands on the agent that asked.
            return self._clarify_or_escalate(
                state,
                "guardrails",
                # Model-written and shown verbatim, so scrubbed like `reason`.
                self.s.output_guard.scrub(
                    result.clarification, "Which deliverable would you like produced?"
                ),
                {
                    "target_agent_id": agent.id,
                    # The screen found the subject in-domain, so this is a real
                    # user mid-conversation, not a probe.
                    "guardrail_block_streak": 0,
                    "guardrail": {
                        "passed": True,
                        "tier": result.tier,
                        "reason": result.reason,
                        "considered": list(screened.considered),
                        "underspecified": True,
                    },
                },
            )

        # The screen may have retargeted to another agent the caller can reach. Safe because
        # every candidate came from the permitted set — the choice cannot widen access.
        retargeted = agent.id != state["target_agent_id"]
        trail = [
            _entry(
                "guardrails",
                "pass",
                f"{result.tier}: {result.reason}"
                + (
                    f" (considered {', '.join(screened.considered)})" if screened.considered else ""
                ),
            )
        ]
        if retargeted:
            trail.append(
                _entry(
                    "guardrails",
                    "retargeted",
                    f"{state['target_agent_id']} -> {agent.id}: the query belongs to {agent.name}",
                )
            )

        if getattr(result, "contested", ()):
            # Recorded even though the user was asked: "two agents claimed it" is invisible in
            # a trail that only names the winner, and is what a reviewer asks about.
            trail.append(
                _entry(
                    "guardrails",
                    "contested",
                    "more than one permitted agent claimed the request: "
                    + ", ".join(result.contested),
                )
            )

        # A second deliverable in the same message, carried on the guardrail record: it is
        # offered back only once the first request has produced something.
        additional = (getattr(result, "additional_request", "") or "").strip()
        # Never offer back the request being answered now: on a resumed turn the held text
        # *is* the query, and a screen reporting it as a second deliverable would loop forever.
        if additional and deferred_resumed:
            if additional.strip().lower() == deferred_resumed.strip().lower():
                additional = ""
        if additional:
            trail.append(
                _entry(
                    "guardrails",
                    "second_request",
                    f"the message carried a second deliverable: {_excerpt(additional)}",
                )
            )

        if deferred_resumed:
            trail.append(
                _entry(
                    "guardrails",
                    "deferred_resumed",
                    f"held request taken up on the user's confirmation: "
                    f"{_excerpt(deferred_resumed)} — screened as a fresh request",
                )
            )

        return Command(
            goto="route",
            update={
                "target_agent_id": agent.id,
                # Consumed. Whether it was taken up or expired, the offer does
                # not survive the turn that answered it.
                "deferred_request": None,
                # The held request, restated as the user's own words, so the
                # transcript says what is being worked on rather than "yes".
                **(
                    {"messages": [HumanMessage(content=deferred_resumed)]}
                    if deferred_resumed
                    else {}
                ),
                # A semantic pass resets the streak — never on small talk: "hi" between
                # probes must not launder a streak back to zero.
                "guardrail_block_streak": 0,
                "guardrail": {
                    "passed": True,
                    "tier": result.tier,
                    "reason": result.reason,
                    "considered": list(screened.considered),
                    "retargeted": retargeted,
                    "contested": list(getattr(result, "contested", ()) or ()),
                    # Read by `dispatch` after a successful answer, which is the
                    # only place it can be offered honestly.
                    "additional_request": additional,
                },
                "audit_trail": state.get("audit_trail", []) + trail,
            },
        )
