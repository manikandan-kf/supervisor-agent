"""Stage 2 — guardrails.

Both tiers of the screen, plus everything the supervisor answers itself:
small talk, session notes, follow-ups, and the deferred second request a
message sometimes carries.
"""

from __future__ import annotations

import logging
from typing import Literal

from agent_governance import review_queue
from agent_governance.resilience import BudgetExhausted, deadline_for
from agent_governance.review_queue import ReviewQueueError
from agent_governance.spend import (
    PROMPT_OVERHEAD_CHARS,
    VERDICT_OUTPUT_TOKENS,
    SpendCaps,
    SpendExhausted,
    TurnSpend,
    estimate_tokens,
)
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from .. import session_notes
from ..guardrail_engine import followup_answer
from ..messages import (
    APPEAL_ACKNOWLEDGED_MESSAGE,
    APPEAL_NOTE,
    APPEAL_UNAVAILABLE_MESSAGE,
    BLOCK_STREAK_ESCALATION_MESSAGE,
    DEFERRED_REQUEST_DROPPED,
    DETERMINISTIC_ESCALATION_MESSAGE,
    GOVERNANCE_UNAVAILABLE_MESSAGE,
    NOTHING_TO_APPEAL_MESSAGE,
    NOTHING_TO_APPEAL_REVIEW_OPEN_MESSAGE,
    REVIEW_IN_PROGRESS_NOTE,
    join_names,
    noted_reply,
    progress,
    sentence,
    small_talk_reply,
)
from ..state import SupervisorContext
from .limits import LimitsMixin
from .turn import (
    _APPEAL_PATTERN,
    _entry,
    _excerpt,
    _history_lines,
    _latest_user_text,
    _narrates,
    _prior_user_text,
    _small_talk_seen,
    _trail,
    _window,
)

logger = logging.getLogger(__name__)


class GuardrailsMixin(LimitsMixin):
    def guardrails(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["route", "respond"]]:
        context = runtime.context or SupervisorContext()
        narrate = _narrates(state)
        if narrate:
            progress("guardrails", "started")

        # ── Emergency stop (kill switch) ─────────────────────────────────────
        # A `kill_switch` in the governed document refuses every turn before the appeal
        # path, any model call or any dispatch; it rides the config publish path. Staged
        # approvals already interrupted are not swept up: that would turn a hold into data loss.
        kill_message = getattr(self.s.guardrails, "kill_switch_message", "") or ""
        if kill_message:
            progress("guardrails", "blocked", "operational hold")
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
                progress("guardrails", "started", "resuming the held request")
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

        # The appeal path out of a block (§06). Checked before the screen: re-screening the
        # *word* "appeal" would block it again — the silent retry the requirement rules out.
        if state.get("appealable") and _APPEAL_PATTERN.match(query or ""):
            appealed = state.get("appealable") or {}

            # Answer the word rather than screening it: screened, "appeal" is off-domain and
            # blocked — and that block would offer the appeal the previous turn withheld.
            if not appealed.get("offered", True):
                progress("guardrails", "blocked", "nothing to appeal")
                return Command(
                    goto="respond",
                    update={
                        # No `**spent`: this branch runs before the screen, so
                        # the turn has made no model call to account for.
                        "guardrail": {
                            "passed": False,
                            "tier": appealed.get("tier", ""),
                            "reason": appealed.get("reason", ""),
                            "appeal_offered": False,
                        },
                        "outcome": "blocked",
                        # Kept, not cleared: clearing it sends the next "appeal" to the screen
                        # as off-domain — blocked with the appeal line attached, the same loop
                        # one turn on. It clears itself when a later query passes.
                        "appealable": appealed,
                        "final_text": appealed.get("no_appeal_reason") or NOTHING_TO_APPEAL_MESSAGE,
                        "audit_trail": _trail(
                            state,
                            "guardrails",
                            "appeal_declined",
                            "the previous refusal offered no appeal path — no review opened",
                        ),
                    },
                )

            blocked_reason = appealed.get("reason", "")
            # The appeal has to *land* before the user is told a human has it. §08: "Fail the
            # operation rather than complete it unaudited."
            try:
                review = self.s.reviews.open_review(
                    kind=review_queue.APPEAL,
                    conversation_id=state.get("conversation_id", ""),
                    reason=blocked_reason,
                    user_key=context.user_key,
                    user_role=context.user_role,
                    target_agent_id=state.get("target_agent_id", ""),
                    request_id=state.get("request_id", ""),
                    correlation_id=context.correlation_id,
                    query_excerpt=_excerpt(_prior_user_text(state.get("messages"))),
                )
            except (ReviewQueueError, ValueError) as exc:
                logger.warning("could not open an appeal: %s", exc)
                progress("guardrails", "error", "appeal not recorded")
                return Command(
                    goto="respond",
                    update={
                        "outcome": "error",
                        "final_text": APPEAL_UNAVAILABLE_MESSAGE,
                        # `appealable` is deliberately kept: the appeal did not
                        # happen, so the user must still be able to take it.
                        "audit_trail": _trail(
                            state,
                            "guardrails",
                            "appeal_failed",
                            f"appeal could not be recorded ({type(exc).__name__}) — "
                            "not reported to the user as flagged",
                        ),
                    },
                )

            progress("guardrails", "error", "appealed to a human")
            return Command(
                goto="respond",
                update={
                    "outcome": "escalated",
                    "final_text": APPEAL_ACKNOWLEDGED_MESSAGE,
                    "appealable": None,
                    # Holds the conversation until a reviewer resolves it, and is
                    # what `_hold_for_review` reads on the next turn.
                    "open_review": {"ref": review.ref, "kind": review.kind},
                    "audit_trail": _trail(
                        state,
                        "guardrails",
                        "appeal",
                        f"appeal {review.ref} opened in the review queue "
                        f"(original block: {blocked_reason})",
                    ),
                },
            )

        # ── A reviewer granted this conversation a retry (§05 Stage 03) ──────
        # `claim_allowance` spends the grant in one conditional UPDATE, so a granted retry is
        # used exactly once even if two turns arrive together. Claimed only on the turn the
        # RBAC gate observed a resolution, so ordinary traffic carries no write round-trip.
        allowance = None
        if state.get("review_resolved"):
            try:
                allowance = self.s.reviews.claim_allowance(state.get("conversation_id", ""))
            except ReviewQueueError as exc:
                # Fails *open*, alone among the review paths: an unclaimed allowance leaves
                # the screen to run normally — the verdict without appealing at all.
                logger.warning("could not check review allowances: %s", exc)

        if allowance is not None:
            trail = _trail(
                state,
                "guardrails",
                "review_allowed",
                f"appeal {allowance.ref} was upheld by {allowance.reviewer or 'a reviewer'}"
                + (f": {allowance.reviewer_note}" if allowance.reviewer_note else "")
                + " — screen bypassed once for this turn",
            )
            if narrate:
                progress("guardrails", "done", "cleared by a reviewer")
            return Command(
                goto="route",
                update={
                    "appealable": None,
                    "open_review": None,
                    "guardrail_block_streak": 0,
                    "guardrail": {
                        "passed": True,
                        "tier": "human_review",
                        "reason": (
                            f"a reviewer resolved appeal {allowance.ref} in the user's favour"
                        ),
                        "reviewer": allowance.reviewer,
                        "review_ref": allowance.ref,
                    },
                    "audit_trail": trail,
                },
            )

        # ── The turn's spend ledger (cost control) ───────────────────────────
        # Rebuilt from state at each node and written back with its update. `ensure` refuses
        # to *start* a screen the turn cannot afford; the charge is what `screen` reports.
        spend = TurnSpend.from_state(state, SpendCaps.from_settings(self.s.settings))
        window = _window(state.get("messages"), self.s.settings.history_max_tokens)
        per_verdict = (
            estimate_tokens(window, extra_chars=len(query or "") + PROMPT_OVERHEAD_CHARS)
            + VERDICT_OUTPUT_TOKENS
        )

        try:
            deadline.ensure("the guardrail screen")
            # One call, not the whole fan-out: the deadline bounds the fan-out, and the worst
            # case up front would refuse a turn one call usually settles. `[:-1]` drops the
            # current turn, which the screen already receives as `query`.
            spend.ensure("the guardrail screen", model_calls=1, tokens=per_verdict)
            screened = self.s.guardrails.screen(
                query, candidates, history=_history_lines(window)[:-1], deadline=deadline
            )
        except BudgetExhausted as exc:
            return self._budget_exhausted(state, "guardrails", exc)
        except SpendExhausted as exc:
            return self._spend_exhausted(state, "guardrails", exc, spend)
        except Exception as exc:
            # A screen failure is a *transport* failure, not a verdict, so §06 holds the query.
            # Type and a bounded message only: client exceptions can carry the request body.
            logger.error(
                "guardrail screening failed for %s: %s: %s",
                state["target_agent_id"],
                type(exc).__name__,
                str(exc)[:200],
            )
            logger.debug("guardrail screening traceback", exc_info=True)
            progress("guardrails", "error", "checks unavailable")
            # Charged even though the screen failed: a call that raised may still have billed,
            # and a cost control takes the conservative reading.
            spend.charge("guardrails", model_calls=1, tokens=per_verdict)
            return Command(
                goto="respond",
                update={
                    "spend": spend.record(),
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

        # `considered` is the agents asked about — one verdict each — so the call count is a
        # fact, not an estimate. A deterministic block or small talk legitimately costs zero.
        verdicts = len(screened.considered)
        if verdicts:
            spend.charge("guardrails", model_calls=verdicts, tokens=per_verdict * verdicts)
        # Merged into every return below, so a block, a clarification and a pass
        # all carry the same accounting into the audit row.
        spent = {"spend": spend.record()}

        if (
            not result.passed
            and getattr(result, "escalate", False)
            and not state.get("open_review")
        ):
            # ── A tier-1 rule published with `action: escalate` ─────────────
            # Refused exactly as a block is, and additionally handed to a reviewer with the
            # conversation held. If the review cannot be recorded, degrade to the plain block:
            # still refused, no reviewer promised (§08). Skipped when a review is already open.
            try:
                review = self.s.reviews.open_review(
                    kind=review_queue.ESCALATION,
                    conversation_id=state.get("conversation_id", ""),
                    reason=f"deterministic escalation: {result.reason}",
                    user_key=context.user_key,
                    user_role=context.user_role,
                    target_agent_id=state.get("target_agent_id", ""),
                    request_id=state.get("request_id", ""),
                    correlation_id=context.correlation_id,
                    query_excerpt=_excerpt(query),
                )
            except (ReviewQueueError, ValueError) as exc:
                logger.warning("could not open a deterministic escalation: %s", exc)
            else:
                progress("guardrails", "error", "escalated for review")
                return Command(
                    goto="respond",
                    update={
                        **spent,
                        "guardrail": {
                            "passed": False,
                            "tier": result.tier,
                            "reason": result.reason,
                            "escalated": True,
                        },
                        "outcome": "escalated",
                        "final_text": DETERMINISTIC_ESCALATION_MESSAGE,
                        "appealable": None,
                        "guardrail_block_streak": state.get("guardrail_block_streak", 0) + 1,
                        "open_review": {"ref": review.ref, "kind": review.kind},
                        "audit_trail": _trail(
                            state,
                            "guardrails",
                            "escalate",
                            f"{result.tier}: {result.reason} — escalation {review.ref} opened, "
                            "conversation held for review",
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
                try:
                    review = self.s.reviews.open_review(
                        kind=review_queue.ESCALATION,
                        conversation_id=state.get("conversation_id", ""),
                        reason=(
                            f"{streak} consecutive guardrail blocks — possible probing "
                            f"or a stuck user (last block: {result.reason})"
                        ),
                        user_key=context.user_key,
                        user_role=context.user_role,
                        target_agent_id=state.get("target_agent_id", ""),
                        request_id=state.get("request_id", ""),
                        correlation_id=context.correlation_id,
                        query_excerpt=_excerpt(query),
                    )
                except (ReviewQueueError, ValueError) as exc:
                    # Degrade to the plain block rather than fail the turn: the preserved
                    # streak makes the next block retry the escalation (§08).
                    logger.warning("could not open a block-streak escalation: %s", exc)
                else:
                    progress("guardrails", "error", "escalated after repeated blocks")
                    return Command(
                        goto="respond",
                        update={
                            **spent,
                            "guardrail": {
                                "passed": False,
                                "tier": result.tier,
                                "reason": result.reason,
                                "block_streak": streak,
                            },
                            "outcome": "escalated",
                            "final_text": BLOCK_STREAK_ESCALATION_MESSAGE,
                            "appealable": None,
                            "guardrail_block_streak": 0,
                            # Terminal until a reviewer resolves it, exactly like
                            # the clarification-cap escalation.
                            "open_review": {"ref": review.ref, "kind": review.kind},
                            "audit_trail": _trail(
                                state,
                                "guardrails",
                                "anomaly_escalate",
                                f"{streak} consecutive blocks reached the streak limit — "
                                f"escalation {review.ref} opened, conversation held for review",
                            ),
                        },
                    )

            progress("guardrails", "blocked", result.reason)
            covered = join_names([a.name for a in reachable])

            # An appeal is withheld only when a reviewer cannot change the outcome: a safety
            # refusal, or a review already open (a second row would orphan the first).
            # Deterministic blocks stay appealable: a regex cannot tell a jailbreak from prose.
            already_reviewing = bool(state.get("open_review"))
            safety_refusal = getattr(result, "safety_refusal", False)
            offer_appeal = not safety_refusal and not already_reviewing

            if offer_appeal:
                closing = APPEAL_NOTE
            elif already_reviewing:
                closing = REVIEW_IN_PROGRESS_NOTE
            else:
                # A safety refusal closes with nothing: inviting the user to
                # rephrase a request for harm is not a way forward.
                closing = ""

            return Command(
                goto="respond",
                update={
                    **spent,
                    "guardrail": {
                        "passed": False,
                        "tier": result.tier,
                        "reason": result.reason,
                        "considered": list(screened.considered),
                        # Recorded so a reviewer auditing a refusal can tell a
                        # withheld appeal from a bug.
                        "safety_refusal": safety_refusal,
                        "appeal_offered": offer_appeal,
                    },
                    "outcome": "blocked",
                    "guardrail_block_streak": streak,
                    # Kept so next turn's "appeal" is recognised rather than screened — even
                    # with `offered: False`, because the word still has to be recognised.
                    "appealable": {
                        "reason": result.reason,
                        "tier": result.tier,
                        "offered": offer_appeal,
                        "no_appeal_reason": (
                            ""
                            if offer_appeal
                            else NOTHING_TO_APPEAL_REVIEW_OPEN_MESSAGE
                            if already_reviewing
                            else NOTHING_TO_APPEAL_MESSAGE
                        ),
                    },
                    # One voice with the small-talk replies; the appeal line closes it (§06).
                    # The reason is model-written under a prompt untrusted content can steer
                    # and is shown word for word, so it passes the output guard's scrub.
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
                            closing,
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
                        )
                        + (
                            " — safety refusal, no appeal offered"
                            if safety_refusal
                            else " — appeal withheld, a review is already open on this conversation"
                            if already_reviewing
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
                    **spent,
                    "guardrail": {
                        "passed": True,
                        "tier": result.tier,
                        "reason": result.reason,
                        # Which kind, not just that it was small talk: a greeting
                        # and a thank-you get different replies.
                        "small_talk": result.small_talk,
                    },
                    "outcome": "answer",
                    "appealable": None,
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

        # ── "…, keep this in mind for later" ────────────────────────────────
        # After the screen, so recognising a note is not a way past a guardrail (see
        # session_notes.py); before the clarification, because the screen's question is
        # "which deliverable?" and the user has just said "not yet".
        note = session_notes.note_from(query) if self.s.settings.session_notes_max else ""
        if note:
            kept, bound_detail = session_notes.record(
                state.get("session_notes"),
                note,
                request_id=state.get("request_id", ""),
                max_notes=self.s.settings.session_notes_max,
                max_chars=self.s.settings.session_note_max_chars,
            )
            if narrate:
                progress("guardrails", "done", result.tier)
            return Command(
                goto="respond",
                update={
                    **spent,
                    "target_agent_id": agent.id,
                    "session_notes": kept,
                    "guardrail": {
                        "passed": True,
                        "tier": result.tier,
                        "reason": result.reason,
                        "considered": list(screened.considered),
                        "deferred": True,
                    },
                    "guardrail_block_streak": 0,
                    "appealable": None,
                    "outcome": "answer",
                    # No worker ran. Said explicitly because the calling UI renders this as
                    # the answering agent.
                    "routed_agent_name": "",
                    "final_text": noted_reply(note, len(kept)),
                    "audit_trail": _trail(
                        state,
                        "guardrails",
                        "noted",
                        "held for this conversation, not dispatched"
                        + (f" ({bound_detail})" if bound_detail else "")
                        + f"; {len(kept)} note(s) now held",
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
                    **spent,
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
                context,
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

        if narrate:
            progress("guardrails", "done", result.tier)
        return Command(
            goto="route",
            update={
                **spent,
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
                # A semantic pass clears the earlier block and resets the streak — never on
                # small talk: "hi" between probes must not launder a streak back to zero.
                "appealable": None,
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
