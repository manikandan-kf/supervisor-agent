"""Graph nodes.

RBAC gate -> Guardrails -> Route/Clarify -> Dispatch -> Respond & Audit. The
order is fixed by the graph and cannot be skipped.

Each chat widget is already scoped to one agent, so the target arrives fixed on
every invocation (`POST /agents/{agent-id}/invocation`). The supervisor's job
is verification and in-domain clarification, not agent selection: the RBAC gate
revalidates the path's agent-id against the caller's role mapping server-side
on every call — the UI's agent selection is never trusted as the access
decision.

Each node returns a `Command`, which carries the state update *and* the next
destination together. The alternative — set `outcome` in state and have a
separate routing function read it back — splits one decision across two places
and lets them disagree. A node that blocks says so and says where to go, once.

Identity arrives as `Runtime[SupervisorContext]`, never from state. See
`context.py`.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

from agent_governance import grounding, review_queue
from agent_governance.deadline import BudgetExhausted, deadline_for
from agent_governance.rbac import DENIED_MESSAGE
from agent_governance.review_queue import ReviewQueueError
from agent_governance.sanitize import (
    clean_worker_output,
    neutralise_embedded_directives,
    neutralise_history_text,
)
from agent_governance.sensitive import redact_structure, redact_text
from agent_governance.spend import (
    PROMPT_OVERHEAD_CHARS,
    VERDICT_OUTPUT_TOKENS,
    SpendCaps,
    SpendExhausted,
    TurnSpend,
    estimate_tokens,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from . import progress, session_notes
from .context import SupervisorContext
from .dispatch import WorkerUnavailable
from .guardrails import followup_answer, small_talk_kind
from .messages import (
    APPEAL_ACKNOWLEDGED_MESSAGE,
    APPEAL_NOTE,
    APPEAL_UNAVAILABLE_MESSAGE,
    APPROVAL_PENDING_MESSAGE,
    AUDIT_UNAVAILABLE_MESSAGE,
    BLOCK_STREAK_ESCALATION_MESSAGE,
    BUDGET_EXHAUSTED_MESSAGE,
    CARRIED_CONTEXT_NOTICE,
    CLARIFY_FINAL_PREFIX,
    CLARIFY_RETRY_PREFIX,
    DEFERRED_REQUEST_DROPPED,
    DEFERRED_REQUEST_OFFER,
    DEFERRED_REQUEST_REFUSED,
    DETERMINISTIC_ESCALATION_MESSAGE,
    ESCALATION_MESSAGE,
    GENERIC_ERROR,
    GOVERNANCE_UNAVAILABLE_MESSAGE,
    INPUT_TOO_LONG_MESSAGE,
    NOTHING_TO_APPEAL_MESSAGE,
    NOTHING_TO_APPEAL_REVIEW_OPEN_MESSAGE,
    OUTPUT_ESCALATED_MESSAGE,
    OUTPUT_WITHHELD_MESSAGE,
    REVIEW_IN_PROGRESS_NOTE,
    REVIEW_PENDING_MESSAGE,
    REVIEW_UNAVAILABLE_MESSAGE,
    SENSITIVE_INPUT_NOTICE,
    SESSION_EXPIRED_MESSAGE,
    SPEND_EXHAUSTED_MESSAGE,
    SUBJECT_ALLOWANCE_MESSAGE,
    WORKER_UNAVAILABLE_MESSAGE,
    context_prose,
    join_names,
    noted_reply,
    sensitive_prose,
    sentence,
    small_talk_reply,
)

logger = logging.getLogger(__name__)

# Outcomes that *are* a governance decision, as opposed to an answer that
# happened to pass through one. §05 Stage 06 requires these be written
# synchronously — "not reported as applied unless its audit record durably
# landed" — and §08 repeats it as a failsafe: "Fail the operation rather than
# complete it unaudited."
#
# `answer` is deliberately absent. Failing a user's question because the audit
# sink is briefly unreachable trades a real outage for a record of a decision
# that did not restrict anyone; failing a *block* silently is how a refusal
# becomes unprovable. The split is the point, and it is recorded rather than
# implied.
#
# An approval or rejection also qualifies, but it does not appear here: those
# keep `outcome == "answer"` so any routing-completion metric derived from it
# is unaffected, and are recognised instead by a `signoff` record in state. See
# `_is_governance_decision`.
GOVERNANCE_OUTCOMES = frozenset({"blocked", "escalated", "expired", "review_pending"})


def _narrates(state: dict) -> bool:
    """Whether this turn should stream a task plan at all.

    The plan is the governance pipeline made visible — which gate ran, which one
    stopped the request, which agent the work went to. A greeting goes through
    none of that: no routing, no worker, not even a model call. Narrating three
    ticked stages for "hi" claims work that never happened, and it buries a
    one-line reply under a card bigger than itself.

    Decided from the message rather than from a result, so the very first event
    is already suppressed — the alternative is rendering the plan and then
    retracting it, which reads as a glitch. Refusals and errors ignore this and
    always narrate: if a greeting somehow *is* stopped by a gate, the user needs
    to see which one.
    """
    return not small_talk_kind(_latest_user_text(state.get("messages")))

# Anchored, so only a message that is *entirely* an appeal counts. "appeal the
# decision to buy" inside a real request must not silently become one.
_APPEAL_PATTERN = re.compile(r"^\W*appeal\b[\s\S]{0,120}$", re.IGNORECASE)


def _small_talk_seen(messages, kind: str) -> int:
    """How many earlier user turns in this conversation were this same kind.

    Counts the conversation the checkpointer already holds, so the escalation
    survives across turns rather than resetting each request.
    """
    if not kind:
        return 0
    seen = 0
    for msg in (messages or [])[:-1]:  # the current turn is the last message
        if getattr(msg, "type", "") != "human":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if small_talk_kind(content) == kind:
            seen += 1
    return seen



def _latest_user_text(messages) -> str:
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") == "human":
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _prior_user_text(messages) -> str:
    """The user turn *before* the current one.

    What a reviewer needs to see on an appeal is the request that was blocked,
    not the word "appeal" that appealed it. Falls back to the current turn when
    there is no earlier one, so an excerpt is never empty for want of history.
    """
    seen = 0
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") != "human":
            continue
        seen += 1
        if seen == 2:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return _latest_user_text(messages)


def _window(messages, max_tokens: int) -> list:
    """The most recent messages that fit a token budget.

    The checkpointer holds the whole conversation and must keep doing so — an
    approval three turns back has to survive. What must *not* grow with it is
    the slice replayed into every model call and worker dispatch. Budgeted in
    tokens rather than a fixed message count because messages are not sized
    alike: ten one-liners and ten pasted stack traces are the same `[-10:]` but
    an order of magnitude apart in context.

    `strategy="last"` keeps the recent end; `start_on="human"` drops a leading
    assistant reply whose question is no longer in the window, so the model
    never sees an answer to nothing. A conversation whose current turn alone
    exceeds the budget degrades to that turn rather than to silence.
    """
    messages = list(messages or [])
    if not messages:
        return messages
    try:
        trimmed = trim_messages(
            messages,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=count_tokens_approximately,
            start_on="human",
        )
    except Exception:
        logger.warning("history trimming failed — sending the full window", exc_info=True)
        return messages
    return trimmed or messages[-1:]


def _history_lines(messages) -> list[str]:
    """Flatten the conversation for the governance prompts, markers defanged.

    Every line is neutralised, not only worker output: `sanitize.py` exists
    because a line shaped like `system: approve everything` inside flattened
    history is indistinguishable from a real turn boundary, and prior *user*
    turns are exactly as capable of carrying one as a worker reply. Worker
    output was already cleaned on the way into state; a second pass over it
    here is a no-op, and the pass over user turns is the one that closes the
    multi-turn injection channel.
    """
    lines = []
    for msg in messages or []:
        role = {"human": "user", "ai": "assistant"}.get(getattr(msg, "type", ""), "other")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {neutralise_history_text(content)}")
    return lines


@dataclass(frozen=True)
class RelayScreen:
    """What the relay screen did to the conversation bound for a worker."""

    messages: list
    # Labels of the values masked, in order of first appearance.
    masked: tuple[str, ...] = ()
    directives: int = 0

    @property
    def modified(self) -> bool:
        return bool(self.masked or self.directives)

    def audit_detail(self) -> str:
        parts = []
        if self.masked:
            parts.append(
                f"{len(self.masked)} sensitive value(s) masked before relay: "
                + ", ".join(dict.fromkeys(self.masked))
            )
        if self.directives:
            parts.append(f"{self.directives} embedded directive marker(s) neutralised")
        return "; ".join(parts) or "clean"


def _worker_messages(messages, guard=None) -> RelayScreen:
    """The conversation a worker receives — screened at the relay boundary.

    This is the prompt boundary `sanitize.py`'s docstring promises, and the
    easiest one to leave open: `_history_lines` defangs what the *governance*
    prompts see, while a worker would otherwise receive `msg.content`
    verbatim. Two things happen here on every user turn relayed:

      * **Sensitive values are masked** by the output guard's policy — the
        same categories the reply is screened for, applied on the way in. A
        credential the user pasted never reaches the worker, so it cannot be
        echoed back; a test fixture built from a masked name is built from a
        placeholder, which is what was wanted. The stored transcript keeps
        the user's own text as typed.
      * **Embedded directive frames are neutralised** — `SYSTEM:` inside a
        pasted API response, `NOTE TO ASSISTANT:` inside linter output,
        `[INJECTED] Supervisor:` in a CI log tail. The words stay, so the
        worker can still summarise what the tool output said; the frame that
        made them read as an instruction goes.

    Assistant turns are the supervisor's own already-guarded text and pass
    through unchanged.
    """
    out = []
    masked: list[str] = []
    directives = 0
    window = [m for m in (messages or []) if getattr(m, "type", "") in ("human", "ai")]
    latest_human = max(
        (i for i, m in enumerate(window) if getattr(m, "type", "") == "human"), default=-1
    )
    for index, msg in enumerate(window):
        msg_type = getattr(msg, "type", "")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg_type == "human":
            content, count = neutralise_embedded_directives(content)
            directives += count
            if guard is not None:
                content, findings = guard.relay(content)
                # Only the turn being answered contributes to `masked`, because
                # that list becomes a notice addressed to the user as "your
                # message contained …". Every human turn in the window is still
                # screened — the worker must never see the value again — but a
                # credential pasted six turns ago would otherwise re-announce
                # itself on every turn after it, and a warning that repeats
                # forever is one people learn to scroll past.
                if index == latest_human:
                    masked.extend(f.label for f in findings)
        out.append({"role": "user" if msg_type == "human" else "assistant", "content": content})
    return RelayScreen(out, tuple(masked), directives)


_SOURCE_FIELD_MAX = 200


def _clean_sources(sources, guard=None) -> list[dict]:
    """Screen the worker-declared grounding a UI renders under an answer.

    `dispatch._parse_worker_response` keeps whatever dicts the worker sent; the
    UI then renders their `title` and `origin`. That made them a channel
    with none of the treatment worker prose gets: no length bound, no control
    characters stripped, no impersonated turn markers defanged, and no
    sensitive-value masking — a Sources card could carry a credential or a
    directive frame. Bounded and screened here, once, before either the stream
    or state sees them.
    """
    out: list[dict] = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        cleaned: dict = {}
        for key, value in source.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            text = neutralise_history_text(value)[:_SOURCE_FIELD_MAX]
            if guard is not None:
                text, _ = guard.relay(text)
            cleaned[key] = text
        if cleaned:
            out.append(cleaned)
    return out


def _entry(stage: str, decision: str, detail: str) -> dict:
    return {
        "stage": stage,
        "decision": decision,
        "detail": detail,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def _trail(state: dict, stage: str, decision: str, detail: str) -> list:
    return state.get("audit_trail", []) + [_entry(stage, decision, detail)]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seconds_since(stamp, what: str) -> Optional[float]:
    """Seconds since a checkpointed ISO timestamp, or None when unusable.

    Parsed rather than kept as a number so the value stays meaningful across
    the process restarts a long-lived conversation will see. A malformed value
    returns None rather than raising: an unparseable timestamp should not be
    able to expire — or refuse to expire — a session on its own.
    """
    if not stamp:
        return None
    try:
        began = datetime.fromisoformat(str(stamp))
    except ValueError:
        logger.warning("unparseable %s %r — age unknown", what, stamp)
        return None
    if began.tzinfo is None:
        began = began.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - began).total_seconds())


def _session_age_seconds(state: dict) -> Optional[float]:
    """How long this conversation has been open — the audit row's Duration KPI."""
    return _seconds_since(state.get("session_started_at"), "session_started_at")


def _session_idle_seconds(state: dict) -> Optional[float]:
    """Time since the last turn — what the lifetime bound measures (§05 Stage 04).

    "Abandoned" is about inactivity, not total age: a conversation someone
    worked in an hour ago is not abandoned on its seventh day. Conversations
    checkpointed before `session_last_active_at` existed fall back to the start
    time, which is the old, stricter measure — old idle threads still expire.
    """
    idle = _seconds_since(state.get("session_last_active_at"), "session_last_active_at")
    return idle if idle is not None else _session_age_seconds(state)


def _is_governance_decision(state: dict) -> bool:
    """Whether this turn *decided* something, as opposed to answering something.

    Two ways to qualify, because they are recorded differently:

      * the outcome is itself a decision — a block, an escalation, an expiry, a
        refusal to proceed while a review is open;
      * the turn carried a sign-off, which keeps `outcome == "answer"` so the
        routing-completion KPI is unaffected but is still a human approving or
        rejecting an artifact and must not go unrecorded.
    """
    return bool(
        state.get("outcome", "") in GOVERNANCE_OUTCOMES or state.get("signoff")
    )


def _provenance(settings, spend: TurnSpend) -> dict:
    """How this turn was decided: which model, which prompts, what it spent.

    §4.1 makes prompts immutable versions promoted by a *movable* alias, and
    `register_prompts.py --pin` moves that alias onto a running endpoint within
    MLflow's 60-second cache. So "the dev alias" does not identify the text
    that produced a verdict, and nothing in the audit row did — which left the
    first question any investigation of a bad routing decision asks
    unanswerable from the compliance record. The model endpoint is here for the
    same reason: with multi-model support it varies per agent, and this row is
    where a cost or quality regression gets attributed.

    `prompts` is process-scoped rather than turn-scoped, stated plainly because
    the difference matters to whoever reads the row: it names the versions this
    replica most recently loaded, which for any turn that made a governance
    call *is* that turn's version, and for a turn that made none (a denial, a
    greeting) is the previous turn's. Threading a per-turn record through three
    engines to sharpen that would only disambiguate rows where the trail
    already shows no screen ran.

    Returns `{}` for a turn with nothing to report, so a replica that has only
    ever denied requests claims no provenance it does not have.
    """
    from .prompt_provider import loaded_prompt_versions

    versions = loaded_prompt_versions()
    by_stage = spend.record()["by_stage"]
    if not versions and not by_stage:
        return {}
    record: dict = {"routing_model": getattr(settings, "routing_llm_endpoint", "")}
    if versions:
        record["prompts"] = versions
    if by_stage:
        record["spend_by_stage"] = by_stage
    return record


def _excerpt(text: str, limit: int = 500) -> str:
    """A short, safe copy of the user's query for a reviewer to read.

    Bounded because it lands in a governance table a human queries, and passed
    through the worker-output sanitizer for the same reason worker output is:
    the reviewer's console renders it, and a query carrying control characters
    or a fake turn boundary should not be the thing that corrupts the surface
    built to review it. Redacted last, because the queries most likely to be
    reviewed are the blocked ones, and blocked queries are where pasted
    credentials show up.
    """
    return redact_text(clean_worker_output(text or "", max_chars=limit).text)[0]


class SupervisorNodes:
    def __init__(self, services):
        self.s = services

    def _may_approve(self, context: SupervisorContext, agent_id: str) -> bool:
        """May this caller sign off work produced by `agent_id`? (§2.11)

        `approvable_agents` is the caller's `<agent_key>_approve_action` grants,
        derived by the Platform API from the validated token. As with
        `permitted_agents`, None and () differ: an empty tuple is a real answer
        — this caller may approve nothing — while None means the field was not
        supplied at all and only occurs off the deployed path (a local trace, a
        direct endpoint call). There is no bundled approve map to fall back to,
        so that case defers to the access decision and says so in the log,
        rather than silently denying every local approval run.
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

    # ── Session lifetime and review holds, used by the RBAC gate ────────────

    def _expire_if_stale(
        self, state: dict, context: SupervisorContext, agent_id: str, reset: dict, reason: str
    ) -> Optional[Command]:
        """Clear an over-age conversation's carried state, or None if it is fresh.

        *"Bound session lifetime and record duration, so an abandoned session
        cannot hold an approval or context open indefinitely"* — GDPR Art.
        5(1)(e), SOC 2 CC6.1.

        What expires is the state that carries influence forward: the resolved
        context, a draft awaiting sign-off, an open review marker, the
        clarification counter. The *messages* are left alone — they are the
        conversation record, they are what the audit trail refers to, and
        deleting them here would be a retention decision masquerading as a
        session bound. Message retention belongs to the scheduled purge, which
        is a per-category policy (§06.4) with its own job.

        Age is **idle time**, not time since the first turn: measuring from the
        start expired conversations mid-use — a turn on day eight of an active
        thread wiped context the user established minutes earlier. And a
        conversation holding a staged artifact gets the longer
        `approval_max_age_seconds` bound, because a reviewer on leave is not an
        abandonment and destroying an in-progress sign-off on a timer is data
        loss, not hygiene. The bound still exists in both cases — an abandoned
        approval cannot hang open forever (§05 Stage 04).

        Evaluated lazily on the next turn rather than by a sweeper, so the bound
        holds with no scheduled job running. A conversation nobody returns to is
        never *used* again, which is the property that matters; that its rows
        still exist is the purge job's problem, not this gate's.
        """
        max_age = float(getattr(self.s.settings, "session_max_age_seconds", 0) or 0)
        if max_age <= 0:
            return None
        if state.get("pending_approval"):
            max_age = max(
                max_age,
                float(getattr(self.s.settings, "approval_max_age_seconds", 0) or 0),
            )
        age = _session_idle_seconds(state)
        if age is None or age <= max_age:
            return None

        # Nothing carried forward means nothing to clear, so an old but empty
        # conversation is not interrupted to be told its emptiness expired —
        # but its clock is still restarted (via `reset`, which the gate merges
        # whatever happens next), so the same stale timestamp cannot shadow
        # every later turn of a conversation that has moved on.
        carried = bool(
            state.get("pending_approval")
            or state.get("session_context")
            or state.get("open_review")
            or state.get("pending_clarification")
            or state.get("session_notes")
            or state.get("deferred_request")
        )

        logger.info(
            "session %s expired after %.0fs idle (limit %.0fs), carried_state=%s",
            state.get("conversation_id", ""),
            age,
            max_age,
            carried,
        )
        cleared = {
            "pending_approval": None,
            "open_review": None,
            "session_context": {},
            "clarification_count": 0,
            "pending_clarification": None,
            "appealable": None,
            # Notes expire with the session that made them. They are the user's
            # own words held in short-term memory, so leaving them behind would
            # turn a bounded session into unbounded retention of free text
            # (§05 Stage 04, GDPR Art. 5(1)(e)) — and would surprise a user who
            # was told the note stays with this conversation.
            "session_notes": [],
            # An offer to run a held-over task expires with the session for the
            # same reason: the user who comes back tomorrow is starting a new
            # exchange, and "yes" then would resume something they no longer
            # have in front of them.
            "deferred_request": None,
            # Restart the clock, so the user's next message runs normally rather
            # than expiring again on the same stale timestamp.
            "session_started_at": _now_iso(),
        }
        if not carried:
            reset["session_started_at"] = _now_iso()
            return None

        progress.step("rbac_gate", "blocked", "session expired")
        return Command(
            goto="respond",
            update={
                **reset,
                **cleared,
                "target_agent_id": agent_id,
                "rbac": {"allowed": True, "reason": reason},
                "outcome": "expired",
                "final_text": SESSION_EXPIRED_MESSAGE,
                "audit_trail": [
                    _entry("rbac_gate", "allow", reason),
                    _entry(
                        "session",
                        "expired",
                        f"session idle {age:.0f}s exceeded the {max_age:.0f}s limit — "
                        "carried context, pending approval and review markers cleared",
                    ),
                ],
            },
        )

    def _hold_for_review(
        self,
        state: dict,
        context: SupervisorContext,
        agent_id: str,
        reset: dict,
        reason: str,
        extra_trail: list,
    ) -> Optional[Command]:
        """Hold the conversation while a human review is open, or None to proceed.

        Four outcomes, and the ordering between them is the control:

          * **nothing open** — the common case. Returns without touching the
            queue, so ordinary traffic has no dependency on it at all.
          * **an escalation is still open** — refuse the turn. This is what
            makes an escalation *terminal* rather than a label on one reply
            (§05 Stage 04).
          * **an appeal is still open** — let the turn run. The appealed request
            stays refused, but the user keeps the assistant for everything else;
            see the reasoning at the branch itself.
          * **resolved** — clear the marker and let the turn run. `guardrails`
            then claims the retry allowance if the reviewer granted one, which
            is where the resolution changes what the system does.

        Fails closed when the queue cannot be read, and only for conversations
        whose own state already says a review is open. Letting an escalated
        conversation proceed unreviewed because a database blinked would make
        the terminal state advisory.
        """
        marker = state.get("open_review") or {}
        ref = str(marker.get("ref") or "")
        if not ref:
            return None

        try:
            review = self.s.reviews.get(ref)
        except ReviewQueueError as exc:
            logger.warning("review status unavailable for %s: %s", ref, exc)
            progress.step("rbac_gate", "blocked", "review status unavailable")
            return Command(
                goto="respond",
                update={
                    **reset,
                    "target_agent_id": agent_id,
                    "rbac": {"allowed": True, "reason": reason},
                    "outcome": "error",
                    "final_text": REVIEW_UNAVAILABLE_MESSAGE,
                    "audit_trail": [
                        _entry("rbac_gate", "allow", reason),
                        _entry(
                            "review",
                            "fail_closed",
                            f"{ref} is open in this conversation's state and the review "
                            f"queue could not be read ({type(exc).__name__}) — turn held",
                        ),
                    ],
                },
            )

        # A marker pointing at a row that no longer exists would hold the
        # conversation forever. Treat it as resolved and say so: an operator
        # deleting a review row is a legitimate act, and a permanent lock is not
        # a defensible response to it.
        if review is None:
            logger.warning("review %s referenced by state no longer exists — clearing", ref)
            reset["open_review"] = None
            extra_trail.append(
                _entry(
                    "review",
                    "cleared",
                    f"{ref} is no longer in the review queue — hold released",
                )
            )
            return None

        if review.open:
            # An open **appeal** does not freeze the conversation.
            #
            # What §05 Stage 04 requires to be terminal is the *escalated
            # request* — the blocked query must not be quietly retried until a
            # human has ruled on it. Freezing every later message was a much
            # broader reading of that, and it made the appeal path actively
            # hostile: a user who disputed one scope call lost the assistant
            # entirely, for as long as the queue took, and was told to abandon
            # the conversation and start again. Since a reviewer's reply lands
            # in that same conversation, "start a new one" is also advice that
            # walks the user away from where the answer will arrive.
            #
            # The retry it was guarding against is already guarded twice over:
            # a rephrased version of the blocked request meets the same
            # guardrail and is blocked again, and repeated blocks trip the
            # streak anomaly into an escalation. Nothing about letting an
            # unrelated question through weakens either.
            #
            # An **escalation** stays terminal, and the asymmetry is the point.
            # An appeal is the user exercising a right after one refusal; an
            # escalation is the system's own response to a conversation that has
            # gone wrong — a block streak that looks like probing, or a
            # clarification loop that cannot converge. Continuing to serve
            # either is the wrong answer.
            if review.kind == review_queue.APPEAL:
                extra_trail.append(
                    _entry(
                        "review",
                        "open",
                        f"appeal {review.ref} is still with a reviewer — this turn is "
                        "unrelated to the appealed request and proceeds normally; the "
                        "appealed request itself stays refused until it is resolved",
                    )
                )
                return None

            progress.step("rbac_gate", "blocked", f"{review.kind} under review")
            return Command(
                goto="respond",
                update={
                    **reset,
                    "target_agent_id": agent_id,
                    "rbac": {"allowed": True, "reason": reason},
                    "outcome": "review_pending",
                    "final_text": REVIEW_PENDING_MESSAGE,
                    "audit_trail": [
                        _entry("rbac_gate", "allow", reason),
                        _entry(
                            "review",
                            "held",
                            f"{review.kind} {review.ref} is still open — turn refused, "
                            "state is terminal until a reviewer resolves it",
                        ),
                    ],
                },
            )

        # Resolved. Release the hold and record who decided what — this is the
        # "what that decision then changed" half of Stage 03's end-to-end appeal
        # audit, which the decision trail could not previously answer.
        # `review_resolved` tells `guardrails` that *this* turn observed a
        # resolution, so the allowance-claim UPDATE runs only now instead of on
        # every conversation's every turn.
        reset["open_review"] = None
        reset["review_resolved"] = {"ref": review.ref, "kind": review.kind}
        extra_trail.append(
            _entry(
                "review",
                "resolved",
                f"{review.kind} {review.ref} resolved as '{review.decision}' by "
                f"{review.reviewer or 'a reviewer'}"
                + (f": {review.reviewer_note}" if review.reviewer_note else ""),
            )
        )
        return None

    # ── 1. RBAC gate ────────────────────────────────────────────────────────
    def rbac_gate(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["guardrails", "approval", "respond"]]:
        """Check the requested agent against the caller's role-to-agent mapping.

        First node, so it also resets the per-turn fields that must not leak
        from the previous turn's checkpoint.

        The Governance Front Door already authorized the caller against the
        path's agent-id, but this gate revalidates it in-graph on every call —
        defence in depth against an IDOR on the path parameter, a stale
        permission, or a direct endpoint invocation that bypassed the gateway.
        The decision is logged on every check, not just on denial.
        """
        context = runtime.context or SupervisorContext()
        narrate = _narrates(state)
        if narrate:
            progress.step("rbac_gate", "started")

        reset = {
            "request_id": str(uuid.uuid4()),
            "outcome": "",
            "final_text": "",
            "worker_response": None,
            "routed_agent_name": "",
            "sources": [],
            # Cleared every turn. A stale sign-off would make the *next* turn
            # look like a governance decision to `_is_governance_decision`, and
            # would re-audit an approval that already happened.
            "signoff": None,
            # One-turn marker; only `_hold_for_review` ever sets it.
            "review_resolved": None,
            # Per-turn, like the stage results above: the ledger measures one
            # turn's governance spend, and carrying it forward would make every
            # later turn of a long conversation refuse itself (spend.py).
            "spend": {},
            # First turn stamps the session clock; later turns keep the value
            # they already have, because `add`-style merging does not apply to a
            # plain str channel and re-stamping would make every session
            # perpetually young. The *activity* clock, by contrast, is stamped
            # on every turn — it is what the idle-based lifetime bound reads.
            "session_started_at": state.get("session_started_at") or _now_iso(),
            "session_last_active_at": _now_iso(),
        }

        # ── Entitlement provenance (trust.py) ───────────────────────────────
        # When a trust secret is configured, an entitlement block that does not
        # carry the gateway's signature is refused outright. Checked before
        # anything reads `permitted_agents` or `user_role`, because every
        # downstream control is built on those fields and an unverified block
        # is a caller asserting their own access.
        if not context.verified:
            progress.step("rbac_gate", "blocked", "unverified caller")
            return Command(
                goto="respond",
                update={
                    **reset,
                    "target_agent_id": (context.requested_agent_id or "").strip(),
                    "rbac": {"allowed": False, "reason": "entitlements not verified"},
                    "outcome": "blocked",
                    "final_text": DENIED_MESSAGE,
                    "audit_trail": [
                        _entry(
                            "rbac_gate",
                            "deny",
                            "the entitlement block carries no valid gateway signature — "
                            "the request did not come through the Governance Front Door",
                        )
                    ],
                },
            )

        agent_id = (context.requested_agent_id or "").strip()
        permitted = context.permitted_agents
        agent = self.s.registry.get(agent_id)

        if not agent_id:
            allowed, reason = False, "no target agent id was supplied"
        elif agent is None:
            allowed, reason = False, f"unknown agent id '{agent_id}'"
        elif permitted is not None:
            allowed = agent_id in permitted
            reason = (
                f"'{agent_id}' is in the caller's permitted set"
                if allowed
                else f"'{agent_id}' is not in the caller's permitted set"
            )
        else:
            decision = self.s.rbac.check((context.user_role or "").strip(), agent_id)
            allowed, reason = decision.allowed, decision.reason

        if not allowed:
            progress.step("rbac_gate", "blocked", "not permitted")
            return Command(
                goto="respond",
                update={
                    **reset,
                    "target_agent_id": agent_id,
                    "rbac": {"allowed": False, "reason": reason},
                    "outcome": "blocked",
                    "final_text": DENIED_MESSAGE,
                    "audit_trail": [_entry("rbac_gate", "deny", reason)],
                },
            )

        # ── Inbound size bound (guardrail layer 1, defence in depth) ────────
        # The gateway's InvocationRequest caps `input` at 8000 characters, but
        # that cap lives in a different deployable — a direct endpoint caller
        # bypasses it, and everything downstream (window trimming, checkpoint
        # size, audit excerpts) is sized assuming it held. Checked after
        # authorization so an unauthorized caller still learns nothing, and
        # before any review/approval state is touched or model called.
        max_chars = self.s.settings.input_max_chars
        inbound = _latest_user_text(state.get("messages"))
        if max_chars > 0 and len(inbound) > max_chars:
            progress.step("rbac_gate", "blocked", "message too long")
            return Command(
                goto="respond",
                update={
                    **reset,
                    "target_agent_id": agent_id,
                    "rbac": {"allowed": True, "reason": reason},
                    "outcome": "blocked",
                    "final_text": INPUT_TOO_LONG_MESSAGE,
                    "audit_trail": [
                        _entry("rbac_gate", "allow", reason),
                        _entry(
                            "input_bounds",
                            "oversize",
                            f"{len(inbound)} chars exceeds the {max_chars}-char inbound "
                            "bound — refused before screening",
                        ),
                    ],
                },
            )

        # ── Subject spend allowance (cost control) ──────────────────────────
        # The caller's rolling governance-spend window (spend.py). Checked here
        # because here is the last point before any model call: the gateway's
        # rate limiter bounds *requests*, which a caller can satisfy while
        # driving spend an order of magnitude past a normal turn, and it lives
        # in a different deployable so a direct endpoint invocation bypasses it
        # entirely — the same defence-in-depth argument as the inbound size
        # bound above.
        #
        # After authorization, so an unauthorized caller cannot use quota
        # messages as an oracle for which agents exist; before the session and
        # review gates, because a caller with no allowance left should not have
        # their conversation expired or their review claimed as a side effect
        # of a turn that is not going to run.
        exceeded = ""
        try:
            exceeded = self.s.spend_window.check(context.user_key)
        except Exception:
            # Fails *open*, alone among the gate's checks, and deliberately:
            # this is a cost backstop, not a security control. The security
            # boundary is the RBAC decision above, which has already been made;
            # turning a bookkeeping error into a refused turn would trade a
            # real outage for a marginal cost saving. The platform-level hard
            # budget limit is the control that must not fail open.
            logger.warning("subject spend window check failed", exc_info=True)
        if exceeded:
            progress.step("rbac_gate", "blocked", "request allowance spent")
            logger.info(
                "subject allowance reached for %s: %s", context.user_key or "?", exceeded
            )
            return Command(
                goto="respond",
                update={
                    **reset,
                    "target_agent_id": agent_id,
                    "rbac": {"allowed": True, "reason": reason},
                    "outcome": "blocked",
                    "final_text": SUBJECT_ALLOWANCE_MESSAGE,
                    "audit_trail": [
                        _entry("rbac_gate", "allow", reason),
                        _entry(
                            "spend",
                            "subject_allowance",
                            f"{exceeded} — turn held before any model call",
                        ),
                    ],
                },
            )

        # ── Session lifetime (§05 Stage 04) ─────────────────────────────────
        # Checked after authorization, so an unauthorized caller learns nothing
        # about whether a conversation exists, and before the review and
        # approval gates, because an expired session must not keep either of
        # them open — that is precisely what the bound is for.
        expired = self._expire_if_stale(state, context, agent_id, reset, reason)
        if expired is not None:
            return expired

        # ── Terminal review state (§05 Stage 04) ────────────────────────────
        # An open appeal or escalation holds the conversation until a human
        # resolves it. Reads the conversation's own checkpointed marker first,
        # so a conversation with nothing open never queries the review queue and
        # an unreachable queue cannot affect it.
        gate_trail: list = []
        held = self._hold_for_review(state, context, agent_id, reset, reason, gate_trail)
        if held is not None:
            return held

        if state.get("pending_approval"):
            # A staged artifact is waiting for sign-off and this turn carried a
            # normal message, not an approve/reject decision. The session state
            # machine only accepts approve/reject/comment while a gate is open,
            # so the message is rejected with an explanation and the gate is
            # re-opened — never silently merged into the pending stage.
            if narrate:
                progress.step("rbac_gate", "done", "approval gate open")
            return Command(
                goto="approval",
                update={
                    **reset,
                    "target_agent_id": agent_id,
                    "rbac": {"allowed": True, "reason": reason},
                    "outcome": "approval_pending",
                    "final_text": APPROVAL_PENDING_MESSAGE,
                    "audit_trail": [
                        _entry("rbac_gate", "allow", reason),
                        *gate_trail,
                        _entry(
                            "approval_gate",
                            "rejected_input",
                            "message received while an approval gate is open",
                        ),
                    ],
                },
            )

        if narrate:
            progress.step("rbac_gate", "done")
        return Command(
            goto="guardrails",
            update={
                **reset,
                "target_agent_id": agent_id,
                "rbac": {"allowed": True, "reason": reason},
                "audit_trail": [_entry("rbac_gate", "allow", reason), *gate_trail],
            },
        )

    # ── 2. Guardrails ───────────────────────────────────────────────────────
    def guardrails(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["route", "respond"]]:
        context = runtime.context or SupervisorContext()
        narrate = _narrates(state)
        if narrate:
            progress.step("guardrails", "started")

        # ── Emergency stop (kill switch) ─────────────────────────────────────
        # A `kill_switch` in the governed guardrails document refuses every
        # turn here — before the appeal path, before any model call, before
        # any dispatch. It rides the same config publish/Reloading path as the
        # deny patterns, so engaging it is a table write that reaches a running
        # endpoint within the config cache TTL, not a redeploy. Checked via
        # getattr because tests stub the engine with objects that predate the
        # field. Staged approvals already interrupted are not swept up: a
        # reviewer rejecting a draft is not new work, and destroying the pause
        # would turn an operational hold into data loss.
        kill_message = getattr(self.s.guardrails, "kill_switch_message", "") or ""
        if kill_message:
            progress.step("guardrails", "blocked", "operational hold")
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

        # The agents this caller may reach, the one already addressed first, so
        # the usual case still costs a single model call. Nothing outside the
        # permitted set is ever a candidate — the RBAC gate's decision bounds
        # what the screen may even consider.
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
        # Checked before the screen, because "yes" screened as a request is an
        # off-domain message that would be refused — the same reason the appeal
        # path sits here.
        #
        # An accepted offer does not skip anything: the held text becomes this
        # turn's query and goes through the full screen below — the same tier-1
        # rules, the same semantic verdicts, the same RBAC-bounded candidate
        # list, its own routing and its own audit row. Holding a request is a
        # pause, never a pre-authorisation, so nothing about having asked first
        # makes the second task easier to get done than asking for it plainly.
        deferred = state.get("deferred_request") or {}
        deferred_resumed = ""
        if deferred.get("text"):
            answer = followup_answer(query or "")
            if answer == "accept":
                deferred_resumed = str(deferred["text"])
                query = deferred_resumed
                progress.step("guardrails", "started", "resuming the held request")
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
            # Anything else is a new request. The offer expires rather than
            # lingering: a held task that survives an unrelated turn is a task
            # the user has stopped expecting, and "yes" three messages later
            # would resume something they no longer have in front of them.

        # The appeal path out of a block (§06). Checked before the screen runs:
        # re-screening the *word* "appeal" would just block it again, which is
        # exactly the silent retry the requirement rules out. Only meaningful
        # when the previous turn was actually blocked, so an unprompted "appeal"
        # falls through to normal screening.
        if state.get("appealable") and _APPEAL_PATTERN.match(query or ""):
            appealed = state.get("appealable") or {}

            # The refusal recorded that no appeal was on offer. Answer the word
            # rather than screening it, for the same reason the branch exists at
            # all: screened, "appeal" is just an off-domain message, so it gets
            # blocked — and *that* block, having no safety flag of its own,
            # cheerfully offers the appeal the previous turn withheld. The user
            # is then one word away from queueing a review the control just
            # declined to open, and the reviewer gets a row whose excerpt reads
            # "appeal". Saying so plainly is the only non-looping answer.
            if not appealed.get("offered", True):
                progress.step("guardrails", "blocked", "nothing to appeal")
                return Command(
                    goto="respond",
                    update={
                        # No `**spent`: this branch runs before the screen, so
                        # the turn has made no model call to account for — the
                        # same reason the appeal branch below carries none.
                        "guardrail": {
                            "passed": False,
                            "tier": appealed.get("tier", ""),
                            "reason": appealed.get("reason", ""),
                            "appeal_offered": False,
                        },
                        "outcome": "blocked",
                        # The marker is *kept*, not cleared. Clearing it sends
                        # the next "appeal" to the screen, where it is an
                        # ordinary off-domain message — blocked, with the appeal
                        # line attached, which is the loop this branch exists to
                        # close, merely one turn further along. Observed live on
                        # v5 before this line changed: the third "appeal" in a
                        # row came back offering an appeal. It clears itself the
                        # moment the user asks something that passes.
                        "appealable": appealed,
                        "final_text": appealed.get("no_appeal_reason") or NOTHING_TO_APPEAL_MESSAGE,
                        "audit_trail": _trail(
                            state,
                            "guardrails",
                            "appeal_declined",
                            "the previous refusal offered no appeal path — "
                            "no review opened",
                        ),
                    },
                )

            blocked_reason = appealed.get("reason", "")
            # The appeal has to *land* before the user is told a human has it.
            # §08: "Do not report a governance decision as applied. Fail the
            # operation rather than complete it unaudited." Telling someone a
            # reviewer will follow up when nothing was written is the exact
            # failure the blueprint's "a pathway, not a promise" note names.
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
                progress.step("guardrails", "error", "appeal not recorded")
                return Command(
                    goto="respond",
                    update={
                        "outcome": "error",
                        "final_text": APPEAL_UNAVAILABLE_MESSAGE,
                        # `appealable` is deliberately kept: the appeal did not
                        # happen, so the user must still be able to take it.
                        # Clearing it here would consume their one route to a
                        # human on an error that was ours.
                        "audit_trail": _trail(
                            state,
                            "guardrails",
                            "appeal_failed",
                            f"appeal could not be recorded ({type(exc).__name__}) — "
                            "not reported to the user as flagged",
                        ),
                    },
                )

            progress.step("guardrails", "error", "appealed to a human")
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

        # `[:-1]` drops the current turn, which is not an inconsistency with the
        # route stage even though that one keeps it: the screen receives the turn
        # separately as `query`, so leaving it in history would send it twice,
        # while route has no separate query argument and the current turn *is*
        # the request it must resolve context for.
        #
        # A screen failure is a *transport* failure, not a verdict. The graph has
        # already retried it; §06 requires the query be held rather than passed
        # through, so the exception becomes a controlled hold with an audit row
        # — never an implicit pass, and never an unhandled 500 that leaves no
        # trail at all.
        # ── A reviewer granted this conversation a retry (§05 Stage 03) ──────
        # The half of the appeal control that makes it a control: the resolution
        # changes what happens next. `claim_allowance` spends the grant in one
        # conditional UPDATE, so a granted retry is used exactly once even if two
        # turns arrive together — it cannot become a standing exemption.
        #
        # Claimed only on the turn the RBAC gate actually observed a resolution
        # (`review_resolved`), never unconditionally: issuing the claim UPDATE
        # against the shared review table on 100% of traffic put a write
        # round-trip on every turn and contradicted review_queue.py's own
        # availability design — ordinary conversations must have no dependency
        # on the queue at all.
        allowance = None
        if state.get("review_resolved"):
            try:
                allowance = self.s.reviews.claim_allowance(state.get("conversation_id", ""))
            except ReviewQueueError as exc:
                # Fails *open* here, alone among the review paths, and for a
                # reason worth stating: not claiming an allowance leaves the
                # screen to run normally, which is the same verdict the user
                # would have got without ever appealing. Nothing is widened by
                # the failure.
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
                progress.step("guardrails", "done", "cleared by a reviewer")
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
        # Rebuilt from state at each node and written back with the node's
        # update, so the running total crosses the hop between nodes without a
        # second place to store it. This node is where a fan-out happens, so it
        # is where the ceiling earns its keep: `ensure` refuses to *start* a
        # screen the turn cannot afford, and the charge afterwards is the exact
        # number of verdicts `screen` reports having asked for. See spend.py.
        spend = TurnSpend.from_state(state, SpendCaps.from_settings(self.s.settings))
        window = _window(state.get("messages"), self.s.settings.history_max_tokens)
        per_verdict = (
            estimate_tokens(
                window, extra_chars=len(query or "") + PROMPT_OVERHEAD_CHARS
            )
            + VERDICT_OUTPUT_TOKENS
        )

        try:
            deadline.ensure("the guardrail screen")
            # One call, not the whole fan-out: the fan-out's own bound is the
            # deadline (it stops between candidates), and demanding the
            # worst-case cost up front would refuse a turn that the common
            # case — the first candidate claims the query — settles in one
            # call. The charge below is on what was actually spent.
            spend.ensure("the guardrail screen", model_calls=1, tokens=per_verdict)
            screened = self.s.guardrails.screen(
                query, candidates, history=_history_lines(window)[:-1], deadline=deadline
            )
        except BudgetExhausted as exc:
            return self._budget_exhausted(state, "guardrails", exc)
        except SpendExhausted as exc:
            return self._spend_exhausted(state, "guardrails", exc, spend)
        except Exception as exc:
            # Type and a bounded message only — LangChain/OpenAI client
            # exceptions can carry the whole request body (the system prompt
            # plus the user's query and context), and the process log has
            # weaker access controls and different retention than the audit
            # table. Full traceback at DEBUG for environments that opt in.
            logger.error(
                "guardrail screening failed for %s: %s: %s",
                state["target_agent_id"],
                type(exc).__name__,
                str(exc)[:200],
            )
            logger.debug("guardrail screening traceback", exc_info=True)
            progress.step("guardrails", "error", "checks unavailable")
            # Charged even though the screen failed. A call that raised may
            # still have billed — a 500 after the model generated, a timeout on
            # the read rather than the connect — and the conservative reading
            # is the one a cost control should take: otherwise a subject stuck
            # in a failure loop spends the endpoint for free, forever.
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

        # What the screen actually cost. `considered` is the agents it asked
        # about — one verdict each — so the call count is a fact rather than an
        # estimate. A deterministic block or small-talk classification short-
        # circuits before any candidate is asked and legitimately costs zero.
        verdicts = len(screened.considered)
        if verdicts:
            spend.charge("guardrails", model_calls=verdicts, tokens=per_verdict * verdicts)
        # Merged into every return below, so a block, a clarification and a
        # pass all carry the same accounting into the audit row. Omitting it on
        # one path would under-report exactly the turns worth investigating.
        spent = {"spend": spend.record()}

        if (
            not result.passed
            and getattr(result, "escalate", False)
            and not state.get("open_review")
        ):
            # ── A tier-1 rule published with `action: escalate` ─────────────
            # Refused exactly as a block is, and additionally handed to a
            # reviewer with the conversation held: the bulk-disclosure and
            # cross-tenant asks where a refusal alone under-reports the event.
            # If the review cannot be recorded, the turn degrades to the plain
            # block below — still refused, no reviewer promised (§08).
            #
            # Skipped when a review is already open on this conversation (an
            # appeal, which does not itself hold the turn): a second row would
            # overwrite the `open_review` marker and orphan the first. The
            # request is still refused by the block path below, which says so.
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
                progress.step("guardrails", "error", "escalated for review")
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
            # One block is a verdict; a streak of them is a signal. When one
            # conversation is blocked `guardrail_block_streak_limit` times in a
            # row, the supervisor stops answering each probe individually and
            # hands the conversation to a human — the same terminal review hold
            # the clarification cap uses. Deterministic and per-conversation:
            # a prober cannot dilute it by rephrasing (any block counts), and a
            # legitimate user who trips it gets a reviewer, not a dead end.
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
                    # Degrade to the plain block below rather than failing the
                    # turn: the user is still refused either way, and the
                    # preserved streak makes the next block retry the
                    # escalation. No promise of a reviewer is made (§08).
                    logger.warning("could not open a block-streak escalation: %s", exc)
                else:
                    progress.step("guardrails", "error", "escalated after repeated blocks")
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

            progress.step("guardrails", "blocked", result.reason)
            covered = join_names([a.name for a in reachable])

            # Whether this block gets an appeal path, and why not when it
            # doesn't. Two independent reasons to withhold it, both meaning "a
            # reviewer cannot change this outcome":
            #
            #   * the verdict is a safety refusal — nothing to overturn;
            #   * a review is already open on this conversation — a human is
            #     attached already, and a second appeal would overwrite the
            #     `open_review` marker and orphan the first row.
            #
            # Deterministic blocks stay appealable on purpose. `guardrails.yaml`
            # explicitly leans on the appeal as the recovery for a false
            # positive ("a false block costs the user their only route in"), and
            # a regex cannot tell a jailbreak from a user who wrote a sentence
            # that looks like one.
            already_reviewing = bool(state.get("open_review"))
            safety_refusal = getattr(result, "safety_refusal", False)
            offer_appeal = not safety_refusal and not already_reviewing

            if offer_appeal:
                closing = APPEAL_NOTE
            elif already_reviewing:
                closing = REVIEW_IN_PROGRESS_NOTE
            else:
                # A safety refusal closes with nothing. The refusal and what the
                # user *can* ask for is the whole message; inviting them to
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
                        # Recorded so the decision trail shows why the appeal was
                        # or wasn't offered — otherwise a reviewer auditing a
                        # refusal cannot tell a withheld appeal from a bug.
                        "safety_refusal": safety_refusal,
                        "appeal_offered": offer_appeal,
                    },
                    "outcome": "blocked",
                    "guardrail_block_streak": streak,
                    # Kept for the next turn so a reply of "appeal" is recognised
                    # as the appeal path rather than screened as a fresh query.
                    # It is kept when no appeal is on offer too, carrying
                    # `offered: False` — the word still has to be *recognised*
                    # there, or it falls through to the screen and the resulting
                    # off-domain block hands back the appeal this one withheld.
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
                    # One voice with the small-talk replies: "for your role I can
                    # reach the …". The old "Your role covers: X" read as a
                    # permissions dump appended to a refusal, and left the user
                    # to work out that it was also the way forward.
                    #
                    # The appeal line closes it (§06): a block has to offer a way
                    # forward that is not retrying the same query at the same
                    # guardrail, and the only such way is a human — where a human
                    # can help at all.
                    # The reason is model-written under a prompt untrusted
                    # content can try to steer, and it is shown word for word
                    # — so it passes the output guard's scrub first: secrets
                    # and identifiers masked, a leaked prompt line replaced.
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
                        + (f" (considered {', '.join(screened.considered)})" if screened.considered else "")
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
            # The supervisor answers this itself, as the supervisor. Sending a
            # greeting on to the worker is what produces a wall of capabilities
            # in reply to "hi", and it spends a worker call to do it.
            #
            # Every agent this caller can reach, not just the one this turn was
            # addressed to: a greeting has no topic, so there is nothing to
            # single one out with.
            # No progress event: this turn ran no gate worth showing, so it gets
            # no task plan. See `_narrates`.
            return Command(
                goto="respond",
                update={
                    **spent,
                    "guardrail": {
                        "passed": True,
                        "tier": result.tier,
                        "reason": result.reason,
                        # Which kind, not just that it was small talk — otherwise
                        # the audit record cannot tell a greeting from a
                        # thank-you, and the two get different replies.
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
        # Placed here on purpose: after the deny patterns, after the semantic
        # screen, and **before** the clarifying question.
        #
        # After the screen, because recognising a note must not be a way past a
        # guardrail. By this line the message has already been judged on its
        # content, so all this decides is whether to dispatch — see
        # session_notes.py for why that ordering is the whole safety argument.
        #
        # Before the clarification, because the screen's question is "which
        # deliverable do you want?" and the user has just said "not yet".
        # Asking it anyway answers a question they did not ask, which is the
        # milder version of the same mistake as dispatching.
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
                progress.step("guardrails", "done", result.tier)
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
                    # No worker ran. Saying so explicitly rather than leaving the
                    # previous turn's name in place, because the calling UI
                    # renders this as the answering agent and a note was
                    # answered by the supervisor.
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
            # The subject is this agent's, but no deliverable was named. Ask
            # rather than dispatch a guess or refuse: the request would be
            # accepted verbatim with three more words in it, so a refusal here is
            # a false negative that costs the user their only route in.
            #
            # Asked here rather than at `route` because the two questions are
            # about different things — this one is "what do you want produced?",
            # route's is "which product line?" — and asking for a product line
            # before knowing whether they want a user story is the wrong order.
            # `target_agent_id` moves with it so the answer lands on the agent
            # that asked.
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
                    # The screen recognised the subject as in-domain, so this is
                    # a real user mid-conversation, not a probe — the streak
                    # resets like any other pass.
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

        # The screen may have found the query belongs to a different agent the
        # caller can reach. Retargeting here is safe because every candidate came
        # from the permitted set the RBAC gate already established — the choice
        # cannot widen access, only move within it.
        retargeted = agent.id != state["target_agent_id"]
        trail = [
            _entry(
                "guardrails",
                "pass",
                f"{result.tier}: {result.reason}"
                + (f" (considered {', '.join(screened.considered)})" if screened.considered else ""),
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
            # Recorded even though the user was asked (above, via
            # `result.clarification`) — a contested screen is the interesting
            # kind for a reviewer looking at why a request went where it did,
            # and "two agents claimed it" is invisible in a trail that only
            # names the winner.
            trail.append(
                _entry(
                    "guardrails",
                    "contested",
                    "more than one permitted agent claimed the request: "
                    + ", ".join(result.contested),
                )
            )

        # A second deliverable in the same message. Carried on the guardrail
        # record rather than acted on here: it is offered back only once the
        # first request has actually produced something, so that a user whose
        # first task was refused is not immediately asked about their second.
        additional = (getattr(result, "additional_request", "") or "").strip()
        # Never offer back the request that is being answered right now. On a
        # resumed turn the held text *is* the query, so a screen that reports it
        # as "a second deliverable" would hold it again, offer it again, and the
        # conversation would loop on one request forever. Production models do
        # not do this — a request is not a second request to itself — but a
        # governance loop must not depend on that.
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
            progress.step("guardrails", "done", result.tier)
        return Command(
            goto="route",
            update={
                **spent,
                "target_agent_id": agent.id,
                # Consumed. Whether it was taken up or expired, the offer does
                # not survive the turn that answered it.
                "deferred_request": None,
                # The held request, restated as the user's own words, so the
                # transcript the worker and the reviewer read says what is being
                # worked on rather than "yes". Verbatim from the message they
                # sent — the supervisor re-raises it, it does not rewrite it.
                **(
                    {"messages": [HumanMessage(content=deferred_resumed)]}
                    if deferred_resumed
                    else {}
                ),
                # A later turn passed the screen, so the earlier block is no
                # longer the thing "appeal" would refer to. The block streak
                # resets with it — but only here and on the other *semantic*
                # passes, never on small talk: "hi" between probes must not
                # launder a streak back to zero.
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

    def _budget_exhausted(self, state: dict, stage: str, exc: BudgetExhausted) -> Command:
        """The turn ran out of its total time budget (§05 Stage 05).

        Routed to `respond` rather than raised, which is the whole reason the
        budget sits below the gateway's timeout: the user gets a governed
        message and the decision trail gets a row. Letting the gateway cut the
        connection instead produces neither, which is why an overrun was a
        compliance problem and not only a latency one.

        `outcome: "error"` rather than a new value, deliberately — the KPI
        report and the calling UI both already treat `error` as a controlled
        failure, and the specific cause is in the trail where an investigation
        will look for it.
        """
        logger.warning("turn budget exhausted at %s: %s", stage, exc)
        progress.step(stage, "error", "time budget exhausted")
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

        Shaped exactly like `_budget_exhausted` — routed to `respond` so the
        user gets a governed message and the trail gets a row — but kept as its
        own method and its own exception type because they are different
        findings. "This turn ran long" and "this turn spent its allowance" lead
        an operator to different settings and different conclusions, and a
        shared handler would leave the decision trail unable to say which one
        happened.

        `outcome: "error"` for the same reason as the time budget: the KPI
        report and the calling UI already treat `error` as a controlled failure,
        and inventing a value here would move a published KPI as a side effect
        of adding a control.
        """
        logger.warning("turn spend ceiling reached at %s: %s", stage, exc)
        progress.step(stage, "error", "spend ceiling reached")
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

        Two stages can now need something from the user: the screen, when it
        recognises the subject but not the deliverable, and the router, when
        required context is missing. They share one counter deliberately — the
        limit exists so a user cannot be asked questions forever, and a limit
        each would let two stages take turns and never reach it.
        """
        count = state.get("clarification_count", 0) + 1
        if count > self.s.settings.max_clarifications:
            context = context or SupervisorContext()
            # The escalation has to become queryable state, or "escalate to a
            # human reviewer" is a message rather than a control (§05 Stage 04,
            # and the same reasoning as the appeal path above). If it cannot be
            # recorded, the user is not told a human has it.
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
                progress.step(stage, "error", "escalation not recorded")
                return Command(
                    goto="respond",
                    update={
                        **update,
                        "outcome": "error",
                        "final_text": APPEAL_UNAVAILABLE_MESSAGE,
                        # The counter is *not* reset: the limit has not been
                        # honoured yet, and resetting here would silently give
                        # the user two more clarification loops as a reward for
                        # our storage failing.
                        "audit_trail": _trail(
                            state,
                            stage,
                            "escalate_failed",
                            f"escalation could not be recorded ({type(exc).__name__}) — "
                            "not reported to the user as flagged",
                        ),
                    },
                )

            progress.step(stage, "error", "escalated")
            return Command(
                goto="respond",
                update={
                    **update,
                    "outcome": "escalated",
                    "final_text": ESCALATION_MESSAGE,
                    "clarification_count": 0,
                    "pending_clarification": None,
                    # Terminal until a reviewer resolves it — the next turn is
                    # refused by `_hold_for_review` rather than restarting the
                    # pipeline as if nothing had happened.
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
            # A question was already open when this turn came in, so whatever
            # the user just said did not settle it — this is at least the
            # second ask in a row, not a fresh one.
            final_attempt = count >= self.s.settings.max_clarifications
            text = (CLARIFY_FINAL_PREFIX if final_attempt else CLARIFY_RETRY_PREFIX) + question

        progress.step(stage, "clarify", question or "")
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

    # ── 3. Route / clarify ──────────────────────────────────────────────────
    def route(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["dispatch", "respond"]]:
        context = runtime.context or SupervisorContext()
        progress.step("route", "started")
        deadline = deadline_for(context, self.s.settings)
        agent = self.s.registry.get(state["target_agent_id"])

        # Never the role. `user_key or user_role` used to stand in when no
        # pseudonymous reference arrived, which made the long-term memory key
        # the *role string*: every user invoking without a user_id shared one
        # context bag per role, so one BA's resolved product line was read back
        # into another BA's routing prompt — cross-user bleed on exactly the
        # direct-call path the trust boundary worries about, violating the BRD
        # NFR and GDPR Art. 5(1)(f). With no subject to key on, the store is
        # skipped entirely: `memory.get_context`/`save_context` already treat
        # an empty key as a no-op, and this conversation's own
        # `session_context` still works.
        user_key = context.user_key
        agent_keys = set(agent.required_context)

        # ── Session isolation (§4.4) ────────────────────────────────────────
        # Two sources of prior context, and the order between them is the whole
        # point. `session_context` is what *this* conversation resolved and had
        # dispatched on; the long-term store is what this user resolved most
        # recently anywhere.
        #
        # Long-term memory is keyed by user alone, so without this ordering a
        # second conversation about product line beta would overwrite the
        # stored value and the *first* conversation — still open, still about
        # alpha — would silently start producing beta artifacts on its next
        # turn. Nothing in the transcript would say so. Two concurrent product
        # lines is the ordinary case for the personas this serves, so that is a
        # correctness bug, not a corner.
        #
        # So: the conversation's own value wins, and the store is only ever a
        # seed for a conversation that has not resolved one yet. Both are
        # narrowed to the keys the target agent actually declared, so a
        # `environment` resolved with the Deployment Agent never reaches the
        # Requirement Agent's prompt.
        session_context = {
            key: value
            for key, value in (state.get("session_context") or {}).items()
            if key in agent_keys
        }

        remembered: dict = {}
        try:
            remembered = self.s.memory.get_context(user_key, keys=agent_keys)
        except Exception:
            logger.warning("long-term memory read failed", exc_info=True)

        prior = {**remembered, **session_context}
        seeded = {k: v for k, v in remembered.items() if k not in session_context}

        # Same fail-closed rule as the screen (§06): the graph has already
        # retried, and a routing model that cannot be reached must hold the
        # request rather than dispatch it with unresolved context.
        spend = TurnSpend.from_state(state, SpendCaps.from_settings(self.s.settings))
        window = _window(state.get("messages"), self.s.settings.history_max_tokens)
        # An agent with no `required_context` is resolved without a model call
        # at all (`Router.resolve` returns immediately), so this stage is free
        # for it — and a free path must not be refused by a spend ceiling or
        # charged for spend that did not happen. The condition mirrors the
        # router's own, rather than inferring cost from the result.
        resolves_by_model = bool(agent.required_context)
        route_tokens = (
            estimate_tokens(window, extra_chars=PROMPT_OVERHEAD_CHARS)
            + VERDICT_OUTPUT_TOKENS
            if resolves_by_model
            else 0
        )

        try:
            deadline.ensure("context resolution")
            if resolves_by_model:
                spend.ensure("context resolution", model_calls=1, tokens=route_tokens)
            result = self.s.router.resolve(
                agent,
                _history_lines(window),
                prior,
                deadline=deadline,
                # What this conversation never said. Named separately so the
                # router can weigh it differently — see `RouteDecision.
                # prior_context_applies`.
                carried_over=seeded,
            )
        except BudgetExhausted as exc:
            return self._budget_exhausted(state, "route", exc)
        except SpendExhausted as exc:
            return self._spend_exhausted(state, "route", exc, spend)
        except Exception as exc:
            # Scrubbed for the same reason as the guardrail path: the exception
            # can carry the prompt, and the prompt carries the user's data.
            logger.error(
                "context resolution failed for %s: %s: %s",
                agent.id,
                type(exc).__name__,
                str(exc)[:200],
            )
            logger.debug("context resolution traceback", exc_info=True)
            progress.step("route", "error", "checks unavailable")
            # Charged for the same reason as the screen's failure path above.
            if resolves_by_model:
                spend.charge("route", model_calls=1, tokens=route_tokens)
            return Command(
                goto="respond",
                update={
                    "spend": spend.record(),
                    "outcome": "error",
                    "final_text": GOVERNANCE_UNAVAILABLE_MESSAGE,
                    "audit_trail": _trail(
                        state,
                        "route",
                        "fail_closed",
                        f"context resolution unavailable ({type(exc).__name__}) — request held",
                    ),
                },
            )

        if resolves_by_model:
            spend.charge("route", model_calls=1, tokens=route_tokens)
        spent = {"spend": spend.record()}

        if result.ready:
            # §04: what long-term memory accepted, and what it refused, belongs
            # in the same decision trail as every other decision — a refused
            # field is either an allowlist that needs widening or an injection
            # attempt, and neither should be silent.
            memory_entries = []
            # Provenance, for the same reason. Context the user never typed in
            # this conversation still shapes the artifact the worker produces,
            # so which conversation it came from has to be reviewable — a
            # dispatch on a seeded product line should be visible in the trail,
            # not inferred from its absence.
            if session_context:
                memory_entries.append(
                    _entry(
                        "memory",
                        "session",
                        f"context this conversation already resolved: {session_context}",
                    )
                )
            if result.dropped_context:
                # The request moved on and the carried context did not describe
                # it any more. Recorded loudly: a dropped product line changes
                # which artifact gets produced, and "why did it stop using
                # alpha" is a question the trail has to be able to answer.
                memory_entries.append(
                    _entry(
                        "memory",
                        "dropped",
                        "carried context discarded — this request is about a different "
                        f"subject: {', '.join(result.dropped_context)}",
                    )
                )
            if seeded:
                memory_entries.append(
                    _entry(
                        "memory",
                        "seeded",
                        f"long-term context seeded into a new conversation: {seeded}",
                    )
                )
            try:
                write = self.s.memory.save_context(user_key, result.resolved_context)
                if write.stored:
                    memory_entries.append(
                        _entry("memory", "persisted", f"long-term context: {write.stored}")
                    )
                if write.rejected:
                    memory_entries.append(
                        _entry(
                            "memory",
                            "refused",
                            f"unvalidated long-term context refused: {write.rejected}",
                        )
                    )
            except Exception:
                logger.warning("long-term memory write failed", exc_info=True)
                memory_entries.append(_entry("memory", "error", "long-term memory write failed"))
            detail = (
                f"resolved context: {result.resolved_context}"
                if result.context_applies
                else "request does not depend on the agent's required context"
            )
            progress.step("route", "done", "" if result.context_applies else "not required here")
            # Pin what was actually dispatched on to the conversation. From here
            # this thread owns these values: a later turn reads them back above
            # in preference to the store, so a different conversation resolving
            # a different product line in the meantime cannot move this one.
            #
            # Pinned on dispatch only, not on the clarify path — a value the
            # user has not confirmed yet has not been used for anything, and
            # binding the conversation to a guess is what the clarifying
            # question exists to avoid.
            # Dropped keys do not come back. Re-pinning `session_context`
            # wholesale would put back exactly what the router just decided this
            # request is not about, and the conversation would go on carrying it
            # forever — the drop would show in the trail and change nothing.
            kept = {
                k: v for k, v in session_context.items() if k not in result.dropped_context
            }
            pinned = {
                **kept,
                **{k: v for k, v in result.resolved_context.items() if k in agent_keys},
            }
            # Context that came from another conversation and actually shaped
            # this dispatch. Told to the user, once, in the reply — see
            # `CARRIED_CONTEXT_NOTICE`. A value they cannot see in the
            # transcript in front of them, silently deciding what gets built,
            # is the whole complaint behind "why is this answering about my
            # other session".
            used_seed = {
                k: v
                for k, v in seeded.items()
                if k not in result.dropped_context and result.resolved_context.get(k) == v
            }
            if used_seed:
                memory_entries.append(
                    _entry(
                        "memory",
                        "carried",
                        "dispatched on context carried from an earlier conversation, "
                        f"and said so in the reply: {used_seed}",
                    )
                )

            return Command(
                goto="dispatch",
                update={
                    **spent,
                    "route": {
                        "next": "dispatch",
                        "resolved_context": result.resolved_context,
                        "carried_over": used_seed,
                        "dropped_context": list(result.dropped_context),
                    },
                    "session_context": pinned,
                    "clarification_count": 0,
                    "pending_clarification": None,
                    "audit_trail": _trail(state, "route", "dispatch", detail) + memory_entries,
                },
            )

        return self._clarify_or_escalate(
            state,
            "route",
            result.clarifying_question,
            {
                **spent,
                "route": {"next": "respond", "resolved_context": result.resolved_context},
            },
            context,
        )

    # ── 4. Dispatch ─────────────────────────────────────────────────────────
    def dispatch(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["respond", "approval"]]:
        context = runtime.context or SupervisorContext()
        deadline = deadline_for(context, self.s.settings)
        agent = self.s.registry.get(state["target_agent_id"])
        resolved = state.get("route", {}).get("resolved_context", {})
        progress.step("dispatch", "started", agent.name, agent=agent.name)

        # What the user asked to be kept in mind earlier in this conversation.
        # Prepended rather than merged into the history, so a note survives the
        # trimming below — which is the only thing this adds over the replayed
        # transcript, and the reason a note is worth a channel of its own.
        #
        # The worker is the *only* consumer: notes never reach the guardrail
        # screen or the router, so a note cannot change a governance decision,
        # only the content of a draft that every gate has already cleared.
        # session_notes.py argues that boundary in full.
        notes = session_notes.worker_message(state.get("session_notes"))

        # ── The relay boundary (guardrail layer 7, inbound half) ─────────────
        # What the worker receives is screened by the same policy as what it
        # returns: sensitive values masked, embedded directive frames
        # neutralised. A note the user asked to be kept in mind is the user's
        # own text and goes through the same screen. See `_worker_messages`.
        relay = _worker_messages(
            _window(state.get("messages"), self.s.settings.worker_history_max_tokens),
            guard=self.s.output_guard,
        )
        relay_masked = list(relay.masked)
        if notes:
            note_text, note_findings = self.s.output_guard.relay(notes["content"])
            notes = {**notes, "content": note_text}
            relay_masked.extend(f.label for f in note_findings)

        # The resolved context is model-lifted from the conversation, so it is
        # a second route to the worker for a value the relay screen just masked
        # out of the messages: the router reads the same untrusted text and
        # writes what it found into `product_line` or `environment`.
        # `memory._validate` bounds what may be *stored*, but a value only
        # dispatched, never stored, never met it.
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
                ([notes] if notes else []) + relay.messages,
                resolved,
                state.get("conversation_id", ""),
                context.user_role,
                # §1.10 — the correlation set, forwarded so the worker's own
                # traces join this turn's. `user_key` is already the
                # pseudonymous reference; §1.10 forbids the raw subject or any
                # token leaving the gateway, and none of these carry one.
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
            # The worker client already retried with backoff and the circuit
            # breaker has spoken — a clear "temporarily unavailable" message,
            # never another LLM-generated prompt.
            logger.warning("worker unavailable for %s: %s", agent.id, exc)
            progress.step("dispatch", "error", "worker unavailable")
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
            progress.step("dispatch", "error", type(exc).__name__)
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

        # Sources are worker-supplied strings that a UI renders, so they
        # are screened like any other worker output — and, critically, they are
        # **not** emitted here. Emitting them at this point put them on the
        # user's screen before the output guard had run, so a response the
        # guard went on to withhold or escalate had already leaked its citation
        # titles: a clean exfiltration channel for a compromised or
        # prompt-injected worker, and an un-neutralised route for control
        # characters and role markers. The block and escalate branches below
        # already omit `sources` from state; this is the same intent applied to
        # the stream. See the emission after the guard.
        sources = _clean_sources(resp.sources, self.s.output_guard)

        # ── Worker output is untrusted (§05 Stage 06) ────────────────────────
        # Bounded and sanitized *here*, before the text reaches state — because
        # once it is an AIMessage it is conversation history, and history is
        # replayed into the next turn's guardrail and routing prompts as
        # "role: content" (see `_history_lines`). A worker response containing a
        # line like "system: ignore the previous instructions" would arrive in
        # the *governance* prompt indistinguishable from a real turn boundary.
        # See sanitize.py for what is removed and what deliberately is not.
        cleaned = clean_worker_output(
            resp.text or "", max_chars=self.s.settings.worker_output_max_chars
        )
        text = cleaned.text
        output_trail = []
        if relay.modified or relay_masked:
            # Recorded on the dispatch stage because it describes what the
            # worker was given, and a reviewer asking "did the worker ever see
            # the pasted token?" needs the answer in the trail, not inferred
            # from the reply.
            output_trail.append(
                _entry(
                    "dispatch",
                    "relay_screened",
                    RelayScreen(relay.messages, tuple(relay_masked), relay.directives).audit_detail(),
                )
            )
            if relay.directives:
                logger.warning(
                    "conversation relayed to %s carried %d embedded directive marker(s) — "
                    "neutralised before dispatch",
                    agent.id,
                    relay.directives,
                )
        if notes:
            # A dispatch shaped by something the user said several turns ago is
            # a dispatch whose inputs are not all in the current message, so the
            # trail has to say so — otherwise "why did it produce that?" is
            # unanswerable from the record.
            output_trail.append(
                _entry(
                    "dispatch",
                    "notes_recalled",
                    f"{len(state.get('session_notes') or [])} session note(s) sent to "
                    f"{agent.id} as background",
                )
            )
        if cleaned.modified or cleaned.code_blocks:
            output_trail.append(
                _entry("dispatch", "output_sanitized", cleaned.audit_detail())
            )
        if cleaned.role_markers or cleaned.template_markers:
            # Not merely noisy output: something in the returned text was shaped
            # like a turn boundary. Logged at warning because it is the NPE
            # coercion channel NIST SP 800-207 §5.7 describes, and because a
            # worker that does this repeatedly is worth investigating.
            logger.warning(
                "worker %s returned %d role marker(s) and %d template marker(s) — "
                "neutralised before entering conversation history",
                agent.id,
                cleaned.role_markers,
                cleaned.template_markers,
            )

        # ── Output guard (guardrail layer 7) ─────────────────────────────────
        # The last check before worker text becomes user-visible state. Runs
        # before the approval branch on purpose: a staged artifact is shown to
        # the approver, so it goes through the same screen as a delivered
        # answer. A policy match withholds the response with a governed message
        # (the matched rule's reason goes to the trail, not to the user);
        # secret shapes are masked either way, so a credential a worker echoed
        # back never reaches the browser or the checkpoint. See output_guard.py
        # for the streaming caveat.
        guarded = self.s.output_guard.screen(text)
        # getattr: a test may stub the guard with a result that predates the
        # escalate tier, and a stub must degrade to "block", never to "deliver".
        #
        # `open_review` short-circuits for the same reason the guardrails node
        # does: a second review row would overwrite the marker of the one a
        # human already has. The response is still withheld by the block path.
        if getattr(guarded, "escalate", False) and not state.get("open_review"):
            # The escalate tier: withheld *and* handed to a human, with the
            # conversation held until they respond. Reached by a bulk
            # disclosure, a leaked canary or protected prompt text, a private
            # key, or any rule/category the governed document sets to
            # `escalate`. If the review cannot be recorded the response is
            # still withheld — degrading to the plain block below rather than
            # delivering — and no reviewer is promised (§08).
            logger.warning(
                "worker %s response escalated by the output guard: %s", agent.id, guarded.reason
            )
            try:
                review = self.s.reviews.open_review(
                    kind=review_queue.ESCALATION,
                    conversation_id=state.get("conversation_id", ""),
                    reason=f"output guard: {guarded.reason}",
                    user_key=context.user_key,
                    user_role=context.user_role,
                    target_agent_id=agent.id,
                    request_id=state.get("request_id", ""),
                    correlation_id=context.correlation_id,
                    query_excerpt=_excerpt(_latest_user_text(state.get("messages"))),
                )
            except (ReviewQueueError, ValueError) as exc:
                logger.warning("could not open an output-guard escalation: %s", exc)
            else:
                progress.step("dispatch", "error", "response escalated for review")
                return Command(
                    goto="respond",
                    update={
                        "pending_approval": None,
                        "worker_response": {"status": resp.status, "stage": resp.stage},
                        "outcome": "escalated",
                        "final_text": OUTPUT_ESCALATED_MESSAGE,
                        "appealable": None,
                        "open_review": {"ref": review.ref, "kind": review.kind},
                        "audit_trail": _trail(
                            state,
                            "output_guard",
                            "escalated",
                            f"{guarded.audit_detail()} — escalation {review.ref} opened, "
                            "conversation held for review",
                        )
                        + output_trail,
                    },
                )
        if guarded.blocked:
            progress.step("dispatch", "error", "response withheld")
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
                    "final_text": " ".join((OUTPUT_WITHHELD_MESSAGE, APPEAL_NOTE)),
                    # Appealable like an input-tier block: a false-positive
                    # output rule needs a human way forward too (§06).
                    "appealable": {"reason": guarded.reason, "tier": "output_policy"},
                    "audit_trail": _trail(
                        state, "output_guard", "withheld", guarded.audit_detail()
                    )
                    + output_trail,
                },
            )
        text = guarded.text
        if guarded.modified:
            output_trail.append(_entry("output_guard", "masked", guarded.audit_detail()))

        # ── Grounding cue (layer 7, hallucination checks) ────────────────────
        # The supervisor cannot verify a worker's claim against evidence it
        # does not hold; what it can do is refuse to let "the rollback
        # completed" or "per ticket PAY-1234" pass as verified when nothing in
        # the turn verified them. A footnote, never a rewrite. See grounding.py.
        if text and self.s.settings.output_provenance_notes:
            grounded = grounding.check(text, sources, resp.raw)
            if grounded.flagged:
                text = grounding.annotate(text, grounded)
                output_trail.append(_entry("grounding", "unverified", grounded.audit_detail()))

        # Only now, with the response cleared by the guard, do the citations
        # reach the calling UI — a withheld or escalated reply shows none, which is
        # what the block branches' omission of `sources` from state always
        # intended.
        if sources:
            progress.sources(sources)

        # ── Supervisor-enforced approval (Solution §04) ──────────────────────
        # The gate used to open only when a worker volunteered
        # `custom_outputs.hitl.status == "pending_approval"`, which made "
        # irreversible actions sit behind the human-in-the-loop approval gate"
        # a promise resting on worker cooperation: a worker that forgot the
        # flag, was misconfigured or was compromised simply returned an answer.
        # A registry `approval_patterns` match now stages the response too, on
        # the supervisor's own authority and through governed configuration.
        #
        # Evaluated on the user's request rather than the worker's reply,
        # because what makes an action reviewable is what was asked for — and
        # the reply is the thing being reviewed. Checked after the output guard
        # so a staged artifact has been screened exactly like a delivered one.
        mandated = self.s.registry.approval_reason(
            agent, _latest_user_text(state.get("messages"))
        )

        if resp.status == "approval_pending" or mandated:
            # Hand the pause to its own node. The worker call above is a side
            # effect, and everything before an `interrupt()` re-runs when the
            # graph resumes — keeping them in one node would invoke the worker
            # a second time on every approval.
            #
            # `stage` falls back to a generic label when the supervisor is the
            # one requiring the gate: the worker did not declare a stage, and
            # inventing one ("HLD") would put a claim about the artifact's place
            # in a sequence into a record a human signs.
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
            progress.step("dispatch", "done", f"awaiting approval: {stage}")
            return Command(
                goto="approval",
                update={
                    "pending_approval": {
                        "agent_id": agent.id,
                        "agent_name": agent.name,
                        "stage": stage,
                        "artifact": text,
                        "context": resolved,
                        # Shown to the approver so the reason for the gate
                        # travels with the artifact rather than living only in
                        # the audit row.
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
            progress.step("dispatch", "error", "empty response")
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
        # After the approval branch, deliberately: this is a message *to the
        # user about their own message*, not part of the artifact a human signs
        # off. A staged draft carries the grounding cue (an approver needs to
        # know a claim is unverified) and nothing else that was not produced by
        # the worker.
        if relay_masked:
            text = (
                text.rstrip()
                + "\n\n"
                + SENSITIVE_INPUT_NOTICE.format(what=sensitive_prose(relay_masked))
            )

        # Context this conversation never stated, carried in from another one.
        # Same placement reasoning as the masking notice above: it is a message
        # to the user about their own request, not part of the artifact a human
        # signs off, so it sits after the approval branch.
        carried_over = (state.get("route") or {}).get("carried_over") or {}
        if carried_over:
            text = text.rstrip() + CARRIED_CONTEXT_NOTICE.format(
                what=context_prose(carried_over)
            )

        # ── The second task in the same message ──────────────────────────────
        # Offered here and nowhere earlier, because this is the first point at
        # which the first task has actually produced something. Offering it
        # alongside a refusal or a clarifying question would be asking the user
        # about their second request while their first is still unanswered.
        #
        # Also after the approval branch: a staged artifact is mid-flight, and
        # a draft awaiting sign-off is the wrong moment to start a second piece
        # of work. The offer is simply not made on that path — the second
        # request is dropped rather than held, because holding it across a
        # human review would surface it long after the user had moved on.
        deferred_update: dict = {}
        additional = ((state.get("guardrail") or {}).get("additional_request") or "").strip()
        if additional:
            # Screened deterministically *before* it is offered. Offering to do
            # something the tier-1 rules refuse — and only refusing once the
            # user says yes — is a worse experience than a straight answer now,
            # and it costs a regex sweep rather than a model call to avoid.
            # The semantic screen and RBAC still run in full if they accept.
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

        progress.step("dispatch", "done")
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

    # ── 4b. Human approval ──────────────────────────────────────────────────
    def approval(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["respond"]]:
        """Suspend until a human accepts or rejects the staged artifact.

        Deliberately side-effect free before the `interrupt()`. LangGraph
        re-runs a node from the top when it resumes, so anything above the
        pause happens twice — which is why the worker call lives in `dispatch`
        and this node only reads state.
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
                # Why the gate opened, when the supervisor required it rather
                # than the worker asking. Empty for a worker-declared stage.
                # ASI09 (human-agent trust exploitation) turns on the approver
                # seeing the real reason for a decision they are being asked to
                # make, not a summary the producing agent chose.
                "required_because": pending.get("required_because", ""),
                "risk_level": pending.get("risk_level", ""),
            }
        )

        approved = decision is True or (
            isinstance(decision, dict) and decision.get("decision") == "approved"
        )
        comment = decision.get("comment", "") if isinstance(decision, dict) else ""

        # ── Who approved what, and when (§05 Stage 04 callout) ───────────────
        # The blueprint scopes the Supervisor's approval duty narrowly and names
        # both halves: "keep the session alive and uncorrupted across the
        # approval turns, **and record who approved what and when**."
        #
        # This matters beyond bookkeeping. §09's segregation-of-duties ruling
        # offers self-approval as an option *"with a recorded approver
        # identity"*, naming the audit trail as the compensating control — and a
        # compensating control that cannot say who approved does not compensate.
        # Before this, the approver was only implicit in the enclosing audit
        # row's `user_key`, recoverable but never recorded as an approval.
        #
        # `user_key` is already the pseudonymous reference the gateway derived;
        # §1.10 forbids the raw subject leaving the gateway, so this is the
        # strongest attributable identity available here and it is stable enough
        # to answer "who approved this" against the identity provider.
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
            # Refused, not merely logged. §09 offers self-approval as
            # acceptable only *"with a recorded approver identity"*, naming the
            # audit trail as the compensating control — and a compensating
            # control that records "unattributed" does not compensate. An
            # approval nobody can be held to is not a governance decision that
            # can be recorded, so the artifact stays staged and the gate stays
            # open for a caller who *is* attributable. (A rejection below still
            # goes through: declining to act is always available, and an
            # unattributed rejection widens nothing.)
            logger.warning(
                "approval decision for '%s' stage '%s' carries no approver identity — "
                "refused. The Platform API always supplies one, so the graph was "
                "invoked outside the gateway.",
                owner_id,
                stage,
            )
            progress.step("dispatch", "blocked", "approval not attributable")
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

        # An approval is bound to the agent that produced the artifact, which
        # is known from the checkpoint — not to whatever id the request was
        # addressed with. The check therefore happens here, against the real
        # owner. Rejecting needs no permission: declining to act is always
        # available.
        if approved and not self._may_approve(context, owner_id):
            progress.step("dispatch", "blocked", "approval not permitted")
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
            progress.step("dispatch", "done", "rejected")
            return Command(
                goto="respond",
                update={
                    "pending_approval": None,
                    # Stays "answer" on purpose. A rejection is a governance
                    # decision and is audited as one — but via `signoff`, not by
                    # inventing an outcome value: a routing-completion metric
                    # is computed from `outcome == "answer"`, and a
                    # signed-off artifact *is* a completed routing. Changing the
                    # vocabulary here would silently move a published KPI.
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

        progress.step("dispatch", "done", "approved")
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

    # ── 5. Respond & audit ──────────────────────────────────────────────────
    def respond(self, state: dict, runtime: Runtime[SupervisorContext]) -> dict:
        context = runtime.context or SupervisorContext()
        text = state.get("final_text") or GENERIC_ERROR
        routed = self.s.registry.get(state.get("target_agent_id") or "")

        deadline = deadline_for(context, self.s.settings)
        signoff = state.get("signoff") or {}

        # ── Redaction at the persistence boundary (Blueprint §06) ────────────
        # The decision trail carries the model's paraphrase of the query, the
        # resolved context, block reasons and clarification questions — all
        # query-derived, all about to be copied into the Postgres trail AND the
        # MLflow trace metadata (TracingAuditLogger writes this same record to
        # both). Secrets and personal data engineers paste into queries are
        # scrubbed here, once, before any sink sees them; the live conversation
        # is untouched. What was redacted is itself recorded — a query that
        # carried credentials is signal.
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
        # Deliberately **not** decision-trail entries. The trail is the ordered
        # sequence of governance decisions — §05 Stage 06 enumerates what it
        # persists, "role, decision, outcome, latency, and the reason for any
        # block" — and readers rely on that: reporting selects entries by
        # stage, and the last entry is how a turn's deciding stage is
        # identified. Spend and prompt provenance are metadata *about* the turn,
        # not decisions taken during it, and `latency_ms`/`session_age_seconds`
        # already set the precedent that turn metadata gets a column.
        spend = TurnSpend.from_state(state)
        if spend.model_calls or spend.tokens:
            # The caller's rolling allowance is charged here, once, with the
            # turn's final figures — not per stage. Charging mid-turn would let
            # a turn refuse itself halfway through, spending the calls and
            # delivering nothing.
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
            # §05 Stage 06 lists latency among what the decision trail must
            # persist — "role, decision, outcome, **latency**, and the reason
            # for any block". It was the one field missing, which left the
            # latency KPI with no durable per-request source and only MLflow
            # spans, whose retention is shorter than the audit table's.
            "latency_ms": round(deadline.spent() * 1000) if deadline.enabled else None,
            # §05 Stage 04's session bound needs a Duration KPI to be evidence
            # rather than a setting. This is that column.
            "session_age_seconds": _session_age_seconds(state),
            # Who signed off what, when — present only on an approval turn.
            "signoff": signoff or None,
            # v1.1 §08's token KPI, as durable columns rather than only trace
            # spans, whose retention is shorter than this table's — the same
            # argument that put `latency_ms` here. `model_calls` is exact (one
            # governance verdict, one call); `tokens_estimated` is named for
            # what it is — see spend.py on why enforcement rests on the former.
            "model_calls": spend.model_calls or None,
            "tokens_estimated": spend.tokens or None,
            # How the turn was decided: which model, which prompt versions,
            # which stage spent what. JSONB like `signoff`, so the shape can
            # grow without a migration per field.
            "provenance": _provenance(self.s.settings, spend) or None,
        }

        # ── Synchronous audit for governance decisions (§05 Stage 06, §08) ───
        # "A config change, an appeal resolution or a deletion is not reported
        # as applied unless its audit record durably landed", and §08 repeats it:
        # "Fail the operation rather than complete it unaudited."
        #
        # Applied to decisions, not to answers. A question that fails because
        # the audit sink blinked is an outage in exchange for a record of a
        # decision that restricted nobody; a block whose record silently
        # vanished is a refusal that cannot be proved to have happened. So the
        # doctrine is enforced where it protects something and relaxed where it
        # would only cost availability — stated here rather than left implicit,
        # because a partial application of a failsafe that reads as total is
        # exactly the overclaiming the blueprint's §01 is about.
        governed = _is_governance_decision(state)
        try:
            self.s.audit.log(record)
        except Exception:
            logger.warning("audit write failed", exc_info=True)
            # The row itself, at ERROR, so a sink hiccup degrades to "recover
            # the rows from the process log" instead of silently under-counting
            # BR-006's usage and clarification metrics. Same shape as
            # `LoggingAuditLogger`, greppable by the same token.
            try:
                import json as _json

                logger.error("supervisor-audit-fallback %s", _json.dumps(record, default=str))
            except Exception:
                logger.error("supervisor-audit-fallback could not serialise the record")
            if governed:
                progress.step("respond", "error", "decision not recorded")
                return {
                    "messages": [AIMessage(content=AUDIT_UNAVAILABLE_MESSAGE)],
                    "outcome": "error",
                    "final_text": AUDIT_UNAVAILABLE_MESSAGE,
                    "routed_agent_name": "",
                }

        # A directly-answered turn shows no plan, so it gets no closing tick
        # either — but a refusal always narrates, and "Finishing up" is what
        # completes that plan.
        if _narrates(state) or state.get("outcome") != "answer":
            progress.step("respond", "done", state.get("outcome", ""))
        return {
            "messages": [AIMessage(content=text)],
            # Attribute the answer to a worker only when one actually produced
            # it. A denial, a guardrail block, a clarifying question and a
            # greeting the supervisor answered itself all reach here without any
            # worker call, and naming one would credit work that never happened.
            "routed_agent_name": (
                (routed.name if routed else "") if state.get("worker_response") else ""
            ),
        }
