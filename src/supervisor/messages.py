"""Everything the user is shown: the supervisor's own sentences, and the streamed
progress checklist.

Kept apart from `nodes/` on purpose: wording is reviewed by different people than
the control flow, and a wording change must not be a logic change. Several strings
are the user-visible half of a governance control, so each carries the reasoning
behind its wording.
"""

from __future__ import annotations

import logging

GENERIC_ERROR = (
    "Something went wrong while handling your request. Please try again, and contact "
    "support if the problem persists."
)
# A second turn arrived while the previous one holds the conversation lock
# (`agent_governance.locking`). Refused, not run concurrently; nothing reads as lost.
BUSY_MESSAGE = (
    "Another message on this conversation is still being processed. Please wait for it "
    "to finish before sending the next one — nothing you sent has been lost."
)
ESCALATION_MESSAGE = (
    "I wasn't able to pin down the details needed to route your request, so I've "
    "flagged this conversation for a human reviewer. They will follow up with you."
)
# Code failsafe after retry-with-backoff and the circuit breaker both gave up: a
# fixed "temporarily unavailable", never another LLM-generated prompt.
WORKER_UNAVAILABLE_MESSAGE = (
    "The agent is temporarily unavailable. Please try again in a few minutes — "
    "your conversation has been kept."
)
# Only approve / reject / comment are accepted while a gate is open; anything
# else is refused with an explanation, never merged into the pending stage.
APPROVAL_PENDING_MESSAGE = (
    "A staged draft is still waiting for a decision. Please approve or reject it "
    "first — your message has not been sent to the agent."
)
# §06: an unreachable guardrail model fails *closed* — held, never an implicit
# pass. Worded as a hold, not a refusal: nothing was judged, and "I can't help
# with that" would attribute a governance decision to a transport failure.
GOVERNANCE_UNAVAILABLE_MESSAGE = (
    "I couldn't complete the safety and routing checks on your request just now, so "
    "I've held it rather than passing it on. Nothing was sent to an agent — please "
    "try again in a moment."
)
# §06: a block "offers an appeal path to a human/admin queue — not a silent
# retry against the same guardrail". The way forward has to be a human.
APPEAL_NOTE = (
    "If you believe this is in scope, reply with **appeal** and I'll flag it for a human reviewer."
)
# Offered only where a reviewer could change the answer: a `safety_refusal`
# block gets none (no reviewer authorises malware). Presentation only — the
# refusal is identical. A second appeal is not offered while one is open.
REVIEW_IN_PROGRESS_NOTE = (
    "A reviewer is already looking at an earlier request from this conversation, so "
    "I haven't raised a second one."
)
# A bare "appeal" after a refusal that offered none. Without this it is screened
# as a fresh off-domain block — which *does* offer the appeal the safety refusal
# withheld: a one-word route around the control.
NOTHING_TO_APPEAL_MESSAGE = (
    "There's nothing for a reviewer to overturn here — that request isn't something "
    "I can pass to an agent whatever a reviewer decided. If you have a software "
    "question, ask it and I'll pick it up from there."
)
NOTHING_TO_APPEAL_REVIEW_OPEN_MESSAGE = (
    "A reviewer is already looking at an earlier request from this conversation, so "
    "there's nothing further to raise — they'll follow up. In the meantime you can "
    "carry on asking me other questions."
)


def noted_reply(note: str, held: int) -> str:
    """The acknowledgement for a "keep this in mind" turn.

    Must say what was recorded, that it is conversation-scoped, and that the work
    has *not* started — the failure being fixed is a turn that quietly did it.
    """
    shown = note if len(note) <= 160 else note[:160].rstrip() + "…"
    count = "" if held <= 1 else f" That's {held} things I'm holding for this conversation."
    return (
        f"Noted for this conversation — I haven't started on it: *{shown}*.{count} "
        "Say the word when you want to pick it up. It stays with this conversation "
        "only; a new one starts with a clean slate."
    )


# The human/admin queue entry point: the turn is recorded as an appeal in the
# decision trail, which is what an admin reviews. The guardrail is not re-run.
APPEAL_ACKNOWLEDGED_MESSAGE = (
    "I've flagged your last request for a human reviewer, with the reason it was "
    "blocked. They'll follow up with you — you don't need to resend it."
)
# §08: an appeal that could not be *recorded* is not reported as flagged —
# "pathway, not a promise".
APPEAL_UNAVAILABLE_MESSAGE = (
    "I couldn't record your appeal just now, so I haven't flagged it — nothing has "
    "been lost, but please try again in a moment so it reaches a reviewer."
)
# An escalation is open. §05 Stage 04: escalated state is "genuinely terminal
# until the review resolves", so the turn is refused rather than restarting the
# pipeline. `review_queue.ESCALATION` only; an open appeal goes to `_hold_for_review`.
REVIEW_PENDING_MESSAGE = (
    "This conversation is with a human reviewer and is paused until they respond. "
    "Your message hasn't been sent to an agent — please start a new conversation if "
    "you need something else in the meantime."
)
# A review is open but the queue cannot be reached. Fails closed: otherwise an
# escalated conversation continues unreviewed whenever the store blinks.
REVIEW_UNAVAILABLE_MESSAGE = (
    "This conversation is paused for human review and I can't confirm its status "
    "right now, so I've held your message rather than acting on it. Please try again "
    "shortly."
)
# Turn time budget exhausted (§05 Stage 05). Worded as a hold: nothing was
# judged, so nothing should read as a refusal.
BUDGET_EXHAUSTED_MESSAGE = (
    "Your request took longer than the time I'm allowed to spend on one turn, so "
    "I've stopped rather than leaving it running. Nothing was sent to an agent — "
    "please try again, and consider narrowing the request."
)
# §05 Stage 04: an abandoned session must not hold an approval or resolved
# context open indefinitely (GDPR Art. 5(1)(e)).
SESSION_EXPIRED_MESSAGE = (
    "This conversation sat inactive longer than the retention window allows, so its "
    "saved context and any draft awaiting sign-off have been cleared. Please start "
    "again — I've kept nothing from it."
)
# The decision trail for a governance decision did not land, so the decision is
# not reported as applied (§05 Stage 06, §08).
AUDIT_UNAVAILABLE_MESSAGE = (
    "I reached a decision on your request but couldn't record it, and I don't report "
    "a governance decision that isn't recorded. Nothing was sent to an agent — please "
    "try again in a moment."
)
# Layer 1 in-graph size bound, the defence-in-depth twin of the gateway's 422.
# Worded as a bound, not a judgment — nothing was screened.
INPUT_TOO_LONG_MESSAGE = (
    "Your message is longer than I can accept in one turn. Please shorten it or "
    "split it into smaller requests — nothing was sent to an agent."
)
# Governance spend ceiling reached (spend.py). A hold, like the time budget.
# Narrowing is named because a turn spends per candidate agent.
SPEND_EXHAUSTED_MESSAGE = (
    "Your request needed more checking than I'm allowed to spend on one turn, so "
    "I've stopped rather than leaving it running. Nothing was sent to an agent — "
    "please try again with a more specific request."
)
# Rolling per-subject allowance spent (spend.py `SubjectWindow`). A quota, not a
# refusal — saying the request is accepted later separates it from a block.
SUBJECT_ALLOWANCE_MESSAGE = (
    "You've used the request allowance for this period, so I've held this one "
    "rather than running it. Nothing was sent to an agent — please try again "
    "shortly, or contact your administrator if you need a larger allowance."
)
# Layer 7 output policy. The matched rule's reason goes to the decision trail,
# never to the user — it would tell a prober exactly which rule fired.
OUTPUT_WITHHELD_MESSAGE = (
    "The agent produced a response, but it didn't pass the output checks, so I "
    "haven't delivered it. The event has been recorded for review."
)
# Output guard escalate tier: withheld *and* handed to a human. Terminal until
# the reviewer resolves it — continuing is how one data dump becomes two.
OUTPUT_ESCALATED_MESSAGE = (
    "The agent produced a response that the output checks flagged for human review, "
    "so I haven't delivered it and I've paused this conversation until a reviewer "
    "has looked at it. They will follow up with you."
)
# A tier-1 rule published with `action: escalate` fired. Refused like a block,
# held like an escalation; no appeal offered because a human already has it.
DETERMINISTIC_ESCALATION_MESSAGE = (
    "I can't route this request, and because of what it asks for I've handed this "
    "conversation to a human reviewer rather than simply declining. It's paused until "
    "they respond — they will follow up with you."
)
# Second deliverable in one message. Offered, not done: running it unasked is an
# unauthorised action; dropping it is the failure users notice.
DEFERRED_REQUEST_OFFER = (
    "\n\nYou also asked me to {what}. I haven't started that — shall I go ahead "
    "with it now? Reply **yes** and I'll take it as the next request, or **no** "
    "to leave it."
)
# The second request would hit a tier-1 rule, so it is never offered — better
# said now than after the user says yes.
DEFERRED_REQUEST_REFUSED = "\n\n_You also asked me to {what}. I can't take that one on: {reason}._"
# The user said no.
DEFERRED_REQUEST_DROPPED = "Understood — I've left that one. Ask again whenever you want it."
# Dispatch used context resolved in a DIFFERENT conversation. A value the user
# cannot see in the transcript must be disclosed once, so it is correctable.
CARRIED_CONTEXT_NOTICE = (
    "\n\n_I carried {what} over from an earlier conversation of yours. Say so if "
    "this one is about something else and I'll re-ask._"
)
# The relayed conversation carried values the guard masked on the way in. The
# user pasted them, so the user is told — chiefly to rotate the credential.
SENSITIVE_INPUT_NOTICE = (
    "_Note: your message contained {what}, which I masked before sending it to the "
    "agent — the agent worked from placeholders. Please don't paste live credentials "
    "or personal data into this chat; if a credential was real, rotate it._"
)
# Catalogue label → mid-sentence phrase. Labels are built for the decision trail;
# read back verbatim they produce "your message contained person".
_SENSITIVE_LABEL_PROSE = {
    "person": "a name",
    "email": "an email address",
    "phone": "a phone number",
    "date-of-birth": "a date of birth",
    "us-ssn": "a social security number",
    "uk-nino": "a National Insurance number",
    "passport-number": "a passport number",
    "card-number": "a payment card number",
    "card-cvv": "a card security code",
    "card-expiry": "a card expiry date",
    "card-fragment": "part of a payment card number",
    "iban": "a bank account number (IBAN)",
    "bank-account": "a bank account number",
    "nhs-number": "an NHS number",
    "medical-record-number": "a medical record number",
    "patient-record-id": "a patient record identifier",
    "health-condition": "health information",
    "medication": "medication details",
    "url-credentials": "credentials inside a URL",
    "secret-assignment": "a secret value",
    "prose-secret": "a secret value",
    "private-key": "private key material",
    "internal-hostname": "an internal hostname",
    "private-ip": "an internal IP address",
    "server-path": "a server file path",
}


def sensitive_prose(labels) -> str:
    """ "a name, an email address and a phone number" — for the notice above."""
    named = list(
        dict.fromkeys(
            _SENSITIVE_LABEL_PROSE.get(label, label.replace("-", " ")) for label in labels
        )
    )
    if len(named) == 1:
        return named[0]
    return ", ".join(named[:-1]) + f" and {named[-1]}"


def context_prose(carried: dict) -> str:
    """ "the product line (alpha)" — for the carried-context notice.

    Safe to show: keys are registry-declared, values passed the long-term store's
    validated-identifier allowlist, and both are the user's own.
    """
    parts = [f"{key.replace('_', ' ')} ({value})" for key, value in sorted(carried.items())]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


# Generic on purpose, and identical whether the asked-about agent exists or not:
# confirming is a disclosure, denying is an oracle. Naming the reachable set is fine.
META_AGENTS_REPLY = (
    "I don't share details about which agents exist, how they're configured, or what "
    "they can access or hold — those aren't mine to disclose, whatever the answer "
    "would be.{options} If you have a software task, tell me what you need and I'll "
    "route it."
)
# Layer 6 anomaly: the block streak limit hands the conversation to a human.
# Deliberately not "you look like you're probing" — a stuck user makes the same streak.
BLOCK_STREAK_ESCALATION_MESSAGE = (
    "Several requests in a row have been outside what I can route for you, so I've "
    "paused this conversation and asked a human reviewer to take a look. They will "
    "follow up with you."
)

# A clarifying question that comes round again was not resolved; the models
# regenerate the same text, and repeating it verbatim reads as not having listened.
CLARIFY_RETRY_PREFIX = "That doesn't tell me yet — "
CLARIFY_FINAL_PREFIX = "Last check before I bring in a human reviewer — "


def sentence(text: str) -> str:
    """Make a fragment stand as its own sentence.

    Block reasons arrive as lowercase unpunctuated fragments (bundled rules) or
    whole sentences (model verdicts). Only a leading lowercase letter is upgraded.
    """
    text = (text or "").strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


# One reply per kind, escalating on repeats (answer; shorter + options; ask for a
# real task) — a verbatim repeat is what makes an assistant read as broken.
# `{options}` fills on the second pass only, mid-sentence, so each reply ends on a question.
_SMALL_TALK_REPLIES: dict[str, tuple[str, ...]] = {
    "how_are_you": (
        "Doing well, thanks for asking — ready when you are. What are you working on?",
        "Still running fine{options}. What would you like to look at?",
        "All good here. Give me a concrete task and I'll route it to the right specialist.",
    ),
    "greeting": (
        "Hello — I'm the Supervisor. Tell me what you need and I'll route it to the right "
        "specialist agent.",
        "Hello again{options}. What are you working on?",
        "Still here. Ask me something concrete and I'll route it to the right specialist.",
    ),
    "thanks": (
        "You're welcome. Anything else you'd like me to look at?",
        "Any time{options}. What's next?",
        "Happy to help — ask whenever you're ready.",
    ),
    # A closing gets no question back: a farewell is the user handing the turn
    # back, and "What are you working on?" would reopen what they just ended.
    "farewell": (
        "Goodbye — I'll be here whenever you need something routed.",
        "See you. Your conversation history stays here for next time.",
        "Take care.",
    ),
    "ack": (
        "Noted. What would you like to work on?",
        "Got it{options}. What's next?",
        "Understood — ready when you are.",
    ),
}


def small_talk_reply(kind: str, reachable: list[str], seen: int = 0) -> str:
    """The supervisor's own answer to small talk, varied by how often it recurs.

    Answered as the supervisor, naming no single worker — small talk has no topic,
    so picking one would be invention. `seen` = earlier turns of the same kind.
    """
    listed = join_names(reachable)

    if kind == "meta_agents":
        # A refusal to disclose, not a pleasantry. Identical on every repeat: a
        # second phrasing would invite a third attempt.
        return META_AGENTS_REPLY.format(
            options=f" For your role I can reach {listed}." if listed else ""
        )

    if kind == "meta":
        # A direct question about what this is. The substance has to survive every
        # retelling — only the framing shortens.
        opening = (
            "I'm the Supervisor: I check your access, screen your request, then route it to "
            "the right specialist and relay the answer back."
            if seen == 0
            else "As before: I check your access, screen the request, and route it."
        )
        return " ".join(
            part
            for part in (
                opening,
                f"For your role that's {listed}." if listed else "",
                "What do you need?",
            )
            if part
        )

    variants = _SMALL_TALK_REPLIES.get(kind) or _SMALL_TALK_REPLIES["greeting"]
    # Past the last variant, hold rather than cycle: a fourth "hi" answered with
    # the introduction restarts a conversation the user is three turns into.
    reply = variants[min(seen, len(variants) - 1)]

    # Options on the second pass. Naming reachable agents is the honest form of
    # "examples" — invented sample queries put words in an agent's mouth.
    return reply.format(options=f" — for your role I can reach {listed}" if listed else "")


def join_names(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return f"the {names[0]}"
    return "the " + ", the ".join(names[:-1]) + f" or the {names[-1]}"


# ---------------------------------------------------------------------------
# The streamed task plan — LangGraph's custom stream channel
# (`stream_mode=["custom", "messages"]`). `get_stream_writer()` returns None when
# nothing is streaming, so the same node code works unchanged under `invoke()`.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Stage -> the label a user should see. Deliberately plain language: this is
# shown in the chat, not in an operations console.
PROGRESS_LABELS: dict[str, str] = {
    "rbac_gate": "Checking your access",
    "guardrails": "Screening the request",
    "route": "Resolving the details",
    "dispatch": "Drafting the answer",
    "respond": "Finishing up",
}


def _stream_writer():
    """LangGraph's custom-channel writer for the running graph, or None.

    None under plain `invoke()` or when the lookup fails. Progress is additive,
    so its absence must never affect the answer.
    """
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:
        return None


def progress(stage: str, status: str, detail: str = "", **extra) -> None:
    """Emit one progress event.

    `status`: started | done | blocked | clarify | error.
    """
    writer = _stream_writer()
    if writer is None:
        return
    try:
        writer(
            {
                "channel": "progress",
                "stage": stage,
                "label": PROGRESS_LABELS.get(stage, stage.replace("_", " ").capitalize()),
                "status": status,
                "detail": detail,
                **extra,
            }
        )
    except Exception:
        logger.debug("progress emit failed for %s/%s", stage, status, exc_info=True)


def progress_sources(items: list[dict]) -> None:
    """Emit the grounding sources the calling UI shows under the answer.

    Items are `{"title", "origin"}`; the simulated worker's are clearly labelled.
    """
    if not items:
        return
    writer = _stream_writer()
    if writer is None:
        return
    try:
        writer({"channel": "sources", "items": items})
    except Exception:
        logger.debug("sources emit failed", exc_info=True)
