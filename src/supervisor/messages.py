"""Every sentence the supervisor says in its own voice, and the helpers that
compose them.

Kept apart from the pipeline in `nodes.py` on purpose: what a refusal, a hold,
an acknowledgement or a small-talk reply *says* is reviewed by different people
than the control flow that decides *when* to say it, and a wording change must
not be a logic change. Each constant carries the reasoning behind its wording,
because several of them are the user-visible half of a governance control
(an appeal offered or withheld, a hold that must not read as a refusal).
"""

from __future__ import annotations

GENERIC_ERROR = (
    "Something went wrong while handling your request. Please try again, and contact "
    "support if the problem persists."
)
# A second turn arrived on a conversation whose previous turn is still running
# (`agent_governance.locking`). Refused rather than run concurrently, and worded
# so the user knows nothing was dropped.
BUSY_MESSAGE = (
    "Another message on this conversation is still being processed. Please wait for it "
    "to finish before sending the next one — nothing you sent has been lost."
)
ESCALATION_MESSAGE = (
    "I wasn't able to pin down the details needed to route your request, so I've "
    "flagged this conversation for a human reviewer. They will follow up with you."
)
# The code failsafe for a worker timeout or tool error: after retry-with-backoff
# and the circuit breaker have both given up, the user gets a clear
# "temporarily unavailable" message rather than another LLM-generated prompt.
WORKER_UNAVAILABLE_MESSAGE = (
    "The agent is temporarily unavailable. Please try again in a few minutes — "
    "your conversation has been kept."
)
# The session state machine only accepts approve / reject / comment while an
# approval gate is open. An out-of-turn message is rejected with an explanation,
# never silently merged into the pending stage.
APPROVAL_PENDING_MESSAGE = (
    "A staged draft is still waiting for a decision. Please approve or reject it "
    "first — your message has not been sent to the agent."
)
# The guardrail/routing model is unreachable after the graph's retries. §06 is
# explicit that this fails *closed*: the query is held rather than allowed
# through, and the user is told to retry — it is never treated as an implicit
# pass. Worded as a hold, not a refusal, because nothing was judged: saying
# "I can't help with that" would attribute a governance decision to a
# transport failure.
GOVERNANCE_UNAVAILABLE_MESSAGE = (
    "I couldn't complete the safety and routing checks on your request just now, so "
    "I've held it rather than passing it on. Nothing was sent to an agent — please "
    "try again in a moment."
)
# §06: a block "states the scope limitation clearly and offers an appeal path to
# a human/admin queue — not a silent retry against the same guardrail". Repeating
# the query verbatim would only hit the same verdict, so the way forward has to
# be a human, and the user has to be told that it is available.
APPEAL_NOTE = (
    "If you believe this is in scope, reply with **appeal** and I'll flag it for a "
    "human reviewer."
)
# ...but only where a reviewer could actually change the answer. A block whose
# verdict carries `safety_refusal` is a refusal of the request itself, not a
# ruling about which agent owns it, and no reviewer is going to authorise
# weapons or malware. Offering the appeal there is the "pathway, not a promise"
# failure in its purest form — it invites the user to queue work for a human
# whose only possible answer is no, and it puts a request for harm in front of
# that human as though it were a scope dispute. Same reasoning that already
# withholds the appeal from the spend ceiling (`SUBJECT_ALLOWANCE_MESSAGE`).
#
# This is a presentation decision, never a blocking one: the request is refused
# identically either way. See `guardrails.GuardrailVerdict.safety_refusal`.
#
# The taxonomy is standard — Ai2's noncompliance work separates "safety
# concerns", which warrant explicit refusal, from capability and scope
# categories, which warrant a constructive way forward. This system's appeal
# path is that way forward, and it belongs to the second group only.
#
# A second appeal is likewise not offered while one is already open: the
# conversation already has a human attached, and a second row would overwrite
# the `open_review` marker and orphan the first.
REVIEW_IN_PROGRESS_NOTE = (
    "A reviewer is already looking at an earlier request from this conversation, so "
    "I haven't raised a second one."
)
# What a bare "appeal" gets when the previous refusal offered none. Without
# this the word falls through to the screen, comes back as an ordinary
# off-domain block, and *that* block offers the appeal the safety refusal had
# just withheld — a one-word route around the control, and a review row whose
# excerpt reads "appeal". Stored on the `appealable` marker so the answer can
# name the actual reason rather than guessing at it.
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

    Says three things, because leaving any of them out is what would make the
    acknowledgement misleading: what was recorded, that it is scoped to this
    conversation and will not follow the user out of it, and that the work has
    *not* started. The last matters most — the failure being fixed is a turn
    that quietly did the work instead.

    The note is echoed back so the user can see it was read correctly, shortened
    for display only; the stored note keeps its own (larger) bound.
    """
    shown = note if len(note) <= 160 else note[:160].rstrip() + "…"
    count = (
        ""
        if held <= 1
        else f" That's {held} things I'm holding for this conversation."
    )
    return (
        f"Noted for this conversation — I haven't started on it: *{shown}*.{count} "
        "Say the word when you want to pick it up. It stays with this conversation "
        "only; a new one starts with a clean slate."
    )
# The user took that appeal path. This is the human/admin queue entry point: the
# turn is recorded as an appeal in the decision trail, which is what an admin
# reviews — the supervisor does not re-run the guardrail against the same query
# and hope for a different answer.
APPEAL_ACKNOWLEDGED_MESSAGE = (
    "I've flagged your last request for a human reviewer, with the reason it was "
    "blocked. They'll follow up with you — you don't need to resend it."
)
# The appeal could not be *recorded*, so it must not be reported as flagged
# (§08). Distinguished from the acknowledgement above because telling a user a
# human will follow up when nothing was written is the specific failure the
# blueprint's "make the appeal a pathway, not a promise" note is about.
APPEAL_UNAVAILABLE_MESSAGE = (
    "I couldn't record your appeal just now, so I haven't flagged it — nothing has "
    "been lost, but please try again in a moment so it reaches a reviewer."
)
# An **escalation** is open on this conversation. §05 Stage 04 requires the
# escalated state be "genuinely terminal until the review resolves", so a new
# message is refused rather than quietly restarting the pipeline.
#
# Reached only for `review_queue.ESCALATION` — the system's own response to a
# conversation that has gone wrong (a block streak that looks like probing, a
# clarification loop that will not converge). An open *appeal* deliberately does
# not come here: see `_hold_for_review`.
REVIEW_PENDING_MESSAGE = (
    "This conversation is with a human reviewer and is paused until they respond. "
    "Your message hasn't been sent to an agent — please start a new conversation if "
    "you need something else in the meantime."
)
# The conversation's own review state says a review is open, but the queue that
# would say whether it has been resolved cannot be reached. Fails closed: the
# alternative is letting an escalated conversation continue unreviewed whenever
# the store blinks. Only reachable for conversations already under review.
REVIEW_UNAVAILABLE_MESSAGE = (
    "This conversation is paused for human review and I can't confirm its status "
    "right now, so I've held your message rather than acting on it. Please try again "
    "shortly."
)
# The reviewer resolved an appeal in the user's favour. The next turn gets one
# pass through the screen — this is the "outcome changes what the system does
# next" half of the appeal control.
REVIEW_ALLOWED_NOTE = (
    "A reviewer has looked at your appeal and agreed this is in scope, so I've gone "
    "ahead with it."
)
# The turn's total time budget ran out (§05 Stage 05). Worded as a hold, like
# the governance-unavailable message: nothing was judged, so nothing should read
# as a refusal.
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
# Layer 1 in-graph size bound. The gateway already refuses oversize input with a
# 422; this is the defence-in-depth twin for callers that reached the endpoint
# directly, worded as a bound rather than a judgment — nothing was screened.
INPUT_TOO_LONG_MESSAGE = (
    "Your message is longer than I can accept in one turn. Please shorten it or "
    "split it into smaller requests — nothing was sent to an agent."
)
# The turn reached its governance spend ceiling (spend.py). Worded as a hold
# like the time budget and for the same reason: nothing was judged, so nothing
# should read as a refusal. It names narrowing the request because that is the
# one thing the user can actually do — a turn spends per candidate agent, and a
# request specific enough to be claimed by the first one costs a single call.
SPEND_EXHAUSTED_MESSAGE = (
    "Your request needed more checking than I'm allowed to spend on one turn, so "
    "I've stopped rather than leaving it running. Nothing was sent to an agent — "
    "please try again with a more specific request."
)
# The caller's own rolling allowance is spent (spend.py `SubjectWindow`). A
# quota message, not a refusal of the request: the same request will be
# accepted once the window rolls forward, and saying so is what separates this
# from a guardrail block. No appeal path is offered — there is nothing for a
# reviewer to overturn, and an appeal that a reviewer cannot action is the
# "pathway, not a promise" failure the appeal control exists to avoid.
SUBJECT_ALLOWANCE_MESSAGE = (
    "You've used the request allowance for this period, so I've held this one "
    "rather than running it. Nothing was sent to an agent — please try again "
    "shortly, or contact your administrator if you need a larger allowance."
)
# Layer 7 output policy. The worker produced a response, but it matched an
# output rule in the governed guardrails document, so it is withheld rather
# than delivered. The matched rule's reason goes to the decision trail, never
# to the user — repeating it here would tell a prober exactly which rule fired.
OUTPUT_WITHHELD_MESSAGE = (
    "The agent produced a response, but it didn't pass the output checks, so I "
    "haven't delivered it. The event has been recorded for review."
)
# The output guard's escalate tier: the response was withheld *and* the
# conversation handed to a human — a bulk disclosure, a leaked canary, a
# private key in the reply. Terminal until the reviewer resolves it, like the
# other escalations, and for the same reason: continuing as if nothing had
# happened is how a data dump becomes two.
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
# Appended to a reply when the same message asked for a second, separate
# deliverable. Offered rather than done: the second half of a compound request
# is still a request, and running it unasked is an action nobody authorised.
# Offered rather than dropped, because a silently half-answered message is the
# failure users actually notice.
DEFERRED_REQUEST_OFFER = (
    "\n\nYou also asked me to {what}. I haven't started that — shall I go ahead "
    "with it now? Reply **yes** and I'll take it as the next request, or **no** "
    "to leave it."
)
# The second request would be refused by a tier-1 rule, so it is never offered.
# Saying so here, at the moment the user can still ask why, is better than
# offering it and refusing once they say yes.
DEFERRED_REQUEST_REFUSED = (
    "\n\n_You also asked me to {what}. I can't take that one on: {reason}._"
)
# The user said yes to the offer. The held request re-enters the pipeline as a
# fresh turn — full screen, full RBAC, its own routing — so this only narrates
# what is about to be judged, and promises nothing about the outcome.
DEFERRED_REQUEST_RESUMED = "Picking up the request you held over: {what}"
# The user said no.
DEFERRED_REQUEST_DROPPED = (
    "Understood — I've left that one. Ask again whenever you want it."
)
# Appended once when a dispatch used context the user resolved in a DIFFERENT
# conversation. Long-term memory exists so nobody retypes their product line
# every session, and that is worth keeping — but a value the user cannot see in
# the transcript in front of them, silently deciding which artifact gets built,
# is how an answer ends up belonging to the previous conversation. Stating it
# costs one line and makes it correctable in the next message.
CARRIED_CONTEXT_NOTICE = (
    "\n\n_I carried {what} over from an earlier conversation of yours. Say so if "
    "this one is about something else and I'll re-ask._"
)
# Appended to a reply when the conversation relayed to the worker carried
# values the guard masked on the way *in*. The user pasted them, so the user
# is told — silently swapping a placeholder into their script would leave them
# wondering why the agent never used the credential they supplied, and the
# thing they most need to hear is to rotate it.
SENSITIVE_INPUT_NOTICE = (
    "_Note: your message contained {what}, which I masked before sending it to the "
    "agent — the agent worked from placeholders. Please don't paste live credentials "
    "or personal data into this chat; if a credential was real, rotate it._"
)
# Catalogue label → what to call it in that sentence. The labels are built for
# a decision trail ("[redacted:url-credentials]"), and reading one back at a
# user mid-sentence produces "your message contained person". Anything not
# listed falls back to the label with its hyphens opened out, which reads
# acceptably for the shapes whose names are already plain English.
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
    """"a name, an email address and a phone number" — for the notice above."""
    named = list(
        dict.fromkeys(
            _SENSITIVE_LABEL_PROSE.get(label, label.replace("-", " ")) for label in labels
        )
    )
    if len(named) == 1:
        return named[0]
    return ", ".join(named[:-1]) + f" and {named[-1]}"


def context_prose(carried: dict) -> str:
    """"the product line (alpha)" — for the carried-context notice.

    Keys are registry-declared (`required_context`), so they are safe to show;
    values come from the long-term store, whose allowlist admits validated
    identifiers only. Both are the user's own, said back to them.
    """
    parts = [f"{key.replace('_', ' ')} ({value})" for key, value in sorted(carried.items())]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


# The supervisor's own answer to a question *about* the agents — whether one
# exists, what it can reach, what it holds. Generic on purpose, and the same
# whether the thing asked about exists or not: confirming is a disclosure,
# denying is an oracle. Naming the reachable set is not a leak — every
# refusal already does, and it is the user's own entitlement.
META_AGENTS_REPLY = (
    "I don't share details about which agents exist, how they're configured, or what "
    "they can access or hold — those aren't mine to disclose, whatever the answer "
    "would be.{options} If you have a software task, tell me what you need and I'll "
    "route it."
)
# Layer 6 anomaly signal: this conversation hit the guardrail block streak
# limit, so instead of refusing again the supervisor hands the conversation to
# a human. Deliberately does not say "you look like you're probing" — the same
# streak is produced by a genuinely stuck user, and the reviewer decides which.
BLOCK_STREAK_ESCALATION_MESSAGE = (
    "Several requests in a row have been outside what I can route for you, so I've "
    "paused this conversation and asked a human reviewer to take a look. They will "
    "follow up with you."
)

# A clarifying question that comes round again is not the first one repeated:
# the reply that came back did not resolve it, and the guardrail/route models
# have no memory of their own previous phrasing, so left alone they reliably
# regenerate the same question (same missing information in, same question
# out). Repeating it verbatim reads as not having read the reply — the same
# principle behind `_SMALL_TALK_REPLIES` above, applied here as a two-step
# escalation instead of a fixed set of variants, since the question text
# itself is generated per-stage and can't be pre-written.
CLARIFY_RETRY_PREFIX = "That doesn't tell me yet — "
CLARIFY_FINAL_PREFIX = "Last check before I bring in a human reviewer — "


def sentence(text: str) -> str:
    """Make a fragment stand as its own sentence.

    A block reason comes from two places with different habits: a bundled rule
    ("the request looks like a prompt-injection attempt" — lowercase, no full
    stop) or a model verdict ("The query asks about a cooking recipe…" — a whole
    sentence). The refusal concatenates whichever it got with the role line, so
    both ends need normalising: capitalise the first letter, and close it.

    Only ever upgrades a leading lowercase letter, so a reason opening on a
    proper noun or an acronym ("HLD generation is…") is left alone.
    """
    text = (text or "").strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


# One reply per kind, then what to say when the same kind comes round again.
#
# Repeating a line verbatim is the thing that makes an assistant read as broken —
# established conversation-design guidance is not to reuse a prompt word for word
# on a second attempt, because the conversational context is no longer the same
# even though the user's input is. So each kind escalates rather than repeats:
#
#   1st  answer the thing that was actually asked, hand the turn back
#   2nd  shorter, and add options — the agents this role can reach
#   3rd+ stop making conversation and ask for a real task
#
# Written out rather than generated: small talk costs no model call today, and a
# model would not improve four fixed lines.
# `{options}` is filled on the second pass only, and sits mid-sentence so the
# reply still *ends* on the question that hands the turn back.
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
    # A closing gets no question back. Every other kind hands the turn to the
    # user because they opened one; a farewell is them handing it back, and
    # "What are you working on?" would reopen a conversation they just ended.
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

    Answered **as the supervisor**, naming no particular worker: small talk
    carries no topic, so there is nothing to say which specialist it belongs to.
    Claiming one would be an invention, and with more than one agent in reach it
    would be an arbitrary pick. Where agents are named it is always *all* of the
    ones the caller can reach, never a pick among them.

    `seen` is how many earlier turns in this conversation were the same kind, so
    the second "hi" does not get the first one's sentence back.
    """
    listed = join_names(reachable)

    if kind == "meta_agents":
        # A question about the agents' existence, access or configuration —
        # the one small-talk kind that is a *refusal to disclose* rather than
        # a pleasantry. Identical on every repeat: there is nothing to vary,
        # and a second phrasing would invite a third attempt.
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
    # Past the last variant, hold on it rather than cycling back to the opener —
    # a fourth "hi" answered with "Hello, I'm the Supervisor" would restart the
    # conversation the user is already three turns into.
    reply = variants[min(seen, len(variants) - 1)]

    # The second time round, add options. Naming the reachable agents is the
    # honest form of "here are examples" — inventing sample queries would put
    # words in an agent's mouth about what it accepts.
    return reply.format(options=f" — for your role I can reach {listed}" if listed else "")


def join_names(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return f"the {names[0]}"
    return "the " + ", the ".join(names[:-1]) + f" or the {names[-1]}"
