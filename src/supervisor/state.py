"""LangGraph state for the supervisor pipeline.

What lives here is what must survive a turn: the conversation, the resolved
routing decision, and the per-turn stage results.

What does **not** live here is identity — persona, entitlements, correlation.
Those are runtime context (`context.SupervisorContext`), because state is
checkpointed and §4.4 forbids persisting authentication context alongside a
conversation. See the module docstring there for why that matters.
"""

from __future__ import annotations

from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SupervisorState(TypedDict, total=False):
    # ── Conversation, accumulated across turns by the checkpointer ──────────
    messages: Annotated[list, add_messages]
    conversation_id: str
    request_id: str

    # ── The target agent, fixed by the invocation path each turn ────────────
    # Copied from the request's agent-id by the RBAC gate after validation.
    target_agent_id: str

    # ── Per-turn stage results, reset at the RBAC gate ──────────────────────
    rbac: dict
    guardrail: dict
    route: dict
    worker_response: Optional[dict]
    # answer | blocked | clarify | escalated | approval_pending | error
    # | review_pending | expired
    outcome: str
    final_text: str
    # Who approved or rejected a staged artifact, and when — §05 Stage 04's
    # "record who approved what and when". Present only on a turn that carried
    # an approval decision.
    #
    # Deliberately a separate channel rather than a new `outcome` value: an
    # approved artifact is still a completed routing, and a routing-completion
    # metric is computed from `outcome == "answer"`. Adding
    # "approved" to the vocabulary would have moved a published KPI as a side
    # effect of improving an audit record.
    signoff: Optional[dict]
    routed_agent_name: str
    sources: list
    audit_trail: list
    # What this turn spent on the supervisor's *own* governance model calls —
    # `{"model_calls": n, "tokens": n, "by_stage": {...}}`. See spend.py.
    #
    # In state rather than runtime context, unlike the time budget, and for the
    # opposite reason: a resumed approval turn must get a fresh *clock* (hours
    # passed while a human decided) but spend already made is spend already
    # made. Reset per turn at the RBAC gate with the other stage results, so it
    # measures one turn and never accumulates across a conversation.
    spend: dict

    # ── Carried across turns by the checkpointer ────────────────────────────
    # Context this conversation resolved and dispatched on — its product line,
    # its environment. Checkpointed, so it is scoped to this thread and this
    # thread only, which is what makes two concurrent conversations on
    # different product lines independent (§4.4 session isolation).
    #
    # It takes precedence over the per-user long-term store on every later turn:
    # the store is keyed by user, so it is shared between a user's open
    # conversations and would otherwise let the most recent one silently retarget
    # all the others. See `nodes.route`.
    session_context: dict
    # Things the user asked to be kept in mind — "…, keep this in mind for
    # later". `[{"text": ..., "at": ..., "request_id": ...}]`, oldest first.
    #
    # Short-term memory on purpose, and checkpointed for the same reason
    # `session_context` is: a note belongs to the conversation it was made in,
    # and it must die with the session rather than follow the user into the
    # next one. Nothing here is ever written to the long-term store, whose
    # allowlist admits validated identifiers and would refuse a sentence.
    #
    # Reaches worker dispatches and no governance model — see session_notes.py
    # for why that boundary is the security design rather than a detail.
    session_notes: list
    # A SECOND deliverable found in one message, held rather than done.
    # `{"text": ..., "asked_at": request_id, "agent_hint": ...}`.
    #
    # Held, not dispatched, because doing the second half of a compound request
    # without being asked is an unrequested action — the user may have been
    # thinking aloud, may want the first result before deciding, or may not have
    # meant the two to run together at all. Dropping it silently is the other
    # failure, and it is the one users notice. So it is offered back, and the
    # offer expires the moment the reply is anything but yes or no.
    #
    # Checkpointed with the conversation and scoped to it, like
    # `session_context` and `session_notes`: an offer belongs to the exchange
    # that produced it and must not follow the user into the next conversation.
    deferred_request: Optional[dict]
    clarification_count: int
    pending_clarification: Optional[str]
    # The last guardrail block, kept so a reply of "appeal" is recognised as the
    # appeal path (§06) instead of being screened as a fresh query — which would
    # block it again, the silent retry the requirement rules out. Cleared when
    # the appeal is taken, and whenever a later turn passes the screen.
    appealable: Optional[dict]
    # Consecutive guardrail blocks in this conversation — the runtime anomaly
    # signal (layer 6). Incremented on every block, reset to zero by any turn
    # that passes the screen; when it reaches the configured limit the
    # conversation is escalated to a human reviewer instead of refused again,
    # because N identical refusals in a row is either probing or a stuck user,
    # and both are reviewer work. Checkpointed so the streak survives across
    # turns but stays scoped to this one conversation.
    guardrail_block_streak: int
    # The staged artifact awaiting human sign-off. Set when a worker pauses;
    # cleared when the human resumes. The pause itself is a LangGraph
    # `interrupt()`, so the graph genuinely suspends rather than returning and
    # being re-entered on the next turn.
    pending_approval: Optional[dict]

    # Set by the RBAC gate for exactly one turn when it releases a review hold
    # because the reviewer resolved it — {"ref": ..., "kind": ...}. It is what
    # lets `guardrails` claim a granted retry allowance *only* on the turn a
    # resolution was actually observed, instead of issuing the claim UPDATE
    # against the shared review table on 100% of traffic (which contradicted
    # review_queue.py's own off-the-hot-path availability design). Reset at the
    # gate every turn, so it can never linger.
    review_resolved: Optional[dict]

    # ── Review state (§05 Stage 03/04) ──────────────────────────────────────
    # The reference of the appeal or escalation this conversation is waiting on,
    # e.g. {"ref": "rev_…", "kind": "escalation"}. Set when a review is opened,
    # cleared when it resolves.
    #
    # Checkpointed on purpose, and it is what makes the escalation state
    # "genuinely terminal until the review resolves" without putting the review
    # queue on the request path for everyone: a turn consults the queue only
    # when *this* field says something is open. Conversations with nothing open
    # never touch it, so an unreachable queue fails closed exactly for the
    # conversations that must not proceed unreviewed, and is invisible to the
    # rest. See review_queue.py.
    open_review: Optional[dict]

    # When this conversation began, as an ISO-8601 UTC timestamp. Set on the
    # first turn and rewritten only when an expiry restarts the session, so it
    # survives as the duration the audit row records (§05 Stage 04).
    # Stored as a string rather than a float because it is checkpointed, and a
    # monotonic clock reading is meaningless once the process that took it is
    # gone.
    session_started_at: str

    # When the most recent turn ran, same format. This — not the start — is
    # what the session lifetime bound measures against: "abandoned" means
    # nobody has touched the conversation, and a legitimately long-running
    # approval flow must not have its staged artifact destroyed by a timer
    # that started ticking on the first hello. Absent in conversations
    # checkpointed before the field existed, in which case expiry falls back
    # to the start time (the old, stricter behaviour).
    session_last_active_at: str
