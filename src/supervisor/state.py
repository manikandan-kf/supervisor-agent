"""The two schemas a turn runs against: checkpointed state, and per-run context.

`SupervisorState` must survive a turn; `SupervisorContext` is identity (persona,
entitlements, correlation), supplied by the Platform API on every invocation.
The boundary is a governance requirement: §4.4 says auth context "shall not be
checkpointed" — identity in state would restore yesterday's entitlements from a
checkpoint, while runtime context is never persisted, so every turn is
authorized against what the caller holds *now*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated, Optional

from agent_governance.trust import trust_secret, verify_entitlements
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SupervisorState(TypedDict, total=False):
    # ── Conversation, accumulated across turns by the checkpointer ──────────
    messages: Annotated[list, add_messages]
    conversation_id: str
    request_id: str

    # Copied from the request's agent-id by the RBAC gate, after validation.
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
    # Who approved/rejected a staged artifact (§05 Stage 04). Its own channel, not
    # an `outcome` value: the routing-completion KPI counts `outcome == "answer"`.
    signoff: Optional[dict]
    routed_agent_name: str
    sources: list
    audit_trail: list
    # This turn's governance spend (spend.py). In state, unlike the time budget: a
    # resumed approval turn needs a fresh clock, but spend already made stays made.
    spend: dict

    # ── Carried across turns by the checkpointer ────────────────────────────
    # Context resolved in this thread; checkpointed so it is thread-scoped (§4.4
    # isolation) and takes precedence over the shared per-user long-term store.
    session_context: dict
    # "Keep this in mind" notes, oldest first. Die with the session; never written
    # to the long-term store; reach worker dispatches only (see session_notes.py).
    session_notes: list
    # A second deliverable from one message, held rather than done, offered back,
    # and expired the moment the reply is anything but yes or no.
    deferred_request: Optional[dict]
    clarification_count: int
    pending_clarification: Optional[str]
    # The last guardrail block, so "appeal" is recognised as the §06 appeal path
    # rather than screened again. Cleared on appeal or when a later turn passes.
    appealable: Optional[dict]
    # Layer 6 anomaly signal; at the limit the conversation is escalated, not refused.
    guardrail_block_streak: int
    # Staged artifact awaiting sign-off. The pause is a LangGraph `interrupt()`, so
    # the graph genuinely suspends.
    pending_approval: Optional[dict]

    # Set for one turn when the RBAC gate releases a review hold, so `guardrails`
    # claims a granted retry allowance only then, not on all traffic.
    review_resolved: Optional[dict]

    # ── Review state (§05 Stage 03/04) ──────────────────────────────────────
    # The open appeal/escalation. Checkpointing it keeps escalation terminal without
    # putting the queue on every request path; an unreachable queue fails closed.
    open_review: Optional[dict]

    # ISO-8601 UTC, not monotonic: it is checkpointed. Rewritten only when an
    # expiry restarts the session.
    session_started_at: str

    # What the session lifetime bound measures — "abandoned" means untouched, not
    # old. Absent in older checkpoints, where expiry falls back to the start.
    session_last_active_at: str


# ---------------------------------------------------------------------------
# Per-run context: identity and entitlements (§4.4). Never checkpointed.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SupervisorContext:
    """Everything the graph needs about *who* is asking, for this turn only."""

    # Persona, for display and audit attribution. Never used for authorization.
    user_role: str = ""

    # Pseudonymous reference (usr_<sha256>) keying long-term memory. Never the
    # raw subject claim — §1.10 forbids raw identifiers leaving the front door.
    user_key: str = ""

    # Authoritative for the RBAC gate. `None` (nothing supplied; bundled role map
    # stands in) differs from `()` ("permitted to use nothing"), which must not widen.
    permitted_agents: Optional[tuple[str, ...]] = None

    # `<agent_key>_approve_action` grants (§2.11). Separate because approval needs
    # *two* permissions and is checked against the producing worker (§2.12).
    approvable_agents: Optional[tuple[str, ...]] = None

    # From the invocation path; always a concrete worker id, revalidated every call.
    requested_agent_id: str = ""

    # §1.10 correlation, carried into the decision trail.
    correlation_id: str = ""
    environment: str = "dev"

    # `time.monotonic()` anchor for the turn budget (§05 Stage 05). Never in state: a
    # checkpointed clock would be restored hours later on approval resume and refuse
    # itself. 0.0 = no clock supplied, which disables the budget (offline tests).
    turn_started_at: float = 0.0

    # False only when a trust secret is configured and the request arrived unsigned
    # or wrongly signed; the RBAC gate refuses it. Defaults True for hand-built contexts.
    verified: bool = True

    @classmethod
    def from_custom_inputs(cls, custom: dict) -> "SupervisorContext":
        """Build from the Platform API's `custom_inputs` payload.

        The turn clock starts here, never from the payload: a client-supplied value
        is trivially settable to disable the budget.
        """
        permitted = custom.get("permitted_agents")
        approvable = custom.get("approvable_agents")
        return cls(
            verified=verify_entitlements(custom, trust_secret()),
            user_role=str(custom.get("user_role", "")),
            user_key=str(custom.get("user_id") or custom.get("user_reference") or ""),
            permitted_agents=tuple(str(a) for a in permitted) if permitted is not None else None,
            approvable_agents=tuple(str(a) for a in approvable) if approvable is not None else None,
            requested_agent_id=str(custom.get("agent_id", "")),
            correlation_id=str(custom.get("correlation_id", "")),
            environment=str(custom.get("environment", "dev")),
            turn_started_at=time.monotonic(),
        )
