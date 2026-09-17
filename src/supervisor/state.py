"""The two schemas a turn runs against: checkpointed state, and per-run context.

`SupervisorState` must survive a turn; `SupervisorContext` is identity (persona,
entitlements, correlation), supplied by the calling application on every invocation.
The boundary is a governance requirement (solution §04): identity in state would
restore yesterday's entitlements from a
checkpoint, while runtime context is never persisted, so every turn is
authorized against what the caller holds *now*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated, Optional

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
    # answer | blocked | clarify | escalated | approval_pending | error | expired
    outcome: str
    final_text: str
    # Who approved/rejected a staged artifact (solution §02). Its own channel, not
    # an `outcome` value: the routing-completion KPI counts `outcome == "answer"`.
    signoff: Optional[dict]
    routed_agent_name: str
    sources: list
    audit_trail: list

    # ── Carried across turns by the checkpointer ────────────────────────────
    # Context resolved in this thread; checkpointed so it is thread-scoped and takes
    # precedence over the shared per-user long-term store.
    session_context: dict
    # A second deliverable from one message, held rather than done, offered back,
    # and expired the moment the reply is anything but yes or no.
    deferred_request: Optional[dict]
    clarification_count: int
    pending_clarification: Optional[str]
    # Layer 6 anomaly signal; at the limit the conversation is escalated, not refused.
    guardrail_block_streak: int
    # Staged artifact awaiting sign-off. The pause is a LangGraph `interrupt()`, so
    # the graph genuinely suspends.
    pending_approval: Optional[dict]

    # ISO-8601 UTC, not monotonic: it is checkpointed. Rewritten only when an
    # expiry restarts the session.
    session_started_at: str

    # What the session lifetime bound measures — "abandoned" means untouched, not
    # old. Absent in older checkpoints, where expiry falls back to the start.
    session_last_active_at: str


# ---------------------------------------------------------------------------
# Per-run context: identity and entitlements. Never checkpointed.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SupervisorContext:
    """Everything the graph needs about *who* is asking, for this turn only."""

    # Persona, for display and audit attribution. Never used for authorization.
    user_role: str = ""

    # Pseudonymous reference (usr_<sha256>) keying long-term memory. Never the
    # raw subject claim: no raw identifier reaches the supervisor.
    user_key: str = ""

    # Authoritative for the RBAC gate. Trusted because only the calling application's
    # service principal holds CAN QUERY on the endpoint (DEPLOYMENT.md §7a). `None` (nothing
    # supplied; bundled role map stands in) differs from `()` ("permitted to use
    # nothing"), which must not widen.
    permitted_agents: Optional[tuple[str, ...]] = None

    # Which agents this caller may approve work for. Separate because approval needs
    # *two* permissions and is checked against the producing worker.
    approvable_agents: Optional[tuple[str, ...]] = None

    # From the invocation path; always a concrete worker id, revalidated every call.
    requested_agent_id: str = ""

    # Correlation id from the caller, carried into the decision trail and the worker call.
    correlation_id: str = ""
    environment: str = "dev"

    # `time.monotonic()` anchor for the turn budget (solution §05). Never in state: a
    # checkpointed clock would be restored hours later on approval resume and refuse
    # itself. 0.0 = no clock supplied, which disables the budget (offline tests).
    turn_started_at: float = 0.0

    @classmethod
    def from_custom_inputs(cls, custom: dict) -> "SupervisorContext":
        """Build from the request's `custom_inputs` payload.

        The turn clock starts here, never from the payload: a client-supplied value
        is trivially settable to disable the budget.
        """
        permitted = custom.get("permitted_agents")
        approvable = custom.get("approvable_agents")
        return cls(
            user_role=str(custom.get("user_role", "")),
            user_key=str(custom.get("user_id") or custom.get("user_reference") or ""),
            permitted_agents=tuple(str(a) for a in permitted) if permitted is not None else None,
            approvable_agents=tuple(str(a) for a in approvable) if approvable is not None else None,
            requested_agent_id=str(custom.get("agent_id", "")),
            correlation_id=str(custom.get("correlation_id", "")),
            environment=str(custom.get("environment", "dev")),
            turn_started_at=time.monotonic(),
        )
