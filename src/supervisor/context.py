"""Per-run context — identity and entitlements (§4.4).

The Platform API supplies these on every invocation. They describe the *run*,
not the conversation, so they travel as LangGraph **runtime context** rather
than graph state.

That distinction is a governance requirement, not a style preference. Graph
state is checkpointed, and §4.4 is explicit that credentials and raw
authentication context "shall not be checkpointed". Identity in state means
yesterday's entitlements get restored from a checkpoint and reused — so a
permission revoked at the identity provider would keep working for anyone resuming an old
conversation. Runtime context is per-invocation and never persisted, so every
turn is authorized against the entitlements the caller holds *now*.

Nodes read it as:

    def node(self, state: dict, runtime: Runtime[SupervisorContext]):
        permitted = runtime.context.permitted_agents
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from agent_governance.trust import trust_secret, verify_entitlements


@dataclass(frozen=True)
class SupervisorContext:
    """Everything the graph needs about *who* is asking, for this turn only."""

    # Persona, for display and audit attribution. Never used for authorization.
    user_role: str = ""

    # Pseudonymous reference (usr_<sha256>) keying long-term memory. Never the
    # raw subject claim — §1.10 forbids raw identifiers leaving the front door.
    user_key: str = ""

    # Agents the caller may use, derived by the front door from the validated
    # identity token. Authoritative for the RBAC gate; the supervisor may route
    # to these and nothing else.
    #
    # `None` and `()` are different: None means the caller supplied nothing and
    # the bundled role map stands in; an empty tuple means "permitted to use
    # nothing", which is a real answer and must not silently widen.
    permitted_agents: Optional[tuple[str, ...]] = None

    # Agents whose staged artifacts the caller may sign off — the callers'
    # `<agent_key>_approve_action` grants (§2.11), derived by the Platform API
    # from the same token.
    #
    # Separate from `permitted_agents` because §2.11 requires *two* permissions
    # to approve, and because the supervisor cannot be the subject of either:
    # the artifact belongs to the worker that produced it, so the approval must
    # be checked against that worker's key however the request was addressed
    # (§2.12 — routing must not bypass a control). None carries the same
    # meaning as it does above: nothing was supplied, so fall back.
    approvable_agents: Optional[tuple[str, ...]] = None

    # The target agent from the invocation path (POST /agents/{agent-id}/
    # invocation). Each chat widget is scoped to one agent, so this is always a
    # concrete worker id; the RBAC gate revalidates it against the caller's
    # role mapping on every call.
    requested_agent_id: str = ""

    # §1.10 correlation, carried into the decision trail.
    correlation_id: str = ""
    environment: str = "dev"

    # When this invocation started, from `time.monotonic()`. The anchor for the
    # turn's total time budget (§05 Stage 05) — see `deadline.py`.
    #
    # It belongs here rather than in graph state for the same reason identity
    # does, and the consequence is sharper: state is checkpointed, so a budget
    # anchored there would be *restored* when a human answers an approval
    # interrupt hours later, and the resumed turn would measure an elapsed time
    # of hours and refuse itself before doing anything. Runtime context is
    # rebuilt per invocation, so a resumed turn correctly gets a fresh budget.
    #
    # 0.0 means "no clock supplied", which disables the budget rather than
    # expiring it — the path every offline test and direct graph call takes.
    turn_started_at: float = 0.0

    # Whether the entitlement block above was verified against the gateway's
    # HMAC (see `trust.py`). Only ever False when `SUPERVISOR_TRUST_SECRET` is
    # configured and the request arrived unsigned or wrongly signed — i.e. it
    # did not come through the gateway. The RBAC gate refuses such a turn.
    # Defaults True so hand-built contexts (tests, offline runs) and dark-mode
    # deployments behave exactly as before.
    verified: bool = True

    @classmethod
    def from_custom_inputs(cls, custom: dict) -> "SupervisorContext":
        """Build from the Platform API's `custom_inputs` payload.

        The turn clock is started *here*, not taken from the payload: it must
        measure time spent inside this process, and a client-supplied value
        would be both meaningless across machines and trivially settable to
        disable the budget.

        When a trust secret is configured, the entitlement block is verified
        against the gateway's signature before any of it is treated as
        authoritative — the docstring claim "computed server-side, never
        accepted from a browser" becomes something this process checks rather
        than something it hopes.
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
