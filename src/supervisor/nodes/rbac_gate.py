"""Stage 1 — the RBAC gate, and the session lifetime it enforces.

Revalidates the requested agent-id against the caller's role mapping on
every call, and expires a conversation that has gone stale.
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal, Optional

from agent_governance.rbac import DENIED_MESSAGE
from langgraph.runtime import Runtime
from langgraph.types import Command

from ..guardrail_engine import approval_reply
from ..state import SupervisorContext
from ..user_facing_text import (
    APPROVAL_PENDING_MESSAGE,
    INPUT_TOO_LONG_MESSAGE,
    SESSION_EXPIRED_MESSAGE,
)
from .base import (
    NodeBase,
    _entry,
    _latest_user_text,
    _now_iso,
    _session_idle_seconds,
)

logger = logging.getLogger(__name__)


class RbacGateMixin(NodeBase):
    # ── Session lifetime, enforced by the RBAC gate ─────────────────────────

    def _expire_if_stale(
        self, state: dict, context: SupervisorContext, agent_id: str, reset: dict, reason: str
    ) -> Optional[Command]:
        """Clear an over-age conversation's carried state, or None if it is fresh.

        GDPR Art. 5(1)(e) / SOC 2 CC6.1: an abandoned session must not hold
        an approval or context open. Age is **idle time**; messages stay as the audit record.
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

        # Nothing carried forward means nothing to clear, so an empty conversation is not told
        # its emptiness expired — but its clock restarts so the stale stamp cannot recur.
        carried = bool(
            state.get("pending_approval")
            or state.get("session_context")
            or state.get("pending_clarification")
            or state.get("deferred_request")
        )

        logger.info(
            "session %s expired after %.0fs idle (limit %.0fs), carried_state=%s",
            state.get("conversation_id", ""),
            age,
            max_age,
            carried,
        )
        cleared: dict = {
            "pending_approval": None,
            "session_context": {},
            "clarification_count": 0,
            "pending_clarification": None,
            # A held-over offer expires with the session (GDPR Art. 5(1)(e)).
            "deferred_request": None,
            # Restart the clock, so the user's next message runs normally rather
            # than expiring again on the same stale timestamp.
            "session_started_at": _now_iso(),
        }
        if not carried:
            reset["session_started_at"] = _now_iso()
            return None

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
                        "carried context and pending approval cleared",
                    ),
                ],
            },
        )

    def rbac_gate(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["guardrails", "approval", "respond"]]:
        """Check the requested agent against the caller's role-to-agent mapping.

        Revalidates in-graph on every call — defence in depth against an IDOR on the path
        parameter, a stale permission or a bypassed caller. First node, so it resets per-turn fields.
        """
        context = runtime.context or SupervisorContext()

        reset: dict = {
            "request_id": str(uuid.uuid4()),
            "outcome": "",
            "final_text": "",
            "worker_response": None,
            "routed_agent_name": "",
            "sources": [],
            # Cleared every turn: a stale sign-off would make the *next* turn look like a
            # governance decision and re-audit an approval that already happened.
            "signoff": None,
            # A decision read out of the conversation belongs to the turn that carried it.
            "approval_reply": None,
            # Stamped on the first turn only: plain-str channels do not merge, and re-stamping
            # would keep every session perpetually young. The *activity* clock restamps every turn.
            "session_started_at": state.get("session_started_at") or _now_iso(),
            "session_last_active_at": _now_iso(),
        }

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
        # The calling application validates payload size too (solution §05), but in a different
        # deployable. Checked after
        # authorization so an unauthorized caller learns nothing, before any state or model call.
        max_chars = self.s.settings.input_max_chars
        inbound = _latest_user_text(state.get("messages"))
        if max_chars > 0 and len(inbound) > max_chars:
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

        # ── Session lifetime ────────────────────────────────────────────────
        # After authorization, so an unauthorized caller learns nothing about whether a
        # conversation exists; before the approval gate, which it must not keep open.
        expired = self._expire_if_stale(state, context, agent_id, reset, reason)
        if expired is not None:
            return expired

        if state.get("pending_approval"):
            # Only approve / reject / comment are accepted while a gate is open (solution §05).
            # A decision typed into the conversation settles the gate here — the caller may have
            # no way to send a `resume` payload — and the approval node then does not suspend.
            decision = approval_reply(inbound)
            if decision is not None:
                return Command(
                    goto="approval",
                    update={
                        **reset,
                        "target_agent_id": agent_id,
                        "rbac": {"allowed": True, "reason": reason},
                        "approval_reply": decision,
                        "audit_trail": [
                            _entry("rbac_gate", "allow", reason),
                            _entry(
                                "approval_gate",
                                "decision_received",
                                f"{decision['decision']} in the conversation — the gate was "
                                "settled without a resume payload",
                            ),
                        ],
                    },
                )

            # Anything else is out of turn: refused with an explanation, never merged
            # into the pending stage.
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
                        _entry(
                            "approval_gate",
                            "rejected_input",
                            "message received while an approval gate is open",
                        ),
                    ],
                },
            )

        return Command(
            goto="guardrails",
            update={
                **reset,
                "target_agent_id": agent_id,
                "rbac": {"allowed": True, "reason": reason},
                "audit_trail": [_entry("rbac_gate", "allow", reason)],
            },
        )
