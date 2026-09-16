"""Stage 1 — the RBAC gate, and the session lifetime it enforces.

Revalidates the requested agent-id against the caller's role mapping on
every call, expires a conversation that has gone stale, and re-opens a
hold when a reviewer has not yet decided.
"""

from __future__ import annotations

import logging
import uuid
from typing import Literal, Optional

from agent_governance import review_queue
from agent_governance.rbac import DENIED_MESSAGE
from agent_governance.review_queue import ReviewQueueError
from langgraph.runtime import Runtime
from langgraph.types import Command

from ..messages import (
    APPROVAL_PENDING_MESSAGE,
    INPUT_TOO_LONG_MESSAGE,
    REVIEW_PENDING_MESSAGE,
    REVIEW_UNAVAILABLE_MESSAGE,
    SESSION_EXPIRED_MESSAGE,
    SUBJECT_ALLOWANCE_MESSAGE,
    progress,
)
from ..state import SupervisorContext
from .turn import (
    NodeBase,
    _entry,
    _latest_user_text,
    _narrates,
    _now_iso,
    _session_idle_seconds,
)

logger = logging.getLogger(__name__)


class RbacGateMixin(NodeBase):
    # ── Session lifetime and review holds, used by the RBAC gate ────────────

    def _expire_if_stale(
        self, state: dict, context: SupervisorContext, agent_id: str, reset: dict, reason: str
    ) -> Optional[Command]:
        """Clear an over-age conversation's carried state, or None if it is fresh.

        §05 Stage 04 / GDPR Art. 5(1)(e) / SOC 2 CC6.1: an abandoned session must not hold
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
        cleared: dict = {
            "pending_approval": None,
            "open_review": None,
            "session_context": {},
            "clarification_count": 0,
            "pending_clarification": None,
            "appealable": None,
            # Notes and a held-over offer expire with the session: leaving them would turn a
            # bounded session into unbounded free-text retention (§05 Stage 04, GDPR Art. 5(1)(e)).
            "deferred_request": None,
            # Restart the clock, so the user's next message runs normally rather
            # than expiring again on the same stale timestamp.
            "session_started_at": _now_iso(),
        }
        if not carried:
            reset["session_started_at"] = _now_iso()
            return None

        progress("rbac_gate", "blocked", "session expired")
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

        An open escalation is terminal; an open appeal lets the turn run. Fails closed when
        the queue cannot be read, but only where the conversation's own state says a review is open.
        """
        marker = state.get("open_review") or {}
        ref = str(marker.get("ref") or "")
        if not ref:
            return None

        try:
            review = self.s.reviews.get(ref)
        except ReviewQueueError as exc:
            logger.warning("review status unavailable for %s: %s", ref, exc)
            progress("rbac_gate", "blocked", "review status unavailable")
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

        # A marker pointing at a deleted row would hold the conversation forever. Treated as
        # resolved: deleting a review row is a legitimate operator act, a permanent lock is not.
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
            # An open **appeal** does not freeze the conversation: §05 Stage 04 makes the
            # *escalated request* terminal, not the user, and a retry is already guarded by the
            # screen and the streak anomaly. An **escalation** stays terminal on purpose.
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

            progress("rbac_gate", "blocked", f"{review.kind} under review")
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

        # Resolved: release the hold and record who decided what (Stage 03 appeal audit).
        # `review_resolved` tells `guardrails` this turn observed it, so the claim runs only now.
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

    def rbac_gate(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["guardrails", "approval", "respond"]]:
        """Check the requested agent against the caller's role-to-agent mapping.

        Revalidates in-graph on every call — defence in depth against an IDOR on the path
        parameter, a stale permission or a bypassed gateway. First node, so it resets per-turn fields.
        """
        context = runtime.context or SupervisorContext()
        narrate = _narrates(state)
        if narrate:
            progress("rbac_gate", "started")

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
            # One-turn marker; only `_hold_for_review` ever sets it.
            "review_resolved": None,
            # Per-turn, like the stage results above: carrying the ledger forward
            # would make every later turn of a long conversation refuse itself.
            "spend": {},
            # Stamped on the first turn only: plain-str channels do not merge, and re-stamping
            # would keep every session perpetually young. The *activity* clock restamps every turn.
            "session_started_at": state.get("session_started_at") or _now_iso(),
            "session_last_active_at": _now_iso(),
        }

        # ── Entitlement provenance (trust.py) ───────────────────────────────
        # With a trust secret configured, an unsigned entitlement block is refused outright,
        # before anything reads `permitted_agents` or `user_role`: every control is built on them.
        if not context.verified:
            progress("rbac_gate", "blocked", "unverified caller")
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
            progress("rbac_gate", "blocked", "not permitted")
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
        # The gateway caps `input` at 8000 chars, but in a different deployable. Checked after
        # authorization so an unauthorized caller learns nothing, before any state or model call.
        max_chars = self.s.settings.input_max_chars
        inbound = _latest_user_text(state.get("messages"))
        if max_chars > 0 and len(inbound) > max_chars:
            progress("rbac_gate", "blocked", "message too long")
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
        # The gateway's rate limiter bounds *requests*, not spend. After authorization, so quota
        # messages are no oracle for which agents exist; before the session and review gates,
        # so a caller with no allowance does not get their conversation expired by a dead turn.
        exceeded = ""
        try:
            exceeded = self.s.spend_window.check(context.user_key)
        except Exception:
            # Fails *open*, alone among the gate's checks: a cost backstop, not a security
            # control — the boundary is the RBAC decision above; an outage is the worse trade.
            logger.warning("subject spend window check failed", exc_info=True)
        if exceeded:
            progress("rbac_gate", "blocked", "request allowance spent")
            logger.info("subject allowance reached for %s: %s", context.user_key or "?", exceeded)
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
        # After authorization, so an unauthorized caller learns nothing about whether a
        # conversation exists; before the review and approval gates, which it must not keep open.
        expired = self._expire_if_stale(state, context, agent_id, reset, reason)
        if expired is not None:
            return expired

        # ── Terminal review state (§05 Stage 04) ────────────────────────────
        # An open appeal or escalation holds the conversation until a human resolves it. Reads
        # the checkpointed marker first, so a conversation with nothing open never hits the queue.
        gate_trail: list = []
        held = self._hold_for_review(state, context, agent_id, reset, reason, gate_trail)
        if held is not None:
            return held

        if state.get("pending_approval"):
            # A normal message arrived while a staged artifact awaits sign-off. Only approve/
            # reject/comment are accepted while a gate is open; it is never merged into the stage.
            if narrate:
                progress("rbac_gate", "done", "approval gate open")
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
            progress("rbac_gate", "done")
        return Command(
            goto="guardrails",
            update={
                **reset,
                "target_agent_id": agent_id,
                "rbac": {"allowed": True, "reason": reason},
                "audit_trail": [_entry("rbac_gate", "allow", reason), *gate_trail],
            },
        )
