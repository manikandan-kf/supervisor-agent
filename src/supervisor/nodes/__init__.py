"""Graph nodes.

RBAC gate -> Guardrails -> Route/Clarify -> Dispatch -> Respond & Audit; the order is fixed
by the graph. The target arrives fixed on every invocation, so the supervisor verifies and
clarifies rather than selects: the RBAC gate revalidates the path's agent-id server-side on
every call, and the UI's selection is never the access decision. Each node returns a
`Command` carrying the update *and* the destination, so one decision lives in one place.
Identity arrives as `Runtime[SupervisorContext]`, never from state (see `state.py`).
"""

from __future__ import annotations

from .approval import ApprovalMixin
from .dispatch import DispatchMixin
from .guardrails import GuardrailsMixin
from .rbac_gate import RbacGateMixin
from .respond import RespondMixin
from .route import RouteMixin
from .turn import GOVERNANCE_OUTCOMES, NodeBase

__all__ = ["GOVERNANCE_OUTCOMES", "SupervisorNodes"]


class SupervisorNodes(
    RbacGateMixin,
    GuardrailsMixin,
    RouteMixin,
    DispatchMixin,
    ApprovalMixin,
    RespondMixin,
    NodeBase,
):
    """The six graph stages, bound to one service container, listed in pipeline order.

    Naming rule: a module named for a node holds that node only, spelled as `graph.py`
    registers it, so "dispatch failed" in a trace maps to exactly one file.
    """
