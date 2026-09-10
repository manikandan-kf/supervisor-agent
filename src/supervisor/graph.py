"""Supervisor graph wiring.

The stage order — RBAC gate, Guardrails, Route/Clarify, Dispatch, Respond &
Audit — is built into the graph itself and cannot be skipped.

Each chat widget is already scoped to one agent, and the user's role is already
known from the caller's identity, so the target agent arrives fixed on every
invocation.
The supervisor's job is verification and in-domain clarification, not agent
selection: the RBAC gate revalidates the requested agent-id against the
caller's role mapping on every call.

There are no conditional-edge functions here. Each node returns a `Command`
carrying its state update and its destination together, so a stage's decision
and its consequence live in one place. `destinations=` declares where a node may
go, which keeps the graph drawable and catches a typo'd target at build time
rather than at run time.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .context import SupervisorContext
from .memory import build_checkpointer, build_store
from .nodes import SupervisorNodes
from .services import Services, build_services
from .state import SupervisorState

# There is deliberately no graph-level RetryPolicy here. One used to wrap the
# two model-backed nodes, but a RetryPolicy only fires on a *raised* exception
# and both nodes catch everything from the model call to return a controlled
# fail-closed Command — so the policy never saw a failure and every transient
# 429/5xx became an immediate "checks unavailable" hold with zero retries.
# Transient failures are now retried where they happen, around each individual
# governance model call, in `resilience.invoke_with_retries`: deadline-aware,
# and it re-sends only the verdict that failed instead of re-running a whole
# candidate fan-out.


def build_graph(services: Services | None = None, checkpointer=None, store=None):
    services = services or build_services()
    nodes = SupervisorNodes(services)

    graph = StateGraph(SupervisorState, context_schema=SupervisorContext)

    graph.add_node(
        "rbac_gate",
        nodes.rbac_gate,
        # No network, no model — a retry would only repeat a deterministic
        # decision. Goes to "approval" only to re-open an interrupted gate when
        # a normal message arrives while a staged artifact awaits sign-off.
        destinations=("guardrails", "approval", "respond"),
    )
    graph.add_node(
        "guardrails",
        nodes.guardrails,
        destinations=("route", "respond"),
    )
    graph.add_node(
        "route",
        nodes.route,
        destinations=("dispatch", "respond"),
    )
    graph.add_node(
        "dispatch",
        nodes.dispatch,
        # Deliberately NOT retried at graph level: a worker call may have side
        # effects, and replaying the whole node would repeat them. Transient
        # transport failures are retried with backoff inside the worker client,
        # behind a circuit breaker; the node catches what survives that and
        # returns a controlled "temporarily unavailable" message.
        destinations=("respond", "approval"),
    )
    # The pause lives in its own node because LangGraph re-runs a node from the
    # top when it resumes. Keeping `interrupt()` away from the worker call is
    # what stops the worker being invoked twice per approval.
    graph.add_node("approval", nodes.approval, destinations=("respond",))
    graph.add_node("respond", nodes.respond)

    graph.add_edge(START, "rbac_gate")
    graph.add_edge("respond", END)

    return graph.compile(
        checkpointer=checkpointer or build_checkpointer(),
        store=store or build_store(),
    )
