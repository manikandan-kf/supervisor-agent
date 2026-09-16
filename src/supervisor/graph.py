"""Supervisor graph wiring.

The stage order — RBAC gate, Guardrails, Route/Clarify, Dispatch, Respond &
Audit — is built into the graph and cannot be skipped. The target agent arrives
fixed on every invocation (widgets are per-agent), so the supervisor verifies
and clarifies rather than selects; the RBAC gate revalidates it every call.
No conditional-edge functions: each node returns a `Command[Literal[...]]`, so a
decision and its consequence live together and targets are checked at build time.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .memory import build_checkpointer, build_store
from .nodes import SupervisorNodes
from .services import Services, build_services
from .state import SupervisorContext, SupervisorState

# Deliberately no graph-level RetryPolicy: it fires only on *raised* exceptions and
# the model nodes catch everything to fail closed. Retries: `resilience.invoke_with_retries`.


def build_graph(services: Services | None = None, checkpointer=None, store=None):
    services = services or build_services()
    nodes = SupervisorNodes(services)

    graph = StateGraph(SupervisorState, context_schema=SupervisorContext)

    # Destinations come from each node's `Command[Literal[...]]` annotation;
    # `destinations=` here would repeat it in a second place, free to disagree.
    graph.add_node("rbac_gate", nodes.rbac_gate)
    graph.add_node("guardrails", nodes.guardrails)
    graph.add_node("route", nodes.route)
    graph.add_node("dispatch", nodes.dispatch)
    # Own node because LangGraph re-runs a node from the top on resume; keeping
    # `interrupt()` away from the worker call stops a double invocation per approval.
    graph.add_node("approval", nodes.approval)
    graph.add_node("respond", nodes.respond)

    graph.add_edge(START, "rbac_gate")
    graph.add_edge("respond", END)

    return graph.compile(
        checkpointer=checkpointer or build_checkpointer(),
        store=store or build_store(),
    )
