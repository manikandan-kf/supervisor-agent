"""Stubs, fixtures and fake credentials shared by the supervisor's tests.

Imported by the test modules directly (`from helpers import ...`); the
`make_graph` fixture in conftest.py wires these into a `Services`.
"""

from __future__ import annotations

import time

from agent_governance.rbac import RbacPolicy
from langgraph.types import Command

from supervisor.agent_registry import AgentRegistry, WorkerAgent
from supervisor.context_resolver import RouteResult
from supervisor.guardrail_engine import GuardrailResult, ScreenResult
from supervisor.state import SupervisorContext
from supervisor.worker_client import WorkerResponse


class StubGuardrails:
    """A fixed verdict, plus control over which agent the screen picks: `owner` names the
    agent id to route to, None the one already addressed (the single-agent path)."""

    def __init__(self, result=None, owner=None):
        self.result = result or GuardrailResult(True, "semantic", "in domain")
        self.owner = owner
        self.calls = []
        self.screened = []
        # Every flattened history the screen was given, so a test can assert what does
        # *not* reach a governance prompt — session notes in particular.
        self.histories = []
        # What the node passed as the turn's time budget, so a test can assert
        # the deadline reaches the fan-out rather than trusting that it does.
        self.deadlines = []

    # The reason this stub's `deterministic_block` gives when a test wants the
    # held-over-request pre-screen to refuse. Empty means nothing is refused.
    block_reason = ""

    def deterministic_block(self, query):
        return self.block_reason

    def screen(self, query, candidates, history=None, deadline=None):
        self.calls.append(query)
        self.screened.append([a.id for a in candidates])
        self.histories.append(list(history or []))
        self.deadlines.append(deadline)
        chosen = None
        if self.result.passed and not self.result.small_talk:
            chosen = next(
                (a for a in candidates if a.id == self.owner),
                candidates[0] if candidates else None,
            )
        return ScreenResult(chosen, self.result, tuple(a.id for a in candidates))


class StubRouter:
    """Returns queued results in order, repeating the last one."""

    def __init__(self, results=None):
        self.results = list(results or [RouteResult(True, {"product_line": "alpha"})])
        self.calls = 0
        self.histories = []
        self.deadlines = []

    def resolve(self, agent, history, prior_context, deadline=None, carried_over=None):
        self.calls += 1
        self.histories.append(list(history or []))
        self.deadlines.append(deadline)
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


class StubWorkers:
    def __init__(self, responses=None):
        self.responses = list(responses or [WorkerResponse(text="worker answer")])
        self.calls = []

    def invoke(self, agent, messages, context, conversation_id, user_role, trace, deadline=None):
        self.calls.append(
            {
                "agent": agent.id,
                "messages": messages,
                "context": context,
                "trace": trace,
                "deadline": deadline,
            }
        )
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


class StubAudit:
    """Records every audit row, and can be made to fail: `fail_on` is a set of outcomes
    whose write raises, driving the rule that a governance decision needs its record."""

    def __init__(self, fail_on=()):
        self.records = []
        self.fail_on = set(fail_on)
        self.attempts = []

    def log(self, record):
        self.attempts.append(record)
        if record.get("outcome") in self.fail_on:
            raise RuntimeError("audit sink unavailable")
        self.records.append(record)


# ── Credential-shaped fixtures ──────────────────────────────────────────────
#
# Assembled at run time, never written out as one literal: a complete credential-shaped
# string is indistinguishable from a real leak to GitHub push protection and this repo's
# own secret scan, while the split value still exercises the detector. New shapes go here.
FAKE_DATABRICKS_TOKEN = "dapi" + "0123456789abcdef" * 2
FAKE_GITHUB_TOKEN = "ghp_" + "1234567890abcdef" * 2 + "1234"
FAKE_GOOGLE_API_KEY = "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"
# AWS's own documentation example pair — synthetic by publication, but split here for
# the same reason as the rest rather than trusting every scanner to know that.
FAKE_AWS_KEY_ID = "AKIA" + "IOSFODNN7EXAMPLE"
FAKE_AWS_SECRET = "wJalrXUtnFEMI/" + "K7MDENG/bPxRfiCYEXAMPLEKEY"

REGISTRY = AgentRegistry(
    [
        WorkerAgent(
            id="requirement-agent",
            name="Requirement Agent",
            description="Generates requirements",
            endpoint="requirement-agent",
            domain_scope="Software requirements engineering",
            required_context=("product_line",),
        ),
        WorkerAgent(
            id="coding-agent",
            name="Coding Agent",
            description="Implements code",
            endpoint="coding-agent",
            domain_scope="Software implementation",
        ),
    ]
)

RBAC = RbacPolicy({"BA": ["requirement-agent"], "Developer": ["coding-agent"]})


def context_for(
    role="BA",
    agent="requirement-agent",
    permitted=None,
    approvable=None,
    user_key="user-1",
    started_at=None,
):
    """Runtime context for one turn — identity, never state. `started_at` stays a
    live monotonic reading so tests run the deployed budget; a past value exhausts it,
    0 disables it."""
    return SupervisorContext(
        user_role=role,
        user_key=user_key,
        requested_agent_id=agent or "",
        permitted_agents=tuple(permitted) if permitted is not None else None,
        approvable_agents=tuple(approvable) if approvable is not None else None,
        turn_started_at=time.monotonic() if started_at is None else started_at,
    )


def invoke(
    graph,
    text,
    role="BA",
    agent="requirement-agent",
    thread="t1",
    permitted=None,
    approvable=None,
    user_key="user-1",
    started_at=None,
):
    """One turn through the graph. `agent` is fixed by the caller, which always names one
    concrete worker."""
    return graph.invoke(
        {"messages": [{"role": "user", "content": text}], "conversation_id": thread},
        config={"configurable": {"thread_id": thread}},
        context=context_for(role, agent, permitted, approvable, user_key, started_at),
    )


def resume(
    graph,
    decision,
    thread="t1",
    role="BA",
    agent="requirement-agent",
    permitted=None,
    approvable=None,
    user_key="user-1",
):
    """Answer a pending `interrupt()` — approve or reject a staged artifact."""
    return graph.invoke(
        Command(resume=decision),
        config={"configurable": {"thread_id": thread}},
        context=context_for(role, agent, permitted, approvable, user_key),
    )
