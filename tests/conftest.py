import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The suite runs offline. Without this every `get_prompt` attempts a Unity
# Catalog round-trip, fails, logs a WARNING and returns the bundled default —
# so the assertions held against the fallback anyway, just slower and noisier.
# Setting it makes that explicit: tests pin the bundled templates, which is the
# text the deployed endpoint is actually running today.
os.environ.setdefault("PROMPT_REGISTRY_ENABLED", "false")

import time
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from supervisor.context import SupervisorContext
from supervisor.dispatch import WorkerResponse
from supervisor.graph import build_graph
from supervisor.guardrails import GuardrailResult, ScreenResult
from supervisor.memory import LongTermMemory
from supervisor.rbac import RbacPolicy
from supervisor.registry import AgentRegistry, WorkerAgent
from supervisor.review_queue import Review, ReviewQueueError
from supervisor.routing import RouteResult
from supervisor.services import Services
from supervisor.settings import Settings


class StubGuardrails:
    """A fixed verdict, plus control over which agent the screen picks.

    `owner` names the agent id the query should be routed to; None means the one
    already addressed, which is what the single-agent path does.
    """

    def __init__(self, result=None, owner=None):
        self.result = result or GuardrailResult(True, "semantic", "in domain")
        self.owner = owner
        self.calls = []
        self.screened = []
        # Every flattened history the screen was given. Recorded so a test can
        # assert what does *not* reach a governance prompt — session notes in
        # particular, which reach a worker and no governance model.
        self.histories = []
        # What the node passed as the turn's time budget, so a test can assert
        # the deadline reaches the fan-out rather than trusting that it does.
        self.deadlines = []

    # The reason this stub's `deterministic_block` gives, when a test wants the
    # held-over-request pre-screen to refuse. Empty means nothing is refused,
    # which is what every test that does not care about it gets.
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
    """Records every audit row, and can be made to fail.

    `fail_on` is a set of outcomes whose write raises, so a test can drive the
    §05 Stage 06 rule that a *governance decision* is not reported as applied
    unless its record landed — while an ordinary answer still gets through.
    """

    def __init__(self, fail_on=()):
        self.records = []
        self.fail_on = set(fail_on)
        self.attempts = []

    def log(self, record):
        self.attempts.append(record)
        if record.get("outcome") in self.fail_on:
            raise RuntimeError("audit sink unavailable")
        self.records.append(record)


class StubReviews:
    """In-memory stand-in for the appeal / escalation queue.

    Deliberately closer to the real thing than a bare mock: `claim_allowance`
    consumes at most once, and `resolve` refuses an already-resolved review, so
    the tests exercise the same one-winner semantics the Postgres implementation
    gets from its conditional UPDATE.
    """

    def __init__(self, fail=False):
        self.fail = fail
        self.rows = {}
        self.opened = []
        self._n = 0

    def _guard(self):
        if self.fail:
            raise ReviewQueueError("review queue unavailable (test)")

    def open_review(self, *, kind, conversation_id, reason, **extra):
        self._guard()
        self._n += 1
        review = Review(
            ref=f"rev_test_{self._n}",
            kind=kind,
            status="open",
            conversation_id=conversation_id,
            reason=reason,
            user_key=extra.get("user_key", ""),
            user_role=extra.get("user_role", ""),
            target_agent_id=extra.get("target_agent_id", ""),
            query_excerpt=extra.get("query_excerpt", ""),
        )
        self.rows[review.ref] = review
        self.opened.append(review)
        return review

    def resolve(self, ref, *, reviewer, decision, note=""):
        self._guard()
        current = self.rows.get(ref)
        if current is None:
            raise ReviewQueueError(f"no such review: {ref}")
        if current.status != "open":
            raise ReviewQueueError(f"{ref} was already resolved by {current.reviewer}")
        updated = replace(
            current,
            status="resolved",
            decision=decision,
            reviewer=reviewer,
            reviewer_note=note,
            resolved_at=datetime.now(timezone.utc),
        )
        self.rows[ref] = updated
        return updated

    def claim_allowance(self, conversation_id):
        self._guard()
        for ref, review in self.rows.items():
            if review.conversation_id == conversation_id and review.grants_retry:
                claimed = replace(review, consumed_at=datetime.now(timezone.utc))
                self.rows[ref] = claimed
                return claimed
        return None

    def get(self, ref):
        self._guard()
        return self.rows.get(ref)

    def list_open(self, *, kind="", limit=100):
        self._guard()
        return [
            r
            for r in self.rows.values()
            if r.status == "open" and (not kind or r.kind == kind)
        ][:limit]

    def open_for_conversation(self, conversation_id):
        self._guard()
        return next(
            (
                r
                for r in self.rows.values()
                if r.conversation_id == conversation_id and r.status == "open"
            ),
            None,
        )


# ── Credential-shaped fixtures ──────────────────────────────────────────────
#
# Assembled at run time, never written out as one literal.
#
# A redaction test cannot demonstrate anything without a string of exactly the
# shape it has to catch — and a complete one, written out, is indistinguishable
# from a real leaked credential to every scanner that matters: GitHub's push
# protection, which rejects the push outright; this repository's own secret
# scan; and whatever the reader runs. The values are obviously synthetic
# counting patterns, and splitting them at the prefix is what stops a scanner
# matching a contiguous run, while the assembled value still exercises the
# detector exactly as a real credential would.
#
# Put new credential shapes here rather than inline, so the next one does not
# block a push to find out.
FAKE_DATABRICKS_TOKEN = "dapi" + "0123456789abcdef" * 2
FAKE_GITHUB_TOKEN = "ghp_" + "1234567890abcdef" * 2 + "1234"
FAKE_GOOGLE_API_KEY = "AIza" + "SyA1234567890abcdefghijklmnopqrstuv"
# AWS's own documentation example pair — synthetic by publication, and split
# here for the same reason as the rest rather than relying on every scanner
# knowing that.
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


@pytest.fixture
def make_graph():
    def _make(
        guardrails=None,
        router=None,
        workers=None,
        audit=None,
        reviews=None,
        settings=None,
        memory=None,
        output_guard=None,
    ):
        services = Services(
            settings=settings or Settings(),
            registry=REGISTRY,
            rbac=RBAC,
            guardrails=guardrails or StubGuardrails(),
            router=router or StubRouter(),
            workers=workers or StubWorkers(),
            audit=audit or StubAudit(),
            # Same derivation as production (services.build_services): the
            # allowlist is the registry's declared context keys, so tests
            # exercise the real §04 validation rather than an open store.
            memory=memory or LongTermMemory(InMemoryStore(), allowed_keys=REGISTRY.context_keys()),
            reviews=reviews if reviews is not None else StubReviews(),
        )
        if output_guard is not None:
            services.output_guard = output_guard
        graph = build_graph(services, checkpointer=MemorySaver(), store=InMemoryStore())
        return graph, services

    return _make


def context_for(
    role="BA",
    agent="requirement-agent",
    permitted=None,
    approvable=None,
    user_key="user-1",
    started_at=None,
):
    """Runtime context for one turn — identity, never state (§4.4).

    `started_at` anchors the turn's time budget. Left as the current monotonic
    reading so the ordinary test path has a *live* budget rather than a disabled
    one — a suite that ran with the deadline switched off would not be testing
    the deployed configuration. Tests that want an exhausted budget pass a value
    in the past; tests that want it disabled pass 0.
    """
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
    """One turn through the graph.

    `agent` is the target fixed by the invocation path — each chat widget is
    scoped to one agent, so it always names a concrete worker.
    """
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
