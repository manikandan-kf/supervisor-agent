"""Failsafe behaviour required by Solution v1.2 §06.

Two rows of that table are covered here:

  * "Guardrail/routing model service error → Fails closed — the query is held
    (not silently allowed through) and the user is told to retry; never treated
    as an implicit pass."
  * "Guardrail block the user disputes → Block message states the scope
    limitation clearly and offers an appeal path to a human/admin queue — not a
    silent retry against the same guardrail."
"""

from __future__ import annotations

from helpers import StubGuardrails, invoke

from supervisor.guardrails import GuardrailResult
from supervisor.nodes import (
    APPEAL_ACKNOWLEDGED_MESSAGE,
    APPEAL_NOTE,
    GOVERNANCE_UNAVAILABLE_MESSAGE,
)


class ExplodingGuardrails:
    """The screening model is unreachable after the graph's retries."""

    def screen(self, query, candidates, history=None):
        raise RuntimeError("model serving endpoint unavailable")


class ExplodingRouter:
    def resolve(self, agent, history, prior_context, carried_over=None):
        raise RuntimeError("model serving endpoint unavailable")


# ── Fail closed ─────────────────────────────────────────────────────────────


def test_screen_failure_holds_the_request(make_graph):
    graph, services = make_graph(guardrails=ExplodingGuardrails())
    result = invoke(graph, "Write an HLD for billing")

    assert result["outcome"] == "error"
    assert result["final_text"] == GOVERNANCE_UNAVAILABLE_MESSAGE
    # The point of failing closed: nothing reached a worker.
    assert services.workers.calls == []


def test_screen_failure_is_audited(make_graph):
    """A held request still leaves a decision trail — an unhandled exception
    would skip `respond` and record nothing at all."""
    graph, services = make_graph(guardrails=ExplodingGuardrails())
    invoke(graph, "Write an HLD for billing")

    trail = services.audit.records[0]["decision_trail"]
    assert [e["decision"] for e in trail if e["stage"] == "guardrails"] == ["fail_closed"]


def test_route_failure_holds_the_request(make_graph):
    graph, services = make_graph(router=ExplodingRouter())
    result = invoke(graph, "Write an HLD for billing")

    assert result["outcome"] == "error"
    assert result["final_text"] == GOVERNANCE_UNAVAILABLE_MESSAGE
    assert services.workers.calls == []
    trail = services.audit.records[0]["decision_trail"]
    assert any(e["stage"] == "route" and e["decision"] == "fail_closed" for e in trail)


def test_failure_is_not_an_implicit_pass(make_graph):
    """The wording matters: a transport failure must not be reported as a
    governance refusal, and must not read as an answer either."""
    graph, _ = make_graph(guardrails=ExplodingGuardrails())
    text = invoke(graph, "Write an HLD for billing")["final_text"]
    assert "held" in text.lower()
    assert "try again" in text.lower()


# ── Appeal path ─────────────────────────────────────────────────────────────


def _blocked_graph(make_graph):
    return make_graph(
        guardrails=StubGuardrails(GuardrailResult(False, "semantic", "this is out of scope"))
    )


def test_block_offers_an_appeal_path(make_graph):
    graph, _ = _blocked_graph(make_graph)
    result = invoke(graph, "Give me a pasta recipe")

    assert result["outcome"] == "blocked"
    assert APPEAL_NOTE in result["final_text"]
    # The scope limitation is still stated — the appeal is offered in addition,
    # not instead.
    assert "out of scope" in result["final_text"].lower()


def test_appeal_reaches_the_human_queue_without_rescreening(make_graph):
    """Re-screening the word "appeal" would block it again — the silent retry
    §06 rules out. The appeal is recognised before the screen runs."""
    guardrails = StubGuardrails(GuardrailResult(False, "semantic", "this is out of scope"))
    graph, services = make_graph(guardrails=guardrails)

    invoke(graph, "Give me a pasta recipe", thread="appeal-1")
    screens_after_block = len(guardrails.calls)

    result = invoke(graph, "appeal", thread="appeal-1")

    assert result["outcome"] == "escalated"
    assert result["final_text"] == APPEAL_ACKNOWLEDGED_MESSAGE
    assert len(guardrails.calls) == screens_after_block, "the appeal was re-screened"

    trail = services.audit.records[-1]["decision_trail"]
    appeal = [e for e in trail if e["decision"] == "appeal"]
    assert appeal, "the appeal is not in the decision trail for an admin to review"
    assert "out of scope" in appeal[0]["detail"]


def test_appeal_without_a_preceding_block_is_screened_normally(make_graph):
    """An unprompted "appeal" is just a message; it must not manufacture an
    escalation out of nothing."""
    graph, _ = make_graph()
    result = invoke(graph, "appeal", thread="appeal-2")
    assert result["outcome"] != "escalated"


def test_a_passing_turn_clears_the_appeal_offer(make_graph):
    """Once a later request passes the screen, the earlier block is no longer
    what "appeal" refers to."""
    guardrails = StubGuardrails(GuardrailResult(False, "semantic", "out of scope"))
    graph, _ = make_graph(guardrails=guardrails)
    invoke(graph, "Give me a pasta recipe", thread="appeal-3")

    guardrails.result = GuardrailResult(True, "semantic", "in domain")
    invoke(graph, "Write an HLD for billing", thread="appeal-3")

    result = invoke(graph, "appeal", thread="appeal-3")
    assert result["outcome"] != "escalated"
