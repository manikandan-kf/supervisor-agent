"""Failsafe behaviour required by Solution v1.2 §06 — two rows of that table:

* model service error → fails closed: the query is held and the user told to retry,
  never treated as an implicit pass;
* a disputed block → states the scope limitation and offers an appeal path to a queue.
"""

from __future__ import annotations

import time

import pytest
from agent_governance.resilience import (
    BudgetExhausted,
    Deadline,
    invoke_with_retries,
    is_transient,
)
from agent_governance.spend import SpendCaps, SubjectWindow
from helpers import StubGuardrails, invoke

from supervisor.guardrail_engine import GuardrailResult
from supervisor.messages import (
    APPEAL_ACKNOWLEDGED_MESSAGE,
    APPEAL_NOTE,
    GOVERNANCE_UNAVAILABLE_MESSAGE,
)
from supervisor.settings import Settings


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


# ── Code failsafes: the turn budget and the retry inside it (resilience.py) ──
#
# §05's "code failsafe" rows: what a transient failure, a permanent one and an
# exhausted budget each do, so the retry authority cannot become two authorities again.


class _Transient(Exception):
    status_code = 503


def test_a_transient_failure_is_retried_and_the_result_returned():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _Transient("upstream blip")
        return "ok"

    assert invoke_with_retries(flaky, what="verdict", attempts=3, initial_backoff=0.001) == "ok"
    assert len(calls) == 3


def test_a_permanent_failure_is_raised_after_exactly_one_attempt():
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("schema mismatch")

    with pytest.raises(ValueError):
        invoke_with_retries(broken, what="verdict", attempts=3, initial_backoff=0.001)
    assert len(calls) == 1


def test_an_exhausted_budget_is_never_retried():
    calls = []

    def over_budget():
        calls.append(1)
        raise BudgetExhausted("verdict", 130.0, 120.0)

    with pytest.raises(BudgetExhausted):
        invoke_with_retries(over_budget, what="verdict", attempts=3, initial_backoff=0.001)
    assert len(calls) == 1


def test_the_last_transient_failure_is_raised_once_attempts_run_out():
    calls = []

    def always_transient():
        calls.append(1)
        raise _Transient("still down")

    with pytest.raises(_Transient):
        invoke_with_retries(always_transient, what="verdict", attempts=2, initial_backoff=0.001)
    assert len(calls) == 2


def test_a_retry_that_cannot_finish_inside_the_deadline_is_not_attempted():
    # ~0.05s left; the first backoff would be 0.5–1.0s. Raise now, do not sleep.
    deadline = Deadline(started_at=time.monotonic() - 119.95, budget_seconds=120.0)
    calls = []

    def transient():
        calls.append(1)
        raise _Transient("blip")

    started = time.monotonic()
    with pytest.raises(_Transient):
        invoke_with_retries(
            transient, what="verdict", deadline=deadline, attempts=3, initial_backoff=1.0
        )
    assert len(calls) == 1
    assert time.monotonic() - started < 0.4


def test_a_deadline_with_no_clock_is_disabled_and_one_past_its_budget_refuses():
    assert Deadline().enabled is False
    assert Deadline().remaining() == float("inf")
    Deadline().ensure("anything")  # never raises

    class _Ctx:
        turn_started_at = 0.0

    assert Deadline.from_context(_Ctx(), 120.0).enabled is False

    spent = Deadline(started_at=time.monotonic() - 10.0, budget_seconds=5.0)
    assert spent.exhausted is True
    with pytest.raises(BudgetExhausted) as exc:
        spent.ensure("the guardrail screen")
    assert exc.value.what == "the guardrail screen"
    assert exc.value.budget == 5.0


def test_what_counts_as_transient():
    class _Status(Exception):
        def __init__(self, code):
            super().__init__(f"http {code}")
            self.status_code = code

    assert is_transient(_Status(429)) is True
    assert is_transient(_Status(503)) is True
    assert is_transient(_Status(400)) is False
    assert is_transient(_Status(403)) is False
    assert is_transient(TimeoutError("Read timed out")) is True
    assert is_transient(ConnectionError("connection reset")) is True
    assert is_transient(ValueError("nope")) is False


# ── Cost control: one subject's rolling allowance (spend.py) ─────────────────


def test_the_shipped_defaults_cap_the_turn_and_leave_the_subject_window_dark():
    caps = SpendCaps.from_settings(Settings())
    assert caps.turn_model_calls == 8
    assert caps.turn_tokens == 60000
    assert caps.turn_enabled is True
    assert caps.subject_enabled is False


def test_a_dark_subject_window_never_refuses():
    window = SubjectWindow()
    assert window.enabled is False
    window.charge("usr_1", model_calls=50, tokens=500000)
    assert window.check("usr_1") == ""


def test_the_call_allowance_refuses_the_next_turn_not_the_current_one():
    window = SubjectWindow(SpendCaps(subject_model_calls=2, subject_window_seconds=60))
    assert window.check("usr_1") == ""
    window.charge("usr_1", model_calls=2)
    reason = window.check("usr_1")
    assert "2 governance model calls" in reason
    assert "2-call subject allowance" in reason
    # Another subject is unaffected.
    assert window.check("usr_2") == ""


def test_the_token_allowance_is_independent_of_the_call_allowance():
    window = SubjectWindow(SpendCaps(subject_tokens=100, subject_window_seconds=60))
    window.charge("usr_1", tokens=100)
    assert "~100 governance tokens" in window.check("usr_1")


def test_spend_older_than_the_window_stops_counting():
    window = SubjectWindow(SpendCaps(subject_model_calls=1, subject_window_seconds=0.05))
    window.charge("usr_1", model_calls=1)
    assert window.check("usr_1") != ""
    time.sleep(0.08)
    assert window.check("usr_1") == ""


def test_an_unattributable_caller_is_allowed_through_rather_than_pooled():
    window = SubjectWindow(SpendCaps(subject_model_calls=1, subject_window_seconds=60))
    window.charge("", model_calls=5)
    assert window.check("") == ""


def test_zero_charges_leave_no_entry():
    window = SubjectWindow(SpendCaps(subject_model_calls=1, subject_window_seconds=60))
    window.charge("usr_1", model_calls=0, tokens=0)
    assert window.check("usr_1") == ""
