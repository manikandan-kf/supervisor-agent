"""What a turn is allowed to carry, and what it must ask about.

Three failures reported from real use, all of them the supervisor being
confidently helpful about something nobody asked for:

  * **context outliving its subject** — a question in a new conversation
    answered in terms of the previous one, because long-term memory seeds a
    fresh thread with the user's most recent values and nothing checks whether
    the new request is about them;
  * **half a compound request** — "write the user stories and generate the test
    cases" produced the stories and dropped the rest, silently;
  * **ownership decided by candidate order** — the screen stopped at the first
    agent above the acting threshold, and `candidates` is ordered target-first,
    so which agent got a request that two of them legitimately own depended on
    which chat widget was open.

The first two are about a *turn's* scope; the third is about a *request's*. All
three end the same way: the supervisor asks instead of assuming.
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest
from conftest import REGISTRY, StubGuardrails, StubRouter, StubWorkers, invoke

from supervisor.dispatch import WorkerResponse
from supervisor.guardrails import (
    GuardrailEngine,
    GuardrailResult,
    GuardrailVerdict,
    followup_answer,
)
from supervisor.nodes import (
    CARRIED_CONTEXT_NOTICE,
    DEFERRED_REQUEST_DROPPED,
)
from supervisor.registry import WorkerAgent
from supervisor.routing import RouteResult
from supervisor.settings import Settings

REQUIREMENT = REGISTRY.get("requirement-agent")

# Two agents whose scopes genuinely overlap, which is the situation the fix is
# about. These are the shipped registry's own words: the Coding Agent's scope
# names "the unit tests written alongside that code" and the Test Case Agent's
# names test generation, and both are true of "write a unit test for this".
CODING = WorkerAgent(
    id="coding-agent",
    name="Coding Agent",
    description="Implements code",
    endpoint="coding-agent",
    domain_scope=(
        "Software implementation — code generation, code review, refactoring and "
        "debugging, including the unit tests written alongside that code."
    ),
)
TESTCASE = WorkerAgent(
    id="test-case-agent",
    name="Test Case Agent",
    description="Generates test cases",
    endpoint="test-case-agent",
    domain_scope=(
        "Functional and acceptance testing — test case generation, test plans and "
        "coverage analysis for documented requirements."
    ),
)


def verdict(in_domain=True, confidence=0.95, reading="a unit test", **kw):
    return GuardrailVerdict(
        sdlc_reading=reading,
        in_domain=in_domain,
        confidence=confidence,
        reason=kw.pop("reason", "this is my subject"),
        **kw,
    )


class PerAgent:
    """A model that answers differently per candidate, and remembers who asked.

    The screen interpolates each agent's own scope into the system turn, so the
    scope text is what identifies the candidate — the same seam the production
    prompt uses, rather than a counter that would pass whatever the order.
    """

    def __init__(self, by_agent):
        self.by_agent = by_agent
        self.asked: list[str] = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        text = str(messages)
        for agent in (CODING, TESTCASE, REQUIREMENT):
            if agent.id in self.by_agent and agent.domain_scope[:45] in text:
                self.asked.append(agent.id)
                return self.by_agent[agent.id]
        raise AssertionError(f"no candidate matched the prompt: {text[:300]}")


def engine(by_agent, *, decisive=0.9, margin=0.15, threshold=0.7):
    model = PerAgent(by_agent)
    return (
        GuardrailEngine(
            model,
            [],
            threshold,
            decisive_threshold=decisive,
            contested_margin=margin,
        ),
        model,
    )


class OnceOffering(StubGuardrails):
    """Reports a second deliverable on the FIRST screen only.

    A real screen reports what it finds in the message it is given, so the turn
    that answers the held request does not find that request inside itself. The
    shared stub replays one result forever, which would be a different — and
    much more forgiving — world than production.
    """

    def screen(self, query, candidates, history=None, deadline=None):
        first = len(self.calls) == 0
        screened = super().screen(query, candidates, history=history, deadline=deadline)
        if first:
            return screened
        return replace(screened, result=replace(screened.result, additional_request=""))


# ── 1. Ownership is a comparison, not an ordering ───────────────────────────


UNIT_TEST_QUERY = (
    "Here's a customer record: name John Mercer, email john.mercer83@gmail.com, "
    "phone 555-014-2231. Can you write a unit test that uses this exact data?"
)


@pytest.mark.parametrize("order", [[CODING, TESTCASE], [TESTCASE, CODING]])
def test_which_agent_owns_a_request_does_not_depend_on_which_widget_was_open(order):
    """The reported bug, stated as a property.

    Both scopes truthfully mention tests — the Coding Agent's names the unit
    tests written alongside code, the Test Case Agent's names test generation —
    so both claim it. Before, whichever was asked first won and the other was
    never consulted.
    """
    guard, model = engine(
        {
            CODING.id: verdict(confidence=0.88, reading="a unit test"),
            TESTCASE.id: verdict(confidence=0.80, reading="a functional test case"),
        }
    )
    screened = guard.screen(UNIT_TEST_QUERY, order)

    assert screened.agent.id == "coding-agent", "the stronger claim wins, whatever the order"
    assert set(model.asked) == {"coding-agent", "test-case-agent"}, "both were asked"


def test_a_decisive_claim_still_costs_one_model_call():
    """The cost argument for the fix has to hold, or the fix is a tax.

    A candidate that is decisively the owner short-circuits exactly as before,
    so the common case is unchanged.
    """
    guard, model = engine(
        {
            CODING.id: verdict(confidence=0.97),
            TESTCASE.id: verdict(confidence=0.10, in_domain=False),
        }
    )
    screened = guard.screen("refactor this function", [CODING, TESTCASE])

    assert screened.agent.id == "coding-agent"
    assert model.asked == ["coding-agent"], "the second candidate was never asked"


def test_a_sole_reachable_agent_is_never_fanned_out_to():
    """With one candidate there is no ordering to be misled by.

    The higher bar exists to stop candidate order deciding ownership; a sole
    candidate has no rival, so a merely-confident verdict settles it and the
    user is not asked a question that has only one possible answer.
    """
    guard, model = engine({CODING.id: verdict(confidence=0.72)})
    screened = guard.screen("refactor this function", [CODING])

    assert screened.agent.id == "coding-agent"
    assert "best available match" not in screened.result.reason
    assert model.asked == ["coding-agent"]


def test_two_agents_within_the_margin_ask_the_user_which_deliverable():
    """A 0.72-to-0.71 split is a coin toss with a decimal point.

    The user is the only one who knows whether they wanted a unit test or a
    functional test case, and they can answer in one word — so they are asked,
    in terms of the deliverables rather than the agent names.
    """
    guard, _ = engine(
        {
            CODING.id: verdict(confidence=0.72, reading="a unit test"),
            TESTCASE.id: verdict(confidence=0.71, reading="a functional test case"),
        }
    )
    screened = guard.screen(UNIT_TEST_QUERY, [CODING, TESTCASE])

    assert screened.result.passed is True, "contested is a question, never a refusal"
    assert set(screened.result.contested) == {"coding-agent", "test-case-agent"}
    question = screened.result.clarification
    assert "unit test" in question and "functional test case" in question
    assert "Agent" not in question, "asked in deliverables, not agent names"


def test_a_clear_winner_outside_the_margin_is_not_contested():
    guard, _ = engine(
        {
            CODING.id: verdict(confidence=0.88, reading="a unit test"),
            TESTCASE.id: verdict(confidence=0.71, reading="a functional test case"),
        }
    )
    screened = guard.screen(UNIT_TEST_QUERY, [CODING, TESTCASE])

    assert screened.agent.id == "coding-agent"
    assert screened.result.contested == ()
    assert not screened.result.clarification


def test_a_weak_claim_is_not_dragged_into_a_contest():
    """Only claims that cleared the acting threshold can contest.

    Otherwise a 0.71-versus-0.60 pair would ask the user to choose between an
    agent that owns the request and one that nearly disclaimed it.
    """
    guard, _ = engine(
        {
            CODING.id: verdict(confidence=0.71, reading="a unit test"),
            TESTCASE.id: verdict(confidence=0.60, reading="a functional test case"),
        }
    )
    screened = guard.screen(UNIT_TEST_QUERY, [CODING, TESTCASE])

    assert screened.agent.id == "coding-agent"
    assert screened.result.contested == ()


def test_contested_routing_never_widens_the_permitted_set():
    """The candidate list is the RBAC gate's decision, so a contest is bounded.

    Whichever way the user answers, they are choosing between agents they can
    already reach — the question cannot surface an agent their role does not
    permit, because one was never a candidate.
    """
    guard, model = engine(
        {
            CODING.id: verdict(confidence=0.72, reading="a unit test"),
            TESTCASE.id: verdict(confidence=0.71, reading="a functional test case"),
        }
    )
    screened = guard.screen(UNIT_TEST_QUERY, [CODING])

    assert screened.result.contested == (), "one permitted agent cannot be contested"
    assert model.asked == ["coding-agent"]


# ── 2. A second task is offered, never assumed and never dropped ────────────


def test_a_second_deliverable_is_carried_off_the_screen():
    guard, _ = engine(
        {
            REQUIREMENT.id: verdict(
                confidence=0.95,
                reading="user stories for password reset",
                additional_request="generate the test cases for the checkout flow",
            )
        }
    )
    screened = guard.screen(
        "Write user stories for the password reset feature, and then also generate "
        "the test cases for the checkout flow.",
        [REQUIREMENT],
    )

    assert screened.result.additional_request == "generate the test cases for the checkout flow"


@pytest.mark.parametrize("found", ["", "   ", "do it", "x" * 401])
def test_an_implausible_second_request_is_ignored(found):
    """Model-written, and it becomes the *query* of a follow-up turn if accepted.

    So it is bounded here rather than trusted: too short to be a request, or
    longer than one, and it is dropped.
    """
    guard, _ = engine(
        {REQUIREMENT.id: verdict(confidence=0.95, additional_request=found)}
    )
    screened = guard.screen("write user stories", [REQUIREMENT])

    assert screened.result.additional_request == ""


def test_a_two_task_message_answers_the_first_and_offers_the_second(make_graph):
    second = "generate the test cases for the checkout flow"
    guard = StubGuardrails(
        result=GuardrailResult(True, "semantic", "in domain", additional_request=second)
    )
    workers = StubWorkers([WorkerResponse(text="Here are the user stories.")])
    graph, _ = make_graph(guardrails=guard, workers=workers)
    result = invoke(graph, "write the user stories and also " + second, thread="two-1")

    assert result["outcome"] == "answer"
    assert "Here are the user stories." in result["final_text"]
    assert second in result["final_text"], "the second task is named back to the user"
    assert "shall I go ahead" in result["final_text"]
    assert result["deferred_request"]["text"] == second
    assert len(workers.calls) == 1, "only the first task was dispatched"


def test_saying_yes_runs_the_held_task_through_the_whole_pipeline(make_graph):
    """The point of holding rather than doing: acceptance is not authorisation.

    The held text re-enters at the screen — same tier-1 rules, same semantic
    verdict, same RBAC-bounded candidates, its own routing and its own audit
    row. Nothing about having asked first makes it easier to get done.
    """
    second = "generate the test cases for the checkout flow"
    guard = OnceOffering(
        result=GuardrailResult(True, "semantic", "in domain", additional_request=second)
    )
    workers = StubWorkers(
        [WorkerResponse(text="Here are the user stories."), WorkerResponse(text="Here are the tests.")]
    )
    graph, _ = make_graph(guardrails=guard, workers=workers)
    invoke(graph, "write the user stories and also " + second, thread="two-2")
    result = invoke(graph, "yes", thread="two-2")

    assert result["outcome"] == "answer"
    assert "Here are the tests." in result["final_text"]
    assert len(workers.calls) == 2
    # The screen saw the held request, not the word "yes".
    assert guard.calls[-1] == second
    assert result.get("deferred_request") in (None, {}), "the offer is consumed"
    assert any(
        e["decision"] == "deferred_resumed" for e in result["audit_trail"]
    ), "the trail records that a held request was taken up"


def test_the_held_request_enters_the_transcript_as_the_users_own_words(make_graph):
    """A worker reading the conversation must see the request, not "yes".

    Verbatim from the message the user actually sent — the supervisor re-raises
    it, it does not paraphrase it into something the user never wrote.
    """
    second = "generate the test cases for the checkout flow"
    guard = OnceOffering(
        result=GuardrailResult(True, "semantic", "in domain", additional_request=second)
    )
    workers = StubWorkers(
        [WorkerResponse(text="stories"), WorkerResponse(text="tests")]
    )
    graph, _ = make_graph(guardrails=guard, workers=workers)
    invoke(graph, "write the user stories and also " + second, thread="two-3")
    invoke(graph, "yes", thread="two-3")

    relayed = "\n".join(
        str(m.get("content") if isinstance(m, dict) else m.content)
        for m in workers.calls[-1]["messages"]
    )
    assert second in relayed


def test_saying_no_drops_the_held_task_without_dispatching(make_graph):
    second = "generate the test cases for the checkout flow"
    guard = StubGuardrails(
        result=GuardrailResult(True, "semantic", "in domain", additional_request=second)
    )
    workers = StubWorkers([WorkerResponse(text="stories")])
    graph, _ = make_graph(guardrails=guard, workers=workers)
    invoke(graph, "write the user stories and also " + second, thread="two-4")
    result = invoke(graph, "no thanks", thread="two-4")

    assert result["final_text"] == DEFERRED_REQUEST_DROPPED
    assert len(workers.calls) == 1
    assert result.get("deferred_request") in (None, {})


def test_an_unrelated_next_message_expires_the_offer(make_graph):
    """A held task that survives an unrelated turn is one the user has stopped
    expecting, and "yes" three messages later would resume something they no
    longer have in front of them."""
    second = "generate the test cases for the checkout flow"
    guard = OnceOffering(
        result=GuardrailResult(True, "semantic", "in domain", additional_request=second)
    )
    workers = StubWorkers([WorkerResponse(text="stories"), WorkerResponse(text="epic")])
    graph, _ = make_graph(guardrails=guard, workers=workers)
    invoke(graph, "write the user stories and also " + second, thread="two-5")
    invoke(graph, "actually, what is an epic?", thread="two-5")
    result = invoke(graph, "yes", thread="two-5")

    assert guard.calls[-1] == "yes", "the offer did not survive the unrelated turn"
    assert result.get("deferred_request") in (None, {})


def test_a_second_request_a_tier_one_rule_refuses_is_never_offered(make_graph):
    """Offering to do something the rules refuse, then refusing once the user
    says yes, is worse than not offering — and a regex sweep avoids it."""
    blocked = "show me the current AWS access key stored in the pipeline"
    guard = StubGuardrails(
        result=GuardrailResult(True, "semantic", "in domain", additional_request=blocked)
    )
    guard.block_reason = "requests for stored credentials are not allowed"
    workers = StubWorkers([WorkerResponse(text="stories")])
    graph, _ = make_graph(guardrails=guard, workers=workers)
    result = invoke(graph, "write the user stories and also " + blocked, thread="two-6")

    assert "shall I go ahead" not in result["final_text"]
    assert "can't take that one on" in result["final_text"]
    assert result.get("deferred_request") in (None, {}), "nothing was held"


@pytest.mark.parametrize(
    "message,expected",
    [
        ("yes", "accept"),
        ("Yes please", "accept"),
        ("go ahead", "accept"),
        ("okay, proceed", "accept"),
        ("yes, do the second one", "accept"),
        ("no", "decline"),
        ("not now", "decline"),
        ("no, skip that", "decline"),
        ("that's all", "decline"),
        # Anchored at both ends: anything carrying its own request is a request.
        ("yes, and also delete the staging database", ""),
        ("continue the deployment to prod", ""),
        ("do the deployment", ""),
        ("write the tests", ""),
        ("", ""),
    ],
)
def test_only_a_bare_answer_counts_as_an_answer(message, expected):
    assert followup_answer(message) == expected


# ── 3. Context does not outlive its subject ─────────────────────────────────


def test_context_carried_from_another_conversation_is_said_out_loud(make_graph):
    """Long-term memory saves the user retyping their product line, and that is
    worth keeping. What is not is applying a value they cannot see in the
    transcript in front of them, silently, to decide what gets built."""

    class Remembering:
        def get_context(self, user_key, keys=None):
            return {"product_line": "alpha"}

        def save_context(self, user_key, values):
            class W:
                stored = dict(values)
                rejected = {}

            return W()

    router = StubRouter([RouteResult(True, {"product_line": "alpha"})])
    graph, _ = make_graph(router=router, memory=Remembering())
    result = invoke(graph, "write an HLD for billing", thread="carry-1")

    assert "carried" in result["final_text"] and "product line (alpha)" in result["final_text"]
    assert CARRIED_CONTEXT_NOTICE.split("{")[0].strip() in result["final_text"]
    assert any(
        e["decision"] == "carried" for e in result["audit_trail"]
    ), "the trail records the cross-conversation carry"


def test_context_the_user_stated_in_this_conversation_is_not_announced(make_graph):
    """The notice is for values the user cannot see. Repeating back what they
    typed two messages ago is noise."""
    router = StubRouter([RouteResult(True, {"product_line": "alpha"})])
    graph, _ = make_graph(router=router)
    result = invoke(graph, "write an HLD for billing on alpha", thread="carry-2")

    assert "carried" not in result["final_text"]


def test_a_change_of_subject_drops_the_pinned_context(make_graph):
    """The router says the carried values no longer describe the request, so
    they are discarded rather than dispatched on — and they do not come back."""
    router = StubRouter(
        [
            RouteResult(True, {"product_line": "alpha"}),
            RouteResult(True, {"product_line": "beta"}, dropped_context=("product_line",)),
        ]
    )
    graph, _ = make_graph(router=router)
    invoke(graph, "write an HLD for billing on alpha", thread="drop-1")
    result = invoke(graph, "now write one for the beta payments rewrite", thread="drop-1")

    assert result["session_context"] == {"product_line": "beta"}
    assert any(
        e["decision"] == "dropped" for e in result["audit_trail"]
    ), "a dropped product line changes what gets built, so the trail has to say so"


def test_a_dropped_key_with_nothing_to_replace_it_does_not_linger(make_graph):
    """Re-pinning wholesale would put back exactly what the router just decided
    the request is not about, and the drop would change nothing."""
    router = StubRouter(
        [
            RouteResult(True, {"product_line": "alpha"}),
            RouteResult(True, {}, dropped_context=("product_line",)),
        ]
    )
    graph, _ = make_graph(router=router)
    invoke(graph, "write an HLD for billing on alpha", thread="drop-2")
    result = invoke(graph, "what is an epic?", thread="drop-2")

    assert result["session_context"] == {}


def test_the_router_drops_only_what_was_carried_in():
    """Anything resolved from *this* request survives the drop."""
    from supervisor.routing import RouteDecision, Router, _ContextItem

    class Model:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return RouteDecision(
                prior_context_applies=False,
                ready=True,
                resolved_context=[_ContextItem(key="product_line", value="beta")],
            )

    result = Router(Model()).resolve(
        REQUIREMENT,
        ["write an HLD for the beta payments rewrite"],
        {"product_line": "alpha"},
        carried_over={"product_line": "alpha"},
    )

    assert result.resolved_context == {"product_line": "beta"}
    assert result.dropped_context == ("product_line",)


def test_the_router_keeps_prior_context_when_the_subject_has_not_changed():
    from supervisor.routing import RouteDecision, Router

    class Model:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return RouteDecision(prior_context_applies=True, ready=True, resolved_context=[])

    result = Router(Model()).resolve(
        REQUIREMENT,
        ["and now the acceptance criteria"],
        {"product_line": "alpha"},
        carried_over={"product_line": "alpha"},
    )

    assert result.resolved_context == {"product_line": "alpha"}
    assert result.dropped_context == ()


def test_the_router_tells_the_model_which_values_the_conversation_never_stated():
    """The two sources carry different weight, so they are named apart.

    A value the user typed in this conversation is above them in the transcript;
    a value read back from another one is not, and the model has to be able to
    treat them differently.
    """
    from supervisor.routing import RouteDecision, Router

    seen = {}

    class Model:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            seen["payload"] = str(messages[-1].content)
            return RouteDecision(ready=True, resolved_context=[])

    Router(Model()).resolve(
        REQUIREMENT,
        ["write an HLD"],
        {"product_line": "alpha"},
        carried_over={"product_line": "alpha"},
    )

    assert "carried_over_from_earlier" in seen["payload"]


def test_an_expired_session_leaves_no_held_task_behind(make_graph):
    """An offer belongs to the exchange that produced it. The user who comes
    back tomorrow is starting a new one, and "yes" then would resume something
    they no longer have in front of them."""
    second = "generate the test cases for the checkout flow"
    guard = StubGuardrails(
        result=GuardrailResult(True, "semantic", "in domain", additional_request=second)
    )
    workers = StubWorkers([WorkerResponse(text="stories")])
    graph, _ = make_graph(
        guardrails=guard, workers=workers, settings=Settings(session_max_age_seconds=0.05)
    )
    invoke(graph, "write the user stories and also " + second, thread="expire-1")
    time.sleep(0.1)
    result = invoke(graph, "yes", thread="expire-1")

    assert result.get("deferred_request") in (None, {})
    assert len(workers.calls) == 1
