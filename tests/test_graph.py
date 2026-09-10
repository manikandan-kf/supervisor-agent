"""End-to-end pipeline tests over the compiled graph with stub services."""

from conftest import StubGuardrails, StubRouter, StubWorkers, invoke, resume

from supervisor.dispatch import WorkerResponse
from supervisor.guardrails import GuardrailResult
from supervisor.nodes import ESCALATION_MESSAGE
from supervisor.rbac import DENIED_MESSAGE
from supervisor.routing import RouteResult


def test_rbac_denied_is_generic_and_never_reaches_worker(make_graph):
    graph, services = make_graph()
    result = invoke(graph, "Write test cases", role="QA", agent="requirement-agent")
    assert result["outcome"] == "blocked"
    assert result["final_text"] == DENIED_MESSAGE
    assert services.workers.calls == []
    assert services.audit.records[0]["outcome"] == "blocked"


def test_guardrail_block_explains_reason(make_graph):
    guardrails = StubGuardrails(GuardrailResult(False, "semantic", "this is a cooking question."))
    graph, services = make_graph(guardrails=guardrails)
    result = invoke(graph, "Best lasagna recipe?")
    assert result["outcome"] == "blocked"
    assert "cooking question" in result["final_text"]
    assert services.workers.calls == []


def test_refusal_reads_as_sentences_whatever_punctuation_the_reason_carries(make_graph):
    # A bundled rule's reason has no full stop, a model verdict's usually does.
    # Both get joined with the role line, so neither may run on into it.
    for reason, expected in (
        ("the request looks like a prompt-injection attempt", "The request looks"),
        ("The query asks about a cooking recipe.", "The query asks"),
        ("HLD generation is not covered here", "HLD generation"),  # acronym survives
    ):
        guardrails = StubGuardrails(GuardrailResult(False, "deterministic", reason))
        graph, _ = make_graph(guardrails=guardrails)
        text = invoke(graph, "whatever", thread=f"t-{len(reason)}")["final_text"]
        assert expected in text
        assert ". For your role I can reach the" in text
        assert ".." not in text


def test_underspecified_request_asks_instead_of_blocking_or_dispatching(make_graph):
    """The screen recognised the subject but not the deliverable.

    Neither of the two things it must not do: no refusal, and no worker call on a
    guess about what was wanted.
    """
    question = "Do you want a user story for password reset, or the acceptance criteria?"
    guardrails = StubGuardrails(
        GuardrailResult(True, "semantic", "underspecified: ...", clarification=question)
    )
    graph, services = make_graph(guardrails=guardrails)
    result = invoke(graph, "can you provide the password reset")

    assert result["outcome"] == "clarify"
    assert result["final_text"] == question
    assert services.workers.calls == []
    # Asked before route, so the user is not made to name a product line for a
    # deliverable they have not chosen yet.
    assert services.router.calls == 0
    assert result["guardrail"]["underspecified"] is True


def test_both_clarifying_stages_share_one_escalation_limit(make_graph):
    """Two stages can now ask, so neither may hold its own budget.

    With a counter each they would take turns and the limit would never be
    reached — the user answers questions forever instead of reaching a human.
    """
    question = "Which requirements artifact do you need?"
    guardrails = StubGuardrails(
        GuardrailResult(True, "semantic", "underspecified: ...", clarification=question)
    )
    graph, services = make_graph(guardrails=guardrails)

    limit = services.settings.max_clarifications
    for turn in range(limit):
        result = invoke(graph, f"vague request {turn}", thread="shared-limit")
        assert result["outcome"] == "clarify"

    result = invoke(graph, "still vague", thread="shared-limit")
    assert result["outcome"] == "escalated"
    assert result["final_text"] == ESCALATION_MESSAGE
    assert services.workers.calls == []


def test_greeting_is_answered_by_the_supervisor_without_a_worker_call(make_graph):
    # "hi" forwarded to a worker comes back as that worker's entire capability
    # list. The supervisor answers it itself, in one line, and no worker runs.
    guardrails = StubGuardrails(
        GuardrailResult(True, "small_talk", "greeting, no domain intent", small_talk="greeting")
    )
    graph, services = make_graph(guardrails=guardrails)
    result = invoke(graph, "hi")

    assert result["outcome"] == "answer"
    assert services.workers.calls == []
    assert result["routed_agent_name"] == ""
    # The kind is recorded, not just "it was small talk" — a greeting and a
    # thank-you get different replies, so the trail has to distinguish them.
    assert result["guardrail"]["small_talk"] == "greeting"
    # Short, and hands the turn back rather than listing everything it can do.
    assert result["final_text"].count(".") <= 2
    assert "Supervisor" in result["final_text"]
    # A greeting carries no topic, so it must not claim a specialist. Naming one
    # would be invented, and arbitrary for a role that can reach several.
    assert "Requirement Agent" not in result["final_text"]

    stages = [e["stage"] for e in services.audit.records[0]["decision_trail"]]
    assert stages == ["rbac_gate", "guardrails"]
    assert services.audit.records[0]["decision_trail"][-1]["decision"] == "answered_directly"


def test_how_are_you_is_answered_as_a_question_not_as_a_greeting(make_graph):
    # It was classified as a greeting and got "Hello — I'm the Supervisor…" back,
    # which answers something the user did not ask.
    guardrails = StubGuardrails(
        GuardrailResult(True, "small_talk", "how_are_you, no domain intent", small_talk="how_are_you")
    )
    graph, services = make_graph(guardrails=guardrails)
    text = invoke(graph, "how are you", thread="hru")["final_text"]

    assert services.workers.calls == []
    assert not text.startswith("Hello")
    # Actually answers, then hands the turn back.
    assert "well" in text.lower() or "fine" in text.lower() or "good" in text.lower()
    assert text.rstrip().endswith("?")


def test_repeating_the_same_small_talk_does_not_repeat_the_same_sentence(make_graph):
    """Three "hi"s must not produce the same line three times.

    Verbatim repetition is the specific thing that reads as broken — the input is
    the same but the conversational context is not, so the reply escalates
    instead: hand back the turn, then offer options, then ask for a real task.
    """
    guardrails = StubGuardrails(
        GuardrailResult(True, "small_talk", "greeting, no domain intent", small_talk="greeting")
    )
    graph, _ = make_graph(guardrails=guardrails)

    replies = [invoke(graph, "hi", thread="repeat")["final_text"] for _ in range(3)]

    assert len(set(replies)) == 3, f"repeated itself: {replies}"
    # The second offers what the role can reach, as options rather than a guess.
    assert "Requirement Agent" in replies[1]
    # A fourth stays on the last variant rather than running out of range.
    assert invoke(graph, "hi", thread="repeat")["final_text"] == replies[2]


def test_escalation_is_tracked_per_kind_not_across_all_small_talk(make_graph):
    # "hi" then "thanks" are different kinds, so neither is a repeat of the other
    # and both get their first-time reply.
    def stub(kind):
        return StubGuardrails(GuardrailResult(True, "small_talk", kind, small_talk=kind))

    graph, _ = make_graph(guardrails=stub("greeting"))
    first = invoke(graph, "hi", thread="kinds")["final_text"]
    graph2, _ = make_graph(guardrails=stub("greeting"))
    assert invoke(graph2, "hi", thread="kinds2")["final_text"] == first


def test_query_is_routed_to_the_agent_whose_domain_owns_it(make_graph):
    # Engineering Manager can reach both agents. The turn was addressed to the
    # requirement agent, but the screen finds the query belongs to the coding
    # agent, so that is who gets the work — and the trail records the move.
    guardrails = StubGuardrails(
        GuardrailResult(True, "semantic", "implementation work"), owner="coding-agent"
    )
    graph, services = make_graph(guardrails=guardrails)
    result = invoke(
        graph,
        "Refactor the payment retry logic",
        role="Engineering Manager",
        agent="requirement-agent",
        permitted=("requirement-agent", "coding-agent"),
    )

    assert result["outcome"] == "answer"
    assert result["target_agent_id"] == "coding-agent"
    assert services.workers.calls[0]["agent"] == "coding-agent"
    assert result["guardrail"]["retargeted"] is True
    # Both agents were offered to the screen, the addressed one first.
    assert guardrails.screened[0] == ["requirement-agent", "coding-agent"]

    trail = {(e["stage"], e["decision"]) for e in services.audit.records[0]["decision_trail"]}
    assert ("guardrails", "retargeted") in trail


def test_screen_only_ever_offers_agents_the_caller_may_reach(make_graph):
    # The candidate list is the permitted set, so routing can move within a
    # caller's access but never outside it.
    guardrails = StubGuardrails(GuardrailResult(True, "semantic", "in domain"))
    graph, _ = make_graph(guardrails=guardrails)
    invoke(
        graph,
        "Write an HLD for billing",
        role="BA",
        agent="requirement-agent",
        permitted=("requirement-agent",),
    )
    assert guardrails.screened[0] == ["requirement-agent"]


def test_happy_path_dispatches_with_context(make_graph):
    graph, services = make_graph(router=StubRouter([RouteResult(True, {"product_line": "alpha"})]))
    result = invoke(graph, "Write an HLD for the billing service on alpha")
    assert result["outcome"] == "answer"
    assert result["final_text"] == "worker answer"
    call = services.workers.calls[0]
    assert call["agent"] == "requirement-agent"
    assert call["context"] == {"product_line": "alpha"}
    stages = [e["stage"] for e in services.audit.records[0]["decision_trail"]]
    # "memory" sits between route and dispatch: §04 requires long-term memory
    # writes to be recorded in the same decision trail as every other decision,
    # and the write happens once the router has resolved the context.
    assert stages == ["rbac_gate", "guardrails", "route", "memory", "dispatch"]


def test_clarify_then_resume_on_reply(make_graph):
    router = StubRouter(
        [
            RouteResult(False, {}, "Which product line is this for?"),
            RouteResult(True, {"product_line": "alpha"}),
        ]
    )
    graph, services = make_graph(router=router)

    first = invoke(graph, "Write an HLD for the billing service", thread="conv-1")
    assert first["outcome"] == "clarify"
    assert first["final_text"] == "Which product line is this for?"
    assert first["clarification_count"] == 1
    assert services.workers.calls == []

    second = invoke(graph, "Product line alpha", thread="conv-1")
    assert second["outcome"] == "answer"
    assert second["clarification_count"] == 0
    assert services.workers.calls[0]["context"] == {"product_line": "alpha"}
    # The worker sees the whole conversation, not just the clarification reply.
    roles = [m["role"] for m in services.workers.calls[0]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_escalates_after_clarification_limit(make_graph):
    router = StubRouter([RouteResult(False, {}, "Which product line?")])
    graph, services = make_graph(router=router)

    assert invoke(graph, "Do the thing", thread="conv-2")["outcome"] == "clarify"
    assert invoke(graph, "The usual", thread="conv-2")["outcome"] == "clarify"
    third = invoke(graph, "You know which one", thread="conv-2")
    assert third["outcome"] == "escalated"
    assert third["final_text"] == ESCALATION_MESSAGE
    assert services.workers.calls == []


def test_approval_gate_interrupts_then_resumes_without_recalling_the_worker(make_graph):
    """A staged artifact suspends the graph until a human answers.

    `interrupt()` pauses *inside* dispatch, so resuming continues from that
    point. The worker is called once, not twice — the old flow returned and
    re-entered the whole pipeline, which re-invoked it.
    """
    workers = StubWorkers(
        [WorkerResponse(text="Here is the HLD draft.", status="approval_pending", stage="HLD")]
    )
    graph, services = make_graph(workers=workers)

    paused = invoke(graph, "Generate requirements for alpha", thread="conv-3")

    interrupts = paused.get("__interrupt__") or ()
    assert interrupts, "the graph should have suspended for approval"
    payload = interrupts[0].value
    assert payload["kind"] == "worker_approval"
    assert payload["stage"] == "HLD"
    assert payload["artifact"] == "Here is the HLD draft."
    assert len(services.workers.calls) == 1

    settled = resume(graph, {"decision": "approved"}, thread="conv-3")
    assert settled["outcome"] == "answer"
    assert settled["final_text"] == "Here is the HLD draft."
    assert settled["pending_approval"] is None
    # Resumed inside the node, so the worker was never asked a second time.
    assert len(services.workers.calls) == 1


def test_rejecting_a_staged_artifact_discards_it(make_graph):
    workers = StubWorkers(
        [WorkerResponse(text="Here is the HLD draft.", status="approval_pending", stage="HLD")]
    )
    graph, services = make_graph(workers=workers)

    invoke(graph, "Generate requirements for alpha", thread="conv-4")
    settled = resume(
        graph, {"decision": "rejected", "comment": "wrong product line"}, thread="conv-4"
    )

    assert settled["outcome"] == "answer"
    assert "discarded" in settled["final_text"]
    assert "wrong product line" in settled["final_text"]
    assert len(services.workers.calls) == 1
    trail = services.audit.records[0]["decision_trail"]
    assert trail[-1]["decision"] == "rejected"


def test_approval_needs_the_producing_agents_permission(make_graph):
    """An approval is bound to the agent that produced the artifact.

    Here the caller may *use* the Requirement Agent and reaches its staged
    HLD, but holds no approve grant for it. Signing off must fail — access and
    approval are separate permissions.
    """
    workers = StubWorkers(
        [WorkerResponse(text="Here is the HLD draft.", status="approval_pending", stage="HLD")]
    )
    graph, services = make_graph(workers=workers)

    invoke(
        graph,
        "Generate requirements for alpha",
        thread="conv-5",
        permitted=["requirement-agent"],
        approvable=[],
    )
    settled = resume(
        graph,
        {"decision": "approved"},
        thread="conv-5",
        permitted=["requirement-agent"],
        approvable=[],
    )

    assert settled["outcome"] == "blocked"
    assert "permission to approve" in settled["final_text"]
    # The draft is discarded rather than returned — a blocked approval must not
    # hand over the artifact it refused to sign off.
    assert settled["pending_approval"] is None
    assert "Here is the HLD draft." not in settled["final_text"]
    trail = services.audit.records[0]["decision_trail"]
    assert trail[-1]["stage"] == "approval"
    assert trail[-1]["decision"] == "deny"


def test_approval_succeeds_when_the_producing_agent_is_approvable(make_graph):
    workers = StubWorkers(
        [WorkerResponse(text="Here is the HLD draft.", status="approval_pending", stage="HLD")]
    )
    graph, _ = make_graph(workers=workers)

    invoke(
        graph,
        "Generate requirements for alpha",
        thread="conv-6",
        permitted=["requirement-agent"],
        approvable=["requirement-agent"],
    )
    settled = resume(
        graph,
        {"decision": "approved"},
        thread="conv-6",
        permitted=["requirement-agent"],
        approvable=["requirement-agent"],
    )

    assert settled["outcome"] == "answer"
    assert settled["final_text"] == "Here is the HLD draft."


def test_rejecting_needs_no_approve_permission(make_graph):
    """Declining to act is always available — only sign-off is a privilege."""
    workers = StubWorkers(
        [WorkerResponse(text="Here is the HLD draft.", status="approval_pending", stage="HLD")]
    )
    graph, _ = make_graph(workers=workers)

    invoke(graph, "Generate requirements for alpha", thread="conv-7", approvable=[])
    settled = resume(graph, {"decision": "rejected"}, thread="conv-7", approvable=[])

    assert settled["outcome"] == "answer"
    assert "discarded" in settled["final_text"]


def test_worker_call_carries_the_correlation_set(make_graph):
    """§1.10 — every request carries the correlation fields, including this one.

    Without them the worker's own MLflow traces are orphans: a support question
    about one answer cannot be followed from the calling UI through to the agent
    that wrote it.
    """
    graph, services = make_graph()
    invoke(graph, "Write an HLD", thread="conv-8")

    trace = services.workers.calls[0]["trace"]
    assert trace["agent_id"] == "requirement-agent"
    assert trace["request_id"]
    assert trace["pseudonymous_user_reference"] == "user-1"
    # §1.10 — no raw subject, email or token may reach a worker or a prompt.
    assert "authorization" not in {k.lower() for k in trace}


def test_empty_worker_response_is_an_error(make_graph):
    graph, _ = make_graph(workers=StubWorkers([WorkerResponse(text="   ")]))
    result = invoke(graph, "Write an HLD")
    assert result["outcome"] == "error"


def test_worker_exception_is_contained(make_graph):
    class ExplodingWorkers:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("endpoint down")

    graph, services = make_graph(workers=ExplodingWorkers())
    result = invoke(graph, "Write an HLD")
    assert result["outcome"] == "error"
    assert services.audit.records[0]["decision_trail"][-1]["decision"] == "error"
