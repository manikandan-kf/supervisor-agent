"""The layer-7 output guard, through the graph — nothing reaches the user unchecked.

Applied in `dispatch` before worker text becomes state: a recognised credential
withholds the reply, a secret assignment is masked with its key kept as
evidence, a governed output rule withholds with a governed message, and a
staged artifact goes through the same screen as a delivered answer. The guard
itself is tested in `libs/agent_governance/tests/test_output_guard.py`.
"""

from __future__ import annotations

from agent_governance.output_guard import OutputGuard
from helpers import FAKE_DATABRICKS_TOKEN, StubAudit, StubWorkers, invoke

from supervisor.messages import OUTPUT_WITHHELD_MESSAGE
from supervisor.worker_client import WorkerResponse


def test_a_worker_leaked_credential_never_reaches_the_user_or_the_transcript(make_graph):
    workers = StubWorkers([WorkerResponse(text=f"Connect with {FAKE_DATABRICKS_TOKEN} today.")])
    graph, _ = make_graph(workers=workers)
    result = invoke(graph, "write an HLD for billing on alpha")

    assert result["outcome"] == "blocked"
    assert FAKE_DATABRICKS_TOKEN not in result["final_text"]
    assert OUTPUT_WITHHELD_MESSAGE in result["final_text"]
    # The message that entered conversation history — what the checkpointer
    # persists and later turns replay — is the withheld message, never the raw reply.
    assert FAKE_DATABRICKS_TOKEN[:20] not in result["messages"][-1].content


def test_a_masked_secret_assignment_is_what_enters_the_transcript(make_graph):
    workers = StubWorkers([WorkerResponse(text="Use DB_PASSWORD=hunter2s3cret to connect.")])
    graph, _ = make_graph(workers=workers)
    result = invoke(graph, "write an HLD for billing on alpha")

    assert result["outcome"] == "answer"
    assert "hunter2s3cret" not in result["final_text"]
    assert "DB_PASSWORD=[redacted:secret-assignment]" in result["final_text"]
    assert "hunter2s3cret" not in result["messages"][-1].content


def test_masking_is_recorded_in_the_decision_trail(make_graph):
    audit = StubAudit()
    workers = StubWorkers([WorkerResponse(text="set api_key=abcdef123456 in the env")])
    graph, _ = make_graph(workers=workers, audit=audit)
    invoke(graph, "write an HLD for billing on alpha")

    trail = audit.records[-1]["decision_trail"]
    entry = next(e for e in trail if e["stage"] == "output_guard" and e["decision"] == "masked")
    assert "secret value(s) masked" in entry["detail"]


def test_a_clean_response_adds_no_output_guard_noise(make_graph):
    audit = StubAudit()
    graph, _ = make_graph(audit=audit)
    invoke(graph, "write an HLD for billing on alpha")
    trail = audit.records[-1]["decision_trail"]
    assert not [e for e in trail if e["stage"] == "output_guard"]


def test_a_policy_match_blocks_delivery_without_leaking_the_rule(make_graph):
    audit = StubAudit()
    guard = OutputGuard([{"pattern": "(?i)do not ship", "reason": "release-block marker"}])
    workers = StubWorkers([WorkerResponse(text="Draft ready. DO NOT SHIP without review.")])
    graph, _ = make_graph(workers=workers, audit=audit, output_guard=guard)
    result = invoke(graph, "write an HLD for billing on alpha")

    assert result["outcome"] == "blocked"
    assert OUTPUT_WITHHELD_MESSAGE in result["final_text"]
    assert "release-block marker" not in result["final_text"], "the rule stays in the trail"
    trail = audit.records[-1]["decision_trail"]
    entry = next(e for e in trail if e["stage"] == "output_guard")
    assert entry["decision"] == "withheld"
    assert "release-block marker" in entry["detail"]


def test_a_withheld_response_offers_the_appeal_path(make_graph):
    """§06: a block offers a way forward that is not retrying the same text."""
    guard = OutputGuard([{"pattern": "forbidden", "reason": "policy"}])
    workers = StubWorkers([WorkerResponse(text="forbidden content")])
    reviews_graph, services = make_graph(workers=workers, output_guard=guard)
    invoke(reviews_graph, "write an HLD for billing on alpha")

    appealed = invoke(reviews_graph, "appeal")
    assert appealed["outcome"] == "escalated"
    assert services.reviews.opened, "the appeal must land in the review queue"
    assert "policy" in services.reviews.opened[0].reason


def test_a_staged_artifact_goes_through_the_same_screen(make_graph):
    """An approval artifact is shown to a human approver — a policy match must
    stop it from being staged at all, not surface it for sign-off."""
    guard = OutputGuard([{"pattern": "forbidden", "reason": "policy"}])
    workers = StubWorkers(
        [WorkerResponse(text="forbidden draft", status="approval_pending", stage="HLD")]
    )
    graph, _ = make_graph(workers=workers, output_guard=guard)
    result = invoke(graph, "write an HLD for billing on alpha")

    assert result["outcome"] == "blocked"
    assert not result.get("pending_approval"), "a withheld artifact is never staged"


def test_a_staged_artifact_is_masked_before_the_approver_sees_it(make_graph):
    workers = StubWorkers(
        [
            WorkerResponse(
                text="HLD draft. token=abcdef123456", status="approval_pending", stage="HLD"
            )
        ]
    )
    graph, _ = make_graph(workers=workers)
    result = invoke(graph, "write an HLD for billing on alpha")
    assert "abcdef123456" not in result["pending_approval"]["artifact"]
