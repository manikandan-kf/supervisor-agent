"""The layer-7 output guard — nothing reaches the user unchecked.

Applied in `dispatch` before worker text becomes state. The decision is
tiered rather than binary — allow, mask, block, escalate per finding category
— and these pin the two ends of it:

  * secret shapes never reach the browser or the checkpoint — a recognised
    credential withholds the reply, a secret assignment is masked with its key
    kept as evidence;
  * a deterministic output policy screen (`output_deny_patterns` in the
    governed guardrails document) — a match withholds the response with a
    governed message and records the matched reason in the trail, never in
    the user-facing text.
"""

from __future__ import annotations

from conftest import FAKE_DATABRICKS_TOKEN, StubAudit, StubWorkers, invoke

from supervisor.config_store import ConfigError, validate
from supervisor.dispatch import WorkerResponse
from supervisor.nodes import OUTPUT_WITHHELD_MESSAGE
from supervisor.output_guard import OutputGuard
from supervisor.redact import redact_text

# ── the guard itself ────────────────────────────────────────────────────────


def test_a_clean_response_is_left_alone():
    text = "# HLD\n\nThe billing service exposes three endpoints."
    result = OutputGuard().screen(text)
    assert result.text == text
    assert not result.blocked
    assert not result.modified
    assert result.action == "allow"


def test_a_credential_token_in_the_reply_withholds_it():
    """Block tier: a recognised credential is never delivered, masked or not."""
    result = OutputGuard().screen(f"Use the token {FAKE_DATABRICKS_TOKEN} here.")
    assert result.blocked
    assert result.action == "block"
    assert result.text == ""
    assert "databricks-token" in result.reason
    assert FAKE_DATABRICKS_TOKEN[:20] not in result.reason


def test_a_secret_assignment_keeps_its_key_as_evidence():
    result = OutputGuard().screen("Set DB_PASSWORD=hunter2s3cret in the env.")
    assert not result.blocked
    assert "hunter2s3cret" not in result.text
    assert "DB_PASSWORD" in result.text, "the key is evidence; only the value leaks"
    assert result.secrets_masked == 1


def test_pii_masking_is_a_policy_choice():
    text = "Contact ops@example.com for access."
    on = OutputGuard(mask_pii=True).screen(text)
    off = OutputGuard(mask_pii=False).screen(text)
    assert "ops@example.com" not in on.text
    assert on.pii_masked == 1
    assert "ops@example.com" in off.text, "an address may be the content, not a leak"
    assert off.pii_masked == 0


def test_secrets_are_acted_on_even_with_pii_masking_off():
    assignment = OutputGuard(mask_pii=False).screen("token=abcdefghijklmnop1234 mail a@b.io")
    assert "abcdefghijklmnop1234" not in assignment.text
    assert assignment.secrets_masked == 1
    assert "a@b.io" in assignment.text
    credential = OutputGuard(mask_pii=False).screen("bearer abcdefghijklmnop1234")
    assert credential.blocked


def test_a_policy_match_withholds_the_response():
    guard = OutputGuard(
        [{"pattern": r"(?i)internal use only", "reason": "classification marker in output"}]
    )
    result = guard.screen("This document is INTERNAL USE ONLY.")
    assert result.blocked
    assert result.reason == "classification marker in output"
    assert result.text == ""


def test_the_audit_detail_names_what_happened():
    detail = OutputGuard().screen("token=abcdef123456 mail a@b.io").audit_detail()
    assert "secret value(s) masked" in detail
    assert "PII value(s) masked" in detail
    assert "found: secret-assignment, email" in detail
    assert OutputGuard().screen("clean").audit_detail() == "clean"


def test_from_mapping_reads_the_governed_document_shape():
    guard = OutputGuard.from_mapping(
        {"output_deny_patterns": [{"pattern": "forbidden", "reason": "policy"}]}
    )
    assert guard.screen("this is forbidden output").blocked
    assert not OutputGuard.from_mapping({}).screen("this is forbidden output").blocked


# ── the redact tier split it relies on ──────────────────────────────────────


def test_redact_without_pii_leaves_addresses_but_not_tokens():
    text, count = redact_text(f"mail a@b.io token {FAKE_DATABRICKS_TOKEN}", pii=False)
    assert "a@b.io" in text
    assert "dapi" not in text
    assert count == 1


def test_redact_with_pii_is_the_persistence_default():
    text, count = redact_text("mail a@b.io", pii=True)
    assert "a@b.io" not in text
    assert count == 1


# ── config validation of the new sections ───────────────────────────────────


def test_output_deny_patterns_are_validated_at_publish_time():
    validate(
        "guardrails",
        {"output_deny_patterns": [{"pattern": "x", "reason": "why"}]},
    )
    try:
        validate("guardrails", {"output_deny_patterns": [{"pattern": "("}]})
    except ConfigError as exc:
        assert "output_deny_patterns[0]" in str(exc)
    else:
        raise AssertionError("an invalid output regex must refuse the publish")


# ── through the graph ───────────────────────────────────────────────────────


def test_a_worker_leaked_credential_never_reaches_the_user_or_the_transcript(make_graph):
    workers = StubWorkers(
        [WorkerResponse(text=f"Connect with {FAKE_DATABRICKS_TOKEN} today.")]
    )
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
