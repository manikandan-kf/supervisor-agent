"""The sensitive-data catalogue — what it must catch, what it must not.

`sensitive.py` is read by three boundaries (the persistence redactor, the
worker relay, the output guard), so a shape pinned here is pinned for all
three. The attack strings are worst-case worker replies; the benign strings
are the SDLC prose this platform carries every day, and every one of them must
come back untouched — a placeholder in an HLD a human was meant to read is an
over-redaction failure, and it costs the reader more than the masking saved.
"""

from __future__ import annotations

import pytest
from agent_governance import sensitive
from agent_governance.sensitive import (
    iban_valid,
    luhn_valid,
    nhs_valid,
    nino_valid,
    scan,
    ssn_valid,
)
from fixtures import FAKE_AWS_KEY_ID, FAKE_AWS_SECRET, FAKE_GITHUB_TOKEN, FAKE_GOOGLE_API_KEY


def labels(text: str) -> list[str]:
    return [f.label for f in scan(text)]


def categories(text: str) -> set[str]:
    return {f.category for f in scan(text)}


# ── validators ──────────────────────────────────────────────────────────────


def test_luhn_accepts_the_industry_test_numbers_and_rejects_a_digit_run():
    for card in ("4242424242424242", "4000 0566 5566 5556", "378282246310005", "6011111111111117"):
        assert luhn_valid(card), card
    assert not luhn_valid("4242424242424241")
    assert not luhn_valid("1234567890123456")


def test_iban_mod97_and_country_length():
    assert iban_valid("GB29 NWBK 6016 1331 9268 19")
    assert iban_valid("DE89370400440532013000")
    assert not iban_valid("GB29 NWBK 6016 1331 9268 18"), "checksum off by one"
    assert not iban_valid("GB29NWBK60161331926819123"), "wrong length for GB"


def test_ssn_issuance_rules():
    assert ssn_valid("123-45-6789"), "the sample number is still masked — over-masking is free"
    for bad in ("000-12-3456", "666-12-3456", "912-34-5678", "123-00-4567", "123-45-0000", "111-11-1111"):
        assert not ssn_valid(bad), bad


def test_nhs_mod11():
    assert nhs_valid("943 476 5919")
    assert not nhs_valid("943 476 5918")


def test_nino_prefix_rules():
    assert nino_valid("AB123456C")
    assert not nino_valid("QQ123456C"), "Q is never a prefix letter"
    assert not nino_valid("BG123456C"), "BG is an excluded prefix"
    assert not nino_valid("AB123456E"), "suffix must be A-D"


# ── worst-case replies ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        # A name, email and phone in a test fixture.
        (
            "name John Mercer, email john.mercer83@gmail.com, phone 555-014-2231",
            {"person", "email", "phone"},
        ),
        # An SSN in a test case description.
        ("customer: Jane Whitfield, SSN 123-45-6789", {"person", "us-ssn"}),
        # A date of birth in a requirement summary.
        ("Customer Alex Rivera, DOB 1990-04-12, called about billing", {"person", "date-of-birth"}),
        # Credentials in a connection string — the host survives.
        ("postgres://admin:Sup3rSecret!@db.internal.example.com:5432/appdb", {"url-credentials"}),
        # A GitHub token.
        (FAKE_GITHUB_TOKEN, {"github-token"}),
        # An AWS key pair, the secret in prose next to its context word.
        (
            f"AWS_ACCESS_KEY_ID={FAKE_AWS_KEY_ID}; the aws secret is "
            f"{FAKE_AWS_SECRET}",
            {"aws-key-id", "aws-secret-key"},
        ),
        # A record number, a diagnosis and a medication.
        (
            "patient ID MRN-88213, diagnosed with Type 2 Diabetes, currently on Metformin 500mg",
            {"medical-record-number", "health-condition", "medication"},
        ),
        # A named patient and a treatment.
        (
            "patient Sarah Kim, treatment plan for stage 2 hypertension",
            {"person", "health-condition"},
        ),
        # A fixture object with a record id in patient context.
        (
            "{patient: 'Tom Reyes', diagnosis: 'HIV positive', record_id: 445521}",
            {"person", "health-condition", "patient-record-id"},
        ),
        # A server path and a cluster-local hostname.
        (
            'File "/opt/app/internal/services/auth_service.py", line 214: could not reach '
            "auth-internal.prod.svc.cluster.local",
            {"server-path", "internal-hostname"},
        ),
        # A private IP with a port.
        ("calls http://10.0.4.12:8080/health", {"private-ip"}),
        # A placeholder-looking password is still a password.
        ("DB_PASSWORD=changeme123", {"secret-assignment"}),
        # Luhn-valid card numbers.
        ("4242424242424242 and 4111 1111 1111 1111", {"card-number"}),
        # The last-four fragment and the expiry next to card context.
        (
            "Last 4 digits: 4242, expiration month: 09/27 for the card on file",
            {"card-fragment", "card-expiry"},
        ),
        # A secret handed over in prose, with no `=` for the assignment rule.
        ("the token dGVzdC1zZWNyZXQtdmFsdWU= is invalid", {"prose-secret"}),
        # Bank and government identifiers with checksums.
        ("Pay to IBAN GB29 NWBK 6016 1331 9268 19", {"iban"}),
        ("NI number AB123456C", {"uk-nino"}),
        ("NHS number 943 476 5919", {"nhs-number"}),
        # Other credential formats gitleaks names.
        (f"key {FAKE_GOOGLE_API_KEY}", {"google-api-key"}),
        ("bearer abcdefghijklmnop1234", {"bearer-token"}),
    ],
)
def test_worst_case_replies_are_recognised(text, expected):
    assert expected <= set(labels(text)), f"{text!r} -> {labels(text)}"


def test_the_placeholder_replaces_only_the_value():
    text, findings = sensitive.redact("Set DB_PASSWORD=hunter2s3cret in the env.")
    assert text == "Set DB_PASSWORD=[redacted:secret-assignment] in the env."
    assert len(findings) == 1


def test_url_credentials_keep_the_host_as_evidence():
    text, _ = sensitive.redact("postgres://admin:Sup3rSecret!@db.internal.example.com:5432/appdb")
    assert text == "postgres://[redacted:url-credentials]@db.internal.example.com:5432/appdb"


def test_a_second_pass_is_a_no_op():
    once, first = sensitive.redact("token=abcdef123456 and 4242424242424242")
    twice, second = sensitive.redact(once)
    assert twice == once
    assert not second, "a placeholder must not be re-matched on the next pass"


def test_a_credential_that_is_also_an_assignment_is_reported_as_the_credential():
    findings = scan(f"AWS_SECRET_ACCESS_KEY={FAKE_AWS_SECRET}")
    assert [f.label for f in findings] == ["aws-secret-key"]
    assert findings[0].category == sensitive.CREDENTIAL


def test_categories_are_the_policy_vocabulary():
    assert categories("SSN 123-45-6789") == {sensitive.GOVERNMENT_ID}
    assert categories("ops@example.com") == {sensitive.CONTACT}
    assert categories("auth-internal.prod.svc.cluster.local") == {sensitive.NETWORK}
    assert sensitive.NEVER_ALLOW <= set(sensitive.CATEGORIES)
    assert sensitive.PII_TIER.isdisjoint(sensitive.NEVER_ALLOW)


# ── what must never match ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "The add function returns the sum of a and b.",
        "User Story: As a Customer Support Team member, I want the Customer Data Model to show "
        "User Acceptance Testing results.",
        "Deployed build 2024.09.10-1234 to src/app/main.py; commit "
        "3f2a9c1e5b7d8e9f0a1b2c3d4e5f60718293a4b5; version 1.2.3; run at 2026-09-10 14:00.",
        "The API returns a token bucket rate limit of 1000 requests. Password requirements: 12 chars.",
        "Contact api.example.com or localhost:8080; see https://docs.example.com/guide.",
        "Mr. Requirements Agent handles user stories; the Test Case Agent owns test plans.",
        "Order 123456789 shipped on 2026-01-05; ticket PAY-4242; release 2026-09-01.",
        "Epoch 1736294400 was the cutoff; the sprint runs 2026-09-01 to 2026-09-14.",
        "Requirement REQ-1234: the diabetes tracker shall store readings.",
        "record_id: 1 is the fixture row; case_id: 42 is the smoke test.",
        "Set LOG_LEVEL=info and PORT=8080; the token_count field holds an integer.",
        "The migration touched /src/migrations/0002_add_index.py and 8.8.8.8 is the DNS example.",
    ],
)
def test_ordinary_sdlc_prose_is_left_alone(text):
    assert scan(text) == [], f"false positive: {[f.label for f in scan(text)]} in {text!r}"


def test_a_person_needs_a_role_word():
    """Unanchored names are the recorded residual: NER, not regex."""
    assert labels("The reviewer was Jane Whitfield.") == []
    assert labels("Reviewer: Jane Whitfield; approver: Tom Reyes.") == []
    assert labels("customer Jane Whitfield") == ["person"]


@pytest.mark.parametrize(
    "fixture",
    [
        # The form a name actually arrives in — a dict or JSON literal, where
        # the closing quote of the key sits before the colon. A single optional
        # quote in the separator never matched this, and a name in that form
        # then travelled through the whole stack in the clear.
        "customer = {'name': 'John Mercer', 'city': 'Leeds'}",
        '{"customer": "Jane Whitfield"}',
        "{'patient': \"Tom Reyes\"}",
        'payload = {"full name": "John Mercer"}',
        "name: John Mercer",
        "Customer: Jane Whitfield",
    ],
)
def test_a_name_in_a_quoted_key_fixture_is_found(fixture):
    assert "person" in labels(fixture), f"{fixture!r} -> {labels(fixture)}"
