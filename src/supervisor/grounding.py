"""Grounding cues — what a worker *claimed* versus what the supervisor can vouch for.

Layer 7 asks for hallucination checks: "verify claims against the evidence
that was actually retrieved". The supervisor cannot do that verification — it
performs no retrieval and runs no tools, so it holds no evidence to check a
claim against. What it *can* do, deterministically and honestly, is refuse to
let two kinds of claim pass as verified when nothing in the turn verified
them:

  * **Execution claims** — "the rollback completed", "all tests passed", "I
    have deployed this". A worker that ran nothing cannot have observed any of
    these. The supervisor has no execution record for the turn unless the
    worker's wire response carries one (`custom_outputs.evidence` or
    `tool_calls`), so a bare claim is labelled as the agent's statement rather
    than a confirmed outcome.
  * **Citations without a source** — "per ticket PAY-1234", "see the
    architecture doc". The wire contract carries the sources a worker actually
    consulted (`WorkerResponse.sources`); a reference in the prose that names
    nothing in that list is unbacked, and a simulated worker's sources back
    nothing at all.

This is a *cue*, appended to the reply and recorded in the decision trail,
never a rewrite: the supervisor relays a governed worker's artifact as
produced, and a footnote that says "unverified" is the confidence-weighted
signal OWASP ASI09 asks for ("low-certainty", "unverified source"), in the
form NIST AI 600-1 frames as confabulation monitoring — surface it, record it,
do not silently present it as fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Assertions that something was *done* or *observed*. Past tense, first
# person or definitive — an HLD that says "once the migration has completed"
# is describing a future step, and the conditional openers are excluded for
# that reason. A miss here costs a missing footnote; a false positive costs a
# footnote on a sentence that did not need one. Both are cheap, and the second
# is the safer direction.
_EXECUTION_CLAIMS = (
    re.compile(
        r"(?i)\bI(?:'ve| have)?\s+(?:ran|run|executed|verified|confirmed|checked|tested|"
        r"deployed|rolled\s+back|applied|completed|validated|reviewed the logs)\b"
    ),
    re.compile(
        r"(?i)\b(?:all\s+)?(?:the\s+)?(?:tests?|test\s+suite|build|deployment|rollback|pipeline|"
        r"migration|release|job)\s+(?:has\s+|have\s+|was\s+|were\s+)?(?:passed|succeeded|"
        r"completed\s+successfully|finished\s+successfully|is\s+complete|are\s+passing|"
        r"pass(?:es)?\s+now)\b"
    ),
    re.compile(
        r"(?i)\b(?:successfully|has\s+been|have\s+been)\s+(?:deployed|rolled\s+back|completed|"
        r"applied|verified|executed|released|promoted)\b"
    ),
    re.compile(r"(?i)\b(?:confirmed|verified)\s*:\s*(?:the\s+)?(?:rollback|deployment|release|tests?)\b"),
)
_CONDITIONAL_OPENERS = re.compile(
    r"(?i)\b(?:once|after|when|if|until|before|should|assuming|provided)\b[^.\n]{0,60}$"
)
# A Gherkin line is a specification of behaviour, not a claim about what
# happened. `Given the migration has completed successfully, when the user logs
# in…` and `Then all tests passed` are what a worker asked for acceptance
# criteria produces on every line, and cueing every one of them as an
# unverified execution claim is how the cue loses its meaning.
_SPEC_LINE = re.compile(
    r"^\s*(?:[-*+>]\s*|\d+[.)]\s*|\|)?"
    r"(?:given|when|then|and|but|scenario(?:\s+outline)?|feature|background|examples?|rule)\b",
    re.IGNORECASE,
)

# References to a specific external record. Jira-style keys, ticket numbers,
# and named documents — the things a Requirement Agent would cite.
_CITATIONS = (
    # A tracker key counts only when something *cites* it. Unqualified,
    # `[A-Z]+-\d+` is the shape of every standards identifier an SDLC artifact
    # legitimately names — ISO-27001, RFC-7519, CVE-2021-44228, SHA-512,
    # AES-128, HTTP-401, SP-800 — and cueing those as unbacked sources put an
    # "unverified source" footnote on ordinary security requirements. Worse,
    # the match truncated them (`CVE-2021`), so the cue named a reference the
    # response had not made.
    re.compile(
        r"(?i)\b(?:ticket|issue|story|epic|card|task|bug|defect|jira|ref(?:erence)?|"
        r"per|see|from|documented\s+in|according\s+to|raised\s+in|tracked\s+in)\b"
        r"[^.\n]{0,30}?\b([A-Z][A-Z0-9]{1,9}-\d{1,6})\b"
    ),
    re.compile(r"(?i)\b(?:ticket|issue|story|epic|incident|change\s+request|cr)\s*#?\s*(\d{2,8})\b"),
    re.compile(r"(?i)\b(?:per|see|from|according\s+to|as\s+documented\s+in|cited\s+in)\s+(?:the\s+)?([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,4}\s+(?:doc(?:ument)?|spec(?:ification)?|ADR|RFC|page|runbook|wiki))\b"),
)
# Prefixes that name a published standard rather than a tracker item, for the
# cases where someone does write "per RFC-7519". An enumerated list is the
# right shape here — standards families are a bounded, well-known set, while
# project keys are not — and it is checked on the prefix, so RFC-7519 and
# RFC-6749 are both covered by one entry.
_STANDARDS_PREFIXES = frozenset(
    """
    ISO IEC IEEE ANSI ECMA RFC BCP STD CVE CWE CAPEC CVSS NIST SP FIPS PCI DSS PA
    SHA SHA1 SHA2 SHA3 MD AES DES RSA ECDSA ECDH HMAC TLS SSL SSH GPG PGP
    HTTP HTTPS UTF ASCII UTC ISO8601 GDPR HIPAA CCPA SOC SOX ITIL COBIT OWASP
    LLM ASI ATLAS MITRE ATT JSR PEP EIP ERC UML BPMN WCAG ARIA XSS CSRF SQL
    COVID IPV IP TCP UDP DNS SAML JWT OAUTH OIDC SCIM LDAP AMQP MQTT GRPC
    """.split()
)


def _is_standard(reference: str) -> bool:
    prefix = reference.split("-", 1)[0].upper()
    return prefix in _STANDARDS_PREFIXES


@dataclass(frozen=True)
class GroundingCheck:
    """What the reply asserted that nothing in the turn backs."""

    execution_claims: tuple[str, ...] = ()
    unbacked_citations: tuple[str, ...] = ()
    # Whether the worker supplied an execution record of its own.
    evidence_supplied: bool = False
    simulated: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def flagged(self) -> bool:
        return bool(self.execution_claims or self.unbacked_citations)

    def audit_detail(self) -> str:
        parts = []
        if self.execution_claims:
            parts.append(
                f"{len(self.execution_claims)} execution/verification claim(s) with no execution "
                "record for this turn"
            )
        if self.unbacked_citations:
            parts.append(
                f"{len(self.unbacked_citations)} citation(s) not backed by a declared source: "
                + ", ".join(self.unbacked_citations[:5])
            )
        return "; ".join(parts) or "grounded"


# What the user reads. Worded as a statement about the record, not an
# accusation about the agent: the claim may well be true — the supervisor
# simply cannot vouch for it, and says so.
UNVERIFIED_EXECUTION_NOTE = (
    "_Unverified: this response reports an action or result as completed, but no "
    "execution record for it reached the supervisor. Treat it as the agent's statement, "
    "not a confirmed outcome, and check the system of record before relying on it._"
)
UNBACKED_CITATION_NOTE = (
    "_Unverified source: this response cites {refs}, which is not among the sources the "
    "agent declared for this answer. Confirm the reference exists before relying on it._"
)


def _evidence_supplied(raw) -> bool:
    """Whether the worker declared an execution record on the wire.

    Every access is type-checked, because every value here is worker-supplied
    and this function runs *outside* the dispatch node's exception handling. A
    worker returning `"custom_outputs": ["…"]` or `"custom_outputs": "yes"`
    used to raise `AttributeError` from `.get`, which propagated out of
    `graph.stream()` — no `respond` node, so no audit row, no governed error
    message, and an HTTP 500 after the worker call had already been paid for.
    `dispatch._parse_worker_response` already guards the identical shape, so
    the asymmetry was the bug, not the input.
    """
    if not isinstance(raw, dict):
        return False
    custom = raw.get("custom_outputs")
    if not isinstance(custom, dict):
        return False
    return any(
        custom.get(key)
        for key in ("evidence", "tool_calls", "execution", "actions", "verification")
    )


def _sources_text(sources) -> str:
    if not isinstance(sources, (list, tuple)):
        return ""
    parts = []
    for source in sources:
        if isinstance(source, dict):
            parts.extend(str(v) for v in source.values() if v)
        else:
            parts.append(str(source))
    return " ".join(parts).lower()


def _is_simulated(sources) -> bool:
    if not isinstance(sources, (list, tuple)):
        return False
    return any(
        isinstance(s, dict) and str(s.get("origin", "")).lower() == "simulated"
        for s in sources
    )


def check(text, sources=None, raw=None) -> GroundingCheck:
    """Assess one worker reply. Cheap, deterministic, never raises.

    "Never raises" is a contract, not a hope: the call site in
    `nodes.dispatch` sits after the worker's `except Exception` has closed and
    the graph declares no `RetryPolicy`, so anything escaping here becomes an
    unaudited 500. Every argument is worker-controlled and every access below
    is typed accordingly.
    """
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    evidence = _evidence_supplied(raw or {})
    simulated = _is_simulated(sources)

    claims: list[str] = []
    if not evidence:
        for pattern in _EXECUTION_CLAIMS:
            for match in pattern.finditer(text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                if _SPEC_LINE.match(text[line_start : match.start() + 1]):
                    continue
                lead = text[max(0, match.start() - 60) : match.start()]
                if _CONDITIONAL_OPENERS.search(lead):
                    continue
                claims.append(match.group(0).strip())

    declared = _sources_text(sources)
    citations: list[str] = []
    for pattern in _CITATIONS:
        for match in pattern.finditer(text):
            ref = (match.group(1) if match.groups() else match.group(0)).strip()
            if not ref or _is_standard(ref):
                continue
            # A key that is a redaction placeholder or a version is not a citation.
            if ref.lower() in declared and not simulated:
                continue
            if ref not in citations:
                citations.append(ref)

    notes: list[str] = []
    if claims:
        notes.append(UNVERIFIED_EXECUTION_NOTE)
    if citations:
        shown = ", ".join(f"`{c}`" for c in citations[:3])
        if len(citations) > 3:
            shown += f" and {len(citations) - 3} more"
        notes.append(UNBACKED_CITATION_NOTE.format(refs=shown))

    return GroundingCheck(
        execution_claims=tuple(dict.fromkeys(claims)),
        unbacked_citations=tuple(citations),
        evidence_supplied=evidence,
        simulated=simulated,
        notes=tuple(notes),
    )


def annotate(text: str, result: GroundingCheck) -> str:
    """Append the cue(s) to a reply. The artifact itself is untouched."""
    if not result.notes:
        return text
    return (text or "").rstrip() + "\n\n" + "\n\n".join(result.notes)
