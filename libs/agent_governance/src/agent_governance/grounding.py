"""Grounding cues — what a worker *claimed* versus what the supervisor can vouch for.

Layer 7 asks for hallucination checks, but the supervisor retrieves nothing and runs no tools, so
it has no evidence to verify against. What it can do deterministically is refuse to let two kinds
of claim pass as verified: execution claims ("all tests passed") with no execution record on the
wire (`custom_outputs.evidence` / `tool_calls`), and citations naming nothing in the declared
`sources`. The cue is appended and recorded in the trail, never a rewrite: the artifact is relayed
as produced, carrying the "unverified" signal OWASP ASI09 and NIST AI 600-1 ask for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Assertions that something was *done* or *observed*; conditional openers are excluded because
# "once the migration has completed" is a future step. A false positive costs only a footnote.
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
    re.compile(
        r"(?i)\b(?:confirmed|verified)\s*:\s*(?:the\s+)?(?:rollback|deployment|release|tests?)\b"
    ),
)
_CONDITIONAL_OPENERS = re.compile(
    r"(?i)\b(?:once|after|when|if|until|before|should|assuming|provided)\b[^.\n]{0,60}$"
)
# A Gherkin line specifies behaviour rather than claiming it happened; cueing every `Then all
# tests passed` in acceptance criteria is how the cue loses its meaning.
_SPEC_LINE = re.compile(
    r"^\s*(?:[-*+>]\s*|\d+[.)]\s*|\|)?"
    r"(?:given|when|then|and|but|scenario(?:\s+outline)?|feature|background|examples?|rule)\b",
    re.IGNORECASE,
)

# References to a specific external record. Jira-style keys, ticket numbers,
# and named documents — the things a Requirement Agent would cite.
_CITATIONS = (
    # A tracker key counts only when something *cites* it: unqualified, `[A-Z]+-\d+` is also
    # the shape of every standards identifier (ISO-27001, CVE-2021-44228, SHA-512), and cueing
    # those put "unverified source" footnotes on ordinary security requirements.
    re.compile(
        r"(?i)\b(?:ticket|issue|story|epic|card|task|bug|defect|jira|ref(?:erence)?|"
        r"per|see|from|documented\s+in|according\s+to|raised\s+in|tracked\s+in)\b"
        r"[^.\n]{0,30}?\b([A-Z][A-Z0-9]{1,9}-\d{1,6})\b"
    ),
    re.compile(
        r"(?i)\b(?:ticket|issue|story|epic|incident|change\s+request|cr)\s*#?\s*(\d{2,8})\b"
    ),
    re.compile(
        r"(?i)\b(?:per|see|from|according\s+to|as\s+documented\s+in|cited\s+in)\s+(?:the\s+)?([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,4}\s+(?:doc(?:ument)?|spec(?:ification)?|ADR|RFC|page|runbook|wiki))\b"
    ),
)
# Prefixes naming a published standard, for "per RFC-7519". An enumerated list fits: standards
# families are a bounded set while project keys are not, and one prefix covers every number.
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


# What the user reads. A statement about the record, not an accusation: the claim may well be
# true — the supervisor simply cannot vouch for it, and says so.
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

    Every access is type-checked: every value is worker-supplied and this runs *outside* the
    dispatch node's exception handling, so a stray `AttributeError` would escape `graph.stream()`
    as an unaudited 500 after the worker call was already paid for.
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
    parts: list[str] = []
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
        isinstance(s, dict) and str(s.get("origin", "")).lower() == "simulated" for s in sources
    )


def check(text, sources=None, raw=None) -> GroundingCheck:
    """Assess one worker reply. Cheap, deterministic, never raises.

    "Never raises" is a contract: the call site in `nodes.dispatch` sits after the worker's
    `except Exception` has closed and the graph declares no `RetryPolicy`, so anything escaping
    here becomes an unaudited 500. Every argument is worker-controlled and typed accordingly.
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
