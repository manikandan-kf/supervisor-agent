"""Regression evaluation for a governed guardrails document.

The guardrails document travels the least supervised path in the system: `publish_config.py
--apply` writes it and a running endpoint picks it up within the cache TTL, with no CI run in
between. This is that missing check: a labelled corpus run through the same deterministic
engines the endpoint uses (`deny_rules` for input, `OutputGuard` for replies). Deterministic on
purpose, so it can be a *gate*; model-graded evaluation belongs alongside, not inside. Roughly
half the corpus is ordinary SDLC work that must pass untouched — over-blocking is a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from .deny_rules import compile_rules, first_match
from .output_guard import OutputGuard

# Input-tier expectations.
ALLOW = "allow"
BLOCK = "block"
ESCALATE = "escalate"
# Output-tier expectations add masking.
MASK = "mask"

_INPUT_EXPECTATIONS = (ALLOW, BLOCK, ESCALATE)
_OUTPUT_EXPECTATIONS = (ALLOW, MASK, BLOCK, ESCALATE)


@dataclass(frozen=True)
class Case:
    """One labelled expectation about the policy.

    Exactly one of `text` (a user message, screened at the input tier) or
    `reply` (a worker response, screened at the output tier) is set.
    """

    id: str
    expect: str
    text: str = ""
    reply: str = ""
    why: str = ""
    # Output case: labels the screen must report. A subset check, since a policy that finds
    # *more* than the case names is not a regression.
    labels: tuple[str, ...] = ()

    @property
    def tier(self) -> str:
        return "output" if self.reply else "input"


@dataclass(frozen=True)
class Result:
    case: Case
    actual: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Report:
    results: tuple[Result, ...] = field(default_factory=tuple)

    @property
    def failures(self) -> tuple[Result, ...]:
        return tuple(r for r in self.results if not r.passed)

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        total = len(self.results)
        failed = len(self.failures)
        if not total:
            return "no cases evaluated"
        if not failed:
            return f"{total}/{total} policy cases passed"
        return f"{total - failed}/{total} policy cases passed, {failed} FAILED"

    def report_lines(self) -> list[str]:
        """One line per failure, naming what was expected and what happened."""
        lines = []
        for r in self.failures:
            subject = r.case.text or r.case.reply
            excerpt = subject if len(subject) <= 88 else subject[:85] + "..."
            lines.append(
                f"  {r.case.id} [{r.case.tier}] expected {r.case.expect}, got {r.actual}"
                + (f" — {r.detail}" if r.detail else "")
                + f"\n      {excerpt!r}"
                + (f"\n      why: {r.case.why}" if r.case.why else "")
            )
        return lines


def parse_cases(data) -> list[Case]:
    """Build cases from a parsed YAML/JSON document. Raises on a malformed one.

    Strict, because a case that silently fails to load is a check that silently
    stops running — the failure mode this whole module exists to remove.
    """
    raw = (data or {}).get("cases") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError("policy suite must carry a list under `cases:`")
    cases: list[Case] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"case {index} is not a mapping")
        case_id = str(entry.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"case {index} has no id")
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id!r}")
        seen.add(case_id)
        text = str(entry.get("text") or "")
        reply = str(entry.get("reply") or "")
        if bool(text) == bool(reply):
            raise ValueError(f"case {case_id} must set exactly one of `text` or `reply`")
        expect = str(entry.get("expect") or "").strip().lower()
        permitted = _OUTPUT_EXPECTATIONS if reply else _INPUT_EXPECTATIONS
        if expect not in permitted:
            raise ValueError(
                f"case {case_id}: expect must be one of {', '.join(permitted)}, got {expect!r}"
            )
        labels = entry.get("labels") or ()
        cases.append(
            Case(
                id=case_id,
                expect=expect,
                text=text,
                reply=reply,
                why=str(entry.get("why") or ""),
                labels=tuple(str(label) for label in labels),
            )
        )
    return cases


def evaluate(document: dict, cases: Iterable[Case], mask_pii: bool = True) -> Report:
    """Run every case against `document` and report what the policy did.

    `document` is the guardrails mapping as it would be published — so this can
    be run against a candidate before `publish_config.py` writes it.
    """
    rules = compile_rules((document or {}).get("global_deny_patterns", []))
    guard = OutputGuard.from_mapping(document or {}, mask_pii=mask_pii)

    results = []
    for case in cases:
        if case.tier == "input":
            rule = first_match(rules, case.text)
            actual = rule.action if rule else ALLOW
            passed = actual == case.expect
            detail = rule.reason if rule else ""
            results.append(Result(case, actual, passed, detail))
            continue

        screened = guard.screen(case.reply)
        actual = screened.action
        detail = screened.reason
        passed = actual == case.expect
        if passed and case.labels:
            missing = [label for label in case.labels if label not in screened.labels]
            if missing:
                passed = False
                detail = f"expected label(s) not found: {', '.join(missing)}"
        results.append(Result(case, actual, passed, detail))
    return Report(tuple(results))


def load_suite(path) -> list[Case]:
    """Read and parse a corpus file.

    The single definition of what a suite file is, shared by `publish_config.py --apply` and the
    CI test, so the two cannot drift into checking different things under one name.
    """
    return parse_cases(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
