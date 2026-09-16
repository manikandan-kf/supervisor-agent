"""Output guardrail (layer 7) — the last check before generated text reaches a user.

Entry points: `OutputGuard` (`screen` a reply, `relay` text to a worker, `scrub` model-written
governance text, `stream_should_hold`), `StreamGuard` (hold-back window over live tokens),
`OutputPolicy` (category → allow/mask/block/escalate, overridable per governed document) and
`PROCESS_CANARY` / `usable_canary` for prompt-leak detection. Tiers follow Google DLP, Bedrock
Guardrails and Databricks `detect_sensitive_data`; not a toxicity/LLM pass (AI Gateway's job).
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

from . import sensitive
from .deny_rules import compile_rules
from .sensitive import Finding

# ── actions ─────────────────────────────────────────────────────────────────
ALLOW = "allow"
MASK = "mask"
BLOCK = "block"
ESCALATE = "escalate"
ACTIONS: tuple[str, ...] = (ALLOW, MASK, BLOCK, ESCALATE)
_SEVERITY = {ALLOW: 0, MASK: 1, BLOCK: 2, ESCALATE: 3}

# The shipped defaults. High tier withheld, moderate tier masked — see the
# module docstring for the vendor tiers these mirror.
DEFAULT_CATEGORY_ACTIONS: dict[str, str] = {
    sensitive.CREDENTIAL: BLOCK,
    sensitive.SECRET_ASSIGNMENT: MASK,
    sensitive.GOVERNMENT_ID: BLOCK,
    sensitive.PAYMENT: BLOCK,
    sensitive.BANK: BLOCK,
    sensitive.HEALTH_ID: BLOCK,
    sensitive.HEALTH_CONDITION: MASK,
    sensitive.PERSON: MASK,
    sensitive.CONTACT: MASK,
    sensitive.DOB: MASK,
    sensitive.NETWORK: MASK,
}
# Five or more high-tier values in one reply is a data dump, not an echo.
DEFAULT_ESCALATE_AT_FINDINGS = 5

# The canary this process plants in prompts it controls. Random per process (AWS guidance),
# matched after whitespace/case normalisation so letter-spacing does not evade it.
PROCESS_CANARY = "CANARY-" + secrets.token_hex(6).upper()


def _normalise(text: str) -> str:
    """Collapse whitespace and case so a leak split across tokens still matches."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _condense(text: str) -> str:
    """Strip everything but letters and digits — the interspersion-proof form."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


#: The floor a canary must clear to be armed: matched on the condensed form, a punctuation-only
#: token never fires. `config_store` refuses to publish one — validator and detector must agree.
CANARY_MIN_CONDENSED = 6


def usable_canary(token: str) -> bool:
    """Whether a canary string can actually be detected if it surfaces."""
    return len(_condense(token)) >= CANARY_MIN_CONDENSED


@dataclass(frozen=True)
class OutputScreenResult:
    """The guarded text, plus what was done to it — for the decision trail."""

    text: str
    # allow | mask | block | escalate — the strongest action any finding earned.
    action: str = ALLOW
    # Why the response was refused — audit-facing, never shown to the user.
    reason: str = ""
    secrets_masked: int = 0
    pii_masked: int = 0
    # Which categories were found, whatever was done about them.
    categories: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    # Set when a canary or protected prompt text surfaced — the critical case.
    leak: str = ""
    findings: tuple[Finding, ...] = field(default_factory=tuple, repr=False)

    @property
    def blocked(self) -> bool:
        """Withheld from the user, whether by block or by escalation."""
        return self.action in (BLOCK, ESCALATE)

    @property
    def escalate(self) -> bool:
        return self.action == ESCALATE

    @property
    def modified(self) -> bool:
        return bool(self.secrets_masked or self.pii_masked)

    def audit_detail(self) -> str:
        """One line for `_entry(...)`, mirroring `CleanOutput.audit_detail`."""
        parts = []
        if self.blocked:
            parts.append(f"response withheld: {self.reason}")
        if self.secrets_masked:
            parts.append(f"{self.secrets_masked} secret value(s) masked")
        if self.pii_masked:
            parts.append(f"{self.pii_masked} PII value(s) masked")
        if self.labels:
            parts.append("found: " + ", ".join(self.labels))
        return "; ".join(parts) or "clean"


@dataclass(frozen=True)
class OutputPolicy:
    """What to do about each category of finding.

    Built from the governed document's `output_policy` section, else the defaults.
    `mask_pii=False` relaxes the PII tier to `allow`; it never touches the high tier.
    """

    category_actions: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CATEGORY_ACTIONS))
    escalate_at_findings: int = DEFAULT_ESCALATE_AT_FINDINGS

    @classmethod
    def from_mapping(cls, data: Optional[dict], mask_pii: bool = True) -> "OutputPolicy":
        section = (data or {}).get("output_policy") or {}
        actions = dict(DEFAULT_CATEGORY_ACTIONS)
        for category, action in (section.get("categories") or {}).items():
            if category in actions and action in ACTIONS:
                actions[category] = action
        if not mask_pii:
            for category in sensitive.PII_TIER:
                if actions.get(category) == MASK:
                    actions[category] = ALLOW
        threshold = section.get("escalate_at_findings", DEFAULT_ESCALATE_AT_FINDINGS)
        try:
            threshold = int(threshold)
        except (TypeError, ValueError):
            threshold = DEFAULT_ESCALATE_AT_FINDINGS
        return cls(actions, max(0, threshold))

    def action_for(self, category: str) -> str:
        return self.category_actions.get(category, MASK)

    def active_categories(self) -> frozenset[str]:
        """Categories whose findings do anything at all."""
        return frozenset(c for c, a in self.category_actions.items() if a != ALLOW)


class OutputGuard:
    """Deterministic screen over one outbound response.

    Built from the same governed guardrails document as the input engine, so a published
    pattern change reaches a running endpoint through the `Reloading` proxy — no redeploy.
    """

    def __init__(
        self,
        deny_rules: Optional[list[dict]] = None,
        mask_pii: bool = True,
        policy: Optional[OutputPolicy] = None,
        canaries: Iterable[str] = (),
        protected_texts: Iterable[str] = (),
    ):
        self._rules = compile_rules(deny_rules, default_reason="matched an output policy rule")
        self._policy = policy or OutputPolicy.from_mapping(None, mask_pii)
        self._canaries = tuple(
            _condense(c) for c in (*canaries, PROCESS_CANARY) if c and usable_canary(c)
        )
        self._protected = tuple(
            _normalise(line) for line in protected_texts if line and len(_normalise(line)) >= 40
        )

    @classmethod
    def from_mapping(
        cls,
        data: dict,
        mask_pii: bool = True,
        protected_texts: Iterable[str] = (),
    ) -> "OutputGuard":
        """Build from the parsed guardrails document (file or governed table)."""
        data = data or {}
        return cls(
            data.get("output_deny_patterns", []),
            mask_pii,
            policy=OutputPolicy.from_mapping(data, mask_pii),
            canaries=[c for c in (data.get("canary_tokens") or []) if isinstance(c, str)],
            protected_texts=protected_texts,
        )

    @property
    def policy(self) -> OutputPolicy:
        return self._policy

    # ── leak detection ───────────────────────────────────────────────────

    def leak_in(self, text: str) -> str:
        """Why `text` is a prompt leak, or "" if it is not.

        Canaries match on the condensed form (`C A N A R Y - 7F3A` counts); protected prompt
        lines match whitespace-normalised — forty-plus characters reproduced is a leak.
        """
        if not text:
            return ""
        condensed = _condense(text)
        for canary in self._canaries:
            if canary in condensed:
                return "a planted canary token surfaced in the response"
        normalised = _normalise(text)
        for line in self._protected:
            if line in normalised:
                return "the response reproduces protected system-prompt text"
        return ""

    # ── the reply ────────────────────────────────────────────────────────

    def screen(self, text: str) -> OutputScreenResult:
        """Screen one response. Leak check, policy rules, then the catalogue.

        Policy rules run on the raw text so a rule can match what masking would rewrite.
        """
        raw = text or ""
        leak = self.leak_in(raw)
        if leak:
            return OutputScreenResult(text="", action=ESCALATE, reason=leak, leak=leak)

        for rule in self._rules:
            if rule.regex.search(raw):
                return OutputScreenResult(text="", action=rule.action, reason=rule.reason)

        # `redact` rather than `scan` + `mask`: it sweeps until nothing new appears, because
        # masking creates the word boundaries a strict shape needs.
        masked, findings = sensitive.redact(raw, self._policy.active_categories())
        if not findings:
            return OutputScreenResult(text=raw)

        strongest = ALLOW
        strongest_label = ""
        for finding in findings:
            # Every category in the span, not just the primary: a merged span
            # is acted on at its strongest tier (see `sensitive._merge`).
            for category in finding.categories or (finding.category,):
                action = self._policy.action_for(category)
                if _SEVERITY[action] > _SEVERITY[strongest]:
                    strongest, strongest_label = action, finding.label

        bulk = [
            f for f in findings if set(f.categories or (f.category,)) & sensitive.BULK_CATEGORIES
        ]
        threshold = self._policy.escalate_at_findings
        if threshold and len(bulk) >= threshold and _SEVERITY[strongest] < _SEVERITY[ESCALATE]:
            strongest = ESCALATE
            strongest_label = f"{len(bulk)} high-tier values"

        categories = tuple(
            dict.fromkeys(c for f in findings for c in (f.categories or (f.category,)))
        )
        labels = tuple(dict.fromkeys(x for f in findings for x in (f.labels or (f.label,))))
        if strongest in (BLOCK, ESCALATE):
            return OutputScreenResult(
                text="",
                action=strongest,
                reason=f"{strongest_label} in the response ({', '.join(labels)})",
                categories=categories,
                labels=labels,
                findings=tuple(findings),
            )

        secret_count = sum(
            1
            for f in findings
            if set(f.categories or (f.category,))
            & {sensitive.CREDENTIAL, sensitive.SECRET_ASSIGNMENT}
        )
        return OutputScreenResult(
            text=masked,
            action=MASK,
            secrets_masked=secret_count,
            pii_masked=len(findings) - secret_count,
            categories=categories,
            labels=labels,
            findings=tuple(findings),
        )

    # ── the relay to a worker ────────────────────────────────────────────

    def relay(self, text: str) -> tuple[str, tuple[Finding, ...]]:
        """Mask what must not reach a worker. Never withholds.

        Masked rather than blocked: a user who pasted a connection string still gets their
        script, minus the credential. What was masked is returned for the trail.
        """
        masked, findings = sensitive.redact(text or "", self._policy.active_categories())
        return masked, tuple(findings)

    # ── governance text shown to the user ────────────────────────────────

    def scrub(self, text: str, fallback: str = "") -> str:
        """Make model-written governance text safe to show.

        A verdict `reason` or clarifying question is generated under a prompt untrusted content
        can steer: secrets are masked, and a canary or protected line yields `fallback`.
        """
        if not text:
            return text or ""
        if self.leak_in(text):
            return fallback
        masked, _ = self.relay(text)
        return masked

    # ── streaming ────────────────────────────────────────────────────────

    def stream_should_hold(self, buffer: str) -> bool:
        """Whether streaming must stop: a withhold-tier finding or a leak."""
        if self.leak_in(buffer):
            return True
        if "-----BEGIN" in buffer:
            return True
        for rule in self._rules:
            if rule.regex.search(buffer):
                return True
        for finding in sensitive.scan(buffer, self._policy.active_categories()):
            for category in finding.categories or (finding.category,):
                if self._policy.action_for(category) in (BLOCK, ESCALATE):
                    return True
        return False

    def stream_findings(self, buffer: str) -> tuple[Finding, ...]:
        """Every finding in a stream buffer, with spans into that buffer."""
        return tuple(sensitive.scan(buffer, self._policy.active_categories()))


class StreamGuard:
    """Hold-back window over live worker tokens.

    Masks against `context + pending`, never the chunk alone: context-gated shapes and values
    straddling the cut would otherwise leak. The release point is clamped back to the start of
    any finding extending past it, and a withhold-tier finding anywhere stops emission.
    """

    #: `github_pat_` tokens and JWTs reach ~260 chars (private-key blocks are caught on
    #: `-----BEGIN` instead). A smaller hold is still safe: partly-seen values are never released.
    LONGEST_SHAPE_HINT = 260

    def __init__(self, guard: OutputGuard, hold: int = 160):
        self._guard = guard
        self._hold = max(0, hold)
        self._pending = ""
        # The tail of what has already gone out, kept only as scanning context.
        self._context = ""
        self.suppressed = False

    def feed(self, text: str) -> str:
        """Accept one token; return the (masked) text safe to emit now."""
        if not text or self.suppressed:
            return ""
        self._pending += text
        if self._guard.stream_should_hold(self._context + self._pending):
            self.suppressed = True
            return ""
        if self._hold == 0:
            return self._release(len(self._pending))
        surplus = len(self._pending) - self._hold
        if surplus <= 0:
            return ""
        window = self._pending[:surplus]
        cut = max(window.rfind("\n"), window.rfind(". "))
        if cut < 0 and surplus > 3 * self._hold:
            cut = window.rfind(" ")
        if cut < 0:
            # No line, sentence or word boundary (base64 blob, minified output, CJK prose):
            # rescanning an unbounded buffer per token goes quadratic, so past this ceiling
            # cut mid-token — safe, because `_release` clamps out any partly-seen value.
            if surplus <= 4 * self._hold:
                return ""
            cut = surplus - 1
        return self._release(cut + 1)

    def flush(self) -> str:
        """Emit whatever remains once the stream has ended.

        No clamping: nothing can grow further once the stream ends, so masking it all is correct.
        """
        if self.suppressed or not self._pending:
            return ""
        return self._release(len(self._pending), clamp=False)

    def _release(self, count: int, clamp: bool = True) -> str:
        """Emit the first `count` characters of `pending`, masked in context.

        `_context` is the already-masked tail of what went out, so a gating word is still present
        and no finding can *start* inside it, so `start - offset` is a valid span in `pending`.
        """
        buffer = self._context + self._pending
        offset = len(self._context)
        findings = self._guard.stream_findings(buffer)

        if clamp:
            limit = offset + count
            for finding in findings:
                if finding.start < limit < finding.end:
                    limit = min(limit, finding.start)
            count = max(0, limit - offset)
            if count <= 0:
                return ""

        raw = self._pending[:count]
        inside = [
            replace(finding, start=finding.start - offset, end=finding.end - offset)
            for finding in findings
            if finding.start >= offset and finding.end <= offset + count
        ]
        emitted = sensitive.mask(raw, inside) if inside else raw
        self._pending = self._pending[count:]
        self._context = (self._context + emitted)[-max(self._hold, 40) :]
        return emitted
