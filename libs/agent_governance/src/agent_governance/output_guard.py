"""Output guardrail — the last check before generated text reaches a user.

Layer 7 of the guardrail stack ("nothing reaches the user unchecked"). Three
boundaries, one policy:

  * **the reply** — every worker response is screened in `nodes.dispatch`
    before the text enters state, so checkpoints, the approval gate and the
    user only ever see guarded text;
  * **the relay** — the conversation forwarded *to* a worker is screened the
    same way (`relay`), so a credential or identifier the user pasted never
    reaches the worker in the first place and cannot be echoed back;
  * **the stream** — tokens relayed live are masked with a hold-back window
    and stopped outright once a block-tier finding appears (`StreamGuard`),
    so the closing item is not the first place the screen applies.

Two structural properties carry most of the layer's weight, and both are
worth stating because a screen without them looks present and catches almost
nothing:

  1. **The catalogue has to be wide.** A screen that knows credential tokens
     and email addresses passes names, phones, national identifiers, payment
     cards, bank accounts, health identifiers, dates of birth, internal
     hostnames and server paths through verbatim. `sensitive.py` is the
     catalogue, with checksums and context gates where the vendors use them.
  2. **The decision must not be binary.** If masking is the only reachable
     action — `output_deny_patterns` ships empty and nothing classifies a
     finding — then "block" and "escalate" exist in the design and nowhere in
     the code. `OutputPolicy` maps every finding category to an action, the
     governed guardrails document can override each one, and a bulk
     disclosure (five or more high-tier values in one reply) escalates to a
     human instead of being quietly masked value by value.

Defaults follow the sensitivity tiers Google DLP, AWS Bedrock Guardrails and
Databricks `detect_sensitive_data` agree on: credentials, national
identifiers, payment and bank data and health identifiers are withheld;
names, contact details, dates of birth, health conditions and network detail
are masked. Every default is one line in `guardrails.yaml` to change, and a
publish that sets a high-tier category to `allow` is refused.

Two controls that are easy to leave out and expensive to add later:

  * **Canary and prompt-leak detection.** A per-process canary is planted in
    the simulated worker's prompt, operators can register the canaries they
    plant in real workers (`canary_tokens`), and distinctive lines of the
    supervisor's own governance prompts are protected text. Any of them
    surfacing in a reply is a critical finding: the response is withheld and
    the conversation escalated, whatever the request looked like.
  * **Governance text scrubbing.** The screen's `reason` and clarification
    questions are model-written and shown to the user word for word; they
    pass through `scrub` so a prompt-injected verdict cannot carry a secret
    or the prompt itself out through the refusal message.

What this module is still *not*: a toxicity classifier or an LLM moderation
pass. Running a second model over every reply doubles cost and latency to
judge workers that are themselves governed internal agents, and a model-based
check can be argued with — the same reasoning as `guardrails.py`'s tier 1.
Platform-level moderation (Databricks AI Gateway guardrails on the serving
endpoints) is the right home for classifier-based screening and composes with
this module rather than replacing it.
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

# The canary this process plants in prompts it controls. Random per process
# (AWS's guidance: unique, unlikely in legitimate output), matched after
# whitespace/case normalisation so letter-spacing does not evade it. A worker
# that reproduces it has reproduced its instructions.
PROCESS_CANARY = "CANARY-" + secrets.token_hex(6).upper()


def _normalise(text: str) -> str:
    """Collapse whitespace and case so a leak split across tokens still matches."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _condense(text: str) -> str:
    """Strip everything but letters and digits — the interspersion-proof form."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


#: The floor a canary must clear to be armed. Matching happens on the condensed
#: form, so a token made only of punctuation, or written in a non-Latin script,
#: condenses to nothing and would never fire. `config_store` refuses to publish
#: one — the validator and the detector have to agree about what counts, or a
#: publish succeeds and the honeytoken is silently inert.
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

    Built from the governed guardrails document's `output_policy` section, or
    from the defaults when it is absent. `mask_pii=False` (the
    `OUTPUT_PII_MASKING` switch) relaxes the PII tier — names, contact
    details, dates of birth, health conditions — to `allow`; it never touches
    the high tier, which the switch was never meant to govern.
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

    Built from the same governed guardrails document as the input engine, so a
    published pattern change reaches a running endpoint through the existing
    `Reloading` proxy — no redeploy, no second config plane.
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
            _normalise(line)
            for line in protected_texts
            if line and len(_normalise(line)) >= 40
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

        Canaries are matched on the condensed form (letters and digits only),
        so `C A N A R Y - 7F3A` and `canary_7f3a` both count. Protected prompt
        lines are matched on the whitespace-normalised form: a worker that
        reproduces forty-plus characters of a governance prompt verbatim has
        reproduced its instructions, whatever the request was.
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

        Policy rules run on the raw text so a rule can match what masking
        would rewrite; a withheld response is discarded by the caller, so
        masking it would be work done on text nobody will see.
        """
        raw = text or ""
        leak = self.leak_in(raw)
        if leak:
            return OutputScreenResult(text="", action=ESCALATE, reason=leak, leak=leak)

        for rule in self._rules:
            if rule.regex.search(raw):
                return OutputScreenResult(text="", action=rule.action, reason=rule.reason)

        # `redact` rather than `scan` + `mask`: it sweeps until nothing new
        # appears, because masking creates the word boundaries a strict shape
        # needs when a permissive neighbour has swallowed them. See its
        # docstring for the half-a-card-number case that motivates it.
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
            f
            for f in findings
            if set(f.categories or (f.category,)) & sensitive.BULK_CATEGORIES
        ]
        threshold = self._policy.escalate_at_findings
        if threshold and len(bulk) >= threshold and _SEVERITY[strongest] < _SEVERITY[ESCALATE]:
            strongest = ESCALATE
            strongest_label = f"{len(bulk)} high-tier values"

        categories = tuple(
            dict.fromkeys(c for f in findings for c in (f.categories or (f.category,)))
        )
        labels = tuple(
            dict.fromkeys(x for f in findings for x in (f.labels or (f.label,)))
        )
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

        The same categories the reply screen acts on, masked rather than
        blocked: a user who pasted a connection string still gets their
        script, minus the credential the worker never needed. What was masked
        is returned so the dispatch node can tell the user and the trail.
        """
        masked, findings = sensitive.redact(text or "", self._policy.active_categories())
        return masked, tuple(findings)

    # ── governance text shown to the user ────────────────────────────────

    def scrub(self, text: str, fallback: str = "") -> str:
        """Make model-written governance text safe to show.

        A screen verdict's `reason` and a clarifying question are generated
        under a prompt that untrusted content can try to steer. Secrets and
        identifiers in them are masked; if the text carries a canary or
        protected prompt line, `fallback` is shown instead.
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

    Tokens are not relayed the instant they arrive: they wait in `pending`
    until at least `hold` characters have accumulated *behind* them, and the
    part that has cleared the window is masked and emitted at a sentence or
    line boundary. A withhold-tier finding anywhere in what has streamed stops
    further emission entirely — the closing item then replaces the client's
    buffer with the guarded (or withheld) text, which it always did; what is
    new is that a block-tier value no longer sits on screen in the meantime.

    **Masking is done against a context window, not against the chunk.** This
    is the part that is easy to get wrong, and getting it wrong leaks exactly
    the values the catalogue exists to find:

      * Half the shapes are *context-gated* — a bare date is only a date of
        birth because "DOB" sits within thirty characters of it. The release
        point prefers a sentence boundary, which is precisely where that
        keyword sits, so masking the chunk alone left `1990-01-02` in the clear
        after `…the applicant's DOB. ` had already been emitted.
      * A multi-token value straddles the cut. `customer Sarah Kim` split
        across the boundary matched nothing in either half.

    So the guard keeps the last `hold` characters of *already emitted* text as
    context, scans `context + pending` as one buffer, and refuses to release
    into any span a finding occupies — the release point is clamped back to the
    start of the earliest finding that extends past it. Nothing is emitted
    until the guard has seen the whole of every value inside it.

    The scan window is bounded (context plus pending, both bounded by `hold`
    and its fallback multiple) so the per-token cost does not grow with the
    length of the answer.
    """

    #: The catalogue's longest contiguous shape is a private key block, which
    #: `stream_should_hold` catches on its `-----BEGIN` marker instead. Of the
    #: rest, `github_pat_`/`gh?_` tokens and JWTs are the longest at up to ~260
    #: characters, so a hold below that cannot see one whole. The clamping
    #: above is what makes a smaller window safe anyway: a value the guard can
    #: only partly see is a value it will not release.
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
            # A reply with no line, sentence or word boundary — a long base64
            # blob, minified output, CJK prose — must not grow the buffer
            # without limit: the guard rescans `context + pending` on every
            # token, so an unbounded buffer makes the whole stream quadratic
            # (measured at ~16 seconds of pure CPU for a 24000-character
            # single-line reply). Past this ceiling the release point is the
            # window itself. Cutting mid-token is safe rather than merely
            # tolerable: `_release` clamps out of any value it can only partly
            # see, and the closing item is what the client renders in the end.
            if surplus <= 4 * self._hold:
                return ""
            cut = surplus - 1
        return self._release(cut + 1)

    def flush(self) -> str:
        """Emit whatever remains once the stream has ended.

        No clamping here: the stream has ended, so a finding that extends past
        the release point cannot grow any further and masking it is correct.
        """
        if self.suppressed or not self._pending:
            return ""
        return self._release(len(self._pending), clamp=False)

    def _release(self, count: int, clamp: bool = True) -> str:
        """Emit the first `count` characters of `pending`, masked in context.

        `self._context` holds the **already-masked** tail of what went out, so
        two properties hold together and make the offset arithmetic sound:

          * a context word that gated a finding is still in the buffer, so the
            finding is still found;
          * no finding can *start* inside the context. Findings entirely
            within it were masked to placeholders when they were emitted, and
            no shape matches a placeholder; findings straddling the release
            point are clamped away below rather than half-emitted. By
            induction every finding starts at or after the context, so
            subtracting the context length gives a valid span in `pending`.
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
