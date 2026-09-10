"""Cost control — the turn's spend envelope, and the subject's rolling one.

`deadline.py` bounds how *long* a turn may take. Nothing bounded how *much* it
may spend, and the two are not the same control:

  * a turn can stay comfortably inside 120 seconds while making one governance
    model call per candidate agent, each carrying a 4000-token history window
    and an 8000-character query — so the time budget is satisfied by a turn
    that costs an order of magnitude more than a normal one;
  * a request-rate limit at the front door bounds *requests* per principal, not
    spend. 20 requests a minute of the expensive shape is a sustained,
    authenticated, entirely in-policy way to drive cost.

The quantity that matters is token consumption by the Supervisor Agent itself
per turn — routing, guardrails, clarification — tracked separately from each
worker agent's own consumption. Reporting that after the fact tells an operator
what a turn cost and stops nothing; this module is the enforcement half.

Scope: the supervisor's **own** governance calls
────────────────────────────────────────────────
The screen verdicts and the context resolution, and nothing else. §01 puts
"token and credit consumption" for worker agents out of scope — it is "handled
at each individual agent level" — so a dispatch is deliberately not charged
here: the tokens it spends are the worker's, on the worker's endpoint, under
the worker's own budget. (The dev-only `SimulatedWorkerClient` does spend the
supervisor's model; it is a mock behind `SUPERVISOR_MOCK_WORKERS` and is not
charged, so the ledger reports the same number in dev as the deployed path
would.)

What is enforced, and on what evidence
──────────────────────────────────────
Two units, deliberately, because they carry different weight:

  * **model calls — exact.** One governance verdict is one call, and
    `ScreenResult.considered` reports exactly how many were made. Since each
    call's input is already bounded (`INPUT_MAX_CHARS`, `history_max_tokens`),
    bounding calls bounds cost with no estimation anywhere in the enforcement
    path. This is the primitive a ceiling should rest on.
  * **tokens — estimated.** `count_tokens_approximately`, the same counter
    `nodes._window` already trims history with. It is an estimate and is
    recorded as one: the ledger's token figure is for attribution and for a
    secondary ceiling, never the sole basis of a refusal. The authority on what
    was actually billed stays the trace (MLflow) and, at platform level, Unity
    AI Gateway's token-level cost attribution.

Retries are not counted separately. A verdict retried through
`resilience.invoke_with_retries` is one governance decision, and
`GOVERNANCE_LLM_ATTEMPTS` already bounds how many attempts it gets — counting
them here would make the ceiling mean something different depending on how
flaky the endpoint was that minute.

Why the subject window is in the graph as well as at the gateway
────────────────────────────────────────────────────────────────
Same reasoning as `INPUT_MAX_CHARS`: the gateway's bound lives in a different
deployable and protects only callers who came through it. A direct endpoint
invocation bypasses it entirely. And the quantities differ — the gateway counts
requests, this counts governance calls and tokens.

`SubjectWindow` counters are **per replica**, stated rather than implied, for
exactly the trade `ratelimit.py` and `dispatch.CircuitBreaker` already document:
the effective ceiling multiplies by replica count. It is a backstop against
runaway spend by one subject, not a billing control — the billing control is
platform-level (Unity AI Gateway hard budget limits), and the durable per-turn
figures this module writes into the audit row are what a cross-replica report
aggregates.

Everything is off by default (`0` disables each cap), so an existing deployment
behaves exactly as it did until an operator sets a ceiling.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages.utils import count_tokens_approximately

logger = logging.getLogger(__name__)

# Roughly what one governance system prompt costs. The bundled
# `supervisor_domain_screen` template is ~4.5 KB and `supervisor_routing` ~2.5
# KB, and a registry-loaded version is the same order. Counted as a constant
# rather than measured per call because measuring means calling `get_prompt`
# a second time from the node — a registry round-trip to improve an estimate
# that is explicitly not the enforcement basis.
PROMPT_OVERHEAD_CHARS = 4500

# The structured verdict a governance call generates: a sentence of reasoning,
# a boolean, a float, sometimes a question. Generous rather than tight — an
# estimate that under-reports output is worse than one that over-reports it.
VERDICT_OUTPUT_TOKENS = 300


class SpendExhausted(Exception):
    """A spend ceiling was reached before this call could be made.

    Deliberately **not** a subclass of `deadline.BudgetExhausted`. The two are
    caught in the same places and turned into the same shape of governed
    outcome, but they are different findings with different operator actions —
    "this turn ran long" versus "this turn, or this user, has spent its
    allowance" — and collapsing them would make the decision trail unable to
    say which happened.
    """

    def __init__(
        self,
        what: str = "",
        *,
        scope: str = "turn",
        unit: str = "model calls",
        spent: float = 0.0,
        limit: float = 0.0,
    ):
        self.what = what
        self.scope = scope
        self.unit = unit
        self.spent = spent
        self.limit = limit
        detail = f" before {what}" if what else ""
        super().__init__(
            f"{scope} budget of {limit:g} {unit} exhausted{detail} "
            f"(spent {spent:g})"
        )


@dataclass(frozen=True)
class SpendCaps:
    """The ceilings, all optional. `0` disables one without disabling the rest."""

    turn_model_calls: int = 0
    turn_tokens: int = 0
    subject_model_calls: int = 0
    subject_tokens: int = 0
    subject_window_seconds: float = 0.0

    @classmethod
    def from_settings(cls, settings) -> "SpendCaps":
        return cls(
            turn_model_calls=int(getattr(settings, "turn_max_model_calls", 0) or 0),
            turn_tokens=int(getattr(settings, "turn_max_tokens", 0) or 0),
            subject_model_calls=int(getattr(settings, "subject_max_model_calls", 0) or 0),
            subject_tokens=int(getattr(settings, "subject_max_tokens", 0) or 0),
            subject_window_seconds=float(
                getattr(settings, "subject_spend_window_seconds", 0) or 0
            ),
        )

    @property
    def turn_enabled(self) -> bool:
        return self.turn_model_calls > 0 or self.turn_tokens > 0

    @property
    def subject_enabled(self) -> bool:
        return self.subject_window_seconds > 0 and (
            self.subject_model_calls > 0 or self.subject_tokens > 0
        )


def estimate_tokens(messages=None, *, extra_chars: int = 0) -> int:
    """An approximate token count for one governance call's input.

    `messages` is the trimmed history window the node is about to send —
    already `BaseMessage` objects, so the same counter that trimmed it counts
    it. `extra_chars` covers what is not in that list: the query passed
    separately, and the system prompt.

    Failure returns a character-based estimate rather than raising. This value
    feeds attribution and a secondary ceiling; a counter that can fail a turn
    would be a cost control that causes outages.
    """
    total = 0
    if messages:
        try:
            total += int(count_tokens_approximately(messages))
        except Exception:
            logger.debug("token estimation fell back to character count", exc_info=True)
            total += sum(
                len(str(getattr(m, "content", m) or "")) for m in messages
            ) // 4
    if extra_chars > 0:
        total += extra_chars // 4
    return max(0, total)


class TurnSpend:
    """This turn's governance ledger: what it has spent, and what it may still.

    Mutable and per-turn, which is why it is not a frozen dataclass like
    `Deadline`: a deadline is derived from a fixed start time, whereas spend
    accumulates across nodes within one turn. It is rebuilt from graph state at
    each node and written back, so the running total survives the hop between
    nodes without a second place to store it.

    Carried in **state** rather than runtime context, unlike the deadline, and
    for the opposite reason: the deadline must reset when a human answers an
    approval interrupt hours later, while spend already made in this
    conversation's turn is spend already made. It resets per turn at the RBAC
    gate along with the other per-turn stage results.
    """

    STATE_KEY = "spend"

    def __init__(
        self,
        caps: Optional[SpendCaps] = None,
        *,
        model_calls: int = 0,
        tokens: int = 0,
        by_stage: Optional[dict] = None,
    ):
        self.caps = caps or SpendCaps()
        self.model_calls = max(0, int(model_calls))
        self.tokens = max(0, int(tokens))
        self.by_stage: dict[str, dict[str, int]] = {
            str(k): {"model_calls": int(v.get("model_calls", 0)), "tokens": int(v.get("tokens", 0))}
            for k, v in (by_stage or {}).items()
            if isinstance(v, dict)
        }

    @classmethod
    def from_state(cls, state: dict, caps: Optional[SpendCaps] = None) -> "TurnSpend":
        record = (state or {}).get(cls.STATE_KEY) or {}
        return cls(
            caps,
            model_calls=record.get("model_calls", 0) or 0,
            tokens=record.get("tokens", 0) or 0,
            by_stage=record.get("by_stage") or {},
        )

    @property
    def enabled(self) -> bool:
        return self.caps.turn_enabled

    def ensure(self, what: str = "", *, model_calls: int = 1, tokens: int = 0) -> None:
        """Raise `SpendExhausted` if this turn cannot afford `what`.

        Called *before* an outbound call, for the same reason `Deadline.ensure`
        is: called after, it only reports an overrun that already happened.
        The check is "would this call take the turn past its ceiling", not
        "is the ceiling already reached" — a turn whose next call is the one
        that would breach it is stopped before making it, so the ceiling is a
        ceiling rather than a threshold the last call always crosses.
        """
        caps = self.caps
        if caps.turn_model_calls > 0 and self.model_calls + model_calls > caps.turn_model_calls:
            raise SpendExhausted(
                what,
                scope="turn",
                unit="model calls",
                spent=self.model_calls,
                limit=caps.turn_model_calls,
            )
        if caps.turn_tokens > 0 and self.tokens + tokens > caps.turn_tokens:
            raise SpendExhausted(
                what,
                scope="turn",
                unit="tokens",
                spent=self.tokens,
                limit=caps.turn_tokens,
            )

    def charge(self, stage: str, *, model_calls: int = 1, tokens: int = 0) -> None:
        """Record what a stage actually spent. Never raises.

        Charging cannot fail the turn it is recording: the stage has already
        made its calls, and refusing at this point would lose the accounting
        for spend that happened. `ensure` is where a refusal belongs.
        """
        model_calls = max(0, int(model_calls))
        tokens = max(0, int(tokens))
        self.model_calls += model_calls
        self.tokens += tokens
        bucket = self.by_stage.setdefault(stage, {"model_calls": 0, "tokens": 0})
        bucket["model_calls"] += model_calls
        bucket["tokens"] += tokens

    def record(self) -> dict:
        """The state / audit shape. Stable, because a KPI reads it."""
        return {
            "model_calls": self.model_calls,
            "tokens": self.tokens,
            "by_stage": {k: dict(v) for k, v in self.by_stage.items()},
        }

    def audit_detail(self) -> str:
        """One line for a decision-trail entry, mirroring `CleanOutput.audit_detail`."""
        stages = ", ".join(
            f"{stage} {spent['model_calls']}call/{spent['tokens']}tok"
            for stage, spent in sorted(self.by_stage.items())
        )
        head = (
            f"{self.model_calls} governance model call(s), ~{self.tokens} token(s) estimated"
        )
        return f"{head} ({stages})" if stages else head


class SubjectWindow:
    """One subject's governance spend over a rolling window, in this process.

    Same sliding-window shape as the gateway's rate limiter, counting a
    different quantity: governance model calls and estimated tokens rather than
    requests. Entries older than the window are discarded on every read, so
    there is nothing to sweep.

    Per replica — see the module docstring. `check` is called before any model
    call in a turn and `charge` after the turn's spend is known, so a subject
    who breaches the window is refused on their *next* turn rather than
    mid-turn: refusing halfway through would spend the calls and deliver
    nothing, which is the worst of both.
    """

    def __init__(self, caps: Optional[SpendCaps] = None):
        self.caps = caps or SpendCaps()
        # subject -> deque of (monotonic timestamp, model_calls, tokens)
        self._entries: dict[str, deque] = {}
        # One lock over the whole structure. The window is read on every turn
        # and expires entries *during* that read, so two threads reading one
        # subject raced each other's `popleft`: one saw
        # `RuntimeError: deque mutated during iteration` from the summing pass,
        # the other `IndexError: pop from an empty deque` from the expiry loop.
        #
        # That is a cost control that stops applying under exactly the
        # concurrency it exists to bound, and it stops applying *silently*:
        # `check` runs inside the RBAC gate's `try/except`, so a raised read
        # fails **open** and the subject allowance is simply not enforced for
        # that turn, while a raised `charge` in `respond` loses the turn's
        # spend altogether.
        #
        # Contention is not a consideration here. The critical section is a
        # deque walk over at most 1024 entries, and `locking.py` already
        # serialises execution per conversation — this lock only ever contends
        # between different conversations belonging to the same subject.
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.caps.subject_enabled

    def _totals(self, subject: str) -> tuple[int, int]:
        with self._lock:
            return self._totals_locked(subject)

    def _totals_locked(self, subject: str) -> tuple[int, int]:
        entries = self._entries.get(subject)
        if not entries:
            return 0, 0
        cutoff = time.monotonic() - self.caps.subject_window_seconds
        while entries and entries[0][0] <= cutoff:
            entries.popleft()
        if not entries:
            self._entries.pop(subject, None)
            return 0, 0
        return sum(e[1] for e in entries), sum(e[2] for e in entries)

    def check(self, subject: str) -> str:
        """"" when this subject may run a turn, else why they may not.

        A missing subject key is allowed through rather than pooled under one
        bucket. Pooling would make every unattributable caller share one
        allowance and starve each other; the surface that reaches the graph
        without a `user_key` is the direct-endpoint one, which the RBAC gate
        and the entitlement HMAC are the controls for.
        """
        if not self.enabled or not subject:
            return ""
        calls, tokens = self._totals(subject)
        caps = self.caps
        window = int(caps.subject_window_seconds)
        if caps.subject_model_calls > 0 and calls >= caps.subject_model_calls:
            return (
                f"{calls} governance model calls in the last {window}s reached the "
                f"{caps.subject_model_calls}-call subject allowance"
            )
        if caps.subject_tokens > 0 and tokens >= caps.subject_tokens:
            return (
                f"~{tokens} governance tokens in the last {window}s reached the "
                f"{caps.subject_tokens}-token subject allowance"
            )
        return ""

    def charge(self, subject: str, *, model_calls: int = 0, tokens: int = 0) -> None:
        if not self.enabled or not subject or (model_calls <= 0 and tokens <= 0):
            return
        with self._lock:
            entries = self._entries.setdefault(subject, deque())
            entries.append(
                (time.monotonic(), max(0, int(model_calls)), max(0, int(tokens)))
            )
            # Bound the memory a single subject can occupy. The window already
            # discards old entries on read, but a subject who never returns
            # leaves its deque behind until then, and one entry per turn is
            # enough that a sustained caller should not be able to grow it
            # without limit.
            if len(entries) > 1024:
                entries.popleft()


def build_spend_window(settings) -> SubjectWindow:
    """The process-wide subject window, from settings."""
    return SubjectWindow(SpendCaps.from_settings(settings))
