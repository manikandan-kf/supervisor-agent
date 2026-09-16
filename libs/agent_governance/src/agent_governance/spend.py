"""Cost control — the turn's spend envelope, and the subject's rolling one.

Bounds how much a turn (`TurnSpend`) and a subject (`SubjectWindow`) may spend on the
supervisor's own governance calls; `SpendCaps` holds the ceilings and `SpendExhausted` is raised
before a call would breach one. Worker tokens are out of scope (§01). Model calls are counted
exactly; tokens are only estimated (`estimate_tokens`), never the sole basis of a refusal. A
retried verdict is one decision, counters are per replica, and every cap is off by default (`0`).
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

# Roughly what one governance system prompt costs (~4.5 KB screen, ~2.5 KB routing). A constant
# rather than measured: measuring costs a `get_prompt` round-trip to refine a non-enforcing guess.
PROMPT_OVERHEAD_CHARS = 4500

# The structured verdict a governance call generates. Generous rather than tight: an estimate
# that under-reports output is worse than one that over-reports it.
VERDICT_OUTPUT_TOKENS = 300


class SpendExhausted(Exception):
    """A spend ceiling was reached before this call could be made.

    Deliberately *not* a subclass of `deadline.BudgetExhausted`: "this turn ran long" and "this
    user has spent its allowance" are different findings with different operator actions, and
    collapsing them would leave the decision trail unable to say which happened.
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
        super().__init__(f"{scope} budget of {limit:g} {unit} exhausted{detail} (spent {spent:g})")


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
            subject_window_seconds=float(getattr(settings, "subject_spend_window_seconds", 0) or 0),
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

    `extra_chars` covers what is not in `messages`: the query passed separately, and the system
    prompt. Failure returns a character-based estimate rather than raising — a counter that can
    fail a turn would be a cost control that causes outages.
    """
    total = 0
    if messages:
        try:
            total += int(count_tokens_approximately(messages))
        except Exception:
            logger.debug("token estimation fell back to character count", exc_info=True)
            total += sum(len(str(getattr(m, "content", m) or "")) for m in messages) // 4
    if extra_chars > 0:
        total += extra_chars // 4
    return max(0, total)


class TurnSpend:
    """This turn's governance ledger: what it has spent, and what it may still.

    Mutable and rebuilt from graph state at each node, since spend accumulates across nodes.
    Carried in state rather than runtime context, unlike the deadline, and for the opposite
    reason: spend already made stays made across an approval interrupt. Resets at the RBAC gate.
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

        Called *before* an outbound call. The check is "would this call take the turn past its
        ceiling", not "is the ceiling already reached", so the ceiling is a ceiling rather than
        a threshold the last call always crosses.
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

        The stage has already made its calls, so refusing here would only lose the accounting
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


class SubjectWindow:
    """One subject's governance spend over a rolling window, in this process.

    Per replica — see the module docstring. `check` runs before any model call and `charge`
    after the turn's spend is known, so a subject who breaches the window is refused on their
    *next* turn: refusing halfway would spend the calls and deliver nothing.
    """

    def __init__(self, caps: Optional[SpendCaps] = None):
        self.caps = caps or SpendCaps()
        # subject -> deque of (monotonic timestamp, model_calls, tokens)
        self._entries: dict[str, deque] = {}
        # One lock over the whole structure: the window expires entries during a read, so
        # concurrent reads of one subject raced each other's `popleft`. That failure was
        # silent — `check` runs inside the RBAC gate's `try/except`, so a raised read fails open.
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
        """ "" when this subject may run a turn, else why they may not.

        A missing subject key is allowed through rather than pooled under one bucket, which
        would make every unattributable caller share an allowance. The surface that arrives
        without a `user_key` is the direct endpoint, guarded by RBAC and the entitlement HMAC.
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
            entries.append((time.monotonic(), max(0, int(model_calls)), max(0, int(tokens))))
            # Bound the memory one subject can occupy: entries expire on read,
            # but a subject who never returns leaves its deque behind until then.
            if len(entries) > 1024:
                entries.popleft()


def build_spend_window(settings) -> SubjectWindow:
    """The process-wide subject window, from settings."""
    return SubjectWindow(SpendCaps.from_settings(settings))
