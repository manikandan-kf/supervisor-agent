"""The turn's time budget, and the bounded retries that must fit inside it.

Governance Blueprint Stage 05 (bound the blocking call): `Deadline` / `deadline_for` hold the
turn's budget, `invoke_with_retries` is the per-call retry that must fit inside it, re-checked
before every attempt and backoff sleep. Per-call bounds multiply - 4 agents x 30s x 3 attempts =
361s against a 180s gateway - and an overrun loses the audit row. Retries live here because a
LangGraph `RetryPolicy` fires only on a raised exception, and both model-backed nodes fail closed.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar


class BudgetExhausted(Exception):
    """The turn's total time budget ran out before this call could be made."""

    def __init__(self, what: str = "", spent: float = 0.0, budget: float = 0.0):
        self.what = what
        self.spent = spent
        self.budget = budget
        detail = f" before {what}" if what else ""
        super().__init__(f"turn budget of {budget:.0f}s exhausted{detail} (spent {spent:.1f}s)")


@dataclass(frozen=True)
class Deadline:
    """How much of this turn's budget is left.

    `budget_seconds <= 0` disables the deadline entirely: `remaining()` reports infinity and
    `ensure()` never raises. That is the offline-test and direct-invocation path, and it keeps
    this module from becoming something tests must know about in order to call a node.
    """

    started_at: float = 0.0
    budget_seconds: float = 0.0

    @classmethod
    def from_context(cls, context, budget_seconds: float) -> "Deadline":
        """The deadline for the turn `context` describes.

        No `turn_started_at` yields a disabled deadline, not an expired one: failing closed on
        a missing *clock* would break every offline caller without protecting anything. The
        budget is an availability bound; the controls that must fail closed are elsewhere.
        """
        started = float(getattr(context, "turn_started_at", 0.0) or 0.0)
        if started <= 0:
            return cls(0.0, 0.0)
        return cls(started, max(0.0, float(budget_seconds)))

    @property
    def enabled(self) -> bool:
        return self.budget_seconds > 0 and self.started_at > 0

    def spent(self) -> float:
        if not self.enabled:
            return 0.0
        return max(0.0, time.monotonic() - self.started_at)

    def remaining(self) -> float:
        if not self.enabled:
            return math.inf
        return self.budget_seconds - self.spent()

    @property
    def exhausted(self) -> bool:
        return self.remaining() <= 0

    def ensure(self, what: str = "") -> None:
        """Raise `BudgetExhausted` if there is no time left to do `what`.

        Call this *before* an outbound call: called before it stops a fan-out or retry loop
        from starting work it cannot finish; called after, it only reports the overrun.
        """
        if self.enabled and self.exhausted:
            raise BudgetExhausted(what, self.spent(), self.budget_seconds)


def deadline_for(context, settings) -> Deadline:
    """The turn deadline, from runtime context and settings.

    A context with no clock means a caller bypassed `from_custom_inputs` — the direct-call
    surface — so off the local path the budget falls back to "from now" rather than running
    unbounded. Each node calls this separately: the bound degrades to per-node, not per-turn.
    """
    budget = getattr(settings, "turn_budget_seconds", 0.0)
    deadline = Deadline.from_context(context, budget)
    if not deadline.enabled and budget > 0:
        from .environment import is_local_environment

        if not is_local_environment(getattr(settings, "environment", "local")):
            return Deadline(time.monotonic(), max(0.0, float(budget)))
    return deadline


# ---------------------------------------------------------------------------
# Bounded retries at the model boundary
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

T = TypeVar("T")


def _default_attempts() -> int:
    """`Settings.governance_llm_attempts`, read per call (env-per-call, like `prompt_provider`)
    so engines built from governed config need no extra constructor plumbing to honour it."""
    try:
        return max(1, int(os.getenv("GOVERNANCE_LLM_ATTEMPTS", "3")))
    except ValueError:
        return 3


def jittered(delay: float) -> float:
    """`delay` with full jitter — a uniform draw from [delay/2, delay].

    Deterministic backoff synchronises retries into a thundering herd on the recovering
    endpoint. Half-to-full jitter never *lengthens* a wait beyond what the caller budgeted, so
    the deadline arithmetic stays a bound. `random`, not `secrets`: load spreading, not crypto.
    """
    return random.uniform(delay / 2.0, delay)  # noqa: S311 - load spreading, not crypto


def is_transient(exc: Exception) -> bool:
    """Whether a failure is worth retrying: a temporary 429/5xx or a timeout."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or 500 <= status_code < 600
    text = str(exc).lower()
    return "timeout" in text or "timed out" in text or "connection" in text


def invoke_with_retries(
    call: Callable[[], T],
    *,
    what: str,
    deadline=None,
    attempts: Optional[int] = None,
    initial_backoff: float = 0.5,
    backoff_factor: float = 2.0,
) -> T:
    """Run `call`, retrying transient failures with backoff inside the budget.

    Non-transient failures and `BudgetExhausted` propagate immediately: a permission or
    contract error will not heal, and an exhausted budget must never be retried because the
    retries are what spent it. The caller's fail-closed path still catches the last failure.
    """
    attempts = max(1, attempts) if attempts is not None else _default_attempts()
    for attempt in range(attempts):
        # The caller gates the *first* attempt (both governance call sites `ensure()`
        # immediately before this), so the wrapper gates only the retries it adds.
        if attempt > 0 and deadline is not None:
            deadline.ensure(what)
        try:
            return call()
        except BudgetExhausted:
            raise
        except Exception as exc:
            if not is_transient(exc) or attempt + 1 >= attempts:
                raise
            delay = jittered(initial_backoff * (backoff_factor**attempt))
            if deadline is not None and deadline.remaining() <= delay:
                logger.warning(
                    "%s failed (%s) with %.1fs of budget left — not retrying",
                    what,
                    type(exc).__name__,
                    max(0.0, deadline.remaining()),
                )
                raise
            logger.warning(
                "%s failed (%s), retrying in %.1fs (attempt %d/%d)",
                what,
                type(exc).__name__,
                delay,
                attempt + 1,
                attempts,
            )
            time.sleep(delay)
    raise RuntimeError(f"unreachable: {what} exited its retry loop")  # pragma: no cover
