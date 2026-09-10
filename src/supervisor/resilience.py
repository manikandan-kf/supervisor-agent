"""Bounded retries for transient failures at the model boundary.

One retry authority, and it lives *here*, at the individual call — not in the
graph. `graph.py` used to wrap the guardrail and route nodes in
`RetryPolicy(max_attempts=3)`, but a RetryPolicy only fires on a *raised*
exception and both nodes catch every exception from the model call and return a
controlled fail-closed Command. The policy never saw a failure, so a single
transient 429/5xx produced an immediate "checks unavailable" hold with zero
retries — safe, but not what the comments promised, and at fleet scale it turns
routine endpoint blips into user-visible errors and appeal load.

Retrying the single failed call is also the cheaper shape: a node-level retry
re-runs the guardrail fan-out across every candidate agent, re-spending one
model call per agent to recover one failure. Here only the verdict that failed
is retried.

The deadline is checked before each attempt *and* before each backoff sleep,
for the same reason `dispatch.ModelServingWorkerClient.invoke` does: a retry
that cannot finish inside the turn budget is not a retry, it is an overrun, and
sleeping through the remaining budget burns the headroom `respond` needs to
write the audit row.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Callable, Optional, TypeVar

from .deadline import BudgetExhausted

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _default_attempts() -> int:
    """`Settings.governance_llm_attempts`, read per call (env-per-call, like
    `prompt_provider`) so the engines built from governed config need no extra
    constructor plumbing to honour it."""
    try:
        return max(1, int(os.getenv("GOVERNANCE_LLM_ATTEMPTS", "3")))
    except ValueError:
        return 3


def jittered(delay: float) -> float:
    """`delay` with full jitter — a uniform draw from [delay/2, delay].

    Deterministic backoff synchronises retries. Every replica that saw the same
    endpoint blip waits exactly 0.5s, then exactly 1.0s, then exactly 2.0s, so
    the recovering endpoint is hit by the whole fleet at once and blips again —
    the thundering herd, produced by the mechanism meant to prevent it. Half-
    to-full jitter is the standard shape: it never *lengthens* a wait beyond
    what the caller budgeted (so the deadline arithmetic above stays a bound)
    and it spreads the retries.

    `random`, not `secrets`, on purpose: this is load spreading, not a security
    decision, and `secrets` would draw from the system entropy pool on every
    retry to no benefit.
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

    Non-transient failures and `BudgetExhausted` propagate immediately —
    a permission or contract error will not heal with a retry, and an exhausted
    budget must never be retried because the retries are what spent it. The
    caller's own `except Exception` fail-closed path still catches whatever
    survives the last attempt, so this changes how often the hold happens, never
    whether it happens.
    """
    attempts = max(1, attempts) if attempts is not None else _default_attempts()
    for attempt in range(attempts):
        # The *first* attempt is gated by the caller (both governance call
        # sites `ensure()` immediately before invoking this); the wrapper gates
        # only the retries it adds, so wrapping a call does not double-charge
        # the deadline's bookkeeping.
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
