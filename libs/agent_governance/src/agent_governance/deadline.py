"""The turn's total time budget — Governance Blueprint §05 Stage 05.

The blueprint's "bound the blocking call" control names four things: a
per-attempt timeout, **a total time budget**, bounded retries with backoff, and
a circuit breaker. Three of those existed before this module. This is the
fourth, and without it the other three do not compose.

Why a per-call timeout is not enough
────────────────────────────────────
Per-call bounds multiply. Two multipliers apply here, and neither is visible
from the call site:

  * **fan-out** — `guardrails.screen` calls the semantic tier once per candidate
    agent (`guardrails.py`, the loop over `candidates`). Four agents in
    `agents.yaml` means up to four 30-second calls in one node;
  * **retries** — `resilience.invoke_with_retries` gives each governance model
    call up to three attempts with backoff.

4 x 30s x 3 = 361s for the guardrails node alone, against a gateway that gives
up at `INVOCATION_TIMEOUT_SECONDS` (180s). Even with zero retries, four slow
calls plus routing plus dispatch is 225s. The arithmetic was already over
budget with every individual bound set correctly.

What overrunning actually costs
───────────────────────────────
Nothing propagates the gateway's cancellation into the graph, so when the
gateway cuts the connection the turn keeps running and then finishes into a
socket nobody is holding. The user gets a transport timeout instead of the
governed message `respond` would have produced, **and no audit row is written**
— the compliance trail is lost precisely on the failure path, which is the path
it exists for. That is the reason this is a governance control and not a
performance tweak.

The shape of the fix
────────────────────
One budget for the turn, drawn down by every outbound call: `ensure()` before
a call, so a call that cannot finish in time never starts and a fan-out or a
retry loop stops instead of overrunning. The call's own per-attempt timeout
stays fixed (rebinding it per call would defeat the chat-model client cache),
so the worst case is the budget plus one in-flight call — which is why the
budget sits well under the gateway's bound.

`BudgetExhausted` is deliberately a distinct type rather than a `TimeoutError`:
`resilience.is_transient` matches on "timeout", and an exhausted budget must not
be retried — retrying is what spent it. The nodes catch it and route to
`respond`, so exhaustion becomes a governed outcome with an audit row rather
than a severed connection.

Why the start time lives in runtime context, not state
──────────────────────────────────────────────────────
`SupervisorContext` is rebuilt from `custom_inputs` on every invocation and is
never checkpointed (see `context.py`). Graph state is the opposite. A budget
anchored in state would be restored from the checkpoint when a human answers an
approval interrupt hours later, and the resumed turn would compute an elapsed
time of hours and refuse itself immediately. Anchoring it in per-invocation
context makes a resumed turn get a fresh budget, which is the correct answer.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


class BudgetExhausted(Exception):
    """The turn's total time budget ran out before this call could be made."""

    def __init__(self, what: str = "", spent: float = 0.0, budget: float = 0.0):
        self.what = what
        self.spent = spent
        self.budget = budget
        detail = f" before {what}" if what else ""
        super().__init__(
            f"turn budget of {budget:.0f}s exhausted{detail} (spent {spent:.1f}s)"
        )


@dataclass(frozen=True)
class Deadline:
    """How much of this turn's budget is left.

    `budget_seconds <= 0` disables the deadline entirely: `remaining()` reports
    infinity and `ensure()` never raises. That is the path every offline test and every direct graph
    invocation takes, and it keeps this module from becoming a thing tests have
    to know about in order to call a node.
    """

    started_at: float = 0.0
    budget_seconds: float = 0.0

    @classmethod
    def from_context(cls, context, budget_seconds: float) -> "Deadline":
        """The deadline for the turn `context` describes.

        A context with no `turn_started_at` — a hand-built one in a test, a
        direct graph call — yields a disabled deadline rather than one that has
        already expired. Failing closed on a missing *clock* would break every
        offline caller without protecting anything: the budget is an
        availability bound, and the controls that must fail closed (the RBAC
        gate, the guardrail screen) are elsewhere and unaffected.
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

        Call this *before* an outbound call, not after. Called before, it stops
        a fan-out or a retry loop from starting work it cannot finish; called
        after, it only reports that the overrun already happened.
        """
        if self.enabled and self.exhausted:
            raise BudgetExhausted(what, self.spent(), self.budget_seconds)


def deadline_for(context, settings) -> Deadline:
    """The turn deadline, from runtime context and settings.

    A context with no clock yields a disabled deadline — the offline-test path
    `Deadline.from_context` documents. In a *deployed* environment that same
    shape means a caller reached the endpoint without going through
    `from_custom_inputs` — the direct-call surface — and running such a
    turn with no time ceiling hands an unauthenticated caller unbounded
    fan-out. So off the local path the budget falls back to "from now": each
    node calls this separately, so the bound degrades from per-turn to per-node
    — conservative rather than exact, and strictly better than none. Model
    Serving's own request timeout remains the outer wall either way.
    """
    budget = getattr(settings, "turn_budget_seconds", 0.0)
    deadline = Deadline.from_context(context, budget)
    if not deadline.enabled and budget > 0:
        from .environment import is_local_environment

        if not is_local_environment(getattr(settings, "environment", "local")):
            return Deadline(time.monotonic(), max(0.0, float(budget)))
    return deadline
