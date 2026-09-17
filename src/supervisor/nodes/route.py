"""Stage 3 — route / clarify.

The target agent is already fixed, so routing here resolves the context the
worker needs and asks one clarifying question when it cannot.
"""

from __future__ import annotations

import logging
from typing import Literal

from agent_governance.retry_and_deadline import BudgetExhausted, deadline_for
from langgraph.runtime import Runtime
from langgraph.types import Command

from ..state import SupervisorContext
from ..user_facing_text import (
    GOVERNANCE_UNAVAILABLE_MESSAGE,
)
from .base import (
    _entry,
    _history_lines,
    _trail,
    _window,
)
from .failsafes import FailsafesMixin

logger = logging.getLogger(__name__)


class RouteMixin(FailsafesMixin):
    def route(
        self, state: dict, runtime: Runtime[SupervisorContext]
    ) -> Command[Literal["dispatch", "respond"]]:
        context = runtime.context or SupervisorContext()
        deadline = deadline_for(context, self.s.settings)
        agent = self.s.registry.get(state["target_agent_id"])

        # Never the role. `user_key or user_role` once made the memory key the *role string*,
        # so one BA's product line was read back into another's routing prompt (BRD NFR, GDPR
        # Art. 5(1)(f)). With no subject to key on, the store is skipped entirely.
        user_key = context.user_key
        agent_keys = set(agent.required_context)

        # ── Session isolation ───────────────────────────────────────────────
        # The conversation's own `session_context` wins and the long-term store is only a seed:
        # otherwise a second conversation about beta would silently turn the first, still about
        # alpha, into beta artifacts. Both are narrowed to the keys the target agent declared.
        session_context = {
            key: value
            for key, value in (state.get("session_context") or {}).items()
            if key in agent_keys
        }

        remembered: dict = {}
        try:
            remembered = self.s.memory.get_context(user_key, keys=agent_keys)
        except Exception:
            logger.warning("long-term memory read failed", exc_info=True)

        prior = {**remembered, **session_context}
        seeded = {k: v for k, v in remembered.items() if k not in session_context}

        # Same fail-closed rule as the screen (solution §05): a routing model that cannot be reached
        # holds the request rather than dispatching it with unresolved context.
        window = _window(state.get("messages"), self.s.settings.history_max_tokens)

        try:
            deadline.ensure("context resolution")
            result = self.s.router.resolve(
                agent,
                _history_lines(window),
                prior,
                deadline=deadline,
                # What this conversation never said. Named separately so the
                # router can weigh it differently.
                carried_over=seeded,
            )
        except BudgetExhausted as exc:
            return self._budget_exhausted(state, "route", exc)
        except Exception as exc:
            # Scrubbed for the same reason as the guardrail path: the exception
            # can carry the prompt, and the prompt carries the user's data.
            logger.error(
                "context resolution failed for %s: %s: %s",
                agent.id,
                type(exc).__name__,
                str(exc)[:200],
            )
            logger.debug("context resolution traceback", exc_info=True)
            return Command(
                goto="respond",
                update={
                    "outcome": "error",
                    "final_text": GOVERNANCE_UNAVAILABLE_MESSAGE,
                    "audit_trail": _trail(
                        state,
                        "route",
                        "fail_closed",
                        f"context resolution unavailable ({type(exc).__name__}) — request held",
                    ),
                },
            )

        if result.ready:
            # §04: what long-term memory accepted and refused belongs in the trail — a refused
            # field is either an allowlist to widen or an injection attempt; neither is silent.
            memory_entries = []
            # Provenance: context the user never typed here still shapes the artifact, so its
            # origin has to be reviewable rather than inferred from its absence.
            if session_context:
                memory_entries.append(
                    _entry(
                        "memory",
                        "session",
                        f"context this conversation already resolved: {session_context}",
                    )
                )
            if result.dropped_context:
                # The request moved on. Recorded loudly: "why did it stop using alpha" is a
                # question the trail has to be able to answer.
                memory_entries.append(
                    _entry(
                        "memory",
                        "dropped",
                        "carried context discarded — this request is about a different "
                        f"subject: {', '.join(result.dropped_context)}",
                    )
                )
            if seeded:
                memory_entries.append(
                    _entry(
                        "memory",
                        "seeded",
                        f"long-term context seeded into a new conversation: {seeded}",
                    )
                )
            try:
                write = self.s.memory.save_context(user_key, result.resolved_context)
                if write.stored:
                    memory_entries.append(
                        _entry("memory", "persisted", f"long-term context: {write.stored}")
                    )
                if write.rejected:
                    memory_entries.append(
                        _entry(
                            "memory",
                            "refused",
                            f"unvalidated long-term context refused: {write.rejected}",
                        )
                    )
            except Exception:
                logger.warning("long-term memory write failed", exc_info=True)
                memory_entries.append(_entry("memory", "error", "long-term memory write failed"))
            detail = (
                f"resolved context: {result.resolved_context}"
                if result.context_applies
                else "request does not depend on the agent's required context"
            )
            # Pin what was dispatched on to this conversation so another conversation cannot
            # move it. Pinned on dispatch only, never on clarify: binding to a guess is what the
            # question exists to avoid. Dropped keys stay dropped — the router just decided so.
            kept = {k: v for k, v in session_context.items() if k not in result.dropped_context}
            pinned = {
                **kept,
                **{k: v for k, v in result.resolved_context.items() if k in agent_keys},
            }
            # Seed context that shaped this dispatch is told to the user once: a value they cannot
            # see in the transcript, silently deciding what gets built, is the whole complaint.
            used_seed = {
                k: v
                for k, v in seeded.items()
                if k not in result.dropped_context and result.resolved_context.get(k) == v
            }
            if used_seed:
                memory_entries.append(
                    _entry(
                        "memory",
                        "carried",
                        "dispatched on context carried from an earlier conversation, "
                        f"and said so in the reply: {used_seed}",
                    )
                )

            return Command(
                goto="dispatch",
                update={
                    "route": {
                        "next": "dispatch",
                        "resolved_context": result.resolved_context,
                        "carried_over": used_seed,
                        "dropped_context": list(result.dropped_context),
                    },
                    "session_context": pinned,
                    "clarification_count": 0,
                    "pending_clarification": None,
                    "audit_trail": _trail(state, "route", "dispatch", detail) + memory_entries,
                },
            )

        return self._clarify_or_escalate(
            state,
            "route",
            result.clarifying_question,
            {"route": {"next": "respond", "resolved_context": result.resolved_context}},
        )
