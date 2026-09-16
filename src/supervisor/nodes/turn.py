"""The turn itself — what every stage reads, records and measures about it.

Reading the conversation, shaping an audit entry, timing a session, deciding
whether the turn narrates. Plus `NodeBase`, the single place the service
container is bound so every stage can rely on `self.s`. Not `common.py` or
`utils.py` on purpose: everything here is about one turn, and anything that is
not does not belong here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from agent_governance.sanitize import (
    clean_worker_output,
    neutralise_embedded_directives,
    neutralise_history_text,
)
from agent_governance.sensitive import redact_text
from agent_governance.spend import (
    TurnSpend,
)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from ..guardrail_engine import small_talk_kind

if TYPE_CHECKING:
    from ..services import Services

logger = logging.getLogger(__name__)


class NodeBase:
    """The service container every stage reads its collaborators from.

    Stages are mixins over this class rather than free functions so they share
    `self.s` without threading the container through every signature.
    """

    s: Services

    def __init__(self, services: Services) -> None:
        self.s = services


# Outcomes that *are* a governance decision, which §05 Stage 06 requires be written
# synchronously. `answer` is deliberately absent: a failed audit write must fail a block, not
# a plain answer. Sign-offs keep `outcome == "answer"`; see `_is_governance_decision`.
GOVERNANCE_OUTCOMES = frozenset({"blocked", "escalated", "expired", "review_pending"})


def _narrates(state: dict) -> bool:
    """Whether this turn should stream a task plan at all.

    Small talk goes through no gate, so narrating stages for "hi" claims work that never
    happened. Decided from the message, not a result, so the first event is suppressed too.
    """
    return not small_talk_kind(_latest_user_text(state.get("messages")))


# Anchored, so only a message that is *entirely* an appeal counts. "appeal the
# decision to buy" inside a real request must not silently become one.
_APPEAL_PATTERN = re.compile(r"^\W*appeal\b[\s\S]{0,120}$", re.IGNORECASE)


def _small_talk_seen(messages, kind: str) -> int:
    """How many earlier user turns in this conversation were this same kind.

    Counted from the checkpointed conversation so the escalation survives across turns.
    """
    if not kind:
        return 0
    seen = 0
    for msg in (messages or [])[:-1]:  # the current turn is the last message
        if getattr(msg, "type", "") != "human":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if small_talk_kind(content) == kind:
            seen += 1
    return seen


def _latest_user_text(messages) -> str:
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") == "human":
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _prior_user_text(messages) -> str:
    """The user turn *before* the current one — on an appeal, the request that was blocked.

    Falls back to the current turn so an excerpt is never empty for want of history.
    """
    seen = 0
    for msg in reversed(messages or []):
        if getattr(msg, "type", "") != "human":
            continue
        seen += 1
        if seen == 2:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return _latest_user_text(messages)


def _window(messages, max_tokens: int) -> list:
    """The most recent messages that fit a token budget.

    The checkpointer keeps the whole conversation; only the slice replayed into model calls
    is bounded. `start_on="human"` so the model never sees an answer to a dropped question.
    """
    messages = list(messages or [])
    if not messages:
        return messages
    try:
        trimmed = trim_messages(
            messages,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=count_tokens_approximately,
            start_on="human",
        )
    except Exception:
        logger.warning("history trimming failed — sending the full window", exc_info=True)
        return messages
    return trimmed or messages[-1:]


def _history_lines(messages) -> list[str]:
    """Flatten the conversation for the governance prompts, markers defanged.

    Every line is neutralised, not only worker output: prior *user* turns carry a fake
    `system:` boundary as readily as a worker reply (the multi-turn injection channel).
    """
    lines = []
    for msg in messages or []:
        role = {"human": "user", "ai": "assistant"}.get(getattr(msg, "type", ""), "other")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {neutralise_history_text(content)}")
    return lines


@dataclass(frozen=True)
class RelayScreen:
    """What the relay screen did to the conversation bound for a worker."""

    messages: list
    # Labels of the values masked, in order of first appearance.
    masked: tuple[str, ...] = ()
    directives: int = 0

    @property
    def modified(self) -> bool:
        return bool(self.masked or self.directives)

    def audit_detail(self) -> str:
        parts = []
        if self.masked:
            parts.append(
                f"{len(self.masked)} sensitive value(s) masked before relay: "
                + ", ".join(dict.fromkeys(self.masked))
            )
        if self.directives:
            parts.append(f"{self.directives} embedded directive marker(s) neutralised")
        return "; ".join(parts) or "clean"


def _worker_messages(messages, guard=None) -> RelayScreen:
    """The conversation a worker receives — screened at the relay boundary.

    Without this a worker got `msg.content` verbatim while `_history_lines` defanged only
    the governance prompts. User turns are masked and directive frames neutralised on relay.
    """
    out = []
    masked: list[str] = []
    directives = 0
    window = [m for m in (messages or []) if getattr(m, "type", "") in ("human", "ai")]
    latest_human = max(
        (i for i, m in enumerate(window) if getattr(m, "type", "") == "human"), default=-1
    )
    for index, msg in enumerate(window):
        msg_type = getattr(msg, "type", "")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if msg_type == "human":
            content, count = neutralise_embedded_directives(content)
            directives += count
            if guard is not None:
                content, findings = guard.relay(content)
                # Only the turn being answered feeds `masked`: it becomes a notice to the
                # user, and a credential pasted six turns ago must not re-announce itself.
                if index == latest_human:
                    masked.extend(f.label for f in findings)
        out.append({"role": "user" if msg_type == "human" else "assistant", "content": content})
    return RelayScreen(out, tuple(masked), directives)


_SOURCE_FIELD_MAX = 200


def _clean_sources(sources, guard=None) -> list[dict]:
    """Screen the worker-declared grounding a UI renders under an answer.

    `title` and `origin` are rendered as sent, so unscreened they were a channel for a
    credential or directive frame. Bounded and screened once, before the stream or state.
    """
    out: list[dict] = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        cleaned: dict = {}
        for key, value in source.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            text = neutralise_history_text(value)[:_SOURCE_FIELD_MAX]
            if guard is not None:
                text, _ = guard.relay(text)
            cleaned[key] = text
        if cleaned:
            out.append(cleaned)
    return out


def _entry(stage: str, decision: str, detail: str) -> dict:
    return {
        "stage": stage,
        "decision": decision,
        "detail": detail,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def _trail(state: dict, stage: str, decision: str, detail: str) -> list:
    return state.get("audit_trail", []) + [_entry(stage, decision, detail)]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seconds_since(stamp, what: str) -> Optional[float]:
    """Seconds since a checkpointed ISO timestamp, or None when unusable.

    A malformed value returns None rather than raising: an unparseable timestamp must not
    be able to expire — or refuse to expire — a session on its own.
    """
    if not stamp:
        return None
    try:
        began = datetime.fromisoformat(str(stamp))
    except ValueError:
        logger.warning("unparseable %s %r — age unknown", what, stamp)
        return None
    if began.tzinfo is None:
        began = began.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - began).total_seconds())


def _session_age_seconds(state: dict) -> Optional[float]:
    """How long this conversation has been open — the audit row's Duration KPI."""
    return _seconds_since(state.get("session_started_at"), "session_started_at")


def _session_idle_seconds(state: dict) -> Optional[float]:
    """Time since the last turn — what the lifetime bound measures (§05 Stage 04).

    "Abandoned" is about inactivity, not total age. Conversations checkpointed before
    `session_last_active_at` existed fall back to the start time, the stricter measure.
    """
    idle = _seconds_since(state.get("session_last_active_at"), "session_last_active_at")
    return idle if idle is not None else _session_age_seconds(state)


def _is_governance_decision(state: dict) -> bool:
    """Whether this turn *decided* something, as opposed to answering something.

    A sign-off keeps `outcome == "answer"` so the routing-completion KPI is unaffected, but a
    human approving or rejecting an artifact must not go unrecorded.
    """
    return bool(state.get("outcome", "") in GOVERNANCE_OUTCOMES or state.get("signoff"))


def _provenance(settings, spend: TurnSpend) -> dict:
    """How this turn was decided: which model, which prompts, what it spent.

    §4.1 promotes prompt versions by a *movable* alias, so "the dev alias" does not identify
    the text behind a verdict. `prompts` is process-scoped; `{}` when nothing is to report.
    """
    from ..prompt_provider import loaded_prompt_versions

    versions = loaded_prompt_versions()
    by_stage = spend.record()["by_stage"]
    if not versions and not by_stage:
        return {}
    record: dict = {"routing_model": getattr(settings, "routing_llm_endpoint", "")}
    if versions:
        record["prompts"] = versions
    if by_stage:
        record["spend_by_stage"] = by_stage
    return record


def _excerpt(text: str, limit: int = 500) -> str:
    """A short, safe copy of the user's query for a reviewer to read.

    Bounded and sanitized because a reviewer's console renders it; redacted last because
    blocked queries are where pasted credentials show up.
    """
    return redact_text(clean_worker_output(text or "", max_chars=limit).text)[0]
