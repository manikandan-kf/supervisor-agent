"""Tier-1 deterministic guardrail rules — the shape every agent's screen shares.

A rule is `{pattern, reason, action?}`; `action` is `block` (default) or `escalate` (refuse and
hand to a human reviewer). One shape governs input deny patterns and output policy rules.
Precision over recall: a false block costs the user their route in, and the semantic tier judges
what these do not catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

BLOCK = "block"
ESCALATE = "escalate"


@dataclass(frozen=True)
class DenyRule:
    regex: re.Pattern[str]
    reason: str
    action: str = BLOCK

    @property
    def escalate(self) -> bool:
        return self.action == ESCALATE


def compile_rules(
    rules: Optional[Iterable[dict]], *, default_reason: str = "matched a blocked pattern"
) -> tuple[DenyRule, ...]:
    """Compile a governed rule list once, at construction.

    A pattern that does not compile raises here. `config_store` refuses such a
    document at publish time, so on the deployed path this never fires; a
    hand-built engine in a test gets the error at the point it made the mistake.
    """
    compiled = []
    for rule in rules or ():
        action = rule.get("action")
        compiled.append(
            DenyRule(
                re.compile(rule["pattern"]),
                rule.get("reason", default_reason),
                action if action in (BLOCK, ESCALATE) else BLOCK,
            )
        )
    return tuple(compiled)


def first_match(rules: Iterable[DenyRule], text: str) -> Optional[DenyRule]:
    """The first rule that matches `text`, or None. First match decides."""
    for rule in rules:
        if rule.regex.search(text or ""):
            return rule
    return None


# Shown when the kill switch is engaged and the operator supplied no message of
# their own. Worded as an operational hold, not a refusal — nothing was judged.
KILL_SWITCH_MESSAGE = (
    "The assistant is temporarily paused by the operations team. Nothing was sent "
    "to an agent — please try again later."
)


def kill_switch_message(declared) -> str:
    """The operator's hold message, or "" when the switch is off.

    Accepts both shapes a governed document may carry — `kill_switch: true` and
    `kill_switch: {enabled: true, message: "..."}` — so an emergency stop is one
    line in a publish. Anything else means off.
    """
    if declared is True:
        return KILL_SWITCH_MESSAGE
    if isinstance(declared, dict) and declared.get("enabled") is True:
        return str(declared.get("message") or "").strip() or KILL_SWITCH_MESSAGE
    return ""
