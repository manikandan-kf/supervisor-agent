"""Short-term (conversation) and long-term (persistent) memory.

The Postgres plumbing is the shared `agent_governance.lakebase`; this module pins it to
the supervisor's schema and adds `LongTermMemory`, the validated per-user context store.
`LAKEBASE_SCHEMA` derives from `ENVIRONMENT` (`supervisor_dev`, `supervisor_prod`), which
keeps a dev conversation out of the prod checkpoint table on a shared Lakebase instance.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass

from agent_governance import lakebase

from .settings import environment_schema

logger = logging.getLogger(__name__)


def _lakebase_schema() -> str:
    """The Postgres schema this environment's durable state lives in.

    It must not default to `public`: a process reaching Lakebase without this
    set would write its checkpoints, audit rows and governed config into
    whatever else is already there.
    """
    return os.getenv("LAKEBASE_SCHEMA") or environment_schema()


def build_checkpointer():
    """Short-term memory: conversation state, keyed by thread_id."""
    return lakebase.build_checkpointer(schema=_lakebase_schema())


def build_store():
    """Long-term memory backing store: per-user context reused across conversations."""
    return lakebase.build_store(schema=_lakebase_schema())


def audit_connection_source():
    """Connection source for the governance tables, or None when no Postgres is configured."""
    return lakebase.audit_connection_source(schema=_lakebase_schema())


# Long-term memory is an *instruction* channel: everything in it is read back into the
# routing prompt. So values must be identifiers under allowlisted keys, and four bounds apply
# because any one alone is porous — charset, length, word count, and an imperative opener.
_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-_./+&]{0,59}$")
_MAX_VALUE_LENGTH = 60
# Room for the longest plausible real label ("Building Management Systems
# Platform"), and nothing like enough for an instruction.
_MAX_VALUE_WORDS = 4
# First word only: "Route 66 Migration" is a plausible label, but nothing legitimate *leads*
# with "ignore". A rejected genuine label shows in the trail's `refused` entry.
_IMPERATIVE_OPENERS = frozenset(
    {
        "act",
        "allow",
        "always",
        "answer",
        "approve",
        "bypass",
        "disregard",
        "escalate",
        "execute",
        "follow",
        "forget",
        "forward",
        "give",
        "grant",
        "ignore",
        "make",
        "never",
        "obey",
        "override",
        "pretend",
        "reject",
        "reply",
        "respond",
        "return",
        "route",
        "run",
        "say",
        "send",
        "set",
        "skip",
        "tell",
        "treat",
        "use",
    }
)

# ── Memory expiry (guardrail layer 3) ───────────────────────────────────────
# Stale context keeps driving decisions and the user is never re-asked, so every entry
# carries its write time and reads drop the expired ones lazily (no scheduled job needed).
# Stamps sit in a sibling map in the same bag so a write stays one `put` and cannot half-apply.
_STAMP_KEY = "_written_at"

# Ninety days: long enough not to re-ask a returning user, short enough that a stale answer
# cannot outlive what it describes by a release cycle.
DEFAULT_TTL_SECONDS = 90 * 24 * 3600


@dataclass(frozen=True)
class MemoryWrite:
    """What a `save_context` call actually persisted, for the decision trail."""

    stored: dict
    # key -> why it was refused. A legitimate miss means the allowlist needs widening; a
    # rejected injection attempt is a security event.
    rejected: dict

    @property
    def clean(self) -> bool:
        return not self.rejected


def _validate(key: str, value, allowed: frozenset) -> str:
    """Why this key/value pair may not be persisted, or "" when it may.

    An empty allowlist rejects everything: a security control has to fail closed, and if no
    agent declares required context there is nothing worth carrying between conversations.
    """
    if key not in allowed:
        return "key is not a declared required_context key"
    if not isinstance(value, str):
        return f"value is {type(value).__name__}, not a string"
    if not value.strip():
        return "value is empty"
    if len(value) > _MAX_VALUE_LENGTH:
        return f"value exceeds {_MAX_VALUE_LENGTH} characters"
    if not _VALUE_PATTERN.match(value.strip()):
        return "value is not a plain identifier"
    words = value.split()
    if len(words) > _MAX_VALUE_WORDS:
        return f"value is prose, not a label (over {_MAX_VALUE_WORDS} words)"
    if len(words) > 1 and words[0].lower() in _IMPERATIVE_OPENERS:
        return "value opens with an instruction verb, not a label"
    return ""


class LongTermMemory:
    """Per-user persistent context (e.g. product line) reused across conversations.

    Reads are filtered by the same rules as writes: rows written before this validation or
    by another writer are still there, and the read path feeds the model. `allowed_keys` may
    be a set or a callable; a callable, because the registry can change under a running process.
    """

    NAMESPACE = ("supervisor", "resolved_context")

    def __init__(self, store, allowed_keys=(), ttl_seconds: float = DEFAULT_TTL_SECONDS):
        self._store = store
        self._allowed_source = allowed_keys
        # <= 0 keeps everything forever, which is why `Settings.validate`
        # refuses to serve a deployed environment configured that way.
        self._ttl = float(ttl_seconds or 0)

    @property
    def _allowed(self) -> frozenset:
        source = self._allowed_source
        keys = source() if callable(source) else source
        return frozenset(keys or ())

    def get_context(self, user_key: str, keys=None) -> dict:
        """The user's stored context, optionally narrowed to `keys`.

        `keys` keeps an agent from seeing context it never declared: the store holds one bag
        per user across every agent's `required_context`, and leaking it across agents widens
        the memory-poisoning surface (§04).
        """
        if not user_key:
            return {}
        values, stamps = self._read(user_key)
        if not values:
            return {}
        wanted = frozenset(keys) if keys is not None else None
        allowed = self._allowed
        now = time.time()
        safe = {}
        for key, value in values.items():
            if wanted is not None and key not in wanted:
                continue
            reason = _validate(key, value, allowed)
            if reason:
                logger.warning("long-term memory: dropping stored %r on read (%s)", key, reason)
                continue
            if self._expired(key, stamps, now):
                continue
            safe[key] = value.strip()
        return safe

    def _read(self, user_key: str) -> tuple[dict, dict]:
        """The stored bag split into remembered values and their write stamps."""
        item = self._store.get(self.NAMESPACE, user_key)
        if not item:
            return {}, {}
        bag = dict(item.value or {})
        stamps = bag.pop(_STAMP_KEY, None)
        return bag, dict(stamps) if isinstance(stamps, dict) else {}

    def _expired(self, key: str, stamps: dict, now: float) -> bool:
        """Whether this entry is past the retention ceiling.

        An entry with no usable stamp is expired, not fresh: only pre-bound or foreign writers
        produce one, and nothing can vouch for its age. Failing open would give an unaged value
        permanent residency in every routing prompt; failing closed costs one question.
        """
        if self._ttl <= 0:
            return False
        written = stamps.get(key)
        age = now - written if isinstance(written, (int, float)) else None
        if age is not None and age <= self._ttl:
            return False
        logger.info(
            "long-term memory: %r expired on read (%s)",
            key,
            "no write stamp" if age is None else f"{int(age // 86400)} days old",
        )
        return True

    def save_context(self, user_key: str, context: dict) -> MemoryWrite:
        """Persist only the validated, declared fields. Returns what happened.

        Merges rather than replaces: a turn that resolves only `environment`
        must not erase a `product_line` learned earlier.
        """
        if not user_key or not context:
            return MemoryWrite({}, {})

        allowed = self._allowed
        stored, rejected = {}, {}
        for key, value in dict(context).items():
            reason = _validate(key, value, allowed)
            if reason:
                rejected[key] = reason
            else:
                stored[key] = value.strip()

        if rejected:
            logger.warning(
                "long-term memory: refused %d field(s) for %s: %s",
                len(rejected),
                user_key,
                rejected,
            )

        if stored:
            # `get_context` rather than the raw bag: an expired or invalid entry is not
            # carried forward by a write that touches a different key.
            surviving = self.get_context(user_key)
            _, stamps = self._read(user_key)
            now = time.time()
            merged = {**surviving, **stored}
            merged[_STAMP_KEY] = {
                key: (now if key in stored else stamps.get(key, now)) for key in merged
            }
            self._store.put(self.NAMESPACE, user_key, merged)

        return MemoryWrite(stored, rejected)
