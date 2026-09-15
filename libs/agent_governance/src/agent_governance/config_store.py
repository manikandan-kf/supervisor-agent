"""Governance configuration held in a table, not in the model artifact.

The table *is* the source of record, read at runtime, so a guardrail rule or a
policy change is a publish that reaches a running endpoint within the cache
TTL — not a redeploy. Each agent declares which documents it keeps here
(`validators`: document name → validation function) and ships the bundled YAML
seed for each; this module owns everything that is the same for all of them.

**Which table.** A UC Delta table read over the Statement Execution API cannot
work from inside Model Serving: the endpoint runs as a system service
principal that cannot be granted `databricks-sql-access`. What does work is
Lakebase Postgres, which an agent already holds an authenticated pool for and
which Unity Catalog can register as a read-only catalog for discovery. One
table, two access paths, no new entitlement.

**The row is untrusted input.** Config used to be a file inside a signed model
artifact; it is now a row in a shared database, which is a channel. Every read
is therefore checked before it may replace anything:

  * the payload is validated against the shape its consumer expects —
    including compiling every regex, so a malformed pattern is refused here
    rather than raising inside an engine on a live request;
  * the stored checksum is recomputed and compared, so a row edited in place
    rather than published through `publish()` is detected;
  * anything that fails either check is refused, loudly, and the last document
    that *was* legitimately published keeps serving — or the bundled seed, for
    a process that has never read the table at all.

**Freshness.** Reads are cached for `ttl_seconds` (default 60, matching
MLflow's prompt-alias TTL). Nothing is cached across a checksum change:
`ConfigProvider` rebuilds a consumer object only when the document differs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import yaml

from .sql import safe_identifier, table_exists_here

logger = logging.getLogger(__name__)

Validator = Callable[[Any], None]

_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
  id          BIGSERIAL PRIMARY KEY,
  name        TEXT        NOT NULL,
  version     INTEGER     NOT NULL,
  payload     JSONB       NOT NULL,
  checksum    TEXT        NOT NULL,
  active      BOOLEAN     NOT NULL DEFAULT FALSE,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_by  TEXT        NOT NULL DEFAULT '',
  comment     TEXT        NOT NULL DEFAULT '',
  UNIQUE (name, version)
)
"""

# Exactly one active version per document, enforced by the database rather than
# by the publisher being careful.
_DDL_ACTIVE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS {table}_one_active_idx
  ON {table} (name) WHERE active
"""


class ConfigError(Exception):
    """The document cannot be used — invalid, tampered with, or unknown."""


# ── canonical form and checksum ─────────────────────────────────────────────


def canonical(payload: Any) -> str:
    """The exact bytes the checksum is taken over.

    Sorted keys and no incidental whitespace, so the same document published
    twice hashes the same; `ensure_ascii` so a homoglyph substitution changes
    the digest rather than hiding inside an identical-looking string.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def checksum_of(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


# ── validation helpers, for every agent's validators ────────────────────────


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def compile_pattern(pattern: str, where: str) -> None:
    """Compile a stored regex here, where a failure is a refused publish.

    Left to the consumer, a bad pattern raises while building services for a
    live request and takes the whole turn down with it.
    """
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ConfigError(f"{where} is not a valid regular expression: {exc}") from exc


# Text a governed document injects into a *system prompt* as trusted
# instruction — an agent's name, description or domain scope — cannot be
# demoted to the untrusted JSON turn the way user input is. The compensating
# control is content validation at publish time: bounded length, no control
# characters, and none of the chat-template or turn-boundary markers
# `sanitize.py` strips from worker output.
PROMPT_FIELD_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
PROMPT_FIELD_MARKERS = re.compile(
    r"(?im)<\|[a-z_]{0,32}\|>|<\/?(?:system|assistant|user|human|im_start|im_end)>"
    r"|\[/?INST\]|\[/?SYS\]|###\s*(?:system|instruction)s?\s*:?"
    r"|^\s*(?:system|assistant|developer|tool)\s*:",
)


def validate_prompt_field(value: str, where: str, field: str, limit: int) -> None:
    """Refuse a prompt-bound string that is oversized or carries markers."""
    require(
        len(value) <= limit,
        f"{where}.{field} exceeds {limit} characters — this text is injected into "
        "governance system prompts and must stay remit-sized",
    )
    require(not PROMPT_FIELD_CONTROL.search(value), f"{where}.{field} contains control characters")
    require(
        not PROMPT_FIELD_MARKERS.search(value),
        f"{where}.{field} contains chat-template or turn-boundary markers, which have "
        "no place in prompt-bound configuration",
    )


def validate_deny_rules(payload: dict, section: str) -> None:
    """One list of {pattern, reason, action?} rules — `deny_rules.DenyRule`."""
    rules = payload.get(section, []) or []
    require(isinstance(rules, list), f"guardrails '{section}' must be a list")
    for index, rule in enumerate(rules):
        where = f"{section}[{index}]"
        require(isinstance(rule, dict), f"{where} must be a mapping")
        pattern = rule.get("pattern")
        require(isinstance(pattern, str) and pattern != "", f"{where}.pattern must be a string")
        compile_pattern(pattern, where)
        if "reason" in rule:
            require(isinstance(rule["reason"], str), f"{where}.reason must be a string")
        # A typo must refuse the publish — silently reading "escalte" as
        # "block" would downgrade a control the operator believed they raised.
        if "action" in rule:
            require(
                rule["action"] in ("block", "escalate"),
                f"{where}.action must be 'block' or 'escalate'",
            )


def _validate_output_policy(payload: dict) -> None:
    """The per-category tier map consumed by `output_guard.OutputPolicy`."""
    from . import sensitive
    from .output_guard import ACTIONS

    section = payload.get("output_policy")
    if section is None:
        return
    require(isinstance(section, dict), "guardrails 'output_policy' must be a mapping")
    categories = section.get("categories", {}) or {}
    require(isinstance(categories, dict), "guardrails 'output_policy.categories' must be a mapping")
    for category, action in categories.items():
        require(
            category in sensitive.CATEGORIES,
            f"output_policy.categories has an unknown category {category!r} — "
            f"known: {', '.join(sensitive.CATEGORIES)}",
        )
        require(
            action in ACTIONS,
            f"output_policy.categories[{category!r}] must be one of {', '.join(ACTIONS)}",
        )
        # The high tier can be masked, withheld or escalated — never delivered raw.
        require(
            not (category in sensitive.NEVER_ALLOW and action == "allow"),
            f"output_policy.categories[{category!r}] may not be 'allow': this category is "
            "never delivered unmasked",
        )
    if "escalate_at_findings" in section:
        threshold = section["escalate_at_findings"]
        require(
            isinstance(threshold, int) and not isinstance(threshold, bool) and threshold >= 0,
            "output_policy.escalate_at_findings must be a non-negative integer (0 disables)",
        )


def _validate_canaries(payload: dict) -> None:
    """Registered honeytokens: strings an operator planted in a worker's prompt."""
    from .output_guard import usable_canary

    canaries = payload.get("canary_tokens")
    if canaries is None:
        return
    require(isinstance(canaries, list), "guardrails 'canary_tokens' must be a list")
    for index, token in enumerate(canaries):
        require(
            isinstance(token, str) and 8 <= len(token.strip()) <= 64,
            f"canary_tokens[{index}] must be a string of 8 to 64 characters",
        )
        require(
            not PROMPT_FIELD_CONTROL.search(token),
            f"canary_tokens[{index}] contains control characters",
        )
        # The detector matches on the condensed form (letters and digits only),
        # so a token that condenses to nothing would be accepted here and never
        # armed. Same rule on both sides.
        require(
            usable_canary(token),
            f"canary_tokens[{index}] must contain at least 6 ASCII letters or digits — "
            "the detector matches on the condensed form, so anything less is never armed",
        )


def validate_guardrails(payload: Any) -> None:
    """The guardrails document shape every agent shares.

    Input deny patterns, output policy rules, the per-category output policy,
    canary tokens and the kill switch.
    """
    require(isinstance(payload, dict), "guardrails config must be a mapping")
    validate_deny_rules(payload, "global_deny_patterns")
    validate_deny_rules(payload, "output_deny_patterns")
    _validate_output_policy(payload)
    _validate_canaries(payload)
    if "kill_switch" in payload:
        declared = payload["kill_switch"]
        if isinstance(declared, dict):
            require(
                isinstance(declared.get("enabled"), bool),
                "guardrails 'kill_switch.enabled' must be a boolean",
            )
            if "message" in declared:
                require(
                    isinstance(declared["message"], str) and len(declared["message"]) <= 500,
                    "guardrails 'kill_switch.message' must be a string of at most 500 chars",
                )
        else:
            require(
                isinstance(declared, bool),
                "guardrails 'kill_switch' must be a boolean or a {enabled, message} mapping",
            )


# ── the document ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfigDocument:
    name: str
    payload: dict
    # "table" when it came from the governed table, "files" when the bundled
    # YAML stood in. Recorded rather than inferred: an operator looking at a
    # config that is not what they published needs to see which one is live.
    source: str
    version: Optional[int] = None
    checksum: str = ""

    @property
    def label(self) -> str:
        return f"{self.name}@{self.source}" + (f":v{self.version}" if self.version else "")


# ── the store ───────────────────────────────────────────────────────────────


class ConfigStore:
    """Reads and publishes governance documents in the Postgres table.

    Takes a *connection source* — a zero-arg callable yielding a
    context-managed connection — because on Lakebase the pool rotates
    credentials, so a pinned connection would outlive its token. `validators`
    maps each document name this agent keeps to its validation function.
    """

    def __init__(self, connection_source, table: str, validators: Mapping[str, Validator]):
        self._source = connection_source
        # Validated once, here, so every f-string below interpolates a value
        # already proven to be a bare identifier (see `sql.safe_identifier`).
        self._table = safe_identifier(table)
        self._validators = dict(validators)
        self._ready = False

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._validators)

    def validate(self, name: str, payload: Any) -> None:
        """Raise `ConfigError` unless `payload` is a usable `name` document."""
        validator = self._validators.get(name)
        if validator is None:
            raise ConfigError(f"unknown configuration document {name!r}")
        validator(payload)

    # `CREATE INDEX IF NOT EXISTS` takes an ownership check before an existence
    # check, so it raises for any identity that did not create the table — and
    # aborts the surrounding transaction with it. Check first, run DDL only
    # when there is genuinely nothing there.
    def _ensure_table(self, conn) -> None:
        if self._ready:
            return
        with conn.cursor() as cur:
            if not table_exists_here(cur, self._table):
                cur.execute(_DDL.format(table=self._table))
                cur.execute(_DDL_ACTIVE_INDEX.format(table=self._table))
        self._ready = True

    @staticmethod
    def _value(row, key: str, position: int):
        return row[key] if isinstance(row, dict) else row[position]

    def read(self, name: str) -> Optional[ConfigDocument]:
        """The active version of `name`, or None when nothing is published.

        Raises `ConfigError` when a row exists but is unusable — a failed
        checksum or a payload that does not validate. "Nothing published yet"
        is an ordinary first-run state; "published but wrong" is a tamper or a
        bad publish and has to be visible.
        """
        with self._source() as conn:
            self._ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT version, payload, checksum FROM {self._table} "
                    "WHERE name = %s AND active LIMIT 1",
                    (name,),
                )
                row = cur.fetchone()

        if row is None:
            return None

        version = self._value(row, "version", 0)
        payload = self._value(row, "payload", 1)
        stored_checksum = self._value(row, "checksum", 2)

        # psycopg returns jsonb as a parsed object; a driver that hands back
        # text should not silently produce a string-shaped "config".
        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            raise ConfigError(f"{name} v{version}: payload is not a mapping")

        actual = checksum_of(payload)
        if stored_checksum and actual != stored_checksum:
            raise ConfigError(
                f"{name} v{version}: checksum mismatch — the row was modified outside "
                f"publish() (stored {stored_checksum[:12]}…, computed {actual[:12]}…)"
            )

        self.validate(name, payload)
        return ConfigDocument(name, payload, "table", int(version), actual)

    def publish(self, name: str, payload: dict, *, actor: str = "", comment: str = "") -> int:
        """Store `payload` as a new version of `name` and make it the active one.

        Validated before it is written, so an invalid document is refused at
        publish time rather than discovered by the endpoint a minute later.
        """
        self.validate(name, payload)

        with self._source() as conn:
            self._ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COALESCE(MAX(version), 0) AS v FROM {self._table} WHERE name = %s",
                    (name,),
                )
                row = cur.fetchone()
                version = int(self._value(row, "v", 0)) + 1

                # Deactivate then insert, in one transaction, so the partial
                # unique index never sees two active rows.
                cur.execute(
                    f"UPDATE {self._table} SET active = FALSE WHERE name = %s AND active",
                    (name,),
                )
                cur.execute(
                    f"INSERT INTO {self._table} "
                    "(name, version, payload, checksum, active, created_by, comment) "
                    "VALUES (%s, %s, %s, %s, TRUE, %s, %s)",
                    (
                        name,
                        version,
                        json.dumps(payload, sort_keys=True, ensure_ascii=True),
                        checksum_of(payload),
                        actor,
                        comment,
                    ),
                )
        return version

    def activate(self, name: str, version: int) -> None:
        """Roll the active pointer to an existing version."""
        with self._source() as conn:
            self._ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT 1 FROM {self._table} WHERE name = %s AND version = %s",
                    (name, version),
                )
                if cur.fetchone() is None:
                    raise ConfigError(f"{name} has no version {version}")
                cur.execute(
                    f"UPDATE {self._table} SET active = FALSE WHERE name = %s AND active",
                    (name,),
                )
                cur.execute(
                    f"UPDATE {self._table} SET active = TRUE WHERE name = %s AND version = %s",
                    (name, version),
                )

    def history(self, name: str) -> list[dict]:
        """Every published version of `name`, newest first."""
        with self._source() as conn:
            self._ensure_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT version, checksum, active, created_at, created_by, comment "
                    f"FROM {self._table} WHERE name = %s ORDER BY version DESC",
                    (name,),
                )
                rows = cur.fetchall()

        return [
            {
                "version": self._value(row, "version", 0),
                "checksum": self._value(row, "checksum", 1),
                "active": self._value(row, "active", 2),
                "created_at": self._value(row, "created_at", 3),
                "created_by": self._value(row, "created_by", 4),
                "comment": self._value(row, "comment", 5),
            }
            for row in rows or []
        ]


# ── the provider an agent's services actually use ───────────────────────────


def load_bundled(config_dir: Path, name: str) -> dict:
    """The YAML that ships in the artifact — the fallback, and the seed."""
    path = Path(config_dir) / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class ConfigProvider:
    """Serves the current configuration to an agent, TTL-cached.

    Two caches, deliberately separate: the *document* cache bounds how often a
    live endpoint queries Postgres, and the *object* cache holds each built
    consumer keyed on the document's checksum, so a TTL expiry that finds the
    config unchanged does not recompile every regex.

    `connection_source` is a zero-arg callable that returns a connection source
    (itself a zero-arg context-manager factory) or None when no Postgres is
    configured; it is called lazily, on the first load. `source` is `auto`
    (table when Postgres is configured, else files), `table` (require the
    table; log an error and fall back if it is missing) or `files` (bundled
    YAML only — offline tests and local iteration).
    """

    def __init__(
        self,
        *,
        validators: Mapping[str, Validator],
        bundled_dir: Path,
        table: str,
        connection_source: Optional[Callable[[], Any]] = None,
        ttl_seconds: float = 60.0,
        source: str = "auto",
        store: Optional[ConfigStore] = None,
    ):
        self._validators = dict(validators)
        self._bundled_dir = Path(bundled_dir)
        self._table = table
        self._connection_source = connection_source
        self._ttl = max(0.0, float(ttl_seconds))
        self._source_mode = (source or "auto").strip().lower()
        self._store = store
        self._store_resolved = store is not None
        self._docs: dict[str, tuple[float, ConfigDocument]] = {}
        self._objects: dict[str, tuple[str, Any]] = {}
        # Complain once per document per failure mode; a 60-second TTL on a
        # broken table would otherwise write a log line a minute forever.
        self._reported: dict[str, str] = {}
        # The last document successfully read *from the table*, per name.
        # "Fall back to the bundled YAML" is not the safe default it looks like:
        # published config moves in both directions, and reverting to the
        # bundle when the table is briefly unreadable can silently *widen* the
        # policy. Serving the last known-good table document is the fail-closed
        # reading that is also available.
        self._last_good: dict[str, ConfigDocument] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._validators)

    # ── plumbing ────────────────────────────────────────────────────────────

    def _resolve_store(self) -> Optional[ConfigStore]:
        if self._store_resolved:
            return self._store
        self._store_resolved = True

        if self._source_mode == "files":
            logger.info("configuration source: bundled files (source=files)")
            return None

        source = self._connection_source() if self._connection_source else None
        if source is None:
            if self._source_mode == "table":
                logger.error(
                    "configuration source=table but no Postgres is configured — "
                    "falling back to the bundled files"
                )
            else:
                logger.info("configuration source: bundled files (no Postgres configured)")
            return None

        self._store = ConfigStore(source, self._table, self._validators)
        logger.info("configuration source: table %s", self._table)
        return self._store

    def _report(self, name: str, kind: str, message: str) -> None:
        if self._reported.get(name) == kind:
            return
        self._reported[name] = kind
        logger.error("configuration %s: %s — using the fallback instead", name, message)

    def _bundled(self, name: str) -> ConfigDocument:
        payload = load_bundled(self._bundled_dir, name)
        return ConfigDocument(name, payload, "files", None, checksum_of(payload))

    # ── documents ───────────────────────────────────────────────────────────

    def load(self, name: str) -> ConfigDocument:
        cached = self._docs.get(name)
        now = time.monotonic()
        if cached and cached[0] > now:
            return cached[1]

        document = self._load_uncached(name)
        self._docs[name] = (now + self._ttl, document)
        return document

    def _fallback(self, name: str) -> ConfigDocument:
        """The last known-good table document, or the bundled file."""
        held = self._last_good.get(name)
        if held is not None:
            logger.warning(
                "configuration %s: holding the last published version (%s) rather than "
                "reverting to the bundled file — a revert could loosen policy",
                name,
                held.label,
            )
            return held
        return self._bundled(name)

    def _load_uncached(self, name: str) -> ConfigDocument:
        store = self._resolve_store()
        if store is None:
            return self._bundled(name)

        try:
            document = store.read(name)
        except ConfigError as exc:
            # A row exists and is wrong — the security-relevant branch.
            self._report(name, "rejected", str(exc))
            return self._fallback(name)
        except Exception as exc:  # transport, permissions, table missing
            self._report(name, "unavailable", f"could not be read ({type(exc).__name__}: {exc})")
            return self._fallback(name)

        if document is None:
            self._report(name, "empty", "nothing published in the table yet")
            # Not `_fallback`: nothing published is not a failure to read, and a
            # document later deleted from the table should not be resurrected
            # from this process's memory. The bundle is the answer.
            return self._bundled(name)

        if self._reported.pop(name, None):
            logger.info("configuration %s: recovered — now serving %s", name, document.label)
        self._last_good[name] = document
        return document

    # ── built consumers ─────────────────────────────────────────────────────

    def built(self, name: str, builder: Callable[[dict], Any], cache_key: str = "") -> Any:
        """Build (or reuse) the object form of a document.

        `cache_key` separates two consumers of the *same* document — an input
        engine and an output guard both built from "guardrails" — so each keeps
        its own object without evicting the other's on every call.
        """
        key = cache_key or name
        document = self.load(name)
        cached = self._objects.get(key)
        if cached and cached[0] == document.checksum:
            return cached[1]

        obj = builder(document.payload)
        self._objects[key] = (document.checksum, obj)
        logger.info("configuration %s: now serving %s", key, document.label)
        return obj


class Reloading:
    """Delegates every attribute to whatever the provider currently serves.

    Services are built once per process, so a config object captured at build
    time would pin the configuration to the moment of the last deploy — exactly
    what moving config into a table is meant to end. Attribute lookup goes
    through the provider instead: a TTL check and two dict lookups in the common
    case. Callers cannot tell a live-reloading object from a fixed one.
    """

    __slots__ = ("_load",)

    def __init__(self, load: Callable[[], Any]):
        object.__setattr__(self, "_load", load)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Reloading({self._load()!r})"
