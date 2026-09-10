"""Governance configuration held in a table, not in the model artifact.

R1 requires the supervisor's configuration to live "directly in Unity Catalog
tables (pre-wrapper-API)" — the table *is* the source of record, read at
runtime, with no validating write service in front of it (that was Solution
v1.1 §05, which v1.2 dropped and this module deliberately does not reimplement).

Three documents are stored, matching the three files that used to ship inside
the artifact:

    agents      the worker registry     (registry.AgentRegistry)
    rbac        the fallback role map   (rbac.RbacPolicy)
    guardrails  tier-1 deny patterns    (guardrails.GuardrailEngine)

**Which table, and why this one.** The literal reading — a UC Delta table read
over the Statement Execution API — cannot work from inside Model Serving, and
that is settled, not untried: the endpoint runs as a System Service Principal
that does not appear in SCIM, so it can never be granted `databricks-sql-access`
and every warehouse statement it issues is refused. That is the same wall that
rules out the Delta audit sink.

What does work is the table this project already reaches: Lakebase Postgres,
registered in Unity Catalog as the read-only catalog `supervisor_memory`.
So `supervisor_memory.public.supervisor_config` is a genuine Unity
Catalog table — governed, SQL-queryable through a warehouse, visible in Catalog
Explorer — while the endpoint reads the same rows over the Postgres protocol it
already holds an authenticated pool for. One table, two access paths, no new
entitlement.

**The row is untrusted input.** Config used to be a file inside a signed model
artifact; it is now a row in a shared database, which is a channel. Every read
is therefore checked three ways before it is allowed to replace anything:

  * the payload is validated against the shape its consumer expects — including
    compiling every regex, so a malformed deny pattern is refused here rather
    than raising inside the guardrail engine on a live request;
  * the stored checksum is recomputed and compared, so a row edited in place
    (rather than published through `publish()`) is detected;
  * anything that fails either check is refused and the **bundled YAML is used
    instead**, loudly. Failing closed onto a known-good config beats failing
    open onto an attacker-supplied one, and beats crashing the endpoint.

**Freshness.** Reads are cached for `CONFIG_CACHE_TTL_SECONDS` (default 60,
matching MLflow's prompt-alias TTL), so a published change reaches a running
endpoint within a minute with no redeploy — which is what "configuration held
in tables" is for. Nothing is cached across a checksum change: `ConfigProvider`
rebuilds the consumer object only when the document actually differs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

logger = logging.getLogger(__name__)

CONFIG_NAMES = ("agents", "rbac", "guardrails")

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
# by the publisher being careful. A partial unique index is the only way to say
# that in Postgres, and it means a half-finished rollback cannot leave two rows
# claiming to be live.
_DDL_ACTIVE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS {table}_one_active_idx
  ON {table} (name) WHERE active
"""


class ConfigError(Exception):
    """The stored document cannot be used."""


# SQL cannot parameterize an *identifier*, so a table name has to be
# interpolated into the statement text — which is what every static analyser
# flags here (Bandit B608 / ruff S608), correctly, because interpolation is the
# SQL-injection shape whether or not this particular value is reachable.
#
# The value comes from `SUPERVISOR_CONFIG_TABLE`, so an attacker would already
# need to control the process environment, at which point the config table is
# not the interesting target. That argument is true and it is also the argument
# every injection post-mortem contains, so it is not relied on: the identifier
# is validated on the way in instead, and the suppression downstream then rests
# on an enforced property rather than on trust.
#
# Optionally qualified (`catalog.schema.table`) because the Delta sink and the
# export job both address tables that way.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}$")


def safe_identifier(name: str, *, what: str = "table") -> str:
    """Return `name` if it is a plain SQL identifier, else raise.

    Deliberately strict: letters, digits and underscores, up to three
    dot-separated parts. No quotes, no whitespace, no semicolons, no comment
    markers — nothing that can end the identifier and start a new clause.
    """
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise ConfigError(f"unsafe {what} identifier: {name!r}")
    return name


# The schema an unqualified CREATE would write to — which is not the schema an
# unqualified SELECT would read from. See `table_exists_here`.
_TABLE_PRESENT_SQL = (
    "SELECT to_regclass(quote_ident(current_schema()) || '.' || quote_ident(%s)) "
    "IS NOT NULL AS present"
)


def table_exists_here(cur, table: str) -> bool:
    """Does `table` exist in the schema this connection would *create* it in?

    Deliberately not `to_regclass('<table>')`, which is the question this used
    to ask and the wrong one. Postgres resolves an unqualified *reference* to
    the first schema in `search_path` that holds the name, but an unqualified
    `CREATE TABLE` always targets the first schema in the path. Those differ the
    moment an environment schema sits ahead of `public`.

    That is exactly the deployed shape once dev and prod are separate schemas:
    `search_path = supervisor_dev, public`, with same-named tables left in
    `public` by an earlier single-schema deployment. An unqualified probe finds
    `public.supervisor_audit_log`, answers "already there", skips the DDL — and
    every read and write then resolves to `public`. Two environments share one
    audit trail, one governed configuration and one review queue, with nothing
    in any log to say so.

    `current_schema()` is the first schema in the path that actually exists —
    precisely where the CREATE would land — so the probe and the DDL agree. In
    the single-schema shape (`"$user", public`, no user schema) it is `public`,
    and nothing changes.

    Used by all three governed stores, which is why it lives here beside
    `safe_identifier` rather than being restated in each.
    """
    cur.execute(_TABLE_PRESENT_SQL, (table,))
    row = cur.fetchone()
    return bool(row["present"] if isinstance(row, dict) else row[0])


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


# ── validation ──────────────────────────────────────────────────────────────

# Agent ids reach the invocation path, the role names derived from it
# and the Postgres audit rows. Constraining them here keeps an id that would
# break one of those from ever being published.
_AGENT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_CONTEXT_KEY = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
# Tool/agent risk classification (registry.WorkerAgent.risk_level).
_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})

# `name`, `description` and `domain_scope` are interpolated into the guardrail
# and routing **system prompts** as trusted instruction text — the domain scope
# *is* the remit the screen judges against, so it cannot be demoted to the
# untrusted JSON turn the way user input is. The compensating control is
# content validation at publish time: bounded length (a scope is a remit, not a
# treatise — and an unbounded one is prompt real estate for smuggled
# instructions), no control characters, and none of the chat-template / turn-
# boundary markers `sanitize.py` strips from worker output. A value that fails
# is a refused publish, not a live surprise. This narrows what a compromised or
# careless publish can smuggle into the system prompt; the checksum + publish()
# pipeline above remains the control over *who* can change these at all.
_PROMPT_FIELD_LIMITS = {"name": 120, "description": 2000, "domain_scope": 4000}
_PROMPT_FIELD_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_PROMPT_FIELD_MARKERS = re.compile(
    r"(?im)<\|[a-z_]{0,32}\|>|<\/?(?:system|assistant|user|human|im_start|im_end)>"
    r"|\[/?INST\]|\[/?SYS\]|###\s*(?:system|instruction)s?\s*:?"
    r"|^\s*(?:system|assistant|developer|tool)\s*:",
)


def _validate_prompt_field(value: str, where: str, field: str) -> None:
    limit = _PROMPT_FIELD_LIMITS[field]
    _require(
        len(value) <= limit,
        f"{where}.{field} exceeds {limit} characters — this text is injected into "
        "governance system prompts and must stay remit-sized",
    )
    _require(
        not _PROMPT_FIELD_CONTROL.search(value),
        f"{where}.{field} contains control characters",
    )
    _require(
        not _PROMPT_FIELD_MARKERS.search(value),
        f"{where}.{field} contains chat-template or turn-boundary markers, which have "
        "no place in prompt-bound configuration",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _validate_agents(payload: Any) -> None:
    _require(isinstance(payload, dict), "agents config must be a mapping")
    agents = payload.get("agents")
    _require(isinstance(agents, list) and agents, "agents config needs a non-empty 'agents' list")

    seen: set[str] = set()
    for index, row in enumerate(agents):
        where = f"agents[{index}]"
        _require(isinstance(row, dict), f"{where} must be a mapping")

        agent_id = row.get("id")
        _require(
            isinstance(agent_id, str) and bool(_AGENT_ID.match(agent_id)),
            f"{where}.id must be a lowercase slug, got {agent_id!r}",
        )
        _require(agent_id not in seen, f"duplicate agent id {agent_id!r}")
        seen.add(agent_id)

        _require(
            isinstance(row.get("endpoint"), str) and row["endpoint"].strip() != "",
            f"{where}.endpoint must be a non-empty string",
        )
        for field in ("name", "description", "domain_scope"):
            if field in row:
                _require(isinstance(row[field], str), f"{where}.{field} must be a string")
                _validate_prompt_field(row[field], where, field)

        # Multi-model support: optional per-agent model reference,
        # a serving-endpoint name. Validated whenever present — the flag that
        # gates *using* it is runtime configuration, and a document published
        # while the feature is dark must still be a document that works the day
        # it is lit. Present-but-empty is refused rather than treated as "no
        # override": a blank here is an unfinished edit, not a decision.
        if "model" in row and row["model"] is not None:
            _require(
                isinstance(row["model"], str) and row["model"].strip() != "",
                f"{where}.model must be a non-empty serving-endpoint name when present",
            )

        required = row.get("required_context", []) or []
        _require(isinstance(required, list), f"{where}.required_context must be a list")
        for key in required:
            _require(
                isinstance(key, str) and bool(_CONTEXT_KEY.match(key)),
                f"{where}.required_context contains an invalid key {key!r}",
            )

        patterns = row.get("deny_patterns", []) or []
        _require(isinstance(patterns, list), f"{where}.deny_patterns must be a list")
        for pattern in patterns:
            _require(isinstance(pattern, str), f"{where}.deny_patterns must contain strings")
            _compile(pattern, f"{where}.deny_patterns")

        # Supervisor-enforced approval (registry.approval_reason). Same
        # {pattern, reason} shape as the guardrails document's deny rules, and
        # validated the same way — a pattern that cannot compile must be a
        # refused publish, not an exception inside a dispatch. The `reason` is
        # shown to the human approver, so it is bounded and marker-free for the
        # same reasons the prompt-bound fields above are.
        approvals = row.get("approval_patterns", []) or []
        _require(isinstance(approvals, list), f"{where}.approval_patterns must be a list")
        for position, rule in enumerate(approvals):
            at = f"{where}.approval_patterns[{position}]"
            _require(isinstance(rule, dict), f"{at} must be a mapping")
            pattern = rule.get("pattern")
            _require(
                isinstance(pattern, str) and pattern != "", f"{at}.pattern must be a string"
            )
            _compile(pattern, at)
            reason = rule.get("reason")
            _require(
                isinstance(reason, str) and 0 < len(reason) <= 500,
                f"{at}.reason must be a non-empty string of at most 500 characters — it is "
                "shown to the human being asked to approve",
            )
            _require(
                not _PROMPT_FIELD_CONTROL.search(reason) and not _PROMPT_FIELD_MARKERS.search(reason),
                f"{at}.reason contains control characters or turn-boundary markers",
            )

        # Classification only, but a typo must not silently become a class of
        # its own — a report that groups by risk level would then under-count
        # the very traffic it exists to surface.
        if "risk_level" in row and row["risk_level"] is not None:
            _require(
                isinstance(row["risk_level"], str)
                and row["risk_level"].strip().lower() in _RISK_LEVELS,
                f"{where}.risk_level must be one of {', '.join(sorted(_RISK_LEVELS))}",
            )


def _validate_rbac(payload: Any) -> None:
    _require(isinstance(payload, dict), "rbac config must be a mapping")
    roles = payload.get("roles")
    _require(isinstance(roles, dict), "rbac config needs a 'roles' mapping")
    for role, agents in roles.items():
        _require(isinstance(role, str) and role.strip() != "", f"invalid role name {role!r}")
        _require(isinstance(agents, list), f"roles[{role!r}] must be a list of agent ids")
        for agent_id in agents:
            _require(
                isinstance(agent_id, str) and bool(_AGENT_ID.match(agent_id)),
                f"roles[{role!r}] contains an invalid agent id {agent_id!r}",
            )


def _validate_deny_rules(payload: dict, section: str) -> None:
    """One list of {pattern, reason} rules — the shape both tiers share."""
    rules = payload.get(section, []) or []
    _require(isinstance(rules, list), f"guardrails '{section}' must be a list")
    for index, rule in enumerate(rules):
        where = f"{section}[{index}]"
        _require(isinstance(rule, dict), f"{where} must be a mapping")
        pattern = rule.get("pattern")
        _require(isinstance(pattern, str) and pattern != "", f"{where}.pattern must be a string")
        _compile(pattern, where)
        if "reason" in rule:
            _require(isinstance(rule["reason"], str), f"{where}.reason must be a string")
        # Optional tier: a match refuses (`block`, the default) or refuses
        # *and* hands the conversation to a reviewer (`escalate`). A typo must
        # refuse the publish — silently reading "escalte" as "block" would
        # downgrade a control the operator believed they had raised.
        if "action" in rule:
            _require(
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
    _require(isinstance(section, dict), "guardrails 'output_policy' must be a mapping")
    categories = section.get("categories", {}) or {}
    _require(isinstance(categories, dict), "guardrails 'output_policy.categories' must be a mapping")
    for category, action in categories.items():
        _require(
            category in sensitive.CATEGORIES,
            f"output_policy.categories has an unknown category {category!r} — "
            f"known: {', '.join(sensitive.CATEGORIES)}",
        )
        _require(
            action in ACTIONS,
            f"output_policy.categories[{category!r}] must be one of {', '.join(ACTIONS)}",
        )
        # The high tier can be masked, withheld or escalated — never delivered
        # raw. A publish that says otherwise is refused rather than honoured.
        _require(
            not (category in sensitive.NEVER_ALLOW and action == "allow"),
            f"output_policy.categories[{category!r}] may not be 'allow': this category is "
            "never delivered unmasked",
        )
    if "escalate_at_findings" in section:
        threshold = section["escalate_at_findings"]
        _require(
            isinstance(threshold, int) and not isinstance(threshold, bool) and threshold >= 0,
            "output_policy.escalate_at_findings must be a non-negative integer (0 disables)",
        )


def _validate_canaries(payload: dict) -> None:
    """Registered honeytokens: strings an operator planted in a worker's prompt."""
    canaries = payload.get("canary_tokens")
    if canaries is None:
        return
    _require(isinstance(canaries, list), "guardrails 'canary_tokens' must be a list")
    for index, token in enumerate(canaries):
        _require(
            isinstance(token, str) and 8 <= len(token.strip()) <= 64,
            f"canary_tokens[{index}] must be a string of 8 to 64 characters",
        )
        _require(
            not _PROMPT_FIELD_CONTROL.search(token),
            f"canary_tokens[{index}] contains control characters",
        )
        # The consumer matches on the condensed form — letters and digits only,
        # so that letter-spacing and punctuation cannot evade it — and skips
        # anything with fewer than six of those. Validating only the raw length
        # let `"--------"` and a Cyrillic token pass the publish and then be
        # silently dropped: an accepted publish, a green `verify_deployment`,
        # and an inert honeytoken. Same rule on both sides.
        from .output_guard import usable_canary

        _require(
            usable_canary(token),
            f"canary_tokens[{index}] must contain at least 6 ASCII letters or digits — "
            "the detector matches on the condensed form, so anything less is never armed",
        )


def _validate_guardrails(payload: Any) -> None:
    _require(isinstance(payload, dict), "guardrails config must be a mapping")
    _validate_deny_rules(payload, "global_deny_patterns")
    # Layer 7: deterministic output policy rules, consumed by output_guard.py.
    _validate_deny_rules(payload, "output_deny_patterns")
    _validate_output_policy(payload)
    _validate_canaries(payload)
    # The emergency stop (see guardrails.GuardrailEngine). `true` or
    # `{enabled: bool, message: str}`; anything else is a refused publish, so a
    # typo'd switch cannot silently be "off" when the operator meant "on".
    if "kill_switch" in payload:
        declared = payload["kill_switch"]
        if isinstance(declared, dict):
            _require(
                isinstance(declared.get("enabled"), bool),
                "guardrails 'kill_switch.enabled' must be a boolean",
            )
            if "message" in declared:
                _require(
                    isinstance(declared["message"], str) and len(declared["message"]) <= 500,
                    "guardrails 'kill_switch.message' must be a string of at most 500 chars",
                )
        else:
            _require(
                isinstance(declared, bool),
                "guardrails 'kill_switch' must be a boolean or a {enabled, message} mapping",
            )


def _compile(pattern: str, where: str) -> None:
    """Compile a stored regex here, where a failure is a refused publish.

    Left to the consumer, a bad pattern raises inside `GuardrailEngine.__init__`
    — which runs while building services for a live request, and would take the
    whole turn down with it.
    """
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ConfigError(f"{where} is not a valid regular expression: {exc}") from exc


_VALIDATORS: dict[str, Callable[[Any], None]] = {
    "agents": _validate_agents,
    "rbac": _validate_rbac,
    "guardrails": _validate_guardrails,
}


def validate(name: str, payload: Any) -> None:
    """Raise `ConfigError` unless `payload` is a usable `name` document."""
    validator = _VALIDATORS.get(name)
    if validator is None:
        raise ConfigError(f"unknown configuration document {name!r}")
    validator(payload)


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
    """Reads and publishes governance documents in the Postgres/UC table.

    Takes a *connection source* — a zero-arg callable yielding a
    context-managed connection — for the same reason `PostgresAuditLogger`
    does: on Lakebase the pool rotates credentials, so a pinned connection
    would outlive its token.
    """

    def __init__(self, connection_source, table: str):
        self._source = connection_source
        # Validated once, here, rather than at each of the eight statements
        # below — so every f-string in this class interpolates a value that has
        # already been proven to be a bare identifier.
        self._table = safe_identifier(table)
        self._ready = False

    # The §24.3 trap, again: `CREATE INDEX IF NOT EXISTS` takes an ownership
    # check before an existence check, so it raises for any identity that did
    # not create the table — and aborts the surrounding transaction with it.
    # Check first, run DDL only when there is genuinely nothing there.
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
        checksum or a payload that does not validate. That distinction matters
        to the caller: "nothing published yet" is an ordinary first-run state,
        while "published but wrong" is a tamper or a bad publish and has to be
        visible.
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

        validate(name, payload)
        return ConfigDocument(name, payload, "table", int(version), actual)

    def publish(self, name: str, payload: dict, *, actor: str = "", comment: str = "") -> int:
        """Store `payload` as a new version of `name` and make it the active one.

        Validated before it is written, so an invalid document is refused at
        publish time rather than discovered by the endpoint a minute later.
        """
        validate(name, payload)

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

        out = []
        for row in rows or []:
            out.append(
                {
                    "version": self._value(row, "version", 0),
                    "checksum": self._value(row, "checksum", 1),
                    "active": self._value(row, "active", 2),
                    "created_at": self._value(row, "created_at", 3),
                    "created_by": self._value(row, "created_by", 4),
                    "comment": self._value(row, "comment", 5),
                }
            )
        return out


# ── the provider the graph actually uses ────────────────────────────────────


def load_bundled(config_dir: Path, name: str) -> dict:
    """The YAML that ships in the artifact — the fallback, and the seed."""
    path = Path(config_dir) / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class ConfigProvider:
    """Serves the current configuration to the graph, TTL-cached.

    Two caches, deliberately separate:

      * the *document* cache holds the parsed payload for `ttl_seconds`, which
        bounds how often a live endpoint queries Postgres for config;
      * the *object* cache holds the built consumer (`AgentRegistry` and
        friends) keyed on the document's checksum, so a TTL expiry that finds
        the config unchanged does not recompile every regex on the next
        request.

    Falls back to the bundled YAML whenever the table cannot be read or the
    row it returns is unusable, and says so at ERROR — this is the one moment
    where an operator's published config is silently not what is running, so it
    must not be silent.
    """

    def __init__(self, settings, store: Optional[ConfigStore] = None):
        self._settings = settings
        self._store = store
        self._store_resolved = store is not None
        self._docs: dict[str, tuple[float, ConfigDocument]] = {}
        self._objects: dict[str, tuple[str, Any]] = {}
        # Only complain once per document per failure mode; a 60-second TTL on
        # a broken table would otherwise write a log line a minute forever.
        self._reported: dict[str, str] = {}
        # The last document successfully read *from the table*, per name.
        #
        # This exists because "fall back to the bundled YAML" is not the safe
        # default it looks like. The bundled copy is a point-in-time snapshot,
        # and published config moves in both directions: someone tightens a
        # guardrail, or removes an agent from a role. If the table then becomes
        # briefly unreadable, reverting to the bundle silently **widens** the
        # policy — a guardrail that was tightened last week stops being applied,
        # and nothing in the request path says so.
        #
        # Blueprint §08 puts "audit or config store unavailable" in the
        # fail-closed column. Serving the last known-good *table* document is the
        # fail-closed reading that is also available: the policy in force cannot
        # get looser than the last thing an operator actually published. The
        # bundle stays the fallback for the one case where it is the only answer
        # — a process that has never read the table at all.
        self._last_good: dict[str, ConfigDocument] = {}

    # ── plumbing ────────────────────────────────────────────────────────────

    def _resolve_store(self) -> Optional[ConfigStore]:
        if self._store_resolved:
            return self._store
        self._store_resolved = True

        if self._settings.config_source == "files":
            logger.info("configuration source: bundled files (SUPERVISOR_CONFIG_SOURCE=files)")
            self._store = None
            return None

        from .memory import audit_connection_source

        # The same shared Postgres the audit sink writes to. Config reads are
        # TTL-cached, so this adds a few statements a minute per process.
        source = audit_connection_source()
        if source is None:
            if self._settings.config_source == "table":
                logger.error(
                    "SUPERVISOR_CONFIG_SOURCE=table but no Postgres is configured — "
                    "falling back to the bundled files"
                )
            else:
                logger.info(
                    "configuration source: bundled files (no Postgres configured)"
                )
            self._store = None
            return None

        self._store = ConfigStore(source, self._settings.config_table)
        logger.info("configuration source: table %s", self._settings.config_table)
        return self._store

    def _report(self, name: str, kind: str, message: str) -> None:
        if self._reported.get(name) == kind:
            return
        self._reported[name] = kind
        logger.error("configuration %s: %s — using the bundled file instead", name, message)

    def _bundled(self, name: str) -> ConfigDocument:
        payload = load_bundled(self._settings.config_dir, name)
        return ConfigDocument(name, payload, "files", None, checksum_of(payload))

    # ── documents ───────────────────────────────────────────────────────────

    def load(self, name: str) -> ConfigDocument:
        cached = self._docs.get(name)
        now = time.monotonic()
        if cached and cached[0] > now:
            return cached[1]

        document = self._load_uncached(name)
        self._docs[name] = (now + max(0.0, self._settings.config_cache_seconds), document)
        return document

    def _fallback(self, name: str) -> ConfigDocument:
        """The last known-good table document, or the bundled file.

        See `_last_good` for why the order is this way round and not the other.
        """
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
            # Validation or checksum failure: a row exists and is wrong. This is
            # the security-relevant branch — refuse it, and keep serving what was
            # last legitimately published rather than whatever the bundle
            # happens to say.
            self._report(name, "rejected", str(exc))
            return self._fallback(name)
        except Exception as exc:  # transport, permissions, table missing
            self._report(name, "unavailable", f"could not be read ({type(exc).__name__}: {exc})")
            return self._fallback(name)

        if document is None:
            self._report(name, "empty", "nothing published in the table yet")
            # No `_fallback` here: nothing published is not a failure to read,
            # and a document that was later deleted from the table should not be
            # resurrected from this process's memory. The bundle is the answer.
            return self._bundled(name)

        if self._reported.pop(name, None):
            logger.info("configuration %s: recovered — now serving %s", name, document.label)
        self._last_good[name] = document
        return document

    # ── built consumers ─────────────────────────────────────────────────────

    def _built(self, name: str, builder: Callable[[dict], Any], cache_key: str = "") -> Any:
        """Build (or reuse) the object form of a document.

        `cache_key` separates two consumers of the *same* document — the input
        engine and the output guard are both built from "guardrails" — so each
        keeps its own object without evicting the other's on every call.
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

    def registry(self):
        from .registry import AgentRegistry

        return self._built("agents", AgentRegistry.from_mapping)

    def rbac(self):
        from .rbac import RbacPolicy

        return self._built("rbac", RbacPolicy.from_mapping)

    def guardrails(
        self,
        llm,
        confidence_threshold: float,
        model_for=None,
        decisive_threshold: float = 0.9,
        contested_margin: float = 0.15,
    ):
        from .guardrails import GuardrailEngine

        return self._built(
            "guardrails",
            lambda payload: GuardrailEngine.from_mapping(
                llm,
                payload,
                confidence_threshold,
                model_for=model_for,
                decisive_threshold=decisive_threshold,
                contested_margin=contested_margin,
            ),
        )

    def output_guard(self, mask_pii: bool = True):
        """Layer-7 output screen, built from the same governed guardrails doc.

        Same document, separate object cache: a publish that changes
        `output_deny_patterns` reaches both the input engine and this guard on
        the next request, through the same `Reloading` proxy.
        """
        from .output_guard import OutputGuard
        from .prompt_provider import protected_lines

        # The governance prompts' distinctive lines are protected text: a
        # worker reply that reproduces one has reproduced the supervisor's
        # instructions. Read here, at build time, so a prompt-registry change
        # reaches the guard on the same reload as a policy change.
        return self._built(
            "guardrails",
            lambda payload: OutputGuard.from_mapping(
                payload, mask_pii=mask_pii, protected_texts=protected_lines()
            ),
            cache_key="output_guard",
        )

    def describe(self) -> dict[str, str]:
        """What is live right now, for `verify_deployment.py` and the tests."""
        return {name: self.load(name).label for name in CONFIG_NAMES}


class Reloading:
    """Delegates every attribute to whatever the provider currently serves.

    The graph holds one `Services` for the life of the process, so a config
    object captured at build time would pin the configuration to the moment of
    the last deploy — exactly what moving config into a table is meant to end.
    Attribute lookup goes through the provider instead, which is a TTL check
    and two dict lookups in the common case.

    Deliberately not a `__getattr__` on `Services` itself: nodes should not be
    able to tell the difference between a live-reloading registry and a fixed
    one, and the tests pass fixed ones.
    """

    __slots__ = ("_load",)

    def __init__(self, load: Callable[[], Any]):
        object.__setattr__(self, "_load", load)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Reloading({self._load()!r})"
