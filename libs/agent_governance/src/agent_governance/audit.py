"""The decision-trail sink every agent writes to — the record a compliance query runs against.

Entry points: `build_audit_logger` picks `PostgresAuditLogger` (same Postgres as the checkpointer,
no extra entitlement) or the process-log `LoggingAuditLogger` fallback, and wraps either in
`TracingAuditLogger` so each decision also lands on the request's MLflow trace. `verify_chain` /
`ChainVerification` walk the tamper-evidence hash chain. No SQL-warehouse sink: the endpoint's
service principal cannot hold `databricks-sql-access`. Trace destination is a deployment choice
(`deploy/log_and_deploy.py --trace-catalog-schema`).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from .lakebase import safe_identifier, table_exists_here

logger = logging.getLogger(__name__)

_PG_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
  id                  BIGSERIAL PRIMARY KEY,
  event_time          TIMESTAMPTZ NOT NULL DEFAULT now(),
  request_id          TEXT,
  correlation_id      TEXT,
  conversation_id     TEXT,
  user_role           TEXT,
  user_key            TEXT,
  target_agent_id     TEXT,
  outcome             TEXT,
  decision_trail      JSONB,
  latency_ms          INTEGER,
  session_age_seconds DOUBLE PRECISION,
  signoff             JSONB,
  model_calls         INTEGER,
  tokens_estimated    INTEGER,
  provenance          JSONB,
  prev_hash           TEXT,
  row_hash            TEXT
)
"""

# Columns added after the table shipped (§05 Stage 06 latency; Stage 04 session age and signoff;
# cost fields). *Detected* via `information_schema`, never ALTERed: `ALTER TABLE` takes the same
# ownership check as `CREATE INDEX` (see `_ensure_table`) and would abort the INSERT with it.
_OPTIONAL_COLUMNS = (
    "latency_ms",
    "session_age_seconds",
    "signoff",
    "model_calls",
    "tokens_estimated",
    "provenance",
    "prev_hash",
    "row_hash",
)

# Dict-valued columns, serialised on the way in. A set, so a third one added later cannot slip
# past an equality check and be inserted as a Python repr into a jsonb column.
_JSON_COLUMNS = frozenset({"signoff", "provenance"})

# Digest input columns, kept visibly parallel to what `log` appends (order is immaterial): a column
# added to the writer and forgotten here would be reported as tampering by `verify_chain`.
_CHAIN_COLUMNS = (
    "event_time",
    "request_id",
    "correlation_id",
    "conversation_id",
    "user_role",
    "user_key",
    "target_agent_id",
    "outcome",
    "decision_trail",
    *(c for c in _OPTIONAL_COLUMNS if c not in ("prev_hash", "row_hash")),
)

# ── Tamper evidence (§10) ───────────────────────────────────────────────────
# `row_hash` = digest(previous `row_hash` + canonical row content), anchored at `_GENESIS`: editing
# or deleting history breaks every later digest (`verify_chain`). Writers serialise on an advisory
# lock so replicas cannot fork. Evident, not impossible: the runtime identity is append-only.
_GENESIS = "0" * 64
_HASH_COLUMNS = ("prev_hash", "row_hash")


def _row_digest(prev_hash: str, payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


# Usage reporting is by user, agent, environment and run (§2.13), so those are
# the columns worth an index.
_PG_INDEX = """
CREATE INDEX IF NOT EXISTS {table}_lookup_idx
  ON {table} (user_key, target_agent_id, event_time DESC)
"""


class LoggingAuditLogger:
    """Fallback — writes the decision trail to the process log.

    Never lost, but not queryable, so it does not satisfy BR-006 on its own.
    """

    def log(self, record: dict) -> None:
        logger.info("supervisor-audit %s", json.dumps(record, default=str))


class PostgresAuditLogger:
    """Appends one row per request to a Postgres table.

    Takes a zero-arg connection source, not a pinned connection: Lakebase pool credentials rotate
    every ~15 minutes. The table is created on first use, so no separate migration step.
    """

    def __init__(self, connection_source, table: str):
        self._source = connection_source
        # SQL cannot parameterize an identifier, so the name is interpolated below; validated
        # here so interpolation rests on an enforced property (`lakebase.safe_identifier`).
        self._table = safe_identifier(table)
        self._ready = False
        # Which of `_OPTIONAL_COLUMNS` this table actually has. Populated by
        # `_ensure_table`; empty until then.
        self._columns: tuple[str, ...] = ()
        # The advisory-lock key chain writers serialize on — per table, so two
        # audit tables on one instance do not contend with each other.
        digest = hashlib.sha256(f"supervisor_audit_chain:{self._table}".encode()).digest()
        self._chain_lock_key = int.from_bytes(digest[:8], "big", signed=True)

    def _detect_columns(self, cur) -> tuple[str, ...]:
        """Which optional columns exist, read from the catalogue.

        Scoped to `current_schema()`: matching on name alone searches every schema on the path
        and could report the *other* environment's column set (see `_ensure_table`).
        """
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s "
            "AND column_name = ANY(%s)",
            (self._table, list(_OPTIONAL_COLUMNS)),
        )
        rows = cur.fetchall() or []
        found = {(row["column_name"] if isinstance(row, dict) else row[0]) for row in rows}
        return tuple(name for name in _OPTIONAL_COLUMNS if name in found)

    def _ensure_table(self, conn) -> None:
        """Create the table on first write — only if it is not already there.

        Not an optimisation: `CREATE INDEX IF NOT EXISTS` checks ownership *before* existence, so
        it raises for any identity that did not create the table and aborts the INSERT with it.
        """
        if self._ready:
            return
        with conn.cursor() as cur:
            # Scoped to the schema the DDL would write to, not the whole search_path — getting
            # this wrong appends this environment's trail to another environment's table.
            if not table_exists_here(cur, self._table):
                cur.execute(_PG_DDL.format(table=self._table))
                cur.execute(_PG_INDEX.format(table=self._table))
                self._columns = tuple(_OPTIONAL_COLUMNS)
            else:
                self._columns = self._detect_columns(cur)
                missing = [c for c in _OPTIONAL_COLUMNS if c not in self._columns]
                if missing:
                    # ERROR, not WARNING: every request from here silently loses these fields
                    # (and, for the hash columns, its tamper evidence) — a compliance gap.
                    logger.error(
                        "audit table %s predates %s — those fields will not be "
                        "persisted until the audit-table migration in DEPLOYMENT.md "
                        "is run as the table's owner.",
                        self._table,
                        ", ".join(missing),
                    )
        self._ready = True

    def log(self, record: dict) -> None:
        with self._source() as conn:
            self._ensure_table(conn)

            columns = [
                "event_time",
                "request_id",
                "correlation_id",
                "conversation_id",
                "user_role",
                "user_key",
                "target_agent_id",
                "outcome",
                "decision_trail",
            ]
            values: list = [
                datetime.now(timezone.utc),
                str(record.get("request_id", "")),
                str(record.get("correlation_id", "")),
                str(record.get("conversation_id", "")),
                str(record.get("user_role", "")),
                str(record.get("user_key", "")),
                str(record.get("target_agent_id", "")),
                str(record.get("outcome", "")),
                json.dumps(record.get("decision_trail", []), default=str),
            ]

            # Only the columns this table actually has — see `_OPTIONAL_COLUMNS`.
            # The hash pair is computed below, never taken from the record.
            for name in self._columns:
                if name in _HASH_COLUMNS:
                    continue
                value = record.get(name)
                columns.append(name)
                values.append(
                    json.dumps(value, default=str) if name in _JSON_COLUMNS and value else value
                )

            chained = all(name in self._columns for name in _HASH_COLUMNS)
            # The lock is transaction-scoped, so the chain read and the INSERT must share one
            # transaction; `conn.transaction()` works whatever the autocommit mode.
            transaction = (
                conn.transaction()
                if chained and hasattr(conn, "transaction")
                else contextlib.nullcontext()
            )
            with transaction:
                if chained:
                    with conn.cursor() as cur:
                        cur.execute("SELECT pg_advisory_xact_lock(%s)", (self._chain_lock_key,))
                        cur.execute(f"SELECT row_hash FROM {self._table} ORDER BY id DESC LIMIT 1")
                        row = cur.fetchone()
                    prev = ""
                    if row is not None:
                        prev = str((row.get("row_hash") if isinstance(row, dict) else row[0]) or "")
                    prev = prev or _GENESIS
                    # Digest over *parsed* values, not the JSON strings: jsonb preserves neither
                    # key order nor whitespace, so a string digest could never be re-verified.
                    payload = dict(zip(columns, values, strict=True))
                    payload["decision_trail"] = record.get("decision_trail", [])
                    for name in _JSON_COLUMNS:
                        if name in payload:
                            payload[name] = record.get(name)
                    row_hash = _row_digest(prev, payload)
                    columns.extend(_HASH_COLUMNS)
                    values.extend([prev, row_hash])

                placeholders = ", ".join(["%s"] * len(columns))
                with conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES ({placeholders})",
                        tuple(values),
                    )


def _tag_trace(record: dict) -> None:
    """Mirror one decision onto this request's active MLflow trace.

    Best-effort and silent by design: offline tests, local runs and direct sink calls have no
    active trace, and that is expected.
    """
    try:
        import json

        import mlflow

        mlflow.update_current_trace(
            client_request_id=str(record.get("request_id") or "") or None,
            session_id=str(record.get("conversation_id") or "") or None,
            user=str(record.get("user_key") or "") or None,
            tags={
                "outcome": str(record.get("outcome", "")),
                "target_agent_id": str(record.get("target_agent_id", "")),
                "user_role": str(record.get("user_role", "")),
            },
            metadata={"decision_trail": json.dumps(record.get("decision_trail", []), default=str)},
        )
    except Exception:
        logger.debug("trace tagging skipped — no active MLflow trace", exc_info=True)


class TracingAuditLogger:
    """Wraps any audit sink so every write also tags the request's MLflow trace (`_tag_trace`).

    Applied by `build_audit_logger()` to whichever durable sink it picks.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def log(self, record: dict) -> None:
        _tag_trace(record)
        self._inner.log(record)


def build_audit_logger(connection_source, table: str):
    """The Postgres sink over `connection_source`, or the process-log fallback.

    `connection_source` is a zero-arg callable yielding a context-managed connection (see
    `lakebase.audit_connection_source`), or None. Either sink is wrapped in `TracingAuditLogger`.
    """
    if connection_source is not None:
        logger.info("audit sink: Postgres table %s", table)
        return TracingAuditLogger(PostgresAuditLogger(connection_source, table))

    logger.warning(
        "audit sink: process log only — the decision trail will not be queryable. "
        "Configure Lakebase (LAKEBASE_INSTANCE)."
    )
    return TracingAuditLogger(LoggingAuditLogger())


# ── chain verification ──────────────────────────────────────────────────────
# In the library, not an operator script: a control that lives only in a script is one `rm` away
# from an unverifiable claim. Run on a schedule and anchor `head_id`/`head_hash` outside operator
# write scope; a clean result then proves no row since the anchor was altered or removed.


@dataclass(frozen=True)
class ChainVerification:
    """What a walk of the audit chain found."""

    intact: bool
    checked: int
    head_id: int = 0
    head_hash: str = ""
    # The id of the first row whose digest did not match, or 0 when intact.
    broken_at: int = 0
    detail: str = ""

    def audit_detail(self) -> str:
        if self.intact:
            return f"audit chain intact: {self.checked} row(s) to head {self.head_id}"
        return f"audit chain BROKEN at row {self.broken_at}: {self.detail}"


def _verifiable_payload(row: dict, columns: Iterable[str]) -> dict:
    """Rebuild the digest input for one row exactly as `log` built it.

    Timestamps are normalised to UTC (`log` hashed `datetime.now(timezone.utc)`; Postgres returns
    the session's zone). JSON columns pass through parsed, which is what `log` hashed.
    """
    payload: dict = {}
    for name in columns:
        if name in _HASH_COLUMNS:
            continue
        value = row.get(name)
        if isinstance(value, datetime):
            value = value.astimezone(timezone.utc)
        payload[name] = value
    return payload


def verify_chain(connection_source, table: str, after_id: int = 0) -> ChainVerification:
    """Walk the chain from `after_id` and report whether history is intact.

    `after_id` continues from an anchored head. A broken chain is a *finding*, returned; a
    connection failure propagates — "could not check" must never read as "intact".
    """
    name = safe_identifier(table)
    with connection_source() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = %s",
                (name,),
            )
            present = {(r["column_name"] if isinstance(r, dict) else r[0]) for r in cur.fetchall()}
            if not {"prev_hash", "row_hash"} <= present:
                return ChainVerification(
                    intact=True,
                    checked=0,
                    detail="table predates the hash chain — nothing to verify",
                )
            columns = [c for c in _CHAIN_COLUMNS if c in present]
            selected = ", ".join(["id", *columns, *_HASH_COLUMNS])
            cur.execute(
                f"SELECT {selected} FROM {name} WHERE id > %s ORDER BY id ASC",  # noqa: S608
                (int(after_id),),
            )
            rows = cur.fetchall() or []

    prev = ""
    checked = 0
    head_id = 0
    head_hash = ""
    for raw in rows:
        row = (
            raw
            if isinstance(raw, dict)
            else dict(zip(["id", *columns, *_HASH_COLUMNS], raw, strict=True))
        )
        row_id = int(row.get("id") or 0)
        stored_prev = str(row.get("prev_hash") or "")
        stored_hash = str(row.get("row_hash") or "")
        # The first row of a partial walk has no in-scope predecessor, so its `prev_hash` is
        # taken on trust — that is what anchoring the head elsewhere is for.
        if prev and stored_prev != prev:
            return ChainVerification(
                intact=False,
                checked=checked,
                head_id=head_id,
                head_hash=head_hash,
                broken_at=row_id,
                detail=f"prev_hash {stored_prev[:12]}… does not follow {prev[:12]}…",
            )
        expected = _row_digest(stored_prev or _GENESIS, _verifiable_payload(row, columns))
        if expected != stored_hash:
            return ChainVerification(
                intact=False,
                checked=checked,
                head_id=head_id,
                head_hash=head_hash,
                broken_at=row_id,
                detail="row content does not match its recorded digest",
            )
        prev, checked, head_id, head_hash = stored_hash, checked + 1, row_id, stored_hash

    return ChainVerification(intact=True, checked=checked, head_id=head_id, head_hash=head_hash)
