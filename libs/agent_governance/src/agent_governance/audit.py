"""The decision-trail sink every agent writes to.

The full decision trail of every request is written to an admin-only store,
while MLflow tracing captures the request trace itself.

Two sinks. `PostgresAuditLogger` is the one to prefer: it writes to the same
Postgres the checkpointer uses (on Lakebase, via a pooled connection with
rotating credentials), needs no additional entitlement, and works from inside
Model Serving. A SQL-warehouse sink is deliberately absent: the endpoint's
system service principal cannot be granted `databricks-sql-access`, so every
INSERT through a warehouse is refused. `LoggingAuditLogger` is the fallback —
never lost, but not queryable.

One hard-won detail about writing from more than one identity: see
`PostgresAuditLogger._ensure_table`. The serving endpoint runs as its own System
Service Principal, which is *not* the identity that created the table, and
owner-only DDL fails for it in a way that silently swallows the row.

**Every sink is also wrapped in `TracingAuditLogger`**, which mirrors the same
decision onto the request's own MLflow trace via `mlflow.update_current_trace()`,
so routing decisions, guardrail triggers and clarification loops all land in one
governed, queryable layer whichever durable sink is active.

This does **not** replace the durable sink below it. Sending traces straight to
Unity Catalog Delta tables (`mlflow.entities.UnityCatalog(...)`) needs an
external storage location; against Databricks-managed default storage it is
refused with `INVALID_PARAMETER_VALUE: ... Tables created in default storage are
not supported` — the same platform constraint that rules out the Delta audit
table below. Postgres is the durable sink of record.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from datetime import datetime, timezone

from .sql import safe_identifier, table_exists_here

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

# Columns added after the table shipped (§05 Stage 06 asks for latency; Stage 04
# for a session-duration measure; the Stage 04 callout for who approved what).
#
# `CREATE TABLE IF NOT EXISTS` does nothing for a table that already exists, so a
# deployment writing rows since before these three existed has a narrower table
# than the DDL above describes. Naming a missing column in the INSERT would fail
# every audit write — and `ALTER TABLE` is not the fix, because it takes the same
# ownership check that makes `CREATE INDEX` unusable here (see `_ensure_table`):
# the endpoint's service principal is not the identity that created the table,
# so the ALTER would raise, abort the transaction, and take the INSERT with it.
#
# So the schema is *detected*, never altered by the writer. `information_schema`
# is readable by anyone, the probe runs once per process, and the INSERT is built
# from what is actually there. A narrow table keeps working and says what it is
# dropping — at ERROR, because "the compliance record is quietly missing fields"
# is per-request data loss, not a configuration nuance. Widening it is a
# migration an owner runs — the ALTER TABLE block in the supervisor's DEPLOYMENT.md.
#
# `model_calls`, `tokens_estimated` and `provenance` joined them for the
# cost-control pass: the supervisor's own token KPI (v1.1 §08) and the prompt
# version / model endpoint that produced a decision previously existed only as
# MLflow spans, whose retention is shorter than this table's — the same reason
# `latency_ms` became a column rather than staying a span.
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

# Columns whose value is a dict and therefore serialised on the way in. Kept as
# a set rather than an `== "signoff"` check because there are now two of them,
# and a third that quietly missed the check would be inserted as a Python repr
# into a jsonb column and fail the write.
_JSON_COLUMNS = frozenset({"signoff", "provenance"})

# ── Tamper evidence (§10) ───────────────────────────────────────────────────
# The audit table is the record that proves controls operated, and a plain
# INSERT-only table proves nothing about later edits: any identity with write
# access to the instance could alter or delete history undetectably. (The
# *config* table got checksum protection; the actual compliance record did
# not.) So each row now carries a hash chain: `row_hash` digests the previous
# row's `row_hash` plus this row's own canonical content, anchored at
# `_GENESIS`. Editing or deleting any historical row breaks every digest after
# it, which a walk of the chain detects and reports.
#
# Chain writers serialize on a transaction-scoped advisory lock, so concurrent
# replicas cannot fork the chain. That serializes one short INSERT per turn —
# milliseconds against a turn that spends seconds in model calls; if it ever
# shows up at fleet scale, shard the chain by a bucket column and verify per
# bucket rather than dropping the lock.
#
# This makes tampering *evident*, not impossible: an attacker with write access
# could rewrite the whole suffix of the chain. Two things bound that: the
# runtime identity is append-only (the least-privilege grants in DEPLOYMENT.md
# revoke UPDATE/DELETE), and the chain head can
# be anchored externally by recording the latest `row_hash` somewhere outside
# operator write scope.
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

    Takes a *connection source* — a zero-arg callable yielding a
    context-managed connection — rather than one pinned connection. On
    Lakebase that borrows from a pool whose credentials rotate (~15 min); a
    connection held for the process lifetime would outlive its token. The
    table is created on first use so a fresh deployment does not need a
    separate migration step.
    """

    def __init__(self, connection_source, table: str):
        self._source = connection_source
        # SQL cannot parameterize an identifier, so the table name is
        # interpolated into the INSERT below. Validated here so that
        # interpolation rests on an enforced property rather than on the
        # configured name being trustworthy — see `sql.safe_identifier`.
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

        A read, not a write — see the note on `_OPTIONAL_COLUMNS`.

        Scoped to `current_schema()`, for the same reason `_ensure_table`'s
        probe is. Matching on the table name alone searches every schema on the
        path, so with `search_path = supervisor_dev, public` this could report
        the column set of the *other* environment's table — and the INSERT built
        from it would then be wrong for the table actually being written. It
        also matched tables in schemas that are not on the path at all, since
        `information_schema.columns` is not filtered by `search_path`.
        """
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s "
            "AND column_name = ANY(%s)",
            (self._table, list(_OPTIONAL_COLUMNS)),
        )
        rows = cur.fetchall() or []
        found = {
            (row["column_name"] if isinstance(row, dict) else row[0]) for row in rows
        }
        return tuple(name for name in _OPTIONAL_COLUMNS if name in found)

    def _ensure_table(self, conn) -> None:
        """Create the table on first write — but only if it is not already there.

        The existence check is not an optimisation. `CREATE TABLE IF NOT EXISTS`
        is happy for anyone, but `CREATE INDEX IF NOT EXISTS` takes an ownership
        check *before* the existence check, so it raises
        `InsufficientPrivilege: must be owner of table` for any identity that
        did not create the table — and because that aborts the transaction, the
        INSERT behind it is lost too.

        That is the normal case in a shared database, not an edge case: whoever
        runs first owns the table, and everyone after only holds INSERT. So the
        DDL runs when there is nothing there, and never again.
        """
        if self._ready:
            return
        with conn.cursor() as cur:
            # Scoped to the schema the DDL below would write to, not to the whole
            # search_path — see `sql.table_exists_here`. Getting this wrong
            # appends this environment's decision trail to another
            # environment's table.
            if not table_exists_here(cur, self._table):
                cur.execute(_PG_DDL.format(table=self._table))
                cur.execute(_PG_INDEX.format(table=self._table))
                self._columns = tuple(_OPTIONAL_COLUMNS)
            else:
                self._columns = self._detect_columns(cur)
                missing = [c for c in _OPTIONAL_COLUMNS if c not in self._columns]
                if missing:
                    # ERROR, not WARNING: every request from here on silently
                    # loses these fields (and, for the hash columns, its tamper
                    # evidence), which is a compliance gap an operator must
                    # act on, not an FYI.
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
                    json.dumps(value, default=str)
                    if name in _JSON_COLUMNS and value
                    else value
                )

            chained = all(name in self._columns for name in _HASH_COLUMNS)
            # The lock is transaction-scoped, so the chain read and the INSERT
            # must share one transaction. `conn.transaction()` works on psycopg
            # connections whatever their autocommit mode; a test double without
            # it simply writes unchained-serialization semantics, which is fine
            # off the deployed path.
            transaction = (
                conn.transaction()
                if chained and hasattr(conn, "transaction")
                else contextlib.nullcontext()
            )
            with transaction:
                if chained:
                    with conn.cursor() as cur:
                        cur.execute("SELECT pg_advisory_xact_lock(%s)", (self._chain_lock_key,))
                        cur.execute(
                            f"SELECT row_hash FROM {self._table} ORDER BY id DESC LIMIT 1"
                        )
                        row = cur.fetchone()
                    prev = ""
                    if row is not None:
                        prev = str(
                            (row.get("row_hash") if isinstance(row, dict) else row[0]) or ""
                        )
                    prev = prev or _GENESIS
                    # The digest is taken over *parsed* values, not the JSON
                    # strings the INSERT carries: jsonb does not preserve key
                    # order or whitespace, so a digest over the inserted string
                    # could never be re-verified from what the table returns.
                    # Canonical serialisation (sorted keys, compact, default=str)
                    # is applied inside `_row_digest`, identically at write and
                    # at verify time.
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
                        f"INSERT INTO {self._table} ({', '.join(columns)}) "
                        f"VALUES ({placeholders})",
                        tuple(values),
                    )


def _tag_trace(record: dict) -> None:
    """Mirror one decision onto this request's own active MLflow trace.

    Best-effort and silent by design: most call sites where an audit sink runs
    without a traced request in progress — an offline test, a local run
    without `mlflow.langchain.autolog()`, a sink invoked directly — have no
    active trace, and that is expected, not a failure worth logging above
    debug.
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
    """Wraps any audit sink so every write also tags the request's own MLflow
    trace with the same decision (`_tag_trace`) before writing to the durable
    sink underneath. Applied by `build_audit_logger()` to whichever sink it
    picks, so the trace is enriched the same way whether the durable copy
    ends up in Postgres, Delta, or only the process log.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def log(self, record: dict) -> None:
        _tag_trace(record)
        self._inner.log(record)


def build_audit_logger(connection_source, table: str):
    """The Postgres sink over `connection_source`, or the process-log fallback.

    `connection_source` is a zero-arg callable yielding a context-managed
    connection (see `lakebase.audit_connection_source`), or None when no
    Postgres is configured. Either sink is wrapped in `TracingAuditLogger`.
    """
    if connection_source is not None:
        logger.info("audit sink: Postgres table %s", table)
        return TracingAuditLogger(PostgresAuditLogger(connection_source, table))

    logger.warning(
        "audit sink: process log only — the decision trail will not be queryable. "
        "Configure Lakebase (LAKEBASE_INSTANCE)."
    )
    return TracingAuditLogger(LoggingAuditLogger())
