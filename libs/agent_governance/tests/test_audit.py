"""The evidence trail: what is recorded, that it cannot be altered undetectably, that it
can still be erased on request, and that the logs around it correlate.

The sink is the record; the hash chain is what makes it evidence; retention is the one
lawful way a row leaves it; the structured log is how a turn is found in the first place.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import timedelta, timezone

import pytest
from agent_governance import observability as obs
from agent_governance import retention
from agent_governance.audit import (
    LoggingAuditLogger,
    PostgresAuditLogger,
    TracingAuditLogger,
    verify_chain,
)

RECORD = {
    "request_id": "req-1",
    "correlation_id": "corr-1",
    "conversation_id": "thr-1",
    "user_role": "BA",
    "user_key": "usr_1",
    "target_agent_id": "requirement-agent",
    "outcome": "answer",
    "decision_trail": [{"stage": "rbac_gate", "decision": "allow", "detail": "x"}],
}


class FakeCursor:
    def __init__(self, conn, table_present: bool):
        self._conn = conn
        self._present = table_present
        self._last = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._conn.statements.append(" ".join(sql.split()))
        self._last = sql
        if "must be owner" in self._conn.fail_on and "CREATE INDEX" in sql:
            raise RuntimeError("InsufficientPrivilege: must be owner of table")

    def fetchone(self):
        if "to_regclass" in self._last:
            return {"present": self._present}
        return None

    def fetchall(self):
        # The column probe (`information_schema.columns`). An existing table is modelled
        # as having the optional columns; the narrow-table case gets its own test.
        if "information_schema.columns" in self._last:
            return [
                {"column_name": name} for name in ("latency_ms", "session_age_seconds", "signoff")
            ]
        return []


class FakeConnection:
    def __init__(self, table_present: bool = False, fail_on: str = ""):
        self.statements: list[str] = []
        self.table_present = table_present
        self.fail_on = fail_on

    def cursor(self):
        return FakeCursor(self, self.table_present)


def source_for(conn):
    import contextlib

    def _source():
        return contextlib.nullcontext(conn)

    return _source


def test_the_table_is_created_when_it_does_not_exist():
    conn = FakeConnection(table_present=False)
    PostgresAuditLogger(source_for(conn), "supervisor_audit_log").log(RECORD)

    joined = " | ".join(conn.statements)
    assert "CREATE TABLE IF NOT EXISTS" in joined
    assert "CREATE INDEX IF NOT EXISTS" in joined
    assert "INSERT INTO supervisor_audit_log" in joined


def test_no_ddl_runs_when_the_table_is_already_there():
    """The fix for `must be owner of table`: `CREATE INDEX IF NOT EXISTS` checks ownership
    before existence, so issuing it unconditionally aborted the transaction with the
    INSERT — every request looked healthy while the table stayed empty."""
    conn = FakeConnection(table_present=True)
    PostgresAuditLogger(source_for(conn), "supervisor_audit_log").log(RECORD)

    joined = " | ".join(conn.statements)
    assert "CREATE TABLE" not in joined
    assert "CREATE INDEX" not in joined
    assert "INSERT INTO supervisor_audit_log" in joined


def test_a_non_owner_can_still_write_to_an_existing_table():
    """End to end on the failure the endpoint actually hit."""
    conn = FakeConnection(table_present=True, fail_on="must be owner")
    PostgresAuditLogger(source_for(conn), "supervisor_audit_log").log(RECORD)
    assert any(s.startswith("INSERT INTO") for s in conn.statements)


def test_the_ddl_check_happens_once_per_process():
    conn = FakeConnection(table_present=False)
    sink = PostgresAuditLogger(source_for(conn), "supervisor_audit_log")
    sink.log(RECORD)
    sink.log(RECORD)

    assert sum("to_regclass" in s for s in conn.statements) == 1
    assert sum(s.startswith("INSERT INTO") for s in conn.statements) == 2


def test_the_trail_is_serialised_as_json():
    conn = FakeConnection(table_present=True)
    PostgresAuditLogger(source_for(conn), "supervisor_audit_log").log(RECORD)
    assert any("decision_trail" in s for s in conn.statements)


def test_the_logging_sink_never_raises(caplog):
    """The fallback must be safe: `respond` swallows sink failures, so a sink
    that throws would hide the trail without anyone noticing."""
    with caplog.at_level("INFO"):
        LoggingAuditLogger().log(RECORD)
    assert "supervisor-audit" in caplog.text
    assert "requirement-agent" in caplog.text


@pytest.mark.parametrize("missing", ["decision_trail", "outcome", "user_role"])
def test_a_partial_record_still_writes(missing):
    record = {k: v for k, v in RECORD.items() if k != missing}
    conn = FakeConnection(table_present=True)
    PostgresAuditLogger(source_for(conn), "supervisor_audit_log").log(record)
    assert any(s.startswith("INSERT INTO") for s in conn.statements)


class FakeInnerLogger:
    def __init__(self):
        self.records: list[dict] = []

    def log(self, record: dict) -> None:
        self.records.append(record)


def test_tracing_wrapper_still_writes_the_durable_sink_with_no_active_trace():
    """Solution v1.2 §08 wants the decision on the trace *in addition to* the durable sink;
    most call sites have no active MLflow trace, which must never block the durable write."""
    inner = FakeInnerLogger()
    TracingAuditLogger(inner).log(RECORD)
    assert inner.records == [RECORD]


def test_tracing_wrapper_tags_a_genuinely_active_trace(tmp_path, monkeypatch):
    # `tracing` is an optional extra of the wheel; the library's own suite must
    # run against a minimal install, so this test skips rather than errors.
    mlflow = pytest.importorskip("mlflow")

    # Reading a trace back in the same process that wrote it needs the
    # synchronous exporter: the async one returns before the trace is queryable.
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'traces.db').as_posix()}")
    mlflow.set_experiment("test-tracing-audit-logger")

    inner = FakeInnerLogger()
    with mlflow.start_span(name="audit-tag-test"):
        TracingAuditLogger(inner).log(RECORD)
        trace_id = mlflow.get_current_active_span().request_id

    trace = mlflow.get_trace(trace_id)
    assert trace.info.tags.get("outcome") == "answer"
    assert trace.info.tags.get("target_agent_id") == "requirement-agent"
    assert trace.info.client_request_id == "req-1"
    assert inner.records == [RECORD]


# ═══ The hash chain ══════════════════════════════════════════════════════════════
#
# `PostgresAuditLogger` computes `row_hash` at write time and `verify_chain` recomputes it
# ~250 lines away; the two must agree over the same columns, parsed JSON and timestamp
# normalisation, so everything here round-trips the real writer into a faithful fake store.

ALL_COLUMNS = (
    "id",
    "event_time",
    "request_id",
    "correlation_id",
    "conversation_id",
    "user_role",
    "user_key",
    "target_agent_id",
    "outcome",
    "decision_trail",
    "latency_ms",
    "session_age_seconds",
    "signoff",
    "model_calls",
    "tokens_estimated",
    "provenance",
    "prev_hash",
    "row_hash",
)
_JSON = ("decision_trail", "signoff", "provenance")


def record(n: int) -> dict:
    return {
        "request_id": f"req-{n}",
        "correlation_id": f"corr-{n}",
        "conversation_id": "thr-1",
        "user_role": "BA",
        "user_key": "usr_1",
        "target_agent_id": "requirement-agent",
        "outcome": "answer",
        "decision_trail": [{"stage": "rbac_gate", "decision": "allow", "detail": f"x{n}"}],
        "latency_ms": 100 + n,
        "session_age_seconds": 1.5 * n,
        "model_calls": n,
        "tokens_estimated": 10 * n,
    }


class FakeStore:
    """A tiny stand-in for the audit table, shared by writer and verifier."""

    def __init__(self, session_offset_hours: int = 0):
        self.rows: list[dict] = []
        # Postgres hands a timestamptz back in the *session's* zone. Non-UTC by
        # default here, because a verifier that only works on a UTC session is
        # a verifier that reports tampering on somebody's laptop.
        self.tz = timezone(timedelta(hours=session_offset_hours))


class ChainCursor:
    def __init__(self, store: FakeStore):
        self.s = store
        self._last = ""
        self._params: tuple = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._last = " ".join(sql.split())
        self._params = params or ()
        if self._last.startswith("INSERT INTO"):
            columns = self._last.split("(", 1)[1].split(")", 1)[0]
            names = [c.strip() for c in columns.split(",")]
            row = dict(zip(names, params, strict=True))
            for key in _JSON:
                # jsonb: stored parsed, returned parsed — never the string.
                if isinstance(row.get(key), str):
                    import json

                    row[key] = json.loads(row[key])
            row["id"] = len(self.s.rows) + 1
            self.s.rows.append(row)

    def fetchone(self):
        if "to_regclass" in self._last:
            return {"present": bool(self.s.rows)}
        if "ORDER BY id DESC LIMIT 1" in self._last:
            return {"row_hash": self.s.rows[-1]["row_hash"]} if self.s.rows else None
        return None

    def fetchall(self):
        if "information_schema.columns" in self._last:
            return [{"column_name": c} for c in ALL_COLUMNS if c != "id"]
        if self._last.startswith("SELECT id,"):
            after = int(self._params[0])
            out = []
            for row in self.s.rows:
                if row["id"] <= after:
                    continue
                copy = dict(row)
                stamp = copy.get("event_time")
                if stamp is not None:
                    copy["event_time"] = stamp.astimezone(self.s.tz)
                out.append(copy)
            return out
        return []


class ChainConnection:
    def __init__(self, store: FakeStore):
        self.s = store

    def cursor(self):
        return ChainCursor(self.s)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def chain_source(store: FakeStore):
    def _source():
        return ChainConnection(store)

    return _source


def write(store: FakeStore, count: int) -> None:
    sink = PostgresAuditLogger(chain_source(store), "supervisor_audit_log")
    for n in range(1, count + 1):
        sink.log(record(n))


# ── the round trip ──────────────────────────────────────────────────────────


def test_a_chain_the_writer_produced_verifies_clean():
    store = FakeStore()
    write(store, 5)
    result = verify_chain(chain_source(store), "supervisor_audit_log")
    assert result.intact, result.detail
    assert result.checked == 5
    assert result.head_id == 5
    assert result.head_hash == store.rows[-1]["row_hash"]


@pytest.mark.parametrize("offset", [0, 5, -8])
def test_verification_does_not_depend_on_the_session_timezone(offset):
    """The writer hashes UTC; Postgres returns the session's zone."""
    store = FakeStore(session_offset_hours=offset)
    write(store, 3)
    assert verify_chain(chain_source(store), "supervisor_audit_log").intact


def test_a_partial_walk_from_an_anchor_checks_only_the_rows_since():
    store = FakeStore()
    write(store, 6)
    result = verify_chain(chain_source(store), "supervisor_audit_log", after_id=4)
    assert result.intact
    assert result.checked == 2
    assert result.head_id == 6


# ── what tampering looks like ───────────────────────────────────────────────


def test_editing_a_row_breaks_its_own_digest():
    store = FakeStore()
    write(store, 4)
    store.rows[1]["outcome"] = "blocked"
    result = verify_chain(chain_source(store), "supervisor_audit_log")
    assert not result.intact
    assert result.broken_at == 2
    assert "digest" in result.detail


def test_editing_a_nested_json_field_breaks_the_digest():
    """The decision trail is the part worth editing, so it must be covered."""
    store = FakeStore()
    write(store, 3)
    store.rows[2]["decision_trail"][0]["decision"] = "deny"
    result = verify_chain(chain_source(store), "supervisor_audit_log").intact
    assert not result


def test_deleting_a_row_breaks_the_link():
    store = FakeStore()
    write(store, 5)
    del store.rows[2]
    result = verify_chain(chain_source(store), "supervisor_audit_log")
    assert not result.intact
    assert "does not follow" in result.detail


def test_an_intact_prefix_is_reported_before_the_break():
    store = FakeStore()
    write(store, 5)
    store.rows[3]["user_role"] = "Admin"
    result = verify_chain(chain_source(store), "supervisor_audit_log")
    assert not result.intact
    assert result.checked == 3
    assert result.broken_at == 4


def test_audit_detail_reads_as_a_finding():
    store = FakeStore()
    write(store, 2)
    assert "intact" in verify_chain(chain_source(store), "supervisor_audit_log").audit_detail()
    store.rows[0]["outcome"] = "tampered"
    assert "BROKEN" in verify_chain(chain_source(store), "supervisor_audit_log").audit_detail()


# ═══ Erasure and retention ═══════════════════════════════════════════════════════
#
# What separates this from a hand-run `DELETE`: a dry run unless told otherwise, verified
# before it reports success, no partial erasure called complete, and recorded through the
# audit logger rather than around it.


class EraseCursor:
    def __init__(self, db):
        self.db = db
        self.rowcount = 0
        self._result = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        text = " ".join(sql.split())
        self.db.statements.append(text)
        self._result = []
        self.rowcount = 0

        if text.startswith("SELECT count(*) FROM store"):
            prefix, key = params
            self._result = [{"count": len(self.db.store.get((prefix, key), []))}]
        elif text.startswith("SELECT count(*) FROM checkpoint"):
            table = text.split()[3]
            threads = set(params[0])
            self._result = [
                {"count": sum(1 for t in self.db.checkpoints.get(table, []) if t in threads)}
            ]
        elif "SELECT DISTINCT conversation_id" in text:
            (key,) = params
            self._result = [{"conversation_id": c} for c in self.db.subjects.get(key, [])]
        elif "HAVING max(event_time)" in text:
            self._result = [{"conversation_id": c} for c in self.db.stale]
        elif text.startswith("DELETE FROM store"):
            prefix, key = params
            self.rowcount = len(self.db.store.pop((prefix, key), []))
        elif text.startswith("DELETE FROM checkpoint"):
            table = text.split()[2]
            threads = set(params[0])
            if table in self.db.undeletable:
                self.rowcount = 0
                return
            kept = [t for t in self.db.checkpoints.get(table, []) if t not in threads]
            self.rowcount = len(self.db.checkpoints.get(table, [])) - len(kept)
            self.db.checkpoints[table] = kept

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return self._result


class FakeDB:
    def __init__(self, undeletable=()):
        self.store = {(retention.STORE_PREFIX, "usr_1"): ["row"]}
        self.checkpoints = {t: ["thr-1", "thr-2", "thr-9"] for t in retention.CHECKPOINT_TABLES}
        self.subjects = {"usr_1": ["thr-1", "thr-2"]}
        self.stale = ["thr-9"]
        self.statements: list[str] = []
        self.undeletable = set(undeletable)

    def cursor(self):
        return EraseCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def erase_source(db):
    def _source():
        return db

    return _source


class RecordingAudit:
    def __init__(self, fail=False):
        self.records = []
        self.fail = fail

    def log(self, record):
        if self.fail:
            raise RuntimeError("sink down")
        self.records.append(record)


# ── dry run is the default ──────────────────────────────────────────────────


def test_erasure_defaults_to_a_dry_run():
    db = FakeDB()
    result = retention.erase_subject(erase_source(db), "usr_1", "supervisor_audit_log")
    assert not result.applied
    assert result.memory_rows == 1
    assert result.conversations == ("thr-1", "thr-2")
    assert not any(s.startswith("DELETE") for s in db.statements)


def test_a_sweep_defaults_to_a_dry_run():
    db = FakeDB()
    result = retention.sweep_conversations(erase_source(db), "supervisor_audit_log", 180)
    assert not result.applied
    assert result.conversations == ("thr-9",)
    assert not any(s.startswith("DELETE") for s in db.statements)


# ── applying ────────────────────────────────────────────────────────────────


def test_erasure_removes_memory_and_every_checkpoint_table():
    db = FakeDB()
    audit = RecordingAudit()
    result = retention.erase_subject(
        erase_source(db), "usr_1", "supervisor_audit_log", apply=True, audit_logger=audit
    )
    assert result.clean
    assert result.memory_rows == 1
    # one row per thread per table
    assert result.checkpoint_rows == 2 * len(retention.CHECKPOINT_TABLES)
    for table in retention.CHECKPOINT_TABLES:
        assert db.checkpoints[table] == ["thr-9"]


def test_memory_only_leaves_conversations_alone():
    db = FakeDB()
    result = retention.erase_subject(
        erase_source(db), "usr_1", "supervisor_audit_log", apply=True, memory_only=True
    )
    assert result.conversations == ()
    assert db.checkpoints["checkpoints"] == ["thr-1", "thr-2", "thr-9"]


def test_a_sweep_only_touches_idle_conversations():
    db = FakeDB()
    result = retention.sweep_conversations(
        erase_source(db), "supervisor_audit_log", 180, apply=True
    )
    assert result.conversations == ("thr-9",)
    assert db.checkpoints["checkpoints"] == ["thr-1", "thr-2"]


# ── it must not report an erasure it did not perform ────────────────────────


def test_residual_data_raises_rather_than_reporting_success():
    db = FakeDB(undeletable={"checkpoint_blobs"})
    with pytest.raises(RuntimeError, match="erasure incomplete"):
        retention.erase_subject(erase_source(db), "usr_1", "supervisor_audit_log", apply=True)


def test_the_failed_erasure_is_still_recorded():
    """An incomplete erasure is exactly the one an auditor needs to see."""
    db = FakeDB(undeletable={"checkpoint_blobs"})
    audit = RecordingAudit()
    with pytest.raises(RuntimeError):
        retention.erase_subject(
            erase_source(db), "usr_1", "supervisor_audit_log", apply=True, audit_logger=audit
        )
    assert audit.records
    assert "RESIDUAL" in audit.records[0]["decision_trail"][0]["detail"]


# ── evidence ────────────────────────────────────────────────────────────────


def test_an_erasure_is_written_through_the_audit_logger():
    """Through `log()`, so the hash chain stays valid — never a raw INSERT."""
    audit = RecordingAudit()
    retention.erase_subject(
        erase_source(FakeDB()),
        "usr_1",
        "supervisor_audit_log",
        apply=True,
        audit_logger=audit,
        requested_by="alice@corp",
    )
    record = audit.records[0]
    assert record["outcome"] == "erasure"
    assert record["user_key"] == "usr_1"
    assert record["user_role"] == "alice@corp"


def test_a_sweep_is_recorded_too():
    audit = RecordingAudit()
    retention.sweep_conversations(
        erase_source(FakeDB()), "supervisor_audit_log", 180, apply=True, audit_logger=audit
    )
    assert audit.records[0]["outcome"] == "retention"


def test_a_failing_audit_sink_does_not_lose_the_erasure(caplog):
    """The rows are already gone; the operator must be told loudly."""
    with caplog.at_level("ERROR"):
        result = retention.erase_subject(
            erase_source(FakeDB()),
            "usr_1",
            "supervisor_audit_log",
            apply=True,
            audit_logger=RecordingAudit(fail=True),
        )
    assert result.clean
    assert any("NOT recorded" in r.message for r in caplog.records)


def test_audit_rows_are_never_deleted():
    """They carry no message text and deleting them breaks the chain."""
    db = FakeDB()
    retention.erase_subject(erase_source(db), "usr_1", "supervisor_audit_log", apply=True)
    assert not any("DELETE FROM supervisor_audit_log" in s for s in db.statements)


# ── argument guards ─────────────────────────────────────────────────────────


def test_an_empty_user_key_is_refused():
    with pytest.raises(ValueError, match="user_key is required"):
        retention.erase_subject(erase_source(FakeDB()), "", "supervisor_audit_log")


@pytest.mark.parametrize("days", [0, -1])
def test_a_nonsensical_retention_window_is_refused(days):
    with pytest.raises(ValueError, match="must be positive"):
        retention.sweep_conversations(erase_source(FakeDB()), "supervisor_audit_log", days)


def test_an_unsafe_table_name_cannot_reach_sql():
    from agent_governance.lakebase import UnsafeIdentifier

    with pytest.raises(UnsafeIdentifier):
        retention.sweep_conversations(erase_source(FakeDB()), "audit; DROP TABLE x", 30)


# ═══ Structured logging and correlation ══════════════════════════════════════════
#
# Two properties decide whether this helps in an incident: every line inside a turn
# carries that turn's ids unaided, and the formatter never loses a log line.


@pytest.fixture
def capture():
    """A root logger writing JSON into a buffer, restored afterwards."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(obs.JsonFormatter())
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    obs.clear()
    try:
        yield buffer
    finally:
        root.handlers, root.level = saved_handlers, saved_level
        obs.clear()


def lines(buffer) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().strip().splitlines() if line]


# ── the shape of a line ─────────────────────────────────────────────────────


def test_a_record_becomes_one_json_object(capture):
    logging.getLogger("supervisor.nodes.dispatch").info("worker call started")
    (record,) = lines(capture)
    assert record["level"] == "INFO"
    assert record["logger"] == "supervisor.nodes.dispatch"
    assert record["message"] == "worker call started"
    assert record["ts"].endswith("+00:00"), "timestamps must be UTC, not container-local"


def test_extra_fields_are_carried_through(capture):
    logging.getLogger("x").warning("retrying", extra={"attempt": 2, "agent": "deployment-agent"})
    (record,) = lines(capture)
    assert record["attempt"] == 2
    assert record["agent"] == "deployment-agent"


def test_an_exception_is_serialized_not_dropped(capture):
    try:
        raise ValueError("worker exploded")
    except ValueError:
        logging.getLogger("x").exception("dispatch failed")
    (record,) = lines(capture)
    assert "worker exploded" in record["exception"]


def test_an_unserializable_extra_degrades_instead_of_raising(capture):
    """A formatter that throws takes out the line reporting the problem."""

    class Opaque:
        def __repr__(self):
            return "<opaque>"

    logging.getLogger("x").info("odd", extra={"thing": Opaque()})
    (record,) = lines(capture)
    assert record["thing"] == "<opaque>"


# ── correlation ─────────────────────────────────────────────────────────────


def test_bound_ids_appear_on_every_later_line(capture):
    obs.bind(conversation_id="thr-1", correlation_id="corr-9")
    logging.getLogger("a").info("one")
    logging.getLogger("b").warning("two")
    assert [r["conversation_id"] for r in lines(capture)] == ["thr-1", "thr-1"]
    assert [r["correlation_id"] for r in lines(capture)] == ["corr-9", "corr-9"]


def test_binding_is_additive(capture):
    obs.bind(conversation_id="thr-1")
    obs.bind(agent_id="coding-agent")
    logging.getLogger("a").info("one")
    (record,) = lines(capture)
    assert record["conversation_id"] == "thr-1"
    assert record["agent_id"] == "coding-agent"


def test_empty_values_are_not_bound(capture):
    """An absent correlation id should be absent, not an empty string."""
    obs.bind(correlation_id="", conversation_id=None, agent_id="coding-agent")
    logging.getLogger("a").info("one")
    (record,) = lines(capture)
    assert "correlation_id" not in record
    assert "conversation_id" not in record
    assert record["agent_id"] == "coding-agent"


def test_clear_ends_the_turn(capture):
    """Turns must not leak ids into each other on a reused worker process."""
    obs.bind(conversation_id="thr-1")
    logging.getLogger("a").info("during")
    obs.clear()
    logging.getLogger("a").info("after")
    during, after = lines(capture)
    assert during["conversation_id"] == "thr-1"
    assert "conversation_id" not in after


def test_current_reports_what_is_bound():
    obs.clear()
    assert obs.current() == {}
    obs.bind(conversation_id="thr-1")
    assert obs.current() == {"conversation_id": "thr-1"}
    obs.clear()


# ── configure ───────────────────────────────────────────────────────────────


def test_configure_is_idempotent():
    """The entrypoint may import more than once; handlers must not stack up."""
    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        root.handlers = []
        obs.configure()
        obs.configure()
        obs.configure()
        assert len([h for h in root.handlers if getattr(h, "_governance_handler", False)]) == 1
    finally:
        root.handlers = saved


def test_plain_format_opts_out_of_json():
    root = logging.getLogger()
    saved = root.handlers[:]
    try:
        root.handlers = []
        obs.configure(fmt="plain")
        handler = next(h for h in root.handlers if getattr(h, "_governance_handler", False))
        assert not isinstance(handler.formatter, obs.JsonFormatter)
    finally:
        root.handlers = saved
