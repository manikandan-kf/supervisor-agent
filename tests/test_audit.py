"""Audit sink behaviour — BR-006.

The decision trail is the deliverable for BR-006, so the sink has to write from
*every* identity that runs the graph, not just whichever one happened to create
the table.
"""

from __future__ import annotations

import pytest

from supervisor.audit import LoggingAuditLogger, PostgresAuditLogger, TracingAuditLogger

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
        # The column probe (`information_schema.columns`). An existing table is
        # modelled as already having the optional columns, which is the state a
        # table created by the current DDL is in — the narrow-table case gets its
        # own test rather than being the default everything else inherits.
        if "information_schema.columns" in self._last:
            return [
                {"column_name": name}
                for name in ("latency_ms", "session_age_seconds", "signoff")
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
    """The fix for `must be owner of table`.

    Whoever creates the table owns it, and Postgres gives a table one owner. The
    serving endpoint's System Service Principal is therefore *not* the owner
    wherever a deploy or a local run got there first — it only holds INSERT.
    `CREATE INDEX IF NOT EXISTS` checks ownership before it checks existence, so
    issuing it unconditionally raised InsufficientPrivilege and aborted the
    transaction, losing the INSERT with it. Every request looked healthy and the
    table stayed empty.
    """
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
    """Solution v1.2 §08 wants the decision on the trace *in addition to* the
    durable sink — most call sites (offline, no autolog) have no active
    MLflow trace, and that must never block the durable write."""
    inner = FakeInnerLogger()
    TracingAuditLogger(inner).log(RECORD)
    assert inner.records == [RECORD]


def test_tracing_wrapper_tags_a_genuinely_active_trace(tmp_path, monkeypatch):
    import mlflow

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
