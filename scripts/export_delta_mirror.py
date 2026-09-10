"""Mirror the audit log and MLflow traces into Unity Catalog Delta tables.

Why a batch job instead of writing Delta directly from the agent (§24.9):

* The serving endpoint's system service principal is not in SCIM, so it can
  never hold the `databricks-sql-access` entitlement a live INSERT through a
  SQL warehouse needs — that ended `DeltaAuditLogger` as a live sink.
* MLflow's native UC Delta trace destination
  (`set_experiment(trace_location=UnityCatalog(...))`) is rejected by this
  workspace outright: "Unsupported table kind. Tables created in default
  storage are not supported" — the metastore has no customer external
  location, only Databricks-managed default storage.

Both walls are specific to *who* writes and *how*. Plain Delta tables in
default storage work fine, and this job runs on serverless job compute as the
deploying user — an identity with full UC and Lakebase access. So:

    Lakebase `supervisor_audit_log`  ──►  {catalog}.{schema}.audit_log
    MLflow experiment traces         ──►  {catalog}.{schema}.mlflow_traces

Postgres (Lakebase) stays the durable low-latency sink of record; these Delta
tables are the governed, SQL-queryable mirror the solution doc's §08 asks for.
Incremental on every run: audit rows by their BIGSERIAL id, traces by
trace_id anti-join, so re-runs never duplicate.

Runs as the DAB job `supervisor_delta_export` (hourly — see databricks.yml),
or on demand:

    databricks bundle run supervisor_delta_export -t dev
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

AUDIT_DELTA_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
  source_id       BIGINT COMMENT 'id of the mirrored row in Lakebase supervisor_audit_log — the incremental watermark',
  event_time      TIMESTAMP,
  request_id      STRING,
  correlation_id  STRING,
  conversation_id STRING,
  user_role       STRING,
  user_key        STRING,
  target_agent_id STRING,
  outcome         STRING,
  decision_trail  STRING COMMENT 'JSON array — the full decision trail (BR-006)',
  exported_at     TIMESTAMP
) USING DELTA
COMMENT 'Governed mirror of the Lakebase audit sink, maintained by the supervisor_delta_export job'
"""

TRACES_DELTA_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
  trace_id              STRING,
  request_time          TIMESTAMP,
  state                 STRING COMMENT 'OK / ERROR / IN_PROGRESS',
  execution_duration_ms BIGINT,
  request_preview       STRING,
  response_preview      STRING,
  tags                  STRING COMMENT 'JSON object — includes the audit mirror written by TracingAuditLogger',
  trace_metadata        STRING COMMENT 'JSON object',
  trace_json            STRING COMMENT 'full trace with all spans, as JSON',
  exported_at           TIMESTAMP
) USING DELTA
COMMENT 'Governed mirror of MLflow traces, maintained by the supervisor_delta_export job'
"""


def export_audit(spark, instance: str, delta_table: str, schema: str = "") -> int:
    """Copy Lakebase audit rows newer than the Delta watermark. Returns rows written.

    `schema` names the environment whose trail is being mirrored. The query
    below is unqualified, so without it the connection reads through `public`
    and mirrors either nothing or another environment's rows into this
    environment's Delta table.
    """
    spark.sql(AUDIT_DELTA_DDL.format(table=delta_table))
    watermark = spark.sql(f"SELECT coalesce(max(source_id), 0) FROM {delta_table}").collect()[0][0]

    from databricks_ai_bridge.lakebase import LakebaseClient
    from psycopg.rows import tuple_row

    rows: list[tuple] = []
    with LakebaseClient(instance_name=instance, **({"schema": schema} if schema else {})) as client:
        with client.pool.connection() as conn:
            with conn.cursor(row_factory=tuple_row) as cur:
                cur.execute("SELECT to_regclass('supervisor_audit_log') IS NOT NULL")
                if not cur.fetchone()[0]:
                    # No request has reached the endpoint since the instance was
                    # (re)created — the sink makes its table on first write.
                    print("audit: supervisor_audit_log does not exist in Lakebase yet — nothing to mirror")
                    return 0
                cur.execute(
                    "SELECT id, event_time, request_id, correlation_id, conversation_id, "
                    "user_role, user_key, target_agent_id, outcome, decision_trail "
                    "FROM supervisor_audit_log WHERE id > %s ORDER BY id",
                    (int(watermark),),
                )
                rows = cur.fetchall()

    if not rows:
        print(f"audit: up to date (watermark id {watermark})")
        return 0

    now = datetime.now(timezone.utc)
    records = [
        (
            int(r[0]),
            r[1],
            r[2],
            r[3],
            r[4],
            r[5],
            r[6],
            r[7],
            r[8],
            json.dumps(r[9], default=str) if not isinstance(r[9], str) else r[9],
            now,
        )
        for r in rows
    ]
    schema = (
        "source_id BIGINT, event_time TIMESTAMP, request_id STRING, correlation_id STRING, "
        "conversation_id STRING, user_role STRING, user_key STRING, target_agent_id STRING, "
        "outcome STRING, decision_trail STRING, exported_at TIMESTAMP"
    )
    spark.createDataFrame(records, schema=schema).write.mode("append").saveAsTable(delta_table)
    print(f"audit: mirrored {len(records)} row(s) (ids {records[0][0]}..{records[-1][0]}) -> {delta_table}")
    return len(records)


def export_traces(spark, experiment: str, delta_table: str, max_traces: int) -> int:
    """Copy traces not yet in the Delta mirror. Returns traces written.

    Fetches the newest `max_traces` and anti-joins on trace_id rather than
    trusting a timestamp filter — idempotent under re-runs and clock skew, and
    the fetch window just needs to be larger than one schedule interval's
    traffic (hourly, so 1000 is generous).
    """
    import mlflow

    mlflow.set_tracking_uri("databricks")
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        print(f"traces: experiment {experiment!r} not found — nothing to mirror")
        return 0

    spark.sql(TRACES_DELTA_DDL.format(table=delta_table))
    known = {
        row[0]
        for row in spark.sql(f"SELECT trace_id FROM {delta_table}").collect()
    }

    traces = mlflow.search_traces(
        locations=[exp.experiment_id],
        order_by=["timestamp_ms DESC"],
        max_results=max_traces,
        return_type="list",
    )
    fresh = [t for t in traces if t.info.trace_id not in known]
    if not fresh:
        print(f"traces: up to date ({len(traces)} checked, all already mirrored)")
        return 0

    def _as_millis(value):
        # `execution_duration` is typed timedelta but observed as a plain int
        # of milliseconds on this workspace (mlflow 3.15.1) — accept both.
        if value is None:
            return None
        if hasattr(value, "total_seconds"):
            return int(value.total_seconds() * 1000)
        return int(value)

    def _as_datetime(value):
        # Same defensive shape for `request_time`: datetime or ms epoch.
        if value is None or isinstance(value, datetime):
            return value
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)

    now = datetime.now(timezone.utc)
    records = []
    for t in fresh:
        info = t.info
        records.append(
            (
                info.trace_id,
                _as_datetime(getattr(info, "request_time", None)),
                str(getattr(info, "state", "")),
                _as_millis(getattr(info, "execution_duration", None)),
                getattr(info, "request_preview", None),
                getattr(info, "response_preview", None),
                json.dumps(dict(info.tags or {}), default=str),
                json.dumps(dict(getattr(info, "trace_metadata", None) or {}), default=str),
                t.to_json(),
                now,
            )
        )
    schema = (
        "trace_id STRING, request_time TIMESTAMP, state STRING, execution_duration_ms BIGINT, "
        "request_preview STRING, response_preview STRING, tags STRING, trace_metadata STRING, "
        "trace_json STRING, exported_at TIMESTAMP"
    )
    spark.createDataFrame(records, schema=schema).write.mode("append").saveAsTable(delta_table)
    print(f"traces: mirrored {len(records)} trace(s) -> {delta_table}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lakebase-instance", required=True)
    parser.add_argument(
        "--lakebase-schema",
        default="",
        help="Postgres schema of the environment being mirrored (supervisor_dev, "
        "supervisor_prod). Empty reads `public`.",
    )
    parser.add_argument("--audit-delta-table", required=True)
    parser.add_argument("--traces-delta-table", required=True)
    parser.add_argument("--experiment", default="/Shared/supervisor-agent")
    parser.add_argument("--max-traces", type=int, default=1000)
    args = parser.parse_args()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    audit_rows = export_audit(
        spark, args.lakebase_instance, args.audit_delta_table, args.lakebase_schema
    )
    trace_rows = export_traces(spark, args.experiment, args.traces_delta_table, args.max_traces)
    print(f"done: {audit_rows} audit row(s), {trace_rows} trace(s)")


if __name__ == "__main__":
    main()
