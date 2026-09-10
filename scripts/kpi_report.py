"""Solution v1.2 §07 KPIs, computed from data this project already has.

No new service, no Databricks dashboard, no separate metrics store — this is
the reporting module named in the solution's KPI section, and
SOLUTION_STATUS_SUMMARY.md, closed the same way the rest of this project
favors custom code over a managed Databricks capability: plain SQL against
the audit log this project already writes (`supervisor_audit_log`, BR-006),
plus the MLflow traces it already produces. Reuses `inspect_lakebase.py`'s
connection handling and `trace_turn.py`'s trace-parsing helpers — in
particular `token_usage()`, which already corrects for MLflow's streamed-chunk
token-count inflation — rather than re-deriving either.

Two sources, because the two halves of §07 live in different places:

    Governance & usage KPIs   `supervisor_audit_log` — one row per turn,
    (routing, RBAC, guardrail,  written by `respond` (BR-006). Exact
    clarification, escalation,  decision strings this reads are pinned in
    usage, sessions, HITL)      `src/supervisor/nodes.py` (see the docstring
                                 on each compute_* function below).

    Cost & performance KPIs   MLflow traces — one trace per turn, autologged
    (latency, tokens)          by `mlflow.langchain.autolog()` (agent.py).

Honesty over false precision: two of the twelve KPIs — routing accuracy and
guardrail precision/recall — need a labeled ground truth (was this query
*actually* routable to that agent? was that block *actually* wrong?) that
live traffic alone cannot supply. Those sections print a clearly-labeled
proxy instead of a fabricated number; see each docstring for exactly what
the proxy measures and why it is not the real thing.

Usage:

    .venv\\Scripts\\python.exe scripts\\kpi_report.py
    .venv\\Scripts\\python.exe scripts\\kpi_report.py --since-hours 24
    .venv\\Scripts\\python.exe scripts\\kpi_report.py --tracking databricks --json

Read-only against both sources: every audit-log query is a SELECT, and MLflow
trace search never mutates anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_lakebase import _cursor, load_dotenv, raw_connection  # noqa: E402
from trace_turn import NODES, ms_of, span_index, token_usage  # noqa: E402

# §07 defines "Supervisor latency" as time to a *routing decision* — so only
# these three stages count. Excluding by name, not by "everything but
# dispatch", matters here: `approval` is a wait on a human, potentially
# minutes or hours, and `respond` is post-routing formatting — either one
# leaking into this sum would corrupt the P95 far worse than dispatch would.
_ROUTING_STAGES = ("rbac_gate", "guardrails", "route")
# "Token consumption... tracked separately from each worker agent's own
# consumption" (§07) — dispatch is the worker's call (today, the simulated
# stand-in for one), so it is excluded from the supervisor-only figure below
# and reported on its own line instead.
_WORKER_STAGE = "dispatch"


# ── audit-log side: governance & usage KPIs ─────────────────────────────────


def fetch_audit_rows(conn, since: datetime) -> list[dict]:
    with _cursor(conn) as cur:
        cur.execute(
            "SELECT event_time, conversation_id, user_role, user_key, "
            "       target_agent_id, outcome, decision_trail, "
            # The durable half of the token KPI (spend.py). MLflow spans carry
            # the *billed* figures and are still the authority on them, but
            # their retention is shorter than this table's — so the question
            # "what did this user spend last quarter" has an answer here and
            # nowhere else. NULL for any row written before the columns
            # existed, which `spend_kpis` treats as unknown rather than zero.
            "       model_calls, tokens_estimated "
            "FROM supervisor_audit_log WHERE event_time >= %s ORDER BY event_time",
            (since,),
        )
        cols = [d.name for d in cur.description]
        # strict=True: a row whose width does not match the cursor description
        # means the SELECT and the column list have drifted apart, and silently
        # truncating would drop a KPI's input rather than say so.
        rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
    for row in rows:
        trail = row.get("decision_trail")
        if isinstance(trail, str):
            try:
                trail = json.loads(trail)
            except json.JSONDecodeError:
                trail = []
        row["decision_trail"] = trail if isinstance(trail, list) else []
    return rows


def _entries(row: dict, stage: str) -> list[dict]:
    return [e for e in row["decision_trail"] if isinstance(e, dict) and e.get("stage") == stage]


def governance_kpis(rows: list[dict]) -> dict:
    """Routing, RBAC, guardrail, clarification and escalation — one row per turn.

    Decision strings read here are exactly what `src/supervisor/nodes.py` writes
    via `_trail()`/`_entry()`; a rename there without a matching update here
    would silently zero out the affected KPI, so keep the two in sync.
    """
    total = len(rows)
    if total == 0:
        return {"turns": 0}

    rbac_denies = [r for r in rows if _entries(r, "rbac_gate") and
                   any(e.get("decision") == "deny" for e in _entries(r, "rbac_gate"))]
    rbac_denies_enforced = [r for r in rbac_denies if r["outcome"] == "blocked"]

    guardrail_blocks = [r for r in rows if r["outcome"] == "blocked"]
    clarifies = [r for r in rows if r["outcome"] == "clarify"]
    escalations = [r for r in rows if r["outcome"] == "escalated"]

    dispatched = [r for r in rows if _entries(r, "dispatch")]
    worker_failures = [
        r for r in dispatched
        if any(e.get("decision") in ("unavailable", "error", "invalid_response")
               for e in _entries(r, "dispatch"))
    ]

    # PROXY, not the real KPI: "routing accuracy" needs a ground-truth label
    # (was this the *correct* agent?) that production traffic does not carry.
    # This instead measures completion — a routable turn that actually reached
    # and got an answer from a worker — which is necessary but not sufficient
    # for "accurate". A labeled eval set (see evaluations/) is the only way to
    # measure the real thing.
    routable = [r for r in rows if r["outcome"] not in ("blocked",)]
    completed = [r for r in routable if r["outcome"] == "answer" and r["target_agent_id"]]

    return {
        "turns": total,
        "routing_completion_proxy": _ratio(len(completed), len(routable)),
        "routing_completion_proxy_note": (
            "PROXY for 'routing accuracy' — measures completed routes, not "
            "correct ones; needs a labeled eval set for the real KPI"
        ),
        "rbac_enforcement_rate": _ratio(len(rbac_denies_enforced), len(rbac_denies)),
        "rbac_denies_seen": len(rbac_denies),
        "guardrail_block_rate": _ratio(len(guardrail_blocks), total),
        "guardrail_precision_recall_note": (
            "NOT COMPUTABLE from live traffic alone — precision/recall need "
            "labeled off-domain/in-domain examples, not just observed outcomes"
        ),
        "clarification_loop_rate": _ratio(len(clarifies), total),
        "human_escalation_rate": _ratio(len(escalations), total),
        "worker_error_rate": _ratio(len(worker_failures), len(dispatched)),
        "worker_dispatches_seen": len(dispatched),
    }


def usage_kpis(rows: list[dict]) -> dict:
    """Usage by role/agent, active users, sessions, session duration, HITL turnaround."""
    if not rows:
        return {}

    by_role_agent: dict[str, int] = {}
    for r in rows:
        key = f"{r['user_role'] or '?'} / {r['target_agent_id'] or '?'}"
        by_role_agent[key] = by_role_agent.get(key, 0) + 1

    users = {r["user_key"] for r in rows if r["user_key"]}
    sessions: dict[str, list[datetime]] = {}
    for r in rows:
        cid = r["conversation_id"]
        if cid:
            sessions.setdefault(cid, []).append(r["event_time"])

    durations = [
        (max(ts) - min(ts)).total_seconds() for ts in sessions.values() if len(ts) > 1
    ]

    turnarounds = _hitl_turnarounds(rows)

    return {
        "usage_by_role_and_agent": by_role_agent,
        "active_users": len(users),
        "session_count": len(sessions),
        "session_duration_seconds_avg": _avg(durations),
        "session_duration_seconds_median": _median(durations),
        "sessions_with_multiple_turns": len(durations),
        "hitl_turnaround_seconds_avg": _avg(turnarounds),
        "hitl_turnaround_seconds_median": _median(turnarounds),
        "hitl_approvals_resolved": len(turnarounds),
    }


def _hitl_turnarounds(rows: list[dict]) -> list[float]:
    """Time from a staged artifact ('dispatch'/'approval_pending') to a human's
    decision ('approval'/'approved' or 'approval'/'rejected') on the same
    conversation — these are always two different turns (two different
    requests), so two different audit rows, correlated by `conversation_id`.
    """
    by_thread: dict[str, list[dict]] = {}
    for r in rows:
        cid = r["conversation_id"]
        if cid:
            by_thread.setdefault(cid, []).append(r)

    out: list[float] = []
    for thread_rows in by_thread.values():
        thread_rows = sorted(thread_rows, key=lambda r: r["event_time"])
        pending_at: datetime | None = None
        for r in thread_rows:
            if any(e.get("decision") == "approval_pending" for e in _entries(r, "dispatch")):
                pending_at = r["event_time"]
            elif pending_at is not None and any(
                e.get("decision") in ("approved", "rejected") for e in _entries(r, "approval")
            ):
                out.append((r["event_time"] - pending_at).total_seconds())
                pending_at = None
    return out


def spend_kpis(rows: list[dict]) -> dict:
    """Governance spend from the audit table — the durable cost KPI.

    Separate from `performance_kpis` below, which reads MLflow traces, and the
    two answer different questions on purpose:

      * traces carry the **billed** token counts, per span, and are the
        authority on cost — but retention is shorter than this table's;
      * this reads what the graph **charged itself** (`spend.py`) per turn,
        durably, attributable per user and per agent for as long as the audit
        table is kept.

    They will not agree exactly and are not meant to: one is measured, one is
    estimated. A large divergence is itself the signal — it means the estimate
    in `spend.py` needs re-basing.

    `top_spenders` exists to answer the operational question the subject
    allowance needs before it can be switched on: what does a *normal* heavy
    user spend? Setting SUBJECT_MAX_MODEL_CALLS from a guess refuses real
    users; setting it from this refuses outliers.
    """
    accounted = [r for r in rows if r.get("model_calls") is not None]
    if not accounted:
        return {
            "turns_accounted": 0,
            "note": (
                "no rows carry model_calls — either the audit table predates the "
                "column (run scripts/provision_lakebase.py --migrate-audit) or "
                "every turn in the window was refused before any model call"
            ),
        }

    calls = [int(r["model_calls"] or 0) for r in accounted]
    tokens = [int(r["tokens_estimated"] or 0) for r in accounted]

    by_subject: dict[str, dict[str, int]] = {}
    for row in accounted:
        key = row.get("user_key") or "?"
        bucket = by_subject.setdefault(key, {"turns": 0, "model_calls": 0, "tokens": 0})
        bucket["turns"] += 1
        bucket["model_calls"] += int(row["model_calls"] or 0)
        bucket["tokens"] += int(row["tokens_estimated"] or 0)

    return {
        "turns_accounted": len(accounted),
        "turns_unaccounted": len(rows) - len(accounted),
        "model_calls_total": sum(calls),
        "model_calls_per_turn_avg": _avg(calls),
        "model_calls_per_turn_max": max(calls),
        "tokens_estimated_total": sum(tokens),
        "tokens_estimated_per_turn_avg": _avg(tokens),
        "top_spenders": dict(
            sorted(by_subject.items(), key=lambda kv: kv[1]["model_calls"], reverse=True)[:5]
        ),
        "note": (
            "estimated by the graph's own ledger (spend.py), not billed — the "
            "MLflow figures below are the measured ones"
        ),
    }


# ── MLflow side: cost & performance KPIs ────────────────────────────────────


def fetch_traces(mlflow, since_ms: int, max_traces: int) -> list:
    """Most recent traces, filtered client-side to the window.

    `search_traces` returns newest-first with no filter needed for "how far
    back", but its own filter-string grammar is a moving target across MLflow
    versions — filtering the returned `TraceInfo.request_time` (ms since
    epoch) in plain Python is simpler and version-stable. `max_traces` bounds
    how far back this can see in one call; traces older than the
    `max_traces`'th-most-recent are silently outside this report, not
    silently wrong — raise --max-traces if the window looks incomplete.
    """
    found = mlflow.search_traces(max_results=max_traces, return_type="list")
    return [t for t in (found or []) if t.info.request_time >= since_ms]


def performance_kpis(traces: list) -> dict:
    if not traces:
        return {"traces": 0}

    stage_latencies_ms: dict[str, list[float]] = {n: [] for n in NODES}
    supervisor_latencies_ms: list[float] = []
    tokens_supervisor_only: list[int] = []
    tokens_all_stages: list[int] = []
    tokens_by_stage: dict[str, int] = {n: 0 for n in NODES}
    token_note_seen = False

    for trace in traces:
        spans, stage_of = span_index(trace)
        per_stage_ms: dict[str, float] = {}
        trace_tokens_supervisor = 0
        trace_tokens_all = 0

        for span in spans:
            if span.end_time_ns is None:
                continue
            # The node's own span is named after the node (`stage_of`'s
            # ancestor walk assumes this too) — summing only these avoids
            # double-counting a node's nested child spans under the same stage.
            if span.name in stage_latencies_ms:
                per_stage_ms[span.name] = per_stage_ms.get(span.name, 0.0) + ms_of(span)

            if str(span.span_type) not in ("CHAT_MODEL", "LLM"):
                continue
            usage, note = token_usage(span.attributes or {})
            if note:
                token_note_seen = True
            total = usage.get("total_tokens")
            if isinstance(total, int):
                trace_tokens_all += total
                stage = stage_of(span)
                if stage in tokens_by_stage:
                    tokens_by_stage[stage] += total
                if stage != _WORKER_STAGE:
                    trace_tokens_supervisor += total

        for stage, ms in per_stage_ms.items():
            stage_latencies_ms[stage].append(ms)

        routing_ms = sum(per_stage_ms.get(s, 0.0) for s in _ROUTING_STAGES)
        if routing_ms:
            supervisor_latencies_ms.append(routing_ms)
        if trace_tokens_all:
            tokens_all_stages.append(trace_tokens_all)
        if trace_tokens_supervisor:
            tokens_supervisor_only.append(trace_tokens_supervisor)

    return {
        "traces": len(traces),
        "supervisor_latency_ms_p50": _percentile(supervisor_latencies_ms, 50),
        "supervisor_latency_ms_p95": _percentile(supervisor_latencies_ms, 95),
        "supervisor_latency_stages": _ROUTING_STAGES,
        "token_consumption_supervisor_avg_per_turn": _avg(tokens_supervisor_only),
        "token_consumption_all_stages_avg_per_turn": _avg(tokens_all_stages),
        "token_consumption_by_stage_total": tokens_by_stage,
        "token_note": (
            "MLflow's raw input-token/cost figures are inflated on streamed "
            "calls; totals above use trace_turn.py's corrected total-output figure"
            if token_note_seen else ""
        ),
    }


# ── small stats helpers — no numpy, this is twelve numbers ─────────────────


def _ratio(numerator: int, denominator: int) -> "float | None":
    return round(numerator / denominator, 4) if denominator else None


def _avg(values: list[float]) -> "float | None":
    return round(sum(values) / len(values), 2) if values else None


def _median(values: list[float]) -> "float | None":
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return round(s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2, 2)


def _percentile(values: list[float], pct: float) -> "float | None":
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


# ── report ───────────────────────────────────────────────────────────────


def render(gov: dict, usage: dict, spend: dict, perf: dict, since: datetime) -> None:
    print(f"Supervisor Agent KPIs (Solution v1.2 §07) — since {since.isoformat()}")
    print("=" * 72)

    print("\n── Routing & governance " + "─" * 47)
    if gov.get("turns"):
        print(f"  turns observed              : {gov['turns']}")
        print(f"  routing completion (PROXY)  : {_pct(gov['routing_completion_proxy'])}")
        print(f"    — {gov['routing_completion_proxy_note']}")
        print(f"  RBAC enforcement rate       : {_pct(gov['rbac_enforcement_rate'])}"
              f"  ({gov['rbac_denies_seen']} denies seen)")
        print(f"  guardrail block rate        : {_pct(gov['guardrail_block_rate'])}")
        print(f"    — {gov['guardrail_precision_recall_note']}")
        print(f"  clarification loop rate     : {_pct(gov['clarification_loop_rate'])}")
        print(f"  human escalation rate       : {_pct(gov['human_escalation_rate'])}")
    else:
        print("  (no audit rows in this window)")

    print("\n── Team & usage " + "─" * 55)
    if usage:
        print(f"  active users                : {usage['active_users']}")
        print(f"  sessions                    : {usage['session_count']}")
        print(f"  session duration avg/median : {_secs(usage['session_duration_seconds_avg'])}"
              f" / {_secs(usage['session_duration_seconds_median'])}"
              f"  ({usage['sessions_with_multiple_turns']} multi-turn sessions)")
        print(f"  HITL turnaround avg/median  : {_secs(usage['hitl_turnaround_seconds_avg'])}"
              f" / {_secs(usage['hitl_turnaround_seconds_median'])}"
              f"  ({usage['hitl_approvals_resolved']} resolved)")
        print("  usage by role / agent:")
        for key, count in sorted(usage["usage_by_role_and_agent"].items(), key=lambda kv: -kv[1]):
            print(f"    {key:<45} {count}")
    else:
        print("  (no audit rows in this window)")

    print("\n── Governance spend (audit table, durable) " + "─" * 29)
    if spend.get("turns_accounted"):
        print(f"  turns accounted             : {spend['turns_accounted']}"
              f"  ({spend['turns_unaccounted']} without spend columns)")
        print(f"  model calls total / per turn: {spend['model_calls_total']}"
              f" / {spend['model_calls_per_turn_avg']} avg,"
              f" {spend['model_calls_per_turn_max']} max")
        print(f"  tokens (estimated) total    : {spend['tokens_estimated_total']}"
              f"  ({spend['tokens_estimated_per_turn_avg']} avg/turn)")
        print("  top spenders (set SUBJECT_MAX_MODEL_CALLS from these, not a guess):")
        for subject, totals in spend["top_spenders"].items():
            print(f"    {subject:<45} {totals['model_calls']} calls"
                  f" / {totals['turns']} turns")
    else:
        print(f"  ({spend.get('note', 'no audit rows in this window')})")
    if spend.get("note") and spend.get("turns_accounted"):
        print(f"  note: {spend['note']}")

    print("\n── Cost & performance (MLflow traces) " + "─" * 33)
    if perf.get("traces"):
        stages = " + ".join(perf["supervisor_latency_stages"])
        print(f"  traces observed             : {perf['traces']}")
        print(f"  supervisor latency P50/P95  : {_ms(perf['supervisor_latency_ms_p50'])}"
              f" / {_ms(perf['supervisor_latency_ms_p95'])}"
              f"  ({stages} only — the routing decision, not dispatch or approval wait)")
        print(f"  worker error/failure rate   : {_pct(gov.get('worker_error_rate'))}"
              f"  ({gov.get('worker_dispatches_seen', 0)} dispatches)")
        print(f"  tokens/turn, supervisor only: "
              f"{perf['token_consumption_supervisor_avg_per_turn']}"
              f"  (excludes '{_WORKER_STAGE}' — the worker's own consumption)")
        print(f"  tokens/turn, all stages     : {perf['token_consumption_all_stages_avg_per_turn']}")
        print("  tokens by stage (total):")
        for stage, total in perf["token_consumption_by_stage_total"].items():
            if total:
                print(f"    {stage:<12} {total}")
        if perf.get("token_note"):
            print(f"  note: {perf['token_note']}")
    else:
        print("  (no traces in this window — check --tracking/--experiment)")


def _pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def _secs(value) -> str:
    return f"{value:.0f}s" if value is not None else "n/a"


def _ms(value) -> str:
    return f"{value:.0f}ms" if value is not None else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--since-hours", type=float, default=168.0,
        help="how far back to look (default: 168 = 1 week)",
    )
    parser.add_argument("--instance", default="", help="Lakebase instance, overriding .env")
    parser.add_argument(
        "--tracking", choices=("local", "databricks"), default="local",
        help="MLflow store to read traces from — same meaning as trace_turn.py",
    )
    parser.add_argument(
        "--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "/Shared/supervisor-agent"),
        help="experiment name when --tracking databricks",
    )
    parser.add_argument(
        "--max-traces", type=int, default=1000,
        help="most recent traces to scan before filtering to the window",
    )
    parser.add_argument("--json", dest="json_out", action="store_true", help="print JSON instead")
    args = parser.parse_args()

    loaded = load_dotenv(ROOT / ".env")
    if not args.json_out and loaded:
        print(f"(loaded from .env: {', '.join(loaded)})")
    if args.instance:
        os.environ["LAKEBASE_INSTANCE"] = args.instance

    since = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)

    with raw_connection() as (conn, _label):
        rows = fetch_audit_rows(conn, since)
    gov = governance_kpis(rows)
    usage = usage_kpis(rows)
    spend = spend_kpis(rows)

    import mlflow

    if args.tracking == "databricks":
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(args.experiment)
    else:
        store = ROOT / ".mlflow-local"
        store.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{(store / 'traces.db').as_posix()}")
        mlflow.set_experiment("supervisor-local-traces")

    since_ms = int(since.timestamp() * 1000)
    traces = fetch_traces(mlflow, since_ms, args.max_traces)
    perf = performance_kpis(traces)

    if args.json_out:
        print(json.dumps({"since": since.isoformat(), "governance": gov, "usage": usage,
                           "spend": spend, "performance": perf}, default=str, indent=2))
    else:
        render(gov, usage, spend, perf, since)
    return 0


if __name__ == "__main__":
    sys.exit(main())
