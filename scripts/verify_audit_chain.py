"""Walk the audit table's hash chain and report whether history is intact.

    # verify the whole chain
    python scripts/verify_audit_chain.py

    # print just the chain head (row id + hash), for external anchoring
    python scripts/verify_audit_chain.py --head

    # verify a range (e.g. since the last anchored head)
    python scripts/verify_audit_chain.py --after-id 148213

Each audit row carries `row_hash = sha256(prev_hash + canonical(row content))`,
written under a transaction-scoped advisory lock so the chain never forks (see
`supervisor.audit`). Editing or deleting any historical row breaks every digest
after it, which is what this walk detects.

What a clean result proves, precisely: no row between the anchor and the head
was altered or removed *without rewriting the whole suffix of the chain*. An
attacker with UPDATE on the table could rewrite that suffix — which is why the
runtime identity is append-only (`provision_lakebase.py
--restrict-governance-tables`) and why `--head` exists: record its output
somewhere outside operator write scope (a ticket, a signed commit, a WORM
bucket) on a schedule, and a suffix rewrite becomes detectable too.

Rows from before the migration (`prev_hash IS NULL`) are reported and skipped:
they predate the control and cannot retroactively join the chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_GENESIS = "0" * 64

# The columns hashed, in the exact order the writer builds them — must match
# `supervisor.audit.PostgresAuditLogger.log`.
_BASE_COLUMNS = (
    "event_time",
    "request_id",
    "correlation_id",
    "conversation_id",
    "user_role",
    "user_key",
    "target_agent_id",
    "outcome",
    "decision_trail",
)
_OPTIONAL = (
    "latency_ms",
    "session_age_seconds",
    "signoff",
    "model_calls",
    "tokens_estimated",
    "provenance",
)

# Columns added to the audit table *after* the hash chain shipped. A row
# written before a `--migrate-audit` that added them was digested without those
# keys, so verifying it with them present — as NULLs — would report a broken
# chain for a row nobody touched.
#
# Rows predating the *hash* columns need no such handling: they carry no
# `row_hash` at all and are counted as unchained. These do carry one, which is
# why they need it. Each row is therefore checked against the full key set
# first and against the earlier one only if that fails, so a genuinely tampered
# row still fails both.
_LATER_ADDITIONS = ("model_calls", "tokens_estimated", "provenance")


def _digest(prev: str, payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + canonical).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--table",
        default=None,
        help="audit table name (default: AUDIT_PG_TABLE or supervisor_audit_log)",
    )
    parser.add_argument("--head", action="store_true", help="print the chain head and exit")
    parser.add_argument(
        "--after-id", type=int, default=0, help="verify only rows with id greater than this"
    )
    args = parser.parse_args()

    from supervisor.memory import audit_connection_source
    from supervisor.settings import Settings

    table = args.table or Settings().audit_pg_table
    source = audit_connection_source()
    if source is None:
        print("no Postgres configured (SUPERVISOR_PG_DSN or Lakebase)", file=sys.stderr)
        return 2

    with source() as conn:
        with conn.cursor() as cur:
            if args.head:
                cur.execute(
                    f"SELECT id, row_hash FROM {table} "  # noqa: S608 - operator tool, name from config
                    "WHERE row_hash IS NOT NULL ORDER BY id DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row is None:
                    print("no chained rows yet")
                    return 0
                rid = row["id"] if isinstance(row, dict) else row[0]
                digest = row["row_hash"] if isinstance(row, dict) else row[1]
                print(f"head id={rid} row_hash={digest}")
                return 0

            columns = ", ".join(("id",) + _BASE_COLUMNS + _OPTIONAL + ("prev_hash", "row_hash"))
            cur.execute(
                f"SELECT {columns} FROM {table} WHERE id > %s ORDER BY id ASC",  # noqa: S608
                (args.after_id,),
            )
            rows = cur.fetchall() or []

    names = ("id",) + _BASE_COLUMNS + _OPTIONAL + ("prev_hash", "row_hash")
    checked = broken = unchained = 0
    expected_prev = None

    for raw in rows:
        row = raw if isinstance(raw, dict) else dict(zip(names, raw, strict=False))
        if not row.get("row_hash"):
            unchained += 1
            expected_prev = None  # a pre-migration gap legitimately resets the anchor
            continue

        # The writer digests *parsed* values (see `supervisor.audit._row_digest`)
        # precisely so this read-back can reproduce them: jsonb returns the
        # trail parsed, timestamps come back as datetimes whose `str()` matches
        # what `default=str` produced at write time.
        payload = {name: row.get(name) for name in _BASE_COLUMNS}
        for name in _OPTIONAL:
            if name in row:
                payload[name] = row.get(name)

        prev = row.get("prev_hash") or _GENESIS
        if expected_prev is not None and prev != expected_prev:
            broken += 1
            print(f"row id={row.get('id')}: prev_hash does not match the previous row's hash")
        expected = row.get("row_hash")
        if _digest(prev, payload) != expected:
            # Retry against the pre-migration key set before calling it broken
            # — see `_LATER_ADDITIONS`.
            earlier = {k: v for k, v in payload.items() if k not in _LATER_ADDITIONS}
            if _digest(prev, earlier) != expected:
                broken += 1
                print(f"row id={row.get('id')}: row_hash does not match the row's content")
        expected_prev = row.get("row_hash")
        checked += 1

    print(f"{checked} chained row(s) verified, {unchained} pre-migration row(s) skipped")
    if broken:
        print(f"CHAIN BROKEN: {broken} inconsistenc(ies) — treat the trail as tampered")
        return 1
    print("chain intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
