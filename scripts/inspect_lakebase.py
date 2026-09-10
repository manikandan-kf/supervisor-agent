"""See what's actually stored in short-term and long-term memory.

`memory.py` picks one of three backends — Lakebase, plain Postgres, or
in-memory — from whatever is configured in `.env` (see that module's
docstring for the selection order). This script connects to whichever one is
active and shows the real rows, so "is memory actually persisting?" has an
answer you can look at instead of trust.

Two things it can show:

    Overview (default) — table row counts, the most recently active
    conversation threads (short-term memory), the long-term context stored per
    user, and the latest audit rows.

    One conversation, decoded — `--thread <thread_id>` loads a single
    checkpoint through the real `CheckpointSaver`/`PostgresSaver` object (not
    raw SQL), so the message list comes back as actual `HumanMessage`/
    `AIMessage` objects rather than the checkpoint's serialized bytes.

Usage:

    # Whatever this environment is configured for (Lakebase if LAKEBASE_INSTANCE
    # is set in .env, else SUPERVISOR_PG_DSN, else nothing to inspect)
    .venv\\Scripts\\python.exe scripts\\inspect_lakebase.py

    # A specific conversation, by the internal thr_... id (from a trace, the
    # audit table, or gateway logs — not the browser's conv-... handle, see
    # derived, not the browser's handle)
    .venv\\Scripts\\python.exe scripts\\inspect_lakebase.py --thread thr_f6d24b8ee16c33a973f874eb5e5d8b54

    # Force the Lakebase instance rather than whatever .env resolves to
    .venv\\Scripts\\python.exe scripts\\inspect_lakebase.py --instance supervisor-memory

This is read-only: every query is a SELECT, and `build_checkpointer()` /
`build_store()` only issue `CREATE TABLE IF NOT EXISTS` on `.setup()` — safe to
run against a live environment.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    try:
        here = Path(__file__)
    except NameError:  # spark_python_task execs the source; no __file__
        import inspect

        here = Path(inspect.currentframe().f_code.co_filename)
    return here.resolve().parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "src"))


def load_dotenv(path: Path) -> list[str]:
    """Minimal .env reader, same as trace_turn.py's — existing env vars win."""
    loaded = []
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


# Tables this project's three sinks create, in the order they're introduced in
# LangGraph's checkpointer (short-term/conversation),
# LangGraph's store (long-term/per-user), then the supervisor's own audit sink.
_KNOWN_TABLES = (
    "checkpoints",
    "checkpoint_writes",
    "checkpoint_blobs",
    "store",
    "supervisor_audit_log",
)


@contextlib.contextmanager
def raw_connection():
    """A plain DB-API connection for read-only introspection.

    Independent of which saver/store *classes* the graph itself uses — this
    just needs a socket to the same database, so raw SQL can look at what
    those classes already wrote.
    """
    from supervisor.memory import lakebase_target, resolve_dsn

    target = lakebase_target()
    if target:
        from databricks_ai_bridge.lakebase import LakebaseClient

        label = "Lakebase (" + ", ".join(f"{k}={v}" for k, v in target.items()) + ")"
        with LakebaseClient(**target) as client:
            with client.pool.connection() as conn:
                yield conn, label
        return

    dsn = resolve_dsn()
    if dsn:
        import psycopg

        # Don't print the DSN verbatim — it may carry a password.
        label = f"Postgres ({dsn.split('@')[-1] if '@' in dsn else '<dsn>'})"
        with psycopg.connect(dsn) as conn:
            yield conn, label
        return

    raise SystemExit(
        "Nothing configured to inspect: no LAKEBASE_INSTANCE / "
        "LAKEBASE_AUTOSCALING_ENDPOINT / LAKEBASE_PROJECT, and no "
        "SUPERVISOR_PG_DSN / LAKEBASE_DSN either. With none of those set, "
        "memory.py falls back to InMemorySaver / InMemoryStore — nothing "
        "durable exists to look at."
    )


def _cursor(conn):
    """A cursor that returns plain tuples, whatever row factory the connection
    itself defaults to (Lakebase's pool hands out dict-like rows, on which
    positional unpacking silently does the wrong thing: `a, b = some_dict`
    assigns the dict's *keys*, not its values, when it has exactly two)."""
    from psycopg.rows import tuple_row

    return conn.cursor(row_factory=tuple_row)


def _row_count(conn, table: str) -> "int | None":
    try:
        with _cursor(conn) as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            return cur.fetchone()[0]
    except Exception:
        conn.rollback()
        return None


def overview(conn, limit: int) -> None:
    print("── Table row counts " + "─" * 58)
    for table in _KNOWN_TABLES:
        count = _row_count(conn, table)
        print(f"  {table:<24} {count if count is not None else '(not found)'}")

    print()
    print(f"── Short-term memory: {limit} most recently active conversation threads " + "─" * 8)
    print("   (the LangGraph checkpointer — one row group per `thr_...` thread, §22.1)")
    try:
        with _cursor(conn) as cur:
            cur.execute(
                """
                SELECT thread_id, checkpoint_ns, count(*) AS steps, max(checkpoint_id) AS latest
                FROM checkpoints
                GROUP BY thread_id, checkpoint_ns
                ORDER BY latest DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        if not rows:
            print("  (no checkpoints yet — no conversation has run against this backend)")
        for thread_id, ns, steps, latest in rows:
            ns_label = f" ns={ns!r}" if ns else ""
            print(f"  {thread_id}{ns_label}  {steps} checkpoint(s), latest {latest}")
        print(
            "  Full content of one thread: rerun with --thread <thread_id>, or "
            "scripts/trace_turn.py --replay last --tracking databricks for the "
            "governance trail alongside it."
        )
    except Exception as exc:
        conn.rollback()
        print(f"  (couldn't read checkpoints: {exc})")

    print()
    print(f"── Long-term memory: {limit} most recently written per-user fields " + "─" * 8)
    print("   (the LangGraph store — survives across conversations, keyed by `usr_...`, never by thread)")
    try:
        with _cursor(conn) as cur:
            cur.execute(
                "SELECT prefix, key, value, updated_at FROM store ORDER BY updated_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        if not rows:
            print("  (empty — no agent has asked for required_context yet, or none was resolved)")
        for prefix, key, value, updated_at in rows:
            print(f"  [{prefix}] {key} = {value}  (updated {updated_at})")
    except Exception as exc:
        conn.rollback()
        print(f"  (couldn't read store: {exc})")

    print()
    print(f"── Decision trail: {limit} most recent audit rows (BR-006) " + "─" * 8)
    try:
        with _cursor(conn) as cur:
            cur.execute(
                """
                SELECT event_time, conversation_id, user_role, target_agent_id, outcome
                FROM supervisor_audit_log
                ORDER BY event_time DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        if not rows:
            print("  (empty)")
        for event_time, conversation_id, user_role, target_agent_id, outcome in rows:
            print(f"  {event_time}  {conversation_id}  {user_role:<10} {target_agent_id:<20} {outcome}")
    except Exception as exc:
        conn.rollback()
        print(f"  (couldn't read supervisor_audit_log: {exc})")


def show_thread(thread_id: str) -> None:
    """Decode one conversation the way the graph itself would read it back."""
    from supervisor.memory import build_checkpointer

    saver = build_checkpointer()
    config = {"configurable": {"thread_id": thread_id}}
    tup = saver.get_tuple(config)
    if tup is None:
        print(f"No checkpoint found for thread_id={thread_id!r}.")
        print("Note: this is the internal thr_... id, derived by the caller, not the")
        print("browser's conv-... handle — find it in a trace or the audit table's")
        print("conversation_id column.")
        return

    values = tup.checkpoint.get("channel_values", {})
    messages = values.get("messages") or []

    print(f"thread_id      : {thread_id}")
    print(f"checkpoint_id  : {tup.config.get('configurable', {}).get('checkpoint_id', '')}")
    print(f"messages       : {len(messages)}")
    print()
    for msg in messages:
        role = getattr(msg, "type", "?")
        content = getattr(msg, "content", "")
        print(f"  [{role}] {content}")

    print()
    for key in ("outcome", "target_agent_id", "clarification_count", "pending_clarification"):
        if key in values:
            print(f"{key:<22}: {values[key]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--thread", default="", help="decode one conversation by its internal thr_... id"
    )
    parser.add_argument(
        "--instance", default="", help="Lakebase instance name, overriding .env / LAKEBASE_INSTANCE"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="rows to show per section in overview mode (default 10)"
    )
    args = parser.parse_args()

    loaded = load_dotenv(ROOT / ".env")
    if loaded:
        print(f"(loaded from .env: {', '.join(loaded)})")

    if args.instance:
        # Overrides whatever .env resolved to, for both modes below — both go
        # through supervisor.memory's own env-var-driven backend selection.
        os.environ["LAKEBASE_INSTANCE"] = args.instance

    if args.thread:
        show_thread(args.thread)
        return 0

    with raw_connection() as (conn, label):
        print(f"Connected to: {label}")
        print()
        overview(conn, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
