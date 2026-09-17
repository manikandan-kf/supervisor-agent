"""Lakebase / Postgres plumbing every agent shares.

Entry points: `build_checkpointer` / `build_store` (Lakebase via `LAKEBASE_INSTANCE`,
`LAKEBASE_AUTOSCALING_ENDPOINT` or `LAKEBASE_PROJECT`+`LAKEBASE_BRANCH`; else in-memory, refused
off a workstation by `refuse_non_durable`), `audit_connection_source` (pooled, credentials
rotate ~15 min), `safe_identifier` and `table_exists_here`. No DSN mode — a
second connection mode is a second set of semantics. Every function takes the schema explicitly.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

from .environment import is_local_environment

logger = logging.getLogger(__name__)

# Checkpoint rows come from a shared database — a poisoning surface. Strict msgpack restricts
# deserialization to plain types and LangChain serializables, so a tampered row cannot execute
# code on load. `setdefault`, so an operator can still widen it deliberately.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

# The governance tables (audit sink, config store) share one small pool.
_audit_pool = None


def _pool_kwargs() -> dict:
    """Pool settings that make a long-idle replica survive its own connections.

    Lakebase closes idle connections silently (the socket looks open), so `check` validates on
    borrow, `max_idle` retires after 5 min, and `max_lifetime` (30 min) stays inside the 60-min
    OAuth credential's life. TCP keepalives do not detect a politely-closed peer.
    """
    kwargs: dict = {"max_idle": 300.0, "max_lifetime": 1800.0}
    try:
        from psycopg_pool import ConnectionPool

        kwargs["check"] = ConnectionPool.check_connection
    except Exception:
        logger.warning("psycopg_pool unavailable — pooled connections will not be checked")
    return kwargs


def _new_pool(*, min_size: int, max_size: int, **target):
    """A `LakebasePool` for `target`, with its schema guaranteed to exist.

    `LakebasePool(schema=...)` sets `search_path` but does not create the schema, and the
    saver/store `setup()` is not reliably the first writer. Idempotent; skipped for `public`.
    """
    from databricks_ai_bridge.lakebase import LakebaseClient, LakebasePool

    pool = LakebasePool(**target, min_size=min_size, max_size=max_size, **_pool_kwargs())
    if target.get("schema") not in (None, "", "public"):
        LakebaseClient.create_schema(pool)
    return pool


def lakebase_instance() -> Optional[str]:
    """The provisioned Lakebase instance name, when one is configured."""
    return os.getenv("LAKEBASE_INSTANCE") or None


def lakebase_target(schema: Optional[str] = None) -> Optional[dict]:
    """Keyword arguments addressing the configured Lakebase database, or None.

    `{"instance_name"}` (provisioned), `{"autoscaling_endpoint"}` or `{"project", "branch"}`
    (Autoscaling); autoscaling keys win when both are set (an operator migrating forward).
    `schema` falls back to `LAKEBASE_SCHEMA`; deployed callers should always pass one.
    """
    endpoint = os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT") or None
    if endpoint:
        target: dict = {"autoscaling_endpoint": endpoint}
    else:
        project = os.getenv("LAKEBASE_PROJECT") or None
        branch = os.getenv("LAKEBASE_BRANCH") or None
        if project:
            target = {"project": project}
            if branch:
                target["branch"] = branch
        elif lakebase_instance():
            target = {"instance_name": lakebase_instance()}
        else:
            return None

    schema = schema or os.getenv("LAKEBASE_SCHEMA") or None
    if schema:
        target["schema"] = schema
    return target


def _label(target: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in target.items())


def _setup_tolerating_races(obj, what: str) -> None:
    """Run `.setup()`, tolerating the concurrent-worker DDL race.

    `CREATE TABLE IF NOT EXISTS` is not atomic across sessions: concurrent workers race on the
    catalog's unique indexes and losers get `UniqueViolation`. The retry finds the work done.
    """
    from psycopg.errors import UniqueViolation

    for attempt in (1, 2, 3):
        try:
            obj.setup()
            return
        except UniqueViolation:
            if attempt == 3:
                raise
            logger.info("%s setup raced a concurrent worker (attempt %d) — retrying", what, attempt)
            time.sleep(1.5 * attempt)


def refuse_non_durable(what: str) -> None:
    """Raise instead of degrading, anywhere a fallback would lose durability.

    Off a workstation a failed durable store is fatal — **`dev` counts as deployed**.
    `ENVIRONMENT=local` is the workstation value; `MEMORY_ALLOW_INMEMORY=true` is break-glass.
    """
    if is_local_environment() or os.getenv("MEMORY_ALLOW_INMEMORY", "").lower() == "true":
        return
    raise RuntimeError(
        f"{what} could not be initialised and this is a deployed environment "
        f"(ENVIRONMENT={os.getenv('ENVIRONMENT', '')!r}) — refusing to degrade to "
        "in-memory state. Fix the store; set ENVIRONMENT=local if this is a "
        "workstation run; or set MEMORY_ALLOW_INMEMORY=true to accept losing "
        "conversations and approvals on restart."
    )


def build_checkpointer(schema: Optional[str] = None):
    """Short-term memory: conversation state, keyed by thread_id."""
    target = lakebase_target(schema)
    if target:
        try:
            from databricks_langchain import CheckpointSaver

            saver = CheckpointSaver(**target, **_pool_kwargs())
            _setup_tolerating_races(saver, "checkpointer")
            logger.info("checkpointer: Lakebase %s (pooled, rotating credentials)", _label(target))
            return saver
        except Exception:
            logger.exception("Lakebase checkpointer failed for %s", _label(target))
            refuse_non_durable(f"the Lakebase checkpointer ({_label(target)})")
    else:
        refuse_non_durable("conversation durability (no Lakebase instance configured)")

    from langgraph.checkpoint.memory import InMemorySaver

    logger.warning(
        "no Lakebase instance configured — using an in-memory checkpointer. "
        "Conversation history will be lost when this process restarts."
    )
    return InMemorySaver()


def build_store(schema: Optional[str] = None):
    """Long-term memory: per-user context reused across conversations.

    No semantic index and no in-band TTL: a handful of identifiers per user, read by exact key.
    Retention is the scheduled purge's job.
    """
    target = lakebase_target(schema)
    if target:
        try:
            from databricks_langchain import DatabricksStore

            store = DatabricksStore(**target, **_pool_kwargs())
            _setup_tolerating_races(store, "store")
            logger.info("store: Lakebase %s (pooled, rotating credentials)", _label(target))
            return store
        except Exception:
            logger.exception("Lakebase store failed for %s", _label(target))
            refuse_non_durable(f"the Lakebase store ({_label(target)})")
    else:
        refuse_non_durable("long-term memory durability (no Lakebase instance configured)")

    from langgraph.store.memory import InMemoryStore

    return InMemoryStore()


def audit_connection_source(schema: Optional[str] = None):
    """A zero-arg callable yielding a context-managed connection for the governance tables
    (audit sink, config store), or None when no Lakebase is configured.

    Borrows from a small dedicated pool, built on first use.
    """
    target = lakebase_target(schema)
    if not target:
        return None
    global _audit_pool
    if _audit_pool is None:
        try:
            _audit_pool = _new_pool(min_size=1, max_size=2, **target)
        except Exception:
            logger.exception("Lakebase audit pool failed for %s", _label(target))
            return None
    return _audit_pool.connection  # zero-arg context manager


# ---------------------------------------------------------------------------
# SQL identifiers: SQL cannot parameterize an *identifier*, so a table name reaches a statement
# as text — the injection shape. Validated once on the way in; every interpolation rests on it.
# ---------------------------------------------------------------------------
# Letters, digits, underscores, up to three dot-separated parts — nothing that can end the
# identifier and start a new clause.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}$")


class UnsafeIdentifier(ValueError):
    """A table or column name that must not be interpolated into SQL."""


def safe_identifier(name: str, *, what: str = "table") -> str:
    """Return `name` if it is a plain SQL identifier, else raise."""
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise UnsafeIdentifier(f"unsafe {what} identifier: {name!r}")
    return name


# The schema an unqualified CREATE would write to — which is not the schema an
# unqualified SELECT would read from. See `table_exists_here`.
_TABLE_PRESENT_SQL = (
    "SELECT to_regclass(quote_ident(current_schema()) || '.' || quote_ident(%s)) "
    "IS NOT NULL AS present"
)


def table_exists_here(cur, table: str) -> bool:
    """Does `table` exist in the schema this connection would *create* it in?

    Not `to_regclass('<table>')`: an unqualified reference resolves to the first schema on
    `search_path` holding the name, but an unqualified CREATE targets `current_schema()`. A
    same-named table left in `public` would otherwise silently capture every read and write.
    """
    cur.execute(_TABLE_PRESENT_SQL, (table,))
    row = cur.fetchone()
    return bool(row["present"] if isinstance(row, dict) else row[0])
