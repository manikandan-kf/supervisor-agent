"""Lakebase / Postgres plumbing every agent shares.

Pools that survive an idle serving replica, the checkpointer and store
builders, and the connection sources the audit sink, the config store, the
review queue and the thread lock borrow from. Mode is selected by
configuration:

    Mode      Configured by                Checkpointer / Store
    ────────  ───────────────────────────  ─────────────────────────────────────
    Lakebase  LAKEBASE_INSTANCE, or        databricks_langchain CheckpointSaver /
              LAKEBASE_AUTOSCALING_        DatabricksStore — pooled connections
              ENDPOINT (or LAKEBASE_       with OAuth tokens rotated automatically
              PROJECT + LAKEBASE_BRANCH)
    Fallback  neither                      InMemorySaver / InMemoryStore (local only)

The Lakebase pools mint a fresh M2M OAuth credential per connection (cached ~15
minutes, recycled before expiry), which is the rotation a long-running Model
Serving replica needs; a hand-built DSN with an embedded token goes stale within
the hour. There is deliberately no "any Postgres by DSN" mode: the deployed
path is Lakebase, a workstation that needs durable state points
`LAKEBASE_INSTANCE` at the dev instance with the developer's own Databricks
credentials, and a second connection mode was a second set of semantics
(shared long-lived connection, re-entrant advisory locks) to keep correct.

**Every function takes the Postgres schema explicitly.** The schema is what
separates one deployed environment's durable state from another's when they
share an instance, and its *default* is an agent-level decision (the supervisor
derives `supervisor_<env>`), so the library never guesses one. A schema has to
*exist* before anything writes, or the separation collapses silently: Postgres
accepts a `search_path` naming a schema that does not exist and an unqualified
`CREATE TABLE` then lands in `public`. `_new_pool` therefore creates the schema
on every pool this module builds.

With nothing configured both builders fall back to in-memory implementations.
That is a *local* convenience with a real cost — history dies with the process
and replicas share nothing — so outside a workstation it is refused rather than
logged. See `refuse_non_durable`.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from .environment import is_local_environment

logger = logging.getLogger(__name__)

# Checkpoint rows come back from a shared database, which is a poisoning
# surface. LangGraph's serializer will, by default, reconstruct arbitrary
# importable classes referenced in a checkpoint; strict mode restricts
# deserialization to plain types and LangChain's own serializable classes, so a
# row tampered with in Postgres cannot execute code on load. `setdefault`, so an
# operator can still widen it deliberately from the environment.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

# One small pool for the governance tables (audit sink, config store, review
# queue), and a separate one for the thread lock: a lock holds its connection
# for the whole turn, and borrowing that from the two-connection audit pool
# would starve the audit write at the end of the same turn.
_audit_pool = None
_lock_pool = None


def _pool_kwargs() -> dict:
    """Pool settings that make a long-idle replica survive its own connections.

    Lakebase closes an idle connection (and an Autoscaling endpoint scales its
    compute to zero) while the pool has no way to notice: the socket looks open
    until something is written to it, so the *next* borrower is handed a corpse
    and fails with `SSL error: unexpected eof while reading` before the graph
    runs a single node. TCP keepalives detect a silently vanished peer, not one
    that closed politely while the pool was idle.

      * `check` — validate a connection when it is borrowed; psycopg_pool then
        discards a dead one and opens a replacement. One round-trip per borrow,
        noise next to the model call in the same turn.
      * `max_idle` — retire an idle connection after 5 minutes, inside any
        server-side idle timeout.
      * `max_lifetime` — recycle every connection after 30 minutes, comfortably
        younger than the 60-minute OAuth credential it was opened with.
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

    `LakebasePool(schema=...)` sets `search_path` and stops there — it does not
    create the schema. `CheckpointSaver.setup()` and `DatabricksStore.setup()`
    do, but they are not the only writers and not reliably the first: the audit
    sink, the config store and the thread lock all borrow from the bare pools
    built here. Idempotent, and skipped for `public`, which always exists and
    whose owner is not the serving identity.
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

    One of two shapes, matching the two Lakebase generations
    (`databricks_ai_bridge.lakebase` picks the matching credential API):

        {"instance_name": ...}                       provisioned instance
        {"autoscaling_endpoint": ...}                Autoscaling project
        {"project": ..., "branch": ...}              Autoscaling project, by branch

    The autoscaling keys win when both are set: an operator adding the new
    variables to an environment that still carries the old one is migrating
    forward. `schema` falls back to `LAKEBASE_SCHEMA`; callers on a deployed
    path should always pass one (see the module docstring).
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

    A serving replica boots several worker processes at once and each runs
    setup() against the same schema. `CREATE TABLE IF NOT EXISTS` is not atomic
    across sessions: concurrent creators race on the catalog's unique indexes
    and every loser gets `UniqueViolation`. That means another worker is
    creating exactly what this one needs, so wait and rerun; the retry finds the
    work done and no-ops.
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

    Outside a workstation a configured-but-failed durable store is fatal: Model
    Serving restarts the replica, and a replica that cannot reach its store does
    not pretend otherwise. **`dev` counts as deployed.** `ENVIRONMENT=local` is
    the workstation value; `MEMORY_ALLOW_INMEMORY=true` is the explicit,
    visible override for break-glass debugging on a deployed replica.
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

    No semantic-search index and no in-band TTL: what this store holds is a
    handful of validated identifiers per user, read back by exact key.
    Retention is the scheduled purge's job, not a read-path TTL.
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
    """A zero-arg callable yielding a context-managed connection for the
    governance tables (audit sink, config store, review queue), or None when
    no Lakebase instance is configured.

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


def lock_connection_source(schema: Optional[str] = None):
    """A connection source for the per-thread execution lock, or None.

    Its own pool, sized for concurrency: a lock is held for the duration of a
    turn, so N in-flight turns need N connections. `max_size=6` bounds what one
    worker process can take from the instance. None — no Lakebase configured —
    sends `locking.thread_lock` to its per-process fallback.
    """
    target = lakebase_target(schema)
    if not target:
        return None
    global _lock_pool
    if _lock_pool is None:
        try:
            _lock_pool = _new_pool(min_size=1, max_size=6, **target)
        except Exception:
            logger.exception("Lakebase lock pool failed for %s", _label(target))
            return None
    return _lock_pool.connection
