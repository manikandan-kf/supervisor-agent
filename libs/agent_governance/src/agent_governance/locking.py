"""One execution per conversation thread at a time.

Model Serving routes nothing on the payload, so two turns on one conversation (double-submit,
retry, two tabs) both load the checkpoint and both write back; the loser silently overwrites the
winner. Postgres advisory locks, not a lease table: session-held, so a killed replica releases at
once, while any lease TTL either wedges a conversation or expires under a slow worker call. Keys
are 64-bit and namespaced (32 bits collide by ~77k conversations). Contention fails closed after
`timeout`; plumbing failure fails open with a warning — pool sizing must not refuse governed
traffic. With no connection source the lock is per-process only (the local dev shape).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import threading
import time
from typing import Any, Callable, ContextManager, Iterator, Optional, cast

logger = logging.getLogger(__name__)

_local_locks: dict[str, threading.Lock] = {}
_local_guard = threading.Lock()


def thread_key(thread_id: str, namespace: str = "agent") -> int:
    """A stable signed 64-bit advisory-lock key for one thread id.

    Signed because `pg_advisory_lock` takes `bigint`. A collision only serializes two unrelated
    conversations (latency, not correctness); `namespace` keeps agents sharing an instance apart.
    """
    digest = hashlib.sha256(f"{namespace}\x00{thread_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def _local_lock(thread_id: str) -> threading.Lock:
    with _local_guard:
        lock = _local_locks.get(thread_id)
        if lock is None:
            lock = threading.Lock()
            _local_locks[thread_id] = lock
        return lock


@contextlib.contextmanager
def _in_process(thread_id: str, timeout: float) -> Iterator[bool]:
    lock = _local_lock(thread_id)
    acquired = lock.acquire(timeout=max(0.0, timeout))
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()


def _acquire(conn, objid: int, timeout: float, poll: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s) AS locked", (objid,))
            row = cur.fetchone()
            if bool(row["locked"] if isinstance(row, dict) else row[0]):
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)


def _release(conn, objid: int, thread_id: str) -> None:
    """Release the advisory lock, or destroy the connection holding it.

    The lock is session-scoped and `psycopg_pool` only rolls back on return, so a leaked lock
    would be inherited by the next borrower and wedge that thread for the life of the process;
    if the unlock fails, closing the connection is what guarantees Postgres drops it.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (objid,))
    except Exception:
        logger.exception(
            "failed to release the advisory lock for thread %s — closing the connection "
            "so the lock cannot outlive it",
            thread_id,
        )
        with contextlib.suppress(Exception):
            conn.close()


@contextlib.contextmanager
def thread_lock(
    thread_id: str,
    *,
    connection_source: Optional[Callable[[], object]] = None,
    enabled: bool = True,
    timeout: float = 15.0,
    poll: float = 0.25,
    namespace: str = "agent",
) -> Iterator[bool]:
    """Hold the execution lock for one conversation. Yields whether it was taken.

    `connection_source` is a zero-arg callable yielding a context-managed Postgres connection
    (`lakebase.lock_connection_source`), or None for the per-process fallback. A caller handed
    `False` must not run the graph and should answer with a busy message instead.
    """
    if not enabled or not thread_id:
        yield True
        return

    if connection_source is None:
        with _in_process(thread_id, timeout) as acquired:
            yield acquired
        return

    objid = thread_key(thread_id, namespace)
    with contextlib.ExitStack() as stack:
        try:
            # `connection_source` is caller-supplied and untyped; the cast documents the
            # contract the docstring states rather than widening it.
            conn: Any = stack.enter_context(cast(ContextManager[Any], connection_source()))
            acquired = _acquire(conn, objid, timeout, poll)
        except Exception:
            # Plumbing, not contention. Fail open — see the module docstring.
            logger.warning(
                "thread serialization unavailable for %s; running unserialized",
                thread_id,
                exc_info=True,
            )
            yield True
            return

        if acquired:
            # Registered after the connection, so it unwinds first: release the
            # lock, then hand the connection back to the pool.
            stack.callback(_release, conn, objid, thread_id)
        yield acquired
