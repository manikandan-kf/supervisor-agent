"""One execution per conversation thread at a time.

Model Serving does not serialize requests by conversation. An endpoint runs
several replicas, each replica several worker processes, and every request
goes to whichever has capacity — nothing routes on the payload. So two turns
arriving on one conversation (a double-submit, an impatient retry, two browser
tabs on the same chat) both load the same checkpoint, both run the gates, and
both write back. The loser's write silently replaces the winner's, so a
clarification can be answered against a state that no longer exists and an
approval gate can be resolved twice. LangGraph Platform enforces one run per
thread for its own deployments ("double texting" — reject, enqueue, interrupt
or rollback); for a graph served anywhere else nothing does, which is why
every agent on this platform takes this lock around its graph run.

**Postgres advisory locks, not a lease table.** An advisory lock is held by the
database *session*, which makes the crash behaviour correct for free: if a
serving replica is killed mid-turn its connection drops and Postgres releases
the lock immediately. A lease row would need a TTL, and any TTL is either long
enough to wedge a conversation after a crash or short enough to expire under a
slow worker call. The lock costs one round-trip against a turn that already
spends seconds in model calls.

**64-bit key space, namespaced.** The single-bigint `pg_try_advisory_lock(key)`
form, with the key derived from a namespace plus the thread id. A two-integer
form would leave 32 bits for the thread — a birthday bound of ~50% collision by
~77,000 distinct conversations, so at enterprise scale colliding pairs would be
constant and each one silently serializes two unrelated users' turns against
each other. 64 bits pushes that bound past five billion conversations; the
namespace keeps one agent's keys disjoint in expectation from any other
advisory-lock user sharing the instance.

**Under contention** the second turn polls for up to `timeout` seconds and is
then refused rather than run concurrently: `thread_lock` yields `False`, and
the caller answers with its own busy message. A refusal the user can act on
("your previous message is still working") is strictly better than two turns
racing to overwrite each other.

**Where it degrades, stated plainly.** With no connection source (local
development, the in-memory fallbacks) the lock is a per-process
`threading.Lock`: it covers concurrent requests inside one worker process, not
the several worker processes a serving replica runs. That is the same shape as
the in-memory checkpointer it arrives with — with no shared database there is
nothing to serialize *through*. The Postgres path is the deployed path.

Acquisition failing for an *infrastructural* reason (pool exhausted, connection
error) **fails open** with a warning, and the turn runs unserialized.
Deliberate: failing closed would let an agent's own pool sizing refuse governed
traffic, and the condition being protected against — two turns on the *same*
conversation — is rare next to the many-threads-one-pool case that would trip
it. Contention itself always fails closed; only the plumbing fails open.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import threading
import time
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

_local_locks: dict[str, threading.Lock] = {}
_local_guard = threading.Lock()


def thread_key(thread_id: str, namespace: str = "agent") -> int:
    """A stable signed 64-bit advisory-lock key for one thread id.

    Signed, because `pg_advisory_lock(key bigint)` takes `bigint`. A collision
    still only serializes two unrelated conversations against each other —
    latency, never correctness — but at 64 bits that stays a curiosity instead
    of a fleet-wide constant. `namespace` is the agent's tag, mixed in so that
    two agents sharing one Lakebase instance live in different regions of the
    key space.
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

    Mandatory, not tidiness: the lock is session-scoped and `psycopg_pool` only
    rolls back on return, which does not release it. A leaked lock would be
    inherited by the next borrower of this pooled connection and wedge that
    thread for the life of the process — so if the unlock statement itself
    fails, the connection is closed instead. Ending the session is what
    guarantees Postgres drops everything it held.
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

    `connection_source` is a zero-arg callable yielding a context-managed
    Postgres connection (`lakebase.lock_connection_source`), or None for the
    per-process fallback. A caller handed `False` must not run the graph — that
    is the entire point — and should answer with a busy message instead.
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
            conn = stack.enter_context(connection_source())
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
