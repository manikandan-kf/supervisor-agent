"""One execution per conversation thread at a time (framework §4.4).

§4.4 requires that "only one execution updates a thread at a time". Nothing
enforced that: two turns arriving on one conversation — a double-submit, an
impatient retry, two browser tabs on the same chat — both loaded the same
checkpoint, both ran the gates, and both wrote back. The loser's write silently
replaced the winner's, so a clarification could be answered against a state that
no longer existed and an approval gate could be resolved twice.

**Postgres advisory locks, not a lease table.** An advisory lock is held by the
database *session*, which makes the crash behaviour correct for free: if a
serving replica is killed mid-turn its connection drops and Postgres releases the
lock immediately. A lease row would need a TTL, and any TTL is either long enough
to wedge a conversation after a crash or short enough to expire under a slow
worker call. The lock costs one round-trip against a turn that already spends
seconds in model calls.

**64-bit key space, project-namespaced.** The single-bigint
`pg_try_advisory_lock(key)` form, with the key derived from a project tag plus
the thread id. The earlier two-integer form left only 32 bits for the thread —
a birthday bound of ~50% collision by ~77,000 distinct conversations, so at
enterprise scale colliding pairs would be constant and each one silently
serializes two unrelated users' turns against each other. 64 bits pushes that
bound past five billion conversations; the tag prefix keeps this project's keys
disjoint from any other advisory-lock user sharing the instance in expectation,
which is what the old `classid` bought.

**Under contention** the second turn polls for up to
`THREAD_LOCK_TIMEOUT_SECONDS` and is then refused with `BUSY_MESSAGE` rather than
run concurrently. A refusal the user can act on ("your previous message is still
working") is strictly better than two turns racing to overwrite each other.

**Where it degrades, stated plainly.** With no Postgres configured (local dev, the
in-memory fallbacks) the lock is a per-process `threading.Lock`: it covers
concurrent requests inside one worker process, not the several worker processes a
serving replica runs. That is a real partial guarantee, and it is the same shape
as the in-memory checkpointer it arrives with — with no shared database there is
nothing to serialize *through*. The Postgres path is the deployed path.

Acquisition failing for an *infrastructural* reason (pool exhausted, connection
error) **fails open** with a warning, and the turn runs unserialized. Deliberate:
failing closed would let this project's own pool sizing refuse governed traffic,
and the condition being protected against — two turns on the *same* conversation
— is rare next to the many-threads-one-pool case that would trip it. Contention
itself always fails closed; only the plumbing fails open.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import threading
import time
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

# The project tag mixed into every key, so this project's advisory locks live
# in their own region of the 64-bit space rather than wherever a bare thread-id
# hash happens to land.
_LOCK_NAMESPACE = b"SUPV\x00"

_local_locks: dict[str, threading.Lock] = {}
_local_guard = threading.Lock()

BUSY_MESSAGE = (
    "Another message on this conversation is still being processed. Please wait for it "
    "to finish before sending the next one — nothing you sent has been lost."
)


def thread_key(thread_id: str) -> int:
    """A stable signed 64-bit advisory-lock key for one thread id.

    Signed, because `pg_advisory_lock(key bigint)` takes `bigint`. A collision
    still only serializes two unrelated conversations against each other —
    latency, never correctness — but at 64 bits that stays a curiosity instead
    of a fleet-wide constant (the 32-bit form crossed 50% collision probability
    at ~77k conversations; this one at ~5 billion).
    """
    digest = hashlib.sha256(_LOCK_NAMESPACE + thread_id.encode("utf-8")).digest()
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
    enabled: bool = True,
    timeout: float = 15.0,
    poll: float = 0.25,
    connection_source: Optional[object] = None,
) -> Iterator[bool]:
    """Hold the execution lock for one conversation. Yields whether it was taken.

    A caller handed `False` must not run the graph — that is the entire point —
    and should return `BUSY_MESSAGE` instead.
    """
    if not enabled or not thread_id:
        yield True
        return

    source = connection_source
    if source is None:
        try:
            from .memory import lock_connection_source

            source = lock_connection_source()
        except Exception:
            logger.exception("could not build a lock connection source")
            source = None

    if source is None:
        with _in_process(thread_id, timeout) as acquired:
            yield acquired
        return

    objid = thread_key(thread_id)
    with contextlib.ExitStack() as stack:
        try:
            conn = stack.enter_context(source())
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
