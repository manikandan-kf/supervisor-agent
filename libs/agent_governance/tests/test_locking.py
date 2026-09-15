"""The per-conversation execution lock.

Model Serving does not serialize turns by conversation, so the lock is what
stops two turns on one thread racing each other's checkpoint. These tests pin
the key derivation the Postgres path relies on and the contention semantics
both paths share.
"""

from __future__ import annotations

import contextlib
import threading

from agent_governance.locking import thread_key, thread_lock


def test_thread_key_is_a_stable_signed_bigint():
    key = thread_key("thr_abc")
    assert key == thread_key("thr_abc")
    assert -(2**63) <= key < 2**63
    assert key != thread_key("thr_abd")


def test_namespace_separates_two_agents_on_one_instance():
    assert thread_key("thr_abc", "supervisor") != thread_key("thr_abc", "coding-agent")


def test_disabled_or_anonymous_turns_are_never_blocked():
    with thread_lock("thr_1", enabled=False) as acquired:
        assert acquired is True
    with thread_lock("", enabled=True) as acquired:
        assert acquired is True


def test_in_process_fallback_refuses_a_concurrent_turn_on_the_same_thread():
    """With no connection source the lock is per process — and still a lock."""
    holding = threading.Event()
    release = threading.Event()

    def first_turn():
        with thread_lock("thr_2", timeout=1.0) as acquired:
            assert acquired
            holding.set()
            release.wait(5)

    worker = threading.Thread(target=first_turn)
    worker.start()
    holding.wait(5)
    try:
        with thread_lock("thr_2", timeout=0.1) as acquired:
            assert acquired is False
        with thread_lock("thr_other", timeout=0.1) as acquired:
            assert acquired is True
    finally:
        release.set()
        worker.join(5)


class _Cursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._conn.statements.append((" ".join(sql.split()), params))

    def fetchone(self):
        sql, _ = self._conn.statements[-1]
        if "pg_try_advisory_lock" in sql:
            return {"locked": self._conn.grant}
        return None


class _Connection:
    def __init__(self, grant: bool):
        self.grant = grant
        self.statements: list = []
        self.closed = False

    def cursor(self):
        return _Cursor(self)

    def close(self):
        self.closed = True


def test_postgres_path_locks_and_unlocks_the_same_key():
    conn = _Connection(grant=True)
    with thread_lock("thr_3", connection_source=lambda: contextlib.nullcontext(conn)) as ok:
        assert ok is True
    statements = [sql for sql, _ in conn.statements]
    assert any("pg_try_advisory_lock" in s for s in statements)
    assert any("pg_advisory_unlock" in s for s in statements)
    keys = {params[0] for _, params in conn.statements}
    assert keys == {thread_key("thr_3")}


def test_postgres_contention_is_refused_and_nothing_is_unlocked():
    conn = _Connection(grant=False)
    with thread_lock(
        "thr_4", connection_source=lambda: contextlib.nullcontext(conn), timeout=0, poll=0
    ) as ok:
        assert ok is False
    assert not any("pg_advisory_unlock" in sql for sql, _ in conn.statements)


def test_a_broken_connection_source_fails_open():
    """Plumbing failure runs the turn unserialized; only contention refuses it."""

    def broken():
        raise RuntimeError("pool exhausted")

    with thread_lock("thr_5", connection_source=broken) as ok:
        assert ok is True
