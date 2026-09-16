"""Erasure and retention — making deletion operable rather than theoretical.

GDPR Art. 17 erasure and storage retention both come down to deleting Postgres rows and showing
afterwards that you did; nothing else here deletes (session expiry and memory TTL are lazy on
read). Three properties make this a control rather than a console `DELETE`: it writes evidence
through the same `log()` that computes the audit hash chain (a hand-inserted row without its
digest breaks every verification after it); it re-reads before it reports; and it defaults to a
dry run. Run as the table owner: the §7b grants deliberately leave the runtime without DELETE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .lakebase import safe_identifier

logger = logging.getLogger(__name__)

# The LangGraph Postgres checkpointer's tables, most dependent first so a
# partial failure cannot orphan a child row.
CHECKPOINT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")

# The long-term store's table and the namespace the supervisor writes under.
STORE_TABLE = "store"
STORE_PREFIX = "supervisor.resolved_context"


@dataclass(frozen=True)
class Erasure:
    """What an erasure removed, or would remove."""

    user_key: str
    applied: bool
    memory_rows: int = 0
    conversations: tuple[str, ...] = ()
    checkpoint_rows: int = 0
    residual: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        """Applied, and a re-read found nothing left."""
        return self.applied and not self.residual

    def audit_detail(self) -> str:
        verb = "erased" if self.applied else "would erase"
        return (
            f"{verb} {self.memory_rows} memory row(s) and {self.checkpoint_rows} "
            f"checkpoint row(s) across {len(self.conversations)} conversation(s)"
            + (f"; RESIDUAL: {', '.join(self.residual)}" if self.residual else "")
        )


@dataclass(frozen=True)
class Sweep:
    """What a retention sweep removed, or would remove."""

    older_than_days: int
    applied: bool
    conversations: tuple[str, ...] = ()
    checkpoint_rows: int = 0

    def audit_detail(self) -> str:
        verb = "purged" if self.applied else "would purge"
        return (
            f"{verb} {self.checkpoint_rows} checkpoint row(s) across "
            f"{len(self.conversations)} conversation(s) idle over "
            f"{self.older_than_days} day(s)"
        )


def _scalar(cur, sql, params=()):
    cur.execute(sql, params)
    row = cur.fetchone()
    if row is None:
        return 0
    return list(row.values())[0] if isinstance(row, dict) else row[0]


def _column(cur, sql, params=()) -> list[str]:
    cur.execute(sql, params)
    return [
        str(list(r.values())[0] if isinstance(r, dict) else r[0]) for r in (cur.fetchall() or [])
    ]


def _delete_threads(cur, conversations) -> int:
    """Remove every checkpoint row for these threads. Returns rows deleted."""
    if not conversations:
        return 0
    removed = 0
    for table in CHECKPOINT_TABLES:
        name = safe_identifier(table)
        cur.execute(
            f"DELETE FROM {name} WHERE thread_id = ANY(%s)",  # noqa: S608 — validated identifier
            (list(conversations),),
        )
        removed += max(0, int(getattr(cur, "rowcount", 0) or 0))
    return removed


def conversations_for(cur, audit_table: str, user_key: str) -> list[str]:
    """Thread ids this subject appears in, from the audit trail.

    The checkpoint tables carry no subject column, so the audit trail is the only link between a
    person and their conversations: a deployment without the Postgres audit sink cannot resolve
    an erasure request at all.
    """
    name = safe_identifier(audit_table)
    return _column(
        cur,
        f"SELECT DISTINCT conversation_id FROM {name} "  # noqa: S608 — validated identifier
        "WHERE user_key = %s AND conversation_id <> ''",
        (user_key,),
    )


def erase_subject(
    connection_source,
    user_key: str,
    audit_table: str,
    *,
    apply: bool = False,
    memory_only: bool = False,
    audit_logger=None,
    requested_by: str = "",
) -> Erasure:
    """Erase one subject's long-term memory and conversation checkpoints.

    `apply=False` (default) reports what would go and changes nothing. Audit rows are deliberately
    kept: they carry no message text, they are the record that the subject's requests were
    governed, and removing them would break the hash chain for every row after them.
    """
    if not user_key:
        raise ValueError("user_key is required")

    with connection_source() as conn:
        with conn.cursor() as cur:
            store = safe_identifier(STORE_TABLE)
            memory_rows = int(
                _scalar(
                    cur,
                    f"SELECT count(*) FROM {store} WHERE prefix = %s AND key = %s",  # noqa: S608
                    (STORE_PREFIX, user_key),
                )
                or 0
            )
            conversations = [] if memory_only else conversations_for(cur, audit_table, user_key)

            if not apply:
                return Erasure(
                    user_key=user_key,
                    applied=False,
                    memory_rows=memory_rows,
                    conversations=tuple(conversations),
                )

            cur.execute(
                f"DELETE FROM {store} WHERE prefix = %s AND key = %s",  # noqa: S608
                (STORE_PREFIX, user_key),
            )
            checkpoint_rows = _delete_threads(cur, conversations)

            # Verify before reporting — a re-read, not the delete's own word.
            residual = []
            if (
                int(
                    _scalar(
                        cur,
                        f"SELECT count(*) FROM {store} WHERE prefix = %s AND key = %s",  # noqa: S608
                        (STORE_PREFIX, user_key),
                    )
                    or 0
                )
                > 0
            ):
                residual.append("long-term memory")
            for table in CHECKPOINT_TABLES if conversations else ():
                name = safe_identifier(table)
                if (
                    int(
                        _scalar(
                            cur,
                            f"SELECT count(*) FROM {name} WHERE thread_id = ANY(%s)",  # noqa: S608
                            (list(conversations),),
                        )
                        or 0
                    )
                    > 0
                ):
                    residual.append(table)

    result = Erasure(
        user_key=user_key,
        applied=True,
        memory_rows=memory_rows,
        conversations=tuple(conversations),
        checkpoint_rows=checkpoint_rows,
        residual=tuple(residual),
    )
    _record(audit_logger, "erasure", user_key, requested_by, result.audit_detail())
    if residual:
        raise RuntimeError(
            f"erasure incomplete for {user_key}: data remains in {', '.join(residual)}"
        )
    return result


def sweep_conversations(
    connection_source,
    audit_table: str,
    older_than_days: int,
    *,
    apply: bool = False,
    audit_logger=None,
    requested_by: str = "",
) -> Sweep:
    """Purge checkpoints for conversations idle longer than `older_than_days`.

    Idleness comes from the audit trail's `event_time`: the checkpoint tables carry no timestamp,
    and inferring one from a checkpoint id would couple this to a LangGraph detail free to change.
    A conversation with no audit row is never swept — the conservative direction.
    """
    if older_than_days <= 0:
        raise ValueError("older_than_days must be positive")
    name = safe_identifier(audit_table)

    with connection_source() as conn:
        with conn.cursor() as cur:
            stale = _column(
                cur,
                f"SELECT conversation_id FROM {name} "  # noqa: S608 — validated identifier
                "WHERE conversation_id <> '' "
                "GROUP BY conversation_id "
                "HAVING max(event_time) < now() - make_interval(days => %s)",
                (int(older_than_days),),
            )
            if not apply:
                return Sweep(older_than_days, False, tuple(stale))
            removed = _delete_threads(cur, stale)

    result = Sweep(older_than_days, True, tuple(stale), removed)
    _record(audit_logger, "retention", "", requested_by, result.audit_detail())
    return result


def _record(audit_logger, outcome: str, user_key: str, requested_by: str, detail: str) -> None:
    """Write the deletion through the runtime's own sink, chain and all."""
    if audit_logger is None:
        logger.warning("%s not recorded in the audit trail: no sink supplied — %s", outcome, detail)
        return
    try:
        audit_logger.log(
            {
                "outcome": outcome,
                "user_key": user_key,
                "user_role": requested_by or "operator",
                "decision_trail": [
                    {"stage": outcome, "decision": "applied", "detail": detail},
                ],
            }
        )
    except Exception:
        # Loud, not fatal: the rows are already gone and an operator can record it another way;
        # swallowing it would leave an unevidenced deletion, the one outcome this module prevents.
        logger.error("%s applied but NOT recorded in the audit trail — %s", outcome, detail)


def default_audit_logger(connection_source, audit_table: str) -> Optional[object]:
    """The Postgres sink, for callers that want erasures chained into the trail."""
    from .audit import PostgresAuditLogger

    if connection_source is None:
        return None
    return PostgresAuditLogger(connection_source, audit_table)
