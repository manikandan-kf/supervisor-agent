"""The appeal and escalation queue — Governance Blueprint §05 Stage 03 / 04.

An appeal route is human intervention only if a human can enumerate open appeals (`list_open`)
and `resolve` them, and the outcome changes what follows: `uphold`, or `allow_retry` granting
one more pass, spent exactly once by `claim_allowance`'s conditional `UPDATE ... RETURNING`.
Both halves live here — `open_review`/`claim_allowance` run in an agent's graph, the reviewer
calls the rest — and the queue is consulted only when a review is already open, so an outage
fails closed for conversations that are already escalated and affects no one else.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .lakebase import safe_identifier, table_exists_here
from .sanitize import clean_worker_output
from .sensitive import redact_text

logger = logging.getLogger(__name__)

OPEN = "open"
RESOLVED = "resolved"

APPEAL = "appeal"
ESCALATION = "escalation"

UPHOLD = "uphold"
ALLOW_RETRY = "allow_retry"

KINDS = (APPEAL, ESCALATION)
DECISIONS = (UPHOLD, ALLOW_RETRY)

_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
  ref             TEXT        PRIMARY KEY,
  kind            TEXT        NOT NULL,
  status          TEXT        NOT NULL DEFAULT 'open',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  conversation_id TEXT        NOT NULL,
  request_id      TEXT,
  correlation_id  TEXT,
  user_key        TEXT,
  user_role       TEXT,
  target_agent_id TEXT,
  reason          TEXT,
  query_excerpt   TEXT,
  decision        TEXT,
  reviewer        TEXT,
  reviewer_note   TEXT,
  resolved_at     TIMESTAMPTZ,
  consumed_at     TIMESTAMPTZ
)
"""

# A reviewer asks "what is open, oldest first"; an agent asks "is anything open for *this*
# conversation". One index each; the partial index keeps the reviewer query off resolved rows.
_INDEXES = (
    "CREATE INDEX IF NOT EXISTS {table}_open_idx ON {table} (created_at) WHERE status = 'open'",
    "CREATE INDEX IF NOT EXISTS {table}_conversation_idx "
    "ON {table} (conversation_id, created_at DESC)",
)


class ReviewQueueError(RuntimeError):
    """The queue could not be reached or the write did not land."""


@dataclass(frozen=True)
class Review:
    ref: str
    kind: str
    status: str
    conversation_id: str
    reason: str = ""
    created_at: Optional[datetime] = None
    request_id: str = ""
    correlation_id: str = ""
    user_key: str = ""
    user_role: str = ""
    target_agent_id: str = ""
    query_excerpt: str = ""
    decision: str = ""
    reviewer: str = ""
    reviewer_note: str = ""
    resolved_at: Optional[datetime] = None
    consumed_at: Optional[datetime] = None

    @property
    def open(self) -> bool:
        return self.status == OPEN

    @property
    def grants_retry(self) -> bool:
        """Resolved in the user's favour, and not yet spent."""
        return self.status == RESOLVED and self.decision == ALLOW_RETRY and self.consumed_at is None

    def public(self) -> dict:
        """The reviewer-facing shape. No raw identifiers — `user_key` is already
        the pseudonymous reference the gateway derived (§1.10)."""
        return {
            "ref": self.ref,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "conversation_id": self.conversation_id,
            "user_reference": self.user_key,
            "user_role": self.user_role,
            "target_agent_id": self.target_agent_id,
            "reason": self.reason,
            "query_excerpt": self.query_excerpt,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "reviewer_note": self.reviewer_note,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else "",
        }


_COLUMNS = (
    "ref, kind, status, created_at, conversation_id, request_id, correlation_id, "
    "user_key, user_role, target_agent_id, reason, query_excerpt, decision, "
    "reviewer, reviewer_note, resolved_at, consumed_at"
)


def _row_to_review(row) -> Review:
    """Build a `Review` from either a dict-shaped or tuple-shaped cursor row.

    The Lakebase pools use a dict row factory; a test double or another psycopg config may not.
    """
    names = [c.strip() for c in _COLUMNS.split(",")]
    if isinstance(row, dict):
        values = {name: row.get(name) for name in names}
    else:
        values = dict(zip(names, row, strict=False))

    def text(name: str) -> str:
        value = values.get(name)
        return "" if value is None else str(value)

    return Review(
        ref=text("ref"),
        kind=text("kind"),
        status=text("status"),
        conversation_id=text("conversation_id"),
        reason=text("reason"),
        created_at=values.get("created_at"),
        request_id=text("request_id"),
        correlation_id=text("correlation_id"),
        user_key=text("user_key"),
        user_role=text("user_role"),
        target_agent_id=text("target_agent_id"),
        query_excerpt=text("query_excerpt"),
        decision=text("decision"),
        reviewer=text("reviewer"),
        reviewer_note=text("reviewer_note"),
        resolved_at=values.get("resolved_at"),
        consumed_at=values.get("consumed_at"),
    )


class ReviewQueue:
    """Postgres-backed appeal and escalation queue.

    Takes a *connection source* — a zero-arg callable yielding a context-managed connection —
    for the same reason `PostgresAuditLogger` does: the pool rotates credentials every ~15 min.
    """

    def __init__(self, connection_source, table: str):
        self._source = connection_source
        # SQL cannot parameterize an identifier. Validated at construction so
        # every interpolation below rests on an enforced property.
        self._table = safe_identifier(table)
        self._ready = False

    # ── plumbing ────────────────────────────────────────────────────────────

    def _ensure_table(self, conn) -> None:
        """Create the table on first use, but only when it is absent: `CREATE INDEX IF NOT
        EXISTS` checks ownership *before* existence, so for any identity that did not create
        the table it raises and aborts the transaction — the normal case in a shared database.
        """
        if self._ready:
            return
        with conn.cursor() as cur:
            # Scoped to the schema the DDL below would write to, not to the whole
            # search_path — see `sql.table_exists_here`.
            if not table_exists_here(cur, self._table):
                cur.execute(_DDL.format(table=self._table))
                for statement in _INDEXES:
                    cur.execute(statement.format(table=self._table))
        self._ready = True

    # ── writes ──────────────────────────────────────────────────────────────

    def open_review(
        self,
        *,
        kind: str,
        conversation_id: str,
        reason: str,
        user_key: str = "",
        user_role: str = "",
        target_agent_id: str = "",
        request_id: str = "",
        correlation_id: str = "",
        query_excerpt: str = "",
    ) -> Review:
        """Record a new open review and return it.

        Raises `ReviewQueueError` if the row did not land; the caller must treat that as a
        failure to escalate — §08: *"Do not report a governance decision as applied."*
        """
        if kind not in KINDS:
            raise ValueError(f"unknown review kind: {kind!r}")
        if not conversation_id:
            raise ValueError("a review must name the conversation it belongs to")

        # Cleaned here at the write rather than trusting every call site: `reason` is the
        # model's raw verdict text and both fields are rendered back to the reviewer.
        # Redaction runs after marker neutralisation, since verdict text often quotes input.
        reason = redact_text(clean_worker_output(reason or "", max_chars=2000).text)[0]
        query_excerpt = redact_text(clean_worker_output(query_excerpt or "", max_chars=2000).text)[
            0
        ]

        ref = f"rev_{uuid.uuid4().hex[:16]}"
        try:
            with self._source() as conn:
                self._ensure_table(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        f"INSERT INTO {self._table} "
                        "(ref, kind, status, conversation_id, request_id, correlation_id, "
                        " user_key, user_role, target_agent_id, reason, query_excerpt) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            ref,
                            kind,
                            OPEN,
                            conversation_id,
                            request_id,
                            correlation_id,
                            user_key,
                            user_role,
                            target_agent_id,
                            reason[:2000],
                            query_excerpt[:2000],
                        ),
                    )
        except Exception as exc:
            raise ReviewQueueError(f"could not open a {kind}: {exc}") from exc

        logger.info(
            "review opened ref=%s kind=%s conversation=%s agent=%s",
            ref,
            kind,
            conversation_id,
            target_agent_id,
        )
        return Review(
            ref=ref,
            kind=kind,
            status=OPEN,
            conversation_id=conversation_id,
            reason=reason,
            created_at=datetime.now(timezone.utc),
            user_key=user_key,
            user_role=user_role,
            target_agent_id=target_agent_id,
            request_id=request_id,
            correlation_id=correlation_id,
            query_excerpt=query_excerpt,
        )

    def resolve(self, ref: str, *, reviewer: str, decision: str, note: str = "") -> Review:
        """Close one review. Exactly one caller wins.

        `WHERE status = 'open'` is the concurrency control (§10). The loser gets a
        `ReviewQueueError` naming the winner, not a silent overwrite of their decision.
        """
        if decision not in DECISIONS:
            raise ValueError(f"unknown decision: {decision!r}")
        if not reviewer:
            raise ValueError("a resolution must name its reviewer")

        try:
            with self._source() as conn:
                self._ensure_table(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {self._table} SET status = %s, decision = %s, "
                        "reviewer = %s, reviewer_note = %s, resolved_at = now() "
                        f"WHERE ref = %s AND status = %s RETURNING {_COLUMNS}",
                        (RESOLVED, decision, reviewer, note[:2000], ref, OPEN),
                    )
                    row = cur.fetchone()
                    if row is not None:
                        review = _row_to_review(row)
                        logger.info(
                            "review resolved ref=%s decision=%s reviewer=%s",
                            ref,
                            decision,
                            reviewer,
                        )
                        return review

                    # Nothing updated: either the ref does not exist or someone else already
                    # resolved it. Distinguish — the two need different answers.
                    cur.execute(f"SELECT {_COLUMNS} FROM {self._table} WHERE ref = %s", (ref,))
                    existing = cur.fetchone()
        except ReviewQueueError:
            raise
        except Exception as exc:
            raise ReviewQueueError(f"could not resolve {ref}: {exc}") from exc

        if existing is None:
            raise ReviewQueueError(f"no such review: {ref}")
        already = _row_to_review(existing)
        raise ReviewQueueError(
            f"{ref} was already resolved by {already.reviewer or 'another reviewer'} "
            f"as '{already.decision}'"
        )

    def claim_allowance(self, conversation_id: str) -> Optional[Review]:
        """Spend a granted retry for this conversation, if one is waiting.

        One conditional `UPDATE ... RETURNING`, so the allowance is consumed exactly once
        however many turns arrive together; oldest grant first, so none is left stranded.
        """
        if not conversation_id:
            return None
        try:
            with self._source() as conn:
                self._ensure_table(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        f"UPDATE {self._table} SET consumed_at = now() WHERE ref = ("
                        f"  SELECT ref FROM {self._table} "
                        "   WHERE conversation_id = %s AND status = %s "
                        "     AND decision = %s AND consumed_at IS NULL "
                        "   ORDER BY resolved_at ASC LIMIT 1"
                        f") RETURNING {_COLUMNS}",
                        (conversation_id, RESOLVED, ALLOW_RETRY),
                    )
                    row = cur.fetchone()
        except Exception as exc:
            raise ReviewQueueError(
                f"could not check review allowances for {conversation_id}: {exc}"
            ) from exc

        if row is None:
            return None
        review = _row_to_review(row)
        logger.info(
            "review allowance claimed ref=%s conversation=%s reviewer=%s",
            review.ref,
            conversation_id,
            review.reviewer,
        )
        return review

    # ── reads ───────────────────────────────────────────────────────────────

    def get(self, ref: str) -> Optional[Review]:
        try:
            with self._source() as conn:
                self._ensure_table(conn)
                with conn.cursor() as cur:
                    cur.execute(f"SELECT {_COLUMNS} FROM {self._table} WHERE ref = %s", (ref,))
                    row = cur.fetchone()
        except Exception as exc:
            raise ReviewQueueError(f"could not read {ref}: {exc}") from exc
        return _row_to_review(row) if row is not None else None

    def list_open(self, *, kind: str = "", limit: int = 100) -> list[Review]:
        """Open reviews, oldest first — the reviewer's working set."""
        clause = "WHERE status = %s"
        params: list = [OPEN]
        if kind:
            if kind not in KINDS:
                raise ValueError(f"unknown review kind: {kind!r}")
            clause += " AND kind = %s"
            params.append(kind)
        params.append(max(1, min(int(limit), 500)))

        try:
            with self._source() as conn:
                self._ensure_table(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT {_COLUMNS} FROM {self._table} {clause} "
                        "ORDER BY created_at ASC LIMIT %s",
                        tuple(params),
                    )
                    rows = cur.fetchall() or []
        except Exception as exc:
            raise ReviewQueueError(f"could not list open reviews: {exc}") from exc
        return [_row_to_review(row) for row in rows]


class NullReviewQueue:
    """Stand-in for when no Postgres is configured.

    Every write **raises**: without a durable queue an agent must not tell a user their appeal
    reached a human. Reads report nothing open, so an offline run looks like an empty queue.
    """

    def open_review(self, **_kwargs) -> Review:
        raise ReviewQueueError(
            "no review queue is configured — an appeal cannot be recorded, so it "
            "must not be reported as flagged. Configure Lakebase (LAKEBASE_INSTANCE)."
        )

    def resolve(self, ref: str, **_kwargs) -> Review:
        raise ReviewQueueError("no review queue is configured")

    def claim_allowance(self, conversation_id: str) -> Optional[Review]:
        return None

    def get(self, ref: str) -> Optional[Review]:
        return None

    def list_open(self, **_kwargs) -> list[Review]:
        return []
