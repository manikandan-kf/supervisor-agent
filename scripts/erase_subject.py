"""Erase or purge long-term memory — GDPR Art. 17, Blueprint §05 / §06.5.

    # what is stored for one subject (no changes)
    python scripts/erase_subject.py --show usr_ab12…

    # fulfil an erasure request — long-term memory, conversation checkpoints
    # and review-queue free text (add --memory-only to restrict to memory)
    python scripts/erase_subject.py --erase usr_ab12… --requested-by alice@corp

    # retention sweep, long-term memory category
    python scripts/erase_subject.py --purge --older-than-days 400 --apply

    # retention sweep, conversation-checkpoint category
    python scripts/erase_subject.py --purge-conversations --older-than-days 180 --apply

The blueprint's §06.5 is the requirement this exists for:

    **Make erasure operable, not theoretical.** A documented procedure a named
    role can execute against Lakebase and Unity Catalog, producing a deletion
    audit record. Art. 17 is a request you must be able to *fulfil* on demand.

Three properties that make this a control rather than a `DELETE` someone runs
from a SQL console:

  * **it produces evidence.** Every erasure writes a row to the same audit sink
    as every other governance decision, naming the subject, the field names
    removed and who asked. Art. 17 compliance is demonstrated by the record, not
    by the absence of data.
  * **it never reports an erasure it did not verify.** `LongTermMemory.forget`
    re-reads the key after deleting and raises if anything remains, so a
    deferred or refused write cannot come back as "done" (§08).
  * **it names the field keys and never the values.** The values are the personal
    data; copying them into an audit table to prove they were destroyed would
    create a second copy to destroy.

**Subject identifiers are pseudonymous.** Long-term memory is keyed on the
`usr_<sha256>` reference the caller derives (§1.10) — the raw identity-provider
subject never reaches this agent. So an erasure request arriving as an email
address or a username has to be translated first: derive the reference the same
way the caller does (`--from-subject`), which needs the identity provider's
`sub` claim and the environment, since the digest is salted with it.

Dry by default: `--purge` prints what it would erase and changes nothing until
`--apply`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pseudonymous_reference(subject: str, environment: str) -> str:
    """The gateway's `Principal.pseudonymous_reference`, recomputed here.

    Duplicated deliberately rather than imported: the gateway is a separate
    deployable and this script runs from the repo, so importing it would couple
    an operator tool to a container image. The derivation is four lines and is
    covered by a test that pins it against the gateway's own implementation — if
    they ever diverge, that test fails rather than an erasure quietly targeting
    the wrong subject.
    """
    digest = hashlib.sha256(f"{environment}:{subject}".encode()).hexdigest()
    return f"usr_{digest[:24]}"


def _services():
    from supervisor.services import build_services

    return build_services()


def show(memory, user_key: str) -> int:
    stored = memory.get_context(user_key)
    if not stored:
        print(f"{user_key}: nothing stored")
        return 0
    print(f"{user_key}: {len(stored)} field(s)")
    for key, value in sorted(stored.items()):
        print(f"  {key} = {value}")
    return 0


def _audit_source():
    from supervisor.memory import audit_connection_source

    return audit_connection_source()


def _subject_conversations(user_key: str) -> list[str]:
    """Every conversation id this subject appears in, from the audit table.

    The checkpoints themselves carry no user key — identity is runtime context,
    never state (§4.4) — so the audit table's (user_key, conversation_id)
    pairing is the one durable mapping from a subject to their threads.
    """
    from supervisor.settings import Settings

    source = _audit_source()
    if source is None:
        return []
    table = Settings().audit_pg_table
    try:
        with source() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT conversation_id FROM {table} "  # noqa: S608 - operator tool
                    "WHERE user_key = %s AND conversation_id <> ''",
                    (user_key,),
                )
                rows = cur.fetchall() or []
    except Exception as exc:  # noqa: BLE001 - reported to the operator
        print(f"WARNING: could not enumerate conversations from {table}: {exc}", file=sys.stderr)
        return []
    return [
        str(row["conversation_id"] if isinstance(row, dict) else row[0]) for row in rows
    ]


def _delete_thread(checkpointer, thread_id: str) -> bool:
    """Remove one conversation's checkpoints, via the API or direct SQL."""
    if hasattr(checkpointer, "delete_thread"):
        checkpointer.delete_thread(thread_id)
        return True
    source = _audit_source()
    if source is None:
        return False
    with source() as conn:
        with conn.cursor() as cur:
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                cur.execute(
                    f"DELETE FROM {table} WHERE thread_id = %s",  # noqa: S608 - fixed names
                    (thread_id,),
                )
    return True


def _scrub_review_rows(user_key: str) -> int:
    """Blank the personal free-text on this subject's review rows.

    The rows themselves stay — an appeal that was reviewed is governance
    evidence (Art. 17(3)(b)) — but `query_excerpt` and `reason` are the
    subject's own words and the model's paraphrase of them, which is exactly
    the content an erasure exists to remove.
    """
    from supervisor.settings import Settings

    source = _audit_source()
    if source is None:
        return 0
    table = Settings().review_queue_table
    try:
        with source() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {table} SET query_excerpt = '[erased]', reason = '[erased]' "  # noqa: S608
                    "WHERE user_key = %s",
                    (user_key,),
                )
                return cur.rowcount if cur.rowcount is not None else 0
    except Exception as exc:  # noqa: BLE001 - reported to the operator
        print(f"WARNING: could not scrub review rows in {table}: {exc}", file=sys.stderr)
        return -1


def erase(services, user_key: str, requested_by: str, reason: str, memory_only: bool = False) -> int:
    from supervisor.memory import MemoryErasureFailed

    try:
        result = services.memory.forget(user_key)
    except MemoryErasureFailed as exc:
        # Loudly, and with a non-zero exit: an Art. 17 request answered with a
        # false confirmation is worse than one answered with an error.
        print(f"ERASURE FAILED for {user_key}: {exc}", file=sys.stderr)
        print(
            "The subject's memory is still present. Do NOT report this request as "
            "fulfilled — investigate the store and re-run.",
            file=sys.stderr,
        )
        return 2

    # ── The other stores holding this subject's data (Art. 17 spans them all) ─
    # Long-term memory was one of four: conversation checkpoints (the full
    # message history), the review queue's free text, and MLflow traces also
    # carry the subject's content. Audit rows are retained deliberately —
    # Art. 17(3)(b): they are the evidence that controls operated, they are
    # keyed by pseudonymous reference only, and (since the redaction pass in
    # nodes.respond) their content is minimised at write time.
    conversations_erased = 0
    reviews_scrubbed = 0
    if not memory_only:
        conversations = _subject_conversations(user_key)
        if conversations:
            from supervisor.memory import build_checkpointer

            checkpointer = build_checkpointer()
            for thread_id in conversations:
                try:
                    if _delete_thread(checkpointer, thread_id):
                        conversations_erased += 1
                    else:
                        print(
                            f"WARNING: could not delete thread {thread_id} — no deletion "
                            "path available",
                            file=sys.stderr,
                        )
                except Exception as exc:  # noqa: BLE001 - reported per thread
                    print(f"WARNING: deleting thread {thread_id} failed: {exc}", file=sys.stderr)
            print(f"erased {conversations_erased}/{len(conversations)} conversation thread(s)")
        reviews_scrubbed = _scrub_review_rows(user_key)
        if reviews_scrubbed > 0:
            print(f"scrubbed personal text from {reviews_scrubbed} review row(s)")
        print(
            "NOTE: MLflow traces are not deleted by this script. Search the experiment "
            f"for traces tagged user={user_key!r} and delete them via "
            "mlflow.client.delete_traces, or rely on the trace archive's retention."
        )

    record = result.audit_record()
    record["decision_trail"][0]["requested_by"] = requested_by
    record["decision_trail"][0]["reason"] = reason
    if not memory_only:
        record["decision_trail"][0]["conversations_erased"] = conversations_erased
        record["decision_trail"][0]["review_rows_scrubbed"] = max(0, reviews_scrubbed)
    record["target_agent_id"] = ""
    record["user_role"] = "data-subject-request"
    record["conversation_id"] = ""
    record["correlation_id"] = f"erasure-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    record["request_id"] = record["correlation_id"]

    try:
        services.audit.log(record)
        audited = True
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        audited = False
        print(f"WARNING: the erasure succeeded but its audit row failed: {exc}", file=sys.stderr)

    if result.had_data:
        print(f"erased {len(result.erased)} field(s) for {user_key}: {', '.join(result.erased)}")
    else:
        print(f"{user_key}: nothing was stored — recorded as fulfilled with no data")

    if not audited:
        # The data is gone either way, so this is not a failure of the erasure.
        # It *is* a failure of the evidence, and §06.5 asks for a deletion audit
        # record — so the operator has to know to record it another way.
        print(
            "Record this erasure manually: the deletion is done but unevidenced.",
            file=sys.stderr,
        )
        return 3
    print("deletion audit record written")
    return 0


def purge(services, older_than_days: float, apply: bool, requested_by: str) -> int:
    """Retention sweep across every subject with stored context.

    Blueprint §06.4 asks for retention *per category*, and this is the long-term
    memory category only — conversation state, audit rows and traces have
    different drivers and different floors (HIPAA §164.316(b)(2)(i) keeps audit
    records far longer than context needs to live). Deliberately not a
    one-size-fits-all sweep across the database.

    **The age used here is the checkpoint's, not the memory row's.** The store's
    items carry `updated_at`, which is when the *context* last changed — a
    conversation active weekly with a stable product line would look stale by
    that measure. Where `updated_at` is unavailable the subject is skipped and
    reported rather than erased, because erasing on an unknown age is the one
    mistake this job must not make.
    """
    subjects = services.memory.subjects()
    if not subjects:
        print("no subjects with stored long-term context")
        return 0

    cutoff_seconds = older_than_days * 24 * 3600
    now = datetime.now(timezone.utc)
    stale: list[tuple[str, float]] = []
    unknown: list[str] = []

    for user_key in subjects:
        updated = services.memory.last_updated(user_key)
        if updated is None:
            unknown.append(user_key)
            continue
        age = (now - updated).total_seconds()
        if age > cutoff_seconds:
            stale.append((user_key, age / 86400))

    print(f"{len(subjects)} subject(s); {len(stale)} older than {older_than_days:g} days")
    for user_key, age_days in sorted(stale, key=lambda pair: -pair[1]):
        print(f"  {user_key}  ({age_days:.0f} days)")
    if unknown:
        print(
            f"\n{len(unknown)} subject(s) skipped — no last-updated timestamp, so their "
            "age is unknown and they are not erased on a guess:"
        )
        for user_key in unknown:
            print(f"  {user_key}")

    if not stale:
        return 0
    if not apply:
        print("\n--apply not given; nothing was erased.")
        return 0

    failures = 0
    for user_key, age_days in stale:
        code = erase(
            services,
            user_key,
            requested_by,
            f"scheduled retention purge: {age_days:.0f} days since last update, "
            f"limit {older_than_days:g}",
            # This sweep is the long-term-memory category only (§06.4);
            # conversation retention is `--purge-conversations`, on its own
            # clock, and mixing the categories here would erase active
            # conversations because their *memory* went stale.
            memory_only=True,
        )
        failures += 1 if code else 0
    print(f"\npurge complete: {len(stale) - failures} erased, {failures} failed")
    return 1 if failures else 0


def purge_conversations(older_than_days: float, apply: bool) -> int:
    """Retention sweep for the conversation-checkpoint category (§06.4).

    A conversation's age is its *latest checkpoint's* timestamp — the last time
    anything happened in it — read from the LangGraph `checkpoints` table. The
    lazy in-graph expiry (`nodes._expire_if_stale`) already stops an old
    conversation being *used*; this is the half that makes the storage itself
    expire, which GDPR Art. 5(1)(e) asks for and a checklist review will test.
    Dry by default; `--apply` deletes.
    """
    source = _audit_source()
    if source is None:
        print("no Postgres configured — nothing to purge", file=sys.stderr)
        return 2

    try:
        with source() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT thread_id, MAX(checkpoint->>'ts') AS last_ts "
                    "FROM checkpoints GROUP BY thread_id"
                )
                rows = cur.fetchall() or []
    except Exception as exc:  # noqa: BLE001 - reported to the operator
        print(f"could not read the checkpoints table: {exc}", file=sys.stderr)
        return 2

    cutoff = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
    stale: list[str] = []
    for row in rows:
        thread_id = str(row["thread_id"] if isinstance(row, dict) else row[0])
        last_ts = row["last_ts"] if isinstance(row, dict) else row[1]
        try:
            last = datetime.fromisoformat(str(last_ts).replace("Z", "+00:00"))
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            # Unknown age is never erased on a guess — same rule as the memory
            # purge above.
            continue
        if last.timestamp() < cutoff:
            stale.append(thread_id)

    print(f"{len(rows)} conversation(s); {len(stale)} idle longer than {older_than_days:g} days")
    if not stale:
        return 0
    if not apply:
        for thread_id in stale[:50]:
            print(f"  {thread_id}")
        if len(stale) > 50:
            print(f"  … and {len(stale) - 50} more")
        print("\n--apply not given; nothing was deleted.")
        return 0

    from supervisor.memory import build_checkpointer

    checkpointer = build_checkpointer()
    deleted = 0
    for thread_id in stale:
        try:
            if _delete_thread(checkpointer, thread_id):
                deleted += 1
        except Exception as exc:  # noqa: BLE001 - reported per thread
            print(f"WARNING: deleting thread {thread_id} failed: {exc}", file=sys.stderr)
    print(f"purge complete: {deleted}/{len(stale)} conversation(s) deleted")
    return 0 if deleted == len(stale) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--show", metavar="USER_KEY", help="print what is stored, change nothing")
    action.add_argument("--erase", metavar="USER_KEY", help="erase one subject's memory")
    action.add_argument(
        "--purge", action="store_true", help="retention sweep across all subjects"
    )
    action.add_argument(
        "--purge-conversations",
        action="store_true",
        help="retention sweep for conversation checkpoints: delete threads whose "
        "latest checkpoint is older than --older-than-days (dry run without --apply)",
    )
    action.add_argument(
        "--from-subject",
        metavar="IDP_SUBJECT",
        help="derive and print the pseudonymous reference for a raw identity-provider "
        "subject, so an erasure request naming a person can be translated into a key",
    )
    parser.add_argument(
        "--environment",
        default="dev",
        help="environment the reference was derived under — the digest is salted "
        "with it, so this must match the deployment (default dev)",
    )
    parser.add_argument(
        "--older-than-days",
        type=float,
        default=400,
        help="purge threshold in days (default 400 — over a year, so an annual "
        "planning cycle does not lose its context mid-cycle)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="with --purge, actually erase (default dry run)"
    )
    parser.add_argument(
        "--requested-by",
        default="",
        help="who asked — recorded in the deletion audit row. Required for --erase.",
    )
    parser.add_argument("--reason", default="data subject erasure request (GDPR Art. 17)")
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="with --erase, remove only long-term memory and leave conversations "
        "and review rows in place (the default erases across all stores)",
    )
    args = parser.parse_args()

    if args.from_subject:
        print(pseudonymous_reference(args.from_subject, args.environment))
        return 0

    if args.erase and not args.requested_by:
        parser.error("--erase requires --requested-by so the audit row names a requester")

    services = _services()

    if args.show:
        return show(services.memory, args.show)
    if args.erase:
        return erase(services, args.erase, args.requested_by, args.reason, args.memory_only)
    if args.purge_conversations:
        return purge_conversations(args.older_than_days, args.apply)
    return purge(
        services,
        args.older_than_days,
        args.apply,
        args.requested_by or "scheduled-retention-job",
    )


if __name__ == "__main__":
    sys.exit(main())
