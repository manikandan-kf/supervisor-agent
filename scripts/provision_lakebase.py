"""Provision the Lakebase instance backing durable memory and the audit sink.

One-time setup **per deployed environment**, run with *your* workspace
credentials. Every environment gets its own Postgres schema, which is what keeps
a dev conversation out of the prod checkpoint table when both share an instance:

    set DATABRICKS_CONFIG_PROFILE=<profile>

    # 1. Create the instance and wait for it to become AVAILABLE (~a few min).
    #    One instance serves every environment; run this once.
    .venv\\Scripts\\python.exe scripts\\provision_lakebase.py --instance supervisor-memory

    # 2. Create this environment's schema and grant the endpoint's service
    #    principal a Postgres role on it. Run once per environment, after that
    #    environment's first `bundle run supervisor_agent_deploy`:
    .venv\\Scripts\\python.exe scripts\\provision_lakebase.py --instance supervisor-memory ^
        --pg-schema supervisor_dev ^
        --grant-identity <endpoint-SP-application-id> --identity-type SERVICE_PRINCIPAL

    # 3. Optional — register the instance in Unity Catalog for governance
    #    (read-only mirror of its schema/tables; the checkpointer, store and
    #    audit sink keep writing straight to Postgres, nothing moves):
    .venv\\Scripts\\python.exe scripts\\provision_lakebase.py --instance supervisor-memory ^
        --skip-instance --register-uc-catalog supervisor_memory

`--pg-schema` must match the deployment's `LAKEBASE_SCHEMA` — the bundle sets
both from the same variable, so use `--var lakebase_schema=` as the one source
of truth if you override it. Physical isolation instead of schema isolation is
the same script with a different `--instance` per environment.

What each part does:

* Instance creation — `w.database.create_database_instance`. Smallest capacity
  by default; idempotent (an existing instance is left as-is).
* Schema creation — `CREATE SCHEMA IF NOT EXISTS`, run as you rather than as the
  serving identity, because the environments have to be separated before
  anything writes. A `search_path` naming a schema that does not exist is not an
  error in Postgres: unqualified `CREATE TABLE` falls through to `public`, so a
  missing schema silently merges the environments instead of failing.
* Grants — Lakebase authenticates Databricks identities with short-lived OAuth
  tokens, but Postgres still needs a *role* matching the identity and
  privileges on the schema. `databricks_ai_bridge.lakebase.LakebaseClient`
  wraps exactly that (`create_role`, `grant_schema`, `grant_all_tables_in_schema`).
* UC catalog registration — `w.database.create_database_catalog`. Creates a
  **read-only** UC catalog mirroring this instance's `databricks_postgres`
  database (schemas, tables, columns — not the row data as Delta; Postgres
  stays the live store). This is governance/discovery only: it does not
  replace, sync into, or compete with the checkpointer/store connections
  above. Short-term memory is deliberately not held in UC Delta tables: no
  official LangGraph checkpointer writes to Delta, and the per-row write pattern
  a checkpointer needs is a poor fit for it.

The workspace-level half of the endpoint's access — permission to mint database
credentials at all — is NOT granted here: `deploy/log_and_deploy.py` declares
the instance as a `DatabricksLakebase` model resource, and `agents.deploy()`'s
auth passthrough covers it, the same way it covers the routing LLM.

Find the endpoint's service principal id in the serving endpoint's details page
(Serving → the endpoint → the "Served entities" identity), or from
`scripts/verify_deployment.py` output once deployed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time


def ensure_instance(w, name: str, capacity: str) -> None:
    from databricks.sdk.errors import NotFound

    try:
        instance = w.database.get_database_instance(name)
        print(f"instance {name!r} already exists (state: {instance.state})")
    except NotFound:
        from databricks.sdk.service.database import DatabaseInstance

        print(f"creating instance {name!r} (capacity {capacity}) …")
        w.database.create_database_instance(DatabaseInstance(name=name, capacity=capacity))

    deadline = time.monotonic() + 15 * 60
    while True:
        state = str(getattr(w.database.get_database_instance(name), "state", ""))
        print(f"  state: {state}")
        if "AVAILABLE" in state:
            break
        if time.monotonic() > deadline:
            raise SystemExit("timed out waiting for the instance to become AVAILABLE")
        time.sleep(20)
    print(f"instance {name!r} is AVAILABLE")


def ensure_pg_schema(instance: str, schema: str) -> None:
    """Create this environment's Postgres schema, as the operator.

    `public` needs nothing — it is Lakebase's default and already exists.

    Anything else has to exist before the first write, and cannot be left to the
    runtime. `SET search_path TO supervisor_dev, public` succeeds against a
    schema that is not there; an unqualified `CREATE TABLE` then falls through
    to `public`, so a typo or a skipped step does not fail the deploy — it
    quietly puts both environments' checkpoints, audit rows and governed config
    in the same tables. That is the failure this call exists to make impossible,
    which is why it runs as you (the instance owner) rather than as the serving
    identity, whose privileges are deliberately narrower.
    """
    if schema == "public":
        return

    from databricks_ai_bridge.lakebase import LakebaseClient

    with LakebaseClient(instance_name=instance) as client:
        with client.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote(schema)}")
    print(f"schema {schema!r} exists")


def grant(instance: str, identity: str, identity_type: str, schema: str, database: str) -> None:
    from databricks_ai_bridge.lakebase import (
        LakebaseClient,
        SchemaPrivilege,
        TablePrivilege,
    )

    with LakebaseClient(instance_name=instance) as client:
        print(f"creating Postgres role for {identity_type} {identity!r} …")
        client.create_role(identity, identity_type)
        # The checkpointer/store/audit sink create their own tables, so the
        # role needs CREATE on the schema, and full DML on what exists already.
        client.grant_schema(identity, [SchemaPrivilege.USAGE, SchemaPrivilege.CREATE], [schema])
        client.grant_all_tables_in_schema(identity, [TablePrivilege.ALL], [schema])
        if schema != "public":
            # CREATE on the *database*, which is narrower than it sounds: it
            # permits creating schemas, nothing inside anyone else's.
            #
            # Needed because `CheckpointSaver.setup()` and
            # `DatabricksStore.setup()` issue `CREATE SCHEMA IF NOT EXISTS` on
            # every boot, as the serving identity. `ensure_pg_schema` above has
            # already created it, so the statement is a no-op — but whether
            # Postgres checks the privilege before or after taking the
            # IF NOT EXISTS shortcut is a version-dependent detail, and a
            # replica that refuses to boot is not worth staking on it.
            with client.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f"GRANT CREATE ON DATABASE {_quote(database)} TO {_quote(identity)}"
                )
    print(f"granted USAGE+CREATE on schema {schema!r} and ALL on its tables to {identity!r}")
    print(
        "  note: ALL PRIVILEGES is not ownership. Postgres gives a table one\n"
        "  owner — whichever identity created it — and owner-only DDL such as\n"
        "  CREATE INDEX stays refused for everyone else. The code accounts for\n"
        "  this (audit.py:_ensure_table skips DDL when the table exists), so no\n"
        "  further grant is needed; a non-owner writes rows perfectly well."
    )
    print(
        "  IMPORTANT: this grant is deliberately broad because the runtime\n"
        "  creates its own tables on first use. Run --restrict-governance-tables\n"
        "  once they exist — until then the serving identity can rewrite its own\n"
        "  policy and edit its own audit trail. See DEPLOYMENT.md."
    )


# ── Least privilege on the governance tables ────────────────────────────────
#
# The problem this closes, stated plainly: `grant()` above gives the serving
# identity ALL PRIVILEGES on every table in the schema, because the checkpointer,
# store and audit sink all create their own tables on first use and need DML on
# them. That same grant covers two tables where DML is a privilege-escalation
# path:
#
#   * `supervisor_config`  — the governed role→agent map, the guardrail
#     definitions and the agent registry. Write access here lets the component
#     the policy governs rewrite the policy.
#   * the audit table      — the record that the gate ran. UPDATE/DELETE here
#     lets the component being audited edit its own audit trail.
#
# `config_store`'s checksum detects a row edited outside `publish()`, which stops
# a careless edit and does nothing about a deliberate one: anyone holding the
# runtime role can compute a valid checksum for their own payload. Detection is
# not prevention.
#
# NIST SP 800-207 §5.1 (Subversion of the ZTA Decision Process) is the anchor:
# "the PE and PA components must be properly configured and monitored, and any
# configuration changes must be logged and subject to audit." An audit trail
# writable by the component it audits is not subject to audit.
#
# Blueprint §07 names the same thing from the change-management side: "config
# rights must not become a new privilege-escalation path around the RBAC gate."
#
# Nothing in the request path writes config — the only writer is
# `scripts/publish_config.py`, which runs as the DAB job identity — so this
# breaks no runtime behaviour. Append-only on the audit table likewise: the sink
# only ever INSERTs.
# `TRIGGER` and `REFERENCES` are revoked alongside the DML because
# `GRANT ALL PRIVILEGES` above hands those out too, and the DML-only revoke left
# them in place. `information_schema.role_table_grants` is the only way to see
# that: the REVOKE statements report success either way.
#
# `TRIGGER` is the one that matters. It lets the serving identity attach a
# trigger to a table it can only read, and a trigger function fires with the
# privileges of whoever performs the DML — so a trigger on `supervisor_config`
# would run as the `publish_config` job identity the next time policy is
# published, which is exactly the escalation around the RBAC gate that Blueprint
# §07 names. Revoking DML but leaving TRIGGER closes the front door and leaves
# the window open. `REFERENCES` is milder (a foreign key into the audit table can
# obstruct maintenance, not rewrite a row) and is revoked for the same reason:
# nothing in the runtime uses either, so neither costs anything to remove.
_RESTRICT_SQL = """
-- Policy is read-only from the data plane.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, TRIGGER, REFERENCES ON {config} FROM {role};
GRANT  SELECT                                                ON {config} TO   {role};

-- The audit trail is append-only from the data plane.
REVOKE UPDATE, DELETE, TRUNCATE, TRIGGER, REFERENCES         ON {audit}  FROM {role};
GRANT  INSERT, SELECT                                        ON {audit}  TO   {role};

-- A review is opened by the supervisor and resolved by the gateway, so the
-- runtime keeps INSERT and UPDATE here. DELETE is nobody's business: a resolved
-- review is the evidence that the appeal was handled.
REVOKE DELETE, TRUNCATE, TRIGGER, REFERENCES                 ON {reviews} FROM {role};
GRANT  INSERT, SELECT, UPDATE                                ON {reviews} TO   {role};
"""

_MIGRATE_AUDIT_SQL = """
ALTER TABLE {audit} ADD COLUMN IF NOT EXISTS latency_ms INTEGER;
ALTER TABLE {audit} ADD COLUMN IF NOT EXISTS session_age_seconds DOUBLE PRECISION;
ALTER TABLE {audit} ADD COLUMN IF NOT EXISTS signoff JSONB;
ALTER TABLE {audit} ADD COLUMN IF NOT EXISTS model_calls INTEGER;
ALTER TABLE {audit} ADD COLUMN IF NOT EXISTS tokens_estimated INTEGER;
ALTER TABLE {audit} ADD COLUMN IF NOT EXISTS provenance JSONB;
ALTER TABLE {audit} ADD COLUMN IF NOT EXISTS prev_hash TEXT;
ALTER TABLE {audit} ADD COLUMN IF NOT EXISTS row_hash TEXT;
"""


def _quote(identifier: str) -> str:
    """Quote a Postgres identifier, rejecting anything that needs escaping.

    These statements are assembled as SQL text because `GRANT`/`REVOKE` cannot
    take a parameterized identifier. Same discipline as
    `config_store.safe_identifier`: validate, then interpolate, so the
    interpolation rests on an enforced property. A Databricks service-principal
    application id is a UUID and a table name is an identifier, so anything
    outside this character set is a mistake worth stopping.
    """
    if not re.match(r"^[A-Za-z0-9_.\-]{1,128}$", identifier or ""):
        raise SystemExit(f"refusing to build SQL with unsafe identifier: {identifier!r}")
    return '"' + identifier + '"'


def restrict_governance_tables(
    instance: str,
    identity: str,
    *,
    config_table: str,
    audit_table: str,
    review_table: str,
    schema: str,
    dry_run: bool,
) -> None:
    """Reduce the runtime role to read-only on policy and append-only on audit."""
    statements = _RESTRICT_SQL.format(
        role=_quote(identity),
        config=_quote(config_table),
        audit=_quote(audit_table),
        reviews=_quote(review_table),
    )

    if dry_run:
        print("--dry-run: would execute\n")
        print(statements)
        return

    from databricks_ai_bridge.lakebase import LakebaseClient

    # `LakebaseClient` exposes a psycopg pool, not a `connect()` method — note
    # that `--dry-run` returns before this point, so a mistake here is invisible
    # until the statements are actually applied.
    #
    # `schema=` is not optional here: the table names above are unqualified, so
    # without it the REVOKEs resolve through `public` and restrict a different
    # environment's tables — or none at all.
    with LakebaseClient(instance_name=instance, schema=schema) as client:
        with client.pool.connection() as conn, conn.cursor() as cur:
            for statement in _split_statements(statements):
                print(f"  {statement}")
                cur.execute(statement)
    print(
        f"\nrestricted {identity!r}: SELECT-only on {config_table!r}, "
        f"append-only on {audit_table!r}."
    )
    print(
        "  the publishing identity (the DAB `publish_config` job) is unaffected —\n"
        "  it holds its own grants."
    )
    # S608 flags this as string-built SQL. It is not executed — it is a query
    # *printed* for the operator to paste into a SQL client, and `identity` has
    # already been through `_quote()`'s character-set check above. Suppressed
    # here rather than in ruff.toml so the rule keeps applying to the rest of a
    # file that does build real GRANT/REVOKE statements.
    print(
        "\n  Verify against the catalog rather than trusting the statements above —\n"  # noqa: S608
        "  a REVOKE reports success whether or not it removed anything:\n"
        "\n"
        "    SELECT table_name, privilege_type\n"
        "      FROM information_schema.role_table_grants\n"
        f"     WHERE grantee = '{identity}'\n"
        f"       AND table_schema = '{schema}'\n"
        f"       AND table_name IN ('{config_table}', '{audit_table}',\n"
        f"                          '{review_table}')\n"
        "     ORDER BY table_name, privilege_type;\n"
        "\n"
        f"  Expected: SELECT on {config_table}; INSERT+SELECT on the audit\n"
        "  table; INSERT+SELECT+UPDATE on the review queue. Anything more —\n"
        "  TRIGGER especially — means the revoke did not cover it.\n"
        "  The table_schema filter matters once environments are separated: the\n"
        "  same three table names exist in every environment's schema."
    )


def migrate_audit_table(
    instance: str, *, audit_table: str, schema: str, dry_run: bool
) -> None:
    """Add the columns §05 Stage 06 asks for to an audit table that predates them.

    Run as the table's **owner**: `ALTER TABLE` takes the same ownership check
    that makes `CREATE INDEX` unusable for the serving identity, which is why
    the sink detects its columns instead of adding them (see
    `audit.PostgresAuditLogger._ensure_table`).
    """
    statements = _MIGRATE_AUDIT_SQL.format(audit=_quote(audit_table))
    if dry_run:
        print("--dry-run: would execute\n")
        print(statements)
        return

    from databricks_ai_bridge.lakebase import LakebaseClient

    # Same pool access as restrict_governance_tables above, and `schema=` for
    # the same reason: the table name is unqualified, and each environment has
    # its own copy of it.
    with LakebaseClient(instance_name=instance, schema=schema) as client:
        with client.pool.connection() as conn, conn.cursor() as cur:
            for statement in _split_statements(statements):
                print(f"  {statement}")
                cur.execute(statement)
    print(
        f"\n{audit_table!r} now carries latency_ms, session_age_seconds, signoff, "
        "model_calls, tokens_estimated, provenance and the prev_hash/row_hash chain."
    )


def _split_statements(block: str) -> list[str]:
    """One statement per `;`, comments and blank lines dropped."""
    lines = [
        line for line in block.splitlines() if line.strip() and not line.strip().startswith("--")
    ]
    joined = "\n".join(lines)
    return [s.strip() for s in joined.split(";") if s.strip()]


def ensure_uc_catalog(w, catalog_name: str, instance_name: str, pg_database: str) -> None:
    """Register the instance's Postgres database as a read-only UC catalog.

    Idempotent, same style as `ensure_instance`: a catalog already registered
    under this name is left as-is rather than re-created. Needs `CREATE CATALOG`
    on the metastore for the identity running this — a workspace-admin-level
    grant, separate from anything `grant()` above sets up, which is why this is
    its own opt-in flag rather than bundled into the default run.
    """
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.database import DatabaseCatalog

    try:
        existing = w.database.get_database_catalog(catalog_name)
        print(
            f"UC catalog {catalog_name!r} already registered "
            f"(instance={existing.database_instance_name}, database={existing.database_name})"
        )
        return
    except NotFound:
        pass

    print(f"registering UC catalog {catalog_name!r} -> {instance_name}/{pg_database} …")
    w.database.create_database_catalog(
        DatabaseCatalog(
            name=catalog_name,
            database_instance_name=instance_name,
            database_name=pg_database,
            create_database_if_not_exists=False,
        )
    )
    print(f"registered — browse it in Catalog Explorer as {catalog_name!r}")
    print(
        "  note: read-only, and a metadata mirror only (schemas/tables/columns) —\n"
        "  querying rows through it needs a Serverless SQL Warehouse; the\n"
        "  checkpointer/store/audit sink are unaffected and keep using their own\n"
        "  direct Postgres connections."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instance", required=True, help="Lakebase instance name")
    parser.add_argument(
        "--capacity", default="CU_1", help="capacity unit for a new instance (default CU_1)"
    )
    parser.add_argument(
        "--grant-identity",
        action="append",
        default=[],
        help="Databricks identity to map to a Postgres role and grant schema "
        "access (repeatable). For the serving endpoint use its service "
        "principal application id.",
    )
    parser.add_argument(
        "--identity-type",
        choices=("USER", "SERVICE_PRINCIPAL", "GROUP"),
        default="SERVICE_PRINCIPAL",
    )
    parser.add_argument(
        "--pg-schema",
        default=os.getenv("LAKEBASE_SCHEMA", "public"),
        help="Postgres schema this environment owns — created if missing, then "
        "granted. Must match the deployment's LAKEBASE_SCHEMA (the bundle sets "
        "both from --var lakebase_schema). Default `public`, the "
        "single-environment shape.",
    )
    parser.add_argument(
        "--skip-instance", action="store_true", help="grants only — the instance already exists"
    )
    parser.add_argument(
        "--register-uc-catalog",
        default="",
        help="UC catalog name to register this instance under (optional; governance/"
        "discovery only — see the module docstring). Omit to skip.",
    )
    parser.add_argument(
        "--pg-database",
        default="databricks_postgres",
        help="Postgres database name inside the instance (Lakebase's default)",
    )
    parser.add_argument(
        "--restrict-governance-tables",
        action="store_true",
        help="reduce the granted identity to SELECT-only on the config table and "
        "append-only on the audit table. Run once the runtime has created its "
        "tables. Closes the escalation path where the serving identity can "
        "rewrite its own policy and edit its own audit trail.",
    )
    parser.add_argument(
        "--migrate-audit",
        action="store_true",
        help="add latency_ms, session_age_seconds, signoff, model_calls, "
        "tokens_estimated, provenance and the prev_hash/row_hash tamper-evidence "
        "pair to an audit table created before those columns existed. Run as the "
        "table's owner.",
    )
    parser.add_argument(
        "--config-table",
        default=os.getenv("SUPERVISOR_CONFIG_TABLE", "supervisor_config"),
        help="governed configuration table (default supervisor_config)",
    )
    parser.add_argument(
        "--audit-table",
        default=os.getenv("AUDIT_PG_TABLE", "supervisor_audit_log"),
        help="decision-trail table (default supervisor_audit_log)",
    )
    parser.add_argument(
        "--review-table",
        default=os.getenv("REVIEW_QUEUE_TABLE", "supervisor_review_queue"),
        help="appeal/escalation queue table (default supervisor_review_queue)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the SQL that --restrict-governance-tables or --migrate-audit "
        "would run, and change nothing",
    )
    args = parser.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    print(f"workspace: {w.config.host}")

    if not args.skip_instance:
        ensure_instance(w, args.instance, args.capacity)

    print(f"environment schema: {args.pg_schema}")
    # Unconditional, and `ensure_pg_schema` no-ops for `public` itself.
    #
    # It must not be gated on the other flags: that would make the documented
    # pre-deploy step
    #
    #     provision_lakebase.py --skip-instance --pg-schema supervisor_dev
    #
    # do nothing at all — printing the schema name and creating no schema. That
    # step exists precisely because a missing schema is the one failure Postgres
    # does not report: `search_path` accepts a name that resolves to nothing and
    # the first unqualified CREATE TABLE lands in `public`. A guard against a
    # silent fault must not itself be silent.
    ensure_pg_schema(args.instance, args.pg_schema)

    for identity in args.grant_identity:
        grant(args.instance, identity, args.identity_type, args.pg_schema, args.pg_database)

    if args.migrate_audit:
        print(f"\nmigrating audit table {args.pg_schema}.{args.audit_table!r} …")
        migrate_audit_table(
            args.instance,
            audit_table=args.audit_table,
            schema=args.pg_schema,
            dry_run=args.dry_run,
        )

    if args.restrict_governance_tables:
        if not args.grant_identity:
            raise SystemExit(
                "--restrict-governance-tables needs --grant-identity to name the "
                "role to restrict"
            )
        for identity in args.grant_identity:
            print(f"\nrestricting governance-table privileges for {identity!r} …")
            restrict_governance_tables(
                args.instance,
                identity,
                config_table=args.config_table,
                audit_table=args.audit_table,
                review_table=args.review_table,
                schema=args.pg_schema,
                dry_run=args.dry_run,
            )

    if not args.grant_identity:
        print(
            "\nNo --grant-identity given. After the first deploy, re-run with the "
            "endpoint service principal id so the container can use the database."
        )

    if args.register_uc_catalog:
        ensure_uc_catalog(w, args.register_uc_catalog, args.instance, args.pg_database)
    print(
        "\nNext: deploy this environment with the instance and schema declared —\n"
        f"  databricks bundle deploy -t dev --var lakebase_instance={args.instance} "
        f"--var lakebase_schema={args.pg_schema}\n"
        f"  databricks bundle run supervisor_agent_deploy -t dev "
        f"--var lakebase_instance={args.instance} --var lakebase_schema={args.pg_schema}\n"
        "\n(both variables already default to these values per target in "
        "databricks.yml — pass them only when overriding.)"
    )
    return None


if __name__ == "__main__":
    sys.exit(main())
