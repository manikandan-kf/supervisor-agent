"""Remove everything this project created in Databricks.

    python scripts/teardown_databricks.py              # dry run — lists only
    python scripts/teardown_databricks.py --yes        # actually delete
    python scripts/teardown_databricks.py -e prod --yes

**One environment at a time.** Every name is derived from `--environment`, so a
teardown of dev cannot reach prod's schema, model, endpoint or experiment. The
Lakebase instance is shared between environments and is never deleted here —
drop the environment's Postgres schema by hand if you want its rows gone.

Deletes, in dependency order:

    1. the agent serving endpoint
    2. every version of the registered model, then the model
    3. the registered prompts
    4. the tables in the project schema (audit log, inference payload)
    5. any UC functions in the project schema
    6. the MLflow experiments, active and already-trashed
    7. the schema itself, once it is empty
    8. the bundle's uploaded workspace directory

**Scope is hard-coded to this project's own objects.** The endpoint name, model
name and schema are matched exactly — a workspace's Databricks-provided
foundation-model endpoints share no name with ours and are never enumerated for
deletion. Nothing outside `--catalog`.`--schema` is touched.

This is irreversible. Model versions, prompt versions and the audit table's rows
cannot be recovered. Dry run is the default for that reason.

The serving endpoint is created imperatively by `databricks.agents.deploy()`, so
it is not a bundle resource and `databricks bundle destroy` will not remove it.
Run this first, then `bundle destroy` for the schema and job.
"""

from __future__ import annotations

import argparse
import sys

import _environment


def main() -> int:
    parser = argparse.ArgumentParser()
    _environment.add_arguments(parser)
    parser.add_argument(
        "--endpoint",
        default="",
        help="override the derived name of the agent serving endpoint that "
        "databricks.agents.deploy() created",
    )
    parser.add_argument("--experiment", default="", help="override the derived experiment")
    parser.add_argument(
        "--bundle-root",
        default="",
        help="workspace directory holding the bundle's uploaded files; "
        "defaults to /Users/<you>/.bundle/langgraph-supervisor",
    )
    parser.add_argument("--keep-schema", action="store_true", help="empty it but leave it in place")
    parser.add_argument(
        "--uc-catalog",
        default="",
        help="UC catalog registered over the Lakebase instance by "
        "provision_lakebase.py --register-uc-catalog, if any (empty = skip; this "
        "project's own UC schema is handled by step 7 either way)",
    )
    parser.add_argument("--yes", action="store_true", help="actually delete; otherwise dry run")
    args = parser.parse_args()

    target = _environment.resolve(args)
    args.endpoint = args.endpoint or target.endpoint
    args.experiment = args.experiment or target.experiment
    # `--schema` is empty until the convention fills it in; write the resolved
    # values back so every listing call below is scoped to one environment.
    args.catalog, args.schema, args.model = target.catalog, target.schema, target.model
    print(target.describe() + "\n")

    import mlflow
    from databricks.sdk import WorkspaceClient
    from mlflow.entities import ViewType

    w = WorkspaceClient()
    # Both URIs, explicitly. Without the tracking URI, MLflow resolves to any
    # local `mlruns/` or `mlflow.db` in the working directory — so the script
    # would cheerfully delete a local experiment and report success while the
    # Databricks one survived untouched.
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    full_model = target.uc_model
    schema_name = target.uc_schema
    if not args.bundle_root:
        args.bundle_root = f"/Users/{w.current_user.me().user_name}/.bundle/langgraph-supervisor"
    planned: list[tuple[str, callable]] = []

    # ── 1. serving endpoint ────────────────────────────────────────────────
    try:
        w.serving_endpoints.get(args.endpoint)
        planned.append(
            (f"serving endpoint  {args.endpoint}", lambda: w.serving_endpoints.delete(args.endpoint))
        )
    except Exception:
        print(f"  absent   serving endpoint  {args.endpoint}")

    # ── 2. model versions, then the model ──────────────────────────────────
    try:
        versions = sorted(
            (v.version for v in w.model_versions.list(full_model)), reverse=True
        )
        for version in versions:
            planned.append(
                (
                    f"model version     {full_model} v{version}",
                    lambda v=version: w.model_versions.delete(full_model, v),
                )
            )
        planned.append(
            (f"registered model  {full_model}", lambda: w.registered_models.delete(full_model))
        )
    except Exception:
        print(f"  absent   registered model  {full_model}")

    # ── 3. prompts ─────────────────────────────────────────────────────────
    # Deletion lives on MlflowClient, not on mlflow.genai — the genai module
    # exposes only the alias/tag helpers.
    #
    # A prompt cannot be deleted while it still has versions: MLflow refuses
    # with "still has undeleted versions. Please delete all versions first".
    # Dropping the schema happens to cascade and hide this, so the bug only
    # surfaces under --keep-schema — where it would leave every prompt behind.
    client = mlflow.MlflowClient()

    def drop_prompt(name: str) -> None:
        for version in sorted(
            (int(v.version) for v in client.search_prompt_versions(name)), reverse=True
        ):
            client.delete_prompt_version(name, str(version))
        client.delete_prompt(name)

    try:
        prompts = list(
            mlflow.genai.search_prompts(
                filter_string=f"catalog = '{args.catalog}' AND schema = '{args.schema}'"
            )
        )
        for p in prompts:
            planned.append(
                (f"prompt            {p.name} (all versions)", lambda n=p.name: drop_prompt(n))
            )
        if not prompts:
            print(f"  absent   prompts in {schema_name}")
    except Exception as exc:
        print(f"  skipped  prompts ({type(exc).__name__}: {str(exc)[:100]})")

    # ── 4. tables ──────────────────────────────────────────────────────────
    try:
        tables = list(w.tables.list(catalog_name=args.catalog, schema_name=args.schema))
        for t in tables:
            planned.append(
                (f"table             {t.full_name}", lambda n=t.full_name: w.tables.delete(n))
            )
        if not tables:
            print(f"  absent   tables in {schema_name}")
    except Exception as exc:
        print(f"  skipped  tables ({type(exc).__name__}: {str(exc)[:100]})")

    # ── 5. UC functions ────────────────────────────────────────────────────
    # agents.deploy() registers a UC function alongside the model; it blocks the
    # schema deletion if it is left behind.
    try:
        functions = list(w.functions.list(catalog_name=args.catalog, schema_name=args.schema))
        for f in functions:
            planned.append(
                (f"uc function       {f.full_name}", lambda n=f.full_name: w.functions.delete(n))
            )
        if not functions:
            print(f"  absent   functions in {schema_name}")
    except Exception as exc:
        print(f"  skipped  functions ({type(exc).__name__}: {str(exc)[:100]})")

    # ── 6. experiments, active and already-trashed ─────────────────────────
    # `mlflow.delete_experiment` only moves an experiment to the Trash, so a
    # previous teardown leaves one behind. Both are collected here; the trashed
    # ones are purged through the workspace API, which actually removes them.
    #
    # Matched by exact name, not by substring. Each environment has its own
    # experiment (`/Shared/supervisor-agent-dev`, `…-prod`), and a "contains
    # supervisor" test would make a dev teardown delete prod's traces — every
    # environment's experiment matches it. `--experiment` overrides the derived
    # name for a deployment that predates the convention.
    seen_experiments: set[str] = set()
    for view, label in ((ViewType.ACTIVE_ONLY, "experiment"), (ViewType.DELETED_ONLY, "trashed exp")):
        try:
            for e in mlflow.search_experiments(view_type=view):
                name = e.name or ""
                if name != args.experiment:
                    continue
                if e.experiment_id in seen_experiments:
                    continue
                seen_experiments.add(e.experiment_id)

                if view == ViewType.ACTIVE_ONLY:
                    planned.append(
                        (
                            f"{label:17} {name} (id={e.experiment_id})",
                            lambda i=e.experiment_id: mlflow.delete_experiment(i),
                        )
                    )
                else:
                    planned.append(
                        (
                            f"{label:17} {name} (id={e.experiment_id})",
                            lambda p=name: w.workspace.delete(p, recursive=True),
                        )
                    )
        except Exception as exc:
            print(f"  skipped  {label} ({type(exc).__name__}: {str(exc)[:100]})")
    if not seen_experiments:
        print("  absent   experiments")

    # ── 7. the schema, last ────────────────────────────────────────────────
    if not args.keep_schema:
        try:
            w.schemas.get(schema_name)
            planned.append((f"schema            {schema_name}", lambda: w.schemas.delete(schema_name)))
        except Exception:
            print(f"  absent   schema  {schema_name}")

    # ── 8. the bundle's uploaded workspace files ───────────────────────────
    # `databricks bundle destroy` removes these, but only if the bundle was
    # deployed from this machine with matching state. A half-finished deploy
    # leaves the directory behind.
    if args.bundle_root:
        try:
            w.workspace.get_status(args.bundle_root)
            planned.append(
                (
                    f"workspace dir     {args.bundle_root}",
                    lambda: w.workspace.delete(args.bundle_root, recursive=True),
                )
            )
        except Exception:
            print(f"  absent   workspace dir  {args.bundle_root}")

    # ── 9. the UC catalog registered over the Lakebase instance, if any ─────
    # Independent of `--catalog`/`--schema` — this mirrors a Lakebase instance
    # (see provision_lakebase.py --register-uc-catalog), not the model/prompt
    # schema steps 1-7 already cover. Deleting it only removes the read-only UC
    # mirror; the Lakebase instance and its data are untouched either way.
    if args.uc_catalog:
        try:
            w.database.get_database_catalog(args.uc_catalog)
            planned.append(
                (
                    f"UC catalog        {args.uc_catalog} (Lakebase registration)",
                    lambda: w.database.delete_database_catalog(args.uc_catalog),
                )
            )
        except Exception:
            print(f"  absent   UC catalog  {args.uc_catalog}")

    if not planned:
        print("\nNothing to delete — Databricks is already clean.")
        return 0

    print(f"\n{len(planned)} object(s) to delete:\n")
    for label, _ in planned:
        print(f"  {label}")

    if not args.yes:
        print("\nDRY RUN — nothing was deleted. Re-run with --yes to proceed.")
        return 0

    print("\nDeleting...\n")
    failures = 0
    for label, action in planned:
        try:
            action()
            print(f"  deleted  {label}")
        except Exception as exc:
            # Deleting one object often removes others: dropping the schema
            # takes the prompts with it, and a UC "function" that shares the
            # model's name disappears with the model. Something already gone is
            # the desired end state, not a failure.
            message = str(exc)
            if "does not exist" in message or "NOT_FOUND" in message or "RESOURCE_DOES_NOT_EXIST" in message:
                print(f"  already gone  {label}")
                continue
            failures += 1
            print(f"  FAILED   {label}")
            print(f"           {type(exc).__name__}: {message[:200]}")

    if failures:
        print(f"\n{failures} deletion(s) failed. Endpoint deletion is asynchronous —")
        print("if the schema failed because it was not empty, re-run in a minute.")
        return 1

    print("\nDatabricks is clean. Next: databricks bundle destroy -t dev")
    return 0


if __name__ == "__main__":
    sys.exit(main())
