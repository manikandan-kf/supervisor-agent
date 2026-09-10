"""Grant Unity Catalog access to the MLflow Prompt Registry prompts.

Closes the known gap where the deployed endpoint silently falls back to its
bundled prompt templates: the prompts live in Unity Catalog
(`<catalog>.<schema>.supervisor_*`), and there is no MLflow resource type for
prompts, so `agents.deploy()`'s automatic auth passthrough cannot cover them —
the official guidance is a manual grant. Per the Databricks docs, working with
registry prompts in a schema requires `EXECUTE` (read/load) and
`CREATE FUNCTION` + `MANAGE` (register/alias) on that schema, plus the usual
`USE CATALOG` / `USE SCHEMA` path.

Grants are **per environment**, because each one keeps its prompts in its own
schema. Run with credentials that own that schema (the deploying user):

    set DATABRICKS_CONFIG_PROFILE=<profile>

    # read-only grant — enough for the serving endpoint to LOAD prompts
    .venv\\Scripts\\python.exe scripts\\grant_prompt_access.py -e dev ^
        --principal <endpoint-SP-application-id>

    # read+write grant — for the identity that REGISTERS prompts (CI, deploy SP)
    .venv\\Scripts\\python.exe scripts\\grant_prompt_access.py -e dev ^
        --principal <deploy-SP-application-id> --write

    # confirm what the current credentials can actually load
    .venv\\Scripts\\python.exe scripts\\grant_prompt_access.py -e dev --verify

Both principals need this (see README, Known limitations): the endpoint's system
service principal (shown on the serving endpoint page) and the platform/deploy
service principal. After granting, restart nothing — the endpoint's next
prompt-cache miss loads from the registry, and `scripts/verify_deployment.py`
confirms it end to end.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from supervisor.prompt_provider import prompt_names  # noqa: E402

# Derived, never hardcoded: a second list here would keep granting access to the
# prompts that existed when it was written, and a newly added prompt would fail
# to load in production with nothing but a WARNING to show for it.
PROMPTS = tuple(prompt_names())


def grant(schema_full: str, principal: str, write: bool) -> None:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import PermissionsChange, Privilege, SecurableType

    w = WorkspaceClient()
    catalog_name = schema_full.split(".")[0]

    schema_privileges = [Privilege.USE_SCHEMA, Privilege.EXECUTE]
    if write:
        schema_privileges += [Privilege.CREATE_FUNCTION, Privilege.MANAGE]

    w.grants.update(
        SecurableType.CATALOG.value,
        catalog_name,
        changes=[PermissionsChange(principal=principal, add=[Privilege.USE_CATALOG])],
    )
    w.grants.update(
        SecurableType.SCHEMA.value,
        schema_full,
        changes=[PermissionsChange(principal=principal, add=schema_privileges)],
    )
    mode = "read+write" if write else "read"
    print(f"granted {mode} prompt access on {schema_full} to {principal}")
    print(f"  catalog {catalog_name}: USE_CATALOG")
    print(f"  schema  {schema_full}: {', '.join(p.value for p in schema_privileges)}")


def verify(schema_full: str, alias: str) -> int:
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    failures = 0
    for name in PROMPTS:
        uri = f"prompts:/{schema_full}.{name}@{alias}"
        try:
            prompt = mlflow.genai.load_prompt(uri)
            version = getattr(prompt, "version", "?")
            print(f"  OK    {uri}  (v{version})")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {uri}  {type(exc).__name__}: {str(exc)[:140]}")
    if failures:
        print(f"\n{failures} of {len(PROMPTS)} prompts not loadable by the current credentials.")
    else:
        print("\nAll prompts load. If the *endpoint* still falls back, its service "
              "principal is the one missing the grant — run verify_deployment.py.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--principal",
        action="append",
        default=[],
        help="service principal application id (or group/user name) to grant (repeatable)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="also grant CREATE FUNCTION + MANAGE (register prompts and move aliases)",
    )
    parser.add_argument(
        "-e",
        "--environment",
        default=os.getenv("ENVIRONMENT", "dev"),
        help="deployed environment whose prompts to grant on (dev, prod, …). "
        "Each has its own schema, so a grant is per environment.",
    )
    parser.add_argument(
        "--schema",
        default="",
        help="catalog.schema of the prompts; defaults to workspace.supervisor_<environment>",
    )
    parser.add_argument("--alias", default="", help="alias used by --verify; defaults to the environment")
    parser.add_argument(
        "--verify", action="store_true", help="load each prompt with the current credentials"
    )
    args = parser.parse_args()

    if not args.principal and not args.verify:
        parser.error("nothing to do — pass --principal (repeatable) and/or --verify")

    schema = args.schema or f"workspace.supervisor_{args.environment}"
    alias = args.alias or args.environment
    print(f"prompts: {schema} @{alias}\n")

    for principal in args.principal:
        grant(schema, principal, args.write)

    if args.verify:
        return verify(schema, alias)
    return 0


if __name__ == "__main__":
    sys.exit(main())
