"""Inspect the MLflow Prompt Registry — what is registered, and what loads.

    python scripts/show_prompts.py                 # names, versions, aliases
    python scripts/show_prompts.py --show-template # print the aliased text too
    python scripts/show_prompts.py --diff          # registry vs bundled default

Prompts live in Unity Catalog, which imposes two rules that fail *silently* if
you get them wrong (`prompt_provider.py` falls back to the bundled default and
logs a WARNING rather than raising):

  * the registry URI must be `databricks-uc`;
  * names must be three-part, `catalog.schema.name`.

Both are set here, so this script sees what a correctly configured client sees.

Note what this does NOT tell you: whether the *serving endpoint* can read these.
That is a different identity with different Unity Catalog grants — use
`scripts/verify_deployment.py`, which reads the endpoint's own logs.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path


def _repo_root() -> Path:
    try:
        here = Path(__file__)
    except NameError:  # spark_python_task execs the source; no __file__
        import inspect

        here = Path(inspect.currentframe().f_code.co_filename)
    return here.resolve().parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _authenticate() -> None:
    """Resolve credentials before MLflow does.

    The Databricks CLI finds credentials on its own; the SDK and MLflow do not
    read `.env`, so a script talking to the workspace through them dies on
    "cannot configure default credentials" in a shell that has not exported them
    by hand. `trace_turn.load_dotenv` is the shared reader.

    Reading the registry is the one place where *which* identity matters. The
    `.env` service principal is refused with `User does not have USE SCHEMA on
    Schema '<catalog>.<schema>'` (see README, Known limitations), while a human's
    own profile can read the prompts — so an explicitly chosen profile is left
    alone rather than being overridden by credentials known not to work here.
    """
    import os

    if os.getenv("DATABRICKS_CONFIG_PROFILE"):
        return

    from trace_turn import load_dotenv

    load_dotenv(ROOT / ".env")
    if os.getenv("DATABRICKS_CLIENT_ID") and os.getenv("DATABRICKS_CLIENT_SECRET"):
        os.environ.setdefault("DATABRICKS_AUTH_TYPE", "oauth-m2m")


def main() -> int:
    # Loaded first, so `--environment` can default to $ENVIRONMENT.
    _authenticate()

    import _environment

    parser = argparse.ArgumentParser()
    _environment.add_arguments(parser, model=False)
    parser.add_argument(
        "--aliases",
        default="",
        help="comma-separated aliases to resolve; defaults to this "
        "environment's own alias",
    )
    parser.add_argument("--show-template", action="store_true")
    parser.add_argument(
        "--diff",
        action="store_true",
        help="compare the aliased template against the in-repo bundled default",
    )
    args = parser.parse_args()

    target = _environment.resolve(args)
    args.catalog, args.schema = target.catalog, target.schema
    # Each environment holds its prompts in its own schema, so the interesting
    # alias in this one is its own — listing dev,staging,prod against a single
    # schema reports two misses every time.
    args.aliases = args.aliases or target.prompt_alias
    print(f"{target.uc_schema}  aliases: {args.aliases}\n")

    import mlflow
    from mlflow import MlflowClient

    # Both are required. Without the UC registry URI, load_prompt is refused
    # outright; without three-part names, UC rejects the name before any lookup.
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    client = MlflowClient()
    aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]

    rows = list(
        mlflow.genai.search_prompts(
            filter_string=f"catalog = '{args.catalog}' AND schema = '{args.schema}'"
        )
    )

    print(f"catalog.schema : {args.catalog}.{args.schema}")
    print(f"registered     : {len(rows)} prompt(s)\n")

    if not rows:
        print("None registered. Run:  python scripts/register_prompts.py")
        return 1

    if args.diff:
        from supervisor.prompt_provider import bundled_default

    drifted = 0
    for row in rows:
        print(f"  {row.name}")

        versions = sorted(
            (v.version for v in client.search_prompt_versions(row.name)), key=int
        )
        print(f"    versions : {versions}")

        for alias in aliases:
            try:
                loaded = mlflow.genai.load_prompt(f"prompts:/{row.name}@{alias}")
            except Exception:
                continue  # alias not set — normal for staging/prod in dev
            # An empty variable list almost always means the template was
            # registered in single-brace form: MLflow detects `{{name}}` only,
            # so `{name}` registers as taking no variables at all.
            declared = sorted(loaded.variables)
            print(
                f"    @{alias:<8}: v{loaded.version}  ({len(loaded.template)} chars)"
                f"  variables={declared or 'NONE — single-brace template?'}"
            )

            # The governance metadata register_prompts.py attaches to every
            # version: owner + use-case tags, the run parameters the version
            # was validated with, and the structured-output contract it was
            # written against. Absent on versions registered by older MLflow.
            governance = {
                k: v for k, v in (loaded.tags or {}).items() if k in ("author", "use_case")
            }
            if governance:
                print(f"      governance      : {governance}")
            config = getattr(loaded, "model_config", None)
            if config:
                print(f"      model_config    : {config}")
            contract = getattr(loaded, "response_format", None)
            if contract:
                title = (
                    contract.get("title", "(unnamed schema)")
                    if isinstance(contract, dict)
                    else getattr(contract, "__name__", str(contract))
                )
                print(f"      response_format : {title}")

            if args.show_template:
                print("    " + "-" * 60)
                for line in loaded.template.splitlines():
                    print(f"    | {line}")
                print("    " + "-" * 60)

            if args.diff:
                short = row.name.rsplit(".", 1)[-1]
                try:
                    local = bundled_default(short)
                except KeyError:
                    print(f"      (no bundled default named '{short}' to compare)")
                    continue
                if local == loaded.template:
                    print("      identical to the bundled default")
                else:
                    drifted += 1
                    print("      DIFFERS from the bundled default:")
                    diff = difflib.unified_diff(
                        local.splitlines(),
                        loaded.template.splitlines(),
                        fromfile="bundled default (in repo)",
                        tofile=f"registry @{alias}",
                        lineterm="",
                    )
                    for line in diff:
                        print(f"        {line}")
        print()

    if args.diff and drifted:
        print(f"{drifted} prompt(s) differ from the in-repo defaults.")
        print("That is expected once a prompt has been edited in the registry —")
        print("the registry is authoritative; the bundled text is only the fallback.")

    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        sys.exit(_exit_code)
