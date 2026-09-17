"""Register the supervisor's prompts in the MLflow Prompt Registry.

Each prompt change is an immutable registry version promoted by alias; the in-repo
templates in `prompt_registry.py` are the offline fallback and the initial version. Prompts live
in Unity Catalog, so the URI is `databricks-uc` and names are three-part (mirrored in
`prompt_registry.prompt_uri`). Re-running is safe: an unchanged template is skipped. Rollback is
`--pin`, which only re-points the alias. A moved alias reaches a running endpoint within MLflow's
alias cache TTL (60s); `get_prompt` is not process-cached, so a promotion never needs a redeploy.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Repository root, however this file is run.

    A `spark_python_task` `exec`s the source in a notebook kernel, so `__file__` is unbound;
    the compiled code object still carries the real path.
    """
    try:
        here = Path(__file__)
    except NameError:
        import inspect

        here = Path(inspect.currentframe().f_code.co_filename)
    return here.resolve().parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs" / "agent_governance" / "src"))


def _prompt_specs() -> dict[str, dict]:
    """Per-prompt governance metadata, registered with every new version.

    `use_case` follows the MLflow enterprise tagging convention (author + use-case per version).
    `response_format` is the structured-output contract the code holds the model to; registering
    it with the version means schema and instructions are promoted and rolled back together.
    """
    from supervisor.context_resolver import RouteDecision
    from supervisor.guardrail_engine import GuardrailVerdict

    return {
        "supervisor_domain_screen": {
            "use_case": "guardrail_domain_screen",
            "response_format": GuardrailVerdict,
        },
        "supervisor_routing": {
            "use_case": "context_resolution",
            "response_format": RouteDecision,
        },
        "supervisor_worker_simulation": {
            "use_case": "worker_simulation",
            "response_format": None,
        },
    }


def _model_config() -> dict:
    """The run parameters a version was written and evaluated against.

    Governance prompts run near-deterministic (settings.py); a version promoted on those terms
    should record the model tier and temperature it was validated with.
    """
    return {
        "model_name": os.getenv("ROUTING_LLM_ENDPOINT", "databricks-claude-sonnet-4-5"),
        "temperature": float(os.getenv("ROUTING_TEMPERATURE", "0.0")),
    }


def _pin(args, mlflow, names) -> int:
    """Point the alias at an existing version. Registers nothing."""
    from mlflow import MlflowClient

    client = MlflowClient()
    failed = False

    for name in names:
        qualified = f"{args.catalog_schema}.{name}"
        try:
            available = sorted((int(v.version) for v in client.search_prompt_versions(qualified)))
        except Exception as exc:
            failed = True
            print(f"  FAILED     {qualified}")
            print(f"             {type(exc).__name__}: {str(exc)[:200]}")
            continue

        # Refuse rather than let the alias point at nothing — a dangling alias
        # sends every request to the bundled fallback with no error anywhere.
        if args.pin not in available:
            failed = True
            print(f"  NO SUCH VERSION  {qualified} v{args.pin} — has {available}")
            continue

        if args.dry_run:
            print(f"  would pin  {qualified} @{args.alias} -> v{args.pin}")
            continue

        try:
            mlflow.genai.set_prompt_alias(name=qualified, alias=args.alias, version=args.pin)
            print(f"  pinned     {qualified} @{args.alias} -> v{args.pin}")
        except Exception as exc:
            failed = True
            print(f"  FAILED     {qualified}")
            print(f"             {type(exc).__name__}: {str(exc)[:200]}")

    if failed:
        print("\nOne or more aliases could not be moved.")
        return 1

    if not args.dry_run:
        print("\nAlias moved. A running endpoint picks this up within MLflow's")
        print("alias cache TTL (60s by default) — no redeploy needed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-schema",
        default=os.getenv("PROMPT_CATALOG_SCHEMA", "workspace.supervisor"),
        help="Unity Catalog catalog.schema holding the prompts",
    )
    parser.add_argument(
        "--alias",
        default=os.getenv("PROMPT_ALIAS", os.getenv("ENVIRONMENT", "dev")),
        help="environment alias to point at the registered version",
    )
    parser.add_argument(
        "--pin",
        type=int,
        default=None,
        metavar="VERSION",
        help="move the alias to an existing version and register nothing (rollback)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="move the alias onto the version just registered. Required for any "
        "alias outside DEV, where an evaluation run belongs in between",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="register the version and leave the alias where it is, even on DEV",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Every prompt change is an immutable version, evaluated before manual promotion, and
    # only authorized owners move production aliases. Registering and promoting in one step let
    # nothing run between them; DEV stays a single command (not a production alias), QA/PROD gate.
    auto_promote = args.alias.strip().lower() in {"dev", "development"}
    promote = args.promote or (auto_promote and not args.no_promote)

    import mlflow

    from supervisor.prompt_registry import bundled_default, prompt_names

    mlflow.set_registry_uri("databricks-uc")

    print(f"catalog.schema : {args.catalog_schema}")
    print(f"alias          : {args.alias}")
    if args.pin is not None:
        print(f"pin to version : {args.pin}  (no new versions will be created)")
    print()

    if args.pin is not None:
        return _pin(args, mlflow, prompt_names())

    failed = False
    pending: list[tuple[str, int]] = []
    specs = _prompt_specs()
    model_config = _model_config()
    for name in prompt_names():
        qualified = f"{args.catalog_schema}.{name}"
        template = bundled_default(name)
        spec = specs.get(name, {})

        if args.dry_run:
            destination = (
                f"-> @{args.alias}"
                if promote
                else f"(alias @{args.alias} would NOT move — evaluate, then --pin)"
            )
            print(f"  would register {qualified}  ({len(template)} chars) {destination}")
            continue

        try:
            # An identical template does not deserve a new version — versions are
            # immutable and each one is meant to be evaluated before promotion.
            current = None
            try:
                current = mlflow.genai.load_prompt(f"prompts:/{qualified}@{args.alias}")
            except Exception:
                pass  # not registered yet, or the alias does not exist

            if current is not None and current.template == template:
                print(f"  unchanged  {qualified} @ {args.alias} (v{current.version})")
                continue

            tags = {
                "source": "deploy/register_prompts.py",
                "component": "supervisor",
                # MLflow's enterprise pair: owner and purpose. The owner is whoever may move
                # production aliases, so it is configuration, not code.
                "author": os.getenv("PROMPT_OWNER", "supervisor-agent-team"),
                "use_case": spec.get("use_case", "supervisor"),
            }
            extras = {"model_config": model_config}
            if spec.get("response_format") is not None:
                extras["response_format"] = spec["response_format"]
            try:
                version = mlflow.genai.register_prompt(
                    name=qualified,
                    template=template,
                    commit_message="Registered from the in-repo bundled default",
                    tags=tags,
                    **extras,
                )
            except TypeError:
                # Older MLflow (a Runtime pin) lacks response_format/model_config on
                # register_prompt; losing the contract metadata beats failing the whole run.
                print(
                    f"  note       {qualified}: this MLflow cannot attach "
                    "response_format/model_config — registered without them"
                )
                version = mlflow.genai.register_prompt(
                    name=qualified,
                    template=template,
                    commit_message="Registered from the in-repo bundled default",
                    tags=tags,
                )
            if promote:
                mlflow.genai.set_prompt_alias(
                    name=qualified, alias=args.alias, version=version.version
                )
                print(f"  registered {qualified} v{version.version} -> @{args.alias}")
            else:
                pending.append((name, version.version))
                print(
                    f"  registered {qualified} v{version.version} "
                    f"(alias @{args.alias} NOT moved — evaluate, then promote)"
                )

        except Exception as exc:
            failed = True
            print(f"  FAILED     {qualified}")
            print(f"             {type(exc).__name__}: {str(exc)[:300]}")

    if failed:
        print("\nOne or more prompts could not be registered.")
        print("Check that the catalog.schema exists and that you hold CREATE on it.")
        return 1

    if pending:
        print(f"\n{len(pending)} version(s) registered but NOT promoted — @{args.alias} still")
        print("points where it did. Nothing reaches the endpoint until it moves.")
        print("\nEvaluate the new version, then promote it per prompt:")
        for name, version in pending:
            print(
                f"    python deploy/register_prompts.py --alias {args.alias} --pin {version}"
                f"   # {name}"
            )
        return 0

    if not args.dry_run:
        print("\nDone. A running endpoint picks these up within MLflow's alias cache")
        print("TTL (60s by default) — provided its service principal can read the")
        print("schema (DEPLOYMENT.md §7c). The endpoint log says which source it loaded.")
    return 0


if __name__ == "__main__":
    # A spark_python_task fails on any escaping exception, SystemExit(0) too: exit only on failure.
    _exit_code = main()
    if _exit_code:
        sys.exit(_exit_code)
