"""Register the supervisor's prompts in MLflow Prompt Registry (§4.1).

Framework §4.1 requires Phase 1 prompts to live in the registry, not the
repository, with each change creating an immutable version promoted by alias.
The in-repo templates in `prompt_provider.py` are the offline fallback; this
script publishes them as the initial registered version.

    python scripts/register_prompts.py                 # register + alias
    python scripts/register_prompts.py --dry-run       # show what would happen
    python scripts/register_prompts.py --alias prod    # promote to another alias
    python scripts/register_prompts.py --pin 1         # roll the alias back to v1

Prompts live in Unity Catalog, so:

  * the registry URI must be `databricks-uc`;
  * names must be three-part, `catalog.schema.name`.

Both are handled here and mirrored in `prompt_provider.prompt_uri`.

Re-running is safe. A template identical to the current aliased version is
skipped rather than creating a pointless version; a changed template creates a
new immutable version and moves the alias to it.

Rolling back is `--pin`. Versions are immutable, so a bad prompt is undone by
moving the alias, never by editing a version in place — the old text stays on
the record. `--pin` writes nothing new; it only re-points the alias.

A moved alias reaches a running endpoint on its own, within MLflow's alias cache
TTL (60s by default). `prompt_provider.get_prompt` is deliberately *not*
process-cached for exactly this reason — an `@lru_cache` there used to make
every promotion a redeploy. No restart is needed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Repository root, however this file is being run.

    A Databricks `spark_python_task` does not import the script — it reads the
    source and `exec`s it inside a notebook kernel, so `__file__` is never
    bound and `Path(__file__)` raises NameError. The code object compiled from
    that source still carries the real path, so the frame is the reliable
    fallback.
    """
    try:
        here = Path(__file__)
    except NameError:
        import inspect

        here = Path(inspect.currentframe().f_code.co_filename)
    return here.resolve().parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "src"))


def _prompt_specs() -> dict[str, dict]:
    """Per-prompt governance metadata, registered with every new version.

    `use_case` follows the enterprise tagging convention from the MLflow
    prompt-registry guidance (author + use-case on every version), so the
    registry stays searchable as more agents add prompts to the same schema.

    `response_format` is the structured-output contract the code holds the
    model to. Registering it with the version means the schema and the
    instructions that reference its fields ("Answer `sdlc_reading` first")
    are versioned — and promoted, and rolled back — together, instead of the
    schema silently drifting in code while the alias stays put. The worker
    simulation prompt returns prose, so it declares none.
    """
    from supervisor.guardrails import GuardrailVerdict
    from supervisor.routing import RouteDecision

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

    Recorded so an operator reading the registry sees not just the words but
    the model tier and temperature they were validated with — governance
    prompts run near-deterministic (settings.py), and a version promoted on
    those terms should say so.
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
            available = sorted(
                (int(v.version) for v in client.search_prompt_versions(qualified))
            )
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
            mlflow.genai.set_prompt_alias(
                name=qualified, alias=args.alias, version=args.pin
            )
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
        "alias outside DEV, where §4.1 wants an evaluation run in between",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="register the version and leave the alias where it is, even on DEV",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # §4.1: "Each prompt change shall create an immutable Prompt Registry version
    # and shall be evaluated before manual promotion", and "only authorized prompt
    # owners shall update production aliases".
    #
    # Registering and moving the alias in one step made those two the same action,
    # so nothing could ever run between them. Splitting on the alias keeps the DEV
    # loop a single command — DEV is not a production alias and gating it would
    # only teach people to pass --promote reflexively — while QA and PROD now
    # require the operator to come back after evaluating.
    auto_promote = args.alias.strip().lower() in {"dev", "development"}
    promote = args.promote or (auto_promote and not args.no_promote)

    import mlflow

    from supervisor.prompt_provider import bundled_default, prompt_names

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
                "source": "scripts/register_prompts.py",
                "component": "supervisor",
                # The enterprise pair from the MLflow guidance: who owns the
                # prompt, and what it is for. The owner is whoever §4.1 lets
                # move production aliases, so it is configuration, not code.
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
                # An older MLflow (a Databricks Runtime pin, typically) without
                # response_format/model_config on register_prompt. The version
                # still registers; only the attached contract metadata is lost,
                # and saying so beats failing the whole registration run.
                print(f"  note       {qualified}: this MLflow cannot attach "
                      "response_format/model_config — registered without them")
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
        print("\n  1. Evaluate:  python evaluations/run_evaluation.py")
        print("  2. Promote, per prompt, once the run is acceptable:")
        for name, version in pending:
            print(
                f"       python scripts/register_prompts.py --alias {args.alias} --pin {version}"
                f"   # {name}"
            )
        return 0

    if not args.dry_run:
        print("\nDone. A running endpoint picks these up within MLflow's alias cache")
        print("TTL (60s by default) — provided its service principal can read the")
        print("schema. Confirm with `python scripts/verify_deployment.py`.")
    return 0


if __name__ == "__main__":
    # `sys.exit(0)` raises SystemExit, and a Databricks `spark_python_task`
    # surfaces *any* exception escaping the exec'd source as a task failure —
    # including a zero exit. So only exit explicitly when there is a real
    # failure to report; success falls off the end, which is exit code 0 for a
    # normal CLI run and a clean finish for the job.
    _exit_code = main()
    if _exit_code:
        sys.exit(_exit_code)
