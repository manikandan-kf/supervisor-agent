"""Log the supervisor as an MLflow model, register it in Unity Catalog and
deploy it on Model Serving via the Databricks Agent Framework.

Run as the DAB job task, or locally with Databricks auth configured:
    python deploy/log_and_deploy.py --routing-endpoint <endpoint> --uc-model <catalog.schema.model>

The shared governance library travels *inside* the model artifact. The wheel
`databricks bundle deploy` built is logged next to the model under `wheels/`
and named as `wheels/<file>.whl` in the model's requirements — the layout
MLflow's own `add_libraries_to_model` produces and Model Serving installs from
the model directory. A serving container therefore never depends on a volume
or an index at build time, and the library it runs is the one that was tested.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    """Repository root, however this file is being run.

    A Databricks `spark_python_task` does not import the script — it reads the
    source and `exec`s it inside a notebook kernel, so `__file__` is never
    bound. The code object compiled from that source still carries the real
    path, so the frame is the reliable fallback.
    """
    try:
        here = Path(__file__)
    except NameError:
        import inspect

        here = Path(inspect.currentframe().f_code.co_filename)
    return here.resolve().parents[1]


ROOT = _repo_root()
LIBRARY = ROOT / "libs" / "agent_governance"


def _governance_wheel() -> Path:
    """The agent_governance wheel to bake into the model artifact.

    `databricks bundle deploy` builds it (the `artifacts` block) and syncs it
    with the bundle. A workstation run that skipped the build gets one built
    here from the same source, so the two paths cannot ship different code.
    """
    dist = LIBRARY / "dist"
    wheels = sorted(dist.glob("agent_governance-*.whl"), key=lambda p: p.stat().st_mtime)
    if wheels:
        return wheels[-1]
    dist.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 — fixed argv, no untrusted input
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(dist), str(LIBRARY)],
        check=True,
    )
    return sorted(dist.glob("agent_governance-*.whl"), key=lambda p: p.stat().st_mtime)[-1]


def _runtime_requirements() -> list[str]:
    """requirements.txt as a list, comments and blanks dropped.

    A list rather than the file path, because the wheel line below has to be
    appended to it and MLflow accepts either form.
    """
    lines = []
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uc-model",
        default=os.getenv("UC_MODEL_NAME", "workspace.supervisor_dev.supervisor_agent"),
    )
    parser.add_argument(
        "--routing-endpoint",
        default=os.getenv("ROUTING_LLM_ENDPOINT", "databricks-claude-sonnet-4-5"),
    )
    parser.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "/Shared/supervisor-agent-dev"))
    parser.add_argument("--environment", default=os.getenv("ENVIRONMENT", "dev"))
    parser.add_argument(
        "--prompt-catalog-schema",
        default=os.getenv("PROMPT_CATALOG_SCHEMA", "workspace.supervisor_dev"),
        help="Unity Catalog catalog.schema holding the registered prompts",
    )
    parser.add_argument(
        "--prompt-alias",
        default=os.getenv("PROMPT_ALIAS", ""),
        help="prompt alias to load; defaults to --environment",
    )
    parser.add_argument(
        "--lakebase-instance",
        default=os.getenv("LAKEBASE_INSTANCE", ""),
        help="Lakebase database instance for durable conversation memory, long-term "
        "memory, the governed config table, the review queue and the audit sink. "
        "Declared as a model resource so the endpoint's service identity can mint "
        "database credentials, and passed to the container as LAKEBASE_INSTANCE.",
    )
    parser.add_argument(
        "--lakebase-schema",
        default=os.getenv("LAKEBASE_SCHEMA", ""),
        help="Postgres schema inside the instance that this deployment owns. Set it "
        "per environment (supervisor_dev, supervisor_prod) so dev and prod share an "
        "instance without sharing state; leave empty for the single-environment shape.",
    )
    parser.add_argument("--skip-deploy", action="store_true", help="log and register only")
    parser.add_argument(
        "--workers",
        choices=("simulated", "live"),
        default=os.getenv("SUPERVISOR_WORKERS", "live"),
        help="'simulated' stands in for worker endpoints that are not deployed yet; "
        "'live' declares and calls the real ones",
    )
    parser.add_argument(
        "--multi-model",
        choices=("false", "true"),
        default=os.getenv("MULTI_MODEL_ENABLED", "false").lower(),
        help="'true' turns on per-agent model references: each agents.yaml entry's "
        "optional `model` endpoint is declared as a resource and MULTI_MODEL_ENABLED=true "
        "is stamped onto the container. 'false' (default) deploys with the feature dark.",
    )
    args = parser.parse_args()

    simulated_workers = args.workers == "simulated"
    multi_model = args.multi_model == "true"

    os.environ["ROUTING_LLM_ENDPOINT"] = args.routing_endpoint
    # The library is installed in the job environment by the bundle; the path
    # insert covers a workstation run against a checkout that has not.
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(LIBRARY / "src"))

    import mlflow
    import mlflow.pyfunc
    import yaml
    from mlflow import MlflowClient
    from mlflow.models.resources import DatabricksLakebase, DatabricksServingEndpoint

    if os.name == "nt":
        # MLflow resolves `python_model` with Path.resolve(), which records a
        # backslash Windows path in the MLmodel file; the Linux serving
        # container then can't find the code file. Re-emit the validated path
        # as POSIX so the logged model stays loadable in Model Serving.
        _orig_validate = mlflow.pyfunc._validate_and_get_model_code_path

        def _posix_model_code_path(model_code_path, temp_dir):
            return Path(_orig_validate(model_code_path, temp_dir)).as_posix()

        mlflow.pyfunc._validate_and_get_model_code_path = _posix_model_code_path

    mlflow.set_registry_uri("databricks-uc")
    # Traces stay in the MLflow experiment's native store. The UC Delta
    # destination requires a customer external location, which a metastore
    # with only Databricks-managed default storage rejects.
    mlflow.set_experiment(experiment_name=args.experiment)

    agents_cfg = yaml.safe_load(
        (ROOT / "src" / "supervisor" / "config" / "agents.yaml").read_text(encoding="utf-8")
    )
    worker_endpoints = [a["endpoint"] for a in agents_cfg.get("agents", [])]

    # Declaring the endpoints as resources lets Databricks grant the deployed
    # supervisor's service identity Can Query on them automatically. With
    # simulated workers only the routing LLM is declared, so the supervisor can
    # be deployed before any worker endpoint exists.
    resources = [DatabricksServingEndpoint(endpoint_name=args.routing_endpoint)]
    if not simulated_workers:
        resources += [DatabricksServingEndpoint(endpoint_name=e) for e in worker_endpoints]
    if multi_model:
        # Each agent's optional model endpoint needs the same Can Query grant
        # as the routing LLM. Declared only when the feature is on — declaring
        # a not-yet-created endpoint would fail the registration for nothing.
        agent_model_endpoints = sorted(
            {a["model"].strip() for a in agents_cfg.get("agents", []) if a.get("model")}
            - {args.routing_endpoint}
        )
        resources += [DatabricksServingEndpoint(endpoint_name=e) for e in agent_model_endpoints]
    if args.lakebase_instance:
        # Grants the endpoint's service identity the ability to mint database
        # credentials for the instance — the same auth passthrough that covers
        # the LLM endpoint above. Postgres-level grants are separate.
        resources += [DatabricksLakebase(database_instance_name=args.lakebase_instance)]

    # Release traceability: record which prompt versions this model version
    # shipped with. Best-effort — the alias must already resolve, and this
    # identity may not be able to read the registry.
    prompt_uris: list[str] = []
    try:
        os.environ["PROMPT_CATALOG_SCHEMA"] = args.prompt_catalog_schema
        os.environ["PROMPT_ALIAS"] = args.prompt_alias or args.environment

        from supervisor.prompt_provider import prompt_names, prompt_uri

        prompt_uris = [prompt_uri(name) for name in prompt_names()]
        print(f"Linking {len(prompt_uris)} prompt versions to the model: {prompt_uris}")
    except Exception as exc:  # noqa: BLE001 — lineage is additive, never fatal
        print(f"Prompt lineage skipped ({type(exc).__name__}: {exc})")

    wheel = _governance_wheel()
    print(f"Baking {wheel.name} into the model artifact")

    with mlflow.start_run(run_name="supervisor-agent"):
        logged = mlflow.pyfunc.log_model(
            name="supervisor_agent",
            **({"prompts": prompt_uris} if prompt_uris else {}),
            # as_posix() so the recorded code path stays loadable inside the
            # Linux serving container when logging from Windows.
            python_model=(ROOT / "src" / "supervisor" / "agent.py").as_posix(),
            code_paths=[str(ROOT / "src" / "supervisor")],
            # The runtime pins plus the library wheel, resolved from the model
            # directory at container build — see the module docstring.
            pip_requirements=[*_runtime_requirements(), f"wheels/{wheel.name}"],
            resources=resources,
            input_example={
                "input": [{"role": "user", "content": "Generate an HLD for the new billing service"}],
                "custom_inputs": {"user_role": "BA", "agent_id": "requirement-agent"},
            },
        )
        # The wheel goes in *before* registration: Unity Catalog copies the
        # model directory when the version is created, so anything added to
        # the logged model afterwards would not reach the endpoint.
        with tempfile.TemporaryDirectory() as staging:
            wheels_dir = Path(staging) / "wheels"
            wheels_dir.mkdir()
            shutil.copy2(wheel, wheels_dir / wheel.name)
            MlflowClient().log_model_artifacts(logged.model_id, staging)

    version = mlflow.register_model(logged.model_uri, args.uc_model).version

    if args.skip_deploy:
        print(f"Logged and registered {args.uc_model} v{version}")
        return

    from databricks import agents

    env_vars = {
        "ROUTING_LLM_ENDPOINT": args.routing_endpoint,
        "ENVIRONMENT": args.environment,
        # Prompts load from Unity Catalog by name and environment alias. Stated
        # explicitly so the deployed agent's prompt source is visible in the
        # endpoint configuration.
        "PROMPT_CATALOG_SCHEMA": args.prompt_catalog_schema,
        "PROMPT_ALIAS": args.prompt_alias or args.environment,
    }
    if simulated_workers:
        env_vars["SUPERVISOR_MOCK_WORKERS"] = "true"
    if multi_model:
        env_vars["MULTI_MODEL_ENABLED"] = "true"
    if args.lakebase_instance:
        env_vars["LAKEBASE_INSTANCE"] = args.lakebase_instance
    if args.lakebase_schema:
        # Which schema a deployment writes to is a property of the deployment,
        # so it is stamped explicitly and is reviewable on the endpoint's
        # configuration page next to LAKEBASE_INSTANCE.
        env_vars["LAKEBASE_SCHEMA"] = args.lakebase_schema
    # The Autoscaling-generation Lakebase address and the tuning knobs travel
    # from the deploying environment to the container verbatim, when set.
    #
    # Guardrail knobs are deliberately absent from this list: a control that a
    # deploy shell can weaken by exporting a variable is not a control, so they
    # stay on their code defaults and `Settings.enforce` refuses to serve a
    # deployed environment where one of them is off. OUTPUT_STREAM_WORKER_TOKENS
    # is the exception because its only non-default value *strengthens* the
    # output screen.
    for passthrough in (
        "LAKEBASE_AUTOSCALING_ENDPOINT",
        "LAKEBASE_PROJECT",
        "LAKEBASE_BRANCH",
        "SUPERVISOR_DURABILITY",
        "SUPERVISOR_HISTORY_MAX_TOKENS",
        "WORKER_HISTORY_MAX_TOKENS",
        "OUTPUT_STREAM_WORKER_TOKENS",
    ):
        if os.getenv(passthrough):
            env_vars[passthrough] = os.environ[passthrough]

    deployment = agents.deploy(args.uc_model, version, environment_vars=env_vars)
    print(f"Deployed {args.uc_model} v{version}: {deployment.endpoint_name}")


if __name__ == "__main__":
    main()
