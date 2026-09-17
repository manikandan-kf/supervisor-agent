"""Log the supervisor as an MLflow model, register it in Unity Catalog, deploy it on Serving.

Run as the DAB job task, or locally: `python deploy/deploy_agent.py --uc-model <c.s.m>`.
`--deploy-method agents` (default) uses `databricks.agents.deploy()`; `serving-api` drives the
Model Serving SDK directly for workspaces where the Agent Framework registration or the AI
Gateway permission check fails. Endpoint credentials come from the logged `resources` either
way. The governance wheel is baked into the model artifact (`wheels/<file>.whl` in its
requirements) so a serving container never depends on a volume or index at build time.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

# What `databricks.agents.deploy()` stamps on every agent endpoint; reproduced
# on the serving-api path so a container behaves the same whichever created it.
_AGENT_ENV_VARS = {
    "ENABLE_LANGCHAIN_STREAMING": "true",
    "ENABLE_MLFLOW_TRACING": "true",
    "RETURN_REQUEST_ID_IN_RESPONSE": "true",
}
_MONITOR_TAG = "MONITOR_EXPERIMENT_ID"
_NAME_MAX = 63  # Model Serving's limit for endpoint and served-entity names


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
LIBRARY = ROOT / "libs" / "agent_governance"


def _governance_wheel() -> Path:
    """The agent_governance wheel to bake into the model artifact.

    Built by `databricks bundle deploy`; a workstation run that skipped the build gets one from
    the same source, so the two paths cannot ship different code.
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


def _sanitize(uc_name: str) -> str:
    return uc_name.replace(".", "-").rstrip("-_")


def endpoint_name(uc_model: str) -> str:
    """`agents_<catalog>-<schema>-<model>`, derived exactly as databricks.agents does.

    Docs and grants address the endpoint by this name, so both deploy methods must
    agree. Checked against `_create_endpoint_name` in databricks-agents 1.11.0.
    """
    prefix = "agents_"
    return prefix + _sanitize(uc_model[: _NAME_MAX - len(prefix)])


def served_entity_name(uc_model: str, version: int | str) -> str:
    suffix = f"_{version}"
    return _sanitize(uc_model[: _NAME_MAX - len(suffix)]) + suffix


def deploy_via_serving_api(
    w,
    uc_model: str,
    version: int | str,
    name: str,
    env_vars: dict[str, str],
    experiment_id: str | None,
    *,
    workload_size: str,
    scale_to_zero: bool,
) -> None:
    """Create or roll the endpoint with the Model Serving SDK.

    Unlike `agents.deploy()`: one served entity (no creep towards the 15-entity cap); inference
    tables requested after create and non-fatal, since requesting them at create time trips the
    AI Gateway permission check this path exists to avoid; no Agent Framework/Review App entry.
    """
    from databricks.sdk.errors import NotFound, ResourceConflict
    from databricks.sdk.service.serving import (
        AiGatewayInferenceTableConfig,
        EndpointCoreConfigInput,
        EndpointStateConfigUpdate,
        EndpointTag,
        Route,
        ServedEntityInput,
        TrafficConfig,
    )

    served = ServedEntityInput(
        name=served_entity_name(uc_model, version),
        entity_name=uc_model,
        entity_version=str(version),
        workload_size=workload_size,
        scale_to_zero_enabled=scale_to_zero,
        environment_vars=env_vars,
    )
    traffic = TrafficConfig(routes=[Route(served_model_name=served.name, traffic_percentage=100)])

    try:
        existing = w.serving_endpoints.get(name)
    except NotFound:
        existing = None

    if existing is None:
        tags = [EndpointTag(key=_MONITOR_TAG, value=experiment_id)] if experiment_id else None
        w.serving_endpoints.create(
            name=name,
            config=EndpointCoreConfigInput(
                name=name, served_entities=[served], traffic_config=traffic
            ),
            tags=tags,
        )
        print(f"Created endpoint {name} serving {served.name}")
    else:
        state = existing.state.config_update if existing.state else None
        if state == EndpointStateConfigUpdate.IN_PROGRESS:
            # Same refusal as agents.deploy(): the service rejects a rollout on top of a
            # running one, and returning quietly would strand the version just registered.
            raise SystemExit(
                f"Endpoint {name} is currently updating; wait for NOT_UPDATING "
                f"and rerun (the registered version is not lost)."
            )
        try:
            w.serving_endpoints.update_config(
                name=name, served_entities=[served], traffic_config=traffic
            )
        except ResourceConflict as exc:
            raise SystemExit(f"Endpoint {name} is currently updating: {exc}") from exc
        current = {t.key: t.value for t in (existing.tags or [])}
        if experiment_id and current.get(_MONITOR_TAG) != experiment_id:
            w.serving_endpoints.patch(
                name=name, add_tags=[EndpointTag(key=_MONITOR_TAG, value=experiment_id)]
            )
        print(f"Rolling endpoint {name} to {served.name}")

    gateway = getattr(existing, "ai_gateway", None) if existing else None
    tables = getattr(gateway, "inference_table_config", None)
    if tables is not None and tables.enabled:
        return
    catalog, schema, model = uc_model.split(".")
    try:
        w.serving_endpoints.put_ai_gateway(
            name,
            inference_table_config=AiGatewayInferenceTableConfig(
                enabled=True, catalog_name=catalog, schema_name=schema, table_name_prefix=model
            ),
        )
        print(f"Inference tables enabled: {catalog}.{schema}.{model}_payload")
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print(
            f"Inference tables not enabled ({type(exc).__name__}: {exc}). The endpoint is "
            "deployed; enable payload logging from its AI Gateway tab if you need it."
        )


def wait_until_serving(w, name: str, minutes: int) -> None:
    """Block until the rollout finishes, failing the job if the endpoint does not serve."""
    print(f"Waiting up to {minutes} min for {name} to finish updating")
    try:
        endpoint = w.serving_endpoints.wait_get_serving_endpoint_not_updating(
            name, timeout=timedelta(minutes=minutes)
        )
    except Exception as exc:  # noqa: BLE001 — any waiter failure is a deploy failure
        raise SystemExit(f"Endpoint {name} did not reach NOT_UPDATING: {exc}") from exc
    ready = endpoint.state.ready.value if endpoint.state and endpoint.state.ready else "unknown"
    print(f"Endpoint {name}: ready={ready}")
    if ready != "READY":
        raise SystemExit(f"Endpoint {name} finished updating but is not READY ({ready})")


def _bind_experiment(mlflow, experiment: str, trace_catalog_schema: str) -> None:
    """Point the experiment at Unity Catalog Delta tables for tracing (§08).

    Bound here, not in the agent: the trace destination is a deployment property, and a serving
    container has no business choosing its own audit destination. Needs MLflow >= 3.14 and an
    existing schema; on failure the experiment store is kept — losing it must not lose the deploy.
    """
    if not trace_catalog_schema:
        mlflow.set_experiment(experiment_name=experiment)
        return
    catalog, _, schema = trace_catalog_schema.partition(".")
    if not catalog or not schema:
        raise SystemExit(
            f"--trace-catalog-schema must be catalog.schema, got {trace_catalog_schema!r}"
        )
    from mlflow.entities.trace_location import UnityCatalog

    try:
        mlflow.set_experiment(
            experiment_name=experiment,
            trace_location=UnityCatalog(catalog_name=catalog, schema_name=schema),
        )
        print(f"Traces persist to Unity Catalog Delta tables in {trace_catalog_schema}")
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print(
            f"UC trace location unavailable ({type(exc).__name__}: {exc}); using the experiment store"
        )
        mlflow.set_experiment(experiment_name=experiment)


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
    parser.add_argument(
        "--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "/Shared/supervisor-agent-dev")
    )
    parser.add_argument(
        "--trace-catalog-schema",
        default=os.getenv("TRACE_CATALOG_SCHEMA", ""),
        help="catalog.schema that MLflow traces are persisted to as Unity Catalog "
        "Delta tables (solution §08). Empty leaves traces in the experiment's own "
        "store, which is capped per experiment and not queryable from SQL.",
    )
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
        "memory, the governed config table and the audit sink. "
        "Declared as a model resource so the endpoint's service identity can mint "
        "database credentials, and passed to the container as LAKEBASE_INSTANCE.",
    )
    parser.add_argument(
        "--lakebase-resource",
        choices=("declare", "skip"),
        default=os.getenv("LAKEBASE_RESOURCE", "declare"),
        help="'declare' (default) registers the instance as a model resource, so Model "
        "Serving issues the endpoint short-lived per-resource credentials. 'skip' stamps "
        "LAKEBASE_INSTANCE without declaring it, for a workspace that refuses the "
        "passthrough (it is accepted only from a workspace admin); the container then "
        "authenticates with DATABRICKS_CLIENT_ID/SECRET instead. DEPLOYMENT.md §6a.",
    )
    parser.add_argument(
        "--lakebase-project",
        default=os.getenv("LAKEBASE_PROJECT", ""),
        help="Lakebase Autoscaling project, for a workspace with a project rather than a "
        "provisioned instance. Stamped onto the container as LAKEBASE_PROJECT and, with "
        "--lakebase-branch, is the address the runtime resolves first. No resource is "
        "declared for it — the MLflow resource type names an instance — so the endpoint "
        "reaches it with the credentials from --endpoint-secret-scope. DEPLOYMENT.md §6a.",
    )
    parser.add_argument(
        "--lakebase-branch",
        default=os.getenv("LAKEBASE_BRANCH", ""),
        help="branch inside --lakebase-project; the project's default branch when empty",
    )
    parser.add_argument(
        "--endpoint-secret-scope",
        default=os.getenv("ENDPOINT_SECRET_SCOPE", ""),
        help="Databricks secret scope holding DATABRICKS_HOST, DATABRICKS_CLIENT_ID and "
        "DATABRICKS_CLIENT_SECRET for the serving container. Stamps them as "
        "{{secrets/<scope>/<key>}} references, which is the only way to pass them from a "
        "job task: setting them as real environment variables would break this script's own "
        "workspace calls. Needed only on the fallback path, where no Lakebase resource can "
        "be declared. DEPLOYMENT.md §6a.",
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
        "--deploy-method",
        choices=("agents", "serving-api"),
        default=os.getenv("DEPLOY_METHOD", "agents"),
        help="'agents' uses databricks.agents.deploy(); 'serving-api' creates or rolls the "
        "endpoint with the Model Serving SDK directly, for workspaces where agents.deploy() "
        "fails (see the module docstring). Same endpoint name either way.",
    )
    parser.add_argument(
        "--endpoint-name",
        default=os.getenv("ENDPOINT_NAME", ""),
        help="override the derived agents_<catalog>-<schema>-<model> endpoint name",
    )
    parser.add_argument(
        "--workload-size", default=os.getenv("WORKLOAD_SIZE", "Small"), help="Small|Medium|Large"
    )
    parser.add_argument(
        "--scale-to-zero",
        choices=("false", "true"),
        default=os.getenv("SCALE_TO_ZERO", "false").lower(),
        help="scale the endpoint to zero when idle; the first request then pays a cold start",
    )
    parser.add_argument(
        "--wait-minutes",
        type=int,
        default=int(os.getenv("DEPLOY_WAIT_MINUTES", "0")),
        help="block until the rollout finishes and fail the job if the endpoint is not READY; "
        "0 (default) returns as soon as the rollout is initiated, as agents.deploy() does",
    )
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
        # MLflow records `python_model` via Path.resolve(), i.e. a backslash Windows path the
        # Linux serving container cannot find; re-emit it as POSIX so the model stays loadable.
        _orig_validate = mlflow.pyfunc._validate_and_get_model_code_path

        def _posix_model_code_path(model_code_path, temp_dir):
            return Path(_orig_validate(model_code_path, temp_dir)).as_posix()

        mlflow.pyfunc._validate_and_get_model_code_path = _posix_model_code_path

    mlflow.set_registry_uri("databricks-uc")
    _bind_experiment(mlflow, args.experiment, args.trace_catalog_schema)

    agents_cfg = yaml.safe_load(
        (ROOT / "src" / "supervisor" / "config" / "agents.yaml").read_text(encoding="utf-8")
    )
    worker_endpoints = [a["endpoint"] for a in agents_cfg.get("agents", [])]

    # Declared resources get the supervisor's service identity Can Query automatically. With
    # simulated workers only the routing LLM is declared, so deploy can precede the workers.
    resources = [DatabricksServingEndpoint(endpoint_name=args.routing_endpoint)]
    if not simulated_workers:
        resources += [DatabricksServingEndpoint(endpoint_name=e) for e in worker_endpoints]
    if multi_model:
        # Model endpoints need the same Can Query grant; declared only when the feature is on,
        # since declaring a not-yet-created endpoint would fail the registration for nothing.
        agent_model_endpoints = sorted(
            {a["model"].strip() for a in agents_cfg.get("agents", []) if a.get("model")}
            - {args.routing_endpoint}
        )
        resources += [DatabricksServingEndpoint(endpoint_name=e) for e in agent_model_endpoints]
    if args.lakebase_instance and args.lakebase_resource == "declare":
        # Lets the endpoint's service identity mint database credentials for the instance;
        # Postgres-level grants are separate.
        resources += [DatabricksLakebase(database_instance_name=args.lakebase_instance)]
    elif args.lakebase_instance:
        # LAKEBASE_INSTANCE is still stamped below — the container needs a target or it
        # refuses to boot. Only the credential path changes.
        print(
            "Lakebase declared as a resource: no (--lakebase-resource skip). The endpoint "
            "must reach it with DATABRICKS_CLIENT_ID/SECRET; see DEPLOYMENT.md §6a."
        )
    elif args.lakebase_project:
        # An Autoscaling project cannot be declared at all: DatabricksLakebase names an
        # instance. The address is stamped below and the credentials come from the secret
        # scope, so say so here rather than let a missing resource look like an oversight.
        print(
            f"Lakebase: project={args.lakebase_project}"
            + (f", branch={args.lakebase_branch}" if args.lakebase_branch else "")
            + " — not declarable as a model resource (the resource type names an instance)."
        )
        if not args.endpoint_secret_scope and not os.getenv("DATABRICKS_CLIENT_ID"):
            print(
                "  WARNING: no --endpoint-secret-scope and no DATABRICKS_CLIENT_ID. The "
                "container will have no workspace credentials, fail its first Lakebase "
                "connection and refuse to serve. See DEPLOYMENT.md §6a."
            )

    # Release traceability: which prompt versions this model version shipped with. Best-effort:
    # the alias must resolve and this identity may not be able to read the registry.
    prompt_uris: list[str] = []
    try:
        os.environ["PROMPT_CATALOG_SCHEMA"] = args.prompt_catalog_schema
        os.environ["PROMPT_ALIAS"] = args.prompt_alias or args.environment

        from supervisor.prompt_registry import prompt_names, prompt_uri

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
            python_model=(ROOT / "src" / "supervisor" / "serving_entrypoint.py").as_posix(),
            code_paths=[str(ROOT / "src" / "supervisor")],
            # The runtime pins plus the library wheel, resolved from the model
            # directory at container build — see the module docstring.
            pip_requirements=[*_runtime_requirements(), f"wheels/{wheel.name}"],
            resources=resources,
            input_example={
                "input": [
                    {"role": "user", "content": "Generate an HLD for the new billing service"}
                ],
                "custom_inputs": {"user_role": "BA", "agent_id": "requirement-agent"},
            },
        )
        # The wheel goes in *before* registration: Unity Catalog copies the model directory when
        # the version is created, so anything added afterwards would not reach the endpoint.
        with tempfile.TemporaryDirectory() as staging:
            wheels_dir = Path(staging) / "wheels"
            wheels_dir.mkdir()
            shutil.copy2(wheel, wheels_dir / wheel.name)
            MlflowClient().log_model_artifacts(logged.model_id, staging)

    version = mlflow.register_model(logged.model_uri, args.uc_model).version

    if args.skip_deploy:
        print(f"Logged and registered {args.uc_model} v{version}")
        return

    env_vars = {
        "ROUTING_LLM_ENDPOINT": args.routing_endpoint,
        "ENVIRONMENT": args.environment,
        # Stated explicitly so the prompt source is visible in the endpoint configuration.
        "PROMPT_CATALOG_SCHEMA": args.prompt_catalog_schema,
        "PROMPT_ALIAS": args.prompt_alias or args.environment,
    }
    if simulated_workers:
        env_vars["SUPERVISOR_MOCK_WORKERS"] = "true"
    if multi_model:
        env_vars["MULTI_MODEL_ENABLED"] = "true"
    if args.lakebase_instance:
        env_vars["LAKEBASE_INSTANCE"] = args.lakebase_instance
    if args.lakebase_project:
        # Flags rather than environment passthrough, because a bundle job task can pass
        # parameters and cannot set environment variables. Both default from the environment,
        # so a workstation run behaves as it always did.
        env_vars["LAKEBASE_PROJECT"] = args.lakebase_project
    if args.lakebase_branch:
        env_vars["LAKEBASE_BRANCH"] = args.lakebase_branch
    if args.lakebase_schema:
        # The schema a deployment writes to is a deployment property; stamped explicitly so it
        # is reviewable on the endpoint's configuration page next to LAKEBASE_INSTANCE.
        env_vars["LAKEBASE_SCHEMA"] = args.lakebase_schema
    # Lakebase address and tuning knobs pass through verbatim when set. Guardrail knobs are
    # deliberately absent — a control a deploy shell can weaken is not a control; `Settings.enforce`
    # refuses to serve with one off.
    for passthrough in (
        "LAKEBASE_AUTOSCALING_ENDPOINT",
        "SUPERVISOR_DURABILITY",
        "SUPERVISOR_HISTORY_MAX_TOKENS",
        "WORKER_HISTORY_MAX_TOKENS",
        # Fallback workspace credentials for the SDK's default chain, used when the
        # declared-resource passthrough cannot be granted — today a Lakebase dependency
        # needs a workspace admin to create the endpoint. Pass `{{secrets/<scope>/<key>}}`
        # references, never literals: the value below is stamped onto the endpoint
        # configuration, where a literal would be readable by anyone with CAN_VIEW.
        #
        # This is a downgrade, not a preference. The container then acts as one static
        # principal for *every* SDK call instead of the endpoint's own short-lived,
        # per-resource identity, and the secret has to be rotated by hand. Use a
        # dedicated least-privilege principal, and drop these once the deploy identity
        # can hold the passthrough. DEPLOYMENT.md §6a.
        "DATABRICKS_HOST",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
    ):
        if os.getenv(passthrough):
            env_vars[passthrough] = os.environ[passthrough]
    if args.endpoint_secret_scope:
        # The same three credentials, addressed instead of read. This wins over the
        # passthrough above on purpose: a shell that happens to hold a literal secret must not
        # be able to stamp it onto the endpoint configuration once a scope has been named.
        scope = args.endpoint_secret_scope
        for key in ("DATABRICKS_HOST", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"):
            env_vars[key] = f"{{{{secrets/{scope}/{key}}}}}"
        print(
            f"Workspace credentials for the container: {{{{secrets/{scope}/…}}}} "
            "(references, not values)"
        )

    name = args.endpoint_name or endpoint_name(args.uc_model)
    scale_to_zero = args.scale_to_zero == "true"

    if args.deploy_method == "agents":
        from databricks import agents

        deployment = agents.deploy(
            args.uc_model,
            version,
            environment_vars=env_vars,
            endpoint_name=name,
            workload_size=args.workload_size,
            scale_to_zero=scale_to_zero,
        )
        name = deployment.endpoint_name
        print(f"Deployed {args.uc_model} v{version}: {name}")
    else:
        from databricks.sdk import WorkspaceClient

        experiment_id = None
        try:
            found = mlflow.get_experiment_by_name(args.experiment)
            experiment_id = found.experiment_id if found else None
        except Exception as exc:  # noqa: BLE001 — tracing is additive, never fatal
            print(f"Experiment id unavailable ({type(exc).__name__}: {exc})")
        if not experiment_id:
            print("No experiment id: the endpoint will not stream traces to MLflow")
        env_vars.update(_AGENT_ENV_VARS)
        if experiment_id:
            env_vars["MLFLOW_EXPERIMENT_ID"] = experiment_id
        w = WorkspaceClient()
        deploy_via_serving_api(
            w,
            args.uc_model,
            version,
            name,
            env_vars,
            experiment_id,
            workload_size=args.workload_size,
            scale_to_zero=scale_to_zero,
        )
        print(f"Deployed {args.uc_model} v{version}: {name}")

    if args.wait_minutes > 0:
        from databricks.sdk import WorkspaceClient

        wait_until_serving(WorkspaceClient(), name, args.wait_minutes)


if __name__ == "__main__":
    main()
