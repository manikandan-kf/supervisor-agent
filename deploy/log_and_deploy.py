"""Log the supervisor as an MLflow model, register it in Unity Catalog and
deploy it on Model Serving via the Databricks Agent Framework.

Run locally (with Databricks auth configured) or as the DAB job task:
    python deploy/log_and_deploy.py --routing-endpoint <endpoint> --uc-model <catalog.schema.model>
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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

AUDIT_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {table} (
  event_time      TIMESTAMP,
  request_id      STRING,
  conversation_id STRING,
  user_role       STRING,
  target_agent_id STRING,
  outcome         STRING,
  decision_trail  STRING
) USING DELTA
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uc-model",
        default=os.getenv("UC_MODEL_NAME", "workspace.supervisor.supervisor_agent"),
    )
    parser.add_argument(
        "--routing-endpoint",
        default=os.getenv("ROUTING_LLM_ENDPOINT", "databricks-claude-sonnet-4-5"),
    )
    parser.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "/Shared/supervisor-agent"))
    parser.add_argument("--environment", default=os.getenv("ENVIRONMENT", "dev"))
    parser.add_argument(
        "--prompt-catalog-schema",
        default=os.getenv("PROMPT_CATALOG_SCHEMA", "workspace.supervisor"),
        help="Unity Catalog catalog.schema holding the registered prompts (§4.1)",
    )
    parser.add_argument(
        "--prompt-alias",
        default=os.getenv("PROMPT_ALIAS", ""),
        help="prompt alias to load; defaults to --environment",
    )
    parser.add_argument("--audit-warehouse-id", default=os.getenv("AUDIT_WAREHOUSE_ID", ""))
    parser.add_argument("--audit-table", default=os.getenv("AUDIT_TABLE", "workspace.supervisor.audit_log"))
    parser.add_argument(
        "--lakebase-instance",
        default=os.getenv("LAKEBASE_INSTANCE", ""),
        help="Lakebase database instance for durable conversation memory, "
        "long-term memory and the Postgres audit sink (§4.4). Declared as a "
        "model resource so the endpoint's service identity can mint database "
        "credentials, and passed to the container as LAKEBASE_INSTANCE.",
    )
    parser.add_argument(
        "--lakebase-schema",
        default=os.getenv("LAKEBASE_SCHEMA", ""),
        help="Postgres schema inside the Lakebase instance that this deployment "
        "owns — checkpoints, long-term memory, the governed config table, the "
        "review queue and the audit trail all land in it. Set it per "
        "environment (supervisor_dev, supervisor_prod) so dev and prod share an "
        "instance without sharing state; leave empty for the single-environment "
        "shape, where everything uses `public`. Create it first with "
        "scripts/provision_lakebase.py --pg-schema.",
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
        "--mock-workers",
        action="store_true",
        help="deprecated alias for --workers simulated",
    )
    parser.add_argument(
        "--multi-model",
        choices=("false", "true"),
        default=os.getenv("MULTI_MODEL_ENABLED", "false").lower(),
        help="'true' turns on per-agent model references: each "
        "agents.yaml entry's optional `model` endpoint is declared as a "
        "resource and MULTI_MODEL_ENABLED=true is stamped onto the container. "
        "'false' (default) deploys "
        "with the feature dark: declared `model` values are ignored at "
        "runtime and their endpoints are not declared, so a reference to an "
        "endpoint that does not exist yet cannot fail the registration.",
    )
    args = parser.parse_args()

    # A bundle passes a value it can template; a human types the flag. Both
    # arrive here as one boolean.
    simulated_workers = args.mock_workers or args.workers == "simulated"
    multi_model = args.multi_model == "true"

    os.environ["ROUTING_LLM_ENDPOINT"] = args.routing_endpoint
    sys.path.insert(0, str(ROOT / "src"))

    import mlflow
    import mlflow.pyfunc
    import yaml
    from mlflow.models.resources import (
        DatabricksLakebase,
        DatabricksServingEndpoint,
        DatabricksSQLWarehouse,
        DatabricksTable,
    )

    if os.name == "nt":
        # MLflow resolves `python_model` with Path.resolve(), which records a
        # backslash Windows path in the MLmodel file; the Linux serving
        # container then can't find the code file (posix basename of
        # "C:\...\agent.py" is the whole string). Re-emit the validated path
        # as POSIX so the logged model stays loadable in Model Serving.
        _orig_validate = mlflow.pyfunc._validate_and_get_model_code_path

        def _posix_model_code_path(model_code_path, temp_dir):
            return Path(_orig_validate(model_code_path, temp_dir)).as_posix()

        mlflow.pyfunc._validate_and_get_model_code_path = _posix_model_code_path

    mlflow.set_registry_uri("databricks-uc")
    # Traces stay in the MLflow experiment's native store. The UC Delta
    # destination (`trace_location=mlflow.entities.trace_location.UnityCatalog`)
    # requires a customer external location; a metastore with only
    # Databricks-managed default storage rejects it with "Unsupported table
    # kind. Tables created in default storage are not supported".
    mlflow.set_experiment(experiment_name=args.experiment)

    agents_cfg = yaml.safe_load(
        (ROOT / "src" / "supervisor" / "config" / "agents.yaml").read_text(encoding="utf-8")
    )
    worker_endpoints = [a["endpoint"] for a in agents_cfg.get("agents", [])]

    # The audit table must exist before it is declared as a model resource —
    # Unity Catalog validates declared tables when the model version is created
    # and fails with TABLE_DOES_NOT_EXIST otherwise.
    if args.audit_warehouse_id:
        from databricks.sdk import WorkspaceClient

        # execute_statement is async by default and its wait_timeout caps at
        # 50s, which a cold serverless warehouse can exceed — so poll to a
        # terminal state rather than carrying on believing the table exists.
        client = WorkspaceClient()
        response = client.statement_execution.execute_statement(
            warehouse_id=args.audit_warehouse_id,
            statement=AUDIT_TABLE_DDL.format(table=args.audit_table),
            wait_timeout="50s",
        )

        def _state(resp) -> str:
            return getattr(getattr(getattr(resp, "status", None), "state", None), "value", "")

        deadline = time.monotonic() + 300
        while _state(response) in ("PENDING", "RUNNING"):
            if time.monotonic() > deadline:
                raise RuntimeError("timed out waiting for the audit table DDL (warehouse start?)")
            print(f"  waiting for warehouse... ({_state(response)})")
            time.sleep(10)
            response = client.statement_execution.get_statement(response.statement_id)

        if _state(response) != "SUCCEEDED":
            error = getattr(getattr(response.status, "error", None), "message", "")
            raise RuntimeError(f"audit table DDL did not succeed ({_state(response)}): {error}")
        print(f"Audit table ready: {args.audit_table}")

    # Declaring the endpoints as resources lets Databricks grant the deployed
    # supervisor's service identity Can Query on them automatically. With
    # --mock-workers only the routing LLM is declared, so the supervisor can be
    # deployed before any worker endpoint exists.
    resources = [DatabricksServingEndpoint(endpoint_name=args.routing_endpoint)]
    if not simulated_workers:
        resources += [DatabricksServingEndpoint(endpoint_name=e) for e in worker_endpoints]
    if multi_model:
        # Multi-model support: each agent's optional model endpoint needs the
        # same Can Query grant as the routing LLM, or the first screened turn
        # against that agent fails with a permission error instead of a
        # verdict. Declared only when the feature is on — while it is dark the
        # runtime never calls these, and declaring a not-yet-created endpoint
        # would fail the model registration for nothing.
        agent_model_endpoints = sorted(
            {a["model"].strip() for a in agents_cfg.get("agents", []) if a.get("model")}
            - {args.routing_endpoint}
        )
        resources += [
            DatabricksServingEndpoint(endpoint_name=e) for e in agent_model_endpoints
        ]
    if args.audit_warehouse_id:
        # Without these the endpoint's service identity has no SQL access, and
        # every audit INSERT fails with "This API is disabled for users without
        # the databricks-sql-access ... entitlement" — swallowed by the respond
        # node, so responses look healthy while the audit table stays empty.
        resources += [
            DatabricksSQLWarehouse(warehouse_id=args.audit_warehouse_id),
            DatabricksTable(table_name=args.audit_table),
        ]
    if args.lakebase_instance:
        # Grants the endpoint's service identity the ability to mint database
        # credentials for the instance, the same auth passthrough that covers
        # the LLM endpoint above. The Postgres-level role and schema grants are
        # separate — scripts/provision_lakebase.py applies those.
        resources += [DatabricksLakebase(database_instance_name=args.lakebase_instance)]

    # §9 release traceability: record which prompt versions this model version
    # shipped with, so "which prompt produced that answer?" is answerable from
    # the model version alone rather than by guessing what the alias pointed at
    # on the day. Best-effort — the alias must already resolve, and this identity
    # may not be able to read the registry (see README, Known limitations), in
    # which case the model still logs and only the lineage link is missing.
    prompt_uris: list[str] = []
    try:
        # `prompt_uri()` reads these from the environment, and they are not
        # stamped onto the endpoint until further down — so set them first, or
        # the recorded lineage points at the default catalog and alias rather
        # than the ones this deployment actually uses.
        os.environ["PROMPT_CATALOG_SCHEMA"] = args.prompt_catalog_schema
        os.environ["PROMPT_ALIAS"] = args.prompt_alias or args.environment

        from supervisor.prompt_provider import prompt_names, prompt_uri

        mlflow.set_registry_uri("databricks-uc")
        prompt_uris = [prompt_uri(name) for name in prompt_names()]
        print(f"Linking {len(prompt_uris)} prompt versions to the model: {prompt_uris}")
    except Exception as exc:  # noqa: BLE001 — lineage is additive, never fatal
        print(f"Prompt lineage skipped ({type(exc).__name__}: {exc})")

    with mlflow.start_run(run_name="supervisor-agent"):
        logged = mlflow.pyfunc.log_model(
            name="supervisor_agent",
            **({"prompts": prompt_uris} if prompt_uris else {}),
            # as_posix() so the recorded code path stays loadable inside the
            # Linux serving container when logging from Windows.
            python_model=(ROOT / "src" / "supervisor" / "agent.py").as_posix(),
            code_paths=[str(ROOT / "src" / "supervisor")],
            pip_requirements=str(ROOT / "requirements.txt"),
            resources=resources,
            input_example={
                "input": [{"role": "user", "content": "Generate an HLD for the new billing service"}],
                "custom_inputs": {"user_role": "BA", "agent_id": "requirement-agent"},
            },
            registered_model_name=args.uc_model,
        )

    if args.skip_deploy:
        print(f"Logged and registered {args.uc_model} v{logged.registered_model_version}")
        return

    from databricks import agents

    env_vars = {
        "ROUTING_LLM_ENDPOINT": args.routing_endpoint,
        "ENVIRONMENT": args.environment,
        # §4.1 — prompts load from Unity Catalog by name and environment alias.
        # Stated explicitly rather than left to a code default, so the deployed
        # agent's prompt source is visible in the endpoint configuration.
        "PROMPT_CATALOG_SCHEMA": args.prompt_catalog_schema,
        "PROMPT_ALIAS": args.prompt_alias or args.environment,
    }
    if simulated_workers:
        env_vars["SUPERVISOR_MOCK_WORKERS"] = "true"
    if multi_model:
        # Stamped only when on: the runtime default is off, and an absent
        # variable saying "off" is easier to reason about on the endpoint's
        # config page than a present-but-false one.
        env_vars["MULTI_MODEL_ENABLED"] = "true"
    if args.audit_warehouse_id:
        env_vars["AUDIT_WAREHOUSE_ID"] = args.audit_warehouse_id
        env_vars["AUDIT_TABLE"] = args.audit_table
    if args.lakebase_instance:
        # memory.py switches to the pooled Lakebase checkpointer/store when
        # this is present (§4.4) — durable conversations, long-term memory and
        # the Postgres audit sink, with credentials rotated by the pool.
        env_vars["LAKEBASE_INSTANCE"] = args.lakebase_instance
    if args.lakebase_schema:
        # What separates one deployed environment's durable state from
        # another's. Stamped explicitly rather than left to the passthrough loop
        # below: which schema a deployment writes to is a property of the
        # deployment, not of whatever shell the deploy job happened to run in,
        # and it has to be visible on the endpoint's configuration page next to
        # LAKEBASE_INSTANCE for either to be reviewable.
        env_vars["LAKEBASE_SCHEMA"] = args.lakebase_schema
    # The Autoscaling-generation address (instances created after March 12,
    # 2026) and the tuning knobs travel from the deploying environment to the
    # container verbatim, when set — memory.py and settings.py read them there.
    # DatabricksLakebase above only models provisioned instances; an
    # autoscaling project's credential grant is applied by
    # scripts/provision_lakebase.py instead.
    #
    # Guardrail knobs are deliberately absent from this list: a control that a
    # deploy shell can weaken by exporting a variable is not a control, so
    # OUTPUT_PII_MASKING, INPUT_MAX_CHARS and GUARDRAIL_BLOCK_STREAK_LIMIT stay
    # on their code defaults (and `Settings.enforce` refuses to serve a
    # deployed environment where one of them is off).
    #
    # OUTPUT_STREAM_WORKER_TOKENS is the exception, because its only non-default
    # value *strengthens* the layer-7 screen: `false` stops relaying worker
    # tokens live, so the closing item carries the whole guarded answer and the
    # streaming residual closes entirely. An operator who wants that posture
    # should not need a code change to get it, and passing it through cannot
    # loosen anything.
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

    deployment = agents.deploy(args.uc_model, logged.registered_model_version, environment_vars=env_vars)
    print(f"Deployed {args.uc_model} v{logged.registered_model_version}: {deployment.endpoint_name}")


if __name__ == "__main__":
    main()
