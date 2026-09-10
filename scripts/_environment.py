"""Which deployed environment an operator script is pointed at.

Every deployed environment owns its own Unity Catalog schema, registered model,
serving endpoint, prompt alias, MLflow experiment and Lakebase Postgres schema —
see the table at the top of `databricks.yml`. All of those names follow from two
things: the environment's name and the catalog. This module is the one place
that derivation lives, so a script cannot address dev's endpoint while reading
prod's prompts.

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    target = resolve(args)

    w.serving_endpoints.get(target.endpoint)

`--environment` defaults to `$ENVIRONMENT`, so an operator who has sourced the
`.env` for one environment stays in it, and switching is one flag:

    python scripts/verify_deployment.py -e prod

Every derived name can still be overridden individually, for the cases the
convention does not cover — a second isolated copy of one environment, or a
deployment made before this convention existed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Mirrors `databricks.agents.deployments._create_endpoint_name`. Reproduced
# rather than imported because `databricks-agents` is a deploy-time dependency
# and these scripts run against an endpoint that already exists — including from
# machines that have no reason to install it.
_ENDPOINT_PREFIX = "agents_"
_MAX_ENDPOINT_NAME_LEN = 63
_INVALID_ENDPOINT_NAME_SUFFIX_CHARS = "_-"

DEFAULT_CATALOG = "workspace"
DEFAULT_MODEL = "supervisor_agent"


def endpoint_name(uc_model: str) -> str:
    """The endpoint `databricks.agents.deploy()` creates for a UC model name.

    Not a guess: `agents_` plus the model's fully qualified name with dots
    turned into dashes, truncated to the 63-character endpoint limit and
    stripped of a trailing separator. Getting this wrong is not a small error —
    a script that derives the wrong name reports "endpoint not found" for a
    healthy deployment, or verifies the wrong environment's.
    """
    truncated = uc_model[: _MAX_ENDPOINT_NAME_LEN - len(_ENDPOINT_PREFIX)]
    sanitized = truncated.replace(".", "-").rstrip(_INVALID_ENDPOINT_NAME_SUFFIX_CHARS)
    return _ENDPOINT_PREFIX + sanitized


@dataclass(frozen=True)
class Target:
    """One deployed environment, and everything named after it."""

    environment: str
    catalog: str
    schema: str
    model: str

    @property
    def uc_schema(self) -> str:
        """`catalog.schema` — where the prompts, model and Delta tables live."""
        return f"{self.catalog}.{self.schema}"

    @property
    def uc_model(self) -> str:
        return f"{self.uc_schema}.{self.model}"

    @property
    def endpoint(self) -> str:
        return endpoint_name(self.uc_model)

    @property
    def prompt_alias(self) -> str:
        """The promoted prompt alias, which the bundle sets to the target name."""
        return self.environment

    @property
    def experiment(self) -> str:
        return f"/Shared/supervisor-agent-{self.environment}"

    @property
    def lakebase_schema(self) -> str:
        """The Postgres schema holding this environment's durable state."""
        return self.schema

    def describe(self) -> str:
        return (
            f"environment : {self.environment}\n"
            f"  model     : {self.uc_model}\n"
            f"  endpoint  : {self.endpoint}\n"
            f"  prompts   : {self.uc_schema} @{self.prompt_alias}\n"
            f"  lakebase  : schema {self.lakebase_schema}"
        )


def add_arguments(parser, *, model: bool = True) -> None:
    """Add `--environment` and the per-name overrides to an argument parser."""
    parser.add_argument(
        "-e",
        "--environment",
        default=os.getenv("ENVIRONMENT", "dev"),
        help="deployed environment to address (dev, prod, …). Every other name "
        "below is derived from it; defaults to $ENVIRONMENT, else dev.",
    )
    parser.add_argument(
        "--catalog",
        default=os.getenv("SUPERVISOR_CATALOG", DEFAULT_CATALOG),
        help=f"Unity Catalog catalog (default {DEFAULT_CATALOG})",
    )
    parser.add_argument(
        "--schema",
        default="",
        help="Unity Catalog schema, and the Lakebase schema. Defaults to "
        "supervisor_<environment>; override only for a deployment that does not "
        "follow the convention.",
    )
    if model:
        parser.add_argument("--model", default=DEFAULT_MODEL)


def resolve(args) -> Target:
    """The `Target` an operator asked for, applying the naming convention."""
    environment = (getattr(args, "environment", "") or "dev").strip()
    schema = (getattr(args, "schema", "") or "").strip() or f"supervisor_{environment}"
    return Target(
        environment=environment,
        catalog=getattr(args, "catalog", "") or DEFAULT_CATALOG,
        schema=schema,
        model=getattr(args, "model", "") or DEFAULT_MODEL,
    )
