"""Which environment a process runs in, and what that permits.

`ENVIRONMENT` answers two different questions, and conflating them is a bug:

  1. *Which* deployment is this? It selects the prompt alias, the Unity Catalog
     schema and the Lakebase schema. Every deployed environment has its own.
  2. *May a control degrade here?* Only on a workstation or in CI.

A dev deployment is a deployment: it has a real store and the same failsafe
doctrine as prod. So the degradation set is the *local* one and `dev` is not in
it. A workstation run is `ENVIRONMENT=local`; a test runner leaves the variable
unset, which is the empty string below.
"""

from __future__ import annotations

import os

LOCAL_ENVIRONMENTS = frozenset({"local", "test", "testing", ""})

DEFAULT_CATALOG = "workspace"

# Which deployed environment a *workstation* run addresses. A local process has
# no resources of its own, so it borrows a deployed environment's — dev.
# A constant rather than a variable on purpose: pointing a lenient process at
# prod's tables should take a deliberate override of each name, not one edit.
LOCAL_READS = "dev"


def current_environment(environment: str | None = None) -> str:
    env = environment if environment is not None else os.getenv("ENVIRONMENT", "local")
    return (env or "").strip().lower()


def is_local_environment(environment: str | None = None) -> bool:
    """Whether this process runs somewhere a weakened control is acceptable.

    True only off Databricks. Every deployed environment, `dev` included, is
    held to the full contract: refuse to boot without a durable store, refuse to
    serve with a safety-critical setting disabled, never run a turn unbounded.
    """
    return current_environment(environment) in LOCAL_ENVIRONMENTS


def resource_environment(environment: str | None = None) -> str:
    """Which deployed environment's resources this process addresses.

    The other half of `is_local_environment`: that one answers "may a control
    degrade here?", this one answers "whose prompts, schema and tables?".
    """
    env = current_environment(environment)
    return LOCAL_READS if env in LOCAL_ENVIRONMENTS else env


def catalog() -> str:
    """The Unity Catalog catalog holding the platform's schemas.

    `PLATFORM_CATALOG` is the seam for moving to a catalog per environment;
    `SUPERVISOR_CATALOG` is honoured for deployments configured before the
    library existed.
    """
    return (
        os.getenv("PLATFORM_CATALOG", "").strip()
        or os.getenv("SUPERVISOR_CATALOG", "").strip()
        or DEFAULT_CATALOG
    )


def environment_schema(prefix: str, environment: str | None = None) -> str:
    """`<prefix>_dev`, `<prefix>_prod`, … — an agent's per-environment schema.

    Used unqualified as the Lakebase Postgres schema and qualified with the
    catalog as the Unity Catalog schema, so the two cannot drift.
    """
    return f"{prefix}_{resource_environment(environment)}"
