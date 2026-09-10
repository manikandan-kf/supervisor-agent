"""Poll the agent serving endpoint until it is ready and serving one version.

    python scripts/wait_for_endpoint.py                 # the environment in $ENVIRONMENT
    python scripts/wait_for_endpoint.py -e prod
    python scripts/wait_for_endpoint.py --timeout 1800

`databricks.agents.deploy()` returns as soon as the deployment is *initiated*.
The endpoint keeps serving the previous version until the new container is
healthy, then shifts traffic — which takes another 10-15 minutes. This waits for
that, so a verification step does not run against the old version.

Exits 0 when ready, 1 on timeout.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _environment  # noqa: E402  — needs the path insert above


def _authenticate() -> None:
    """Load `.env` before constructing the SDK client.

    The Databricks *CLI* finds credentials on its own, so `bundle deploy` works
    in a bare shell — but this script uses the *SDK*, which does not read `.env`.
    Without this, chaining `wait_for_endpoint.py` before a deploy dies on
    "cannot configure default credentials" even though the deploy itself would
    have authenticated fine.
    """
    from _dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if os.getenv("DATABRICKS_CLIENT_ID") and os.getenv("DATABRICKS_CLIENT_SECRET"):
        # An OAuth M2M service principal is non-interactive; a CLI profile may
        # want a browser, which this script cannot provide.
        os.environ.setdefault("DATABRICKS_AUTH_TYPE", "oauth-m2m")


def _routes(endpoint) -> dict[str, int]:
    """Traffic split, or empty while the first version is still building.

    On a first-ever deployment `config` is None — there is no live config yet,
    only `pending_config`. Reading `.config.traffic_config` directly raises
    AttributeError until the endpoint has served something.
    """
    for attr in ("config", "pending_config"):
        config = getattr(endpoint, attr, None)
        traffic = getattr(config, "traffic_config", None) if config else None
        routes = getattr(traffic, "routes", None) if traffic else None
        if routes:
            return {r.served_model_name: r.traffic_percentage for r in routes}
    return {}


def main() -> int:
    # Before the parser is built, not after: `--environment` defaults to
    # $ENVIRONMENT, and `.env` is where an operator sets it.
    _authenticate()

    parser = argparse.ArgumentParser()
    _environment.add_arguments(parser)
    parser.add_argument("--endpoint", default="", help="override the derived endpoint name")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds")
    parser.add_argument("--interval", type=int, default=30, help="seconds between polls")
    args = parser.parse_args()

    endpoint = args.endpoint or _environment.resolve(args).endpoint
    print(f"waiting on {endpoint}", flush=True)

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    deadline = time.monotonic() + args.timeout

    while time.monotonic() < deadline:
        try:
            ep = w.serving_endpoints.get(endpoint)
        except Exception as exc:
            print(f"  endpoint not found yet ({type(exc).__name__})", flush=True)
            time.sleep(args.interval)
            continue

        routes = _routes(ep)
        ready = str(getattr(ep.state, "ready", ""))
        updating = str(getattr(ep.state, "config_update", ""))
        print(f"  ready={ready}  update={updating}  {routes or '(building)'}", flush=True)

        # UPDATE_FAILED is terminal — it never becomes NOT_UPDATING, so polling on
        # would burn the whole timeout to report nothing. Say why it failed
        # instead: the per-entity message carries the actual cause, and the
        # commonest one here is the account's served-entity cap, hit because every
        # `agents.deploy()` *adds* an entity rather than replacing the last.
        if "UPDATE_FAILED" in updating:
            print("\nUPDATE FAILED — the previous version is still serving.")
            for entity in getattr(getattr(ep, "pending_config", None), "served_entities", []) or []:
                message = getattr(getattr(entity, "state", None), "deployment_state_message", "")
                if message:
                    print(f"  v{getattr(entity, 'entity_version', '?')}: {message}")
            return 1

        # A live `config` (not just `pending_config`) plus NOT_UPDATING is what
        # actually means "serving"; traffic alone can be reported while the
        # first container is still being built.
        if (
            getattr(ep, "config", None) is not None
            and "NOT_UPDATING" in updating
            and 100 in routes.values()
        ):
            live = [n for n, p in routes.items() if p == 100]
            print(f"\nREADY — serving {live[0]} at 100%")
            return 0

        time.sleep(args.interval)

    print(f"\nTIMED OUT after {args.timeout}s. Check the endpoint's build logs in the UI.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
