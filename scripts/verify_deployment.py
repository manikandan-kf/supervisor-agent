"""Post-deploy verification: what is registered, and what the endpoint reads.

    python scripts/verify_deployment.py            # the environment in $ENVIRONMENT
    python scripts/verify_deployment.py -e prod

Every name it checks — endpoint, schema, prompt alias — is derived from the
environment (`_environment.py`), so verifying dev cannot accidentally report on
prod's deployment.

Answers four questions in one pass:

  1. Is the endpoint serving, and which version?
  2. Are the prompts registered in Unity Catalog?
  3. Does the *endpoint* load them, or fall back to the bundled defaults?
  4. Is the audit table receiving rows?

Question 3 is the one that needs the endpoint's own logs. A client can load the
prompts perfectly well while the endpoint's service principal cannot — they are
different identities with different Unity Catalog grants.
"""

from __future__ import annotations

import argparse
import sys

import _environment


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main() -> int:
    parser = argparse.ArgumentParser()
    _environment.add_arguments(parser)
    parser.add_argument(
        "--endpoint", default="", help="override the derived endpoint name"
    )
    parser.add_argument("--alias", default="", help="override the derived prompt alias")
    # No default: the id is workspace-specific, and the Delta audit table is
    # unwired anyway — the Postgres sink on Lakebase is the real audit store.
    parser.add_argument("--warehouse-id", default="")
    args = parser.parse_args()

    target = _environment.resolve(args)
    endpoint = args.endpoint or target.endpoint
    alias = args.alias or target.prompt_alias
    print(target.describe())

    import mlflow
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    # ── 1. endpoint ────────────────────────────────────────────────────────
    section("1. SERVING ENDPOINT")
    served_name = None
    try:
        ep = w.serving_endpoints.get(endpoint)
        # `config` is None on a first-ever deployment — only `pending_config`
        # exists until the first container is healthy.
        routes: dict[str, int] = {}
        for attr in ("config", "pending_config"):
            config = getattr(ep, attr, None)
            traffic = getattr(config, "traffic_config", None) if config else None
            found = getattr(traffic, "routes", None) if traffic else None
            if found:
                routes = {r.served_model_name: r.traffic_percentage for r in found}
                break

        print(f"  ready  : {getattr(ep.state, 'ready', '')}")
        print(f"  update : {getattr(ep.state, 'config_update', '')}")
        if not routes:
            print("  no traffic routes yet — still building")
        for name, pct in routes.items():
            print(f"  {name}: {pct}%")
            if pct == 100 and getattr(ep, "config", None) is not None:
                served_name = name
    except Exception as exc:
        print(f"  NOT FOUND ({type(exc).__name__}: {str(exc)[:120]})")
        return 1

    # ── 2. prompts registered ──────────────────────────────────────────────
    section("2. PROMPTS IN UNITY CATALOG")
    try:
        rows = list(
            mlflow.genai.search_prompts(
                filter_string=f"catalog = '{target.catalog}' AND schema = '{target.schema}'"
            )
        )
        for r in rows:
            try:
                v = mlflow.genai.load_prompt(f"prompts:/{r.name}@{alias}")
                print(f"  {r.name}  v{v.version} @{alias}  ({len(v.template)} chars)")
            except Exception as exc:
                print(f"  {r.name}  no @{alias} alias ({type(exc).__name__})")
        if not rows:
            print("  NONE registered — run scripts/register_prompts.py")
    except Exception as exc:
        print(f"  cannot list ({type(exc).__name__}: {str(exc)[:120]})")

    # ── 3. does the endpoint read them? ────────────────────────────────────
    section("3. WHAT THE ENDPOINT ACTUALLY LOADS")
    if not served_name:
        print("  no version at 100% traffic yet")
    else:
        try:
            text = w.serving_endpoints.logs(endpoint, served_name).logs or ""
            hits = [
                ln.strip()
                for ln in text.splitlines()
                if "prompt registry" in ln.lower() or "loaded prompt" in ln.lower()
            ]
            if not hits:
                print("  No prompt lines in the log buffer yet.")
                print("  Send one request through the calling UI, then re-run this.")
            else:
                for ln in hits[-6:]:
                    print(f"  {ln[:190]}")
                loaded = any("loaded prompt" in h.lower() for h in hits)
                fell_back = any("unavailable" in h.lower() for h in hits)
                print()
                if loaded and not fell_back:
                    print("  VERDICT: reading from the registry — §4.1 satisfied end to end.")
                elif fell_back:
                    print("  VERDICT: falling back to bundled defaults.")
                    print("           Prompts are still correct; the endpoint's service")
                    print("           principal cannot read them from Unity Catalog.")
        except Exception as exc:
            print(f"  cannot read logs ({type(exc).__name__}: {str(exc)[:120]})")

    # ── 4. audit table ─────────────────────────────────────────────────────
    section("4. AUDIT TABLE")
    if not args.warehouse_id:
        print("  skipped — no --warehouse-id given. The Delta audit table is unwired")
        print("  in this environment; the real audit store is supervisor_audit_log on")
        print("  the Lakebase/Postgres sink.")
        return 0
    table = f"{target.uc_schema}.audit_log"
    try:
        resp = w.statement_execution.execute_statement(
            warehouse_id=args.warehouse_id,
            statement=f"SELECT count(*) AS n FROM {table}",
            wait_timeout="30s",
        )
        state = getattr(getattr(resp.status, "state", None), "value", "")
        if state == "SUCCEEDED" and resp.result and resp.result.data_array:
            count = resp.result.data_array[0][0]
            print(f"  {table}: {count} row(s)")
            if str(count) == "0":
                print("  Expected while the endpoint's service principal lacks")
                print("  databricks-sql-access — see DEPLOYMENT.md.")
        else:
            error = getattr(getattr(resp.status, "error", None), "message", "")
            print(f"  query {state}: {error[:150]}")
    except Exception as exc:
        print(f"  cannot query ({type(exc).__name__}: {str(exc)[:120]})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
