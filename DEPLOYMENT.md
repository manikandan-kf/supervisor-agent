# Deploying the Supervisor Agent to Databricks

Everything this component needs **from** a Databricks workspace, everything it
**creates** there, and the order to do it in. Follow it top to bottom for a
fresh workspace.

The whole deployment is the Databricks Asset Bundle in this repository plus a
short sequence of grants that the bundle cannot express.

**Run it once per environment.** `dev` and `prod` are two deployments that share
nothing but the Lakebase instance, and every name below derives from the bundle
target — `-t dev` and `-t prod` throughout, `-e dev` / `-e prod` for the operator
scripts. §3 lists exactly what each one creates. A third environment is a new
target in `databricks.yml` and no other change.

`dev` is a deployment, not a mode: it is held to the same runtime contract as
prod — durable state or refuse to boot, no safety-critical setting disabled,
every turn under a time budget. `ENVIRONMENT=local` is the only value that
relaxes any of it, and only `.env.example` (a workstation file) uses it.

---

## 1. Prerequisites — what to have before you start

Collect these first. Every one is a question for the workspace or account owner;
none can be discovered by the code.

| # | What | Why it is needed |
|---|---|---|
| 1 | **Workspace URL** and a user with admin (or near-admin) rights | CLI authentication, bundle deploy, creating the service principal, granting ACLs |
| 2 | **Unity Catalog**: a catalog, and permission to create **one schema per environment** in it | Holds the registered model, the prompts and the governed configuration. Defaults to `workspace.supervisor_dev` and `workspace.supervisor_prod` |
| 3 | **A foundation-model serving endpoint** on your approved model list, pay-per-token | The governance model used for guardrail verdicts and context resolution. Dev and prod may use different tiers |
| 4 | **Lakebase (Databricks Postgres)** capability | Conversation checkpoints, long-term memory, the governed configuration table, the review queue and the audit sink. One instance serves every environment, separated by Postgres schema |
| 5 | **Serverless jobs compute** | Runs the deploy job. Enabled by default on most workspaces |
| 6 | **Model Serving** enabled | Hosts the agent |
| 7 | **A service principal** (SCIM-visible) **with one generated secret** | The OAuth M2M identity your front door uses to call the agent endpoint |
| 8 | *(Optional)* A **SQL warehouse id** | Only if you also want a Delta copy of the audit table. The Postgres sink is the primary store and needs no warehouse |

**Deliberately not required:** the `databricks-sql-access` entitlement for the
serving endpoint (its system service principal can never hold it — see §7b), and
any external cloud storage or Unity Catalog external location.

Local tooling: the Databricks CLI (v1.10+), Python 3.11+, and a virtual
environment with `pip install -r requirements.txt -r requirements-dev.txt`.

---

## 2. The three identities

Confusing these is the most common failure. Three separate identities do three
separate jobs.

| Identity | Created by | What it does | What it needs |
|---|---|---|---|
| **Your CLI user** | Your Databricks account | Deploys the bundle, runs the job, creates the other two, owns the Postgres tables | `databricks auth login`; workspace admin |
| **Caller service principal** | You, step 4 | The OAuth M2M identity that invokes the agent endpoint on users' behalf | A generated **secret**; **CAN QUERY** on the serving endpoint (§7a) |
| **The endpoint's own system SP** | `agents.deploy()`, automatically | What the supervisor code runs *as* inside Model Serving | Lakebase access (usually automatic), prompt-schema grants (§7c), then **restricted** governance-table grants (§7b) |

> The endpoint's own service principal does **not** appear in SCIM —
> `databricks service-principals list` will not show it. The endpoint page in
> the UI shows it; §7 gives a CLI route.

---

## 3. What the deployment creates in the workspace

**Everything here is per environment** — `<env>` is the bundle target — except
the Lakebase instance, which is shared and separated by Postgres schema.
Deploying one environment cannot touch another's objects.

| Object | Name | Created by |
|---|---|---|
| Unity Catalog schema | `<catalog>.supervisor_<env>` | `bundle deploy` |
| Job | `supervisor-agent-deploy` | `bundle deploy` |
| MLflow experiment | `/Shared/supervisor-agent-<env>` | the job |
| Registered model | `<catalog>.supervisor_<env>.supervisor_agent` | the job |
| Prompts (3) | `supervisor_domain_screen`, `supervisor_routing`, `supervisor_worker_simulation`, aliased `@<env>` | the job |
| Serving endpoint | `agents_<catalog>-supervisor_<env>-supervisor_agent` | `agents.deploy()` inside the job |
| Lakebase instance | `supervisor-memory` — **shared** across environments | step 5 |
| Postgres schema | `supervisor_<env>` | step 5 |
| Postgres tables | checkpoints, store, `supervisor_config`, `supervisor_review_queue`, `supervisor_audit_log`, all inside that schema | created by the runtime on first use |

The endpoint name is derived from the model's full name — which is precisely how
the per-environment schema gives each environment its own endpoint. Changing the
catalog or schema renames the endpoint, so anything resolving it by name must
change too.

---

## 4. Authenticate and create the caller service principal

```
databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile <profile-name>
databricks auth profiles          # the new profile should show Valid: YES
```

Set the profile once so every terminal and the Python SDK pick it up:

```
# Windows
setx DATABRICKS_CONFIG_PROFILE "<profile-name>"
# macOS / Linux
export DATABRICKS_CONFIG_PROFILE="<profile-name>"
```

> If you keep several workspaces in `.databrickscfg`, leave the `[DEFAULT]`
> section empty. Giving `[DEFAULT]` the same host as a named profile makes every
> SDK call fail with *"DEFAULT and \<profile\> match \<host\> … Use --profile"*.

Create the identity your front door will use to call the agent:

```
databricks service-principals create --display-name "supervisor-platform-api" --active
# note the applicationId (a UUID) and the id (SCIM, numeric)

databricks service-principal-secrets-proxy create <scim-id>
```

The secret is shown **once**. Record it in your secret store; if it scrolls
away, generate another rather than hunting for it — there is no read-back API.
Put the application id and secret wherever your front door reads its
credentials from.

Pre-flight, before going further:

```
databricks current-user me                       # active: true
databricks serving-endpoints get <routing-model>  # state READY
```

If the routing model you intended does not exist here, pick another from the
pay-per-token list and set `routing_llm_endpoint` in `databricks.yml`. Nothing
in `src/` names a model.

---

## 5. Provision Lakebase

The deploy job **fails** if the instance named in `databricks.yml` does not
exist, so this runs first.

```
# the instance — once per workspace, shared by every environment
python scripts/provision_lakebase.py --instance supervisor-memory
# creating instance 'supervisor-memory' (capacity CU_1)
# state: UPDATING … AVAILABLE      (about two minutes)

# this environment's schema — once per environment, before its first deploy
python scripts/provision_lakebase.py --instance supervisor-memory \
  --skip-instance --pg-schema supervisor_dev
# schema 'supervisor_dev' exists
```

The schema is what separates dev's checkpoints, audit rows and governed
configuration from prod's.

> **Do not skip the second command.** Postgres accepts a `search_path` naming a
> schema that does not exist, so an unqualified `CREATE TABLE` falls through to
> `public`. A missing schema does not fail the deploy — it succeeds and quietly
> puts both environments' rows in one set of tables, which is only visible once
> someone reads a row that should not be there.

Physical isolation instead of schema isolation is the same script with a
different `--instance` per environment, plus `lakebase_instance` set on that
target in `databricks.yml`.

> **Cost:** Lakebase bills continuously for as long as the instance exists. It
> is the first thing to remove in a non-permanent environment
> (`databricks database delete-database-instance supervisor-memory`); the code
> falls back to in-memory state without any change, losing durability.

If your workspace creates Autoscaling-generation instances (project/branch
rather than a provisioned instance), use `LAKEBASE_AUTOSCALING_ENDPOINT` /
`LAKEBASE_PROJECT` / `LAKEBASE_BRANCH` instead of `LAKEBASE_INSTANCE`. The
script reports which generation it created.

---

## 6. Deploy the bundle and run the job

```
databricks bundle validate -t dev     # → Validation OK!
databricks bundle deploy   -t dev     # uploads files, creates the UC schema and the job
databricks bundle run supervisor_agent_deploy -t dev
```

Three tasks. `register_prompts` and `publish_config` run in parallel, then
`log_and_deploy` once both succeed:

| Task | What it does |
|---|---|
| `register_prompts` | Registers the three prompts and points the target's alias at them |
| `publish_config` | Seeds `supervisor_config` from `src/supervisor/config/*.yaml`, so the agent is governed before it first serves |
| `log_and_deploy` | Logs the model, registers it in Unity Catalog, and deploys the serving endpoint |

Expect **10–15 minutes**, almost all of it the serving-container build.
`agents.deploy()` returns when the rollout is *initiated*, not when it is
serving — so wait for it explicitly before verifying anything:

```
python scripts/wait_for_endpoint.py -e dev --timeout 1800
```

Repeat the whole step with `-t prod` / `-e prod` for the production
environment — it creates its own schema, model, endpoint, prompts and
experiment.

> Several log lines from `log_and_deploy` look alarming and are expected: the
> job validates the model with a test prediction that runs **without** the
> endpoint's environment, so it reports an in-memory checkpointer, a
> process-log-only audit sink, no review queue, and a failed worker dispatch.
> Confirm those on the endpoint in §8, not from the job log.

> **Do not start a second deploy while a rollout is in progress.**
> `agents.deploy()` refuses an updating endpoint with *"Endpoint … is currently
> updating"* and strands the model version it just registered. Wait for
> `wait_for_endpoint.py` to exit cleanly first.

---

## 7. Post-deploy grants — the part that is easy to miss

Three grants, for three different identities. `agents.deploy()` recreates the
endpoint with a default ACL, so **redo these after any teardown-and-rebuild.**

All three are **per environment**, and each environment's endpoint has its own
service principal. `--pg-schema` and `-e` are not optional once more than one
environment exists: without them the commands resolve through `public` or
through `dev`, and act on the wrong environment's tables — or on none, silently.

### 7a. Caller SP → CAN QUERY on the endpoint

Do this **before** running `wait_for_endpoint.py` if your `.env` carries the
service principal's credentials: the scripts then authenticate as that SP, and
an SP with no grants cannot even read the endpoint's state.

The permissions API takes the endpoint **id**, not its name:

```
databricks serving-endpoints get agents_<catalog>-supervisor_<env>-supervisor_agent -o json
# take .id from the output, then, with a JSON body:
#   {"access_control_list": [
#      {"service_principal_name": "<application-id>", "permission_level": "CAN_QUERY"}]}
databricks serving-endpoints update-permissions <endpoint-id> --json @acl.json
```

Use `update-permissions` (merges), not `set-permissions` (replaces, and would
drop your own CAN_MANAGE). Write the JSON file without a byte-order mark — the
CLI rejects a BOM with *"invalid character 'ï' looking for beginning of value"*.

The resulting ACL should read: the caller SP `CAN_QUERY`, you `CAN_MANAGE`,
`admins` `CAN_MANAGE`.

### 7b. Endpoint SP → Lakebase, then least privilege

The endpoint's own service principal usually gets its Postgres role and tables
automatically on the first turn. Send one message through the endpoint, then
find its application id — it is not in SCIM, but its first Lakebase connection
creates a Postgres role named after it:

```sql
SELECT rolname FROM pg_roles;   -- the UUID-shaped role is the endpoint SP
```

Then **restrict it**, once the runtime has created its tables. Until you do, the
serving identity holds full DML on every table in the schema — including the
configuration that governs it and the audit trail that records it. An audit
trail writable by the component it audits is not an audit trail.

```
python scripts/provision_lakebase.py --instance supervisor-memory --skip-instance \
  --pg-schema supervisor_dev \
  --grant-identity <endpoint-sp-application-id> --restrict-governance-tables --dry-run

# review the SQL, then run it without --dry-run
```

Verify against the catalog rather than the script's output — a `REVOKE` reports
success whether or not it removed anything:

```sql
SELECT table_name, privilege_type
  FROM information_schema.role_table_grants
 WHERE grantee = '<endpoint-sp-application-id>'
   AND table_schema = 'supervisor_dev'
   AND table_name IN ('supervisor_config','supervisor_audit_log','supervisor_review_queue')
 ORDER BY table_name, privilege_type;
```

Expected exactly: `SELECT` on `supervisor_config`; `INSERT, SELECT` on
`supervisor_audit_log`; `INSERT, SELECT, UPDATE` on `supervisor_review_queue`.
Anything more — `TRIGGER` especially — means the revoke did not cover it.

The `table_schema` filter matters: the same three table names exist in every
environment's schema, so without it the query answers a question about all of
them at once.

> The role also holds `CREATE` on the *database*, granted alongside the schema
> privileges. It permits creating schemas and nothing inside anyone else's, and
> it is needed because the checkpointer and store issue
> `CREATE SCHEMA IF NOT EXISTS` on every boot as this identity.

Nothing in the request path writes configuration, so this breaks no runtime
behaviour: the only writer is the `publish_config` job, which runs as a
different identity.

### 7c. Endpoint SP → prompt access

For the endpoint to load prompts from the registry rather than falling back to
the bundled text:

```
python scripts/grant_prompt_access.py -e dev --principal <endpoint-sp-application-id> --write
```

The read-only set is not enough. With MLflow tracing active, every prompt load
also links the version to the trace, which needs create and update rights on the
schema. The schema holds only this agent's artifacts, so the wider grant is
contained; if that is not acceptable, keep the bundled-prompt fallback instead —
it is behaviourally identical while the registry text matches the bundled text.

---

## 8. Verify

```
python scripts/verify_deployment.py -e dev
```

It prints the environment it resolved, then reports the endpoint state, the
prompts in Unity Catalog, what the endpoint actually loaded, and the audit table.
Send one real domain request through the endpoint before trusting the prompt
check — a cold endpoint has loaded nothing, and small talk is classified
deterministically without touching a prompt.

Then confirm durable state exists **in this environment's schema**: after one
governed turn there should be rows in `supervisor_<env>`'s checkpoint tables and
at least one row in its `supervisor_audit_log`.

```sql
SET search_path TO supervisor_dev;
SELECT count(*) FROM supervisor_audit_log;
```

Two failures look similar and are not. If those tables are empty and the endpoint
log mentions an in-memory checkpointer, the Lakebase instance name did not reach
the endpoint. If instead the rows turn up in `public`, the environment's schema
did not exist when the runtime first wrote — go back to §5, then redeploy.

---

## 9. After the first deploy

| Changed | How it reaches the endpoint |
|---|---|
| `agents.yaml`, `rbac.yaml`, `guardrails.yaml` | `python scripts/publish_config.py --apply --lakebase-schema supervisor_dev` — a table write, into that environment's schema only. A running endpoint picks it up within the configuration cache TTL. **No redeploy** |
| `src/supervisor/**` or `deploy/**` | `databricks bundle deploy -t dev` → `databricks bundle run supervisor_agent_deploy -t dev` → `wait_for_endpoint.py -e dev` |
| Prompt text in `prompt_provider.py` | Same redeploy. While the endpoint uses the bundled fallback, a prompt change reaches it by redeploy, not by moving a registry alias |
| `databricks.yml` variables | `bundle deploy` then `bundle run` again |
| A **new** environment | Add the target to `databricks.yml` — every name derives from `${bundle.target}`, so nothing else changes. Then §5's schema command, §6 with `-t <env>`, and §7's grants for the new endpoint |

Promotion from dev to prod is a `bundle run -t prod`, not a copy: prod builds its
own model version from the same source and registers it in its own schema.
Nothing moves between environments, and nothing in dev can change what prod
serves.

Onboarding a new worker agent is a configuration change, not a code change: add
the entry to `agents.yaml` and the role mapping to `rbac.yaml`, publish to each
environment you want it in, and grant the corresponding role in your identity
provider.

To remove one environment: `python scripts/teardown_databricks.py -e dev --yes`
(it deletes the serving endpoint, which the bundle does not own) followed by
`databricks bundle destroy -t dev`. It touches only that environment's objects.
The Lakebase instance is shared — delete it separately, and drop the
environment's Postgres schema by hand if you want its rows gone.

---

## 10. Troubleshooting

| Symptom | First thing to check |
|---|---|
| `bundle run` → *"Triggering new runs … is currently disabled temporarily"* | Account credits or entitlements, not the bundle. `w.warehouses.start(...)` tends to state the real reason |
| A job task dies with `NameError: name '__file__' is not defined` | Serverless `exec`s the script rather than importing it. Use the `_repo_root()` fallback the other job scripts carry |
| A task's log shows success but the task is `FAILED` with `SystemExit: 0` | The script exits explicitly on success. Any escaping exception is a task failure, a zero exit included — exit only on a real failure code |
| `log_and_deploy` → *"Endpoint … is currently updating"* | A previous rollout is still in progress. Wait for `wait_for_endpoint.py`, then re-run |
| *"Could not open requirements file"* in the job | A stale `.databricks/` sync snapshot from another workspace. Delete the folder and redeploy |
| Callers get 403 / "agent unavailable" while your own scripts work | §7a — your scripts authenticate as you, the front door as the service principal |
| Endpoint log says it fell back to bundled prompts | §7c. Harmless while the registry text matches the bundled text |
| Audit table empty | The Postgres sink is the real store. The Delta table is optional and unwired unless you set a warehouse id. Also check `search_path` — you may be reading a different environment's schema from the one the endpoint writes to |
| Conversation history lost between turns | The endpoint has no Lakebase instance — check the bundle variable reached it, and that the instance is AVAILABLE |
| Endpoint refuses to boot: *"refusing to degrade to in-memory state"* | Working as intended. Every deployed environment, `dev` included, requires durable state. Either Lakebase is unreachable, or `ENVIRONMENT` was set to a deployed name on a workstation — use `local` there |
| Two environments' rows in the same tables | The environment's schema was never created, so `search_path` fell through to `public`. Run §5's schema command and redeploy; rows already in `public` stay there |
| An operator script reports the endpoint is missing after a healthy deploy | It derived a different environment's name. Pass `-e <env>`, or check `$ENVIRONMENT` — the scripts default to it |
