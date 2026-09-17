# Deploying the Supervisor Agent to Databricks

Everything this component needs **from** a Databricks workspace, everything it
**creates** there, and the order to do it in. Follow it top to bottom for a
fresh workspace.

The whole deployment is the Databricks Asset Bundle in this repository plus a
short sequence of grants that the bundle cannot express. Every step is a
Databricks CLI command, a SQL statement, or one of the four scripts in
`deploy/` that the bundle's job runs for you.

**Run it once per environment.** `dev` and `prod` are two deployments that share
nothing but the Lakebase instance, and every name below derives from the bundle
target — `-t dev` and `-t prod` throughout. A third environment is a new target
in `databricks.yml` and no other change.

**Where the bundle configuration lives.** `databricks.yml` holds what is true
for the whole bundle: the variables, the library artifact, the sync set and the
targets. Each resource is its own file under `resources/`, merged in by an
`include: resources/*.yml` glob — `schema.yml` (the environment's Unity Catalog
schema), `deploy_job.yml` (the four-task job) and `lakebase.yml` (shared
infrastructure, shipped disabled; the file says why). Adding a resource is a new
file there, not another block in `databricks.yml`; removing one is deleting its
file.

`dev` is a deployment, not a mode: it is held to the same runtime contract as
prod — durable state or refuse to boot, no safety-critical setting disabled,
every turn under a time budget. `ENVIRONMENT=local` is the only value that
relaxes any of it, and only a workstation uses it.

---

## 1. Prerequisites — what to have before you start

| # | What | Why it is needed |
|---|---|---|
| 1 | **Workspace URL** and a user with admin (or near-admin) rights | CLI authentication, bundle deploy, creating the service principal, granting ACLs |
| 2 | **Unity Catalog**: a catalog, and permission to create **one schema per environment** plus one platform schema in it | Holds the registered model, the prompts, the governed configuration, and the volume the shared library is published to. Defaults to `workspace.supervisor_dev`, `workspace.supervisor_prod` and `workspace.agent_platform` |
| 3 | **A foundation-model serving endpoint** on your approved model list, pay-per-token | The governance model used for guardrail verdicts and context resolution. Dev and prod may use different tiers |
| 4 | **Lakebase (Databricks Postgres)** capability | Conversation checkpoints, long-term memory, the governed configuration table and the audit sink. One instance serves every environment, separated by Postgres schema |
| 5 | **Serverless jobs compute** | Runs the deploy job. Enabled by default on most workspaces |
| 6 | **Model Serving** enabled | Hosts the agent |
| 7 | **A service principal** (SCIM-visible) **with one generated secret** | The OAuth M2M identity the calling application uses to call the agent endpoint |

**Deliberately not required:** a SQL warehouse, the `databricks-sql-access`
entitlement for the serving endpoint (its system service principal can never
hold it), and any external cloud storage or Unity Catalog external location.

Local tooling: the Databricks CLI (v1.10+), Python 3.11+ with `pip` (the bundle
builds the shared library wheel locally with `pip wheel`), and a virtual
environment with `pip install -r requirements.txt -r requirements-dev.txt`.

---

## 2. The three identities

Confusing these is the most common failure. Three separate identities do three
separate jobs.

| Identity | Created by | What it does | What it needs |
|---|---|---|---|
| **Your CLI user** | Your Databricks account | Deploys the bundle, runs the job, creates the other two, owns the Postgres tables | `databricks auth login`; workspace admin |
| **Caller service principal** | You, step 4 | The OAuth M2M identity that invokes the agent endpoint on users' behalf | A generated **secret**; **CAN QUERY** on the serving endpoint (§7a) |
| **The endpoint's own system SP** | Model Serving, when the endpoint is created | What the supervisor code runs *as* inside Model Serving | Lakebase access (usually automatic), prompt-schema grants (§7c), then **restricted** governance-table grants (§7b) |

> The endpoint's own service principal does **not** appear in SCIM —
> `databricks service-principals list` will not show it. The endpoint page in
> the UI shows it, and its first Lakebase connection creates a Postgres role
> named after its application id (§7b).

---

## 3. What the deployment creates in the workspace

**Everything here is per environment** — `<env>` is the bundle target — except
the Lakebase instance and the platform schema, which are shared.

| Object | Name | Created by |
|---|---|---|
| Unity Catalog schema | `<catalog>.supervisor_<env>` | `bundle deploy` |
| Job | `supervisor-agent-deploy` | `bundle deploy` |
| Shared library wheel | `agent_governance-<version>-py3-none-any.whl` | `bundle deploy` (built locally, uploaded with the bundle) |
| MLflow experiment | `/Shared/supervisor-agent-<env>` | the job |
| Trace tables | OTel Delta tables in `<catalog>.supervisor_<env>` — the experiment's `trace_location` (solution §08) | the job (`deploy_agent --trace-catalog-schema`) |
| Prompts (3) | `supervisor_domain_screen`, `supervisor_routing`, `supervisor_worker_simulation`, aliased `@<env>` | the job (`register_prompts`) |
| Registered model | `<catalog>.supervisor_<env>.supervisor_agent`, with the wheel under `wheels/` | the job (`deploy_agent`) |
| Serving endpoint | `agents_<catalog>-supervisor_<env>-supervisor_agent` | the job (`deploy_agent`): `agents.deploy()` by default, the Model Serving SDK with `deploy_method: serving-api` (§6a) |
| Platform schema + volume | `<catalog>.agent_platform.libs` — **shared** by every agent | the job (`publish_library`) |
| Lakebase instance | `supervisor-memory` — **shared** across environments | step 5 |
| Postgres schema | `supervisor_<env>` | the runtime, on the first connection (`CREATE SCHEMA IF NOT EXISTS`) |
| Postgres tables | checkpoints, store, `supervisor_config`, `supervisor_audit_log`, all inside that schema | the runtime, on first use |

> **Tracing destination.** The bundle passes `--trace-catalog-schema
> ${var.catalog}.${var.schema}`, so MLflow traces persist as Unity Catalog Delta
> tables rather than in the experiment's own store: governed by UC, queryable
> from SQL, and with no per-experiment cap. It needs MLflow >= 3.14 (pinned at
> 3.15.1) and the deploying identity to hold `USE CATALOG`, `USE SCHEMA` and
> `MODIFY`/`SELECT` on the schema — `bundle deploy` creates the schema, so this
> holds by default. A workspace that cannot satisfy it keeps the experiment
> store and prints why; the deploy is not failed for it.

The endpoint name is derived from the model's full name — which is how the
per-environment schema gives each environment its own endpoint.

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

Create the identity the calling application will use to call the agent:

```
databricks service-principals create --display-name "supervisor-platform-api" --active
# note the applicationId (a UUID) and the id (SCIM, numeric)

databricks service-principal-secrets-proxy create <scim-id>
```

The secret is shown **once**. Record it in your secret store. Put the
application id and secret wherever the calling application reads its credentials from.

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
exist, so this runs first. One instance serves every environment:

```
databricks database create-database-instance supervisor-memory --capacity CU_1
databricks database get-database-instance supervisor-memory      # repeat until state: AVAILABLE (~2 min)
```

**The environment's Postgres schema needs no separate step.** Every pool the
runtime opens issues `CREATE SCHEMA IF NOT EXISTS <lakebase_schema>` first, and
the `publish_config` task in §6 is the first thing to connect — running as
you, the instance owner — so `supervisor_dev` exists before the endpoint ever
boots. That closes the failure the schema step used to guard against: Postgres
accepts a `search_path` naming a schema that does not exist and lets an
unqualified `CREATE TABLE` fall through to `public`, quietly pooling two
environments' rows in one table.

Physical isolation instead of schema isolation is a second instance plus
`lakebase_instance` set on that target in `databricks.yml`.

> **Cost:** Lakebase bills continuously for as long as the instance exists. It
> is the first thing to remove in a non-permanent environment
> (`databricks database delete-database-instance supervisor-memory`).

If your workspace creates Autoscaling-generation instances (project/branch
rather than a provisioned instance), set `LAKEBASE_AUTOSCALING_ENDPOINT` or
`LAKEBASE_PROJECT` / `LAKEBASE_BRANCH` in the deploy shell instead of relying
on `LAKEBASE_INSTANCE`; `deploy_agent.py` passes them to the container.

---

## 6. Deploy the bundle and run the job

```
databricks bundle validate -t dev     # → Validation OK!
databricks bundle deploy   -t dev     # builds the library wheel, uploads files, creates the UC schema and the job
databricks bundle run supervisor_agent_deploy -t dev
```

`bundle deploy` runs `python -m pip wheel --no-deps --wheel-dir dist .` inside
`libs/agent_governance` before uploading, so the machine running it needs
Python and `pip` on the path.

Four tasks. Three run in parallel, then `deploy_agent` once the prompts and
the configuration are in place:

| Task | What it does |
|---|---|
| `register_prompts` | Registers the three prompts and points the target's alias at them |
| `publish_config` | Seeds `supervisor_config` from `src/supervisor/config/*.yaml`, so the agent is governed before it first serves. Creates the environment's Postgres schema as a side effect |
| `publish_library` | Copies the library wheel to `/Volumes/<catalog>/agent_platform/libs/` for the other agents. Skips a version already there |
| `deploy_agent` | Logs the model with the wheel baked in under `wheels/`, registers it in Unity Catalog, and deploys the serving endpoint |

Expect **10–15 minutes**, almost all of it the serving-container build.
`deploy_agent` returns when the rollout is *initiated*, not when it is
serving (pass `--wait-minutes 40` to make the job block until `READY` and fail
otherwise) — so wait for it explicitly before verifying anything:

```
databricks serving-endpoints get agents_<catalog>-supervisor_<env>-supervisor_agent -o json
# ready when:  .state.ready == "READY"  and  .state.config_update == "NOT_UPDATING"
```

`UPDATE_FAILED` is terminal — it never becomes `NOT_UPDATING`. The per-entity
`deployment_state_message` in `pending_config.served_entities` carries the
cause; the commonest is the account's served-entity cap.

Repeat the whole step with `-t prod` for the production environment.

> Several log lines from `deploy_agent` look alarming and are expected: the
> job validates the model with a test prediction that runs **without** the
> endpoint's environment, so it reports an in-memory checkpointer, a
> process-log-only audit sink, and a failed worker dispatch.
> Confirm those on the endpoint in §8, not from the job log.

> **Do not start a second deploy while a rollout is in progress.**
> Both deploy methods refuse an updating endpoint with *"Endpoint … is currently
> updating"* and strand the model version they just registered. Wait for
> `NOT_UPDATING` and rerun the job; the version is still registered.

### 6a. If `agents.deploy()` fails in your workspace

`deploy_agent` has two ways to create the endpoint. The default calls
`databricks.agents.deploy()`, which does four things: creates the endpoint,
requests AI Gateway inference tables **in the same create call**, stamps the
tracing variables, and registers the deployment with the Agent Framework REST
API (the Review App entry). The last two of those are where locked-down
workspaces fail, and neither is something this agent depends on.

`deploy_method: serving-api` creates or rolls the endpoint through the Model
Serving SDK instead. It keeps everything the rest of this document relies on:

| | `agents` (default) | `serving-api` |
|---|---|---|
| Endpoint name | `agents_<catalog>-<schema>-<model>` | identical (same derivation, checked against databricks-agents 1.11.0) |
| Credentials for the routing LLM, workers, Lakebase | from the model's declared `resources` | identical — this is a Model Serving feature, not an `agents.deploy()` one |
| `ENABLE_MLFLOW_TRACING`, `MLFLOW_EXPERIMENT_ID`, `MONITOR_EXPERIMENT_ID` tag | set | set, same values |
| Inference tables | requested at create; a refusal fails the deploy | requested **after** create; a refusal is printed, the deploy succeeds |
| Old versions on the endpoint | kept at 0 % traffic (15-entity cap) | replaced — one served entity |
| Review App entry | yes | no |
| `databricks-agents` needed at deploy time | yes | no |

Switch per environment in `databricks.yml` (`variables.deploy_method`, or
under the target's `variables:`), or for one deploy:

```
databricks bundle deploy -t dev --var deploy_method=serving-api
databricks bundle run supervisor_agent_deploy -t dev
```

By hand: `python deploy/deploy_agent.py … --deploy-method serving-api`.
Add `--wait-minutes 40` to make the job block until the endpoint is `READY`
and fail if it is not, instead of returning when the rollout is initiated.

**What the switch does not fix.** An error of the form
*"PERMISSION_DENIED: Endpoint creator doesn't have permission to access
dependency type: LAKEBASE"* comes from Model Serving validating the model's
declared resources and is raised by both methods. Databricks currently accepts
the passthrough for a Lakebase dependency only when the identity creating the
endpoint is a **workspace admin** — run the job as one, or deploy with
`--lakebase-instance ""` and provide the database credentials through the
endpoint's environment instead.

---

## 7. Post-deploy grants — the part that is easy to miss

Three grants, for three different identities. A recreated endpoint comes with
a default ACL whichever method created it, so **redo these after any
teardown-and-rebuild.**
All three are **per environment**.

### 7a. Caller SP → CAN QUERY on the endpoint

The permissions API takes the endpoint **id**, not its name:

```
databricks serving-endpoints get agents_<catalog>-supervisor_<env>-supervisor_agent -o json
# take .id from the output, then, with a JSON body:
#   {"access_control_list": [
#      {"service_principal_name": "<application-id>", "permission_level": "CAN_QUERY"}]}
databricks serving-endpoints update-permissions <endpoint-id> --json @acl.json
```

Use `update-permissions` (merges), not `set-permissions` (replaces, and would
drop your own CAN_MANAGE). Write the JSON file without a byte-order mark.

### 7b. Endpoint SP → Lakebase, then least privilege

The endpoint's own service principal gets its Postgres role and table grants
automatically on the first turn. Send one message through the endpoint (§8),
then find the role — it is not in SCIM, but its first Lakebase connection
creates a Postgres role named after its application id:

```sql
SELECT rolname FROM pg_roles;   -- the UUID-shaped role is the endpoint SP
```

Then **restrict it**, once the runtime has created its tables. Until you do, the
serving identity holds full DML on every table in the schema — including the
configuration that governs it and the audit trail that records it. An audit
trail writable by the component it audits is not an audit trail. Run this in
the instance's SQL editor (or `psql`) as the table owner, with `<role>` the
application id and the schema set first:

```sql
SET search_path TO supervisor_dev;

-- Policy is read-only from the data plane.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, TRIGGER, REFERENCES ON supervisor_config FROM "<role>";
GRANT  SELECT                                                ON supervisor_config TO   "<role>";

-- The audit trail is append-only from the data plane.
REVOKE UPDATE, DELETE, TRUNCATE, TRIGGER, REFERENCES         ON supervisor_audit_log FROM "<role>";
GRANT  INSERT, SELECT                                        ON supervisor_audit_log TO   "<role>";
```

`TRIGGER` is the one that matters: it would let the serving identity attach a
trigger that fires with the privileges of whoever performs the next DML — the
`publish_config` job identity — which is exactly the escalation around the
RBAC gate this closes.

Verify against the catalog rather than trusting the statements — a `REVOKE`
reports success whether or not it removed anything:

```sql
SELECT table_name, privilege_type
  FROM information_schema.role_table_grants
 WHERE grantee = '<role>'
   AND table_schema = 'supervisor_dev'
   AND table_name IN ('supervisor_config','supervisor_audit_log')
 ORDER BY table_name, privilege_type;
```

Expected exactly: `SELECT` on `supervisor_config`; `INSERT, SELECT` on
`supervisor_audit_log`. The `table_schema` filter matters: the same table
names exist in every environment's schema.

Nothing in the request path writes configuration, so this breaks no runtime
behaviour: the only writer is the `publish_config` task, which runs as you.

### 7c. Endpoint SP → prompt access

For the endpoint to load prompts from the registry rather than falling back to
the bundled text, its service principal needs the catalog and the environment's
schema. There is no MLflow resource type for prompts, so `agents.deploy()`
cannot grant this for you:

```
databricks grants update catalog <catalog> --json '{"changes":[{"principal":"<endpoint-sp-application-id>","add":["USE_CATALOG"]}]}'
databricks grants update schema <catalog>.supervisor_<env> --json '{"changes":[{"principal":"<endpoint-sp-application-id>","add":["USE_SCHEMA","EXECUTE","CREATE_FUNCTION","MANAGE"]}]}'
```

The read-only set (`USE_SCHEMA`, `EXECUTE`) is not enough: with MLflow tracing
active, every prompt load also links the version to the trace, which needs
create and update rights on the schema. The schema holds only this agent's
artifacts, so the wider grant is contained; if that is not acceptable, keep the
bundled-prompt fallback — it is behaviourally identical while the registry text
matches the bundled text.

### 7d. Consuming agents → the library volume

The deploy identity of any agent that installs the shared library needs to read
the volume:

```
databricks grants update volume <catalog>.agent_platform.libs --json '{"changes":[{"principal":"<consumer-deploy-identity>","add":["READ_VOLUME"]}]}'
```

### 7e. What the calling application is responsible for

The endpoint trusts `permitted_agents`, `approvable_agents`, `user_role` and
`user_id` in `custom_inputs` because **only the caller's service principal
holds CAN QUERY** (§7a). That ACL is the trust boundary; keep it to that one
principal. Three things the supervisor deliberately does not do, because the
caller is the right place for them (solution §04, §05):

| Responsibility | Why it lives in the caller |
|---|---|
| Validate the payload and resolve the target agent | Malformed requests are rejected with a 4xx before they reach the graph |
| Rate-limit per user and per API key | AI Gateway rate limits are not available on agent endpoints; the caller sees every user |
| **Send one turn per conversation at a time** | Model Serving runs concurrent requests on any replica with no session affinity. Two turns on one `conversation_id` in flight together would both load the same checkpoint and the second write wins. The caller holds a turn until the previous one for that conversation has returned |

---

## 8. Verify

Send one real domain request through the endpoint, authenticating as yourself:

```
databricks serving-endpoints query agents_<catalog>-supervisor_<env>-supervisor_agent --json '{
  "input": [{"role": "user", "content": "Write user stories for a password reset feature on product line alpha"}],
  "custom_inputs": {"user_role": "BA", "agent_id": "requirement-agent",
                    "permitted_agents": ["requirement-agent"], "user_id": "usr_verify"}
}'
```

A governed refusal is a *successful* call — read `custom_outputs.outcome` with
the text. Then read the endpoint's own log, which is the only place that says
what the endpoint's identity could actually reach:

```
databricks serving-endpoints logs agents_<catalog>-supervisor_<env>-supervisor_agent <served-model-name>
```

Look for these lines, one each:

| Line | Meaning |
|---|---|
| `checkpointer: Lakebase instance_name=…, schema=supervisor_<env>` | durable state is on, in this environment's schema |
| `audit sink: Postgres table supervisor_audit_log` | the decision trail is queryable |
| `configuration source: table supervisor_config` | the governed documents are live |
| `loaded prompt prompts:/…@<env> v…` | prompts come from the registry; `prompt registry unavailable … bundled default` means §7c is missing |

Then confirm durable state exists **in this environment's schema**:

```sql
SET search_path TO supervisor_dev;
SELECT count(*) FROM supervisor_audit_log;
SELECT name, version, active FROM supervisor_config;
```

Two failures look similar and are not. If those tables are empty and the endpoint
log mentions an in-memory checkpointer, the Lakebase instance name did not reach
the endpoint. If instead the rows turn up in `public`, `LAKEBASE_SCHEMA` did
not reach the endpoint — check the endpoint's environment variables.

Confirm the model artifact carries the library: in the registered model
version's artifacts, `wheels/agent_governance-<version>-py3-none-any.whl` is
present and `requirements.txt` ends with that same relative path.

---

## 9. After the first deploy

| Changed | How it reaches the endpoint |
|---|---|
| `agents.yaml`, `rbac.yaml`, `guardrails.yaml` | `python deploy/publish_config.py --apply --lakebase-instance supervisor-memory --lakebase-schema supervisor_<env>` — a table write, into that environment's schema only. A running endpoint picks it up within the configuration cache TTL. **No redeploy** |
| `src/supervisor/**` or `deploy/**` | `databricks bundle deploy -t <env>` → `databricks bundle run supervisor_agent_deploy -t <env>` → wait (§6) |
| `libs/agent_governance/**` | Bump `agent_governance.__version__`, then the same redeploy: the new wheel is baked into the new model version and published to the volume. Consumers pick up the new version when they choose to |
| Prompt text in `prompt_registry.py` | Same redeploy. While the endpoint uses the bundled fallback, a prompt change reaches it by redeploy, not by moving a registry alias |
| `databricks.yml` variables | `bundle deploy` then `bundle run` again |
| A **new** environment | Add the target to `databricks.yml` — every name derives from `${bundle.target}`. Then §6 with `-t <env>` and §7's grants for the new endpoint |

Promotion from dev to prod is a `bundle run -t prod`, not a copy: prod builds its
own model version from the same source and registers it in its own schema.

Onboarding a new worker agent is a configuration change, not a code change: add
the entry to `agents.yaml` and the role mapping to `rbac.yaml`, publish to each
environment you want it in, and grant the corresponding role in your identity
provider.

**Retention.** Nothing on the platform purges Lakebase checkpoints, and the
runtime evaluates session expiry and long-term memory TTL lazily *on read*, so
an abandoned conversation stops being used but its rows stay. Schedule this as
a SQL job running as the **table owner** (the §7b grants deliberately leave the
serving identity without DELETE). It uses the audit trail's `event_time` to
find idle conversations, because the checkpoint tables carry no timestamp:

```sql
SET search_path TO supervisor_<env>;

-- Conversations with no turn in the last 180 days.
CREATE TEMP TABLE stale AS
  SELECT conversation_id FROM supervisor_audit_log
   WHERE conversation_id <> ''
   GROUP BY conversation_id
  HAVING max(event_time) < now() - interval '180 days';

DELETE FROM checkpoint_writes WHERE thread_id IN (SELECT conversation_id FROM stale);
DELETE FROM checkpoint_blobs  WHERE thread_id IN (SELECT conversation_id FROM stale);
DELETE FROM checkpoints       WHERE thread_id IN (SELECT conversation_id FROM stale);
```

The audit rows themselves are never deleted. They carry no message text — the
decision trail is redacted before it is written — and they are the record that
the subject's requests were governed. MLflow trace tables and inference tables
have no automatic purge either; set a retention job on them in the same way.

**Verifying the decision trail.** `agent_governance.audit_trail.verify_chain` walks
the hash chain and reports whether history is intact; record its `head_id` and
`head_hash` somewhere outside operator write scope (a ticket, a signed commit)
so a clean result proves something. Schedule it alongside the sweep.

**Erasing a data subject by hand (GDPR Art. 17).** Two things hold personal data per
user: the long-term memory row (validated identifiers such as a product line,
keyed by the pseudonymous `user_key`) and the conversation checkpoints (keyed
by `thread_id`). Both are ordinary Postgres rows in the environment's schema,
so an erasure is SQL run as the table owner:

```sql
SET search_path TO supervisor_<env>;

-- What is held for the subject. Keys only: the values are the personal data.
SELECT key, jsonb_object_keys(value) AS field
  FROM store WHERE prefix = 'supervisor.resolved_context' AND key = '<user_key>';

DELETE FROM store WHERE prefix = 'supervisor.resolved_context' AND key = '<user_key>';

-- Each conversation the subject asks to have removed (thread ids come from
-- the audit table: SELECT DISTINCT conversation_id FROM supervisor_audit_log WHERE user_key = '<user_key>').
DELETE FROM checkpoint_writes WHERE thread_id = '<conversation_id>';
DELETE FROM checkpoint_blobs  WHERE thread_id = '<conversation_id>';
DELETE FROM checkpoints       WHERE thread_id = '<conversation_id>';
```

Re-run the `SELECT` afterwards to confirm nothing remains, and record the
request and its completion in your change log. Do **not** hand-insert a row
into `supervisor_audit_log` to record it: that table is a hash chain written
only by the runtime, and a row inserted without its digest breaks every
verification after it. The audit rows themselves carry no message text — the
decision trail is redacted before it is written — and stay as the record that
the subject's requests were governed.

To remove one environment: delete the serving endpoint (the bundle does not own
it), then destroy the bundle. Both touch only that environment's objects:

```
databricks serving-endpoints delete agents_<catalog>-supervisor_<env>-supervisor_agent
databricks bundle destroy -t <env>
```

The Lakebase instance and the platform volume are shared — delete them
separately, and drop the environment's Postgres schema by hand if you want its
rows gone.

---

## 10. Troubleshooting

| Symptom | First thing to check |
|---|---|
| `bundle deploy` → *"no wheel"* / build error under `libs/agent_governance` | Python and `pip` on the local path; `python -m pip wheel --no-deps --wheel-dir libs/agent_governance/dist libs/agent_governance` by hand shows the real error |
| `deploy_agent` → `ModuleNotFoundError: agent_governance` | The job environment did not install the wheel — the `dependencies` glob in `resources/deploy_job.yml` must match a built file |
| Endpoint build log → *"No such file: wheels/agent_governance-…whl"* | The wheel was not logged next to the model. Check the registered version's artifacts; it is written by `log_model_artifacts` *before* `register_model` |
| `bundle run` → *"Triggering new runs … is currently disabled temporarily"* | Account credits or entitlements, not the bundle |
| A job task dies with `NameError: name '__file__' is not defined` | Serverless `exec`s the script rather than importing it. Use the `_repo_root()` fallback the deploy scripts carry |
| A task's log shows success but the task is `FAILED` with `SystemExit: 0` | The script exited explicitly on success. Exit only on a real failure code |
| `deploy_agent` → *"Endpoint … is currently updating"* | A previous rollout is still in progress. Wait for `NOT_UPDATING`, then re-run |
| *"Could not open requirements file"* in the job | A stale `.databricks/` sync snapshot from another workspace. Delete the folder and redeploy |
| Callers get 403 / "agent unavailable" while your own calls work | §7a — you authenticate as you, the calling application as the service principal |
| Endpoint log says it fell back to bundled prompts | §7c. Harmless while the registry text matches the bundled text |
| Audit table empty | Check `search_path` — you may be reading a different environment's schema from the one the endpoint writes to |
| Conversation history lost between turns | The endpoint has no Lakebase instance — check the variable reached it, and that the instance is AVAILABLE |
| Endpoint refuses to boot: *"refusing to degrade to in-memory state"* | Working as intended. Every deployed environment, `dev` included, requires durable state. Either Lakebase is unreachable, or `ENVIRONMENT` was set to a deployed name on a workstation — use `local` there |
| Endpoint log: *"audit table … predates latency_ms, …"* | The table was created by an earlier version. Widen it as its owner with the block below |
| `publish_library` → *"already published — left as is"* | Expected on a redeploy without a version bump. Bump `agent_governance.__version__` to release a library change |

Audit-table migration, for a table created before these columns existed (run
as the table's owner, schema set first):

```sql
SET search_path TO supervisor_dev;
ALTER TABLE supervisor_audit_log ADD COLUMN IF NOT EXISTS latency_ms INTEGER;
ALTER TABLE supervisor_audit_log ADD COLUMN IF NOT EXISTS session_age_seconds DOUBLE PRECISION;
ALTER TABLE supervisor_audit_log ADD COLUMN IF NOT EXISTS signoff JSONB;
ALTER TABLE supervisor_audit_log ADD COLUMN IF NOT EXISTS provenance JSONB;
ALTER TABLE supervisor_audit_log ADD COLUMN IF NOT EXISTS prev_hash TEXT;
ALTER TABLE supervisor_audit_log ADD COLUMN IF NOT EXISTS row_hash TEXT;
```

---

## Appendix — consuming the library from another agent

What the requirement, test-case, coding and deployment agents do to run the
same governance code as the supervisor. Their deploy job:

1. Installs the published wheel:
   `pip install /Volumes/<catalog>/agent_platform/libs/agent_governance-<version>-py3-none-any.whl`
   (or names that path in the job environment's `dependencies`).
2. Logs the same file next to their model — `wheels/<file>.whl` in the model's
   `pip_requirements`, the file added with `MlflowClient().log_model_artifacts`
   before `register_model` — exactly as `deploy/deploy_agent.py` does here.
3. In code: `agent_governance.sanitize` on inbound text,
   `agent_governance.output_guard` on every reply, and `agent_governance.audit_trail`
   for the decision trail. The library's README lists the module map. A worker
   trusts a dispatch because only the supervisor endpoint's service identity
   holds CAN QUERY on it — grant nothing else.

Pin the version. A wheel on the volume is never overwritten under the same
name, so a pinned consumer cannot change under you.
