# Custom LangGraph Supervisor Agent

A governed supervisor for an SDLC agent platform. Every user turn is taken
through a fixed pipeline before any worker agent is reached, and every decision
along the way is recorded:

```
rbac_gate → guardrails → route / clarify → dispatch → approval → respond + audit
```

It is served on Databricks Model Serving as an MLflow `ResponsesAgent`, and
deployed with the Databricks Asset Bundle in this repository.

**To deploy it, follow [DEPLOYMENT.md](DEPLOYMENT.md)** — prerequisites, what to
create in the workspace, step order, grants and verification.

---

## What each stage does

| Stage | Responsibility |
|---|---|
| `rbac_gate` | Re-validates the caller's role against the requested agent on **every** turn, including clarification replies and approval decisions. Also holds the conversation when a human review is open, and expires abandoned sessions |
| `guardrails` | Two tiers: deterministic deny patterns first, then a semantic domain screen per candidate agent. Blocks off-domain and unsafe requests, offers an appeal path where a reviewer could change the answer, answers small talk itself, and holds a *"…keep this in mind for later"* as a session note rather than dispatching it as work |
| `route` | Resolves the context a worker needs (`required_context`). Asks one guided question rather than guessing; caps the clarification loop and escalates when it cannot converge |
| `dispatch` | Calls the worker's serving endpoint with a per-attempt timeout, bounded retries and a circuit breaker. Screens the conversation on the way *out* to the worker and the reply on the way back: sensitive values are masked, withheld or escalated by category, embedded directives are defanged, and a claim the turn holds no evidence for is labelled as one |
| `approval` | Stages irreversible work for human sign-off using LangGraph `interrupt()`, and records who approved what |
| `respond` | Emits the answer and writes the decision trail. A governance decision is not reported as applied unless its audit record landed |

## Two packages, one repository

The security and governance primitives are **not** the supervisor's alone: the
requirement, test-case, coding and deployment agents that follow will apply the
same screens and write the same audit shape. So they live in a separate,
installable library, and the supervisor is its first consumer.

```
libs/agent_governance/               the shared library — one wheel every agent installs
  src/agent_governance/
    sensitive.py       the sensitive-shape catalogue every boundary reads, + redact helpers
    output_guard.py    response policy: allow / mask / block / escalate, and the stream guard
    sanitize.py        untrusted text handling: control chars, fake turn boundaries, directives
    deny_rules.py      tier-1 deterministic deny patterns and the kill switch
    prompting.py       untrusted content in its own JSON turn; cacheable system prefix
    grounding.py       execution claims and citations nothing in the turn backs
    trust.py           HMAC over entitlements and dispatches — a worker calls verify_dispatch
    rbac.py            role → agent authorization
    deadline.py        per-turn time budget
    resilience.py      bounded retries for transient model failures
    spend.py           per-turn and per-subject cost ceilings
    audit.py           decision-trail sink with the tamper-evident hash chain
    review_queue.py    appeal / escalation queue — opened by an agent, resolved by a reviewer surface
    locking.py         one execution per conversation: the advisory lock Model Serving does not provide
    config_store.py    governed configuration in a table: checksummed, validated, TTL-cached
    lakebase.py        Lakebase pools, checkpointer and store builders, connection sources
    sql.py             identifier validation; the schema-aware table probe
    environment.py     is_local_environment, resource_environment, catalog, environment_schema
  tests/               the library's own tests, offline

src/supervisor/                      the supervisor agent
  agent.py           MLflow ResponsesAgent wrapper — predict / predict_stream
  graph.py           StateGraph wiring and durability
  nodes.py           the six stages above
  messages.py        every sentence the supervisor says in its own voice
  state.py           the conversation state channels
  context.py         per-request runtime context (role, entitlements, identity)
  registry.py        worker agent registry
  guardrails.py      the semantic domain screen, small-talk classifier, ownership contest
  routing.py         context resolution and clarification
  dispatch.py        worker client: timeout, retry, circuit breaker; the simulated worker
  memory.py          the supervisor's Lakebase schema; long-term memory — validated
                     on write, narrowed on read, aged out on a retention ceiling
  session_notes.py   "keep this in mind for later" — short-term, this thread only
  config.py          the supervisor's governed documents and their validators
  progress.py        the task-plan events streamed while a turn runs
  prompt_provider.py prompts from the MLflow registry, bundled fallbacks
  model_provider.py  the ChatDatabricks client for the routing model (and per-agent models)
  services.py        the dependency container
  settings.py        every tunable, read from the environment
  config/            agents.yaml · rbac.yaml · guardrails.yaml (seed documents)

deploy/
  log_and_deploy.py    logs, registers and deploys the model, with the library wheel baked in
  register_prompts.py  registers the prompts in Unity Catalog under the environment alias
  publish_config.py    publishes the governed documents to the configuration table
  publish_library.py   publishes the library wheel to the platform volume, for other agents
tests/                 the supervisor's governance contract, offline
databricks.yml         the asset bundle: schema, library artifact, deploy job
pyproject.toml         project metadata, dependencies, ruff and pytest config
requirements.txt       what the serving container installs — pinned exactly
```

## How the library reaches every agent

Databricks offers several ways to share code between agents. This project uses
a **Python wheel**, for the reason the alternatives fail here:

* **Unity Catalog functions** are SQL or Python UDFs executed on a warehouse or
  a cluster. They suit a tool an agent *calls*; they do not suit a regex
  catalogue applied to every token of a streamed reply, and the serving
  endpoint's identity cannot hold `databricks-sql-access` in any case.
* **`code_paths` copies of the source** in each agent's repository drift the
  day the second copy is edited — exactly what a shared guardrail must not do.
* **A wheel** is versioned, testable on its own, installed into a job or a
  serving container like any other dependency, and imports at native speed.

Three moments in the supervisor's deploy make it work:

1. `databricks bundle deploy` builds `libs/agent_governance/dist/*.whl` (the
   `artifacts` block) and installs it into the deploy job's environment.
2. `deploy/log_and_deploy.py` logs the wheel *inside* the model artifact under
   `wheels/` and names it as `wheels/<file>.whl` in the model's requirements —
   the layout MLflow's own `add_libraries_to_model` produces and Model Serving
   installs from. The endpoint never depends on a volume or an index at build
   time, and the library it runs is the one that was tested.
3. `deploy/publish_library.py` copies the same wheel to the platform volume,
   `/Volumes/<catalog>/agent_platform/libs/`, where the other agents install it
   from. A version already present is never overwritten: bump
   `agent_governance.__version__` to release a change.

A consuming agent adds two lines to its own deploy: pull the wheel from the
volume, and log it the way step 2 does. The library's
[README](libs/agent_governance/README.md) shows the imports. When the library
gets its own repository — the natural next step once a second agent consumes
it — `libs/agent_governance/` moves as a unit with its tests and pyproject, and
`publish_library.py` goes with it.

## Environments

`dev` and `prod` are two deployments on Databricks that share nothing. Every
per-environment name derives from the bundle target, so a third environment is a
new target in `databricks.yml` and no other edit:

| | `-t dev` | `-t prod` |
|---|---|---|
| Unity Catalog schema | `workspace.supervisor_dev` | `workspace.supervisor_prod` |
| Registered model | `…supervisor_dev.supervisor_agent` | `…supervisor_prod.supervisor_agent` |
| Serving endpoint | `agents_workspace-supervisor_dev-supervisor_agent` | `agents_workspace-supervisor_prod-supervisor_agent` |
| Prompt alias | `@dev` | `@prod` |
| MLflow experiment | `/Shared/supervisor-agent-dev` | `/Shared/supervisor-agent-prod` |
| Lakebase Postgres schema | `supervisor_dev` | `supervisor_prod` |
| Routing model | sonnet | opus |
| Workers | simulated | live |

One Lakebase instance serves both; the **Postgres schema** is what keeps them
apart — checkpoints, long-term memory, the governed configuration table, the
review queue and the audit trail all land in it. The runtime creates the
schema itself on the first connection, so nothing has to be pre-created.

**`dev` is a deployment, not a mode.** It is held to the same runtime contract
as prod: durable state or refuse to boot, no safety-critical setting disabled,
every turn under a time budget. `ENVIRONMENT=local` — the value a workstation
uses — is the only one that relaxes any of it.

**One value selects an environment.** The bundle target drives the deploy and
`ENVIRONMENT` drives the running process; each derives the prompt alias, the
Unity Catalog schema and the Lakebase schema from that single value. Each
derived name can still be overridden on its own (`PROMPT_ALIAS`,
`PROMPT_CATALOG_SCHEMA`, `LAKEBASE_SCHEMA`, `PLATFORM_CATALOG`) for the shapes
the convention does not cover.

## Configuration

Two layers, deliberately separate:

**Environment variables** — infrastructure and tunables (endpoints, timeouts,
budgets, storage). Defined and documented in `settings.py`. The deployed
endpoint gets these from the bundle (`deploy/log_and_deploy.py` stamps them).

**Governed documents** — `agents.yaml`, `rbac.yaml` and `guardrails.yaml` hold
the worker registry, the role mapping and the guardrail rules. These are
**published to a table**, not baked into the image: after the first deploy a
change is `python deploy/publish_config.py --apply` on its own, and a running
endpoint picks it up within the configuration cache TTL. No redeploy. The files
in `src/supervisor/config/` are the seed and the fallback.

The table lives in the environment's own Lakebase schema, so publishing to dev
cannot change what prod serves. Pass `--lakebase-instance` and
`--lakebase-schema` (or set `LAKEBASE_INSTANCE` / `LAKEBASE_SCHEMA`) to say
which; the script prints the target before it writes.

`guardrails.yaml` also carries the **output policy**: what happens to each
category of sensitive finding in a reply — `allow`, `mask`, `block` or
`escalate` — plus the bulk-disclosure threshold and any canary tokens planted
in real workers' prompts. A publish that sets a withheld category to `allow` is
refused rather than honoured.

## Local development

```
python -m venv .venv
.venv/bin/python -m pip install -e libs/agent_governance -e ".[dev]"   # Windows: .venv\Scripts\python.exe
.venv/bin/ruff check .                                                  # lint and the security ruleset
.venv/bin/pytest                                                        # both test trees, offline
```

`ruff check .` and `pytest` run in CI on every push and pull request, and the
results are uploaded as a retained artifact — see `.github/workflows/ci.yml`,
which also builds the library wheel. `CONTRIBUTING.md` covers what a change is
expected to include.

## Tests

```
pytest                 # 200 tests, under 20 seconds
```

Everything runs **offline** — no workspace, no network, no database. External
systems are faked at their client boundary. A bare `pytest` works in a fresh
clone: each tree's `conftest.py` puts its source on the path.

| | |
|---|---|
| `tests/test_graph.py` | the pipeline end to end, every stage in order |
| `tests/test_guardrails.py` | deterministic deny patterns, then the semantic domain screen |
| `tests/test_failsafes.py` | refuse rather than degrade; the appeal path |
| `tests/test_injection_patterns.py` | the shipped tier-1 patterns: what they catch and must not |
| `tests/test_output_guard_pipeline.py` | response policy and redaction before delivery, through the graph |
| `tests/test_memory.py` | what long-term memory accepts, narrows to the asking agent, and ages out |
| `libs/agent_governance/tests/test_output_guard.py` | the guard itself: tiers, masking, policy rules |
| `libs/agent_governance/tests/test_sensitive_detection.py` | what the sensitive-shape catalogue must catch, and must not |
| `libs/agent_governance/tests/test_audit.py` | the decision trail is written from any identity |
| `libs/agent_governance/tests/test_review_queue.py` | appeals are enumerable, resolved by exactly one reviewer, and a granted retry is spent once |
| `libs/agent_governance/tests/test_locking.py` | one execution per conversation: key derivation, contention refused, plumbing failure fails open |
| `libs/agent_governance/tests/test_rbac.py` | which roles may reach which agent |

These are the contract, not a smoke test: each asserts a governance property the
supervisor is supposed to hold, so a change that breaks one is a change to what
the agent guarantees. Add to them rather than around them.

## Working agreement

Default branch `main`, short-lived feature branches, PR review to merge, no
force-push to `main`. Run `ruff check .` and `pytest` before opening a PR.
