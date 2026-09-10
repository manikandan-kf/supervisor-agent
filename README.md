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

## Layout

```
src/supervisor/
  agent.py           MLflow ResponsesAgent wrapper — predict / predict_stream
  graph.py           StateGraph wiring, retry policy, durability
  nodes.py           the six stages above
  state.py           the conversation state channels
  context.py         per-request runtime context (role, entitlements, identity)
  registry.py        worker agent registry
  rbac.py            role → agent authorization
  guardrails.py      two-tier domain screen
  routing.py         context resolution and clarification
  dispatch.py        worker client: timeout, retry, circuit breaker
  sensitive.py       the sensitive-shape catalogue every boundary reads
  output_guard.py    response policy applied before delivery, and on the relay
  grounding.py       execution claims and citations nothing in the turn backs
  sanitize.py        untrusted worker output handling
  progress.py        the task-plan events streamed while a turn runs
  memory.py          checkpointer, long-term store, Postgres wiring
  session_notes.py   "keep this in mind for later" — short-term, this thread only
  config_store.py    governed configuration read from Unity Catalog / Postgres
  review_queue.py    appeal and escalation queue
  audit.py           decision-trail sinks, incl. the tamper-evident hash chain
  spend.py           per-turn and per-subject cost ceilings
  deadline.py        per-turn time budget
  locking.py         one active turn per conversation
  settings.py        every tunable, read from the environment
  config/            agents.yaml · rbac.yaml · guardrails.yaml (seed documents)

deploy/log_and_deploy.py   logs, registers and deploys the model
scripts/                   deploy, verify and operate — see DEPLOYMENT.md
tests/                     the governance contract, offline — see Tests below
databricks.yml             the asset bundle
pyproject.toml             project metadata, dependencies, ruff and pytest config
requirements.txt           what the serving container installs — pinned exactly
.github/workflows/ci.yml   lint, SAST, tests, SBOM, dependency and secret scans
```

## Operating it

Beyond deployment, the scripts that answer the questions this agent gets asked
in production:

| | |
|---|---|
| `verify_audit_chain.py` | walk the audit hash chain — was history altered? |
| `trace_turn.py` | the whole flow of one turn: every stage, every model call |
| `inspect_lakebase.py` | what is actually in short-term and long-term memory |
| `kpi_report.py` | KPIs computed from the data already stored |
| `show_prompts.py` | what is registered, what the endpoint loads, and the diff |
| `verify_stream.py` | does the deployed endpoint really stream |
| `erase_subject.py` | erase or purge a subject's long-term memory (GDPR Art. 17) |
| `export_delta_mirror.py` | backfill the decision trail into governed Delta tables |

Each takes `-e/--environment` (or the equivalent) and derives the rest, so none
of them can address two environments at once. All are read-only except
`erase_subject.py`, which requires an explicit `--apply`.

**This repository is the supervisor agent only.** The identity provider, the
front door that calls this agent's API, and the user interface are separate
deployables owned by other teams. Nothing here contains or imports them; where
the documentation names "the caller", it means whichever service holds the
endpoint's `CAN_QUERY` grant and signs the entitlements the RBAC gate checks.

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
review queue and the audit trail all land in it. Point a target at its own
`lakebase_instance` when the environments need physically separate databases.

**`dev` is a deployment, not a mode.** It is held to the same runtime contract
as prod: durable state or refuse to boot, no safety-critical setting disabled,
every turn under a time budget. `ENVIRONMENT=local` — the value in
`.env.example`, because that file configures a workstation — is the only one that
relaxes any of it.

**One value selects an environment, everywhere.** The bundle target drives the
deploy, `-e/--environment` drives the operator scripts, and `ENVIRONMENT` drives
the running process — each derives the prompt alias, the Unity Catalog schema and
the Lakebase schema from that single value, so they cannot disagree:

```
databricks bundle deploy -t prod          # deploys prod
python scripts/verify_deployment.py -e prod   # verifies prod
ENVIRONMENT=prod                          # the process reads prod
```

Each derived name can still be overridden on its own (`PROMPT_ALIAS`,
`PROMPT_CATALOG_SCHEMA`, `LAKEBASE_SCHEMA`, `SUPERVISOR_CATALOG`) for the shapes
the convention does not cover — a second isolated copy of one environment, a
release candidate pinned under another alias, or a migration from a
single-schema deployment (`LAKEBASE_SCHEMA=public`). Overriding is how you point
a workstation at another environment's data, and it is deliberately more than a
one-word edit.

Create each environment's Lakebase schema before its first deploy —
[DEPLOYMENT.md](DEPLOYMENT.md) has the command and the reason it cannot be
skipped.

## Configuration

Two layers, deliberately separate:

**Environment variables** — infrastructure and tunables (endpoints, timeouts,
budgets, storage). Defined in `settings.py`; copy `.env.example` to `.env` for
local script runs. The deployed endpoint gets these from the bundle, not from
`.env`.

**Governed documents** — `agents.yaml`, `rbac.yaml` and `guardrails.yaml` hold
the worker registry, the role mapping and the guardrail rules. These are
**published to a table**, not baked into the image: after the first deploy a
change is `python scripts/publish_config.py --apply` on its own, and a running
endpoint picks it up within the configuration cache TTL. No redeploy. The files
in `src/supervisor/config/` are the seed and the fallback.

The table lives in the environment's own Lakebase schema, so publishing to dev
cannot change what prod serves. `publish_config.py` prints which schema it is
about to write to; pass `--lakebase-schema` to target another.

`guardrails.yaml` also carries the **output policy**: what happens to each
category of sensitive finding in a reply — `allow`, `mask`, `block` or
`escalate` — plus the bulk-disclosure threshold and any canary tokens planted
in real workers' prompts. Defaults withhold credentials, government
identifiers, payment and bank data and health identifiers, and mask names,
contact details, dates of birth, health conditions and internal network
detail. Every one is a line to change and a publish to apply; a publish that
sets a withheld category to `allow` is refused rather than honoured.

## Local development

```
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"   # Windows: .venv\Scripts\python.exe
.venv/bin/ruff check .                        # lint and the security ruleset
.venv/bin/pytest                              # the governance contract, offline
```

`ruff check .` and `pytest` run in CI on every push and pull request, and the
results are uploaded as a retained artifact — see `.github/workflows/ci.yml`.
`CONTRIBUTING.md` covers what a change is expected to include, and how to move a
dependency pin.

## Tests

```
pytest                 # 219 tests, under 5 seconds
```

Everything runs **offline** — no workspace, no network, no database. External
systems are faked at their client boundary, and each test says which. A bare
`pytest` works in a fresh clone: `tests/conftest.py` puts `src/` on the path, so
nothing needs installing or exporting first.

One file per stage of the pipeline, plus the properties that cut across it:

| | |
|---|---|
| `test_rbac.py` | which roles may reach which agent |
| `test_guardrails.py` | deterministic deny patterns, then the semantic domain screen |
| `test_graph.py` | the pipeline end to end, every stage in order |
| `test_audit.py` | the decision trail is written, and its hash chain |
| `test_failsafes.py` | refuse rather than degrade |
| `test_injection_patterns.py` | untrusted content stays data, never instructions |
| `test_small_talk.py` | deterministic classification before any model call |
| `test_output_guard.py` | response policy and redaction before delivery |
| `test_sensitive_detection.py` | what the sensitive-shape catalogue must catch, and must not |
| `test_windowing.py` | the conversation window is bounded in tokens, not messages |

These are the contract, not a smoke test: each asserts a governance property the
supervisor is supposed to hold, so a change that breaks one is a change to what
the agent guarantees. Add to them rather than around them.

`ruff.toml` is the lint and SAST baseline — the `S` ruleset is a port of
flake8-bandit, so `ruff check .` covers both. Every suppression in that file
documents the enforced property it rests on.

## Working agreement

Default branch `main`, short-lived feature branches, PR review to merge, no
force-push to `main`. Run `ruff check .` before opening a PR.
