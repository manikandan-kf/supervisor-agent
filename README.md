# Custom LangGraph Supervisor Agent

The Supervisor Agent from *Supervisor Agent — Solution Design & Architecture*
(v1.2): a custom LangGraph graph, served on
Databricks Model Serving as an MLflow `ResponsesAgent`, that takes every user
turn through a fixed pipeline before any worker agent is reached:

```
rbac_gate → guardrails → route / clarify → dispatch → approval → respond + audit
```

This repository holds **only the supervisor and what deploys it**. The
application that calls the endpoint (the chat UI and the service in front of it
that authenticates users and resolves their roles) is a separate deliverable and
is not in this repository. Where the supervisor depends on it, this README says
"the caller" and states exactly what the caller must send.

**To deploy it, follow [DEPLOYMENT.md](DEPLOYMENT.md).**

---

## 0. What one request looks like

The endpoint is a standard Databricks agent endpoint. A caller sends the
conversation as `input` and, in `custom_inputs`, who is asking and for which
worker:

| `custom_inputs` field | What it is | Who trusts it, and why |
|---|---|---|
| `agent_id` | The worker the request is for. Every request names exactly one | Re-checked by the RBAC gate on every turn |
| `permitted_agents` | The worker ids this user may reach | Trusted because only the caller's service principal holds CAN QUERY on the endpoint; with it absent, `config/rbac.yaml` stands in |
| `approvable_agents` | The worker ids this user may approve staged work for | Same |
| `user_role`, `user_id` | The persona, for audit attribution, and a pseudonymous user reference for long-term memory | Never used for authorization |
| `conversation_id` | The thread; the checkpointer keys on it | A new id starts a fresh conversation |
| `resume` | `{"decision": "approve" | "reject", "comment": ...}` when answering an approval gate | Resumes the paused graph |

Every turn then walks the fixed pipeline and ends in one of these outcomes,
returned in `custom_outputs.outcome` beside the text:

| Outcome | Meaning |
|---|---|
| `answer` | A worker answered, the reply passed the output guard, here it is |
| `blocked` | Access denied, off-domain, or the reply failed the output guard. Final |
| `clarify` | The supervisor needs one detail before it can route; reply and it resumes |
| `approval_pending` | A staged artifact is waiting for `resume` |
| `escalated` | The clarification cap, an `escalate` rule, a block streak or the output guard handed this to a human; it is recorded in the audit table |
| `expired` | The conversation sat idle past the retention window; start again |
| `error` | Nothing was judged: a model or the audit sink was unavailable, or the turn ran out of time. Retry |

In the AI Playground, `custom_inputs` go in the request's custom inputs panel;
`agent_id` is the only required field for a first call.

---

## 1. What Databricks does for us, and what we build

Before reading the file list, this is the line between platform and code. Each
row is what Databricks Model Serving / Agent Framework provides natively for a
custom agent endpoint, and what this repository therefore does or does not
build. Sources are the Databricks docs as of September 2026.

| Concern | Databricks provides natively | What this repo does |
|---|---|---|
| Serving, scaling, request queuing | Provisioned concurrency, autoscaling, 429 on overload, endpoint metrics (latency, QPS, errors) | Nothing. `deploy/deploy_agent.py` only sets workload size |
| Caller authentication | OAuth M2M service principals; endpoint ACL (`CAN QUERY`) enforced before the request reaches code | Trusts `custom_inputs` **because** only the caller's service principal holds CAN QUERY (DEPLOYMENT.md §7a, §7e). No request signing |
| Credentials for the routing LLM, Lakebase, worker endpoints | Automatic auth passthrough: resources declared at `log_model` get short-lived, rotated M2M tokens | Declares the resources (`deploy_agent.py`); no secrets in the container |
| Tracing | MLflow Tracing (`mlflow.langchain.autolog()`), persisted to Unity Catalog Delta tables; inference tables with request, response, requester, latency | One `autolog()` call in `serving_entrypoint.py`; traces bound to UC tables by the deploy script (solution §08). No custom span or log-correlation code |
| Token accounting and per-user token limits | Token usage on every LLM span of the trace | Nothing. The §07 token KPI is a query over the trace tables. There is no token cap or per-user quota in the code: every call runs as the one service principal, so a per-user limit can only be enforced by the caller, which knows the user (§04) |
| Rate limiting / per-user quotas | AI Gateway rate limits exist for foundation-model and custom-model endpoints, **not for agent endpoints** | Nothing. The solution puts rate limiting in front of the supervisor, in the caller (§04) |
| Per-conversation serialization | **Not provided.** No session affinity; concurrent requests on one `conversation_id` can run on different replicas | Nothing. The caller sends one turn per conversation at a time (DEPLOYMENT.md §7e) |
| Retries, circuit breakers, request deadline | Server-side timeout of 597 s per request; no server-side retry; SDK retries 429/503 | `retry_and_deadline.py` (turn budget, bounded retries) and the circuit breaker in `worker_client.py` — solution §05 asks for exactly this |
| Guardrails on the agent's own input/output | AI Gateway safety/PII guardrails apply to foundation-model and external-model endpoints, **not to agent endpoints**, and not to streamed output | The two-tier input screen and the layer-7 output guard are code (solution §02, §04) |
| Conversation memory | `databricks_langchain.CheckpointSaver` / `DatabricksStore` over Lakebase | `lakebase.py` wraps those two classes (schema per environment, refuse to run without durable state) |
| Prompt versioning | MLflow Prompt Registry in Unity Catalog, aliases per environment | `prompt_registry.py` loads by alias; `deploy/register_prompts.py` registers |
| Audit tables | Trace tables and inference tables are best-effort and delivered within an hour | `audit_trail.py` writes the decision trail synchronously to a Postgres table, because a governance *decision* must not be reported as applied until its record landed (solution §02 "Response & audit") |
| Human review | Nothing equivalent for a custom agent (the Review App is for developer feedback) | Nothing. An escalation is an `escalated` decision in the audit table; a reviewer surface is the caller's, later (§3) |
| Retention / purge | Nothing purges checkpoints, traces or inference tables | SQL recipes in DEPLOYMENT.md §9; no code |

Two platform notes worth knowing. Databricks now recommends **Databricks Apps**
for new agents and keeps Model Serving as the supported path for existing ones;
the solution chose Model Serving and nothing here prevents a later move. And
`agents.deploy()` wires tracing, inference tables and the Review App in one
call; the `serving-api` deploy method (DEPLOYMENT.md §6a) exists only for
workspaces where that call is refused.

## 2. Every file, and why it is here

Each entry says what the file does, which part of the solution document it
serves, and whether it is **required** (the solution asks for it, or the code
cannot run without it) or a **product feature** the solution does not ask for
and that can be removed on its own. Nothing in the tree is there for a reason
not stated below.

### Top level

| File | What it is |
|---|---|
| `README.md` | This file |
| `DEPLOYMENT.md` | Everything the supervisor needs from a workspace, everything it creates, the step order, the grants and the verification |
| `CHANGELOG.md` | What changed between source versions |
| `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE` | How to contribute, how to report a vulnerability, licence |
| `databricks.yml` | The Databricks Asset Bundle: variables, the library wheel artifact, the sync set, the `dev` and `prod` targets (solution §08 "CI/CD with DABs") |
| `resources/deploy_job.yml` | The deploy job: four tasks, one per script in `deploy/` |
| `resources/schema.yml` | The Unity Catalog schema each environment owns (model, prompts, config, trace tables) |
| `resources/lakebase.yml` | The Lakebase instance and library volume as bundle resources. Ships disabled; the file says why |
| `pyproject.toml` | Project metadata, the dependency mirror of `requirements.txt`, ruff / pytest / mypy configuration |
| `requirements.txt` | What the serving container installs, pinned exactly; baked into the model artifact |
| `requirements-dev.txt` | Tooling for local work and CI |
| `.github/workflows/ci.yml` | Lint, security ruleset, tests, wheel build, dependency audit and secret scan on every push |
| `.github/CODEOWNERS`, `.github/dependabot.yml` | Review ownership of the governance surface; dependency update PRs |
| `.gitattributes`, `.gitignore` | Line endings; build output and local state kept out of git |

### `src/supervisor/` — the agent

| File | What it does | Solution reference | Needed? |
|---|---|---|---|
| `serving_entrypoint.py` | The Model Serving entrypoint: an MLflow `ResponsesAgent` whose `predict` runs one synchronous turn of the graph (solution pattern 01). `predict_stream` runs the same turn and emits the finished answer as one event, because the AI Playground and the SDK send every request as a stream. The file the deploy script logs as the model | §02 "Databricks ResponsesAgent", "Synchronous relay" | Required |
| `graph.py` | Wires the six nodes into a `StateGraph` with the Lakebase checkpointer and store. The stage order is the graph's edges, so it cannot be skipped | §03 "Deterministic, hardcoded stage order" | Required |
| `state.py` | Two schemas: `SupervisorState` (checkpointed conversation state) and `SupervisorContext` (who is asking, supplied per turn and never checkpointed) | §04 confused-deputy mitigation | Required |
| `services.py` | The dependency container the nodes read collaborators from, and `build_services()` which assembles the production ones | — (wiring) | Required |
| `settings.py` | Every tunable, read from environment variables, with `validate()` / `enforce()` refusing to serve a deployed environment with a safety control switched off | §05 "code failsafes" | Required |
| `nodes/__init__.py` | `SupervisorNodes`: the stage classes composed into one object the graph registers | — (wiring) | Required |
| `nodes/base.py` | `NodeBase` (the container binding) and the helpers every stage uses: reading the conversation, trimming history, shaping an audit entry, session age, the relay screen | — (shared by the stages) | Required |
| `nodes/rbac_gate.py` | Stage 1. Re-validates the requested agent against the caller's permitted set on every turn; enforces the input size bound and the session lifetime | §02 "RBAC gate", §04 IDOR mitigation | Required |
| `nodes/guardrails.py` | Stage 2. Runs the two-tier screen and answers small talk itself. A block is final; an `escalate` rule or a streak of blocks is recorded as an escalation | §02 "Guardrails" | Required |
| `nodes/route.py` | Stage 3. Resolves the context the worker needs (session context first, long-term memory as a seed) or asks one clarifying question | §02 "Route / clarify" | Required |
| `nodes/dispatch.py` | Stage 4. Calls the worker, screens the conversation on the way out and the reply on the way back, and stages an artifact for approval when required | §02 "Dispatch", §05 "Worker agent timeout" | Required |
| `nodes/approval.py` | Stage 4b. The LangGraph `interrupt()` that pauses for human sign-off, in its own node so a resume does not re-call the worker; records who decided | §02 "Human-in-the-loop approval gates" | Required |
| `nodes/respond.py` | Stage 5. The single exit: writes the audit record (synchronously for governance decisions), then emits the answer | §02 "Response & audit" | Required |
| `nodes/failsafes.py` | What any stage does when the turn's time budget is exhausted or the clarification cap is reached (escalate to a human) | §05 clarification cap, time budget | Required |
| `guardrail_engine.py` | The two-tier screen itself: deterministic deny rules, then a semantic verdict per reachable agent; also the small-talk classifier | §03 "Full control of guardrail logic" | Required |
| `context_resolver.py` | `ContextResolver`, which the route stage calls: structured-output resolution of the context a worker needs (product line, environment …) against the routing LLM, or the one clarifying question to ask | §02 "Route / clarify" | Required |
| `worker_client.py` | The Model Serving client for worker endpoints: per-attempt timeout, bounded retries with jitter, per-agent circuit breaker; plus the simulated worker used until real ones exist | §05 "Retry with backoff, then circuit-break" | Required |
| `agent_registry.py` | `WorkerAgent` and `AgentRegistry`: the worker catalogue built from `config/agents.yaml` | §02 "New agents are onboarded with a registry entry" | Required |
| `governed_config.py` | Validators for the three governed documents and `SupervisorConfig`, which builds the registry, RBAC policy, guardrail engine and output guard from them | §02 "Unity Catalog: config governance" | Required |
| `memory.py` | Pins the shared Lakebase plumbing to this agent's schema, and `LongTermMemory`: validated, allow-listed, expiring per-user context | §02 "Lakebase Postgres", §04 memory poisoning | Required |
| `user_facing_text.py` | Every sentence the supervisor says to the user. Kept apart from the nodes so a wording change is never a logic change | — (every governed message lives here) | Required |
| `prompt_registry.py` | Loads the prompts from the MLflow Prompt Registry by environment alias, with the bundled text as fallback | §06 model strategy, §08 | Required |
| `llm_provider.py` | The `ChatDatabricks` client for the routing LLM, one per endpoint | §06 "approved model list" | Required |
| `config/agents.yaml` | Seed worker registry: id, endpoint, domain scope, required context, approval patterns | §02 | Required |
| `config/rbac.yaml` | Seed role → agent mapping. Fallback only; the caller's `permitted_agents` is authoritative | §01 "Role-to-agent mappings" | Required |
| `config/guardrails.yaml` | Seed tier-1 deny patterns, output policy, canary tokens and the kill switch | §02 "two-tier engine" | Required |
| `config/policy_suite.yaml` | Labelled cases the guardrails document must satisfy; run by the tests and by `publish_config.py` before a publish | §07 "Guardrail block precision / recall" | Required |

### `libs/agent_governance/` — the shared library

The screens, sinks and bounds that every agent on the platform must apply
identically, packaged as one wheel the supervisor bakes into its model and
publishes for the worker agents. [Its README](libs/agent_governance/README.md)
shows how a worker consumes it.

| File | What it does | Solution reference | Needed? |
|---|---|---|---|
| `sensitive_data.py` | The catalogue of sensitive shapes (credentials, government and health identifiers, payment and bank data, PII, internal network detail) with checksums and context gates; `redact_text` for anything persisted | §04, guardrail layer 7 "PII masking" | Required |
| `output_guard.py` | The layer-7 screen: allow / mask / block / escalate per category, canary and prompt-leak detection, applied to the whole reply before it leaves the graph | §02 "The worker's response is validated" | Required |
| `sanitize.py` | Untrusted text at the prompt boundary, both directions: control characters and fake turn boundaries in, JSON-delimited data turns out | Guardrail layers 1 and 2 | Required |
| `deny_rules.py` | The tier-1 rule shape (`pattern`, `reason`, `action`) and the kill switch | §02 "deterministic rules" | Required |
| `rbac.py` | `RbacPolicy` (role → agent) and the startup check that the governance tables' post-deploy grants were applied | §04 "role-to-agent mapping is the sole source of truth" | Required |
| `retry_and_deadline.py` | The turn time budget (`Deadline`) and the bounded, jittered retries that must fit inside it. Model Serving has no server-side retry and a 597 s hard timeout, so this is the only place a hung call is bounded. Time only: nothing here counts or caps tokens | §05 "Code failsafe" | Required |
| `audit_trail.py` | The Postgres decision-trail sink, written synchronously before a governance decision is reported, with a tamper-evident hash chain and a mirror onto the MLflow trace; `verify_chain` | §02 "admin-only audit tables … traced in MLflow 3" | Required |
| `governed_config_store.py` | The governed configuration table the three documents (`agents.yaml`, `rbac.yaml`, `guardrails.yaml`) are published to: checksummed, validated on read, TTL-cached, holding the last known-good version on failure; and `Reloading`, the proxy that lets a published change reach a running endpoint. Without it, onboarding a worker or changing a guardrail rule is a redeploy | §02 "Unity Catalog: config governance", "New agents are onboarded with a registry entry — no supervisor code change" | Required |
| `policy_suite_eval.py` | Runs `config/policy_suite.yaml` against a guardrails document: the regression check the tests and the publish gate share | §07 "Guardrail block precision / recall" | Required |
| `lakebase.py` | The one place the agent touches Lakebase: connection pools that survive idle replicas and credential rotation, the checkpointer (short-term memory) and store (long-term memory) builders, the connection the audit and config tables borrow, SQL identifier validation, and the refusal to run without durable state | §02 "Lakebase Postgres: short-term & long-term memory" | Required |
| `environment.py` | The one place `ENVIRONMENT` is interpreted: which deployment's prompt alias, Unity Catalog schema and Lakebase schema to use (`dev`, `prod`), and whether a control may degrade (only `local`, on a workstation). In the library because `lakebase.py` and `retry_and_deadline.py` need the same answer as the supervisor | §06 two environments, §08 DABs targets | Required |
| `README.md`, `pyproject.toml` | Module map and consumer guide; packaging with floors, not pins | | |
| `tests/` | The library's own offline tests | | |

### `deploy/` — what the bundle's job runs

| File | What it does |
|---|---|
| `deploy_agent.py` | Logs the agent as an MLflow model with the library wheel baked in, registers it in Unity Catalog, binds traces to UC tables, and creates or rolls the serving endpoint |
| `register_prompts.py` | Registers the prompts in the MLflow Prompt Registry and points the environment alias at them |
| `publish_config.py` | Publishes the governed YAML documents to the configuration table, refusing a guardrails document that fails the policy suite |
| `publish_library.py` | Copies the library wheel to the platform volume for the worker agents; never overwrites a version |

### `tests/` — the supervisor's governance contract

All offline: no workspace, no network, no database.

| File | What it pins |
|---|---|
| `test_graph.py` | The pipeline end to end, every stage in order, approval resume, session expiry, escalation |
| `test_guardrails.py` | The two-tier engine, and the policy corpus against the shipped guardrails document |
| `test_failsafes.py` | Fail closed on a model error; a block is final; the time budget and retries |
| `test_injection_patterns.py` | What the shipped tier-1 patterns catch and must not |
| `test_output_guard_pipeline.py` | The output guard through the graph |
| `test_memory.py` | Long-term memory validation and expiry |
| `helpers.py`, `conftest.py` | Stand-ins for the model, workers and audit sink |
| `libs/agent_governance/tests/test_output_guard.py` | The guard and the sanitiser |
| `libs/agent_governance/tests/test_sensitive_detection.py` | The sensitive-shape catalogue |
| `libs/agent_governance/tests/test_audit_trail.py` | The sink and the hash chain |
| `libs/agent_governance/tests/test_rbac.py` | Role → agent decisions and the privilege check |

## 3. Not in the solution document

These behaviours exist in the code and are not asked for by the solution. They
are product decisions rather than platform duplicates, so they were kept; each
is one module or one setting to remove.

- **A second deliverable in one message** is offered back after the first is answered (`guardrails` and `dispatch` nodes).
- **Small talk** answered by the supervisor rather than forwarded (`guardrail_engine.py`).
- **Per-agent governance model** (`MULTI_MODEL_ENABLED`, `model:` in `agents.yaml`); ships dark. §06 describes one model per environment.
- **Hash chain** on the audit table (`audit_trail.py`); §02 asks for admin-only audit tables, which the Postgres grants provide on their own.
- **Kill switch** and **block-streak escalation** in the guardrails stage.

Three things are deliberately **not** built.

- **No streaming.** The solution's pattern 01 is a synchronous relay, and the
  Playground shows a finished answer either way. Token relay, the streamed
  progress checklist and the hold-back guard over live tokens are gone; the
  output guard screens the whole reply inside the graph. When the chat UI
  is integrated and wants tokens as they arrive, that is a change to
  `predict_stream` alone, plus a hold-back guard over live tokens in the library.

- **No appeal path.** §05 says a disputed block "offers an appeal path to a
  human/admin queue". A guardrail block is final: the word "appeal" is screened
  like any other message and nothing can bypass the screen.
- **No review queue and no hold.** "Escalate to a human" (§02, §05) is the
  `escalated` outcome written synchronously to the audit table, which is what a
  reviewer works from; the conversation is not paused. A reviewer surface is
  the caller's to build later, and when it exists it reads escalations from
  that table. The §07 "Human escalation rate" KPI is a query over it.

## 4. Environments

`dev` and `prod` are two deployments that share nothing but the Lakebase
instance. Every per-environment name derives from the bundle target:

| | `-t dev` | `-t prod` |
|---|---|---|
| Unity Catalog schema | `<catalog>.supervisor_dev` | `<catalog>.supervisor_prod` |
| Registered model | `…supervisor_dev.supervisor_agent` | `…supervisor_prod.supervisor_agent` |
| Serving endpoint | `agents_<catalog>-supervisor_dev-supervisor_agent` | `agents_<catalog>-supervisor_prod-supervisor_agent` |
| Prompt alias | `@dev` | `@prod` |
| Lakebase Postgres schema | `supervisor_dev` | `supervisor_prod` |
| Routing model | medium tier | high tier (solution §06) |
| Workers | simulated | live |

`dev` is a deployment, not a mode: it refuses to boot without durable state and
refuses to serve with a safety-critical setting disabled, exactly as prod does.
`ENVIRONMENT=local` is the only value that relaxes that, and only a workstation
uses it.

## 5. Configuration

**Environment variables** hold infrastructure and tunables (endpoints, timeouts,
budgets). They are defined and documented in `settings.py`; the deploy script
stamps them onto the endpoint.

**Governed documents** — `agents.yaml`, `rbac.yaml`, `guardrails.yaml` — are
published to a table in the environment's Lakebase schema with
`python deploy/publish_config.py --apply`. A running endpoint picks a change up
within the cache TTL, so onboarding a worker agent or tightening a rule needs
no redeploy. The files in `src/supervisor/config/` are the seed and the
fallback.

## 6. Local development

```
python -m venv .venv
.venv/bin/python -m pip install -e libs/agent_governance -e ".[dev]"   # Windows: .venv\Scripts\python.exe
ruff check .            # lint and the security ruleset
ruff format --check .
pytest                  # both test trees, offline, about ten seconds
```

CI runs the same on every push and pull request. `CONTRIBUTING.md` says what a
change is expected to include; the one rule that matters most: if a second
agent would need it, it belongs in `libs/agent_governance`, never in
`src/supervisor`.
