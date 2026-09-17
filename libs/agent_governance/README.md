# agent-governance

The governance layer every agent on the SDLC platform shares. One tested copy,
installed as a wheel, so the supervisor and each worker agent (requirement,
test-case, coding, deployment) apply the same screens and write the same audit
shape.

| Module | What it gives an agent |
|---|---|
| `sensitive_data` | the sensitive-data catalogue (credentials, identifiers, payment, health, PII, internal network detail) with checksums and context gates; `redact_text` for anything about to be persisted |
| `output_guard` | the layer-7 screen: `OutputGuard.screen()` on a reply, `.relay()` on text bound for a model, `.scrub()` on model-written text shown to a user |
| `sanitize` | untrusted text handling: control characters, impersonated turn boundaries, embedded directive frames, size bounds; `untrusted_turn` / `system_blocks` for composing a prompt |
| `deny_rules` | tier-1 deterministic deny patterns and the kill switch, one rule shape for input and output |
| `rbac` | role → agent authorization, and `check_privileges` — the startup check that the governance tables' post-deploy grants were actually applied |
| `retry_and_deadline` | the turn time budget and the bounded, jittered retries that must fit inside it |
| `audit_trail` | the Postgres decision-trail sink with its tamper-evident hash chain, mirrored onto the MLflow trace; `verify_chain` |
| `governed_config_store` | governed configuration in a table: checksummed, validated, TTL-cached, holding the last known-good version on failure |
| `policy_suite_eval` | deterministic evaluation of a guardrails document against a labelled corpus, for CI and the publish gate |
| `lakebase` | Lakebase pools that survive idle replicas, the checkpointer and store builders, the connection source the sinks borrow, SQL identifier validation |
| `environment` | `is_local_environment`, `resource_environment`, `catalog`, `environment_schema` |

## What the wheel deliberately does not contain

These are the platform's job, or the calling application's, and an agent
should not rebuild them:

| Concern | Where it lives |
|---|---|
| Caller authentication and the trust in `custom_inputs` | The serving endpoint's ACL: only the calling application's service principal holds CAN QUERY. A worker trusts a dispatch because only the supervisor's service identity holds CAN QUERY on it |
| Credentials for downstream resources | Automatic auth passthrough for resources declared at `log_model` |
| One turn per conversation at a time | The calling application; Model Serving has no session affinity and no lock of its own |
| Rate limits and per-user quotas | The calling application (AI Gateway rate limits do not apply to agent endpoints) |
| Token accounting and latency | The MLflow trace |
| Request-level logging and correlation | The MLflow trace and the inference table |
| Retention and purge | A scheduled SQL job (DEPLOYMENT.md §9) |

There is also no fallback-to-a-smaller-model path. A governance control that
cannot run means the turn is held and the caller is told to retry — degrading
to a weaker screen would report a decision the screen never made.

## The seven guardrail layers, and who owns each

| Layer | What the wheel gives you | What stays yours |
|---|---|---|
| 1 · Input | `sanitize` (control and bidi characters, impersonated turn boundaries, embedded directive frames, size bounds); `deny_rules` (injection and jailbreak patterns, kill switch) | Payload schema validation, rate limits and malware scanning — these belong in the calling application, ahead of the endpoint |
| 2 · Prompt | `sanitize.system_blocks` (system instructions in a channel user text cannot reach, cacheable prefix); `sanitize.untrusted_turn` (untrusted content JSON-encoded in its own turn); `output_guard.relay` on text bound *for* a model | Your own locked instructions, kept in the system block rather than in a user turn |
| 3 · Memory | `sensitive_data.redact_text` before anything is persisted; `lakebase` pools with one Postgres schema per agent and environment (session separation) | The write allowlist, recall narrowing and retention ceiling for the fields *you* store — the supervisor's `LongTermMemory` is the worked example |
| 4 · Retrieval | `sanitize.clean_inbound_text` on every retrieved chunk before it reaches a prompt | Source and metadata filtering, trust scoring and freshness — they depend on your index and your permissions |
| 5 · Tool | `rbac` (role → agent); `retry_and_deadline` (turn budget, per-call timeout, bounded retries) | Your tool allowlist, and the LangGraph `interrupt()` gate in front of anything irreversible |
| 6 · Runtime | `retry_and_deadline` (turn budget); `audit_trail` (the decision trail and its hash chain, mirrored onto the MLflow trace) | Loop bounds for your own graph, and a recursion ceiling on it |
| 7 · Output | `output_guard.screen` / `.scrub` (policy per category — allow, mask, block, escalate — bulk-disclosure threshold, canary and prompt-leak detection); the `sensitive_data` catalogue | Your response schema, validated by your `ResponsesAgent` contract |

## Where the wheel comes from

The supervisor's `databricks bundle deploy` builds it and its deploy job
publishes it to the platform volume, one file per version, never overwritten:

```
/Volumes/<catalog>/agent_platform/libs/agent_governance-<version>-py3-none-any.whl
```

Bump `agent_governance.__version__` to release a change; the next supervisor
deploy publishes the new file beside the old ones, and each consumer moves to
it when it chooses to. (`READ VOLUME` on that volume is the one grant a
consuming agent's deploy identity needs — DEPLOYMENT.md §7d.)

## Deploying an agent that uses it

Three places in a consuming agent's own repository, and nothing else changes:

**1. Its bundle's job environment**, so the deploy job can import the library
when it logs the model:

```yaml
environments:
  - environment_key: default
    spec:
      client: "2"
      dependencies:
        - -r requirements.txt
        - /Volumes/<catalog>/agent_platform/libs/agent_governance-0.3.0-py3-none-any.whl
```

**2. Its deploy step**, which bakes the same file into the model artifact so
the serving container installs it from the model directory and never depends
on the volume at runtime. This is the layout MLflow's own
`add_libraries_to_model` produces; the supervisor's `deploy/deploy_agent.py`
is the worked example:

```python
WHEEL = Path("/Volumes/<catalog>/agent_platform/libs/agent_governance-0.3.0-py3-none-any.whl")

logged = mlflow.pyfunc.log_model(
    name="coding_agent",
    python_model="src/coding_agent/agent.py",
    code_paths=["src/coding_agent"],
    pip_requirements=[*runtime_pins, f"wheels/{WHEEL.name}"],
    resources=[...],
)
with tempfile.TemporaryDirectory() as staging:            # before register_model:
    wheels = Path(staging) / "wheels"                     # UC copies the model dir
    wheels.mkdir()                                        # when the version is created
    shutil.copy2(WHEEL, wheels / WHEEL.name)
    MlflowClient().log_model_artifacts(logged.model_id, staging)
version = mlflow.register_model(logged.model_uri, "<catalog>.<schema>.coding_agent").version
agents.deploy("<catalog>.<schema>.coding_agent", version, environment_vars={...})
```

**3. Its code.** The pieces every worker agent is expected to apply:

```python
from agent_governance.audit_trail import build_audit_logger
from agent_governance.lakebase import audit_connection_source, build_checkpointer
from agent_governance.output_guard import OutputGuard
from agent_governance.sanitize import clean_inbound_text

SCHEMA = "coding_agent_dev"                                    # this agent's own Lakebase schema

text = clean_inbound_text(request_text)                        # layer 1, on the way in
result = graph.invoke(..., config={"configurable": {"thread_id": conversation_id}})

guard = OutputGuard()                                          # or .from_mapping(governed_doc)
screened = guard.screen(result_text)                           # .text, .action, .blocked, .audit_detail()
build_audit_logger(audit_connection_source(SCHEMA), "coding_agent_audit_log").log(record)
```

Each agent keeps its own Lakebase schema and table names; the library never
guesses either. The worker trusts the dispatch it received because only the
supervisor endpoint's service identity holds CAN QUERY on it — grant that and
nothing else.

## Tests

```
pytest libs/agent_governance/tests
```

Offline: no workspace, no network, no database.

## Migrating

**0.2.0 → 0.3.0.** Seven modules were removed. Five because the platform or
the calling application owns the concern (see the table above): `locking`,
`trust`, `observability`, `retention`, `spend`. `review_queue` because a
guardrail block is final (no appeal path) and an escalation is an `escalated`
row in the audit trail, not a held conversation; a reviewer surface is the
caller's to build later. `grounding` because the solution does not ask for it. Four modules
were renamed so the file name says what it is for. `langchain-core` is no
longer a dependency. A wheel pinned at 0.2.0 on the platform volume is
unaffected — `publish_library` never overwrites a published version.

| 0.2.0 | 0.3.0 |
|---|---|
| `agent_governance.sensitive.*` | `agent_governance.sensitive_data.*` (same names) |
| `agent_governance.locking.thread_lock` | removed — the caller sends one turn per conversation |
| `agent_governance.trust.*` | removed — the endpoint ACL is the trust boundary |
| `agent_governance.observability.*` | removed — the MLflow trace is the request log |
| `agent_governance.retention.*` | removed — SQL recipes in DEPLOYMENT.md §9 |
| `agent_governance.spend.*` | removed — token usage is on the trace; rate limits in the caller |
| `agent_governance.lakebase.lock_connection_source` | removed |
| `agent_governance.review_queue.*` | removed — escalations are audit rows; no appeal path |
| `agent_governance.grounding.*` | removed — not in the solution |
| `agent_governance.output_guard.StreamGuard`, `OutputGuard.stream_should_hold`, `.stream_findings` | removed — the reply is screened whole; no token streaming |
| `agent_governance.audit.*` | `agent_governance.audit_trail.*` (same names) |
| `agent_governance.resilience.*` | `agent_governance.retry_and_deadline.*` (same names) |
| `agent_governance.config_store.*` | `agent_governance.governed_config_store.*` (same names) |
| `agent_governance.policy_eval.*` | `agent_governance.policy_suite_eval.*` (same names) |

**0.1.0 → 0.2.0.** Four modules merged into the ones they were inseparable
from; two functions renamed.

| 0.1.0 | 0.2.0 |
|---|---|
| `agent_governance.prompting.untrusted_turn`, `.system_blocks` | `agent_governance.sanitize.untrusted_turn`, `.system_blocks` |
| `agent_governance.deadline.Deadline`, `.deadline_for`, `.BudgetExhausted` | `agent_governance.retry_and_deadline.Deadline`, `.deadline_for`, `.BudgetExhausted` |
| `agent_governance.sql.safe_identifier`, `.table_exists_here`, `.UnsafeIdentifier` | `agent_governance.lakebase.safe_identifier`, `.table_exists_here`, `.UnsafeIdentifier` |
| `agent_governance.privileges.check`, `.report`, `.PrivilegeFinding` | `agent_governance.rbac.check_privileges`, `.report_privileges`, `.PrivilegeFinding` |
