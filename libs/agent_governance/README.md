# agent-governance

The governance layer every agent on the SDLC platform shares. One tested copy,
installed as a wheel, so the supervisor and each worker agent (requirement,
test-case, coding, deployment) apply the same screens and write the same audit
shape.

| Module | What it gives an agent |
|---|---|
| `sensitive` | the sensitive-data catalogue (credentials, identifiers, payment, health, PII) with checksums and context gates; `redact_text` for anything about to be persisted |
| `output_guard` | the layer-7 screen: `OutputGuard.screen()` on a reply, `.relay()` on text bound for a model, `.scrub()` on model-written text shown to a user, `StreamGuard` over live tokens |
| `sanitize` | untrusted text handling: control characters, impersonated turn boundaries, embedded directive frames, size bounds |
| `deny_rules` | tier-1 deterministic deny patterns and the kill switch, one rule shape for input and output |
| `grounding` | execution claims and citations nothing in the turn backs |
| `trust` | HMAC signing and verification of entitlements and dispatches — a worker calls `verify_dispatch` |
| `rbac` | role → agent authorization, and `check_privileges` — the startup check that the governance tables' post-deploy grants were actually applied |
| `resilience`, `spend` | the turn time budget and the bounded retries inside it, and the governance spend ledger |
| `audit` | the Postgres decision-trail sink with its tamper-evident hash chain, mirrored onto the MLflow trace |
| `review_queue` | the appeal / escalation queue: an agent opens and claims, a reviewer surface lists and resolves — same table, same module |
| `locking` | `thread_lock`: one execution per conversation, over a Postgres advisory lock. Model Serving does not serialize turns by conversation; this does |
| `config_store` | governed configuration in a table: checksummed, validated, TTL-cached, falling back to a bundled seed |
| `lakebase` | Lakebase pools that survive idle replicas, the checkpointer and store builders, the connection sources the sinks and the lock borrow |
| `environment` | `is_local_environment`, `resource_environment`, `catalog`, `environment_schema` |

## The seven guardrail layers, and who owns each

The platform's guardrail model has seven layers. This wheel is where the layers
that must be *identical* across agents live — an agent that reimplements one has
already diverged from every other. The rest are each agent's own, because they
depend on what that agent retrieves, stores and calls.

| Layer | What the wheel gives you | What stays yours |
|---|---|---|
| 1 · Input | `sanitize` (control and bidi characters, impersonated turn boundaries, embedded directive frames, size bounds); `deny_rules` (injection and jailbreak patterns, kill switch) | Payload schema and MIME validation, rate limits and malware scanning — these belong at the front door, ahead of the endpoint |
| 2 · Prompt | `sanitize.system_blocks` (system instructions in a channel user text cannot reach, cacheable prefix); `sanitize.untrusted_turn` (untrusted content JSON-encoded in its own turn — context boundaries and role separation); `output_guard.relay` on text bound *for* a model | Your own locked instructions, and keeping them in the system block rather than in a user turn |
| 3 · Memory | `sensitive.redact_text` before anything is persisted; `lakebase` pools with one Postgres schema per agent and environment (session separation) | The write allowlist, recall narrowing and retention ceiling for the fields *you* store — the supervisor's `LongTermMemory` is the worked example |
| 4 · Retrieval | `grounding` (claims and citations the turn holds no evidence for); `sanitize.clean_inbound_text` on every retrieved chunk before it reaches a prompt | Source and metadata filtering, trust scoring and freshness — they depend on your index and your permissions |
| 5 · Tool | `rbac` (role → agent); `trust.verify_dispatch` (act only on a dispatch the supervisor signed); `resilience` (turn budget, per-call timeout, bounded retries); `spend` (per-turn and per-subject ceilings) | Your tool allowlist, and the LangGraph `interrupt()` gate in front of anything irreversible |
| 6 · Runtime | `locking.thread_lock` (one execution per conversation); `resilience` (turn budget); `audit` (the decision trail and its hash chain, mirrored onto the MLflow trace) | Loop bounds for your own graph, and a recursion ceiling on it |
| 7 · Output | `output_guard.screen` / `.scrub` and `StreamGuard` (policy per category — allow, mask, block, escalate — bulk-disclosure threshold, canary and prompt-leak detection); the `sensitive` catalogue; `grounding` cues on unbacked claims | Your response schema, validated by your `ResponsesAgent` contract |

One deliberate absence: there is no fallback-to-a-smaller-model path. A
governance control that cannot run means the turn is held and the caller is
told to retry — degrading to a weaker screen would report a decision the
screen never made.

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
        - /Volumes/<catalog>/agent_platform/libs/agent_governance-0.2.0-py3-none-any.whl
```

**2. Its log-and-deploy step**, which bakes the same file into the model
artifact so the serving container installs it from the model directory and
never depends on the volume at runtime. This is the layout MLflow's own
`add_libraries_to_model` produces; the supervisor's `deploy/log_and_deploy.py`
is the worked example:

```python
WHEEL = Path("/Volumes/<catalog>/agent_platform/libs/agent_governance-0.2.0-py3-none-any.whl")

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
from agent_governance.audit import build_audit_logger
from agent_governance.lakebase import audit_connection_source, build_checkpointer, lock_connection_source
from agent_governance.locking import thread_lock
from agent_governance.output_guard import OutputGuard
from agent_governance.sanitize import clean_inbound_text
from agent_governance.trust import trust_secret, verify_dispatch

SCHEMA = "coding_agent_dev"                                    # this agent's own Lakebase schema

if not verify_dispatch(custom_inputs, trust_secret()):          # the supervisor signed this dispatch
    raise PermissionError("dispatch signature missing or invalid")

with thread_lock(conversation_id, connection_source=lock_connection_source(SCHEMA),
                 namespace="coding-agent") as acquired:         # one turn per conversation
    if not acquired:
        return busy_reply()
    result = graph.invoke(..., config={"configurable": {"thread_id": conversation_id}})

guard = OutputGuard()                                          # or .from_mapping(governed_doc)
screened = guard.screen(result_text)                           # .text, .action, .blocked, .audit_detail()
build_audit_logger(audit_connection_source(SCHEMA), "coding_agent_audit_log").log(record)
```

Each agent keeps its own Lakebase schema and table names; the library never
guesses either.

## Tests

```
pytest libs/agent_governance/tests
```

Offline: no workspace, no network, no database.

## Migrating from 0.1.0

0.2.0 merged four modules into the ones they were inseparable from. No function
changed behaviour; two were renamed because their new module already had a
`check`. A wheel pinned at 0.1.0 on the platform volume is unaffected —
`publish_library` never overwrites a published version.

| 0.1.0 | 0.2.0 |
|---|---|
| `agent_governance.prompting.untrusted_turn`, `.system_blocks` | `agent_governance.sanitize.untrusted_turn`, `.system_blocks` |
| `agent_governance.deadline.Deadline`, `.deadline_for`, `.BudgetExhausted` | `agent_governance.resilience.Deadline`, `.deadline_for`, `.BudgetExhausted` |
| `agent_governance.sql.safe_identifier`, `.table_exists_here`, `.UnsafeIdentifier` | `agent_governance.lakebase.safe_identifier`, `.table_exists_here`, `.UnsafeIdentifier` |
| `agent_governance.privileges.check`, `.report`, `.PrivilegeFinding` | `agent_governance.rbac.check_privileges`, `.report_privileges`, `.PrivilegeFinding` |
