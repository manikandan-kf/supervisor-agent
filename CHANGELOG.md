# Changelog

Notable changes to the supervisor agent. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version here identifies the **source**. A running deployment is identified
by its Unity Catalog registered-model version, assigned at deploy time.

## [1.2.0] — 2026-09-15

Production-readiness pass. Four controls that existed in design but not in a
deployed environment, and the first automated check in front of a config
publish.

### Added

- **A second endpoint deploy path** (`log_and_deploy.py --deploy-method
  serving-api`, bundle variable `deploy_method`). `databricks.agents.deploy()`
  stays the default; the alternative creates or rolls the endpoint through the
  Model Serving SDK for workspaces where `agents.deploy()` fails on the Agent
  Framework registration or the AI Gateway check at create time. It derives the
  same `agents_<catalog>-<schema>-<model>` name (checked against
  databricks-agents 1.11.0's own derivation, truncation and suffix strip
  included), sets the same tracing variables and `MONITOR_EXPERIMENT_ID` tag,
  and requests inference tables after the endpoint exists so a refusal is
  printed rather than fatal. Credentials are unaffected: they come from the
  model's declared `resources` on either path. Also new: `--endpoint-name`,
  `--workload-size`, `--scale-to-zero` and `--wait-minutes` (block until
  `READY`, fail the job otherwise); defaults reproduce the previous behaviour
  exactly.
- **A policy regression suite** (`src/supervisor/config/policy_suite.yaml`,
  `agent_governance.policy_eval`). 56 labelled cases run
  against the governed guardrails document by `pytest` in CI and — the
  reason it exists — by `publish_config.py --apply` against the *candidate*
  document before it is written. A published guardrails change reaches a live
  endpoint within the config cache TTL with no redeploy, so until now nothing
  automated stood between an edited regex and production traffic. Roughly half
  the corpus is ordinary SDLC work that must pass untouched: a suite that only
  measures what gets caught drifts toward refusing everything.
- **`agent_governance.rbac.check_privileges`** — a startup check that the §7b REVOKEs were
  actually run. Until they are, the serving identity holds full DML on the
  configuration that governs it and the audit trail that records it, and
  nothing noticed. Logs at ERROR rather than refusing to serve, because the
  documented bootstrap order has the runtime create its own tables first.
- **`audit.verify_chain`** — walks the hash chain and reports whether history
  is intact, with `head_id`/`head_hash` for external anchoring. The chain was
  written and never read; tamper evidence nothing evaluates is a column pair,
  not a control.
- **`agent_governance.retention`** — retention sweeps and subject erasure,
  restoring the capability the operator-script cleanup removed, as a library
  function called from a job rather than a script of its own. Dry run by default, verifies before reporting success, and records
  the deletion *through the audit logger* so the hash chain stays valid — which
  is what a hand-written INSERT could not do.
- **`agent_governance.observability`** — one JSON object per log line, each
  carrying the turn's correlation, conversation and agent ids, bound once in
  the entrypoint. `correlation_id` previously reached the audit table and the
  MLflow trace and nothing else, so a container log line could not be tied to
  a request.
- mypy, gated in CI on the shared library (clean) and reported on the
  supervisor package; coverage reporting (66% at this commit).

- **MLflow traces persist to Unity Catalog Delta tables.**
  `log_and_deploy.py --trace-catalog-schema <catalog>.<schema>` binds the
  experiment's `trace_location` (MLflow >= 3.14), and the bundle passes the
  target's own schema, so traces land in governed, SQL-queryable OTel Delta
  tables with no per-experiment cap — the observability layer solution §08
  asks for. It had been left in the experiment store on the belief that a
  customer external location was required; that is no longer the case. An
  unavailable destination falls back to the experiment store and says why
  rather than failing the deploy.

- **Forty characterization tests for controls that had none**, added to the
  test files that already own each concern rather than as new files: the
  entitlement and dispatch HMAC (`trust`), the turn deadline and bounded retry
  (`resilience`), the subject spend window (`spend`), session notes, and the
  RBAC gate's two untested paths — session expiry and the review hold. They pin
  current behaviour, including the wire contract of the signature bytes the
  Governance Front Door keeps its own copy of. Coverage 67% -> 70%.
- **`ruff format` adopted and enforced in CI.** Applied to 42 files; an AST
  fingerprint of all 64 modules, docstrings excluded, is byte-identical before
  and after, which is the proof that only layout changed.

### Changed

- **`--lakebase-resource skip`** deploys with `LAKEBASE_INSTANCE` stamped on the
  endpoint but the instance *not* declared as a model resource, so the
  admin-only passthrough check never runs. The obvious alternative — passing an
  empty `--lakebase-instance` — is wrong and is called out as such in the
  runbook: it leaves the container with no Lakebase target, and a deployed
  environment then refuses to boot rather than degrading to in-memory state.
- **`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` can
  now be passed through to the endpoint** by `log_and_deploy.py`, as the
  documented fallback for a workspace that will not grant the declared-resource
  passthrough — today a Lakebase dependency is accepted only from a workspace
  admin, and without this the endpoint has no other way to reach durable state.
  Pass `{{secrets/<scope>/<key>}}` references, not literals: the value is
  stamped onto the endpoint configuration, which anyone with CAN_VIEW can read.
  Recorded as a downgrade, in the code and in the runbook: the container then
  acts as one static principal for every call rather than holding a separate
  short-lived credential per declared resource, and the secret is rotated by
  hand. Nothing changes for a deployment that can use the passthrough — unset
  variables are not stamped.

- **The bundle is split into `databricks.yml` plus `resources/*.yml`.** One file
  per resource, merged by an `include` glob, which is the Databricks Asset
  Bundle idiom and keeps the root file from growing a block per resource:
  `resources/schema.yml` (the environment's Unity Catalog schema),
  `resources/deploy_job.yml` (the four-task job) and `resources/lakebase.yml`.
  `databricks.yml` went from 286 lines to 189 and now holds only what is true
  for the whole bundle — variables, the library artifact, the sync set, targets.
  Adding a resource is a new file; removing one is deleting its file, which is
  also the documented way to deploy into a schema you did not create.
  Behaviour-preserving, and checked rather than asserted: the merged
  configuration is equal, key for key, to the previous single file, with the one
  intended exception of a variable description that now points at
  `resources/lakebase.yml`.
- **`resources/lakebase.yml` ships disabled, deliberately.** It carries ready
  declarations for the Lakebase instance and the platform library volume, both
  commented out, both with `lifecycle.prevent_destroy`. Enabling them as they
  stand is wrong for the current design: one instance and one volume are shared
  across targets, and a second target that declares the same name fails its
  deploy on a resource that already exists. The file states the condition — give
  each target its own instance first — so the choice is documented where it is
  made rather than discovered at deploy time.

- **A third comment pass, this time repo-wide and to a stated shape.** The
  earlier two passes took the worst files; this one set a rule and applied it to
  all 64 modules: a standalone comment block is at most two lines, a module
  docstring eight, a function docstring four — one summary line, then only the
  contract or the reason a reader needs in order not to undo something.
  **2,525 lines of comment and docstring came out** (19,917 → 17,392 source
  lines); comment lines fell 2,579 → 1,443 and blocks of four or more
  consecutive comment lines 267 → 42, none longer than four. What stayed is
  every spec reference, every fail-closed and append-only invariant, every
  per-pattern rationale in `sensitive.py`, and the traps that read as arbitrary
  until someone widens them — the `{0,20}` card-fragment gap, the turn budget
  whose sum must be re-checked against the gateway timeout, the retry that is
  local so a screen is not repeated. What went is narrative: what the code used
  to do, what was tried and reverted, incidents retold at length. Proven
  mechanical rather than claimed: the SHA-256 of every module's AST with
  docstrings blanked is identical before and after for all 64 files, so not one
  code token moved.

- **`nodes.py` is now the `nodes/` package** — one module per graph stage, plus
  `common.py` for what several stages share and `limits.py` for what any of
  them does when time, budget or patience runs out. It had reached 3,052 lines,
  which meant a reviewer looking at the approval gate had to page past the
  worker dispatch to reach it. A pure move: every body is byte-identical, the
  stages are composed back into one `SupervisorNodes` object, and `graph.py`
  sees exactly what it saw before. The one node still worth splitting is
  `guardrails`, at 839 lines; it shares too many locals across its guard blocks
  to extract without changing behaviour, so it was left alone and said so here
  rather than pretended away.
- **The comments were cut back.** They carried the reasoning behind every
  governance decision, and a lot of that reasoning had become narrative — what
  the code used to do, what was tried and taken back out, an incident retold in
  three paragraphs. Roughly 700 lines of comment and docstring came out across
  21 files; prose went from 37% of the source tree to 34%, and from 42-58% to
  26-36% in the files that were worst. What stayed is the *why* a reader needs
  in order not to undo something: the regex that is flat because the nested form
  was a 353-second denial of service, the field ordering that is the mitigation
  for an over-refusal, the checks that fail open and the reason each one does.
  What went is the history git already holds.
- **Two graph declarations collapsed into one.** `graph.py` passed
  `destinations=` to `add_node` for every stage, repeating what each node's own
  `Command[Literal[...]]` return annotation already declares — two places free to
  disagree about the same fact. The annotation sits on the function that returns
  the `Command`, so that is the one kept.
- **No two modules in `src/supervisor/` share a name any more.** The stage split
  had produced `nodes/guardrails.py` beside `guardrails.py` and
  `nodes/dispatch.py` beside `dispatch.py`. Python resolves those fine — they
  are different packages, and this was never a bug — but a traceback, a grep
  and an editor tab all showed the same word for two different files, and
  `from ..guardrails import` inside `nodes/guardrails.py` is a line a reader has
  to stop at. One rule now decides every name: a module named for a node holds
  that node and is spelled exactly as `graph.py` registers it, because that
  string is the graph's own vocabulary (`add_node`, checkpoints, stream events,
  traces); everything else is named for what it is. So `nodes/access.py` →
  `nodes/rbac_gate.py`, and the collaborators the stages call became
  `guardrail_engine.py` and `worker_client.py`. `nodes/common.py` → `turn.py`
  for a related reason: `common`, like `utils`, names where code ended up rather
  than what it is, and is where unrelated things collect.
- **`Services` fields are protocols instead of `object`.** Six collaborators were
  annotated `object`, with the interface each must provide written in a comment
  beside it. `object` says the value has *no* attributes, so every
  `self.s.audit.log(...)` in the graph was a type error — invisible only because
  `self.s` was itself untyped, which meant mypy checked nothing inside three
  thousand lines of node code. Giving `NodeBase.s` a type turned that checking
  on and the count went 21 → 73; typing the fields structurally (`AuditSink`,
  `ReviewSink`, `ResponseGuard`, `ContextResolver`, `SpendWindow`, and the
  `WorkerClient` protocol that already existed) took it to 14, and all 14 that
  remain are third-party stub gaps in MLflow and LangGraph. Structural, so the
  stand-ins the tests pass still satisfy them and nothing changes at runtime.
  `registry`, `rbac` and `guardrails` stay `Any` deliberately: in production
  they are `Reloading` proxies resolving attributes through `__getattr__`, which
  no structural type can describe.
- **Each stage mixin now type-checks on its own.** `GuardrailsMixin`,
  `RouteMixin` and `DispatchMixin` extend `LimitsMixin` rather than `NodeBase`,
  because they call `_budget_exhausted` and `_clarify_or_escalate`; before, they
  were only correct once `SupervisorNodes` composed them, which is the classic
  way a mixin split loses the checking it was supposed to keep.
- **The policy corpus moved to `src/supervisor/config/`**, beside the
  `guardrails.yaml` it constrains, so a rule and the expectation pinning it are
  edited, reviewed and versioned in one place — and `agent_governance.policy_eval`
  grew the `load_suite` both callers now share, replacing an `evaluate_document`
  nobody called and a `sys.path` hack in the publish gate that imported a
  loader out of a sibling script.
- **The newer tests were folded into the per-area modules that already
  existed** rather than added beside them: the hash chain, retention and
  structured logging into `test_audit.py` (one subject — what an incident
  responder can rely on afterwards), privileges into `test_rbac.py`, sanitize
  and grounding into `test_output_guard.py`, the policy corpus into
  `test_guardrails.py`. Same 311 tests, six fewer files.

- **`SUPERVISOR_TRUST_SECRET` is now reachable and required.**
  `deploy/log_and_deploy.py` passes it to the endpoint and `Settings.validate`
  refuses to serve a deployed environment without it. The RBAC gate treats a
  caller-supplied `permitted_agents` set as authoritative, and the HMAC is what
  makes that safe — but the variable appeared nowhere except two source
  comments, so the check was dark in every deployed environment and any
  principal with CAN QUERY could assert its own entitlements. DEPLOYMENT.md
  §7e covers provisioning and rotation.
- `gitleaks` is now a release gate. A committed credential needs no triage to
  be wrong, unlike a new CVE — `pip-audit` stays advisory pending an owner.
- **Seven modules merged into the four they were already inseparable from**,
  cutting the tree from 54 to 47 and removing the hop a reader had to take to
  understand one idea. `sql` -> `lakebase` (identifier validation is only ever
  used by the Postgres stores), `deadline` -> `resilience` (a budget and the
  retries that must fit inside it are one control), `prompting` -> `sanitize`
  (untrusted text at the prompt boundary, both directions), `privileges` ->
  `rbac` as `check_privileges` (access control, both directions), `context` ->
  `state` (the two schemas `StateGraph` is constructed from, with the boundary
  between them spelled out in one place), `progress` -> `messages` as
  `progress`/`progress_sources` (everything the user is shown). No behaviour
  changed; the public names are unchanged except the two noted.
- **`agent_governance` is 0.2.0.** The public module map changed (the four
  merges above), so the version moved with it; the README carries the
  migration table. The 0.1.0 wheel on the volume is untouched.
- **One writer lookup in `messages.py`.** `progress` and `progress_sources`
  each acquired LangGraph's stream writer with the same guarded block; it is
  now `_stream_writer()`, called by both.
- **The library's tracing test skips without MLflow** instead of erroring —
  `mlflow` is the `tracing` extra, and the suite has to run against a minimal
  install of the wheel, which is how it is now verified.
- **Another ~150 lines of comment and docstring removed**, on the same rule as
  the previous pass: narrative out, constraint in. What is left in the heaviest
  files — `settings.py`, `sensitive.py`, `state.py` — is per-setting operator
  documentation and per-pattern security justification, which is the auditable
  part and is staying.

### Fixed

- **A seven-line comment block 79 lines from the code it described.** It
  explained the history slice and the screen-failure handler in
  `nodes/guardrails.py` from a position where neither was in view; an earlier
  edit had moved the code out from under it. Removed, with the two live facts
  folded into the call sites.
- **Three stale claims about library versions and module names.**
  `agent.py` and `settings.py` both said LangGraph's default `recursion_limit`
  is 25 — it is 10007 in 1.2, which makes the explicit ceiling more necessary,
  not less. `audit.py` said traces cannot reach Unity Catalog Delta tables
  without an external location; they can, and now do. `graph.py`'s docstring
  still described `destinations=`, removed in the same pass that wrote it.
- **A duplicated sentence in `agent.py`** left by the previous comment pass.

- **Two message constants nothing read.** `REVIEW_ALLOWED_NOTE` and
  `DEFERRED_REQUEST_RESUMED` were written, reviewed and never referenced — the
  kind of dead string that reads as a shipped behaviour to whoever finds it next.
- **The `WorkerClient` protocol was missing `deadline`.** Both implementations
  take it and the dispatch stage always passes it, but the declared protocol did
  not name it — so a third worker client written against the protocol, as the
  protocol exists to permit, would have been called with an argument it did not
  accept: a `TypeError` on the dispatch path at the first worker call.

- **Quoted JSON keys evaded three detection shapes.** `secret-assignment`,
  `health-condition` and `patient-record-id` required the key and separator to
  be adjacent, so `{"DB_PASSWORD": "…"}` and `{"diagnosis": "…"}` were missed
  entirely while the same content as `KEY=value` or prose was caught.
  `patient-record-id` is `health_id`, a *withhold* tier, so this was a block
  that silently did not happen. The `person` shape had been fixed for exactly
  this and the fix was never carried across.
- **`patient-record-id` matched schema words.** It lacked the
  `_identifier_with_a_digit` guard its sibling shape carries, so "the intake
  form has a patient id: field and a record number: field" captured the literal
  word `field` as a health identifier — and withheld an ordinary healthcare
  requirement from the user. Found by the new policy suite on its first run.
- **`card-fragment` missed the natural phrasing.** "Last 4 digits: 4242,
  expiration month: 09" repeats no payment noun, because the question
  established it, so the context gate refused the one reply the shape exists to
  catch. The gate now includes `expir`/`cvv`/`cvc`/`cardholder`. The proximity
  window is deliberately *not* widened — `_has_context` searches either side of
  the match and never inside it, so a wider gap swallows the word it needs.

## [1.1.0] — 2026-09-15

The governance primitives become a shared library, and the repository is cut
down to what deploying the serving endpoint needs.

### Added

- `libs/agent_governance` — the security, guardrail, audit and runtime-bound
  code every agent on the platform shares, packaged as the `agent-governance`
  wheel (`0.1.0`) with its own tests. The supervisor is its first consumer;
  the requirement, test-case, coding and deployment agents install the same
  wheel.
- The bundle builds the wheel at deploy time (`artifacts`), the deploy job
  installs it, and `deploy/log_and_deploy.py` bakes it into the model artifact
  under `wheels/` so the serving container installs the copy that was tested.
- `deploy/publish_library.py` and the `publish_library` job task: the wheel is
  published to the platform volume `/Volumes/<catalog>/agent_platform/libs/`
  for the other agents, never overwriting a released version.
- `src/supervisor/messages.py` — every user-facing sentence, separated from the
  pipeline logic in `nodes.py`.
- `src/supervisor/config.py` — the supervisor's governed documents and their
  validators, on top of the library's generic store.
- A retention ceiling on long-term memory (`LONG_TERM_MEMORY_TTL_SECONDS`,
  ninety days). Every remembered `required_context` value carries the moment it
  was written and stops being read into later routing prompts once it is past
  the ceiling — the memory-expiry control of guardrail layer 3. An entry with
  no write stamp is treated as expired rather than as fresh, and a deployed
  environment refuses to serve with the ceiling switched off.

### Changed

- The three job scripts the deploy needs (`register_prompts.py`,
  `publish_config.py`, and the new `publish_library.py`) live in `deploy/`
  beside `log_and_deploy.py`.
- `redact.py` folded into `agent_governance.sensitive`; the tier-1 rule
  compiler and kill switch moved out of `guardrails.py` into
  `agent_governance.deny_rules`; prompt hygiene (`untrusted_turn`,
  `system_blocks`) out of `prompt_provider.py` into
  `agent_governance.prompting`; the Lakebase pool plumbing out of `memory.py`
  into `agent_governance.lakebase`.
- The Postgres schema is no longer a pre-deploy step: the runtime issues
  `CREATE SCHEMA IF NOT EXISTS` on every pool it opens, and the `publish_config`
  task connects first, as the deploying user.
- Tests are split into the supervisor's contract (`tests/`) and the library's
  own (`libs/agent_governance/tests/`), with shared helpers in importable
  modules rather than `conftest.py`.
- `locking.py` and `review_queue.py` moved into the library
  (`agent_governance.locking`, `agent_governance.review_queue`). Every
  LangGraph agent served on Model Serving needs the per-conversation lock —
  the platform runs several replicas and worker processes with no affinity by
  conversation, and LangGraph only serializes turns on its own Platform — and
  the reviewer surface needs the queue's `resolve` / `list_open` half without
  depending on the supervisor package. `thread_lock` now takes its connection
  source and a namespace explicitly; the busy message is the supervisor's, in
  `messages.py`. Both modules gained their own tests.
- `model_provider.py` builds `ChatDatabricks` directly. The `init_chat_model`
  provider switch and `ROUTING_LLM_PROVIDER` are gone: on Databricks it always
  raised and fell back, logging a warning on every start, and a different
  vendor is reached through an external-model serving endpoint anyway.

### Removed

- The fourteen operator scripts (trace, KPI report, audit-chain walk, erasure,
  inspection, teardown, verification, waiting, granting) and `.env.example`.
  DEPLOYMENT.md carries the CLI and SQL equivalents of the steps a deploy
  still needs.
- The SQL-warehouse Delta audit sink and its settings (`AUDIT_WAREHOUSE_ID`,
  `AUDIT_TABLE`, `AUDIT_ALLOW_WAREHOUSE`) and the `supervisor_delta_export`
  job. The path could never work from Model Serving — the endpoint's service
  principal cannot hold `databricks-sql-access` — and needed an explicit
  override flag to select. Postgres on Lakebase is the audit store.
- The deprecated `--mock-workers` alias of `--workers simulated`.
- Three peripheral test files (conversation scope, small-talk phrasing,
  windowing). The behaviours they covered are unchanged.
- The plain-Postgres DSN mode (`SUPERVISOR_PG_DSN` / `LAKEBASE_DSN`). Lakebase
  is the one durable backend; a workstation that needs durable state points
  `LAKEBASE_INSTANCE` at the dev instance with its own credentials.
- `LongTermMemory.forget`, `last_updated` and `subjects`, unreachable once the
  erasure script went. DEPLOYMENT.md §9 carries the erasure SQL.
- Seams nothing called: the `from_yaml` constructors, `Deadline.cap`,
  `ConfigProvider.describe`, `TurnSpend.audit_detail`,
  `ReviewQueue.open_for_conversation`, and the `_is_transient` /
  `_RouteDecision` aliases.

## [1.0.0] — 2026-09-10

First delivery: the complete supervisor agent, its asset bundle, the operator
tooling and the governance contract as tests.

### Added

- The six-stage pipeline — `rbac_gate → guardrails → route/clarify → dispatch →
  approval → respond` — served as an MLflow `ResponsesAgent` on Databricks Model
  Serving, with progress events and token streaming.
- Authorization re-validated on every turn, including clarification replies and
  approval decisions, with optional HMAC-signed entitlements.
- A two-tier guardrail screen: deterministic deny patterns, then a semantic
  domain check per candidate agent, with an appeal path and a deterministic
  small-talk classifier ahead of any model call.
- The layer-7 output screen: a sensitive-shape catalogue with checksum and
  context validation, a per-category policy (`allow` / `mask` / `block` /
  `escalate`), the same screen applied to the conversation relayed *to* a
  worker, a streaming guard with a hold-back window, canary and prompt-leak
  detection, and grounding cues on unbacked claims.
- Human-in-the-loop approval via LangGraph `interrupt()`, and a queryable
  review queue for appeals and escalations.
- A tamper-evident decision trail with a row hash chain, mirrored onto the
  request's MLflow trace.
- Durable state in Lakebase Postgres — checkpoints, long-term memory, governed
  configuration, review queue and audit — with one schema per environment, and a
  refusal to serve rather than degrade outside a workstation.
- Governed configuration published to a table: the worker registry, the role
  mapping, the guardrail rules, the output policy and the kill switch all reach
  a running endpoint within the cache TTL, with no redeploy.
- Prompts registered in Unity Catalog with a per-environment alias, and bundled
  templates as the fallback.
- Cost and time bounds: per-turn and per-subject spend ceilings, a turn deadline,
  bounded clarification, an explicit graph step ceiling, and one active turn per
  conversation.
- Session notes: *"…keep this in mind for later"* is acknowledged and held for
  the conversation rather than dispatched as work.
- A Databricks asset bundle with one target per environment, sharing nothing.
