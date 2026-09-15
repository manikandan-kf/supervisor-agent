# Changelog

Notable changes to the supervisor agent. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version here identifies the **source**. A running deployment is identified
by its Unity Catalog registered-model version, assigned at deploy time.

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
