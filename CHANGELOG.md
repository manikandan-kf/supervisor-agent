# Changelog

Notable changes to the supervisor agent. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version here identifies the **source**. A running deployment is identified
by its Unity Catalog registered-model version, assigned at deploy time; the two
are related by the deploy that produced them, not by being the same number.

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
- Seventeen operator scripts: deploy, verify, trace a turn, walk the audit
  chain, erase a subject, inspect state, report KPIs.
- The governance contract as 219 offline tests, plus lint and SAST in CI with a
  retained evidence artifact, a CycloneDX SBOM, a dependency-vulnerability scan
  and a secret scan.
