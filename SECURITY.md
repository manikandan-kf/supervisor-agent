# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities privately, **not** as a public issue or pull
request. Use GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability), or the security contact your organisation
publishes for this project.

Please include: what you observed, the request or input that produced it, which
environment (`dev` or `prod`), and the endpoint or commit if you have it. If a
report involves real data, describe the shape rather than pasting the values.

## What is in scope

This repository holds the **supervisor agent** — the governance layer between a
caller and the worker agents — and the **shared governance library**
(`libs/agent_governance`) that every agent on the platform installs. A finding
in the library affects every consumer, so it is the higher-priority report. In
scope:

- the guardrail layers: input screening, prompt handling, memory writes,
  retrieval and worker relay, runtime bounds, and the output screen
- authorization: the RBAC gate, the endpoint ACL it relies on, and the
  invocation surface
- the decision trail: the audit sink, its hash chain, and what reaches it
- the governed configuration path: what a published document can change, and
  what it cannot
- the library's packaging path: what ends up in the wheel and in the model
  artifact
- dependency and supply-chain issues in the pinned set in `requirements.txt`

Out of scope, because they are separate deployables owned elsewhere: the calling
application (its UI, authentication and rate limiting) and the worker agents'
own behaviour within their remit.

## Supported versions

The `main` branch and the currently deployed Unity Catalog registered-model
version. Fixes are delivered as a new model version through the normal deploy
path in `DEPLOYMENT.md`; a library fix is also a new wheel version on the
platform volume. There is no long-term support branch.

## Security-relevant design decisions

Two are worth stating up front, because they look like bugs and are not:

- **Controls fail closed.** A deployed environment refuses to serve rather than
  degrade — an unreachable durable store, a missing audit sink or a disabled
  guardrail setting stops the turn instead of completing it unaudited. Only
  `ENVIRONMENT=local` relaxes anything.
- **Governance knobs are not settable from the deploy shell.** A control that a
  deploy environment can weaken by exporting a variable is not a control, so the
  guardrail settings stay on their code defaults and `Settings.enforce` refuses
  to serve a deployed environment where one of them is off. The one exception is
  documented in `deploy/deploy_agent.py`, and its only non-default value
  *strengthens* the screen.

## Handling of sensitive data in a report

The output screen masks or withholds credentials, government identifiers,
payment and bank data, health identifiers, names, contact details, dates of
birth and internal network detail. If you find a shape it misses, report the
*shape* (`AKIA` followed by 16 uppercase alphanumerics) rather than a live
value.
