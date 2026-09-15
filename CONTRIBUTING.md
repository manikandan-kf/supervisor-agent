# Contributing

## Setting up

```
python -m venv .venv
.venv/bin/python -m pip install -e libs/agent_governance -e ".[dev]"   # Windows: .venv\Scripts\python.exe
```

Installs the shared library (editable), the supervisor package and the
tooling. `pip install -r requirements.txt -r requirements-dev.txt` gives the
same environment.

## The three checks

```
ruff check .          # lint and the security ruleset (flake8-bandit port)
pytest                # both test trees — offline, no workspace needed
python -m pip wheel --no-deps --wheel-dir libs/agent_governance/dist libs/agent_governance
```

CI runs all three on every push and every pull request, and uploads the
results as a retained artifact. A red pipeline is a blocked merge.

The test suites need no workspace, no network and no database. If a test you
add needs any of those, fake it at the client boundary and say so in the test's
docstring.

## Where code goes

Two packages, one rule: **if a second agent would need it, it belongs in
`libs/agent_governance`.** A sensitive-data shape, an output policy rule, a
sanitiser, an audit column, a retry or budget primitive — library. Anything
that knows about the supervisor's graph, its prompts, its registry or its
settings — `src/supervisor`.

The library must not import from the supervisor. It reads platform-wide
environment variables (`LAKEBASE_INSTANCE`, `ENVIRONMENT`, …) but never
agent-specific defaults; those are passed in (the Lakebase schema, the config
validators, the bundled seed directory).

A library change is a release: bump `agent_governance.__version__`, because
`deploy/publish_library.py` will not overwrite a version another agent may have
pinned.

## What a change is expected to include

- **A test that fails without it.** Add to the file that owns the stage or the
  primitive rather than starting a new one.
- **The reason, in the code.** This codebase carries its reasoning in comments
  and docstrings on purpose — a control whose reason is only in a ticket is a
  control the next person deletes. `E501` is switched off so prose does not
  have to be reflowed.
- **No change to what a deployment can weaken.** New settings that govern a
  control belong on a code default with an `enforce` check, not in the deploy
  job's environment.

## Changing a guardrail rule or a prompt

Neither needs a code change to reach a running endpoint, and both are governed:

- **Guardrail rules, the output policy and canary tokens** live in
  `src/supervisor/config/guardrails.yaml`, which is the *seed*. The live copy is
  published to a table with `python deploy/publish_config.py --apply` and
  reaches the endpoint within the configuration cache TTL. Publishing validates
  the document and refuses a change that would downgrade a control.
- **Prompts** are registered in Unity Catalog by
  `python deploy/register_prompts.py`, one alias per environment. The bundled
  templates in `prompt_provider.py` are the fallback the endpoint uses when the
  registry is unreachable, so a change to a template needs a redeploy as well as
  a registration.

## Dependencies

`requirements.txt` is what the serving container installs, and `pyproject.toml`
mirrors it for local work and CI. **Change both together, and pin exactly.**
The library's own `pyproject.toml` declares floors, not pins: it is a library,
and each installing agent owns its pins. To move a pin: raise it in both files,
run the suite, deploy to `dev`, and check the endpoint's build log for a
resolution conflict before prod.

## Style

`ruff` is the whole style guide — formatting, import order and the security
rules. If a rule is wrong for a specific file, add a scoped
`per-file-ignores` entry in `pyproject.toml` with the reason, rather than a bare
`# noqa`. No `src/` or `libs/**/src/` file is exempt from `S101`, `S105` or
`S106`.

## Reporting a security issue

Do not open a public issue. See `SECURITY.md`.
