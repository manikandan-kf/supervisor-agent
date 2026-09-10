# Contributing

## Setting up

```
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"     # Windows: .venv\Scripts\python.exe
```

`pip install -e ".[dev]"` installs the package and the tooling. If you would
rather not install the package, `pip install -r requirements.txt -r
requirements-dev.txt` gives the same environment — `tests/conftest.py` puts
`src/` on the path itself, so `pytest` works either way.

## The three checks

```
ruff check .          # lint and the security ruleset (flake8-bandit port)
pytest                # the governance contract — offline, no workspace needed
python -m compileall src scripts deploy
```

CI runs the first two on every push and every pull request, and uploads the
results as a retained artifact. A red pipeline is a blocked merge.

The test suite needs no workspace, no network and no database. If a test you add
needs any of those, fake it at the client boundary and say so in the test's
docstring.

## What a change is expected to include

- **A test that fails without it.** The suite is one file per stage of the
  pipeline plus the cross-cutting security properties; add to the file that owns
  the stage rather than starting a new one.
- **The reason, in the code.** This codebase carries its reasoning in comments
  and docstrings on purpose — a control whose reason is only in a ticket is a
  control the next person deletes. `E501` is switched off so prose does not have
  to be reflowed.
- **No change to what a deployment can weaken.** New settings that govern a
  control belong on a code default with an `enforce` check, not in the deploy
  job's environment.

## Changing a guardrail rule or a prompt

Neither needs a code change to reach a running endpoint, and both are governed:

- **Guardrail rules, the output policy and canary tokens** live in
  `src/supervisor/config/guardrails.yaml`, which is the *seed*. The live copy is
  published to a table with `python scripts/publish_config.py --apply` and reaches
  the endpoint within the configuration cache TTL. Publishing validates the
  document and refuses a change that would downgrade a control.
- **Prompts** are registered in Unity Catalog by
  `python scripts/register_prompts.py`, one alias per environment. The bundled
  templates in `prompt_provider.py` are the fallback the endpoint uses when the
  registry is unreachable, so a change to a template needs a redeploy as well as
  a registration.

Both paths are audited. Neither can be used to relax a control the code enforces.

## Dependencies

`requirements.txt` is what MLflow bakes into the served model, and
`pyproject.toml` mirrors it for local work and CI. **Change both together, and
pin exactly.** To move a pin: raise it in both files, run the suite, deploy to
`dev`, and check the endpoint's build log for a resolution conflict before prod.

## Style

`ruff` is the whole style guide — formatting, import order and the security
rules. If a rule is wrong for a specific file, add a scoped
`per-file-ignores` entry in `pyproject.toml` with the reason, rather than a bare
`# noqa`. No `src/` file is exempt from `S101`, `S105` or `S106`.

## Reporting a security issue

Do not open a public issue. See `SECURITY.md`.
