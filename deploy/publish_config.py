"""Publish governance configuration to the Unity Catalog table.

This is how a guardrail rule, a role mapping or a worker registry entry changes
now — a publish, not a redeploy. R1 puts the supervisor's configuration
"directly in Unity Catalog tables (pre-wrapper-API)": the table is the source of
record, `supervisor.config` reads it at runtime, and there is
deliberately no validating write service in front of it. This script is the
write path.

The table is Lakebase Postgres, registered in Unity Catalog as the read-only
catalog `supervisor_memory` — so the same rows are
`supervisor_memory.public.supervisor_config` in the SQL editor and an ordinary
INSERT here. `config_store` explains why that is the governed table this project
can actually use rather than a plain UC Delta one.

    # what is live, and every version behind it
    python deploy/publish_config.py --list

    # would this publish change anything?  (default: nothing is written)
    python deploy/publish_config.py --diff

    # publish the bundled YAML as new versions and make them active
    python deploy/publish_config.py --apply
    python deploy/publish_config.py --apply --only guardrails -m "tighten injection rule"

    # roll one document back to an earlier version
    python deploy/publish_config.py --pin guardrails=3

Nothing is written without `--apply` or `--pin`. A document whose payload is
byte-identical to the active version is skipped rather than republished, so
re-running this in a deploy job does not manufacture a version per deploy.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Repository root, however this file is being run.

    A Databricks `spark_python_task` does not import the script — it reads the
    source and `exec`s it inside a notebook kernel, so `__file__` is never
    bound and `Path(__file__)` raises NameError. The code object compiled from
    that source still carries the real path, so the frame is the reliable
    fallback.

    Without it the task fails on its first real run with `NameError: name
    '__file__' is not defined`, taking every task that depends on it down as an
    upstream failure. Same helper as the other deploy scripts; keep them identical.
    """
    try:
        here = Path(__file__)
    except NameError:
        import inspect

        here = Path(inspect.currentframe().f_code.co_filename)
    return here.resolve().parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs" / "agent_governance" / "src"))

from agent_governance.config_store import (  # noqa: E402
    ConfigError,
    ConfigStore,
    checksum_of,
    load_bundled,
)

from supervisor.config import CONFIG_NAMES, config_store, validate  # noqa: E402
from supervisor.settings import Settings  # noqa: E402


def _store(settings: Settings) -> ConfigStore:
    try:
        return config_store(settings)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


def _actor() -> str:
    for name in ("DATABRICKS_CLIENT_ID", "USER", "USERNAME"):
        value = os.getenv(name)
        if value:
            return value
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _selected(only: str | None) -> tuple[str, ...]:
    if not only:
        return CONFIG_NAMES
    names = tuple(part.strip() for part in only.split(",") if part.strip())
    unknown = [n for n in names if n not in CONFIG_NAMES]
    if unknown:
        raise SystemExit(f"unknown document(s): {', '.join(unknown)}; known: {', '.join(CONFIG_NAMES)}")
    return names


def cmd_list(store: ConfigStore) -> int:
    for name in CONFIG_NAMES:
        rows = store.history(name)
        if not rows:
            print(f"{name}: (nothing published — the endpoint is running the bundled file)")
            continue
        print(f"{name}:")
        for row in rows:
            marker = "*" if row["active"] else " "
            print(
                f"  {marker} v{row['version']:<3} {str(row['created_at'])[:19]}  "
                f"{row['checksum'][:12]}  {row['created_by'] or '-'}  {row['comment'] or ''}"
            )
    print("\n* = active. The endpoint picks a change up within CONFIG_CACHE_TTL_SECONDS (60s).")
    return 0


def cmd_diff(store: ConfigStore, settings: Settings, names: tuple[str, ...]) -> int:
    changed = 0
    for name in names:
        bundled = load_bundled(settings.config_dir, name)
        try:
            validate(name, bundled)
        except ConfigError as exc:
            print(f"{name}: INVALID in the bundled file — {exc}")
            changed += 1
            continue

        try:
            live = store.read(name)
        except ConfigError as exc:
            print(f"{name}: the active row is unusable — {exc}")
            print("       publishing would replace it")
            changed += 1
            continue

        if live is None:
            print(f"{name}: nothing published yet — would create v1")
            changed += 1
        elif live.checksum == checksum_of(bundled):
            print(f"{name}: unchanged (active v{live.version})")
        else:
            print(f"{name}: differs from active v{live.version} — would publish v{live.version + 1}")
            changed += 1
    print(f"\n{changed} document(s) would change. Re-run with --apply to publish.")
    return 0


def cmd_apply(store: ConfigStore, settings: Settings, names: tuple[str, ...], message: str) -> int:
    actor = _actor()
    published = 0
    for name in names:
        bundled = load_bundled(settings.config_dir, name)
        try:
            validate(name, bundled)
        except ConfigError as exc:
            print(f"{name}: REFUSED — {exc}")
            return 1

        try:
            live = store.read(name)
        except ConfigError:
            live = None  # unusable active row; republishing is the fix

        if live is not None and live.checksum == checksum_of(bundled):
            print(f"{name}: unchanged (active v{live.version}) — skipped")
            continue

        version = store.publish(name, bundled, actor=actor, comment=message)
        print(f"{name}: published v{version} ({checksum_of(bundled)[:12]}) and made active")
        published += 1

    if published:
        print(
            f"\n{published} document(s) published. A running endpoint picks them up within "
            f"{int(settings.config_cache_seconds)}s — no redeploy."
        )
    return 0


def cmd_pin(store: ConfigStore, spec: str) -> int:
    if "=" not in spec:
        raise SystemExit("--pin takes name=version, e.g. --pin guardrails=3")
    name, _, raw = spec.partition("=")
    name = name.strip()
    if name not in CONFIG_NAMES:
        raise SystemExit(f"unknown document {name!r}; known: {', '.join(CONFIG_NAMES)}")
    try:
        version = int(raw)
    except ValueError:
        raise SystemExit(f"not a version number: {raw!r}") from None

    store.activate(name, version)
    print(f"{name}: v{version} is now active")
    return 0


def cmd_show(store: ConfigStore, name: str) -> int:
    document = store.read(name)
    if document is None:
        print(f"{name}: nothing published")
        return 0
    print(f"# {name} v{document.version}  sha256:{document.checksum}")
    print(json.dumps(document.payload, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show every published version")
    parser.add_argument("--diff", action="store_true", help="what --apply would change (default)")
    parser.add_argument("--apply", action="store_true", help="publish the bundled YAML")
    parser.add_argument("--pin", metavar="NAME=VERSION", help="make an existing version active again")
    parser.add_argument("--show", metavar="NAME", help="print one document as it is stored")
    parser.add_argument("--only", metavar="NAMES", help="comma-separated subset of documents")
    parser.add_argument("-m", "--message", default="", help="comment recorded with the version")
    parser.add_argument(
        "--lakebase-instance",
        default="",
        help=(
            "Lakebase instance holding the table. Needed when running as a bundle job task, "
            "which has no .env to read; locally it comes from the environment."
        ),
    )
    parser.add_argument(
        "--lakebase-schema",
        default="",
        help=(
            "Postgres schema holding the table. Each deployed environment owns one "
            "(supervisor_dev, supervisor_prod), so publishing to dev cannot change what "
            "prod serves. Omit for the single-environment shape, which uses `public`."
        ),
    )
    args = parser.parse_args()

    if args.lakebase_instance:
        # Set before anything resolves a connection — `audit_connection_source()`
        # reads the environment at call time.
        os.environ["LAKEBASE_INSTANCE"] = args.lakebase_instance
    if args.lakebase_schema:
        # Same timing rule, and the reason it is a flag rather than inherited
        # from the environment: a publish that silently went to `public` would
        # write a version nothing reads while reporting success.
        os.environ["LAKEBASE_SCHEMA"] = args.lakebase_schema
    settings = Settings()
    store = _store(settings)
    # Which environment is about to change. With one schema per environment the
    # only difference between publishing to dev and publishing to prod is this
    # line, so print it rather than leaving it to be inferred from the flags.
    print(
        f"target: {os.getenv('LAKEBASE_SCHEMA') or 'public'}.{settings.config_table} "
        f"on {os.getenv('LAKEBASE_INSTANCE') or '(no LAKEBASE_INSTANCE set)'}\n"
    )

    if args.list:
        return cmd_list(store)
    if args.show:
        return cmd_show(store, args.show)
    if args.pin:
        return cmd_pin(store, args.pin)

    names = _selected(args.only)
    if args.apply:
        return cmd_apply(store, settings, names, args.message)
    return cmd_diff(store, settings, names)


if __name__ == "__main__":
    # `raise SystemExit(main())` raises SystemExit even for a zero return, and a
    # Databricks `spark_python_task` surfaces *any* exception escaping the
    # exec'd source as a task failure — a clean exit included, which reports a
    # task that did its work correctly as FAILED and skips everything
    # downstream. So only exit explicitly when there is a real failure to
    # report; success falls off the end, which is exit code 0 for a normal CLI
    # run and a clean finish for the job. Same epilogue as
    # `register_prompts.py`.
    _exit_code = main()
    if _exit_code:
        sys.exit(_exit_code)
