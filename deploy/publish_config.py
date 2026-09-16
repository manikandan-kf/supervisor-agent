"""Publish governance configuration to the Unity Catalog table.

A guardrail rule, role mapping or worker registry entry changes by publish, not redeploy: R1
puts configuration "directly in Unity Catalog tables (pre-wrapper-API)", so the table is the
source of record, `supervisor.config` reads it at runtime and this script is the write path, with
deliberately no validating service in front. The table is Lakebase Postgres surfaced as the UC
catalog `supervisor_memory` (`config_store` explains why). Nothing is written without `--apply` or
`--pin`; a payload identical to the active version is skipped, so a deploy job mints no versions.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Repository root, however this file is run.

    A `spark_python_task` `exec`s the source in a notebook kernel, so `__file__` is unbound;
    the compiled code object still carries the real path. Same helper as the other deploy scripts.
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


def _policy_report(candidate: dict, config_dir: Path):
    """The policy suite's verdict on a candidate guardrails document, or None.

    None means the check could not run; the caller treats that as a refusal, because "could not
    check" and "checked and clean" must never be the same answer on the path that reaches
    production without a redeploy. The suite sits beside the document so both ship in one diff.
    """
    try:
        from agent_governance.policy_eval import evaluate, load_suite

        return evaluate(candidate, load_suite(config_dir / "policy_suite.yaml"))
    except Exception as exc:  # noqa: BLE001 — reported, then refused by the caller
        print(f"policy suite error: {exc}")
        return None


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
        raise SystemExit(
            f"unknown document(s): {', '.join(unknown)}; known: {', '.join(CONFIG_NAMES)}"
        )
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
            print(
                f"{name}: differs from active v{live.version} — would publish v{live.version + 1}"
            )
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

        # The guardrails document reaches a live endpoint without a redeploy, so this is the only
        # automated check between an edited regex and production traffic. Run on the *candidate*
        # and refuse the publish rather than report afterwards; offline, so it cannot flake.
        if name == "guardrails":
            report = _policy_report(bundled, settings.config_dir)
            if report is None:
                print(f"{name}: REFUSED — the policy suite could not be run")
                return 1
            if not report.passed:
                print(f"{name}: REFUSED — {report.summary()}")
                for line in report.report_lines():
                    print(line)
                print(
                    "\nNothing was published. Fix the rule, or update "
                    "src/supervisor/config/policy_suite.yaml if the new behaviour is "
                    "intended — a deliberate policy change should edit its expectation."
                )
                return 1
            print(f"{name}: {report.summary()}")

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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="show every published version")
    parser.add_argument("--diff", action="store_true", help="what --apply would change (default)")
    parser.add_argument("--apply", action="store_true", help="publish the bundled YAML")
    parser.add_argument(
        "--pin", metavar="NAME=VERSION", help="make an existing version active again"
    )
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
        # Same timing rule; a flag rather than inherited because a publish that silently went to
        # `public` would write a version nothing reads while reporting success.
        os.environ["LAKEBASE_SCHEMA"] = args.lakebase_schema
    settings = Settings()
    store = _store(settings)
    # With one schema per environment this line is the only difference between publishing to dev
    # and to prod, so print it rather than leave it to be inferred from the flags.
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
    # A spark_python_task fails on any escaping exception, SystemExit(0) too: exit only on failure.
    _exit_code = main()
    if _exit_code:
        sys.exit(_exit_code)
