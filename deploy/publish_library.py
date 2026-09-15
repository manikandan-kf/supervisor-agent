"""Publish the shared governance wheel to the platform's Unity Catalog volume.

`databricks bundle deploy` builds `libs/agent_governance/dist/*.whl` (the
`artifacts` block in databricks.yml) and installs it into this job's
environment. The supervisor itself never reads the volume — its deploy bakes
the same wheel into its model artifact (see log_and_deploy.py) — but the worker
agents (requirement, test-case, coding, deployment) install their copy from
here, so every agent on the platform runs one tested library:

    /Volumes/<catalog>/<schema>/<volume>/agent_governance-<version>-py3-none-any.whl

Runs as the DAB job task `publish_library`, or by hand:

    python deploy/publish_library.py --catalog workspace --schema agent_platform --volume libs

A version already present is left alone unless `--overwrite` is passed: a
published wheel is something another agent may have pinned, and the same
version name with different bytes is how two teams end up debugging two
different libraries under one name. Bump `agent_governance.__version__` instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Repository root, however this file is being run.

    A Databricks `spark_python_task` `exec`s the source, so `__file__` is not
    bound; the compiled code object still carries the real path.
    """
    try:
        here = Path(__file__)
    except NameError:
        import inspect

        here = Path(inspect.currentframe().f_code.co_filename)
    return here.resolve().parents[1]


ROOT = _repo_root()
DIST = ROOT / "libs" / "agent_governance" / "dist"


def find_wheel(dist: Path = DIST) -> Path:
    """The newest agent_governance wheel under `dist`."""
    wheels = sorted(dist.glob("agent_governance-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SystemExit(
            f"no agent_governance wheel under {dist}. `databricks bundle deploy` builds it; "
            f"locally: python -m pip wheel --no-deps --wheel-dir {dist} libs/agent_governance"
        )
    return wheels[-1]


def ensure_volume(w, catalog: str, schema: str, volume: str) -> None:
    """Create the schema and the managed volume if they do not exist yet."""
    from databricks.sdk.errors import AlreadyExists, NotFound
    from databricks.sdk.service.catalog import VolumeType

    try:
        w.schemas.get(f"{catalog}.{schema}")
    except NotFound:
        print(f"creating schema {catalog}.{schema}")
        try:
            w.schemas.create(name=schema, catalog_name=catalog)
        except AlreadyExists:
            pass
    try:
        w.volumes.read(f"{catalog}.{schema}.{volume}")
    except NotFound:
        print(f"creating volume {catalog}.{schema}.{volume}")
        try:
            w.volumes.create(
                catalog_name=catalog, schema_name=schema, name=volume, volume_type=VolumeType.MANAGED
            )
        except AlreadyExists:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", default=os.getenv("PLATFORM_CATALOG", "workspace"))
    parser.add_argument(
        "--schema",
        default=os.getenv("PLATFORM_SCHEMA", "agent_platform"),
        help="the platform-wide schema shared by every agent (not an environment schema)",
    )
    parser.add_argument("--volume", default=os.getenv("LIBRARY_VOLUME", "libs"))
    parser.add_argument("--wheel", default="", help="wheel file to publish; default: newest in dist/")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing file of the same name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wheel = Path(args.wheel) if args.wheel else find_wheel()
    destination = f"/Volumes/{args.catalog}/{args.schema}/{args.volume}/{wheel.name}"
    print(f"wheel      : {wheel}")
    print(f"destination: {destination}")
    if args.dry_run:
        return 0

    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound

    w = WorkspaceClient()
    ensure_volume(w, args.catalog, args.schema, args.volume)

    try:
        w.files.get_metadata(destination)
        exists = True
    except NotFound:
        exists = False

    if exists and not args.overwrite:
        print("already published — left as is (pass --overwrite to replace, or bump the version)")
    else:
        with wheel.open("rb") as handle:
            w.files.upload(destination, handle, overwrite=True)
        print("published" if not exists else "replaced")

    print(
        "\nConsuming agents add this to the wheels they bake into their model artifact, or\n"
        f"install it directly:  pip install {destination}"
    )
    return 0


if __name__ == "__main__":
    # A `spark_python_task` surfaces *any* exception escaping the exec'd source
    # as a task failure, a zero SystemExit included — so exit explicitly only on
    # a real failure. Same epilogue as the other deploy scripts.
    _exit_code = main()
    if _exit_code:
        sys.exit(_exit_code)
