"""Minimal `.env` reader shared by the operator scripts.

The Databricks *CLI* finds credentials on its own, so `databricks bundle deploy`
works in a bare shell. The scripts here use the *SDK*, which does not read
`.env` — without this they fail with "cannot configure default credentials" even
where the CLI beside them authenticates fine.

Deliberately not `python-dotenv`: one function, no parsing surface beyond
`KEY=value`, and no dependency added to a deployment path for it.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> list[str]:
    """Load `KEY=value` lines into the environment; return the keys set.

    Existing environment variables win, so an explicitly exported value is never
    silently overridden by the file. Blank lines, comments and lines without an
    `=` are skipped; surrounding quotes are stripped from the value.
    """
    loaded: list[str] = []
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
