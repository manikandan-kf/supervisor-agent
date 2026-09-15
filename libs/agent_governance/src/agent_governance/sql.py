"""The two SQL helpers every Postgres-backed governance store relies on.

SQL cannot parameterize an *identifier*, so a table name has to reach a
statement as text — the SQL-injection shape whether or not the value is
reachable by an attacker. The identifier is therefore validated once, on the
way in, and every interpolation downstream rests on that enforced property.
"""

from __future__ import annotations

import re

# Letters, digits and underscores, up to three dot-separated parts. No quotes,
# whitespace, semicolons or comment markers — nothing that can end the
# identifier and start a new clause.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}$")


class UnsafeIdentifier(ValueError):
    """A table or column name that must not be interpolated into SQL."""


def safe_identifier(name: str, *, what: str = "table") -> str:
    """Return `name` if it is a plain SQL identifier, else raise."""
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise UnsafeIdentifier(f"unsafe {what} identifier: {name!r}")
    return name


# The schema an unqualified CREATE would write to — which is not the schema an
# unqualified SELECT would read from. See `table_exists_here`.
_TABLE_PRESENT_SQL = (
    "SELECT to_regclass(quote_ident(current_schema()) || '.' || quote_ident(%s)) "
    "IS NOT NULL AS present"
)


def table_exists_here(cur, table: str) -> bool:
    """Does `table` exist in the schema this connection would *create* it in?

    Not `to_regclass('<table>')`: Postgres resolves an unqualified *reference*
    to the first schema on `search_path` that holds the name, but an unqualified
    `CREATE TABLE` always targets the first schema on the path. With
    `search_path = supervisor_dev, public` and a same-named table left in
    `public`, the unqualified probe answers "already there", the DDL is skipped,
    and every later read and write resolves to the other environment's table.
    `current_schema()` is precisely where the CREATE would land.
    """
    cur.execute(_TABLE_PRESENT_SQL, (table,))
    row = cur.fetchone()
    return bool(row["present"] if isinstance(row, dict) else row[0])
