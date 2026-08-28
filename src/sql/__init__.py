"""SQL analyses over the impressions table.

Every file in this directory is a standalone DuckDB query. They are stored as
SQL rather than assembled in Python because the queries are the artifact. A
reviewer can read them, run them against their own Parquet, and check the
arithmetic without reading any of the surrounding code.

The queries carry `{}` placeholders for column lists and thresholds. Those are
schema and configuration, not data. Nothing user supplied is ever formatted into
a query string.

Each query assumes a view named `impressions` with the Criteo schema, one
`label` column, the dense columns I1 to I13 with missing values as SQL NULL, the
sparse columns C1 to C26 as strings, and a zero based `row_id` that carries the
original file order.
"""

from __future__ import annotations

import os
from typing import Any

QUERY_DIR = os.path.dirname(os.path.abspath(__file__))


def load_query(name: str, **params: Any) -> str:
    """Read a .sql file from this directory and substitute its placeholders."""
    path = os.path.join(QUERY_DIR, f"{name}.sql")
    with open(path, "r", encoding="utf-8") as handle:
        template = handle.read()
    if not params:
        return template
    return template.format(**params)
