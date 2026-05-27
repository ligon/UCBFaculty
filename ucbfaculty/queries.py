"""Read-only DuckDB query layer over ucbfaculty data sources.

This module is the canonical query entry point.  The MCP server uses it,
and other parts of the package may use it as well.  It does not own data
loading: the heavy lifting (gpg decryption, YAML transforms, spelling
normalization) lives in :mod:`ucbfaculty.faculty` and
:mod:`ucbfaculty.new_feature`.  This module only:

* maintains a singleton DuckDB connection,
* registers pandas DataFrames produced by the existing loaders as
  zero-copy views, and
* exposes a small public API for issuing SQL queries.

Public API
----------
con()
    Cached singleton :class:`duckdb.DuckDBPyConnection` with the static
    views (``vitals``, ``committees``) registered.

execute(sql, *, register=None)
    Run a query.  Feature views referenced in ``sql`` are registered
    lazily on first use.

available_views()
    Names of all views that *could* be queried (statics + every feature
    in :func:`ucbfaculty.new_feature.available_features`).

registered_views()
    Names currently registered in the cached connection.

describe(view=None)
    Schema for one view (columns, dtypes, row count) or a summary of
    available/registered views.

Notes
-----
* The connection is process-local and not thread-safe.  The MCP server
  is single-threaded over stdio, so this is fine for that use case.
* Views are zero-copy references to pandas DataFrames; the load cost is
  paid once per Feature, on first use.
* Static views (``vitals``, ``committees``) register eagerly on first
  :func:`con` call.  Feature views register lazily based on table
  references parsed from the SQL, or via the ``register`` argument.
* Re-registration (after :mod:`ucbfaculty` cache invalidation) is not
  yet supported; restart the process to pick up new data.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Module-level connection cache
# ---------------------------------------------------------------------------
_CON = None
_REGISTERED: set[str] = set()
_STATIC_VIEWS: tuple[str, ...] = ("vitals", "committees")


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def _connect():
    """Open (or reuse) the singleton in-memory DuckDB connection."""
    global _CON
    if _CON is None:
        import duckdb

        _CON = duckdb.connect(":memory:")
    return _CON


# ---------------------------------------------------------------------------
# View loaders
# ---------------------------------------------------------------------------

def _load_vitals() -> pd.DataFrame:
    """Return the faculty-vitals DataFrame ready for registration.

    The canonical :func:`ucbfaculty.faculty.faculty_vitals` returns a
    DataFrame indexed by Email; we move the index to a column so SQL
    sees a flat table.
    """
    from ucbfaculty.faculty import faculty_vitals

    return faculty_vitals().reset_index()


def _load_committees() -> pd.DataFrame:
    """Return the full committee-membership DataFrame.

    :func:`ucbfaculty.faculty.committee_membership` already returns a
    flat DataFrame with no MultiIndex.
    """
    from ucbfaculty.faculty import committee_membership

    return committee_membership()


def _load_feature(name: str) -> pd.DataFrame:
    """Return a feature's latest-year values as a flat ``(Email, <name>)`` table.

    The :class:`ucbfaculty.new_feature.Feature` returns a Series indexed
    by Email (or another canonical key); we ``reset_index()`` and rename
    the value column to the feature name so SQL sees self-describing
    column headers (``SELECT Email, Salary FROM Salary`` works).
    """
    from ucbfaculty.new_feature import Feature

    series = Feature(name)["latest"]
    df = series.reset_index()
    if df.shape[1] >= 2:
        # Rename only the value column; preserve any index columns.
        new_cols = list(df.columns)
        new_cols[-1] = name
        df.columns = new_cols
    return df


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _register(con, name: str, df: pd.DataFrame) -> None:
    """Register a DataFrame as a view and record the name."""
    con.register(name, df)
    _REGISTERED.add(name)


def _ensure_static_views(con) -> None:
    """Register vitals and committees if not already registered."""
    if "vitals" not in _REGISTERED:
        _register(con, "vitals", _load_vitals())
    if "committees" not in _REGISTERED:
        _register(con, "committees", _load_committees())


def _ensure_feature_view(con, name: str) -> bool:
    """Register a feature view by name.

    Returns
    -------
    bool
        ``True`` if the view was registered (or was already registered),
        ``False`` if ``name`` is not a known feature.
    """
    if name in _REGISTERED:
        return True
    if name in _STATIC_VIEWS:
        _ensure_static_views(con)
        return True

    from ucbfaculty.new_feature import available_features

    if name not in set(available_features()):
        return False

    _register(con, name, _load_feature(name))
    return True


# ---------------------------------------------------------------------------
# SQL parsing for lazy view registration
# ---------------------------------------------------------------------------

# Match identifiers that appear after FROM or JOIN.  Handles both bare
# identifiers (``FROM Salary``) and double-quoted identifiers
# (``FROM "Salary"``); DuckDB accepts both forms, and feature names with
# initial caps tend to look quoted in agent-generated SQL.  Conservative:
# misses subqueries and CTE references, but those don't need extra
# registration beyond what the outer FROM/JOIN already pulled in.
_TABLE_REF_RE = re.compile(
    r'(?:FROM|JOIN)\s+"?([a-zA-Z_]\w*)"?',
    re.IGNORECASE,
)


def _referenced_tables(sql: str) -> set[str]:
    """Return identifiers that appear after FROM or JOIN in ``sql``."""
    return set(_TABLE_REF_RE.findall(sql))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def con():
    """Return the cached connection with static views registered."""
    c = _connect()
    _ensure_static_views(c)
    return c


def execute(
    sql: str,
    *,
    register: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Run a SQL query and return the result as a pandas DataFrame.

    Parameters
    ----------
    sql : str
        DuckDB-compatible SQL.  Available tables are ``vitals``,
        ``committees``, and one view per available feature (registered
        lazily on first reference).
    register : list[str] | None
        Force-register these views before running the query.  Useful
        when SQL parsing would miss the reference (e.g. dynamic table
        construction) or when a query must succeed despite an unknown
        feature name (which is silently skipped).

    Returns
    -------
    pandas.DataFrame
        Query result.

    Raises
    ------
    duckdb.Error
        On SQL parse or execution errors.
    """
    c = con()

    explicit = list(register or [])
    referenced = _referenced_tables(sql)
    needed = (set(explicit) | referenced) - _REGISTERED

    for name in needed:
        # _ensure_feature_view handles statics and unknowns gracefully.
        _ensure_feature_view(c, name)

    return c.execute(sql).df()


def available_views() -> list[str]:
    """Return all view names that *could* be queried.

    Includes the static views plus every feature in
    :func:`ucbfaculty.new_feature.available_features`.
    """
    from ucbfaculty.new_feature import available_features

    feats = sorted(set(available_features()))
    return list(_STATIC_VIEWS) + feats


def registered_views() -> list[str]:
    """Return view names currently registered in the cached connection."""
    return sorted(_REGISTERED)


def describe(view: Optional[str] = None) -> dict:
    """Return schema info for one view, or a summary across all views.

    Parameters
    ----------
    view : str | None
        When given, return a detailed schema for that single view
        (registering it if necessary).  When ``None``, return a
        cross-view summary listing what's available and what's already
        loaded.

    Returns
    -------
    dict
        For a specific view::

            {"view": <name>, "columns": [(col, dtype), ...], "rows": <int>}

        For ``view=None``::

            {"available": [...], "registered": [...], "static": [...]}

    Raises
    ------
    KeyError
        If ``view`` is given and is neither a static view nor a known
        feature.
    """
    c = con()
    if view is None:
        return {
            "available": available_views(),
            "registered": registered_views(),
            "static": list(_STATIC_VIEWS),
        }

    if not _ensure_feature_view(c, view):
        raise KeyError(f"Unknown view: {view!r}")

    info = c.execute(f'DESCRIBE "{view}"').df()
    rows = c.execute(f'SELECT COUNT(*) FROM "{view}"').fetchone()[0]
    cols = [
        (str(row["column_name"]), str(row["column_type"]))
        for _, row in info.iterrows()
    ]
    return {"view": view, "columns": cols, "rows": int(rows)}


# ---------------------------------------------------------------------------
# Test-only helper
# ---------------------------------------------------------------------------

def _reset() -> None:
    """Reset the connection cache.  Test-only; not part of the public API."""
    global _CON, _REGISTERED
    if _CON is not None:
        try:
            _CON.close()
        except Exception:
            pass
    _CON = None
    _REGISTERED = set()
