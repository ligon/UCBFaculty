"""Tests for :mod:`ucbfaculty.queries`.

These exercise the DuckDB query layer.  They depend on the same fixture
data that drives the rest of the test suite (see ``tests/conftest.py``)
and skip cleanly when no data is available (real or fixture).
"""
from __future__ import annotations

import pytest

from ucbfaculty import queries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_queries():
    """Reset the connection cache before each test for isolation."""
    queries._reset()
    yield
    queries._reset()


def _data_available() -> bool:
    """Return True iff vitals + at least one feature can be loaded."""
    try:
        from ucbfaculty.faculty import faculty_vitals

        faculty_vitals()
    except Exception:
        return False
    try:
        from ucbfaculty.new_feature import available_features

        return bool(available_features())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Basic connection / static-view tests
# ---------------------------------------------------------------------------

def test_con_returns_duckdb_connection():
    """``con()`` returns a DuckDB connection on which we can run SQL."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    c = queries.con()
    # Smoke: DuckDB-flavored SELECT works.
    assert c.execute("SELECT 1 AS x").df()["x"].iloc[0] == 1


def test_static_views_register_on_first_con():
    """Calling ``con()`` eagerly registers vitals and committees."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    queries.con()
    registered = queries.registered_views()
    assert "vitals" in registered
    assert "committees" in registered


def test_execute_select_on_vitals_returns_dataframe():
    """A SELECT on the static ``vitals`` view returns a non-empty DataFrame."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    df = queries.execute("SELECT * FROM vitals LIMIT 5")
    assert not df.empty
    assert len(df) <= 5
    assert "Email" in df.columns


def test_execute_count_on_committees():
    """``COUNT(*)`` against ``committees`` returns a scalar."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    df = queries.execute("SELECT COUNT(*) AS n FROM committees")
    assert "n" in df.columns
    assert df["n"].iloc[0] >= 0


# ---------------------------------------------------------------------------
# Lazy feature view registration
# ---------------------------------------------------------------------------

def test_feature_view_registers_lazily_via_sql_parse():
    """Referencing a feature in SQL auto-registers its view."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    from ucbfaculty.new_feature import available_features

    feats = available_features()
    if not feats:
        pytest.skip("No features available in this environment")
    target = feats[0]

    # Pre-condition: not registered until referenced.
    queries.con()
    assert target not in queries.registered_views()

    df = queries.execute(f'SELECT * FROM "{target}" LIMIT 3')
    assert target in queries.registered_views()
    assert not df.empty


def test_feature_view_registers_via_explicit_argument():
    """The ``register=`` kwarg registers a view even without SQL reference."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    from ucbfaculty.new_feature import available_features

    feats = available_features()
    if not feats:
        pytest.skip("No features available in this environment")
    target = feats[0]

    queries.execute("SELECT 1", register=[target])
    assert target in queries.registered_views()


def test_unknown_view_in_sql_yields_duckdb_error():
    """Querying a nonexistent view propagates DuckDB's error."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    import duckdb

    with pytest.raises(duckdb.Error):
        queries.execute("SELECT * FROM definitely_not_a_view")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_available_views_includes_statics_and_features():
    """``available_views`` lists statics and every available feature."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    from ucbfaculty.new_feature import available_features

    avail = queries.available_views()
    assert "vitals" in avail
    assert "committees" in avail
    for name in available_features():
        assert name in avail


def test_registered_views_starts_empty_after_reset():
    """``_reset`` clears the registration cache."""
    queries._reset()
    assert queries.registered_views() == []


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------

def test_describe_no_arg_returns_summary():
    """``describe()`` returns the available/registered/static breakdown."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    info = queries.describe()
    assert set(info.keys()) >= {"available", "registered", "static"}
    assert "vitals" in info["available"]
    assert info["static"] == ["vitals", "committees"]


def test_describe_specific_view_returns_columns_and_rows():
    """``describe(view)`` returns column metadata and a row count."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    info = queries.describe("vitals")
    assert info["view"] == "vitals"
    assert isinstance(info["columns"], list)
    assert all(
        isinstance(c, tuple) and len(c) == 2 for c in info["columns"]
    )
    assert "Email" in [c[0] for c in info["columns"]]
    assert info["rows"] >= 0


def test_describe_unknown_view_raises_keyerror():
    """``describe`` raises ``KeyError`` on a name that's not a known view."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    with pytest.raises(KeyError):
        queries.describe("not_a_real_view_name")


# ---------------------------------------------------------------------------
# Feature-view shape: single feature view should expose Email + value
# ---------------------------------------------------------------------------

def test_feature_view_columns_include_email_and_feature_name():
    """A registered feature view has ``Email`` and a column named after the feature."""
    if not _data_available():
        pytest.skip("No fixture or RawData available")
    from ucbfaculty.new_feature import available_features

    feats = available_features()
    if not feats:
        pytest.skip("No features available in this environment")
    target = feats[0]

    info = queries.describe(target)
    col_names = [c[0] for c in info["columns"]]
    assert "Email" in col_names
    assert target in col_names
