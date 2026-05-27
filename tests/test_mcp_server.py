"""Unit tests for the MCP server module.

Tests cover serialization helpers, validation logic, CLI argument parsing,
and individual tool functions via mocking.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from ucbfaculty.mcp_server import (
    _serialize_dataframe,
    _serialize_list,
    _serialize_dict,
    _resolve_format,
    _handle_error,
    _build_parser,
    _DEFAULT_FORMAT,
    _TABLE_ROW_LIMIT,
    _JSON_ROW_LIMIT,
)


# ===================================================================
# _resolve_format
# ===================================================================

class TestResolveFormat:
    def test_explicit_org(self):
        assert _resolve_format("org") == "org"

    def test_explicit_markdown(self):
        assert _resolve_format("markdown") == "markdown"

    def test_explicit_json(self):
        assert _resolve_format("json") == "json"

    def test_case_insensitive(self):
        assert _resolve_format("JSON") == "json"
        assert _resolve_format("Org") == "org"
        assert _resolve_format("Markdown") == "markdown"

    def test_none_returns_default(self):
        assert _resolve_format(None) == "json"

    def test_invalid_returns_default(self):
        assert _resolve_format("csv") == "json"
        assert _resolve_format("") == "json"


# ===================================================================
# _serialize_dataframe
# ===================================================================

class TestSerializeDataframe:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "Email": ["a@b.edu", "c@d.edu"],
            "Name": ["Alice", "Bob"],
        })

    @pytest.fixture
    def sample_series(self):
        return pd.Series(
            ["Alice", "Bob"],
            index=pd.Index(["a@b.edu", "c@d.edu"], name="Email"),
            name="FullName",
        )

    def test_empty_df(self):
        result = _serialize_dataframe(pd.DataFrame())
        assert result == "(No results)"

    def test_empty_series(self):
        result = _serialize_dataframe(pd.Series(dtype=object, name="x"))
        assert result == "(No results)"

    def test_org_format_has_pipes(self, sample_df):
        result = _serialize_dataframe(sample_df, "org")
        assert "|" in result
        assert "Alice" in result
        # orgtbl uses +---+ separators
        assert "+" in result or "-" in result

    def test_markdown_format(self, sample_df):
        result = _serialize_dataframe(sample_df, "markdown")
        assert "|" in result
        assert "Alice" in result

    def test_json_format(self, sample_df):
        result = _serialize_dataframe(sample_df, "json")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["Email"] == "a@b.edu"

    def test_series_converted_to_df(self, sample_series):
        result = _serialize_dataframe(sample_series, "json")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_truncation_org(self):
        big = pd.DataFrame({"x": range(_TABLE_ROW_LIMIT + 50)})
        result = _serialize_dataframe(big, "org")
        assert f"Showing {_TABLE_ROW_LIMIT} of {_TABLE_ROW_LIMIT + 50} rows" in result

    def test_truncation_json(self):
        big = pd.DataFrame({"x": range(_JSON_ROW_LIMIT + 50)})
        result = _serialize_dataframe(big, "json")
        assert f"Showing {_JSON_ROW_LIMIT} of {_JSON_ROW_LIMIT + 50} rows" in result

    def test_no_truncation_when_under_limit(self, sample_df):
        result = _serialize_dataframe(sample_df, "org")
        assert "Showing" not in result

    def test_multiindex_flattened(self):
        idx = pd.MultiIndex.from_tuples(
            [("2024-25", "a@b.edu"), ("2024-25", "c@d.edu")],
            names=["Year", "Email"],
        )
        df = pd.DataFrame({"Salary": [100, 200]}, index=idx)
        result = _serialize_dataframe(df, "json")
        parsed = json.loads(result)
        assert "Year" in parsed[0]
        assert "Email" in parsed[0]


# ===================================================================
# _serialize_list
# ===================================================================

class TestSerializeList:
    def test_empty(self):
        assert _serialize_list([]) == "(No results)"

    def test_default_is_json(self):
        # With the JSON-default switch, no-format calls return JSON.
        result = _serialize_list(["CAPBD", "DIVCO", "CUEP"])
        assert json.loads(result) == ["CAPBD", "DIVCO", "CUEP"]

    def test_newline_joined_for_org(self):
        result = _serialize_list(["CAPBD", "DIVCO", "CUEP"], "org")
        assert "CAPBD\nDIVCO\nCUEP" == result

    def test_json(self):
        result = _serialize_list(["CAPBD", "DIVCO"], "json")
        parsed = json.loads(result)
        assert parsed == ["CAPBD", "DIVCO"]


# ===================================================================
# _serialize_dict
# ===================================================================

class TestSerializeDict:
    def test_empty(self):
        assert _serialize_dict({}) == "(No results)"

    def test_json(self):
        result = _serialize_dict({"ECON": "Economics"}, "json")
        parsed = json.loads(result)
        assert parsed == {"ECON": "Economics"}

    def test_org_has_pipes(self):
        result = _serialize_dict({"ECON": "Economics"}, "org")
        assert "|" in result
        assert "ECON" in result


# ===================================================================
# _handle_error
# ===================================================================

class TestHandleError:
    def test_decryption_error(self):
        from ucbfaculty.rawdata import DecryptionError
        msg = _handle_error(DecryptionError("bad"))
        assert "passphrase" in msg.lower()
        assert "ucbfaculty unlock" in msg

    def test_file_not_found(self):
        msg = _handle_error(FileNotFoundError("missing.csv"))
        assert "File not found" in msg

    def test_key_error(self):
        msg = _handle_error(KeyError("bad_key"))
        assert "bad_key" in msg

    def test_value_error(self):
        msg = _handle_error(ValueError("invalid input"))
        assert "invalid input" in msg

    def test_generic_error(self):
        msg = _handle_error(RuntimeError("boom"))
        assert "RuntimeError" in msg
        assert "boom" in msg


# ===================================================================
# CLI argument parser
# ===================================================================

class TestCLIParser:
    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.default_format == "json"
        assert args.passphrase is None

    def test_passphrase(self):
        parser = _build_parser()
        args = parser.parse_args(["--passphrase", "secret123"])
        assert args.passphrase == "secret123"

    def test_format_markdown(self):
        parser = _build_parser()
        args = parser.parse_args(["--default-format", "markdown"])
        assert args.default_format == "markdown"

    def test_format_json(self):
        parser = _build_parser()
        args = parser.parse_args(["--default-format", "json"])
        assert args.default_format == "json"

    def test_invalid_format_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--default-format", "csv"])


# ===================================================================
# Tool function tests (mocked)
# ===================================================================

class TestSpellTool:
    def test_known_mapping(self):
        from ucbfaculty.mcp_server import spell
        with patch("ucbfaculty.spelling.spell", return_value="BIR"):
            result = spell("budget committee")
            assert result == "BIR"

    def test_passthrough(self):
        from ucbfaculty.mcp_server import spell
        with patch("ucbfaculty.spelling.spell", return_value="Unknown Thing"):
            result = spell("Unknown Thing")
            assert result == "Unknown Thing"


class TestSpellVariantsTool:
    def test_returns_variants(self):
        from ucbfaculty.mcp_server import spell_variants
        variants = ["Budget Committee", "Budget and Interdepartmental Relations", "BIR"]
        with patch("ucbfaculty.spelling.spellings", return_value=variants):
            result = spell_variants("BIR")
            assert "Budget Committee" in result

    def test_no_variants(self):
        from ucbfaculty.mcp_server import spell_variants
        with patch("ucbfaculty.spelling.spellings", return_value=["XYZ"]):
            result = spell_variants("XYZ")
            assert "No variants" in result

    def test_json_format(self):
        from ucbfaculty.mcp_server import spell_variants
        variants = ["Budget Committee", "BIR"]
        with patch("ucbfaculty.spelling.spellings", return_value=variants):
            result = spell_variants("BIR", format="json")
            parsed = json.loads(result)
            assert "Budget Committee" in parsed


class TestLookupFacultyTool:
    def test_exact_match(self):
        from ucbfaculty.mcp_server import lookup_faculty
        df = pd.DataFrame({
            "FullName": ["Panos Papadopoulos", "Alice Smith"],
            "DeptName": ["Mechanical Engineering", "Economics"],
        }, index=pd.Index(["panos@berkeley.edu", "alice@berkeley.edu"], name="Email"))

        with patch("ucbfaculty.faculty.faculty_vitals", return_value=df):
            with patch("ucbfaculty.FACULTY_NAME_DICTIONARY") as mock_dict:
                mock_dict.spell.return_value = "Panos Papadopoulos"
                result = lookup_faculty("Panos Papadopoulos")
                assert "Panos Papadopoulos" in result
                assert "Mechanical Engineering" in result

    def test_no_match(self):
        from ucbfaculty.mcp_server import lookup_faculty
        df = pd.DataFrame({
            "FullName": ["Alice Smith"],
            "DeptName": ["Economics"],
        }, index=pd.Index(["alice@berkeley.edu"], name="Email"))

        with patch("ucbfaculty.faculty.faculty_vitals", return_value=df):
            with patch("ucbfaculty.FACULTY_NAME_DICTIONARY") as mock_dict:
                mock_dict.spell.return_value = "Nobody Real"
                with patch("ucbfaculty.mcp_server.lookup_faculty.__module__", create=True):
                    pass
                result = lookup_faculty("Nobody Real")
                assert "No faculty member" in result

    def test_no_match_includes_closest_names(self):
        """A no-match should suggest the closest known names so callers
        can spot typos without an extra round trip.  Use an input that's
        far enough from every candidate to fall through Strategies 1-3."""
        from ucbfaculty.mcp_server import lookup_faculty
        df = pd.DataFrame({
            "FullName": [
                "Alice Smith", "Bob Jones", "Carol Jensen", "Dan Wu",
            ],
            "DeptName": ["Econ", "Math", "Stats", "Physics"],
        }, index=pd.Index(
            ["alice@b", "bob@b", "carol@b", "dan@b"], name="Email"
        ))
        with patch("ucbfaculty.faculty.faculty_vitals", return_value=df):
            with patch("ucbfaculty.FACULTY_NAME_DICTIONARY") as mock_dict:
                # A spelling-dict miss; nothing close enough to land via
                # substring or fuzzy-match-above-threshold.
                mock_dict.spell.return_value = "Xyzzy McPlugh"
                result = lookup_faculty("Xyzzy McPlugh")
                assert "No faculty member" in result
                assert "Closest names:" in result
                # At least one of the four candidates surfaces as a suggestion.
                assert any(
                    name in result
                    for name in [
                        "Alice Smith", "Bob Jones", "Carol Jensen", "Dan Wu",
                    ]
                )


class TestServiceHistoryTool:
    def _fake_member(self, *, verified, email="ligon@b", full="Ethan Ligon",
                     records=None):
        fm = MagicMock()
        fm.verified = verified
        fm.Email = email
        fm.FullName = full
        if records is None:
            records = [
                {"Academic Year": "2025-26", "Committee": "COMS",
                 "Role": "Chair", "ServiceStreak": 2, "ChairStreak": 1},
            ]
        fm.service_history.return_value = pd.DataFrame(records)
        return fm

    def test_echoes_resolved_subject(self):
        """The response should advertise which person matched, since the
        records themselves don't carry Email or FullName columns."""
        from ucbfaculty.mcp_server import service_history
        fake = self._fake_member(verified=True)
        with patch("ucbfaculty.faculty.FacultyMember", return_value=fake):
            result = service_history("Ethan Ligon", format="json")
        assert "Resolved 'Ethan Ligon' to ligon@b (Ethan Ligon)." in result
        # The JSON records still appear after the header line.
        body = result.split("\n", 1)[1]
        assert "COMS" in body
        assert json.loads(body)  # well-formed JSON list

    def test_no_history_still_echoes_subject(self):
        from ucbfaculty.mcp_server import service_history
        fake = self._fake_member(verified=True, records=[])
        with patch("ucbfaculty.faculty.FacultyMember", return_value=fake):
            result = service_history("Ethan Ligon")
        assert "Resolved 'Ethan Ligon' to ligon@b" in result
        assert "No service history" in result

    def test_unverified_returns_no_match(self):
        from ucbfaculty.mcp_server import service_history
        fake = self._fake_member(verified=False)
        with patch("ucbfaculty.faculty.FacultyMember", return_value=fake):
            result = service_history("Nobody Real")
        assert "No faculty member" in result
        # Importantly, no half-resolved subject leaks through.
        assert "Resolved" not in result


class TestCommitteeNamesTool:
    def test_returns_list(self):
        from ucbfaculty.mcp_server import committee_names
        with patch("ucbfaculty.faculty.committee_names", return_value=["CAPBD", "DIVCO"]):
            result = committee_names()
            assert "CAPBD" in result
            assert "DIVCO" in result

    def test_json_format(self):
        from ucbfaculty.mcp_server import committee_names
        with patch("ucbfaculty.faculty.committee_names", return_value=["CAPBD", "DIVCO"]):
            result = committee_names(format="json")
            parsed = json.loads(result)
            assert parsed == ["CAPBD", "DIVCO"]


class TestCommitteeYearRangeTool:
    def test_normal(self):
        from ucbfaculty.mcp_server import committee_year_range
        with patch("ucbfaculty.faculty.committee_year_range", return_value=(2005, 2025)):
            result = committee_year_range()
            assert "2005" in result
            assert "2025" in result

    def test_empty(self):
        from ucbfaculty.mcp_server import committee_year_range
        with patch("ucbfaculty.faculty.committee_year_range", return_value=(None, None)):
            result = committee_year_range()
            assert "No committee records" in result


class TestGetCurrentYearTool:
    def test_returns_string(self):
        from ucbfaculty.mcp_server import get_current_year
        with patch("ucbfaculty.faculty.get_current_year", return_value="2025-26"):
            assert get_current_year() == "2025-26"


class TestIngestDataframeTool:
    def test_file_not_found(self):
        from ucbfaculty.mcp_server import ingest_dataframe
        result = ingest_dataframe(filepath="/nonexistent/file.csv")
        assert "File not found" in result


class TestProposeSpecTool:
    def test_file_not_found(self):
        from ucbfaculty.mcp_server import propose_spec
        result = propose_spec(filepath="/nonexistent/file.csv")
        assert "File not found" in result


# ===================================================================
# query / describe tools (DuckDB-backed)
# ===================================================================

def _queries_data_available() -> bool:
    """Return True iff the queries layer can register its static views."""
    try:
        from ucbfaculty.faculty import faculty_vitals, committee_membership

        faculty_vitals()
        committee_membership()
        return True
    except Exception:
        return False


@pytest.fixture
def _reset_queries():
    """Reset the queries connection cache between tests."""
    from ucbfaculty import queries

    queries._reset()
    yield
    queries._reset()


class TestQueryTool:
    def test_simple_select_returns_json(self, _reset_queries):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query

        result = query("SELECT 1 AS x")
        parsed = json.loads(result)
        assert parsed[0]["x"] == 1

    def test_select_from_vitals(self, _reset_queries):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query

        result = query("SELECT Email FROM vitals LIMIT 3")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert all("Email" in row for row in parsed)

    def test_explicit_org_format(self, _reset_queries):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query

        result = query("SELECT Email FROM vitals LIMIT 2", format="org")
        # Org tables use pipe separators.
        assert "|" in result

    def test_sql_error_returns_readable_message(self, _reset_queries):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query

        result = query("SELECT * FROM definitely_not_a_view")
        assert "SQL error" in result

    def test_join_across_views(self, _reset_queries):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query

        result = query(
            "SELECT v.DeptName, COUNT(*) AS n "
            "FROM committees c JOIN vitals v ON c.Email = v.Email "
            "GROUP BY v.DeptName ORDER BY n DESC LIMIT 3"
        )
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) <= 3
        if parsed:
            assert "DeptName" in parsed[0]
            assert "n" in parsed[0]


class TestQueryOutputPath:
    """Spill-to-file: ``output_path`` writes full result to disk."""

    def test_parquet_spill_writes_file_and_returns_preview(
        self, _reset_queries, tmp_path,
    ):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query
        import pandas as pd

        spill = tmp_path / "vitals.parquet"
        result = query("SELECT * FROM vitals", output_path=str(spill))

        assert spill.exists()
        assert "wrote" in result and "rows" in result and "cols" in result
        assert "first 5 rows:" in result

        # Round-trip verifies file integrity.
        roundtrip = pd.read_parquet(spill)
        assert not roundtrip.empty
        assert "Email" in roundtrip.columns

    def test_csv_spill_writes_file(self, _reset_queries, tmp_path):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query
        import pandas as pd

        spill = tmp_path / "vitals.csv"
        result = query("SELECT * FROM vitals LIMIT 10", output_path=str(spill))

        assert spill.exists()
        assert "wrote" in result and str(spill) in result

        # CSV is readable as text and parseable by pandas.
        text = spill.read_text()
        assert "Email" in text.splitlines()[0]
        roundtrip = pd.read_csv(spill)
        assert len(roundtrip) > 0

    def test_pq_extension_uses_parquet(self, _reset_queries, tmp_path):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query
        import pandas as pd

        spill = tmp_path / "vitals.pq"
        query("SELECT * FROM vitals LIMIT 5", output_path=str(spill))

        assert spill.exists()
        # Must be readable as Parquet (not CSV).
        roundtrip = pd.read_parquet(spill)
        assert not roundtrip.empty

    def test_extensionless_path_appends_parquet(
        self, _reset_queries, tmp_path,
    ):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query

        bare = tmp_path / "spill"
        result = query("SELECT * FROM vitals LIMIT 5", output_path=str(bare))

        # Original path unchanged; .parquet was appended.
        assert not bare.exists()
        appended = tmp_path / "spill.parquet"
        assert appended.exists()
        assert str(appended) in result

    def test_unknown_extension_appends_parquet(
        self, _reset_queries, tmp_path,
    ):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query

        weird = tmp_path / "spill.foo"
        query("SELECT * FROM vitals LIMIT 5", output_path=str(weird))

        # We append rather than replace, so the user's name is preserved.
        appended = tmp_path / "spill.foo.parquet"
        assert appended.exists()

    def test_empty_result_skips_write(self, _reset_queries, tmp_path):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query

        spill = tmp_path / "empty.parquet"
        result = query(
            "SELECT * FROM vitals WHERE Email = 'nonexistent@nowhere'",
            output_path=str(spill),
        )

        assert "No results" in result
        assert "nothing written" in result
        assert not spill.exists()

    def test_preview_is_json_list_with_5_or_fewer_rows(
        self, _reset_queries, tmp_path,
    ):
        """The preview block must be valid JSON the agent can parse."""
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query

        spill = tmp_path / "preview.parquet"
        result = query("SELECT * FROM vitals", output_path=str(spill))

        # Preview lives after the literal "first 5 rows:" marker.
        marker = "first 5 rows:\n"
        assert marker in result
        preview_text = result.split(marker, 1)[1]
        preview = json.loads(preview_text)
        assert isinstance(preview, list)
        assert 0 < len(preview) <= 5

    def test_full_result_written_not_truncated(
        self, _reset_queries, tmp_path,
    ):
        """Spill must contain *all* rows, not the limit-capped subset."""
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import query
        import pandas as pd

        # Determine total row count without spilling.
        count_result = query("SELECT COUNT(*) AS n FROM vitals")
        total = json.loads(count_result)[0]["n"]

        spill = tmp_path / "full.parquet"
        # Pass a low `limit`; since output_path is set, the full result
        # must still be written.
        query("SELECT * FROM vitals", output_path=str(spill), limit=2)

        roundtrip = pd.read_parquet(spill)
        assert len(roundtrip) == total


class TestDescribeTool:
    def test_no_arg_returns_summary(self, _reset_queries):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import describe

        result = describe()
        parsed = json.loads(result)
        assert "vitals" in parsed["available"]
        assert "committees" in parsed["available"]
        assert parsed["static"] == ["vitals", "committees"]

    def test_specific_view_returns_columns(self, _reset_queries):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import describe

        result = describe(view="vitals")
        parsed = json.loads(result)
        assert parsed["view"] == "vitals"
        assert isinstance(parsed["columns"], list)
        col_names = [c[0] for c in parsed["columns"]]
        assert "Email" in col_names
        assert parsed["rows"] >= 0

    def test_unknown_view_returns_keyerror_message(self, _reset_queries):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import describe

        result = describe(view="not_a_real_view")
        assert "Unknown view" in result

    def test_org_format_renders_table(self, _reset_queries):
        if not _queries_data_available():
            pytest.skip("No fixture or RawData available")
        from ucbfaculty.mcp_server import describe

        result = describe(view="vitals", format="org")
        assert "|" in result  # orgtbl pipes
        assert "view: vitals" in result
