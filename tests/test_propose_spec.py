"""Tests for ucbfaculty.propose_spec and multi-path spec loading."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Type-inference helper tests
# ---------------------------------------------------------------------------

from ucbfaculty.propose_spec import (
    _is_boolean_column,
    _is_integer_column,
    _is_float_column,
    _is_categorical_column,
    _infer_column_type,
    _sanitize_feature_name,
    _classify_columns,
    _pick_index,
    _render_sources_stanza,
    _render_feature_specs_stanza,
    propose_spec,
)


class TestIsBooleanColumn:
    def test_yes_no_values(self):
        s = pd.Series(["Yes", "No", "Yes", "No", "Yes"])
        assert _is_boolean_column(s)

    def test_true_false_values(self):
        s = pd.Series(["True", "False", "true", "false"])
        assert _is_boolean_column(s)

    def test_zero_one_values(self):
        s = pd.Series(["0", "1", "1", "0", "1"])
        assert _is_boolean_column(s)

    def test_non_boolean(self):
        s = pd.Series(["apple", "banana", "cherry"])
        assert not _is_boolean_column(s)

    def test_empty(self):
        s = pd.Series([], dtype="object")
        assert not _is_boolean_column(s)

    def test_with_nulls(self):
        s = pd.Series(["Yes", None, "No", pd.NA, "Yes"])
        assert _is_boolean_column(s)


class TestIsIntegerColumn:
    def test_integer_strings(self):
        s = pd.Series(["1", "2", "3", "42"])
        assert _is_integer_column(s)

    def test_actual_integers(self):
        s = pd.Series([1, 2, 3, 42])
        assert _is_integer_column(s)

    def test_floats_that_are_ints(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _is_integer_column(s)

    def test_real_floats(self):
        s = pd.Series([1.5, 2.7, 3.14])
        assert not _is_integer_column(s)

    def test_mixed_types(self):
        s = pd.Series(["hello", "world"])
        assert not _is_integer_column(s)

    def test_empty(self):
        s = pd.Series([], dtype="object")
        assert not _is_integer_column(s)


class TestIsFloatColumn:
    def test_float_values(self):
        s = pd.Series([1.5, 2.7, 3.14, 0.1])
        assert _is_float_column(s)

    def test_integer_values_not_float(self):
        s = pd.Series([1, 2, 3])
        assert not _is_float_column(s)

    def test_mostly_parseable(self):
        s = pd.Series(["1.5", "2.7", "3.14", "0.5", "N/A"])
        assert _is_float_column(s)

    def test_non_numeric(self):
        s = pd.Series(["hello", "world", "foo"])
        assert not _is_float_column(s)


class TestIsCategoricalColumn:
    def test_low_cardinality(self):
        values = ["A", "B", "C"] * 100
        s = pd.Series(values)
        assert _is_categorical_column(s)

    def test_high_cardinality(self):
        s = pd.Series([f"val_{i}" for i in range(100)])
        assert not _is_categorical_column(s)

    def test_single_value(self):
        s = pd.Series(["A"] * 50)
        assert not _is_categorical_column(s)

    def test_small_dataset(self):
        s = pd.Series(["A", "B", "C", "A", "B"])
        assert _is_categorical_column(s)


class TestInferColumnType:
    def test_boolean(self):
        assert _infer_column_type(pd.Series(["Yes", "No", "Yes"])) == "bool"

    def test_integer(self):
        assert _infer_column_type(pd.Series([1, 2, 3])) == "int"

    def test_float(self):
        assert _infer_column_type(pd.Series([1.5, 2.7, 3.14])) == "float"

    def test_category(self):
        assert _infer_column_type(pd.Series(["A", "B"] * 50)) == "category"

    def test_string(self):
        assert _infer_column_type(pd.Series([f"name_{i}" for i in range(50)])) == "str"


# ---------------------------------------------------------------------------
# Feature name sanitisation
# ---------------------------------------------------------------------------

class TestSanitizeFeatureName:
    def test_space_separated(self):
        assert _sanitize_feature_name("Salary Step") == "SalaryStep"

    def test_underscore_separated(self):
        assert _sanitize_feature_name("first_name") == "FirstName"

    def test_hyphen_separated(self):
        assert _sanitize_feature_name("dept-code") == "DeptCode"

    def test_mixed(self):
        assert _sanitize_feature_name("employee_class - desc") == "EmployeeClassDesc"

    def test_already_clean(self):
        assert _sanitize_feature_name("Email") == "Email"


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

class TestClassifyColumns:
    def test_email_column_detected(self):
        df = pd.DataFrame({
            "contact": ["alice@berkeley.edu", "bob@berkeley.edu", "carol@berkeley.edu"],
            "score": [90, 85, 77],
        })
        analysis = _classify_columns(df)
        assert analysis["contact"]["role"] == "email_index"
        assert analysis["contact"]["feature_name"] == "Email"

    def test_name_column_detected(self):
        df = pd.DataFrame({
            "person": ["Alice Smith", "Bob Jones", "Carol White"],
            "value": [1, 2, 3],
        })
        analysis = _classify_columns(df)
        assert analysis["person"]["role"] == "name"
        assert analysis["person"]["feature_name"] == "FullName"

    def test_last_first_format(self):
        df = pd.DataFrame({
            "name": ["Smith, Alice", "Jones, Bob", "White, Carol"],
            "value": [1, 2, 3],
        })
        analysis = _classify_columns(df)
        assert analysis["name"]["needs_transform"] == "reversename"

    def test_split_name_columns(self):
        df = pd.DataFrame({
            "FirstName": ["Alice", "Bob", "Carol"],
            "LastName": ["Smith", "Jones", "White"],
            "score": [90, 85, 77],
        })
        analysis = _classify_columns(df)
        assert analysis["FirstName"]["role"] == "name_part"
        assert analysis["LastName"]["role"] == "name_part"
        # Virtual composite entry
        composite_keys = [k for k, v in analysis.items() if v["role"] == "name_composite"]
        assert len(composite_keys) == 1

    def test_feature_columns(self):
        df = pd.DataFrame({
            "email": ["a@b.edu", "c@d.edu", "e@f.edu"],
            "salary_step": [3, 5, 7],
        })
        analysis = _classify_columns(df)
        assert analysis["salary_step"]["role"] == "feature"
        assert analysis["salary_step"]["feature_name"] == "SalaryStep"
        assert analysis["salary_step"]["type"] == "int"


class TestPickIndex:
    def test_prefer_email(self):
        analysis = {
            "email": {"role": "email_index", "feature_name": "Email"},
            "name": {"role": "name", "feature_name": "FullName"},
        }
        assert _pick_index(analysis) == "Email"

    def test_fallback_to_name(self):
        analysis = {
            "name": {"role": "name", "feature_name": "FullName"},
            "score": {"role": "feature", "feature_name": "Score"},
        }
        assert _pick_index(analysis) == "FullName"

    def test_no_index(self):
        analysis = {
            "score": {"role": "feature", "feature_name": "Score"},
        }
        assert _pick_index(analysis) is None


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------

class TestRenderSourcesStanza:
    def test_basic_rendering(self):
        analysis = {
            "email": {
                "original_name": "email",
                "role": "email_index",
                "feature_name": "Email",
                "type": "str",
            },
            "score": {
                "original_name": "score",
                "role": "feature",
                "feature_name": "Score",
                "type": "int",
            },
        }
        text = _render_sources_stanza(
            "Proposed/test.csv", analysis, "Email", ["2025-26"], set(),
        )
        assert '"Proposed/test.csv":' in text
        assert "_authority_for:" in text
        assert "- Score" in text
        assert '  _years: ["2025-26"]' in text

    def test_existing_features_commented(self):
        analysis = {
            "email": {
                "original_name": "email",
                "role": "email_index",
                "feature_name": "Email",
                "type": "str",
            },
        }
        text = _render_sources_stanza(
            "Proposed/test.csv", analysis, "Email", ["2025-26"], {"Email"},
        )
        assert "# - Email" in text
        assert "already defined" in text

    def test_reversename_transform(self):
        analysis = {
            "name": {
                "original_name": "name",
                "role": "name",
                "feature_name": "FullName",
                "type": "str",
                "name_format": "last_first",
                "needs_transform": "reversename",
            },
        }
        text = _render_sources_stanza(
            "Proposed/test.csv", analysis, "FullName", ["2025-26"], set(),
        )
        assert "!transform [name, ucbfaculty.transformations.reversename]" in text


class TestRenderFeatureSpecsStanza:
    def test_basic(self):
        analysis = {
            "score": {
                "role": "feature",
                "feature_name": "Score",
                "type": "int",
            },
        }
        text = _render_feature_specs_stanza(analysis, "Email", set())
        assert "Score:" in text
        assert "_type: int" in text
        assert "- Email" in text

    def test_skip_existing(self):
        analysis = {
            "email": {
                "role": "email_index",
                "feature_name": "Email",
                "type": "str",
            },
        }
        text = _render_feature_specs_stanza(analysis, "Email", {"Email"})
        assert "Email:" not in text


# ---------------------------------------------------------------------------
# Multi-path spec loading
# ---------------------------------------------------------------------------

from ucbfaculty.feature_spec_loader import (
    load_feature_specs_multi,
    load_source_specs_multi,
    clear_spec_caches,
    _load_yaml,
)


class TestMultiPathLoading:
    def test_single_path_unchanged(self, tmp_path):
        """Single path behaves identically to load_feature_specs."""
        spec_file = tmp_path / "features_spec.yaml"
        spec_file.write_text(
            'TestFeat:\n  _description: "A test"\n  _type: str\n  _index:\n    - Email\n'
        )
        clear_spec_caches()
        specs = load_feature_specs_multi(spec_file)
        assert "TestFeat" in specs
        assert specs["TestFeat"].description == "A test"

    def test_merge_two_files(self, tmp_path):
        """Later file overrides earlier for duplicate keys."""
        file1 = tmp_path / "specs1.yaml"
        file1.write_text(
            'Feat1:\n  _description: "First"\n  _type: str\n  _index: [Email]\n'
            'Feat2:\n  _description: "From file1"\n  _type: int\n  _index: [Email]\n'
        )
        file2 = tmp_path / "specs2.yaml"
        file2.write_text(
            'Feat2:\n  _description: "From file2"\n  _type: float\n  _index: [Email]\n'
            'Feat3:\n  _description: "Third"\n  _type: str\n  _index: [Email]\n'
        )
        clear_spec_caches()
        specs = load_feature_specs_multi((file1, file2))
        assert "Feat1" in specs
        assert "Feat2" in specs
        assert "Feat3" in specs
        # Feat2 should come from file2 (later wins)
        assert specs["Feat2"].description == "From file2"
        assert specs["Feat2"].type == "float"

    def test_missing_file_skipped(self, tmp_path):
        """A non-existent file in the tuple is silently skipped."""
        file1 = tmp_path / "specs1.yaml"
        file1.write_text(
            'Feat1:\n  _description: "OK"\n  _type: str\n  _index: [Email]\n'
        )
        missing = tmp_path / "does_not_exist.yaml"
        clear_spec_caches()
        specs = load_feature_specs_multi((file1, missing))
        assert "Feat1" in specs

    def test_source_specs_merge(self, tmp_path):
        """Source specs from multiple files merge correctly."""
        file1 = tmp_path / "sources1.yaml"
        file1.write_text(
            '"file1.csv":\n  _authority_for:\n    - Feat1\n  _years: ["2025-26"]\n'
        )
        file2 = tmp_path / "sources2.yaml"
        file2.write_text(
            '"file2.csv":\n  _authority_for:\n    - Feat2\n  _years: ["2025-26"]\n'
        )
        clear_spec_caches()
        specs = load_source_specs_multi((file1, file2))
        assert "file1.csv" in specs
        assert "file2.csv" in specs


# ---------------------------------------------------------------------------
# YAML round-trip (write then load)
# ---------------------------------------------------------------------------

class TestYamlRoundTrip:
    def test_sources_stanza_loadable(self, tmp_path):
        """Generated sources.yaml text can be loaded back by _load_yaml."""
        analysis = {
            "Email": {
                "original_name": "Email",
                "role": "email_index",
                "feature_name": "Email",
                "type": "str",
            },
            "Score": {
                "original_name": "Score",
                "role": "feature",
                "feature_name": "Score",
                "type": "int",
            },
        }
        text = _render_sources_stanza(
            "Proposed/test.csv", analysis, "Email", ["2025-26"], set(),
        )
        yaml_file = tmp_path / "sources.yaml"
        yaml_file.write_text(text)
        clear_spec_caches()
        loaded = _load_yaml(yaml_file)
        assert "Proposed/test.csv" in loaded

    def test_feature_specs_stanza_loadable(self, tmp_path):
        """Generated features_spec.yaml text can be loaded back."""
        analysis = {
            "Score": {
                "role": "feature",
                "feature_name": "Score",
                "type": "int",
            },
        }
        text = _render_feature_specs_stanza(analysis, "Email", set())
        yaml_file = tmp_path / "features_spec.yaml"
        yaml_file.write_text(text)
        clear_spec_caches()
        loaded = _load_yaml(yaml_file)
        assert "Score" in loaded
        assert loaded["Score"]["_type"] == "int"


# ---------------------------------------------------------------------------
# Integration: propose_spec end-to-end
# ---------------------------------------------------------------------------

class TestProposeSpec:
    def test_csv_end_to_end(self, tmp_path):
        """propose_spec reads a CSV, copies it, and writes YAML files."""
        # Create a test CSV.
        csv_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            "email": ["alice@berkeley.edu", "bob@berkeley.edu", "carol@berkeley.edu"],
            "dept": ["EECS", "Math", "EECS"],
            "salary_step": [3, 5, 7],
        })
        df.to_csv(csv_file, index=False)

        proposed_dir = tmp_path / "Proposed"

        # Patch datadir so propose_spec doesn't need real RawData.
        with patch("ucbfaculty.rawdata.get_dataframe", return_value=df):
            result = propose_spec(
                csv_file,
                proposed_dir=proposed_dir,
                years=["2025-26"],
            )

        # Data file was copied.
        assert result["data_file"].exists()
        assert result["data_file"].parent == proposed_dir

        # YAML files were created.
        assert (proposed_dir / "sources.yaml").exists()
        assert (proposed_dir / "features_spec.yaml").exists()

        # Generated text is non-empty.
        assert result["sources_yaml"].strip()
        assert result["features_spec_yaml"].strip()

        # The source key uses the Proposed/ prefix.
        assert "Proposed/test_data.csv" in result["sources_yaml"]

    def test_rejects_gpg_file(self, tmp_path):
        gpg_file = tmp_path / "data.csv.gpg"
        gpg_file.write_bytes(b"fake gpg data")
        with pytest.raises(ValueError, match="unencrypted"):
            propose_spec(gpg_file)

    def test_rejects_unsupported_extension(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            propose_spec(txt_file)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            propose_spec("/nonexistent/path/data.csv")

    def test_xlsx_end_to_end(self, tmp_path):
        """propose_spec handles Excel files."""
        xlsx_file = tmp_path / "test_data.xlsx"
        df = pd.DataFrame({
            "Name": ["Alice Smith", "Bob Jones", "Carol White"],
            "Active": ["Yes", "No", "Yes"],
        })
        df.to_excel(xlsx_file, index=False)

        proposed_dir = tmp_path / "Proposed"

        with patch("ucbfaculty.rawdata.get_dataframe", return_value=df):
            result = propose_spec(
                xlsx_file,
                proposed_dir=proposed_dir,
                years=["2025-26"],
            )

        assert result["data_file"].exists()
        assert "Proposed/test_data.xlsx" in result["sources_yaml"]

    def test_appends_to_existing_yaml(self, tmp_path):
        """Calling propose_spec twice appends to existing YAML files."""
        proposed_dir = tmp_path / "Proposed"
        proposed_dir.mkdir()

        # First call.
        csv1 = tmp_path / "data1.csv"
        df1 = pd.DataFrame({"email": ["a@b.edu"], "val": [1]})
        df1.to_csv(csv1, index=False)

        with patch("ucbfaculty.rawdata.get_dataframe", return_value=df1):
            propose_spec(csv1, proposed_dir=proposed_dir, years=["2025-26"])

        # Second call.
        csv2 = tmp_path / "data2.csv"
        df2 = pd.DataFrame({"email": ["c@d.edu"], "score": [99]})
        df2.to_csv(csv2, index=False)

        with patch("ucbfaculty.rawdata.get_dataframe", return_value=df2):
            propose_spec(csv2, proposed_dir=proposed_dir, years=["2025-26"])

        sources_text = (proposed_dir / "sources.yaml").read_text()
        assert "data1.csv" in sources_text
        assert "data2.csv" in sources_text
