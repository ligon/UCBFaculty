"""Tests for the Feature/Features classes and build_feature_dataframe wrapper."""
from pathlib import Path

import pytest

from ucbfaculty.merge_features import build_feature_dataframe
from ucbfaculty.new_feature import Feature, available_features


def _require_data():
    try:
        Feature("FullName")["latest"]
    except Exception:
        pytest.skip("Feature data not available in test environment")


def test_feature_reports_unknown_feature():
    with pytest.raises(KeyError, match="Unknown feature"):
        Feature("Department")


def test_build_feature_dataframe_returns_expected_columns():
    _require_data()
    df = build_feature_dataframe(["FullName", "DeptName"], interactive=False)

    assert not df.empty
    assert "FullName" in df.columns
    assert "DeptName" in df.columns


def test_first_name_feature_not_identical_to_last_name():
    _require_data()
    first = Feature("FirstName")["latest"]
    assert not first.empty

    merged = build_feature_dataframe(["FullName", "FirstName", "LastName"], interactive=False)
    sample = merged.iloc[0]
    assert sample["FirstName"] != sample["LastName"]


def test_available_features_returns_sorted_list():
    names = available_features()
    assert len(names) > 0
    assert names == sorted(names)
