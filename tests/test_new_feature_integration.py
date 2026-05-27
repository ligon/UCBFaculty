"""Integration tests: load each feature defined in features_spec.yaml.

These tests require the GPG-decrypted data files in RawData.  They are
skipped automatically when a decryption or file-not-found error occurs.
"""
import pytest
import pandas as pd

import ucbfaculty.new_feature as nf
from ucbfaculty.feature_spec_loader import (
    _FEATURE_SPEC_CACHE,
    _SOURCE_SPEC_CACHE,
    load_feature_specs,
    load_source_specs,
)


@pytest.fixture(autouse=True)
def _clear_spec_caches():
    """Ensure fresh specs each run."""
    _FEATURE_SPEC_CACHE.clear()
    _SOURCE_SPEC_CACHE.clear()


def _load_feature(name, year=None):
    """Helper: build Feature and fetch a year, skipping on data errors."""
    try:
        feat = nf.Feature(name)
        yr = year or feat.available_years()[-1]
        return feat, feat[yr], yr
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Data unavailable for {name}: {exc}")
    except Exception as exc:
        if "decrypt" in str(exc).lower() or "secret key" in str(exc).lower():
            pytest.skip(f"Cannot decrypt data for {name}: {exc}")
        raise


# ── simple Email-indexed features from All Senate Faculty ────────────

@pytest.mark.parametrize("name", ["FullName", "LastName", "FirstName", "UID", "DeptCode", "DeptName"])
def test_senate_faculty_feature(name):
    feat, series, yr = _load_feature(name)
    assert isinstance(series, pd.Series), f"{name} should return a Series"
    assert len(series) > 0, f"{name} should have data"
    assert series.index.name == "Email" or (
        isinstance(series.index, pd.MultiIndex) and "Email" in series.index.names
    ), f"{name} should be indexed by Email"


def test_email_feature():
    feat, series, yr = _load_feature("Email")
    assert isinstance(series, pd.Series)
    assert len(series) > 0
    assert series.index.name == "FullName" or (
        isinstance(series.index, pd.MultiIndex) and "FullName" in series.index.names
    )
    # Emails should contain '@'
    assert series.str.contains("@").all(), "All Email values should contain '@'"


# ── COMS-derived features ────────────────────────────────────────────

def test_division():
    feat, series, yr = _load_feature("Division")
    assert len(series) > 0
    assert series.index.name == "Email" or (
        isinstance(series.index, pd.MultiIndex) and "Email" in series.index.names
    )


def test_employment_category():
    feat, series, yr = _load_feature("EmploymentCategory")
    assert len(series) > 0


def test_step():
    feat, series, yr = _load_feature("Step")
    assert len(series) > 0
    assert series.index.name == "Email" or (
        isinstance(series.index, pd.MultiIndex) and "Email" in series.index.names
    )


# ── Affiliations ─────────────────────────────────────────────────────

def test_affiliations():
    feat, series, yr = _load_feature("Affiliations")
    assert len(series) > 0


# ── Committee appointments (multi-year) ──────────────────────────────

def test_appointment_single_year():
    feat, series, yr = _load_feature("Appointment", year="2024-25")
    assert len(series) > 0
    if isinstance(series.index, pd.MultiIndex):
        assert "Email" in series.index.names


def test_appointment_history():
    feat = nf.Feature("Appointment")
    result = feat["history"]
    assert isinstance(result.index, pd.MultiIndex)
    assert "Year" in result.index.names
    assert "Email" in result.index.names
    years = result.index.get_level_values("Year").unique()
    assert len(years) > 1, "History should span multiple years"


def test_role_single_year():
    feat, series, yr = _load_feature("Role", year="2024-25")
    assert len(series) > 0


# ── Volunteer features (sheets_are_years) ────────────────────────────

@pytest.mark.parametrize("name", ["AnyCommittee", "Choice1", "Choice2", "Choice3", "Continue"])
def test_volunteer_feature(name):
    feat, series, yr = _load_feature(name)
    assert isinstance(series, pd.Series)
    assert len(series) > 0
    assert series.index.name == "Email" or (
        isinstance(series.index, pd.MultiIndex) and "Email" in series.index.names
    )


def test_volunteer_feature_multiple_years():
    feat = nf.Feature("AnyCommittee")
    years = feat.available_years()
    assert len(years) >= 2, "AnyCommittee should have multiple years"
    for yr in years:
        try:
            series = feat[yr]
            assert len(series) > 0, f"AnyCommittee[{yr}] should have data"
        except (FileNotFoundError, OSError, ValueError):
            pytest.skip(f"Data unavailable for AnyCommittee year {yr}")


# ── RecWellMember ────────────────────────────────────────────────────

def test_recwell_member():
    feat, series, yr = _load_feature("RecWellMember")
    assert len(series) > 0
    assert series.index.name == "Email" or (
        isinstance(series.index, pd.MultiIndex) and "Email" in series.index.names
    )


# ── Appointment (current year) ────────────────────────────────────────

def test_appointment_current():
    feat, series, yr = _load_feature("Appointment")
    assert len(series) > 0


# ── available_features covers all specs ──────────────────────────────

def test_available_features_complete():
    features = nf.available_features()
    specs = load_feature_specs()
    assert set(features) == set(specs.keys())


# ── available_years returns non-empty for every feature ──────────────

def _all_features():
    """Collect feature names at import time; return empty list if specs are unavailable."""
    try:
        return nf.available_features()
    except (FileNotFoundError, OSError):
        return []

@pytest.mark.parametrize("name", _all_features())
def test_available_years_non_empty(name):
    feat = nf.Feature(name)
    years = feat.available_years()
    assert len(years) > 0, f"{name} should have at least one available year"
