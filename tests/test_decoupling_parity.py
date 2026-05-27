"""Feature integration tests: verify Feature class produces valid output.

These tests require GPG-decrypted data files and are skipped otherwise.
"""
import pytest
import pandas as pd

import ucbfaculty.new_feature as nf


def _load(name):
    """Load feature via Feature class."""
    try:
        feat = nf.Feature(name)
        yr = feat.available_years()[-1]
        return feat[yr]
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Data unavailable: {exc}")
    except Exception as exc:
        if "decrypt" in str(exc).lower():
            pytest.skip(f"Cannot decrypt: {exc}")
        raise


EMAIL_INDEXED_FEATURES = [
    "FullName",
    "LastName",
    "FirstName",
    "UID",
    "DeptCode",
    "DeptName",
    "Step",
    "Affiliations",
    "RecWellMember",
    "Division",
    "EmploymentCategory",
]


@pytest.mark.parametrize("name", EMAIL_INDEXED_FEATURES)
def test_feature_produces_nonempty_series(name):
    """Each feature should resolve to a non-empty Series."""
    result = _load(name)
    assert len(result) > 0
    idx_name = result.index.name if not isinstance(result.index, pd.MultiIndex) else result.index.names[0]
    assert idx_name == "Email", f"Expected Email index, got {idx_name}"


def test_email_feature():
    result = _load("Email")
    assert len(result) > 0


def test_any_committee():
    result = _load("AnyCommittee")
    assert len(result) > 0


def test_appointment_latest():
    result = _load("Appointment")
    assert len(result) > 0
