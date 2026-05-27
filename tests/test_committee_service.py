import re
from pathlib import Path

import pytest

from ucbfaculty import datadir
from ucbfaculty.faculty import committee_membership, committee_names, committee_year_range


def _require_committee_data():
    path = Path(datadir) / "past_committee_appointments.csv"
    if not path.exists():
        pytest.skip("past_committee_appointments.csv not available in test environment")


def test_committee_names_available():
    _require_committee_data()
    committees = committee_names()
    assert committees, "Expected at least one committee name"


def test_committee_membership_filtered_by_year_range():
    _require_committee_data()
    committees = committee_names()
    assert committees, "Expected at least one committee name"
    committee = committees[0]
    start, end = committee_year_range()
    df_all = committee_membership(committee=committee, start_year=start, end_year=end)
    assert not df_all.empty, f"Expected membership records for {committee}"
    pattern = re.compile(r"^\d{4}-\d{2}$")
    assert df_all["Academic Year"].dropna().map(lambda value: bool(pattern.match(value))).all()
    assert df_all["FullName"].dropna().map(lambda name: not str(name).isupper()).all()

    mid = max(start, end - 5)
    df_recent = committee_membership(committee=committee, start_year=mid, end_year=end)
    if not df_recent.empty:
        assert df_recent["YearStart"].min() >= mid
        assert df_recent["YearStart"].max() <= end
