import pandas as pd
import pytest

import ucbfaculty.faculty as fac


def test_normalize_year_input_handles_int_and_strings():
    assert fac.normalize_year_input(2024) == (2024, "2024-25")
    assert fac.normalize_year_input("2024") == (2024, "2024-25")
    assert fac.normalize_year_input("2024-25") == (2024, "2024-25")


def test_committee_membership_filters_by_year(monkeypatch):
    hist_df = pd.DataFrame(
        {
            "Committee": ["COMS", "COMS", "COMS"],
            "Academic Year": ["2023-24", "2024-25", "2025-26"],
            "YearStart": [2023, 2024, 2025],
            "FullName": ["A X", "B Y", "C Z"],
            "Email": ["a@example.com", "b@example.com", "c@example.com"],
            "DeptName": ["Dept", "Dept", "Dept"],
            "Role": ["Member", "Member", "Member"],
        }
    )

    monkeypatch.setattr(fac, "_committee_service_records", lambda: hist_df.copy())
    monkeypatch.setattr(fac.spelling, "spell", lambda x: x)
    if hasattr(fac._committee_service_records, "cache_clear"):
        fac._committee_service_records.cache_clear()

    df = fac.committee_membership(committee="COMS", start_year=2024, end_year=2025)
    assert set(df["YearStart"].unique()) == {2024, 2025}
    assert "FullName" in df.columns
    assert "Email" in df.columns


def test_committee_membership_excludes_out_of_range(monkeypatch):
    hist_df = pd.DataFrame(
        {
            "Committee": ["COMS"],
            "Academic Year": ["2023-24"],
            "YearStart": [2023],
            "FullName": ["A X"],
            "Email": ["a@example.com"],
            "DeptName": ["Dept"],
            "Role": ["Member"],
        }
    )

    monkeypatch.setattr(fac, "_committee_service_records", lambda: hist_df.copy())
    monkeypatch.setattr(fac.spelling, "spell", lambda x: x)

    df = fac.committee_membership(committee="COMS", start_year=2026, end_year=2027)
    assert df.empty
