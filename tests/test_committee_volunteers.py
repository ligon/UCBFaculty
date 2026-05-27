"""Regression tests for ``Committee.volunteers``.

These use monkeypatch to stub out the underlying ``faculty.volunteers``
loader so the test does not require encrypted volunteer workbooks or
fixture data.

Background: ``Committee.volunteers`` previously called
``pd.concat(frames, ignore_index=True, ...)`` on Email-indexed frames,
which silently discarded the Email index, and the column-based recovery
clause never fired (Email was never a column).  See commit history.
"""
import pandas as pd

from ucbfaculty import committee as committee_mod
from ucbfaculty import faculty as fac


def _fake_volunteer_frame(emails, committee_acronym, vol_year):
    """Build an Email-indexed frame matching what ``faculty.volunteers`` returns."""
    df = pd.DataFrame({
        "Committee": [committee_acronym] * len(emails),
        "Choice1": [committee_acronym] * len(emails),
        "VolunteerYear": [vol_year] * len(emails),
    })
    df.index = pd.Index(list(emails), name="Email")
    return df


def test_committee_volunteers_preserves_email_identity(monkeypatch):
    """Email must survive the concat step (was lost via ignore_index=True)."""
    # Spell-pass-through so Committee("X") yields acronym "X".
    monkeypatch.setattr(committee_mod.spelling, "spell", lambda x: x)

    # Stub the per-year loader to return an Email-indexed frame.
    def fake_volunteers(year, data_fn=None):
        return _fake_volunteer_frame(
            ["a@b.edu", "c@d.edu"], "TESTCOM", year
        )
    monkeypatch.setattr(fac, "volunteers", fake_volunteers)
    monkeypatch.setattr(
        fac, "normalize_year_input",
        lambda y: (2025, "2025-26") if y in (2025, "2025-26") else (None, None),
    )

    com = committee_mod.Committee("TESTCOM")
    out = com.volunteers(year="2025-26", as_df=True)

    assert isinstance(out, pd.DataFrame)
    assert not out.empty
    # Bug regression: Email must be the index AND a column (drop=False).
    assert out.index.name == "Email", "Email index was dropped by concat"
    assert "Email" in out.columns, "Email column was not preserved"
    assert set(out.index) == {"a@b.edu", "c@d.edu"}
    assert (out["Email"] == out.index).all()


def test_committee_volunteers_all_years_preserves_email(monkeypatch):
    """The ``year='all'`` path stacks multiple years and must keep Email."""
    monkeypatch.setattr(committee_mod.spelling, "spell", lambda x: x)
    monkeypatch.setattr(fac, "_volunteer_years", lambda data_fn: ["2024-25", "2025-26"])
    monkeypatch.setattr(
        fac, "normalize_year_input",
        lambda y: {
            "2024-25": (2024, "2024-25"),
            "2025-26": (2025, "2025-26"),
        }.get(y, (None, None)),
    )

    def fake_volunteers(year, data_fn=None):
        emails = {
            "2024-25": ["a@b.edu"],
            "2025-26": ["a@b.edu", "c@d.edu"],
        }[year]
        return _fake_volunteer_frame(emails, "TESTCOM", year)
    monkeypatch.setattr(fac, "volunteers", fake_volunteers)

    com = committee_mod.Committee("TESTCOM")
    out = com.volunteers(year="all", as_df=True)

    assert out.index.name == "Email"
    assert "Email" in out.columns
    # 1 row from 2024-25 + 2 rows from 2025-26 = 3 rows; "a@b.edu" appears twice.
    assert len(out) == 3
    assert (out.index == "a@b.edu").sum() == 2
    assert "VolunteerYear" in out.columns
    assert set(out["VolunteerYear"]) == {"2024-25", "2025-26"}
