import pandas as pd

from ucbfaculty import faculty


def test_volunteer_years_handles_encrypted_workbook(monkeypatch, tmp_path):
    # Build a simple workbook and simulate decrypted bytes from an encrypted file.
    workbook = tmp_path / "volunteers.xlsx"
    pd.DataFrame({"AnyCommittee": [True]}).to_excel(workbook, sheet_name="2025-26", index=False)
    decrypted = workbook.read_bytes()

    monkeypatch.setattr(faculty, "datadir", tmp_path)
    monkeypatch.setattr(faculty, "_gpg_decrypt", lambda path: decrypted)
    faculty._volunteer_years_for_path.cache_clear()

    sheets = faculty._volunteer_years("volunteers.xlsx.gpg")
    assert sheets == ["2025-26"]


def test_volunteer_years_caches_decryption(monkeypatch, tmp_path):
    workbook = tmp_path / "volunteers.xlsx"
    pd.DataFrame({"AnyCommittee": [True]}).to_excel(workbook, sheet_name="2025-26", index=False)
    decrypted = workbook.read_bytes()

    calls = {"count": 0}
    def fake_decrypt(path):
        calls["count"] += 1
        return decrypted

    monkeypatch.setattr(faculty, "datadir", tmp_path)
    monkeypatch.setattr(faculty, "_gpg_decrypt", fake_decrypt)
    faculty._volunteer_years_for_path.cache_clear()

    assert faculty._volunteer_years("volunteers.xlsx.gpg") == ["2025-26"]
    assert faculty._volunteer_years("volunteers.xlsx.gpg") == ["2025-26"]
    assert calls["count"] == 1


def test_volunteers_returns_dataframe_with_volunteer_year(monkeypatch):
    """Test that volunteers() returns a DataFrame with VolunteerYear column."""
    from ucbfaculty import new_feature

    # Mock Features to return predictable data
    class FakeFeatures:
        def __init__(self, names):
            pass

        def __getitem__(self, year):
            df = pd.DataFrame({
                "AnyCommittee": [True, False],
                "Choice1": ["COR", "DIVCO"],
            })
            df.index = pd.Index(["a@b.edu", "c@d.edu"], name="Email")
            return df

    monkeypatch.setattr(new_feature, "Features", FakeFeatures)

    out = faculty.volunteers(year="2025-26")

    assert isinstance(out, pd.DataFrame)
    assert "VolunteerYear" in out.columns
    assert out["VolunteerYear"].iloc[0] == "2025-26"
    assert out.index.name == "Email"
