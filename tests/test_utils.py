import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from ucbfaculty.utils import faculties_to_df, df_to_faculties


def _make_member(**kwargs):
    """Create a SimpleNamespace mimicking FacultyMember attributes."""
    defaults = dict(Email="a@b.edu", FullName="A B", FirstName="A", LastName="B", DeptName="CS", UID="123")
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestFacultiesToDf:
    def test_basic(self):
        m1 = _make_member(Email="x@y.edu", FullName="X Y")
        m2 = _make_member(Email="z@y.edu", FullName="Z Y")
        df = faculties_to_df([m1, m2])
        assert df.index.name == "Email"
        assert list(df.index) == ["x@y.edu", "z@y.edu"]
        assert "FullName" in df.columns

    def test_extra_cols(self):
        m = _make_member(Email="a@b.edu")
        df = faculties_to_df([m], extra_cols={"a@b.edu": {"Role": "Chair"}})
        assert "Role" in df.columns
        assert df.loc["a@b.edu", "Role"] == "Chair"

    def test_empty_list(self):
        df = faculties_to_df([])
        assert len(df) == 0


class TestDfToFaculties:
    def test_roundtrip(self):
        mock_fm_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.match_info = {"method": "direct"}
        mock_fm_class.return_value = mock_instance

        df = pd.DataFrame({"FullName": ["A B"]}, index=pd.Index(["a@b.edu"], name="Email"))

        with patch("ucbfaculty.utils.FacultyMember", mock_fm_class):
            result = df_to_faculties(df)
        assert len(result) == 1
        mock_fm_class.assert_called_once_with(Email="a@b.edu")

    def test_email_as_column(self):
        mock_fm_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.match_info = {"method": "direct"}
        mock_fm_class.return_value = mock_instance

        df = pd.DataFrame({"Email": ["a@b.edu"], "FullName": ["A B"]})

        with patch("ucbfaculty.utils.FacultyMember", mock_fm_class):
            result = df_to_faculties(df)
        assert len(result) == 1
