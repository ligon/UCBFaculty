import pandas as pd
import numpy as np
import pytest

from ucbfaculty import transformations


class TestMaskMissing:
    def test_none_passthrough(self):
        assert pd.isna(transformations.lower(None))

    def test_nan_passthrough(self):
        assert pd.isna(transformations.lower(float("nan")))

    def test_valid_scalar(self):
        assert transformations.lower("FOO") == "foo"


class TestReversename:
    def test_last_comma_first(self):
        assert transformations.reversename("Doe, Jane") == "Jane Doe"

    def test_no_comma(self):
        assert transformations.reversename("Jane Doe") == "Jane Doe"

    def test_missing(self):
        assert pd.isna(transformations.reversename(None))


class TestToBoolean:
    @pytest.mark.parametrize("val", [True, "True", "Yes", "yes", "true", 1, "1", "YES", "TRUE"])
    def test_truthy(self, val):
        assert transformations.to_boolean(val) is True

    @pytest.mark.parametrize("val", [False, "False", "No", "no", "false", 0, "0", "NO", "FALSE"])
    def test_falsy(self, val):
        assert transformations.to_boolean(val) is False

    def test_unrecognised(self):
        assert transformations.to_boolean("maybe") == "maybe"


class TestLowerUpperTitle:
    def test_lower_scalar(self):
        assert transformations.lower("FOO") == "foo"

    def test_upper_series(self):
        s = pd.Series(["foo", "bar"])
        result = transformations.upper(s)
        assert list(result) == ["FOO", "BAR"]

    def test_title_scalar(self):
        assert transformations.title("foo bar") == "Foo Bar"


class TestNormalizeAcademicYear:
    def test_int_input(self):
        assert transformations.normalize_academic_year(2023) == "2023-24"

    def test_string_yyyy_yy(self):
        assert transformations.normalize_academic_year("2023-24") == "2023-24"

    def test_six_digit(self):
        assert transformations.normalize_academic_year("202324") == "2023-24"

    def test_eight_digit(self):
        assert transformations.normalize_academic_year("20232024") == "2023-24"

    def test_invalid_passthrough(self):
        assert transformations.normalize_academic_year("foo") == "foo"

    def test_missing(self):
        assert pd.isna(transformations.normalize_academic_year(None))
