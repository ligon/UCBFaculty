import pandas as pd
import numpy as np
import pytest

from ucbfaculty.utils import name_variants as _name_variants, ensure_unique_index


class TestNameVariants:
    def test_simple_name(self):
        variants = _name_variants("Jane Doe")
        assert "Jane Doe" in variants
        assert "jane doe" in variants

    def test_name_with_periods(self):
        variants = _name_variants("J. Doe")
        assert "J Doe" in variants

    def test_single_char_filtering(self):
        # "A B Smith" -> middle initial "B" should be filtered out
        variants = _name_variants("Alice B. Smith")
        assert any("Alice Smith" in v for v in variants)

    def test_non_string(self):
        assert _name_variants(None) == []
        assert _name_variants(123) == []

    def test_no_duplicates(self):
        variants = _name_variants("jane doe")
        assert len(variants) == len(set(variants))


class TestEnsureUniqueIndex:
    def test_unique_passthrough(self):
        s = pd.Series([1, 2, 3], index=["a", "b", "c"])
        result = ensure_unique_index(s)
        pd.testing.assert_series_equal(result, s)

    def test_duplicate_index(self):
        s = pd.Series([1, 2, 3], index=["a", "a", "b"])
        result = ensure_unique_index(s)
        assert result.index.is_unique
        # "a" should be grouped into a list
        assert isinstance(result.loc["a"], list)
        assert set(result.loc["a"]) == {1, 2}
        assert result.loc["b"] == 3

    def test_duplicate_same_value(self):
        s = pd.Series([1, 1], index=["a", "a"])
        result = ensure_unique_index(s)
        # Same value twice -> deduplicated to single scalar
        assert result.loc["a"] == 1
