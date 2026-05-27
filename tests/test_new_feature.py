import logging

import pytest
import warnings
import pandas as pd

from ucbfaculty import new_feature
from ucbfaculty.feature_spec_loader import FeatureSpec, SourceSpec


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_specs(*feature_names):
    return {
        n: FeatureSpec(name=n, description=None, index=["Email"], type=str)
        for n in feature_names
    }


def _make_source(name, authority_for, years=None, year_range=None):
    return SourceSpec(
        name=name,
        authority_for=authority_for,
        years=years or ["2024-25"],
        year_range=year_range,
        index="Email",
        sheet=None,
        sheets_are_years=False,
    )


def _fake_series(feature, value="val", email="a@b.edu"):
    return pd.Series(
        [value],
        index=pd.Index([email], name="Email"),
        name=feature,
    )


def _patch_materialize(monkeypatch, values=None):
    """Patch _materialize_from_source to return a simple series."""
    def fake(feature, src, year, spec):
        val = (values or {}).get((feature, year), f"{feature}-{year}")
        return [_fake_series(feature, value=val)]
    monkeypatch.setattr(new_feature, "_materialize_from_source", fake)


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------

def test_source_spec_assigns_default_year(monkeypatch):
    from ucbfaculty import feature_spec_loader
    monkeypatch.setattr(feature_spec_loader, "_default_year", lambda: "2024-25")
    spec = SourceSpec.from_dict(
        "example.csv",
        {
            "_authority_for": ["Email"],
            "_index": "Email",
        },
    )

    assert spec.years == ["2024-25"]


def test_history_adds_year_level(monkeypatch):
    specs = _make_specs("Division")
    sources = {"source": _make_source("source", ["Division"])}

    _patch_materialize(monkeypatch)
    feature = new_feature.Feature("Division", feature_specs=specs, source_specs=sources)

    result = feature["history"]

    assert isinstance(result.index, pd.MultiIndex)
    assert "Year" in result.index.names
    assert result.index.get_level_values("Year")[0] == "2024-25"


# ---------------------------------------------------------------------------
# __getitem__ – specific year
# ---------------------------------------------------------------------------

def test_getitem_specific_year(monkeypatch):
    specs = _make_specs("Dept")
    sources = {"s": _make_source("s", ["Dept"], years=["2024-25"])}
    _patch_materialize(monkeypatch)

    feat = new_feature.Feature("Dept", feature_specs=specs, source_specs=sources)
    result = feat["2024-25"]

    assert len(result) == 1
    assert result.iloc[0] == "Dept-2024-25"


# ---------------------------------------------------------------------------
# Year fallback to most recent
# ---------------------------------------------------------------------------

def test_getitem_falls_back_to_latest_year(monkeypatch):
    specs = _make_specs("Rank")
    sources = {"s": _make_source("s", ["Rank"], years=["2022-23", "2023-24"])}
    _patch_materialize(monkeypatch, values={("Rank", "2023-24"): "latest"})

    feat = new_feature.Feature("Rank", feature_specs=specs, source_specs=sources)
    # Request a year not covered by any source -> should fall back
    result = feat["2030-31"]
    assert result.iloc[0] == "latest"


def test_getitem_none_falls_back_to_latest_year(monkeypatch):
    specs = _make_specs("Rank")
    sources = {"s": _make_source("s", ["Rank"], years=["2022-23", "2023-24"])}
    _patch_materialize(monkeypatch, values={("Rank", "2023-24"): "latest"})

    feat = new_feature.Feature("Rank", feature_specs=specs, source_specs=sources)
    result = feat[None]
    assert result.iloc[0] == "latest"


# ---------------------------------------------------------------------------
# Conflict detection (multiple sources for same year)
# ---------------------------------------------------------------------------

def test_getitem_raises_on_conflicting_sources(monkeypatch):
    specs = _make_specs("Dept")
    sources = {
        "a": _make_source("a", ["Dept"], years=["2024-25"]),
        "b": _make_source("b", ["Dept"], years=["2024-25"]),
    }
    _patch_materialize(monkeypatch)

    feat = new_feature.Feature("Dept", feature_specs=specs, source_specs=sources)
    with pytest.raises(ValueError, match="Multiple sources"):
        feat["2024-25"]


def test_history_raises_on_conflicting_sources(monkeypatch):
    specs = _make_specs("Dept")
    sources = {
        "a": _make_source("a", ["Dept"], years=["2024-25"]),
        "b": _make_source("b", ["Dept"], years=["2024-25"]),
    }
    _patch_materialize(monkeypatch)

    feat = new_feature.Feature("Dept", feature_specs=specs, source_specs=sources)
    with pytest.raises(ValueError, match="Conflicting sources"):
        feat["history"]


# ---------------------------------------------------------------------------
# available_years
# ---------------------------------------------------------------------------

def test_available_years_from_explicit_years():
    specs = _make_specs("X")
    sources = {"s": _make_source("s", ["X"], years=["2022-23", "2024-25"])}
    feat = new_feature.Feature("X", feature_specs=specs, source_specs=sources)
    assert feat.available_years() == ["2022-23", "2024-25"]


def test_available_years_with_year_range():
    specs = _make_specs("X")
    sources = {"s": _make_source("s", ["X"], years=[], year_range="2020-2022")}
    feat = new_feature.Feature("X", feature_specs=specs, source_specs=sources)
    yrs = feat.available_years()
    assert "2020-21" in yrs
    assert "2021-22" in yrs
    assert "2022-23" in yrs


# ---------------------------------------------------------------------------
# _expand_years helper
# ---------------------------------------------------------------------------

def test_expand_years_empty():
    assert new_feature._expand_years([], None) == []


def test_expand_years_deduplicates():
    result = new_feature._expand_years(["2024-25", "2024-25"], None)
    assert result == ["2024-25"]


# ---------------------------------------------------------------------------
# _latest_year helper
# ---------------------------------------------------------------------------

def test_latest_year_picks_most_recent():
    sources = [
        _make_source("a", [], years=["2020-21"]),
        _make_source("b", [], years=["2023-24"]),
    ]
    assert new_feature._latest_year(sources) == "2023-24"


def test_latest_year_empty():
    assert new_feature._latest_year([]) is None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_unknown_feature_raises_key_error():
    specs = _make_specs("Known")
    sources = {"s": _make_source("s", ["Known"])}
    with pytest.raises(KeyError, match="Unknown feature"):
        new_feature.Feature("Bogus", feature_specs=specs, source_specs=sources)


def test_no_sources_raises_value_error():
    specs = _make_specs("Orphan")
    sources = {"s": _make_source("s", ["Other"])}  # doesn't cover Orphan
    with pytest.raises(ValueError, match="No sources declared"):
        new_feature.Feature("Orphan", feature_specs=specs, source_specs=sources)


# ---------------------------------------------------------------------------
# available_features helper
# ---------------------------------------------------------------------------

def test_available_features(monkeypatch):
    fake_specs = {"Zebra": None, "Alpha": None, "Mid": None}
    monkeypatch.setattr(new_feature, "load_feature_specs", lambda path=None: fake_specs)
    result = new_feature.available_features()
    assert result == ["Alpha", "Mid", "Zebra"]


def test_available_features_filtered_by_year(monkeypatch):
    specs = _make_specs("A", "B")
    sources = {
        "sa": _make_source("sa", ["A"], years=["2024-25"]),
        "sb": _make_source("sb", ["B"], years=["2025-26"]),
    }
    monkeypatch.setattr(new_feature, "load_feature_specs", lambda path=None: specs)
    monkeypatch.setattr(new_feature, "load_source_specs", lambda path=None: sources)
    assert new_feature.available_features("2024-25") == ["A"]
    assert new_feature.available_features("2025-26") == ["B"]
    assert new_feature.available_features() == ["A", "B"]


# ---------------------------------------------------------------------------
# Features class
# ---------------------------------------------------------------------------

def test_features_getitem_merges(monkeypatch):
    specs = _make_specs("A", "B")
    sources = {
        "sa": _make_source("sa", ["A"], years=["2024-25"]),
        "sb": _make_source("sb", ["B"], years=["2024-25"]),
    }
    _patch_materialize(monkeypatch)

    fs = new_feature.Features(
        ["A", "B"], feature_specs=specs, source_specs=sources,
    )
    df = fs["2024-25"]
    assert isinstance(df, pd.DataFrame)
    assert "A" in df.columns
    assert "B" in df.columns


def test_features_single_string(monkeypatch):
    specs = _make_specs("C")
    sources = {"sc": _make_source("sc", ["C"], years=["2024-25"])}
    _patch_materialize(monkeypatch)

    fs = new_feature.Features("C", feature_specs=specs, source_specs=sources)
    df = fs["2024-25"]
    assert "C" in df.columns


def test_features_adds_year_level(monkeypatch):
    """Even simple Email-indexed features get a Year level in the DataFrame."""
    specs = _make_specs("A")
    sources = {"sa": _make_source("sa", ["A"], years=["2024-25"])}
    _patch_materialize(monkeypatch)

    fs = new_feature.Features(["A"], feature_specs=specs, source_specs=sources)
    df = fs["2024-25"]
    assert isinstance(df.index, pd.MultiIndex)
    assert "Year" in df.index.names


def test_features_year_none(monkeypatch):
    """year=None falls back to latest and adds Year level."""
    specs = _make_specs("A")
    sources = {"sa": _make_source("sa", ["A"], years=["2023-24", "2024-25"])}
    _patch_materialize(monkeypatch, values={("A", "2024-25"): "val"})

    fs = new_feature.Features(["A"], feature_specs=specs, source_specs=sources)
    df = fs[None]
    assert isinstance(df.index, pd.MultiIndex)
    assert "Year" in df.index.names
    assert df.index.get_level_values("Year")[0] == "2024-25"


def test_features_mixed_years(monkeypatch):
    """Features with different latest years both show their actual year."""
    specs = _make_specs("A", "B")
    sources = {
        "sa": _make_source("sa", ["A"], years=["2023-24"]),
        "sb": _make_source("sb", ["B"], years=["2024-25"]),
    }
    _patch_materialize(monkeypatch)

    fs = new_feature.Features(["A", "B"], feature_specs=specs, source_specs=sources)
    df = fs[None]
    assert "A" in df.columns
    assert "B" in df.columns
    assert "Year" in df.index.names


def test_features_join_inner(monkeypatch):
    """join='inner' restricts to shared index entries."""
    specs = _make_specs("A", "B")
    sources = {
        "sa": _make_source("sa", ["A"], years=["2024-25"]),
        "sb": _make_source("sb", ["B"], years=["2024-25"]),
    }

    def fake(feature, src, year, spec):
        email = "shared@b.edu" if feature == "A" else "other@b.edu"
        return [_fake_series(feature, value="v", email=email)]
    monkeypatch.setattr(new_feature, "_materialize_from_source", fake)

    fs_outer = new_feature.Features(
        ["A", "B"], feature_specs=specs, source_specs=sources, join="outer",
    )
    fs_inner = new_feature.Features(
        ["A", "B"], feature_specs=specs, source_specs=sources, join="inner",
    )
    assert len(fs_outer["2024-25"]) >= len(fs_inner["2024-25"])


def test_features_available_years():
    """available_years returns the intersection across all features."""
    specs = _make_specs("A", "B")
    sources = {
        "sa": _make_source("sa", ["A"], years=["2022-23", "2023-24", "2024-25"]),
        "sb": _make_source("sb", ["B"], years=["2023-24", "2024-25"]),
    }
    fs = new_feature.Features(["A", "B"], feature_specs=specs, source_specs=sources)
    assert fs.available_years() == ["2023-24", "2024-25"]


def test_features_available_years_no_overlap():
    """No common years -> empty list."""
    specs = _make_specs("A", "B")
    sources = {
        "sa": _make_source("sa", ["A"], years=["2022-23"]),
        "sb": _make_source("sb", ["B"], years=["2024-25"]),
    }
    fs = new_feature.Features(["A", "B"], feature_specs=specs, source_specs=sources)
    assert fs.available_years() == []


def test_features_history(monkeypatch):
    """Features['history'] returns a DataFrame with Year level."""
    specs = _make_specs("A")
    sources = {"sa": _make_source("sa", ["A"], years=["2023-24", "2024-25"])}

    call_years = []
    def fake(feature, src, year, spec):
        call_years.append(year)
        return [_fake_series(feature, value=f"v-{year}")]
    monkeypatch.setattr(new_feature, "_materialize_from_source", fake)

    fs = new_feature.Features(["A"], feature_specs=specs, source_specs=sources)
    df = fs["history"]
    assert isinstance(df, pd.DataFrame)
    assert "Year" in df.index.names


def test_feature_caches_by_year(monkeypatch):
    """Repeated access for the same year should not re-materialize."""
    specs = _make_specs("A")
    sources = {"sa": _make_source("sa", ["A"], years=["2024-25"])}

    call_count = 0
    def counting_fake(feature, src, year, spec):
        nonlocal call_count
        call_count += 1
        return [_fake_series(feature)]
    monkeypatch.setattr(new_feature, "_materialize_from_source", counting_fake)

    feat = new_feature.Feature("A", feature_specs=specs, source_specs=sources)
    feat["2024-25"]
    feat["2024-25"]
    assert call_count == 1


def test_feature_cache_different_years(monkeypatch):
    """Different years should each materialize independently."""
    specs = _make_specs("A")
    sources = {"sa": _make_source("sa", ["A"], years=["2023-24", "2024-25"])}

    call_count = 0
    def counting_fake(feature, src, year, spec):
        nonlocal call_count
        call_count += 1
        return [_fake_series(feature, value=f"v-{year}")]
    monkeypatch.setattr(new_feature, "_materialize_from_source", counting_fake)

    feat = new_feature.Feature("A", feature_specs=specs, source_specs=sources)
    feat["2023-24"]
    feat["2024-25"]
    feat["2023-24"]  # cached
    assert call_count == 2


def test_features_reuses_specs(monkeypatch):
    """Feature objects inside Features share the same spec dicts."""
    specs = _make_specs("A", "B")
    sources = {
        "sa": _make_source("sa", ["A"], years=["2024-25"]),
        "sb": _make_source("sb", ["B"], years=["2024-25"]),
    }
    fs = new_feature.Features(["A", "B"], feature_specs=specs, source_specs=sources)
    # Both Feature instances should reference the same spec dict
    assert fs._features["A"].canonical_features is fs._features["B"].canonical_features


# ---------------------------------------------------------------------------
# _apply_dtype tests
# ---------------------------------------------------------------------------

def test_apply_dtype_int():
    """Test int type conversion to Int64."""
    series = pd.Series(["1", "2", "3", None, "invalid"])
    result = new_feature._apply_dtype(series, "int", "TestFeature")
    assert result.dtype == "Int64"
    assert result.iloc[0] == 1
    assert result.iloc[1] == 2
    assert pd.isna(result.iloc[3])
    assert pd.isna(result.iloc[4])  # "invalid" coerced to NA


def test_apply_dtype_str():
    """Test str type conversion to StringDtype."""
    series = pd.Series([1, 2, "hello", None])
    result = new_feature._apply_dtype(series, "str", "TestFeature")
    assert result.dtype == "string"
    assert result.iloc[0] == "1"
    assert result.iloc[2] == "hello"


def test_apply_dtype_float():
    """Test float type conversion to Float64."""
    series = pd.Series(["1.5", "2.7", None, "invalid"])
    result = new_feature._apply_dtype(series, "float", "TestFeature")
    assert result.dtype == "Float64"
    assert result.iloc[0] == 1.5
    assert result.iloc[1] == 2.7
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])


def test_apply_dtype_bool():
    """Test bool type conversion to BooleanDtype."""
    series = pd.Series(["yes", "no", "True", "False", "1", "0", None, "invalid"])
    result = new_feature._apply_dtype(series, "bool", "TestFeature")
    assert result.dtype == "boolean"
    assert result.iloc[0] == True
    assert result.iloc[1] == False
    assert result.iloc[2] == True
    assert result.iloc[3] == False
    assert result.iloc[4] == True
    assert result.iloc[5] == False
    assert pd.isna(result.iloc[6])
    assert pd.isna(result.iloc[7])  # "invalid" -> NA


def test_apply_dtype_category_inferred():
    """Test category type with inferred categories."""
    series = pd.Series(["A", "B", "A", "C", None])
    result = new_feature._apply_dtype(series, "category", "TestFeature")
    assert result.dtype.name == "category"
    assert set(result.cat.categories) == {"A", "B", "C"}


def test_apply_dtype_category_explicit():
    """Test category type with explicit categories from tuple."""
    series = pd.Series(["Member", "Chair", "Member", None])
    result = new_feature._apply_dtype(series, ("Member", "Chair", "Vice Chair"), "TestFeature")
    assert result.dtype.name == "category"
    assert list(result.cat.categories) == ["Member", "Chair", "Vice Chair"]


def test_apply_dtype_category_tuple_string():
    """Test category type with tuple string notation from YAML."""
    series = pd.Series(["Active Faculty", "Emeritus (Retired)", None])
    # This is how YAML parses the tuple notation
    type_spec = '("Active Faculty", "Emeritus (Retired)", "Emeritus (Recalled)", "Deans/Admin")'
    result = new_feature._apply_dtype(series, type_spec, "TestFeature")
    assert result.dtype.name == "category"
    assert "Active Faculty" in result.cat.categories
    assert "Emeritus (Retired)" in result.cat.categories
    assert "Emeritus (Recalled)" in result.cat.categories
    assert "Deans/Admin" in result.cat.categories


def test_apply_dtype_category_warns_on_unexpected():
    """Test that explicit categories warn on unexpected values."""
    series = pd.Series(["Member", "Chair", "Unknown"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = new_feature._apply_dtype(series, ("Member", "Chair"), "TestFeature")
        assert len(w) == 1
        assert "unexpected values" in str(w[0].message).lower()
        assert "Unknown" in str(w[0].message)
    # Unexpected values should be added to categories
    assert "Unknown" in result.cat.categories


def test_apply_dtype_none_passthrough():
    """Test that None type_spec returns series unchanged."""
    series = pd.Series(["a", "b", "c"])
    result = new_feature._apply_dtype(series, None, "TestFeature")
    assert result.equals(series)


def test_apply_dtype_unknown_warns():
    """Test that unknown type keyword emits a warning."""
    series = pd.Series(["a", "b"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = new_feature._apply_dtype(series, "unknowntype", "TestFeature")
        assert len(w) == 1
        assert "Unknown type" in str(w[0].message)
    # Series should be returned unchanged
    assert result.equals(series)


# ---------------------------------------------------------------------------
# _coerce_bool tests
# ---------------------------------------------------------------------------

def test_coerce_bool_true_values():
    """Test _coerce_bool recognizes true values."""
    assert new_feature._coerce_bool("true") is True
    assert new_feature._coerce_bool("True") is True
    assert new_feature._coerce_bool("TRUE") is True
    assert new_feature._coerce_bool("yes") is True
    assert new_feature._coerce_bool("Yes") is True
    assert new_feature._coerce_bool("1") is True
    assert new_feature._coerce_bool(True) is True


def test_coerce_bool_false_values():
    """Test _coerce_bool recognizes false values."""
    assert new_feature._coerce_bool("false") is False
    assert new_feature._coerce_bool("False") is False
    assert new_feature._coerce_bool("FALSE") is False
    assert new_feature._coerce_bool("no") is False
    assert new_feature._coerce_bool("No") is False
    assert new_feature._coerce_bool("0") is False
    assert new_feature._coerce_bool(False) is False


def test_coerce_bool_na_values():
    """Test _coerce_bool returns NA for invalid/null values."""
    assert pd.isna(new_feature._coerce_bool(None))
    assert pd.isna(new_feature._coerce_bool(pd.NA))
    assert pd.isna(new_feature._coerce_bool("maybe"))
    assert pd.isna(new_feature._coerce_bool("invalid"))


# ---------------------------------------------------------------------------
# Email canonicalisation collision check
#
# Regression coverage for the Mancosu / Wallace identity-swap surfaced on
# 2026-05-21: two distinct people whose original emails both normalised
# (via EMAIL_DICTIONARY.inverse_d) to the same canonical address ended up
# at the same index entry, and downstream lookups silently mixed up the
# rows.  The check warns at this stage so the pipeline still completes
# but the operator can see which identities collided.
# ---------------------------------------------------------------------------

class TestWarnOnEmailCanonicalisationCollision:
    def _df_with_index(self, index, fullnames=None):
        cols = {}
        if fullnames is not None:
            cols["FullName"] = fullnames
        df = pd.DataFrame(cols, index=pd.Index(index, name="Email"))
        return df

    def test_no_warning_when_no_fullname_column(self, caplog):
        df = pd.DataFrame(index=pd.Index(
            ["rjw@berkeley.edu", "rjw@berkeley.edu"], name="Email",
        ))
        with caplog.at_level(logging.WARNING, logger="ucbfaculty.new_feature"):
            new_feature._warn_on_email_canonicalisation_collision(df)
        assert caplog.records == []

    def test_no_warning_when_index_is_unique(self, caplog):
        df = self._df_with_index(
            ["a@berkeley.edu", "b@berkeley.edu"],
            fullnames=["Alice Smith", "Bob Jones"],
        )
        with caplog.at_level(logging.WARNING, logger="ucbfaculty.new_feature"):
            new_feature._warn_on_email_canonicalisation_collision(df)
        assert caplog.records == []

    def test_no_warning_when_duplicate_index_same_fullname(self, caplog):
        # Same person with two source rows that canonicalised together.
        df = self._df_with_index(
            ["mancosu@berkeley.edu", "mancosu@berkeley.edu"],
            fullnames=["Paolo Mancosu", "Paolo Mancosu"],
        )
        with caplog.at_level(logging.WARNING, logger="ucbfaculty.new_feature"):
            new_feature._warn_on_email_canonicalisation_collision(df)
        assert caplog.records == []

    def test_warning_when_duplicate_index_distinct_fullname(self, caplog):
        # The Mancosu / Wallace case.
        df = self._df_with_index(
            ["rjw@berkeley.edu", "rjw@berkeley.edu"],
            fullnames=["R. Jay Wallace", "Paolo Mancosu"],
        )
        with caplog.at_level(logging.WARNING, logger="ucbfaculty.new_feature"):
            new_feature._warn_on_email_canonicalisation_collision(df)
        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "rjw@berkeley.edu" in msg
        assert "Paolo Mancosu" in msg
        assert "R. Jay Wallace" in msg

    def test_nan_fullname_alone_not_enough_to_flag(self, caplog):
        # If only one side has a known FullName, there is not enough
        # evidence to declare an identity collision.
        df = self._df_with_index(
            ["rjw@berkeley.edu", "rjw@berkeley.edu"],
            fullnames=["R. Jay Wallace", float("nan")],
        )
        with caplog.at_level(logging.WARNING, logger="ucbfaculty.new_feature"):
            new_feature._warn_on_email_canonicalisation_collision(df)
        assert caplog.records == []
