"""Spec-driven Feature class for loading data per YAML declarations.

Uses ``features_spec.yaml`` for feature metadata and ``sources.yaml``
for source-file declarations and column mappings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

import logging
import warnings
from typing import Any as TypingAny

import pandas as pd

from .feature_spec_loader import (
    FeatureSpec,
    SourceSpec,
    load_feature_specs_multi as load_feature_specs,
    load_source_specs_multi as load_source_specs,
)
from .transformations import normalize_academic_year
from .var_processing import load_source_dataframe, process_vars, get_vars_outputs, get_index_columns
from .merge_features import fullname_to_email_index
from . import FACULTY_NAME_DICTIONARY

logger = logging.getLogger(__name__)


def _warn_on_email_canonicalisation_collision(df: pd.DataFrame) -> None:
    """Warn when email canonicalisation collapses distinct identities.

    Background.  ``rename(index=EMAIL_DICTIONARY.inverse_d)`` rewrites
    each row's index value to its canonical form.  Same-canonical
    collisions are the *intended* outcome when one person had several
    email variants (e.g. ``mancosu@socrates.berkeley.edu`` collapsing
    to ``mancosu@berkeley.edu``).  Collisions with *different*
    ``FullName`` values are the silent identity-swap mode surfaced on
    2026-05-21, where two distinct people --- Paolo Mancosu (via the
    Philosophy 12A course mailbox) and R. Jay Wallace --- ended up at
    the same canonical email and downstream code mixed up which
    row's metadata went with which email.

    This check is intentionally diagnostic, not corrective: it logs
    at ``WARNING`` and lets the pipeline continue.  Fixing a true
    collision means correcting source data, not branching here.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe whose index has just been rewritten by the
        email dictionary.  Must carry a ``FullName`` column for the
        check to be useful; the function returns silently otherwise.

    Returns
    -------
    None
    """
    if "FullName" not in df.columns:
        return
    if not df.index.has_duplicates:
        return
    dup_rows = df.loc[df.index.duplicated(keep=False)]
    if dup_rows.empty:
        return
    if isinstance(df.index, pd.MultiIndex):
        grouped = dup_rows.groupby(level=list(range(df.index.nlevels)))
    else:
        grouped = dup_rows.groupby(level=0)
    for key, sub in grouped:
        names = sorted(set(sub["FullName"].dropna().unique()))
        if len(names) > 1:
            logger.warning(
                "Email canonicalisation collapses distinct identities to %r: %s",
                key, names,
            )


@dataclass(init=False)
class Feature:
    """A single named feature backed by YAML spec declarations.

    A ``Feature`` represents one column of data (e.g. ``'Salary'``,
    ``'FullName'``) whose extraction rules are declared in
    ``features_spec.yaml`` and whose source files are listed in
    ``sources.yaml``.  Subscripting with a year key materializes the
    data as a :class:`pandas.Series`.

    Attributes
    ----------
    name : str
        The feature's canonical name (must match a key in the spec).
    spec : FeatureSpec
        Parsed metadata for this feature (description, index, type).
    sources : dict[str, SourceSpec]
        Source-file specs that supply data for this feature.
    canonical_features : dict[str, FeatureSpec]
        The full set of feature specs (shared across features).
    """

    name: str
    spec: FeatureSpec
    sources: Dict[str, SourceSpec]
    canonical_features: Dict[str, FeatureSpec]

    def __init__(
        self,
        name: str,
        feature_specs: Optional[Dict[str, FeatureSpec]] = None,
        source_specs: Optional[Dict[str, SourceSpec]] = None,
        spec_path: Optional[Path | str] = None,
        sources_path: Optional[Path | str] = None,
    ) -> None:
        """Construct a Feature by name, loading specs as needed.

        Parameters
        ----------
        name : str
            Feature name (must exist in the feature specs YAML).
        feature_specs : dict[str, FeatureSpec] | None
            Pre-loaded feature specs.  When ``None``, loaded from
            *spec_path* or the default location.
        source_specs : dict[str, SourceSpec] | None
            Pre-loaded source specs.  When ``None``, loaded from
            *sources_path* or the default location.
        spec_path : Path | str | None
            Path to a ``features_spec.yaml`` override.
        sources_path : Path | str | None
            Path to a ``sources.yaml`` override.

        Raises
        ------
        KeyError
            If *name* is not present in the feature specs.
        ValueError
            If no source files declare data for *name*.
        """
        specs = feature_specs or load_feature_specs(spec_path)
        sources = source_specs or load_source_specs(sources_path)
        if name not in specs:
            available = ", ".join(sorted(specs.keys()))
            raise KeyError(f"Unknown feature: {name}. Available features: {available}")

        self.name = name
        self.spec = specs[name]
        self.canonical_features = specs
        self.sources = _sources_for_feature(name, sources)
        self._cache: Dict[Any, pd.Series] = {}
        if not self.sources:
            raise ValueError(f"No sources declared for feature '{name}' in sources spec")

    @classmethod
    def from_specs(
        cls,
        name: str,
        feature_specs: Dict[str, FeatureSpec],
        source_specs: Dict[str, SourceSpec],
    ) -> "Feature":
        return cls(name, feature_specs=feature_specs, source_specs=source_specs)

    def __getitem__(self, year: Any) -> pd.Series:
        """Resolve the feature for a given year and return a Series.

        Parameters
        ----------
        year : str | int
            One of:

            - ``'latest'`` — the most recent year across all sources.
            - ``'history'`` — all available years (returns a Series
              with a ``Year`` index level).
            - An academic-year string (e.g. ``'2023-24'``) or integer
              start year.

        Returns
        -------
        pandas.Series
            Feature values indexed according to the spec (typically
            by ``Email``, possibly with a ``Year`` level).
        """
        if year == "latest":
            year = _latest_year(self.sources.values())
        target_year = normalize_academic_year(year) if year not in ("history", None) else year
        if target_year in self._cache:
            return self._cache[target_year]
        result = self._resolve(target_year)
        self._cache[target_year] = result
        return result

    def _resolve(self, target_year: Any) -> pd.Series:
        """Uncached resolution of feature for a given year."""
        # History: include all years, error on conflicts.
        if target_year == "history":
            frames: list[pd.Series] = []
            by_year: Dict[str, list[SourceSpec]] = {}
            for src in self.sources.values():
                for src_year in _expand_years(src.years, src.year_range):
                    by_year.setdefault(src_year, []).append(src)
            conflicts = {y: [s.name for s in srcs] for y, srcs in by_year.items() if len(srcs) > 1}
            if conflicts:
                msgs = [f"{y}: {', '.join(names)}" for y, names in conflicts.items()]
                raise ValueError(f"Conflicting sources for feature '{self.name}': {'; '.join(msgs)}")

            # Collect unique sources.  Sources whose index already contains
            # Year embed all years in a single file and should be
            # materialized once (with year="history") instead of per-year.
            seen_sources: set[str] = set()
            for src_year, srcs in by_year.items():
                src = srcs[0]
                if src.name in seen_sources:
                    continue
                source_has_year_index = "Year" in _normalize_index(src.index)
                if source_has_year_index:
                    seen_sources.add(src.name)
                    for series in _materialize_from_source(self.name, src, "history", self.spec):
                        frames.append(_ensure_year_level(series, "history", self.spec.index, force=True))
                else:
                    for series in _materialize_from_source(self.name, src, src_year, self.spec):
                        frames.append(_ensure_year_level(series, src_year, self.spec.index, force=True))

            if not frames:
                return pd.Series(dtype=object)
            out = pd.concat(frames)
            return _reindex_to_spec(out, self.spec)

        candidates: list[SourceSpec] = []
        for src in self.sources.values():
            all_years = _expand_years(src.years, src.year_range)
            if target_year in all_years:
                candidates.append(src)
        actual_year = target_year
        if not candidates:
            # fallback to most recent available year
            latest = _latest_year(self.sources.values())
            actual_year = latest
            for src in self.sources.values():
                all_years = _expand_years(src.years, src.year_range)
                if actual_year in all_years:
                    candidates.append(src)
        if not candidates:
            raise ValueError(f"No source available for feature '{self.name}' and year '{target_year}'")
        if len(candidates) > 1:
            names = ", ".join(src.name for src in candidates)
            raise ValueError(f"Multiple sources claim feature '{self.name}' for year '{actual_year}': {names}")

        frames = []
        for src in candidates:
            frames.extend(
                _ensure_year_level(series, actual_year, self.spec.index, force="Year" in (self.spec.index or []))
                for series in _materialize_from_source(self.name, src, actual_year, self.spec)
            )

        if not frames:
            return pd.Series(dtype=object)

        out = pd.concat(frames)
        # Year-range sources embed all years in one file; keep only the
        # year actually requested (no-op for single-year sources).
        out = _select_year(out, actual_year)
        return _reindex_to_spec(out, self.spec)

    def available_years(self) -> list[str]:
        """Return the list of years for which this feature could be constructed (from specs only)."""
        years: list[str] = []
        for source in self.sources.values():
            years.extend(_expand_years(source.years, source.year_range))
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for y in years:
            if y not in seen:
                seen.add(y)
                ordered.append(y)
        return ordered


def load_feature(name: str, spec_path: Optional[str] = None, sources_path: Optional[str] = None) -> Feature:
    return Feature(name, spec_path=spec_path, sources_path=sources_path)


def available_features(
    year: Optional[str] = None,
    spec_path: Optional[Path | str] = None,
    sources_path: Optional[Path | str] = None,
) -> list[str]:
    """Return a sorted list of feature names defined in the YAML spec.

    Parameters
    ----------
    year : str | None
        When given, only features that have data for this academic year
        are included.  Accepts any format handled by
        :func:`normalize_academic_year`.
    spec_path : Path | str | None
        Path to a ``features_spec.yaml`` override.
    sources_path : Path | str | None
        Path to a ``sources.yaml`` override.

    Returns
    -------
    list[str]
        Alphabetically sorted feature names.
    """
    specs = load_feature_specs(spec_path)
    if year is None:
        return sorted(specs.keys())
    sources = load_source_specs(sources_path)
    target = normalize_academic_year(year)
    result = []
    for name in specs:
        try:
            feat = Feature(name, feature_specs=specs, source_specs=sources)
        except (KeyError, ValueError):
            continue
        if target in feat.available_years():
            result.append(name)
    return sorted(result)


class Features:
    """A collection of named features that resolves to a DataFrame.

    Usage::

        fs = Features(["Salary", "Step"])
        df = fs["2023-24"]       # DataFrame with Year+Email index
        df = fs["history"]       # all available years
    """

    def __init__(
        self,
        names: str | list[str],
        *,
        join: str = "outer",
        feature_specs: Optional[Dict[str, FeatureSpec]] = None,
        source_specs: Optional[Dict[str, SourceSpec]] = None,
        spec_path: Optional[Path | str] = None,
        sources_path: Optional[Path | str] = None,
    ) -> None:
        """Construct a collection of features for joint resolution.

        Parameters
        ----------
        names : str | list[str]
            Feature name or list of feature names to include.
        join : str
            Merge strategy when combining features: ``'outer'``
            (default) keeps all rows, ``'inner'`` keeps only rows
            present in every feature.
        feature_specs : dict[str, FeatureSpec] | None
            Pre-loaded feature specs (shared across all features).
        source_specs : dict[str, SourceSpec] | None
            Pre-loaded source specs (shared across all features).
        spec_path : Path | str | None
            Path to a ``features_spec.yaml`` override.
        sources_path : Path | str | None
            Path to a ``sources.yaml`` override.
        """
        if isinstance(names, str):
            names = [names]

        self.names = names
        self.join = join
        self._feature_specs = feature_specs or load_feature_specs(spec_path)
        self._source_specs = source_specs or load_source_specs(sources_path)
        self._features: Dict[str, Feature] = {
            n: Feature(n, feature_specs=self._feature_specs, source_specs=self._source_specs)
            for n in names
        }

    def __getitem__(self, year: Any) -> pd.DataFrame:
        """Resolve all features for *year* and merge into a single DataFrame."""
        series_dict: Dict[str, pd.Series] = {}
        _static_features: list[str] = []  # features whose spec lacks Year
        for name, feat in self._features.items():
            series = feat[year]

            # Ensure Year level is present for uniform concat
            has_year = (
                isinstance(series.index, pd.MultiIndex) and "Year" in series.index.names
            ) or series.index.name == "Year"

            if not has_year:
                if year not in (None, "history", "latest"):
                    resolved_year = normalize_academic_year(year)
                else:
                    resolved_year = _latest_year(feat.sources.values())
                series = _ensure_year_level(series, resolved_year, feat.spec.index, force=True)

            # Track static features (spec index lacks Year) for post-merge fill
            if year in (None, "history") and "Year" not in (feat.spec.index or []):
                _static_features.append(name)

            series_dict[name] = series

        if not series_dict:
            return pd.DataFrame()

        df = _merge_series(series_dict, self.join)

        # Forward/back-fill static features (spec index lacks Year) across
        # years within each Email group so that e.g. FullName propagates to
        # years where only volunteer data exists.
        if _static_features and isinstance(df.index, pd.MultiIndex) and "Email" in df.index.names:
            cols = [c for c in _static_features if c in df.columns]
            if cols:
                with pd.option_context("future.no_silent_downcasting", True):
                    df[cols] = df.groupby(level="Email")[cols].transform(
                        lambda g: g.ffill().bfill()
                    ).infer_objects(copy=False)

        return df

    def available_years(self) -> list[str]:
        """Return years common to all contained features."""
        sets = [set(f.available_years()) for f in self._features.values()]
        if not sets:
            return []
        common = sets[0]
        for s in sets[1:]:
            common &= s
        return sorted(common)


def _merge_series(series_dict: Dict[str, pd.Series], join: str = "outer") -> pd.DataFrame:
    """Merge a dict of named Series into a single DataFrame.

    Uses iterative ``merge`` on shared index columns.  When multiple
    series share a non-unique index, a within-group occurrence counter
    is added as a temporary merge key so that rows align positionally
    rather than producing a cartesian product.  The counter is dropped
    after the merge.
    """
    how = "outer" if join == "outer" else "inner"
    _OCCUR = "__merge_occur__"

    frames = []
    for name, series in series_dict.items():
        s = series.rename(name)
        # Drop index levels that collide with the series value column name
        idx_names = s.index.names if isinstance(s.index, pd.MultiIndex) else [s.index.name]
        if name in idx_names:
            s = s.droplevel(name)
        df = s.reset_index()
        # Add a within-group occurrence counter to disambiguate duplicate
        # index rows during the merge (e.g. multiple appointments per
        # person per year).
        idx_cols = [c for c in df.columns if c != name]
        if idx_cols:
            df[_OCCUR] = df.groupby(idx_cols).cumcount()
        else:
            df[_OCCUR] = 0
        frames.append(df)

    result = frames[0]
    for df in frames[1:]:
        shared_cols = [c for c in result.columns if c in df.columns and c not in series_dict]
        result = result.merge(df, on=shared_cols, how=how)

    # Drop the temporary occurrence counter
    if _OCCUR in result.columns:
        result = result.drop(columns=[_OCCUR])

    # Set index from non-feature columns
    index_cols = [c for c in result.columns if c not in series_dict]
    if index_cols:
        result = result.set_index(index_cols)

    return result




# --- helpers ---
def _sources_for_feature(name: str, source_specs: Dict[str, SourceSpec]) -> Dict[str, SourceSpec]:
    def _covers(feature_name: str, src: SourceSpec) -> bool:
        for entry in src.authority_for:
            if isinstance(entry, str) and entry == feature_name:
                return True
            if isinstance(entry, dict) and feature_name in entry:
                return True
        return False

    return {k: v for k, v in source_specs.items() if _covers(name, v)}


def _fullname_email_mapping() -> pd.Series:
    """Return a Series of FullName values indexed by Email.

    Uses Feature("FullName") which is safe from recursion because the
    FullName source has no index requiring FullName→Email mapping.
    """
    feat = Feature("FullName")
    latest = _latest_year(feat.sources.values())
    series = feat[latest]
    if isinstance(series, pd.DataFrame):
        series = series["FullName"] if "FullName" in series.columns else series.iloc[:, 0]
    return series.rename("FullName")


def _expand_years(years: list[str], year_range: Optional[str]) -> list[str]:
    out: list[str] = []
    out.extend([normalize_academic_year(y) for y in years])
    if year_range:
        try:
            start, end = year_range.split("-")
            start = int(start)
            end = int(end)
            for yr in range(start, end + 1):
                acad = f"{yr}-{(yr+1)%100:02d}"
                out.append(normalize_academic_year(acad))
        except Exception:
            out.append(year_range)
    seen = set()
    ordered = []
    for y in out:
        if y not in seen:
            seen.add(y)
            ordered.append(y)
    return ordered


def _latest_year(sources: Any) -> Optional[str]:
    years: list[str] = []
    for src in sources:
        years.extend(_expand_years(src.years, src.year_range))
    if not years:
        return None
    # assume normalized "YYYY-YY"
    years_sorted = sorted(years, reverse=True)
    return years_sorted[0]


def _normalize_index(idx: Any, return_entries: bool = False):
    if idx is None:
        return ([], []) if return_entries else []
    if isinstance(idx, str):
        return ([idx], []) if return_entries else [idx]
    if isinstance(idx, list):
        out = []
        index_vars = []
        for entry in idx:
            if isinstance(entry, dict):
                out.extend(entry.keys())
                index_vars.append(entry)
            else:
                out.append(entry)
        return (out, index_vars) if return_entries else out
    val = [str(idx)]
    return (val, []) if return_entries else val


def _materialize_from_source(feature: str, source: SourceSpec, year: Any, spec: FeatureSpec) -> list[pd.Series]:
    # Determine file spec for loading
    filespec: Any = source.name
    if source.sheets_are_years:
        filespec = [source.name, year]
    elif source.sheet:
        filespec = [source.name, source.sheet]

    # Load raw DataFrame
    df = load_source_dataframe(filespec)

    # Build vars config: feature vars + index var entries
    index_norm, index_var_entries = _normalize_index(source.index, return_entries=True)
    if not index_norm and spec.index:
        index_norm = [i for i in spec.index if i != "Year"]

    feature_vars: list[Any] = []
    for entry in source.authority_for:
        if isinstance(entry, str) and entry == feature:
            feature_vars.append({feature: feature})
        elif isinstance(entry, dict) and feature in entry:
            feature_vars.append(entry)
    vars_norm = feature_vars + index_var_entries

    # Apply transforms/renames/assigns
    if isinstance(vars_norm, str):
        vars_norm = (vars_norm,)
    processed_df = process_vars(df, vars_norm)

    # Determine output columns and set index
    vars_outputs = get_vars_outputs(vars_norm)
    intended_outputs = sorted(set(vars_outputs + index_norm))

    if index_norm:
        try:
            processed_df = processed_df.set_index(index_norm)
        except KeyError:
            pass

    # Normalize variant emails via the email dictionary
    if "Email" in index_norm:
        from ucbfaculty import EMAIL_DICTIONARY
        if EMAIL_DICTIONARY is not None:
            processed_df = processed_df.rename(index=EMAIL_DICTIONARY.inverse_d)
            _warn_on_email_canonicalisation_collision(processed_df)

    # Select output columns
    final_columns = [col for col in intended_outputs if col in processed_df.columns]
    built = processed_df[final_columns].squeeze() if final_columns else processed_df.squeeze()

    # Ensure Series
    if isinstance(built, pd.DataFrame):
        if feature in built.columns:
            built = built[feature]
        elif len(built.columns) > 0:
            built = built.iloc[:, 0]
        elif isinstance(built.index, pd.MultiIndex) and feature in built.index.names:
            built = built.index.get_level_values(feature).to_series(index=built.index)
            built.name = feature
        else:
            built = pd.Series(dtype=object, name=feature)

    if not isinstance(built, pd.Series):
        built = pd.Series(built, name=feature)

    # Map FullName->Email if spec expects Email but source index is FullName
    # (or has no meaningful index because the source lacked Email entirely).
    if spec.index and "Email" in spec.index:
        has_email = (
            built.index.name == "Email"
            or (isinstance(built.index, pd.MultiIndex) and "Email" in built.index.names)
        )
        has_fullname = (
            built.index.name == "FullName"
            or (isinstance(built.index, pd.MultiIndex) and "FullName" in built.index.names)
        )
        # If no Email and no FullName index, try constructing FullName from
        # FirstName+LastName columns in the raw data.
        if not has_email and not has_fullname:
            if "FirstName" in df.columns and "LastName" in df.columns:
                first = df["FirstName"].fillna("").astype(str).str.strip()
                last = df["LastName"].fillna("").astype(str).str.strip()
                constructed = (first + " " + last).str.strip()
                built.index = pd.Index(constructed, name="FullName")
                has_fullname = True

        if has_fullname:
            canonical = _fullname_email_mapping()
            if isinstance(built.index, pd.MultiIndex) and built.index.nlevels > 1:
                df = built.reset_index()
                # Apply spelling correction to FullName before merging
                df["FullName"] = df["FullName"].map(
                    lambda x: FACULTY_NAME_DICTIONARY.spell(x) if isinstance(x, str) else x
                )
                email_map = canonical.reset_index()
                email_map.columns = ["Email", "FullName"]
                df = df.merge(email_map, on="FullName", how="inner")
                idx_cols = [("Email" if c == "FullName" else c) for c in built.index.names]
                reindexed = df.set_index(idx_cols)
                if feature in reindexed.columns:
                    built = reindexed[feature]
                elif feature in reindexed.index.names:
                    built = reindexed.index.get_level_values(feature).to_series(index=reindexed.index)
                    built.name = feature
                else:
                    built = pd.Series(dtype=object, name=feature, index=reindexed.index)
            else:
                # Skip deduplication so that features with intentionally
                # non-unique indices (e.g. multiple Appointments per
                # person per year) are preserved as separate rows.
                built = fullname_to_email_index(
                    built, canonical_keys=canonical, interactive=False,
                    deduplicate=False,
                )

    # Drop entries with missing index values (e.g. roster rows without Email),
    # but only when the index is a named column (not a default RangeIndex).
    if built.index.name is not None or (isinstance(built.index, pd.MultiIndex) and any(n is not None for n in built.index.names)):
        if isinstance(built.index, pd.MultiIndex):
            mask = pd.Series(True, index=built.index)
            for level in range(built.index.nlevels):
                vals = built.index.get_level_values(level)
                mask &= vals.notna() & (vals.astype(str) != '') & (vals.astype(str) != 'nan')
            built = built[mask]
        else:
            idx = built.index
            valid = idx.notna() & (idx.astype(str) != '') & (idx.astype(str) != 'nan')
            if valid.any():
                built = built[valid]

    # Normalize Year index level if present
    built = _ensure_year_level(built, year, spec.index)
    return [built]


def _ensure_year_level(series: pd.Series, year: Any, spec_index: list[str], force: bool = False) -> pd.Series:
    """Normalize or add a Year index level when requested."""
    normalized_year = normalize_academic_year(year) if year not in (None, "history") else year
    if isinstance(series.index, pd.MultiIndex):
        names = list(series.index.names)
        if "Year" in names:
            level = names.index("Year")
            tuples = []
            for tup in series.index:
                tup = list(tup)
                tup[level] = normalize_academic_year(tup[level])
                tuples.append(tuple(tup))
            series.index = pd.MultiIndex.from_tuples(tuples, names=names)
            return series
        if not (force or "Year" in (spec_index or [])):
            return series
        names = ["Year"] + names
        series.index = pd.MultiIndex.from_tuples(
            [(normalized_year, *idx) for idx in series.index],
            names=names,
        )
        return series

    # Simple Index
    if series.index.name == "Year":
        series.index = series.index.map(normalize_academic_year)
        return series

    if not (force or "Year" in (spec_index or [])):
        return series

    names = ["Year", series.index.name]
    series.index = pd.MultiIndex.from_arrays(
        [[normalized_year] * len(series), series.index],
        names=names,
    )
    return series


def _select_year(series: pd.Series, year: Any) -> pd.Series:
    """Restrict a materialized series to a single academic year.

    Year-range sources (``_year_range``) embed every year in one file, so
    a single-year resolve must keep only the requested year -- otherwise a
    query like ``Feature("Appointment")["2024-25"]`` returns the entire
    history (and every past year resolves to the same blob).  This is a
    no-op for series without a ``Year`` index level, and for single-year
    sources whose rows were just labelled with the requested year.
    """
    if year in (None, "history"):
        return series
    ny = normalize_academic_year(year)
    idx = series.index
    if isinstance(idx, pd.MultiIndex) and idx.names and "Year" in idx.names:
        return series[idx.get_level_values("Year") == ny]
    if getattr(idx, "name", None) == "Year":
        return series[idx == ny]
    return series


def _coerce_bool(val: TypingAny) -> TypingAny:
    """Coerce a value to boolean, returning pd.NA for non-boolean values."""
    if pd.isna(val):
        return pd.NA
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        if val.lower() in ("true", "yes", "1"):
            return True
        if val.lower() in ("false", "no", "0"):
            return False
    return pd.NA


def _parse_tuple_string(s: str) -> list[str] | None:
    """Parse a string like '("A", "B", "C")' into a list of strings.

    Returns None if the string doesn't match the tuple format.
    """
    s = s.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return []
    # Split on commas and strip quotes
    items = []
    for item in inner.split(","):
        item = item.strip()
        # Remove surrounding quotes (double or single)
        if (item.startswith('"') and item.endswith('"')) or \
           (item.startswith("'") and item.endswith("'")):
            item = item[1:-1]
        items.append(item)
    return items


def _apply_dtype(series: pd.Series, type_spec: TypingAny, feature_name: str) -> pd.Series:
    """Convert series to specified dtype.

    Args:
        series: The pandas Series to convert.
        type_spec: The type specification from YAML. Can be:
            - None: no conversion
            - str keyword: "int", "str", "float", "bool", "category"
            - tuple/list: explicit category values
            - str in tuple format: '("A", "B", "C")' - parsed as explicit categories
        feature_name: Name of the feature (for warning messages).

    Returns:
        The converted Series.
    """
    if type_spec is None:
        return series

    # Tuple/list = explicit categories
    if isinstance(type_spec, (tuple, list)):
        categories = list(type_spec)
        observed = set(series.dropna().unique())
        expected = set(categories)
        unexpected = observed - expected
        if unexpected:
            warnings.warn(
                f"Feature '{feature_name}' has unexpected values: {unexpected}",
                stacklevel=3
            )
            categories = list(categories) + sorted(unexpected)
        return series.astype(pd.CategoricalDtype(categories=categories))

    # String - could be keyword or tuple notation
    if isinstance(type_spec, str):
        # Check for tuple notation first: ("A", "B", "C")
        parsed_tuple = _parse_tuple_string(type_spec)
        if parsed_tuple is not None:
            categories = parsed_tuple
            observed = set(series.dropna().unique())
            expected = set(categories)
            unexpected = observed - expected
            if unexpected:
                warnings.warn(
                    f"Feature '{feature_name}' has unexpected values: {unexpected}",
                    stacklevel=3
                )
                categories = list(categories) + sorted(unexpected)
            return series.astype(pd.CategoricalDtype(categories=categories))

        # String keywords
        t = type_spec.lower()
        if t == "int":
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        elif t == "str":
            return series.astype("string")
        elif t == "float":
            return pd.to_numeric(series, errors="coerce").astype("Float64")
        elif t == "bool":
            return series.map(_coerce_bool).astype("boolean")
        elif t == "category":
            return series.astype("category")
        else:
            warnings.warn(f"Unknown type '{type_spec}' for {feature_name}")

    return series


def _reindex_to_spec(series: pd.Series, spec: FeatureSpec) -> pd.Series:
    if not spec.index:
        return _apply_dtype(series, spec.type, spec.name)
    if isinstance(series.index, pd.MultiIndex):
        series = series.reindex(pd.MultiIndex.from_tuples(series.index, names=series.index.names))
    return _apply_dtype(series, spec.type, spec.name)
