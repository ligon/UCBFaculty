"""Loader utilities for Feature spec and sources metadata.

Reads feature specs from ``features_spec.yaml`` and source declarations
from ``sources.yaml``, supporting single or multiple files per type.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from yaml import SafeLoader

from .var_processing import transform_constructor as fb_transform_constructor
from .var_processing import assign_constructor as fb_assign_constructor
from .transformations import normalize_academic_year
from . import datadir, feature_specs_yaml, sources_yaml


def _resolve_path(candidate: Path | str) -> Path:
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path
    return Path(datadir) / path


def _default_year() -> str:
    """Best-effort default academic year for undated sources."""
    try:
        from . import current_year

        if current_year:
            return normalize_academic_year(current_year)
    except Exception:
        pass

    try:
        from .faculty import _guess_year_from_clock

        return normalize_academic_year(_guess_year_from_clock())
    except Exception:
        return "Unknown"


def _load_yaml(path: Path | str) -> dict:
    target = _resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Spec not found: {target}")
    with target.open("r", encoding="utf-8") as handle:
        loader = SafeLoader
        loader.add_constructor("!transform", fb_transform_constructor)
        loader.add_constructor("!assign", fb_assign_constructor)
        return yaml.load(handle, Loader=loader) or {}


_FEATURE_SPEC_CACHE: Dict[Path, Dict[str, "FeatureSpec"]] = {}
_SOURCE_SPEC_CACHE: Dict[Path, Dict[str, "SourceSpec"]] = {}


@dataclass
class FeatureSpec:
    """Parsed metadata for a single feature from ``features_spec.yaml``.

    Attributes
    ----------
    name : str
        The feature's canonical name (matches its YAML key).
    description : str | None
        Human-readable description from the ``_description`` field.
    index : list[str]
        Index columns this feature is keyed on (e.g. ``['Email']``).
    type : Any
        Type specification (``'int'``, ``'str'``, ``'float'``,
        ``'bool'``, ``'category'``, a tuple of categories, or ``None``).
    """

    name: str
    description: Optional[str]
    index: List[str]
    type: Any

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "FeatureSpec":
        return cls(
            name=name,
            description=data.get("_description"),
            index=data.get("_index", []) or [],
            type=data.get("_type"),
        )


@dataclass
class SourceSpec:
    """Parsed metadata for a data source file from ``sources.yaml``.

    Attributes
    ----------
    name : str
        Source-file key (typically the filename relative to the data
        directory).
    authority_for : list[Any]
        Features this source provides.  Each entry is either a plain
        feature name (``str``) or a ``{FeatureName: column_spec}``
        dict describing renames or transforms.
    years : list[str]
        Explicit academic years this source covers.
    year_range : str | None
        A ``'START-END'`` range string (e.g. ``'2018-2024'``) that
        expands to individual years.
    index : Any
        Index column specification from the YAML ``_index`` field.
    sheet : str | None
        Excel sheet name to load, if applicable.
    sheets_are_years : bool
        When ``True``, each sheet in the workbook represents a
        different academic year.
    accessioned : str | None
        ISO date (``YYYY-MM-DD``) the file entered this archive, from
        the ``_accessioned`` field.  Recorded explicitly in the spec
        so provenance survives any storage representation (git, Box,
        plain folder); the source system's own report date, if any,
        is usually part of the filename.
    """

    name: str
    authority_for: List[Any]
    years: List[str]
    year_range: Optional[str]
    index: Any
    sheet: Optional[str]
    sheets_are_years: bool
    accessioned: Optional[str] = None

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "SourceSpec":
        years_val = data.get("_years") or []
        if isinstance(years_val, str):
            years = [years_val]
        else:
            years = list(years_val)
        year_range = data.get("_year_range")
        if not years and not year_range:
            years = [_default_year()]
        accessioned = data.get("_accessioned")
        if accessioned is not None:
            # YAML parses bare YYYY-MM-DD into datetime.date; normalise.
            accessioned = str(accessioned)
        return cls(
            name=name,
            authority_for=data.get("_authority_for", []) or [],
            years=years,
            year_range=year_range,
            index=data.get("_index"),
            sheet=data.get("_sheet"),
            sheets_are_years=bool(data.get("_sheets_are_years")),
            accessioned=accessioned,
        )


def load_feature_specs(path: Path | str | None = None) -> Dict[str, FeatureSpec]:
    """Load feature specs from a single ``features_spec.yaml`` file.

    Parameters
    ----------
    path : Path | str | None
        YAML file to load.  Defaults to the project's configured
        ``features_spec.yaml``.

    Returns
    -------
    dict[str, FeatureSpec]
        Mapping of feature names to their parsed specs.
    """
    spec_path = _resolve_path(path or feature_specs_yaml)
    if spec_path in _FEATURE_SPEC_CACHE:
        return _FEATURE_SPEC_CACHE[spec_path]
    raw = _load_yaml(spec_path)
    specs = {name: FeatureSpec.from_dict(name, data) for name, data in raw.items()}
    _FEATURE_SPEC_CACHE[spec_path] = specs
    return specs


def load_source_specs(path: Path | str | None = None) -> Dict[str, SourceSpec]:
    """Load source specs from a single ``sources.yaml`` file.

    Parameters
    ----------
    path : Path | str | None
        YAML file to load.  Defaults to the project's configured
        ``sources.yaml``.

    Returns
    -------
    dict[str, SourceSpec]
        Mapping of source-file keys to their parsed specs.
    """
    spec_path = _resolve_path(path or sources_yaml)
    if spec_path in _SOURCE_SPEC_CACHE:
        return _SOURCE_SPEC_CACHE[spec_path]
    raw = _load_yaml(spec_path)
    specs = {name: SourceSpec.from_dict(name, data) for name, data in raw.items()}
    _SOURCE_SPEC_CACHE[spec_path] = specs
    return specs


# ---------------------------------------------------------------------------
# Multi-file spec loading (merges specs from multiple YAML files)
# ---------------------------------------------------------------------------

def _ensure_path_tuple(
    paths: Path | str | tuple[Path | str, ...] | None,
    default: Any,
) -> tuple[Path | str, ...]:
    """Normalise *paths* to a tuple so callers can always iterate."""
    if paths is None:
        paths = default
    if isinstance(paths, (str, Path)):
        return (paths,)
    return tuple(paths)


def load_feature_specs_multi(
    paths: Path | str | tuple[Path | str, ...] | None = None,
) -> Dict[str, FeatureSpec]:
    """Load and merge feature specs from one or more YAML files.

    Later files take precedence for duplicate keys.
    Files that do not exist are silently skipped (e.g. Proposed/ may not
    exist yet).
    """
    path_tuple = _ensure_path_tuple(paths, feature_specs_yaml)
    if len(path_tuple) == 1:
        return load_feature_specs(path_tuple[0])
    merged: Dict[str, FeatureSpec] = {}
    for p in path_tuple:
        try:
            merged.update(load_feature_specs(p))
        except FileNotFoundError:
            pass
    return merged


def load_source_specs_multi(
    paths: Path | str | tuple[Path | str, ...] | None = None,
) -> Dict[str, SourceSpec]:
    """Load and merge source specs from one or more YAML files.

    Later files take precedence for duplicate keys.
    Files that do not exist are silently skipped.
    """
    path_tuple = _ensure_path_tuple(paths, sources_yaml)
    if len(path_tuple) == 1:
        return load_source_specs(path_tuple[0])
    merged: Dict[str, SourceSpec] = {}
    for p in path_tuple:
        try:
            merged.update(load_source_specs(p))
        except FileNotFoundError:
            pass
    return merged


def clear_spec_caches() -> None:
    """Invalidate cached specs so newly written YAML files are picked up."""
    _FEATURE_SPEC_CACHE.clear()
    _SOURCE_SPEC_CACHE.clear()
