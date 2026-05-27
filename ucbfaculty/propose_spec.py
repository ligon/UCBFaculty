"""Generate draft YAML spec stanzas for a new data file.

Given a CSV or Excel file, :func:`propose_spec` introspects its columns,
copies the file into a ``Proposed/`` subdirectory of the data directory,
and writes draft ``sources.yaml`` and ``features_spec.yaml`` stanzas
there.  The machine-generated specs are kept separate from the
human-curated originals in the main data directory.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .merge_features import _is_email_column, _is_name_column, _detect_name_format


# ---------------------------------------------------------------------------
# Type-inference helpers
# ---------------------------------------------------------------------------

def _is_boolean_column(series: pd.Series) -> bool:
    """Return True if values are boolean-like (True/False/Yes/No/0/1)."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    bool_values = {"true", "false", "yes", "no", "0", "1", "y", "n"}
    str_vals = non_null.astype(str).str.strip().str.lower()
    return str_vals.isin(bool_values).mean() > 0.8


def _is_integer_column(series: pd.Series) -> bool:
    """Return True if all non-null values parse as integers."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    try:
        numeric = pd.to_numeric(non_null, errors="coerce")
        valid = numeric.dropna()
        if len(valid) == 0:
            return False
        return (valid == valid.astype(int)).all()
    except (ValueError, TypeError):
        return False


def _is_float_column(series: pd.Series) -> bool:
    """Return True if values parse as floats (but not all integers)."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    try:
        numeric = pd.to_numeric(non_null, errors="coerce")
        valid_frac = numeric.notna().mean()
        return valid_frac >= 0.8 and not _is_integer_column(series)
    except (ValueError, TypeError):
        return False


def _is_categorical_column(
    series: pd.Series, max_cardinality_ratio: float = 0.05,
) -> bool:
    """Return True if the column has low cardinality relative to its size."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    n_unique = non_null.nunique()
    if n_unique <= 1:
        return False
    if len(non_null) < 20:
        return n_unique <= 5
    return (n_unique / len(non_null)) < max_cardinality_ratio


def _infer_column_type(series: pd.Series) -> str:
    """Infer a YAML ``_type`` keyword for a pandas Series.

    Returns one of ``'str'``, ``'int'``, ``'float'``, ``'bool'``, or
    ``'category'``.
    """
    if _is_boolean_column(series):
        return "bool"
    if _is_integer_column(series):
        return "int"
    if _is_float_column(series):
        return "float"
    if _is_categorical_column(series):
        return "category"
    return "str"


# ---------------------------------------------------------------------------
# Feature-name helpers
# ---------------------------------------------------------------------------

def _sanitize_feature_name(col_name: str) -> str:
    """Convert a raw column name to PascalCase feature name.

    >>> _sanitize_feature_name("Salary Step")
    'SalaryStep'
    >>> _sanitize_feature_name("employee_class - desc")
    'EmployeeClassDesc'
    """
    parts = re.split(r"[\s_\-]+", col_name.strip())
    return "".join(p.capitalize() for p in parts if p)


# ---------------------------------------------------------------------------
# Column introspection
# ---------------------------------------------------------------------------

def _classify_columns(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Classify every column in *df* by role and inferred type.

    Returns a dict keyed by original column name with values like::

        {'role': 'email_index', 'feature_name': 'Email', 'type': 'str'}
        {'role': 'name', 'feature_name': 'FullName', 'type': 'str',
         'name_format': 'last_first', 'needs_transform': 'reversename'}
        {'role': 'feature', 'feature_name': 'SalaryStep', 'type': 'int'}
    """
    analysis: dict[str, dict[str, Any]] = {}

    # Detect email columns first so we can exclude them from name detection.
    email_cols: list[str] = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if _is_email_column(df[col]):
            email_cols.append(col)

    # Detect FirstName + LastName pair.
    cols_lower = {c.lower(): c for c in df.columns}
    has_split_names = "firstname" in cols_lower and "lastname" in cols_lower

    for col in df.columns:
        series = df[col]
        entry: dict[str, Any] = {"original_name": col}

        if col in email_cols:
            entry["role"] = "email_index"
            entry["feature_name"] = "Email"
            entry["type"] = "str"
        elif has_split_names and col == cols_lower.get("firstname"):
            entry["role"] = "name_part"
            entry["feature_name"] = "FirstName"
            entry["type"] = "str"
        elif has_split_names and col == cols_lower.get("lastname"):
            entry["role"] = "name_part"
            entry["feature_name"] = "LastName"
            entry["type"] = "str"
        elif col not in email_cols and _is_name_column(series):
            fmt = _detect_name_format(series)
            entry["role"] = "name"
            entry["feature_name"] = "FullName"
            entry["type"] = "str"
            entry["name_format"] = fmt
            if fmt == "last_first":
                entry["needs_transform"] = "reversename"
        else:
            entry["role"] = "feature"
            entry["feature_name"] = _sanitize_feature_name(col)
            entry["type"] = _infer_column_type(series)

        analysis[col] = entry

    # Synthesize a virtual FullName entry when split names are detected.
    if has_split_names:
        first_col = cols_lower["firstname"]
        last_col = cols_lower["lastname"]
        analysis[f"({first_col}+{last_col})"] = {
            "original_name": [first_col, last_col],
            "role": "name_composite",
            "feature_name": "FullName",
            "type": "str",
            "needs_transform": "fullname",
            "source_columns": [first_col, last_col],
        }

    return analysis


def _pick_index(analysis: dict[str, dict[str, Any]]) -> str | None:
    """Choose the best index feature name from the column analysis.

    Prefers ``Email``; falls back to ``FullName``.
    """
    for entry in analysis.values():
        if entry.get("role") == "email_index":
            return "Email"
    for entry in analysis.values():
        if entry.get("role") in ("name", "name_composite"):
            return "FullName"
    return None


# ---------------------------------------------------------------------------
# YAML rendering
# ---------------------------------------------------------------------------

def _render_sources_stanza(
    relative_path: str,
    analysis: dict[str, dict[str, Any]],
    index_feature: str | None,
    years: list[str],
    existing_features: set[str],
) -> str:
    """Render a ``sources.yaml`` stanza as a human-readable string."""
    lines = [f'"{relative_path}":']

    # _authority_for
    lines.append("  _authority_for:")
    for col, entry in analysis.items():
        if entry["role"] in ("name_composite",):
            continue  # synthetic entry, not a real column
        feat = entry["feature_name"]
        orig = entry["original_name"]

        if feat in existing_features:
            lines.append(f"    # - {feat}  # already defined elsewhere")
            continue

        if entry.get("needs_transform") == "reversename":
            lines.append(
                f"    - {feat}: !transform [{orig}, ucbfaculty.transformations.reversename]"
            )
        elif orig != feat:
            lines.append(f'    - {feat}: "{orig}"')
        else:
            lines.append(f"    - {feat}")

    # _index
    if index_feature:
        # Find the original column for the index feature.
        for col, entry in analysis.items():
            if entry["feature_name"] == index_feature and entry["role"] != "name_composite":
                orig = entry["original_name"]
                if entry.get("needs_transform") == "reversename":
                    lines.append("  _index:")
                    lines.append(
                        f"    - {index_feature}: !transform [{orig}, ucbfaculty.transformations.reversename]"
                    )
                elif isinstance(orig, str) and orig != index_feature:
                    lines.append("  _index:")
                    lines.append(f'    - {index_feature}: "{orig}"')
                else:
                    lines.append(f"  _index: {index_feature}")
                break
        else:
            # Composite name (FirstName + LastName)
            for col, entry in analysis.items():
                if entry["role"] == "name_composite":
                    src_cols = entry["source_columns"]
                    lines.append("  _index:")
                    lines.append(
                        f"    - FullName: !transform [[{src_cols[0]}, {src_cols[1]}], "
                        f"ucbfaculty.transformations.fullname]"
                    )
                    break

    # _years
    years_str = ", ".join(f'"{y}"' for y in years)
    lines.append(f"  _years: [{years_str}]")

    return "\n".join(lines) + "\n"


def _render_feature_specs_stanza(
    analysis: dict[str, dict[str, Any]],
    index_feature: str | None,
    existing_features: set[str],
) -> str:
    """Render ``features_spec.yaml`` stanzas as a human-readable string."""
    lines: list[str] = []
    seen: set[str] = set()

    for col, entry in analysis.items():
        feat = entry["feature_name"]
        if feat in existing_features or feat in seen:
            continue
        if entry["role"] in ("name_part",):
            continue  # FirstName/LastName handled via FullName composite
        seen.add(feat)

        desc = entry.get("description", feat)
        ftype = entry["type"]
        idx = [index_feature] if index_feature and feat != index_feature else ["Email"]

        # For the index feature itself, the index is the *other* identifier.
        if feat == "Email":
            idx = ["FullName"]
        elif feat == "FullName":
            idx = ["Email"]

        lines.append(f"{feat}:")
        lines.append(f'  _description: "{desc}"')
        lines.append(f"  _type: {ftype}")
        idx_str = "\n".join(f"    - {i}" for i in idx)
        lines.append(f"  _index:\n{idx_str}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def propose_spec(
    filename: str | Path,
    proposed_dir: Path | None = None,
    years: list[str] | None = None,
) -> dict:
    """Read a data file, introspect its columns, and generate draft YAML specs.

    Parameters
    ----------
    filename : str | Path
        Path to a CSV or Excel file (unencrypted) to propose specs for.
    proposed_dir : Path | None
        Directory where proposed specs and the data file copy will be
        placed.  Defaults to ``<data_directory>/Proposed/``.
    years : list[str] | None
        Academic years this source covers.  Defaults to the current year.

    Returns
    -------
    dict
        Keys:

        - ``sources_yaml`` -- generated ``sources.yaml`` text
        - ``features_spec_yaml`` -- generated ``features_spec.yaml`` text
        - ``data_file`` -- :class:`Path` to the copied data file
        - ``columns`` -- column analysis dict
    """
    from . import datadir
    from .rawdata import get_dataframe
    from .feature_spec_loader import (
        _default_year,
        load_feature_specs_multi,
        clear_spec_caches,
    )

    source = Path(filename)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".gpg":
        raise ValueError(
            "propose_spec() expects an unencrypted file. "
            "Decrypt it first, or pass the plain CSV/Excel path."
        )
    if suffix not in (".csv", ".xlsx", ".xls"):
        raise ValueError(
            f"Unsupported file type '{suffix}'. Expected .csv, .xlsx, or .xls."
        )

    # Resolve proposed directory.
    if proposed_dir is None:
        proposed_dir = Path(datadir) / "Proposed"
    proposed_dir.mkdir(parents=True, exist_ok=True)

    # Copy data file.
    dest = proposed_dir / source.name
    shutil.copy2(str(source), str(dest))

    # Load as DataFrame for introspection.
    df = get_dataframe(str(source))
    if isinstance(df, dict):
        # Multi-sheet Excel: use the first sheet for introspection.
        first_sheet = next(iter(df))
        df = df[first_sheet]

    # Determine years.
    if years is None:
        years = [_default_year()]

    # Introspect columns.
    analysis = _classify_columns(df)

    # Check which features are already defined.
    try:
        existing_specs = load_feature_specs_multi()
        existing_features = set(existing_specs.keys())
    except Exception:
        existing_features = set()

    # Determine best index.
    index_feature = _pick_index(analysis)

    # Relative path for the source key (relative to data_directory).
    relative_path = f"Proposed/{source.name}"

    # Render YAML stanzas.
    sources_text = _render_sources_stanza(
        relative_path, analysis, index_feature, years, existing_features,
    )
    specs_text = _render_feature_specs_stanza(
        analysis, index_feature, existing_features,
    )

    # Write (append) to Proposed YAML files.
    _append_yaml(proposed_dir / "sources.yaml", source.name, sources_text)
    _append_yaml(proposed_dir / "features_spec.yaml", source.name, specs_text)

    # Invalidate spec caches so subsequent Feature() calls see the new specs.
    clear_spec_caches()

    return {
        "sources_yaml": sources_text,
        "features_spec_yaml": specs_text,
        "data_file": dest,
        "columns": analysis,
    }


def _append_yaml(path: Path, source_label: str, text: str) -> None:
    """Append *text* to a YAML file, creating it if needed.

    A comment header with the source label is prepended to each block so
    that successive calls produce clearly delimited sections.
    """
    header = f"\n# --- Proposed from: {source_label} ---\n"
    if path.exists():
        with path.open("a", encoding="utf-8") as fh:
            fh.write(header)
            fh.write(text)
            fh.write("\n")
    else:
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"# Auto-generated proposed specs\n")
            fh.write(header)
            fh.write(text)
            fh.write("\n")
