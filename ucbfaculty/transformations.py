from typing import Callable, Dict, Any, List
import pandas as pd
from .spelling import dictionary, inverse_d

def mask_missing(mask_condition=None):
    """
    Decorator that automatically masks rows/elements with missing values
    before applying the function, then reconstructs the full result.
    """
    def decorator(func):
        def wrapper(data):
            # Check if this is a single row/element (called via apply)
            # vs the entire DataFrame/Series (called directly)
            if isinstance(data, pd.Series):
                # If it has a MultiIndex or looks like a single row from a DataFrame
                if hasattr(data, 'name') and isinstance(data.name, (int, str, type(None))):
                    # This is likely a single row - check for missing values
                    if data.isna().any():
                        return pd.NA
                    else:
                        if data.shape[0]==1:
                            return func(data.values[0])
                        else:
                            return func(data)
                # Otherwise treat as full Series
            elif not isinstance(data, pd.DataFrame):
                # Single element - check if missing
                if pd.isna(data):
                    return pd.NA
                else:
                    return func(data)

            # Logic for full DataFrame/Series
            if mask_condition is None:
                if isinstance(data, pd.DataFrame):
                    valid_mask = data.notna().all(axis=1)
                else:  # Series
                    valid_mask = data.notna()
            else:
                valid_mask = mask_condition(data)

            if isinstance(data, pd.DataFrame):
                valid_result = data[valid_mask].apply(func, axis=1)
            else:  # Series
                valid_result = data[valid_mask].apply(func)

            return valid_result.reindex(data.index, fill_value=pd.NA)

        return wrapper
    return decorator


@mask_missing()
def lower(s):
    """Cast a string value to lower case.

    Parameters
    ----------
    s : str
        Input value.

    Returns
    -------
    str
        Lower-cased string, or *s* unchanged on ``AttributeError``.
    """
    try:
        return s.lower()
    except AttributeError:
        return s

@mask_missing()
def fullname(x):
    """Construct a ``FullName`` from separate name fields.

    Designed for use with ``DataFrame.apply(fullname, axis=1)`` where
    each row contains ``FirstName`` and ``LastName`` columns.

    Parameters
    ----------
    x : pandas.Series
        A row (or dict-like) with ``'FirstName'`` and ``'LastName'``
        keys.

    Returns
    -------
    str | pandas.NA
        Title-cased ``'First Last'`` string, or ``pd.NA`` when both
        parts are empty.
    """
    try:
        first = str(x.get('FirstName', '') or '').strip()
        last = str(x.get('LastName', '') or '').strip()
        parts = [p for p in (first, last) if p]
        joined = " ".join(parts)
        if not joined:
            return pd.NA
        return joined.title()
    except (AttributeError, TypeError, KeyError):
        return None

@mask_missing()
def reversename(s):
    """Recompose a ``'Last, First'`` name as ``'First Last'``.

    Parameters
    ----------
    s : str
        A name string, possibly in ``'Last, First'`` format.

    Returns
    -------
    str
        ``'First Last'`` if a comma was found, otherwise *s*
        unchanged.
    """
    try:
        bits = [t.strip() for t in s.split(',')]
        if len(bits) < 2:
            return s
        return f"{bits[1]} {bits[0]}"
    except AttributeError:
        return s

@mask_missing()
def to_boolean(s):
    """Map common truthy/falsy representations to Python booleans.

    Parameters
    ----------
    s : str | int | bool
        Input value (e.g. ``'Yes'``, ``'no'``, ``1``, ``'FALSE'``).

    Returns
    -------
    bool | Any
        ``True`` or ``False`` when recognised, otherwise *s*
        unchanged.
    """
    if s in [True,"True","Yes",'yes','true',1,'1','YES','TRUE']:
        return True
    elif s in [False,'False','No','no','false',0,'0','NO','FALSE']:
        return False
    else:
        return s

# Example
@mask_missing()
def join_with_sep(first: pd.Series, last: pd.Series, sep: str = ' ') -> pd.Series:
    """Join two string columns with a separator.

    Parameters
    ----------
    first : pandas.Series
        Left-hand values.
    last : pandas.Series
        Right-hand values.
    sep : str
        Separator (default ``' '``).

    Returns
    -------
    pandas.Series
        Concatenated string Series.
    """
    return first.astype(str) + sep + last.astype(str)

@mask_missing()
def upper(series: pd.Series) -> pd.Series:
    """Convert a string value or Series to uppercase.

    Parameters
    ----------
    series : pandas.Series | str
        Input values.

    Returns
    -------
    pandas.Series | str
        Upper-cased result.
    """
    return series.str.upper()


@mask_missing()
def title(series: pd.Series) -> pd.Series:
    """Title-case a string value or Series.

    Parameters
    ----------
    series : pandas.Series | str
        Input values.

    Returns
    -------
    pandas.Series | str
        Title-cased result.
    """
    if isinstance(series, pd.Series):
        return series.astype(str).str.title()
    try:
        return str(series).title()
    except (TypeError, AttributeError):
        return series


@mask_missing()
def correct_spelling(value):
    """Normalise spellings through the project spelling dictionary.

    Parameters
    ----------
    value : str | pandas.Series
        A single string or Series of strings.

    Returns
    -------
    str | pandas.Series
        Canonically-spelled value(s).
    """
    if isinstance(value, pd.Series):
        return value.map(lambda v: dictionary.spell(v) if isinstance(v, str) else v)
    if isinstance(value, str):
        return dictionary.spell(value)
    return value


@mask_missing()
def employment_category(value):
    """Map UCPath employment status codes to broader categories.

    Parameters
    ----------
    value : str | list[str]
        One or more employment status strings (e.g.
        ``'9 - Academic: Faculty'``).

    Returns
    -------
    str | list[str]
        Mapped category (e.g. ``'Active Faculty'``,
        ``'Emeritus (Retired)'``).  A list is returned when
        *value* is a list and multiple distinct categories remain.
    """
    mapping = {
        '21 - Academic: Emeriti': 'Emeritus (Retired)',
        '3 - Academic: Recall': 'Emeritus (Recalled)',
        '9 - Academic: Faculty': 'Active Faculty',
        '22 - Academic: Deans/Faculty Admin': 'Deans/Admin',
    }

    def _map_single(item):
        if not isinstance(item, str):
            return item
        return mapping.get(item, item)

    if isinstance(value, list):
        categories = {_map_single(v) for v in value if v is not None}
        categories.discard(None)
        if len(categories) == 1:
            return categories.pop()
        return sorted(categories)

    return _map_single(value)


# --- predict_sex --------------------------------------------------------
#
# INTERNAL/ANALYTIC ONLY -- intended for DIVCO reporting in the
# FacultyExplorer workflow.  DO NOT use the predicted ``Sex`` feature
# for committee selection (Title VII / Prop 209).
#
# This transformation is deliberately a *soft* dependency on the
# ``gender-guesser`` package (Joerg Michael's name dataset).  The
# import is lazy and failure-tolerant: if ``gender_guesser`` is not
# installed in the active venv, ``predict_sex`` returns ``pd.NA``
# rather than raising, so the surrounding feature-loader keeps
# working and any analysis simply sees a higher unknown rate.
#
# To enable predictions, add ``gender-guesser`` to the *consuming*
# repo's pyproject (e.g. FacultyExplorer), not to UCBFaculty's deps.

_SEX_DETECTOR = None


def _get_sex_detector():
    """Lazily build a singleton ``gender_guesser`` detector.

    Returns ``None`` if the package is unavailable, so callers can
    degrade gracefully to ``pd.NA``.
    """
    global _SEX_DETECTOR
    if _SEX_DETECTOR is None:
        try:
            import gender_guesser.detector as _gg
            _SEX_DETECTOR = _gg.Detector(case_sensitive=False)
        except ImportError:
            _SEX_DETECTOR = False  # sentinel for "tried and failed"
    return _SEX_DETECTOR or None


@mask_missing()
def predict_sex(first_name):
    """Predict sex from a first name via gender-guesser.

    Parameters
    ----------
    first_name : str
        Faculty first name.  May contain compound forms
        (``"Mary Ann"``, ``"Jean-Pierre"``); only the leading token
        is used because the underlying dataset is keyed on single
        first names.

    Returns
    -------
    str | pandas.NA
        ``"F"``, ``"M"``, or ``"unknown"``.  ``mostly_female`` /
        ``mostly_male`` collapse to ``"F"`` / ``"M"``; ``andy``
        (androgynous) and ``unknown`` collapse to ``"unknown"``.
        Returns ``pd.NA`` when the input is empty or
        ``gender_guesser`` is not installed.

    Notes
    -----
    Accuracy is high for Anglo first names and degrades on East
    Asian, many South Asian, and many African names.  Downstream
    analyses should report unknown rates by Division to expose the
    asymmetry rather than treat ``"unknown"`` as missing-at-random.
    """
    if not isinstance(first_name, str) or not first_name.strip():
        return pd.NA
    detector = _get_sex_detector()
    if detector is None:
        return pd.NA
    token = first_name.strip().split()[0]
    raw = detector.get_gender(token)
    if raw in ("male", "mostly_male"):
        return "M"
    if raw in ("female", "mostly_female"):
        return "F"
    return "unknown"


@mask_missing()
def faculty_rank(value):
    """Map UCPath ``Job Code Desc`` to academic rank.

    Parameters
    ----------
    value : str
        UCPath job code description, e.g. ``'PROF-AY'``,
        ``'ASSOC PROF-AY-B/E/E'``, ``'ASST PROF OF TEACH-AY'``,
        ``'ACT ASSOC PROF-AY'``, ``'PROF EMERITUS(WOS)'``.

    Returns
    -------
    str
        One of ``'Full'``, ``'Associate'``, ``'Assistant'``,
        ``'Emeritus'``, or ``'Other'``.

    Notes
    -----
    Recognises the common UCB UCPath prefixes:

    - ``ASST`` -> Assistant.
    - ``ASSOC`` / ``ACT ASSOC`` -> Associate (except admin titles
      like ``ASSOC DEAN``, which fall through to ``'Other'``).
    - ``PROF`` / ``ACT PROF`` / ``UNIV PROF`` -> Full.
    - Anything containing ``EMERITUS`` or starting with ``RECALL``
      -> ``'Emeritus'`` (also flagged separately by
      ``EmploymentCategory``).
    """
    if not isinstance(value, str):
        return pd.NA
    v = value.upper().strip()
    if "EMERITUS" in v or v.startswith("RECALL") or v.startswith("SENATE EMERITUS"):
        return "Emeritus"
    if v.startswith("ASST"):
        return "Assistant"
    if v.startswith("ASSOC DEAN"):
        return "Other"
    if v.startswith("ASSOC") or v.startswith("ACT ASSOC"):
        return "Associate"
    if v.startswith("PROF") or v.startswith("ACT PROF") or v.startswith("UNIV PROF"):
        return "Full"
    return "Other"


@mask_missing()
def normalize_academic_year(value):
    """Normalize an academic year value to the canonical ``'YYYY-YY'`` form.

    Parameters
    ----------
    value : str | int | float
        An academic year expressed as a start year (``2023``), a
        full string (``'2023-2024'``, ``'2023-24'``, ``'AY 2023-24'``),
        or similar variant.

    Returns
    -------
    str
        Normalised string in ``'YYYY-YY'`` format (e.g. ``'2023-24'``).
        Returns *value* unchanged when it cannot be parsed.
    """
    # Numeric input: treat as start year
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            start = int(value)
            end = start + 1
            return f"{start}-{str(end)[-2:]}"
        except (ValueError, TypeError, OverflowError):
            return value
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if len(stripped) == 7 and stripped[:4].isdigit() and stripped[4] == '-' and stripped[5:].isdigit():
        return stripped

    digits = ''.join(ch for ch in stripped if ch.isdigit())
    if len(digits) >= 6:
        start = digits[:4]
        if not start.isdigit():
            return stripped
        if len(digits) >= 8:
            end_full = digits[4:8]
        else:
            end_full = digits[4:6]

        if end_full.isdigit():
            end_two = end_full[-2:]
            return f"{start}-{end_two}"

    if stripped[:4].isdigit() and stripped[-2:].isdigit():
        return f"{stripped[:4]}-{stripped[-2:]}"

    return stripped
