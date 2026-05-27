import pandas as pd
import numpy as np
from typing import List, Optional

try:
    from .faculty import FacultyMember
except ImportError:  # pragma: no cover
    FacultyMember = None  # type: ignore


def name_variants(name: str) -> List[str]:
    """Generate spelling variants of a full name for fuzzy matching.

    Produces title-case, lower-case, period-stripped, and
    middle-initial-removed variants.

    Parameters
    ----------
    name : str
        A full name string.

    Returns
    -------
    list[str]
        Unique name variants (order preserved).
    """
    if not isinstance(name, str):
        return []

    stripped = name.strip()
    variants = [stripped]

    title_case = stripped.title()
    lower_case = stripped.lower()
    if title_case not in variants:
        variants.append(title_case)
    if lower_case not in variants:
        variants.append(lower_case)

    tokens = stripped.split()
    if tokens:
        periodless = " ".join(token.replace(".", "") for token in tokens if token)
        if periodless and periodless not in variants:
            variants.append(periodless)

    if len(tokens) > 2:
        filtered = []
        changes = False
        for token in tokens:
            cleaned = token.replace(".", "")
            if len(cleaned) == 1:
                changes = True
                continue
            filtered.append(token)
        if changes:
            filtered_name = " ".join(filtered).strip()
            if filtered_name and filtered_name not in variants:
                variants.append(filtered_name)

    return list(dict.fromkeys(variant for variant in variants if variant))


# Backward-compatible alias
_name_variants = name_variants


def yes_response(prompt, default=None):
    """Prompt the user with a yes/no question and return the answer.

    Parameters
    ----------
    prompt : str
        The question text shown to the user.
    default : str | None
        Default answer (``'y'`` or ``'n'``) used when the user
        presses Enter without typing.

    Returns
    -------
    bool
        ``True`` if the user answered yes, ``False`` otherwise.
    """
    options = {"y", "n", ""}
    while (res := input(prompt + " [Y/n]: ").lower().strip()) not in options:
        pass

    if len(res) == 0 and default is not None:
        res = default.lower()

    return res == 'y'


def ensure_unique_index(series):
    """
    Convert a pandas Series with potentially non-unique index to one with unique index.
    If the index is already unique, returns the series unchanged.
    If the index has duplicates, groups values into lists by index label.

    Parameters:
    -----------
    series : pd.Series
        Input series with potentially non-unique index

    Returns:
    --------
    pd.Series
        Series with unique index. If grouping occurred, dtype will be 'object'
        containing lists of values.
    """
    if series.index.is_unique:
        return series

    def _flatten(values):
        unique = []
        for value in values:
            if isinstance(value, (list, tuple, pd.Series, np.ndarray)):
                for item in value:
                    if pd.isna(item):
                        continue
                    if item not in unique:
                        unique.append(item)
            else:
                if pd.isna(value):
                    continue
                if value not in unique:
                    unique.append(value)
        if len(unique) == 1:
            return unique[0]
        return unique

    if isinstance(series.index, pd.MultiIndex):
        level = list(range(series.index.nlevels))
    else:
        level = 0

    grouped = series.groupby(level=level, sort=False).apply(_flatten)
    grouped.index.names = series.index.names
    return grouped


def faculties_to_df(members: List["FacultyMember"], extra_cols: Optional[dict] = None) -> pd.DataFrame:
    """Convert a list of :class:`FacultyMember` objects to a DataFrame.

    Parameters
    ----------
    members : list[FacultyMember]
        Faculty member instances to serialize.
    extra_cols : dict[str, dict] | None
        Optional mapping of ``Email`` (or ``FullName`` / ``LastName``)
        to a dict of additional column values to attach to each row.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by ``Email`` with columns ``FullName``,
        ``FirstName``, ``LastName``, ``DeptName``, ``UID``, plus
        any extras.
    """
    rows = []
    extras = extra_cols or {}
    for m in members:
        data = {
            "Email": getattr(m, "Email", None),
            "FullName": getattr(m, "FullName", None),
            "FirstName": getattr(m, "FirstName", None),
            "LastName": getattr(m, "LastName", None),
            "DeptName": getattr(m, "DeptName", None),
            "UID": getattr(m, "UID", None),
        }
        extra_data = None
        for key in (data.get("Email"), data.get("FullName"), data.get("LastName")):
            if key in extras:
                extra_data = extras[key]
                break
        if extra_data:
            data.update(extra_data)
        rows.append(data)
    df = pd.DataFrame(rows)
    if "Email" in df.columns:
        df = df.set_index("Email")
    return df


def df_to_faculties(df: pd.DataFrame) -> List["FacultyMember"]:
    """Convert a DataFrame to a list of :class:`FacultyMember` objects.

    Each row is looked up via :class:`FacultyMember`; non-core columns
    are attached as extra attributes on the resulting instances.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with an ``Email`` column or index.

    Returns
    -------
    list[FacultyMember]
        One :class:`FacultyMember` per row.
    """
    global FacultyMember
    if FacultyMember is None:  # late import to avoid circular import issues
        try:  # pragma: no cover - exercised indirectly
            from .faculty import FacultyMember as _FM
            FacultyMember = _FM
        except ImportError:
            return []

    work = df.copy()
    if "Email" not in work.columns and work.index.name == "Email":
        work = work.reset_index()

    members: List[FacultyMember] = []
    for _, row in work.iterrows():
        # Try email first, otherwise pass name fields for lookup
        email = row.get("Email")
        kwargs = {}
        if email is not None and not pd.isna(email):
            kwargs["Email"] = email
        else:
            for key in ("FullName", "FirstName", "LastName"):
                if key in row and not pd.isna(row[key]):
                    kwargs[key] = row[key]
        fm = FacultyMember(**kwargs)
        if fm.match_info.get("method") != "direct":
            fm.verified = False
        for col, val in row.items():
            if col in {"Email", "FullName", "FirstName", "LastName", "DeptName", "UID"}:
                # Skip core attrs that FacultyMember already populates.
                continue
            setattr(fm, col, val)
        members.append(fm)
    return members
