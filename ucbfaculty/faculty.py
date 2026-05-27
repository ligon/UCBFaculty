from functools import cache, cached_property
from collections import defaultdict
import logging
import re
import warnings
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)

import pandas as pd
from ligonlibrary.strings import similar
from .merge_features import build_feature_dataframe, merge_features
from .utils import faculties_to_df, name_variants as _name_variants
from .rawdata import get_dataframe, _gpg_decrypt
from .transformations import normalize_academic_year, to_boolean
from . import datadir, spelling, FACULTY_NAME_DICTIONARY, current_year, set_current_year

VOLUNTEER_WORKBOOK = "List of Volunteers.xlsx.gpg"


@cache
def _volunteer_years_for_path(path: Path) -> list[str]:
    """List sheet names for a volunteer workbook path (supports encrypted files)."""
    try:
        if path.suffix.lower() in {".gpg", ".pgp"}:
            decrypted = _gpg_decrypt(path)
            xls = pd.ExcelFile(BytesIO(decrypted))
        else:
            xls = pd.ExcelFile(path)
        return xls.sheet_names
    except Exception as exc:
        logger.debug("Failed to read volunteer years from %s: %s", path, exc)
        return []


def _guess_year_from_clock():
    """Best-effort guess of the academic year from the current date."""
    import datetime as _dt
    today = _dt.date.today()
    year = today.year
    # Academic year starts in August
    if today.month >= 8:
        start = year
    else:
        start = year - 1
    end_two = (start + 1) % 100
    return f"{start}-{end_two:02d}"


def _year_start_from_academic_year(value):
    """Extract the starting calendar year from an academic year string."""
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (ValueError, TypeError, OverflowError):
            return None
    if not isinstance(value, str):
        return None
    match = re.search(r'(19|20|21)\d{2}', value)
    if match:
        try:
            return int(match.group(0))
        except (ValueError, TypeError):
            return None
    return None


def get_current_year():
    """Resolve the current academic year string.

    Checks several sources in priority order:

    1. Config variable set via :func:`set_current_year`.
    2. Latest year in the ``Appointment`` feature spec.
    3. Latest ``YearStart`` in historical committee service records.
    4. Clock-based guess from today's date.

    Returns
    -------
    str
        Academic year in ``'YYYY-YY'`` format (e.g. ``'2025-26'``).
    """
    if current_year:
        return current_year

    # Try Appointment feature
    try:
        from .new_feature import Feature
        feat = Feature("Appointment")
        years = feat.available_years()
        if years:
            resolved = sorted(years)[-1]
            set_current_year(resolved)
            return resolved
    except Exception as exc:
        logger.debug("Could not resolve current year from Appointment: %s", exc)

    # Try historical appointments
    try:
        df = _committee_service_records()
        if not df.empty and "YearStart" in df.columns:
            max_start = df["YearStart"].max()
            if pd.notna(max_start):
                end_two = (int(max_start) + 1) % 100
                set_current_year(f"{int(max_start)}-{end_two:02d}")
                return current_year
    except Exception as exc:
        logger.debug("Could not resolve current year from service records: %s", exc)

    set_current_year(_guess_year_from_clock())
    return current_year


def _set_current_year_normalized(value: str):
    """Set the current academic year explicitly, normalizing the value first."""
    set_current_year(normalize_academic_year(value))

def df_filtered(df_fn,index_by=None,correct_column_spellings=True,**kwargs):
    """
    Load a DataFrame and apply optional indexing/filters.

    Parameters:
    - df_fn: str
        Ignored; retained for backward compatibility.
    - index_by: str, optional
        Column name to set as the DataFrame's index.
    - **kwargs: selection criteria

    Returns:
    - df: pandas.DataFrame
        A DataFrame with optional re-indexing and column spelling corrections applied.
    """
    df = _df_base_cached(index_by=index_by or 'Email').reset_index()

    if len(kwargs):
        realkw = {k:v for k,v in kwargs.items() if v is not None}
        for k,v in realkw.items():
            column = k
            if column not in df.columns:
                raise ValueError(f"Unknown keyword {k}.\nValid codes:\n{df.columns}")
            if isinstance(v,(list,tuple)):
                use = df[column].isin(v)
            elif callable(v): # Boolean functions
                use = df[column].apply(v)
            else:
                use = df[column]==v

            df = df.loc[use,:]

    if index_by is not None:
        df = df.set_index(index_by)

    if not df.index.is_unique:
        warnings.warn(f"Non-unique index in {df_fn}")

    return df


@cache
def _df_base_cached(index_by='Email'):
    """Cached base load for faculty_vitals without filters."""
    feature_names = [
        "Email",
        "FullName",
        "FirstName",
        "LastName",
        "UID",
        "DeptName",
        "DeptCode",
        "Division",
        "EmploymentCategory",
        "Affiliations",
    ]
    df = build_feature_dataframe(feature_names, interactive=False)
    if df.index.name in df.columns:
        df = df.drop(columns=[df.index.name])
    if index_by != 'Email':
        df = df.reset_index().set_index(index_by)
    return df


def faculty_vitals(df_fn='AS_list.csv.gpg', index_by='Email', **kwargs):
    """Load core demographic data for Berkeley faculty.

    Returns a DataFrame of "vital" columns (name, department, UID, etc.)
    for all current Senate faculty, optionally filtered by keyword
    arguments that correspond to column values.

    Parameters
    ----------
    df_fn : str
        Ignored; retained for backward compatibility.
    index_by : str
        Column name to use as the DataFrame index (default ``'Email'``).
    **kwargs
        Column-name / value pairs used to filter the result.  Each key
        must match a column in the base DataFrame.  Values may be a
        scalar (exact match), a list/tuple (membership test), or a
        callable that returns a boolean Series.

    Returns
    -------
    pandas.DataFrame
        Faculty vitals indexed by *index_by*.  Columns include
        ``FullName``, ``FirstName``, ``LastName``, ``UID``,
        ``DeptName``, ``DeptCode``, ``Division``,
        ``EmploymentCategory``, and ``Affiliations``.
    """
    if not kwargs:
        fac = _df_base_cached(index_by=index_by)
    else:
        fac = df_filtered(df_fn=df_fn,index_by=index_by,**kwargs)

    fac = fac.reset_index().set_index(index_by)

    if not fac.index.is_unique:
        warnings.warn(f"Non-unique index in {df_fn}")

    return fac

def division():
    """Return Division for each faculty member, indexed by Email."""
    from .new_feature import Feature
    return Feature("Division")["latest"]

def full_name(df):
    fac=faculty_vitals()
    full_name = df[['FirstName','LastName']].astype(str).apply(' '.join,axis=1)

    regularizer = find_similar_pairs(full_name,fac.FullName,similarity_threshold=68)
    return full_name.replace(regularizer)

def _volunteer_years(data_fn: str = VOLUNTEER_WORKBOOK) -> list[str]:
    """List available volunteer years (sheet names) in the workbook."""
    return _volunteer_years_for_path(datadir / data_fn)


@cache
def _cached_volunteers_feature(year: object, data_fn: str, data_root: str) -> pd.DataFrame:
    """Cached loader for volunteer feature to avoid repeated decrypt/parse."""
    return _load_volunteers_feature(year, data_fn=data_fn, interactive=False)


_VOLUNTEER_FEATURES = [
    "AnyCommittee", "Choice1", "Choice2", "Choice3",
    "Continue", "Interests",
    "ChoicesReason", "OnLeave",
]


def _load_volunteers_feature(year: object, data_fn: str = VOLUNTEER_WORKBOOK, interactive: bool = False) -> pd.DataFrame:
    """Load volunteer data for a given year, indexed by Email."""
    from .new_feature import Features, Feature

    try:
        df = Features(_VOLUNTEER_FEATURES)[year]
    except Exception as exc:
        warnings.warn(f"Unable to load volunteer features: {exc}")
        return pd.DataFrame()

    # Drop Year index level — callers add VolunteerYear separately
    if isinstance(df.index, pd.MultiIndex) and "Year" in df.index.names:
        df = df.droplevel("Year")

    # Deduplicate (merge can produce cartesian products on non-unique index)
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="first")]

    # Add FullName explicitly if missing
    try:
        fullname = Feature("FullName")["latest"]
        if "FullName" not in df.columns:
            df["FullName"] = df.index.map(fullname.to_dict())
    except Exception:
        pass

    if isinstance(year, str):
        df["AcademicYear"] = normalize_academic_year(year)

    return df


def volunteers(year=None, data_fn: str = VOLUNTEER_WORKBOOK):
    """Return volunteer survey data for the given year.

    Args:
        year: Academic year (e.g., "2024-25"), "all" for all years, or None for current year.
        data_fn: Deprecated, ignored. Volunteer data is loaded via Feature specs.

    Returns:
        DataFrame indexed by Email with volunteer survey columns and VolunteerYear.
    """
    from .new_feature import Features

    if year is None:
        year = get_current_year()

    features = Features(_VOLUNTEER_FEATURES)

    if isinstance(year, str) and year.lower() == "all":
        # Get full history
        try:
            df = features["history"]
        except (KeyError, ValueError):
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Add VolunteerYear from Year index level
        df = df.reset_index()
        if "Year" in df.columns:
            df["VolunteerYear"] = df["Year"].map(normalize_academic_year)
            df = df.set_index("Email")
        return df

    # Single year
    try:
        df = features[year]
    except (KeyError, ValueError):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    # Handle MultiIndex (Year, Email) from Features
    if isinstance(df.index, pd.MultiIndex) and "Year" in df.index.names:
        df = df.reset_index(level="Year", drop=True)

    df["VolunteerYear"] = normalize_academic_year(year)
    return df


def _parse_interests(interests_str: str) -> list[str]:
    """Split comma-delimited interests string into list of categories."""
    if not isinstance(interests_str, str) or not interests_str:
        return []
    return [cat.strip() for cat in interests_str.split(", ") if cat.strip()]


def interest_categories(year=None) -> list[str]:
    """Return sorted list of unique interest categories.

    Args:
        year: Academic year or None for all years combined.

    Returns:
        Sorted list of unique interest category strings.
    """
    from .new_feature import Feature

    if year is None:
        interests = Feature("Interests")["history"]
    else:
        interests = Feature("Interests")[year]

    categories = set()
    for val in interests.dropna():
        if isinstance(val, str):
            for cat in _parse_interests(val):
                categories.add(cat)

    return sorted(categories)


def volunteers_with_interest(interest: str, year=None, as_df: bool = True):
    """Return volunteers who expressed a specific interest.

    Args:
        interest: Interest category to filter by (e.g., "Art Museums").
                  Must match exactly (case-insensitive) one of the split categories.
        year: Academic year, "all" for history, or None for current year.
        as_df: If True return DataFrame, else list of emails.

    Returns:
        DataFrame of matching volunteers, or list of email addresses.
    """
    from .new_feature import Features

    if year is None:
        year = get_current_year()

    features = Features(_VOLUNTEER_FEATURES)

    if isinstance(year, str) and year.lower() == "all":
        df = features["history"].reset_index()
    else:
        df = features[year].reset_index()

    if df.empty or "Interests" not in df.columns:
        return pd.DataFrame() if as_df else []

    # Filter rows where interest appears in the parsed list (case-insensitive)
    interest_lower = interest.lower()
    mask = df["Interests"].apply(
        lambda x: interest_lower in [c.lower() for c in _parse_interests(x)]
    )
    result = df.loc[mask]

    if as_df:
        if "Email" in result.columns:
            result = result.set_index("Email")
        return result
    else:
        return result["Email"].tolist() if "Email" in result.columns else []


def any_committee(year=None):
    """Return list of emails of faculty who volunteered for any committee."""
    from .new_feature import Feature

    if year is None:
        year = get_current_year()

    try:
        any_comm = Feature("AnyCommittee")[year]
    except (KeyError, ValueError):
        return []

    if any_comm.empty:
        return []

    # Filter to True values and get emails
    volunteers_true = any_comm[any_comm == True]
    if volunteers_true.empty:
        return []

    # Extract Email from index (may be MultiIndex or simple Index)
    if isinstance(volunteers_true.index, pd.MultiIndex) and "Email" in volunteers_true.index.names:
        return volunteers_true.index.get_level_values("Email").tolist()
    return volunteers_true.index.tolist()


def unexploited_volunteers(year=None, as_df=False):
    """Return faculty who volunteered for any committee but were not appointed.

    Args:
        year: Academic year (e.g., "2024-25"). Defaults to current year.
        as_df: If True, return DataFrame with FullName and DeptName.

    Returns:
        List of emails, or DataFrame if as_df=True.
    """
    from .new_feature import Feature

    if year is None:
        year = get_current_year()

    # Get volunteers who said AnyCommittee=True
    volunteer_emails = set(any_committee(year=year))
    if not volunteer_emails:
        return pd.DataFrame(columns=["FullName", "DeptName"]) if as_df else []

    # Get appointments for the year
    try:
        appt = Feature("Appointment")[year]
    except (KeyError, ValueError):
        appt = pd.Series(dtype=object)

    # Get emails with appointments
    if not appt.empty:
        if isinstance(appt.index, pd.MultiIndex) and "Email" in appt.index.names:
            appointed_emails = set(appt.index.get_level_values("Email"))
        else:
            appointed_emails = set(appt.index)
    else:
        appointed_emails = set()

    # Unexploited = volunteered but not appointed
    unexploited = volunteer_emails - appointed_emails

    if not as_df:
        return sorted(unexploited)

    # Return DataFrame with faculty info
    if not unexploited:
        return pd.DataFrame(columns=["FullName", "DeptName"])

    try:
        fullname = Feature("FullName")["latest"]
        deptname = Feature("DeptName")["latest"]
        df = pd.DataFrame({"FullName": fullname, "DeptName": deptname})
        df = df.loc[df.index.isin(unexploited)]
        return df.sort_values("FullName")
    except Exception:
        return pd.DataFrame({"Email": sorted(unexploited)})


def normalize_year_input(value):
    """
    Normalize a year input (int or str) into (start_year:int|None, academic_year:str|None).

    Examples:
      2024 -> (2024, "2024-25")
      "2024" -> (2024, "2024-25")
      "2024-25" -> (2024, "2024-25")
    """
    if value is None:
        return None, None
    if isinstance(value, int):
        start = value
        end_two = (start + 1) % 100
        return start, f"{start}-{end_two:02d}"
    if isinstance(value, str):
        norm = normalize_academic_year(value)
        if isinstance(norm, str) and len(norm) >= 4 and norm[:4].isdigit():
            start = int(norm[:4])
            end_two = (start + 1) % 100
            return start, f"{start}-{end_two:02d}"
    return None, None

class FacultyMember:
    # Core attributes typically populated on init, but now extensible via __getattr__
    attrs = ['FirstName', 'LastName', 'FullName', 'Email', 'UID', 'DeptName', 'DeptCode', 'Affiliations']

    def __init__(self, options=False, **kwargs):
        """Look up a faculty member and populate core attributes.

        On construction the class queries :func:`faculty_vitals` with
        the supplied keyword arguments.  If exactly one match is found
        the instance is marked as *verified* and all core attributes
        are populated from the match.

        Parameters
        ----------
        options : bool
            When ``True`` and multiple matches are found, print the
            candidates to stdout for interactive disambiguation.
        **kwargs
            Lookup keys corresponding to columns in the faculty
            vitals DataFrame — typically ``Email``, ``FullName``,
            ``FirstName``, ``LastName``, ``UID``, ``DeptName``,
            ``DeptCode``, or ``Affiliations``.

        Attributes
        ----------
        verified : bool
            ``True`` when exactly one record matched the query.
        match_info : dict
            Diagnostic dict with keys ``method`` (``'direct'``,
            ``'ambiguous'``, or ``'none'``) and ``candidates`` (int).
        """
        self.verified = False
        self.match_info = {}
        self._cache = {}  # Cache for dynamically loaded features

        # Identify keys that match core attributes to drive the initial search
        assign_keys = list(set(kwargs.keys()).intersection(FacultyMember.attrs))
        kv_pairs = {k: kwargs[k] for k in assign_keys}

        # Perform lookup using the legacy faculty_vitals helper
        # This returns a DataFrame of matches
        fac = faculty_vitals(**kv_pairs)

        if fac.shape[0] == 1:
            # Exact match found
            kv_pairs = fac.reset_index().T.to_dict()[0]
            self.verified = True
            self.match_info = {"method": "direct", "candidates": 1}
        elif options and fac.shape[0] > 1:
            # Multiple matches
            warnings.warn("Multiple matches; trying narrowing")
            print(fac)
            self.match_info = {"method": "ambiguous", "candidates": fac.shape[0]}
        else:
            # No match or ambiguous
            self.match_info = {"method": "ambiguous" if fac.shape[0] > 1 else "none", "candidates": fac.shape[0]}

        # Populate attributes from the lookup result
        for attr in FacultyMember.attrs:
            if attr in kv_pairs.keys():
                setattr(self, attr, kv_pairs[attr])
            else:
                setattr(self, attr, None)
        
        # Populate any other kwargs passed that aren't core attrs (e.g. from df_to_faculties)
        for k, v in kwargs.items():
            if k not in FacultyMember.attrs:
                setattr(self, k, v)

    def __getattr__(self, name):
        """
        Dynamic access to any available feature.
        
        If 'name' corresponds to a known feature in the system, fetch its latest value
        for this faculty member.
        """
        # Avoid infinite recursion for internal attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if the requested name is a valid feature
        from .new_feature import available_features, Feature
        
        if name in available_features():
            if name in self._cache:
                return self._cache[name]
                
            if not self.Email:
                return None
                
            try:
                # Fetch the 'latest' value for this feature
                series = Feature(name)["latest"]
                val = series.get(self.Email)
                
                # If series is MultiIndex (e.g. Year, Email), try to get the latest year entry
                if val is None and isinstance(series.index, pd.MultiIndex) and "Email" in series.index.names:
                    try:
                        subset = series.xs(self.Email, level="Email")
                        if not subset.empty:
                            # Assuming standard sorting, take the last one
                            val = subset.iloc[-1]
                    except KeyError:
                        pass

                self._cache[name] = val
                return val
            except Exception as e:
                logger.debug(f"Failed to load dynamic feature '{name}' for {self.Email}: {e}")
                return None
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def attributes(self):
        """Return a dictionary of all core attributes plus any cached dynamic features."""
        data = {attr: getattr(self, attr) for attr in FacultyMember.attrs}
        data.update(self._cache)
        return data

    def __repr__(self):
        return (f"FacultyMember(UID={self.UID}, "
                f"Full Name={self.FullName}, "
                f"Email={self.Email}, "
                f"Department={self.DeptName})")

    def report(self, format="text"):
        """
        Generate a comprehensive report of the faculty member.

        Args:
            format: 'text', 'markdown', 'org', or 'dict'.
        
        Returns:
            String (for text/markdown/org) or Dict.
        """
        # 1. Gather Data
        # Ensure we have essential scalar features loaded
        vital_features = ["EmploymentCategory", "Division", "Step", "RecWellMember", "OnLeave"]
        vitals = self.attributes()
        for feat in vital_features:
            vitals[feat] = getattr(self, feat, None)
            
        service = self.service_history(with_year_start=True)
        volunteer = self.volunteer_info(as_df=True)
        
        if format == "dict":
            return {
                "vitals": vitals,
                "service": service.to_dict(orient="records") if not service.empty else [],
                "volunteer": volunteer.to_dict(orient="records") if not volunteer is None and not volunteer.empty else []
            }
            
        if format == "org":
            return self._report_org(vitals, service, volunteer)
        elif format == "markdown":
            return self._report_markdown(vitals, service, volunteer)
        else:
            return self._report_text(vitals, service, volunteer)

    def _report_text(self, vitals, service, volunteer):
        lines = []
        lines.append(f"FACULTY REPORT: {vitals.get('FullName', 'Unknown')}")
        lines.append("=" * 60)
        
        # Vitals Section
        lines.append("VITALS")
        lines.append("-" * 20)
        core_keys = ['Email', 'UID', 'DeptName', 'Division', 'EmploymentCategory', 'Step', 'Affiliations']
        for k in core_keys:
            val = vitals.get(k)
            if val:
                lines.append(f"{k:<20}: {val}")
        lines.append("")

        # Service Section
        lines.append("SERVICE HISTORY")
        lines.append("-" * 20)
        if not service.empty:
            # Simple table formatting
            df = service.reset_index()
            cols = ["Academic Year", "Committee", "Role"]
            if "ChairStreak" in df.columns:
                cols.append("ChairStreak")
            
            # header
            lines.append(f"{'Year':<10} {'Committee':<30} {'Role':<20} {'ChairStreak'}")
            for _, row in df.iterrows():
                yr = str(row.get('Academic Year', ''))
                comm = str(row.get('Committee', ''))
                role = str(row.get('Role', ''))
                streak = str(row.get('ChairStreak', '')) if pd.notna(row.get('ChairStreak')) else "-"
                lines.append(f"{yr:<10} {comm:<30} {role:<20} {streak}")
        else:
            lines.append("No service history found.")
        lines.append("")

        # Volunteer Section
        lines.append("VOLUNTEER INFO")
        lines.append("-" * 20)
        if volunteer is not None and not volunteer.empty:
            vol_df = volunteer.reset_index()
            for _, row in vol_df.iterrows():
                lines.append(f"Year: {row.get('VolunteerYear', 'Unknown')}")
                choices = [str(row.get(c)) for c in ['Choice1', 'Choice2', 'Choice3'] if pd.notna(row.get(c))]
                if choices:
                    lines.append(f"  Choices: {', '.join(choices)}")
                interests = row.get('Interests')
                if pd.notna(interests):
                    lines.append(f"  Interests: {interests}")
                lines.append("")
        else:
            lines.append("No volunteer records found.")
            
        return "\n".join(lines)

    def _report_markdown(self, vitals, service, volunteer):
        lines = []
        name = vitals.get('FullName', 'Unknown')
        lines.append(f"# {name}")
        
        # Vitals
        lines.append("## Vitals")
        lines.append("| Attribute | Value |")
        lines.append("|---|---|")
        for k, v in vitals.items():
            if v and k not in ['FirstName', 'LastName', 'FullName']: # Skip redundant names
                lines.append(f"| {k} | {v} |")
        lines.append("")
        
        # Service
        lines.append("## Service History")
        if not service.empty:
            df = service.reset_index()
            if "YearStart" in df.columns:
                df = df.drop(columns=["YearStart"])
            lines.append(df.to_markdown(index=False))
        else:
            lines.append("_No service history available._")
        lines.append("")
        
        # Volunteer
        lines.append("## Volunteer Info")
        if volunteer is not None and not volunteer.empty:
            # Select subset of cols for cleanliness
            cols = [c for c in volunteer.columns if c not in ['Email', 'CanonicalName']]
            lines.append(volunteer[cols].to_markdown(index=False))
        else:
            lines.append("_No volunteer info available._")
            
        return "\n".join(lines)

    def _report_org(self, vitals, service, volunteer):
        lines = []
        name = vitals.get('FullName', 'Unknown')
        lines.append(f"* {name}")
        
        # Properties Drawer for Vitals
        lines.append("  :PROPERTIES:")
        for k, v in vitals.items():
            # Handle Series values (extract scalar)
            if isinstance(v, pd.Series):
                v = v.iloc[0] if not v.empty else None
            # Skip NA/None values (pd.isna handles pandas NA, np.nan, None)
            try:
                if v is None or pd.isna(v):
                    continue
            except (ValueError, TypeError):
                pass  # Not a scalar NA check, proceed with value
            if v and isinstance(v, (str, int, float, bool)):
                # Sanitize key for Org property (replace spaces with hyphens, etc)
                safe_key = str(k).replace(" ", "-")
                lines.append(f"  :{safe_key}: {v}")
        lines.append("  :END:")
        
        # Service
        lines.append("** Service History")
        if not service.empty:
            df = service.reset_index()
            if "YearStart" in df.columns:
                df = df.drop(columns=["YearStart"])
            # Org-mode table format
            headers = df.columns.tolist()
            lines.append("  | " + " | ".join(headers) + " |")
            lines.append("  |-" + "-+-".join(["-" * len(h) for h in headers]) + "-|")
            for _, row in df.iterrows():
                vals = [str(x).replace("|", "/") for x in row] # Escape pipes
                lines.append("  | " + " | ".join(vals) + " |")
        else:
            lines.append("  No service history.")

        # Volunteer
        lines.append("** Volunteer Info")
        if volunteer is not None and not volunteer.empty:
             # Org-mode table format
            cols = [c for c in volunteer.columns if c not in ['Email', 'CanonicalName']]
            df = volunteer[cols]
            headers = df.columns.tolist()
            lines.append("  | " + " | ".join(headers) + " |")
            lines.append("  |-" + "-+-".join(["-" * len(h) for h in headers]) + "-|")
            for _, row in df.iterrows():
                vals = [str(x).replace("|", "/") for x in row]
                lines.append("  | " + " | ".join(vals) + " |")
        else:
            lines.append("  No volunteer info.")
            
        return "\n".join(lines)

    def volunteer_info(self, year=None, as_df=False):
        """Return volunteer survey responses for this faculty member.

        Args:
            year: Specific academic year (e.g., "2024-25") or None for all years
            as_df: If True and year is None, return DataFrame instead of dict

        Returns:
            If year specified: dict of field->value, or None if no data
            If year is None: dict of year->info dicts, or DataFrame if as_df=True
        """
        from .new_feature import Features

        if not self.Email:
            return None if year else ({} if not as_df else pd.DataFrame())

        features = Features(_VOLUNTEER_FEATURES)

        if year is not None:
            # Single year lookup
            try:
                df = features[year]
            except (KeyError, ValueError):
                return None

            if df.empty:
                return None

            # Filter to this member's email
            if isinstance(df.index, pd.MultiIndex):
                email_level = df.index.get_level_values("Email")
                if self.Email not in email_level:
                    return None
                member_df = df.loc[df.index.get_level_values("Email") == self.Email]
                row = member_df.iloc[0]
            else:
                if self.Email not in df.index:
                    return None
                row = df.loc[self.Email]

            result = row.to_dict()
            result["VolunteerYear"] = normalize_academic_year(year)
            return result

        else:
            # All years - use available_years() instead of hardcoded list
            available = features.available_years()
            if not available:
                return {} if not as_df else pd.DataFrame()

            results = {}
            for yr in sorted(available, reverse=True):  # Newest first
                info = self.volunteer_info(year=yr)
                if info is not None:
                    results[yr] = info

            if as_df:
                return pd.DataFrame(results).T if results else pd.DataFrame()
            return results

    def service_history(self, include_current: bool = True, with_year_start: bool = False):
        """Return committee service history for this faculty member.

        Args:
            include_current: Unused (kept for API compatibility). History always
                includes all available years from the Appointment/Role features.
            with_year_start: If True, include YearStart column in output.

        Returns:
            DataFrame indexed by (Academic Year, Committee) with columns:
            Role, ServiceStreak, ChairStreak, and optionally YearStart.
        """
        from .new_feature import Features

        if not self.Email:
            return pd.DataFrame()

        # Load history directly from Features - already Email-indexed
        try:
            hist = Features(["Appointment", "Role"])["history"]
        except Exception as exc:
            logger.debug("Failed to load Appointment/Role history: %s", exc)
            return pd.DataFrame()

        if hist.empty:
            return pd.DataFrame()

        # Filter to this member's email
        df = hist.reset_index()
        out = df.loc[df["Email"] == self.Email].copy()

        if out.empty:
            return pd.DataFrame()

        # Rename columns to standard names
        out = out.rename(columns={"Appointment": "Committee", "Year": "Academic Year"})
        out["Academic Year"] = out["Academic Year"].map(normalize_academic_year)
        out["YearStart"] = out["Academic Year"].map(_year_start_from_academic_year)

        # Clean up
        out = out.dropna(subset=["Academic Year", "Committee", "YearStart"])
        out = out.drop_duplicates(subset=["Academic Year", "Committee"])

        # Compute streaks per committee (consecutive academic years)
        out = out.sort_values(["Committee", "YearStart"], ascending=[True, True], kind="stable")
        out["ServiceStreak"] = pd.NA
        out["ChairStreak"] = pd.NA

        for committee, group in out.groupby("Committee", sort=False, observed=True):
            last_year = None
            service_streak = 0
            chair_streak = 0
            last_was_chair = False

            for idx, row in group.iterrows():
                year_start = row.get("YearStart")
                if pd.isna(year_start):
                    continue

                year_int = int(year_start)
                consecutive = last_year is not None and year_int == last_year + 1
                service_streak = service_streak + 1 if consecutive else 1

                role_val = row.get("Role")
                is_chair = isinstance(role_val, str) and "chair" in role_val.lower()
                if is_chair:
                    chair_streak = chair_streak + 1 if (consecutive and last_was_chair) else 1
                else:
                    chair_streak = 0

                out.at[idx, "ServiceStreak"] = service_streak
                out.at[idx, "ChairStreak"] = chair_streak
                last_year = year_int
                last_was_chair = is_chair

        # Final sort: newest first
        out = out.sort_values(["YearStart", "Committee"], ascending=[False, True], kind="stable")

        # Select output columns
        keep = ["Academic Year", "Committee", "Role", "ServiceStreak", "ChairStreak"]
        if with_year_start:
            keep.append("YearStart")
        out = out[[col for col in keep if col in out.columns]]

        # Set index
        if {"Academic Year", "Committee"}.issubset(out.columns):
            out = out.set_index(["Academic Year", "Committee"])

        return out


@cache
def _committee_service_records():
    """
    Return canonicalized committee service records.

    Note: Uses @cache so results persist across calls. If faculty_spellings
    dictionary is updated, this cache must be manually cleared.
    """
    try:
        from .new_feature import Features, Feature
        hist = Features(["Appointment", "Role"])["history"]
    except Exception as exc:
        logger.debug("Failed to load Committees feature: %s", exc)
        return pd.DataFrame(columns=["Committee", "Role", "Academic Year", "YearStart", "FullName", "Email", "DeptName"])

    df = hist.reset_index()
    df = df.rename(columns={"Appointment": "Committee", "Year": "Academic Year"})

    # Normalize committee names via spelling dictionary
    df["Committee"] = df["Committee"].map(spelling.spell)

    # Build FullName lookup from Email
    try:
        full_series = Feature("FullName")["latest"]
        df["FullName"] = df["Email"].map(full_series.to_dict())
    except Exception as exc:
        logger.debug("Failed to map Email->FullName in service records: %s", exc)
        df["FullName"] = pd.NA

    # Build DeptName lookup from Email
    try:
        dept_series = Feature("DeptName")["latest"]
        df["DeptName"] = df["Email"].map(dept_series.to_dict())
    except Exception as exc:
        logger.debug("Failed to map Email->DeptName: %s", exc)
        df["DeptName"] = pd.NA

    df["Academic Year"] = df["Academic Year"].astype(str).str.strip().map(normalize_academic_year)
    df["YearStart"] = df["Academic Year"].map(_year_start_from_academic_year)
    df = df.dropna(subset=["Committee", "Academic Year", "YearStart"])
    df["YearStart"] = df["YearStart"].astype(int)
    df = df.sort_values(["YearStart", "Committee"])
    return df


def committee_names():
    """Return a sorted list of canonical committee acronyms.

    Returns
    -------
    list[str]
        Unique committee acronyms from the service records, sorted
        alphabetically.
    """
    df = _committee_service_records()
    return sorted(df['Committee'].dropna().unique().tolist())


def committee_year_range():
    """Return the year range spanned by committee service records.

    Returns
    -------
    tuple[int | None, int | None]
        ``(min_year_start, max_year_start)`` as integers, or
        ``(None, None)`` if no records exist.
    """
    df = _committee_service_records()
    if df.empty:
        return (None, None)
    return (int(df['YearStart'].min()), int(df['YearStart'].max()))


def committee_membership(committee=None, start_year=None, end_year=None):
    """Return committee membership records filtered by committee and/or year range.

    Parameters
    ----------
    committee : str | None
        Committee acronym (e.g. ``'CAPBD'``).  Spelling is normalised
        via the project dictionary.  ``None`` returns all committees.
    start_year : int | str | None
        Earliest academic-year start to include (inclusive).
    end_year : int | str | None
        Latest academic-year start to include (inclusive).

    Returns
    -------
    pandas.DataFrame
        Filtered service records with columns ``Committee``,
        ``Email``, ``FullName``, ``DeptName``, ``Academic Year``,
        ``YearStart``, and any role/appointment columns present in the
        source data.
    """
    df = _committee_service_records().copy()

    start_year_norm, _ = normalize_year_input(start_year)
    end_year_norm, _ = normalize_year_input(end_year)

    if committee:
        canonical = spelling.spell(committee)
        df = df[df['Committee'] == canonical]
    if start_year_norm is not None:
        df = df[df['YearStart'] >= start_year_norm]
    if end_year_norm is not None:
        df = df[df['YearStart'] <= end_year_norm]

    return df

def dept_name_to_code(DeptName=None, df_fn='2025-01 All Senate Faculty.csv.gpg'):
    """Map department names to their short codes.

    Parameters
    ----------
    DeptName : str | None
        A single department name to look up.  When ``None``, the full
        mapping dictionary is returned.
    df_fn : str
        Data file parameter forwarded to :func:`faculty_vitals`
        (retained for backward compatibility).

    Returns
    -------
    str | dict[str, str] | None
        The department code when *DeptName* is given, the complete
        ``{name: code}`` mapping when ``None``, or ``None`` (with a
        warning) if the name is not found.
    """
    fac = faculty_vitals(df_fn=df_fn)

    d = dict(fac[['DeptName','DeptCode']].values)

    if DeptName is None:
        return d
    else:
        try:
            return d[DeptName]
        except KeyError:
            warnings.warn(f'No department named {DeptName}.')

def dept_code_to_name(DeptCode=None, df_fn='2025-01 All Senate Faculty.csv.gpg'):
    """Map department codes to their full names.

    Parameters
    ----------
    DeptCode : str | None
        A single department code to look up.  When ``None``, the full
        mapping dictionary is returned.
    df_fn : str
        Data file parameter forwarded to :func:`faculty_vitals`
        (retained for backward compatibility).

    Returns
    -------
    str | dict[str, str] | None
        The department name when *DeptCode* is given, the complete
        ``{code: name}`` mapping when ``None``, or ``None`` (with a
        warning) if the code is not found.
    """
    fac = faculty_vitals(df_fn=df_fn)

    d = dict(fac[['DeptCode','DeptName']].values)

    if DeptCode is None:
        return d
    else:
        try:
            return d[DeptCode]
        except KeyError:
            warnings.warn(f'No department code {DeptCode}.')

class Department():
    def __init__(self, DeptName=None, DeptCode=None):
        self.DeptName = DeptName
        self.DeptCode = DeptCode

        if self.DeptName is None:
            if self.DeptCode is not None:
                try:
                    self.DeptName = dept_code_to_name(self.DeptCode)
                except KeyError:
                    warnings.warn(f'No department named {DeptName}.')
        elif self.DeptCode is None:
            if self.DeptName is not None:
                try:
                    self.DeptCode = dept_name_to_code(self.DeptName)
                except KeyError:
                    warnings.warn(f'No department named {DeptName}.')
        elif self.DeptName is not None and self.DeptCode is not None:
            if self.DeptCode != dept_name_to_code(self.DeptName):
                raise ValueError(f"DeptCode {self.DeptCode} does not match DeptName {self.DeptName}")
            if self.DeptName != dept_code_to_name(self.DeptCode):
                raise ValueError(f"DeptName {self.DeptName} does not match DeptCode {self.DeptCode}")


def clear_faculty_caches():
    """Clear all cached faculty data. Call after updating faculty_spellings dictionary."""
    from .rawdata import clear_dataframe_cache
    _committee_service_records.cache_clear()
    clear_dataframe_cache()
