from typing import Dict, Callable, Iterable, Optional, Union, List
import warnings
import pandas as pd
import numpy as np
from ligonlibrary.dataframes import find_similar_pairs
try:
    from ligonlibrary.strings import most_similar
except ImportError:
    from thefuzz import fuzz

    def most_similar(target, candidates, similarity_threshold=85):
        """Fallback similarity helper if ligonlibrary.strings.most_similar is unavailable."""
        if target is None:
            return None
        best = None
        best_score = similarity_threshold
        normalized_target = target.strip().lower()

        for candidate in candidates:
            if candidate is None:
                continue
            score = fuzz.ratio(normalized_target, candidate.strip().lower())
            if score >= best_score:
                best_score = score
                best = candidate

        return best
from ucbfaculty import spelling, FACULTY_NAME_DICTIONARY
from ucbfaculty.utils import name_variants, _name_variants, yes_response, ensure_unique_index

_logged_messages: set[str] = set()


# ---------------------------------------------------------------------------
# Column-detection helpers for reindex_to_email
# ---------------------------------------------------------------------------

def _is_email_column(series: pd.Series) -> bool:
    """Return True if a significant fraction of non-null values contain '@'."""
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return False
    at_frac = non_null.str.contains('@', na=False).mean()
    return at_frac > 0.3


def _is_name_column(series: pd.Series) -> bool:
    """Heuristic: short, multi-word, mostly alphabetic string values."""
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return False
    # Values are mostly short (under 60 chars)
    short = (non_null.str.len() < 60).mean() > 0.8
    # Values are mostly multi-word
    multi_word = (non_null.str.split().str.len() >= 2).mean() > 0.5
    # Values are mostly alphabetic (allowing spaces, commas, hyphens, periods, apostrophes)
    alpha = non_null.str.match(r"^[A-Za-z\s,.\'\-]+$", na=False).mean() > 0.7
    return short and multi_word and alpha


def _detect_name_format(series: pd.Series) -> str:
    """Return 'last_first' if majority of values contain commas, else 'first_last'."""
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return 'first_last'
    comma_frac = non_null.str.contains(',', na=False).mean()
    return 'last_first' if comma_frac > 0.5 else 'first_last'


# ---------------------------------------------------------------------------
# Matching helpers for reindex_to_email
# ---------------------------------------------------------------------------

def _match_emails(
    email_series: pd.Series,
    canonical_emails: set,
    email_dictionary,
) -> pd.Series:
    """Match a series of wild emails to canonical ``@berkeley.edu`` emails.

    Strategy (applied per value, in priority order):
    1. Exact match — email already in *canonical_emails*.
    2. ``EMAIL_DICTIONARY`` lookup via *inverse_d*.
    3. Local-part match — extract part before ``@`` and compare against
       local parts of known Berkeley emails.

    Returns a Series aligned to *email_series* whose values are canonical
    emails (or ``pd.NA`` for unmatched rows).
    """
    result = pd.Series(pd.NA, index=email_series.index, dtype='object')

    # Pre-build local-part lookup for strategy 3 (O(1) per row)
    local_to_canonical: dict[str, str] = {}
    for email in canonical_emails:
        local = str(email).split('@')[0].lower()
        local_to_canonical[local] = email

    for idx, val in email_series.items():
        if pd.isna(val):
            continue
        email = str(val).strip().lower()

        # Strategy 1: exact match
        if email in canonical_emails:
            result[idx] = email
            continue

        # Strategy 2: dictionary lookup
        if email_dictionary is not None:
            corrected = email_dictionary.inverse_d[email]
            if corrected != email and corrected in canonical_emails:
                result[idx] = corrected
                continue

        # Strategy 3: local-part match
        if '@' in email:
            local = email.split('@')[0]
            canonical = local_to_canonical.get(local)
            if canonical is not None:
                result[idx] = canonical

    return result


def _match_names(
    name_series: pd.Series,
    canonical_mapping: pd.Series,
    spelling_dictionary,
    similarity_threshold: int = 93,
) -> pd.Series:
    """Match a series of names to canonical faculty emails.

    *canonical_mapping* is a Series of FullName values indexed by Email
    (as returned by ``_fullname_email_mapping()``).

    Strategy (per value, in priority order):
    1. Spelling-dictionary correction then exact match.
    2. ``name_variants()`` — try case / format variations.
    3. ``most_similar()`` — fuzzy match at *similarity_threshold*.

    Returns a Series aligned to *name_series* whose values are canonical
    emails (or ``pd.NA``).
    """
    canonical_values = canonical_mapping.values.tolist()
    canonical_names = set(canonical_values)

    # Build reverse lookup: FullName -> Email
    name_to_email: dict[str, str] = {}
    for email, name in canonical_mapping.items():
        if isinstance(name, str):
            name_to_email[name] = email

    result = pd.Series(pd.NA, index=name_series.index, dtype='object')

    for idx, val in name_series.items():
        if pd.isna(val) or not isinstance(val, str) or not val.strip():
            continue
        name = val.strip()

        # Strategy 1: spelling correction + exact match
        corrected = spelling_dictionary.inverse_d[name]
        if corrected in canonical_names:
            result[idx] = name_to_email[corrected]
            continue

        # Strategy 2: name variants
        matched = False
        for variant in _name_variants(name):
            canonical_variant = spelling_dictionary.inverse_d[variant]
            if canonical_variant in canonical_names:
                result[idx] = name_to_email[canonical_variant]
                matched = True
                break
        if matched:
            continue

        # Strategy 3: fuzzy matching
        possibility = most_similar(
            name, canonical_values, similarity_threshold=similarity_threshold,
        )
        if possibility is not None:
            result[idx] = name_to_email.get(possibility)

    return result


def fullname_to_email_index(
    s: pd.Series,
    canonical_keys: pd.Series,
    return_unmatched_names=False,
    spelling_dictionary=None,
    interactive=True,
    deduplicate=True,
) -> pd.Series:
    """Re-index a FullName-indexed Series to Email using a canonical mapping.

    Matches names to faculty emails through spelling correction, name
    variants, and fuzzy matching (in that priority order).

    Parameters
    ----------
    s : pandas.Series
        Data series whose index contains full names.
    canonical_keys : pandas.Series
        Series of ``FullName`` values indexed by ``Email`` (the
        canonical name-to-email mapping).
    return_unmatched_names : bool
        When ``True``, return a tuple ``(series, unmatched)`` where
        *unmatched* is a list of names that could not be matched.
    spelling_dictionary : Dictionary | None
        Spelling dictionary to use for name correction.  Defaults to
        ``FACULTY_NAME_DICTIONARY``.
    interactive : bool
        When ``True``, prompt the user to confirm fuzzy matches and
        optionally persist dictionary additions.
    deduplicate : bool
        When ``True`` (default), collapse duplicate email entries
        via :func:`ensure_unique_index`.

    Returns
    -------
    pandas.Series | tuple[pandas.Series, list[str]]
        The re-indexed Series (and optionally the unmatched names).
    """
    if spelling_dictionary is None:
        spelling_dictionary = FACULTY_NAME_DICTIONARY

    # Correct spelling of selected faculty names
    s = s.rename(index=spelling_dictionary.inverse_d)

    canonical_values = canonical_keys.values.tolist()
    canonical_names = set(canonical_values)
    unmatched = set(s.index.to_list()).difference(canonical_names)

    n_additions = 0
    updates = {}
    no_match = []
    canonical_df = canonical_keys.reset_index()
    canonical_df.columns = ["Email", "FullName"]

    for name in list(unmatched):

        matched = False
        for variant in _name_variants(name):
            canonical_variant = spelling_dictionary.inverse_d[variant]
            if canonical_variant in canonical_names:
                updates[name] = canonical_variant
                matched = True
                break
        if matched:
            continue
        possibility = most_similar(name,canonical_keys.values.tolist(),similarity_threshold=93)
        if possibility is not None:
            if interactive:
                if yes_response(f"Matched {name}:{possibility}.  Add to dictionary?",default='y'):
                    updates[name] = possibility
                    spelling_dictionary.add(possibility,name)
                    n_additions += 1
            else:
                updates[name] = possibility
        else:
            no_match.append(name)

    if interactive and n_additions and yes_response("Save updated dictionary?"):
        spelling_dictionary.persist()

    # Apply updates
    s = s.rename(index=updates)

    # Map to get email index
    s = pd.DataFrame(canonical_keys).join(s,on='FullName')[s.name]

    # Remove entries with no email mapping
    s = s.dropna()
    # Combine duplicates per email unless caller opts out (e.g. when
    # duplicate index entries are meaningful, like multiple appointments).
    if deduplicate:
        s = ensure_unique_index(s)

    if return_unmatched_names:
        return s, no_match
    else:
        return s

def merge_features(
    series_dict: Dict[str, pd.Series],
    canonical_email_to_name_key='FullName',
    return_unmatched_names=False,
    spelling_dictionary=None,
    interactive=True,
) -> pd.DataFrame:
    """
    Merge a dictionary of pandas Series into a DataFrame indexed by Email.

    Parameters:
    -----------
    series_dict : Dict[str, pd.Series]
        Dictionary of Series, each indexed by either "Email" or "FullName"

    Returns:
    --------
    pd.DataFrame
        Merged DataFrame indexed by Email with canonical names
    """
    if spelling_dictionary is None:
        spelling_dictionary = FACULTY_NAME_DICTIONARY

    # Get the canonical email-to-name mapping
    canonical_mapping = ensure_unique_index(series_dict[canonical_email_to_name_key].copy())
    canonical_mapping = canonical_mapping.map(
        lambda value: value[0] if isinstance(value, list) and value else value
    )
    canonical_mapping = canonical_mapping.replace(spelling_dictionary.inverse_d)

    # Drop entries with missing emails (pd.NA, None, NaN, empty strings)
    valid_email_mask = canonical_mapping.index.notna() & (canonical_mapping.index != '') & (canonical_mapping.index != 'nan')
    canonical_mapping = canonical_mapping[valid_email_mask]

    if len(canonical_mapping) == 0:
        raise ValueError("No valid emails found in canonical mapping after dropping missing values")

    dropped_count = len(series_dict[canonical_email_to_name_key]) - len(canonical_mapping)
    if dropped_count > 0:
        msg = f"Dropped {dropped_count} entries with missing/invalid emails from canonical mapping"
        if msg not in _logged_messages:
            _logged_messages.add(msg)
            warnings.warn(msg, stacklevel=2)

    # Start with all valid emails from the canonical mapping
    all_emails = set(canonical_mapping.index)

    # Add emails from any other series that are already indexed by Email
    for key, series in series_dict.items():
        if key != canonical_email_to_name_key and series.index.name == 'Email':
            # Only add valid emails
            valid_emails = series.index[series.index.notna() & (series.index != '') & (series.index != 'nan')]
            all_emails.update(valid_emails)

    # Create the result DataFrame
    result_df = pd.DataFrame(index=pd.Index(sorted(all_emails), name='Email'))

    # Add canonical name column
    result_df[canonical_email_to_name_key] = canonical_mapping

    # Process each series
    for series_name, series in series_dict.items():
        series = ensure_unique_index(series.squeeze())  # May result in a series of lists!
        if len(series.shape) != 1:
            raise ValueError(f"Expected a Series for '{series_name}', got shape {series.shape}")

        if series_name == canonical_email_to_name_key:
            # Already added above
            continue

        if series.index.name == 'Email':
            # Series already indexed by Email - direct merge, but only for valid emails
            valid_series = series[series.index.notna() & (series.index != '') & (series.index != 'nan')]
            invalid_count = len(series) - len(valid_series)
            if invalid_count > 0:
                warnings.warn(f"Dropped {invalid_count} entries with missing/invalid emails from series '{series_name}'", stacklevel=2)
            result_df[series_name] = valid_series

        elif series.index.name == 'FullName':
            result_df[series_name] = fullname_to_email_index(
                series,
                canonical_mapping,
                spelling_dictionary=spelling_dictionary,
                interactive=interactive,
            )
        else:
            warnings.warn(f"Series '{series_name}' has unexpected index name: {series.index.name}", stacklevel=2)

    return result_df


def build_feature_dataframe(
    feature_names: Union[str, Iterable[str]],
    feature_builder: Optional[object] = None,
    builder_kwargs: Optional[Dict[str, object]] = None,
    merge_kwargs: Optional[Dict[str, object]] = None,
    interactive: bool = True,
) -> pd.DataFrame:
    """
    Load and merge feature columns in a single call.

    Parameters
    ----------
    feature_names
        A feature name or iterable of feature names declared in the YAML spec.
    feature_builder
        Deprecated; ignored.
    builder_kwargs
        Deprecated; ignored.
    merge_kwargs
        Optional dict of keyword arguments forwarded to :func:`merge_features`.
    interactive
        When ``False``, bypasses interactive spelling-dictionary prompts and uses
        best-guess matches transiently (useful for automated contexts like Panel).

    Returns
    -------
    pandas.DataFrame
        The merged feature data indexed by Email.
    """
    from .new_feature import Features

    if isinstance(feature_names, str):
        feature_names = [feature_names]

    merge_kwargs = merge_kwargs or {}
    merge_kwargs.setdefault("interactive", interactive)

    names = list(feature_names)

    # "Email" defines the index in the old system; the new system treats it
    # as a FullName-indexed feature.  Exclude it from the merge and let Email
    # remain as the natural index from other features.
    include_email = "Email" in names
    merge_names = [n for n in names if n != "Email"]

    if not merge_names:
        # Only "Email" was requested — return it as a column
        from .new_feature import Feature
        s = Feature("Email")["latest"]
        df = s.reset_index()
        df.columns = ["Email", "FullName"] if "FullName" in names or len(df.columns) == 2 else df.columns
        return df.set_index("Email") if "Email" in df.columns else df

    df = Features(merge_names)["latest"]

    # Drop Year index level for backward compatibility
    if isinstance(df.index, pd.MultiIndex) and "Year" in df.index.names:
        df = df.droplevel("Year")

    # Deduplicate: combine values across years, preferring non-null
    if not df.index.is_unique:
        df = df.groupby(df.index).first()

    return df


# ---------------------------------------------------------------------------
# reindex_to_email — match a "wild" DataFrame to Berkeley faculty
# ---------------------------------------------------------------------------

def ingest_dataframe(
    filepath: Union[str, "Path"],
    name_col=None,
    email_col=None,
    interactive: bool = False,
    similarity_threshold: int = 93,
) -> pd.DataFrame:
    """Load a CSV/Excel file and match rows to Berkeley faculty.

    Reads the file, handles multi-sheet Excel workbooks (picks the first
    sheet), and calls :func:`reindex_to_email` to identify which rows
    correspond to known faculty.  Returns matched rows reindexed by
    canonical ``@berkeley.edu`` email.

    Parameters
    ----------
    filepath : str | Path
        Path to a CSV or Excel file on disk.
    name_col : str | list | tuple | None
        Column(s) containing names (auto-detected if omitted).
    email_col : str | None
        Column containing email addresses (auto-detected if omitted).
    interactive : bool
        When ``True``, prompts the user to confirm fuzzy matches.
    similarity_threshold : int
        Threshold (0--100) for fuzzy name matching.

    Returns
    -------
    pandas.DataFrame
        Subset of the input data reindexed by canonical Email.
        Rows that could not be matched are dropped.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    """
    from pathlib import Path
    from .rawdata import get_dataframe

    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = get_dataframe(str(p), correct_column_spellings=False)
    if isinstance(df, dict):
        # Multi-sheet Excel -- pick first sheet
        sheet_name = next(iter(df))
        df = df[sheet_name]
        warnings.warn(
            f"Multiple sheets found; using '{sheet_name}'", stacklevel=2,
        )

    return reindex_to_email(
        df,
        name_col=name_col,
        email_col=email_col,
        interactive=interactive,
        similarity_threshold=similarity_threshold,
    )


def reindex_to_email(
    df: pd.DataFrame,
    name_col=None,
    email_col=None,
    interactive: bool = False,
    similarity_threshold: int = 93,
) -> pd.DataFrame:
    """Reindex a *wild* DataFrame by canonical Berkeley faculty Email.

    Takes external data (scholarly society members, conference attendees, etc.)
    and identifies which rows correspond to Berkeley faculty, returning those
    rows reindexed by their canonical ``@berkeley.edu`` email.  The DataFrame
    content is preserved — this is a **reindexing** operation.

    Parameters
    ----------
    df : pandas.DataFrame
        The external DataFrame to reindex.
    name_col : str | list | tuple | None
        Column(s) containing names.

        * ``str`` — single column with full names.
        * ``list`` / ``tuple`` of ``str`` — columns to compose
          (e.g. ``('FirstName', 'LastName')``).
        * ``None`` — auto-detect.
    email_col : str | None
        Column containing email addresses, or ``None`` for auto-detect.
    interactive : bool
        When ``True``, prompts the user to confirm fuzzy matches and persists
        new entries to ``FACULTY_NAME_DICTIONARY`` and ``EMAIL_DICTIONARY``.
    similarity_threshold : int
        Threshold (0–100) for fuzzy name matching via
        :func:`most_similar`.

    Returns
    -------
    pandas.DataFrame
        Subset of *df* reindexed by canonical Email.  Rows that could not be
        matched to a Berkeley faculty member are dropped.
    """
    # Lazy imports to avoid circular-import issues
    from ucbfaculty import FACULTY_NAME_DICTIONARY, EMAIL_DICTIONARY
    from ucbfaculty.new_feature import _fullname_email_mapping
    from ucbfaculty.transformations import reversename, fullname

    working_df = df.copy()

    # ------------------------------------------------------------------
    # Step 1 — Classify columns & compose names
    # ------------------------------------------------------------------

    # --- Resolve email column(s) ---
    email_candidates: list[str] = []
    if email_col is not None:
        if email_col not in working_df.columns:
            raise ValueError(
                f"email_col={email_col!r} not found in DataFrame columns"
            )
        email_candidates = [email_col]
    else:
        for col in working_df.select_dtypes(include=['object', 'string']).columns:
            if _is_email_column(working_df[col]):
                email_candidates.append(col)

    # --- Resolve name column(s) -> a single working Series of names ---
    name_series: Optional[pd.Series] = None

    if name_col is not None:
        if isinstance(name_col, (list, tuple)):
            # Compose from multiple columns via transformations.fullname()
            sub_df = working_df[list(name_col)].copy()
            if len(name_col) == 2:
                sub_df.columns = ['FirstName', 'LastName']
            name_series = sub_df.apply(fullname, axis=1)
        else:
            if name_col not in working_df.columns:
                raise ValueError(
                    f"name_col={name_col!r} not found in DataFrame columns"
                )
            raw = working_df[name_col].copy()
            if _detect_name_format(raw) == 'last_first':
                name_series = raw.apply(reversename)
            else:
                name_series = raw
            name_series = name_series.map(
                lambda x: str(x).strip().title() if isinstance(x, str) else x
            )
    else:
        # Auto-detect: look for recognised FirstName + LastName pair
        cols_lower = {c.lower(): c for c in working_df.columns}
        if 'firstname' in cols_lower and 'lastname' in cols_lower:
            first_col = cols_lower['firstname']
            last_col = cols_lower['lastname']
            sub_df = working_df[[first_col, last_col]].copy()
            sub_df.columns = ['FirstName', 'LastName']
            name_series = sub_df.apply(fullname, axis=1)
        else:
            # Heuristic: find string columns that look like names
            name_candidates = [
                col
                for col in working_df.select_dtypes(include=['object', 'string']).columns
                if col not in email_candidates and _is_name_column(working_df[col])
            ]

            if len(name_candidates) == 1:
                raw = working_df[name_candidates[0]].copy()
                if _detect_name_format(raw) == 'last_first':
                    name_series = raw.apply(reversename)
                else:
                    name_series = raw
                name_series = name_series.map(
                    lambda x: str(x).strip().title() if isinstance(x, str) else x
                )
            elif len(name_candidates) > 1:
                if interactive:
                    print("Multiple potential name columns detected:")
                    for i, col in enumerate(name_candidates):
                        sample = working_df[col].dropna().head(3).tolist()
                        print(f"  {i}: {col}  (sample: {sample})")
                    choice = input(
                        "Enter column number (or press Enter to skip name matching): "
                    ).strip()
                    if choice.isdigit() and int(choice) < len(name_candidates):
                        chosen = name_candidates[int(choice)]
                        raw = working_df[chosen].copy()
                        if _detect_name_format(raw) == 'last_first':
                            name_series = raw.apply(reversename)
                        else:
                            name_series = raw
                        name_series = name_series.map(
                            lambda x: str(x).strip().title() if isinstance(x, str) else x
                        )
                # else: non-interactive with multiple candidates — defer to
                # Step 3 which will score each by match count.

    # ------------------------------------------------------------------
    # Step 2 — Email matching
    # ------------------------------------------------------------------
    canonical_mapping = _fullname_email_mapping()  # Series: Email -> FullName
    canonical_emails = set(canonical_mapping.index)

    email_matches = pd.Series(pd.NA, index=working_df.index, dtype='object')
    best_email_col: Optional[str] = None

    if email_candidates:
        best_count = -1
        for col in email_candidates:
            col_matches = _match_emails(
                working_df[col], canonical_emails, EMAIL_DICTIONARY,
            )
            match_count = int(col_matches.notna().sum())
            if match_count > best_count:
                best_count = match_count
                email_matches = col_matches
                best_email_col = col

    # ------------------------------------------------------------------
    # Step 3 — Name matching
    # ------------------------------------------------------------------
    name_matches = pd.Series(pd.NA, index=working_df.index, dtype='object')

    if name_series is not None:
        # Single name series already resolved in Step 1
        name_matches = _match_names(
            name_series,
            canonical_mapping,
            FACULTY_NAME_DICTIONARY,
            similarity_threshold,
        )
    elif name_col is None:
        # Auto-detect produced multiple candidates and interactive was False.
        # Score each candidate column by match count and pick the best.
        name_candidates_to_try = [
            col
            for col in working_df.select_dtypes(include=['object', 'string']).columns
            if col not in email_candidates and _is_name_column(working_df[col])
        ]
        best_name_count = -1
        for col in name_candidates_to_try:
            trial = working_df[col].copy()
            if _detect_name_format(trial) == 'last_first':
                trial = trial.apply(reversename)
            trial = trial.map(
                lambda x: str(x).strip().title() if isinstance(x, str) else x
            )
            trial_matches = _match_names(
                trial,
                canonical_mapping,
                FACULTY_NAME_DICTIONARY,
                similarity_threshold,
            )
            count = int(trial_matches.notna().sum())
            if count > best_name_count:
                best_name_count = count
                name_matches = trial_matches

    # ------------------------------------------------------------------
    # Step 4 — Cross-validate and learn (interactive only)
    # ------------------------------------------------------------------
    if interactive:
        n_name_additions = 0
        n_email_additions = 0

        for idx in working_df.index:
            has_email = pd.notna(email_matches.get(idx))
            has_name = pd.notna(name_matches.get(idx))

            if has_email and not has_name and name_series is not None:
                # Email matched but name didn't — possible new spelling
                canonical_email = email_matches[idx]
                canonical_name = canonical_mapping.get(canonical_email)
                wild_name = name_series.get(idx)
                if (
                    canonical_name
                    and wild_name
                    and isinstance(wild_name, str)
                    and wild_name.strip()
                ):
                    sim = most_similar(
                        wild_name, [canonical_name], similarity_threshold=70,
                    )
                    if sim is not None and yes_response(
                        f"Email matched {canonical_email}, name '{wild_name}' "
                        f"looks like '{canonical_name}'. "
                        f"Add to name dictionary?",
                        default='y',
                    ):
                        FACULTY_NAME_DICTIONARY.add(canonical_name, wild_name)
                        n_name_additions += 1

            elif has_name and not has_email and best_email_col is not None:
                # Name matched but email didn't — possible new email alias
                canonical_email = name_matches[idx]
                wild_email = working_df.at[idx, best_email_col]
                if pd.notna(wild_email) and str(wild_email).strip():
                    if yes_response(
                        f"Name matched to {canonical_email}, but email "
                        f"'{wild_email}' is unknown. Add to email dictionary?",
                        default='y',
                    ):
                        EMAIL_DICTIONARY.add(
                            canonical_email, str(wild_email).strip(),
                        )
                        n_email_additions += 1

        if n_name_additions and yes_response("Save updated name dictionary?"):
            FACULTY_NAME_DICTIONARY.persist()
        if n_email_additions and yes_response("Save updated email dictionary?"):
            EMAIL_DICTIONARY.persist()

    # ------------------------------------------------------------------
    # Step 5 — Combine results (email matches take precedence)
    # ------------------------------------------------------------------
    canonical_email = email_matches.copy()
    missing = canonical_email.isna()
    canonical_email[missing] = name_matches[missing]

    # ------------------------------------------------------------------
    # Step 6 — Reindex and return
    # ------------------------------------------------------------------
    matched_mask = canonical_email.notna()
    result = working_df[matched_mask].copy()
    result.index = pd.Index(canonical_email[matched_mask].values, name='Email')

    if not result.index.is_unique:
        warnings.warn(
            f"reindex_to_email: {result.index.duplicated().sum()} duplicate "
            f"email mappings found. Keeping all rows.",
            stacklevel=2,
        )

    return result
