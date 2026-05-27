"""MCP server exposing ucbfaculty query functions as tools.

Runs over **stdio** transport using the FastMCP decorator API so that
Claude Code CLI and Emacs ``mcp.el`` can query faculty data directly.

Key safety invariants
---------------------
* ALL logging goes to **stderr** (stdout is JSON-RPC transport).
* Every library call that accepts ``interactive`` forces it to ``False``
  so that ``input()`` never reads from the JSON-RPC stream.
* Library imports are *lazy* (inside tool functions) to avoid import-time
  side-effects that might write to stdout.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging -- MUST go to stderr, never stdout
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ucbfaculty.mcp")

# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "ucbfaculty",
    instructions=(
        "UC Berkeley faculty data server.\n\n"
        "PRIMARY PATH: use `query` (read-only DuckDB SQL) for slicing, "
        "joining, and aggregating.  Use `describe` to list views and "
        "inspect their schemas.  Together these answer the bulk of "
        "data questions and return *results* (not source rows), so "
        "they're much cheaper in context than older dataframe tools "
        "would have been.  When a result is genuinely too large to "
        "inline, pass `output_path` to `query` and the full result "
        "spills to a Parquet/CSV file (Parquet is queryable directly: "
        "`query(\"SELECT ... FROM '/tmp/spill.parquet'\")`).\n\n"
        "PERSON-SHAPED HELPERS: `lookup_faculty` (fuzzy name search) "
        "and `service_history` (one person's committee record) are "
        "ergonomic shortcuts; `query` can do both with more effort.\n\n"
        "AUTHENTICATION: Most data tools work automatically because "
        "gpg-agent caches the decryption passphrase.  Just call the "
        "tool you need; if you get a DecryptionError, ask the user "
        "to run `ucbfaculty unlock` in a separate terminal to cache "
        "their passphrase via pinentry, then retry.  NEVER ask the "
        "user for their passphrase directly.\n\n"
        "Tools that never need GPG: spell, spell_variants, "
        "get_current_year, propose_spec."
    ),
)

# ---------------------------------------------------------------------------
# Server-level default format (set via CLI --default-format)
# ---------------------------------------------------------------------------
# JSON is the default because every consumer of this server is an LLM, for
# which JSON is more token-dense than padded org tables.  Humans reading
# transcripts can pass ``format="org"`` explicitly.
_DEFAULT_FORMAT: str = "json"

# Truncation limits
_TABLE_ROW_LIMIT = 25        # org / markdown
_JSON_ROW_LIMIT = 100


# ===================================================================
# Serialization helpers
# ===================================================================

def _resolve_format(fmt: Optional[str]) -> str:
    """Return the effective output format, falling back to server default."""
    if fmt and fmt.lower() in {"org", "markdown", "json"}:
        return fmt.lower()
    return _DEFAULT_FORMAT


def _serialize_dataframe(
    df,
    fmt: Optional[str] = None,
    limit: Optional[int] = None,
    hint: str = "",
) -> str:
    """Render a DataFrame (or Series) as text in the requested format.

    Parameters
    ----------
    df : pandas.DataFrame | pandas.Series
        Data to serialize.
    fmt : str | None
        ``"org"``, ``"markdown"``, or ``"json"``.  Falls back to
        the server-level default.
    limit : int | None
        Maximum rows to return.  Falls back to the built-in default
        (25 for org/markdown, 100 for json).
    hint : str
        Appended to the truncation notice to suggest filters.

    Returns
    -------
    str
        The formatted table text.
    """
    import pandas as pd
    from tabulate import tabulate

    fmt = _resolve_format(fmt)

    # Coerce Series -> single-column DataFrame
    if isinstance(df, pd.Series):
        name = df.name or "value"
        df = df.reset_index()
        if df.columns.tolist() == [0]:
            df.columns = [name]

    if df.empty:
        return "(No results)"

    total_rows = len(df)
    default_limit = _JSON_ROW_LIMIT if fmt == "json" else _TABLE_ROW_LIMIT
    effective_limit = limit if limit is not None else default_limit
    truncated = total_rows > effective_limit

    if truncated:
        df = df.head(effective_limit)

    # Always work with a flat DataFrame (index as columns)
    if df.index.name or (hasattr(df.index, 'names') and any(df.index.names)):
        df = df.reset_index()

    if truncated:
        suffix = f"\n(Showing {effective_limit} of {total_rows} rows)"
        if hint:
            suffix += f"  {hint}"
    else:
        suffix = ""

    if fmt == "json":
        return df.to_json(orient="records", default_handler=str) + suffix
    elif fmt == "markdown":
        return df.to_markdown(index=False) + suffix
    else:
        # org -- use tabulate(tablefmt="orgtbl")
        return tabulate(
            df, headers="keys", tablefmt="orgtbl", showindex=False,
        ) + suffix


def _serialize_list(items: list, fmt: Optional[str] = None) -> str:
    """Render a list of strings as text or JSON."""
    import json

    fmt = _resolve_format(fmt)
    if not items:
        return "(No results)"
    if fmt == "json":
        return json.dumps(items)
    # org / markdown -- newline-joined
    return "\n".join(str(i) for i in items)


def _serialize_dict(d: dict, fmt: Optional[str] = None) -> str:
    """Render a dict as text or JSON."""
    import json
    import pandas as pd
    from tabulate import tabulate

    fmt = _resolve_format(fmt)
    if not d:
        return "(No results)"
    if fmt == "json":
        return json.dumps(d, default=str)
    # org / markdown -- render as 2-column table
    df = pd.DataFrame(list(d.items()), columns=["Key", "Value"])
    if fmt == "markdown":
        return df.to_markdown(index=False)
    return tabulate(df, headers="keys", tablefmt="orgtbl", showindex=False)


# ===================================================================
# Error handling helper
# ===================================================================

def _handle_error(exc: Exception) -> str:
    """Convert common library exceptions to user-friendly messages."""
    from ucbfaculty.rawdata import DecryptionError

    if isinstance(exc, DecryptionError):
        return (
            "DecryptionError: gpg-agent does not have the passphrase "
            "cached.  Ask the user to run `ucbfaculty unlock` in a "
            "separate terminal to cache their passphrase via pinentry, "
            "then retry this tool.  NEVER ask the user for their "
            "passphrase directly."
        )
    if isinstance(exc, FileNotFoundError):
        return f"File not found: {exc}"
    if isinstance(exc, (KeyError, ValueError)):
        return str(exc)
    # Fallback
    logger.exception("Unexpected error in MCP tool")
    return f"Error: {type(exc).__name__}: {exc}"


# ===================================================================
# 1. Spelling / lookup tools
# ===================================================================

@mcp.tool()
def spell(text: str) -> str:
    """Resolve a name to its canonical form using the spelling dictionary.

    Works for committee names, department names, and faculty names.
    Lookups are case-insensitive (the input is also tried in title-case).
    Examples: ``"budget committee"`` -> ``"BIR"``,
    ``"agricultural & resource economics"`` ->
    ``"Agricultural & Resource Economics"``.

    Unknown inputs fall back to the stripped, title-cased form (so
    ``"econ"`` -> ``"Econ"``); use ``spell_variants`` to discover what
    canonical names exist.

    Parameters
    ----------
    text : str
        The name or abbreviation to look up.
    """
    from ucbfaculty import spelling as _sp

    try:
        canonical = _sp.spell(text)
        return canonical
    except Exception as exc:
        return _handle_error(exc)


@mcp.tool()
def spell_variants(canonical: str, format: Optional[str] = None) -> str:
    """List all known spelling variants for a canonical name.

    Useful for discovering alternate names for a committee, department,
    or person.  Example: ``spell_variants("BIR")`` returns
    ``"Budget Committee"``, ``"Budget and Interdepartmental Relations"``, etc.

    Note that the ``canonical`` argument must itself be a canonical key
    (the long-form name, e.g. ``"Agricultural & Resource Economics"`` or
    ``"BIR"``), not a short alias --- run ``spell("…")`` first if unsure.

    Parameters
    ----------
    canonical : str
        The canonical name (e.g. ``"BIR"``,
        ``"Agricultural & Resource Economics"``).
    format : str | None
        Output format: ``"org"`` (default), ``"markdown"``, or ``"json"``.
    """
    from ucbfaculty import spelling as _sp

    try:
        variants = _sp.spellings(canonical)
        if not variants or variants == [canonical]:
            return f"No variants found for '{canonical}'."
        return _serialize_list(variants, format)
    except Exception as exc:
        return _handle_error(exc)


@mcp.tool()
def lookup_faculty(
    name: str,
    format: Optional[str] = None,
) -> str:
    """Search for a faculty member by name (fuzzy matching).

    Given a full or partial name, finds matching Berkeley faculty and
    returns their vitals (Email, FullName, DeptName, etc.).

    On a no-match, the response includes the three closest known names
    as suggestions (e.g. for a likely typo).

    Parameters
    ----------
    name : str
        Faculty name to search for (e.g. ``"Panos Papadopoulos"``).
    format : str | None
        Output format: ``"org"`` (default), ``"markdown"``, or ``"json"``.
    """
    import pandas as pd
    from ucbfaculty.faculty import faculty_vitals as _vitals
    from ucbfaculty import FACULTY_NAME_DICTIONARY

    try:
        fac = _vitals()

        # Strategy 1: spelling dictionary correction -> exact match
        canonical_name = FACULTY_NAME_DICTIONARY.spell(name)
        exact = fac[fac["FullName"] == canonical_name]
        if not exact.empty:
            return _serialize_dataframe(exact, format)

        # Strategy 2: case-insensitive substring match on FullName
        mask = fac["FullName"].str.contains(name, case=False, na=False)
        subset = fac[mask]
        if not subset.empty:
            return _serialize_dataframe(subset, format)

        # Strategy 3: fuzzy match via most_similar
        try:
            from ligonlibrary.strings import most_similar
        except ImportError:
            from thefuzz import fuzz

            def most_similar(target, candidates, similarity_threshold=80):
                best, best_score = None, similarity_threshold
                for c in candidates:
                    score = fuzz.ratio(target.lower(), c.lower())
                    if score >= best_score:
                        best_score, best = score, c
                return best

        match = most_similar(name, fac["FullName"].dropna().tolist(), similarity_threshold=80)
        if match:
            result = fac[fac["FullName"] == match]
            return _serialize_dataframe(result, format)

        # No match above the threshold.  Surface the closest few names
        # so the caller can decide whether the input was a typo.
        suggestions = _closest_names(name, fac["FullName"].dropna().tolist(), n=3)
        if suggestions:
            return (
                f"No faculty member matching '{name}'.  "
                f"Closest names: {', '.join(suggestions)}."
            )
        return f"No faculty member matching '{name}'."
    except Exception as exc:
        return _handle_error(exc)


def _closest_names(target: str, candidates: list, n: int = 3) -> list:
    """Return the *n* candidates most similar to *target* (descending).

    Best-effort: requires ``thefuzz``.  Returns an empty list when fuzz
    is unavailable so the caller can fall back to a plain message.
    """
    try:
        from thefuzz import fuzz
    except ImportError:
        return []
    target_l = target.lower()
    scored = sorted(
        (
            (fuzz.ratio(target_l, c.lower()), c)
            for c in candidates
            if isinstance(c, str) and c
        ),
        reverse=True,
    )
    return [c for _, c in scored[:n]]


@mcp.tool()
def service_history(
    name: str,
    limit: Optional[int] = None,
    format: Optional[str] = None,
) -> str:
    """Return committee service history for a faculty member.

    Shows which committees they served on, their role, and how many
    consecutive years they have served (ServiceStreak / ChairStreak).

    The response is prefixed with a one-line ``Resolved '<input>' to
    <email> (<full name>).`` so callers know which person fuzzy
    matching landed on; the records themselves do not carry Email or
    FullName columns.

    Parameters
    ----------
    name : str
        Faculty name or email (e.g. ``"Ethan Ligon"`` or
        ``"ligon@berkeley.edu"``).  Fuzzy matching is supported.
    limit : int | None
        Max rows to return (default 25).
    format : str | None
        Output format: ``"org"`` (default), ``"markdown"``, or ``"json"``.
    """
    from ucbfaculty.faculty import FacultyMember

    try:
        # Try as email first, then as name
        if "@" in name:
            fm = FacultyMember(Email=name)
        else:
            fm = FacultyMember(FullName=name)

        if not fm.verified:
            return f"No faculty member matching '{name}'."

        header = f"Resolved '{name}' to {fm.Email} ({fm.FullName})."
        df = fm.service_history()
        if df.empty:
            return f"{header}\n(No service history on record.)"
        return f"{header}\n{_serialize_dataframe(df, format, limit=limit)}"
    except Exception as exc:
        return _handle_error(exc)


# ===================================================================
# 2. Small scalar / list helpers
# ===================================================================

@mcp.tool()
def committee_names(format: Optional[str] = None) -> str:
    """List all canonical committee acronyms (e.g. CAPBD, DIVCO).

    Parameters
    ----------
    format : str | None
        Output format: ``"org"`` (default), ``"markdown"``, or ``"json"``.
    """
    from ucbfaculty.faculty import committee_names as _committee_names

    try:
        names = _committee_names()
        return _serialize_list(names, format)
    except Exception as exc:
        return _handle_error(exc)


@mcp.tool()
def committee_year_range() -> str:
    """Return the (min, max) year range of committee service records.

    Returns a string like ``"2005 -- 2025"``, or an explanatory message
    if no records exist.
    """
    from ucbfaculty.faculty import committee_year_range as _year_range

    try:
        lo, hi = _year_range()
        if lo is None:
            return "(No committee records found)"
        return f"{lo} -- {hi}"
    except Exception as exc:
        return _handle_error(exc)


@mcp.tool()
def get_current_year() -> str:
    """Return the current academic year string (e.g. ``'2025-26'``)."""
    from ucbfaculty.faculty import get_current_year as _current_year

    try:
        return _current_year()
    except Exception as exc:
        return _handle_error(exc)


@mcp.tool()
def dept_name_to_code(dept_name: str, format: Optional[str] = None) -> str:
    """Map a department name (e.g. ``'Economics'``) to its short code.

    To browse all departments, ``query("SELECT DISTINCT DeptName, DeptCode FROM vitals")``.
    """
    from ucbfaculty.faculty import dept_name_to_code as _lookup

    try:
        result = _lookup(DeptName=dept_name)
        if result is None:
            return f"No department named '{dept_name}'."
        if isinstance(result, dict):
            return _serialize_dict(result, format)
        return str(result)
    except Exception as exc:
        return _handle_error(exc)


@mcp.tool()
def dept_code_to_name(dept_code: str, format: Optional[str] = None) -> str:
    """Map a department code (e.g. ``'SECON'``) to its full name.

    Codes are not always the obvious abbreviation --- ``'SECON'`` ->
    ``'Economics'``, ``'IMMCB'`` -> ``'Molecular & Cell Biology'``,
    ``'MBARC'`` -> ``'Agricultural & Resource Economics'``.  To browse
    all departments, ``query("SELECT DISTINCT DeptName, DeptCode FROM vitals")``.
    """
    from ucbfaculty.faculty import dept_code_to_name as _lookup

    try:
        result = _lookup(DeptCode=dept_code)
        if result is None:
            return f"No department code '{dept_code}'."
        if isinstance(result, dict):
            return _serialize_dict(result, format)
        return str(result)
    except Exception as exc:
        return _handle_error(exc)


# ===================================================================
# 3. Ingest / spec-proposal tools
# ===================================================================

@mcp.tool()
def ingest_dataframe(
    filepath: str,
    name_col: Optional[str] = None,
    email_col: Optional[str] = None,
    limit: Optional[int] = None,
    format: Optional[str] = None,
) -> str:
    """Load a CSV/Excel file and match rows to Berkeley faculty.

    Reads the file, runs ``reindex_to_email()`` to identify which rows
    correspond to known faculty (by name or email matching), and returns
    the matched subset indexed by canonical ``@berkeley.edu`` email.

    Parameters
    ----------
    filepath : str
        Path to a CSV or Excel file on disk.
    name_col : str | None
        Column containing names (auto-detected if omitted).
    email_col : str | None
        Column containing emails (auto-detected if omitted).
    limit : int | None
        Max rows to return (default 25).
    format : str | None
        Output format: ``"org"`` (default), ``"markdown"``, or ``"json"``.
    """
    from ucbfaculty.merge_features import ingest_dataframe as _ingest

    try:
        matched = _ingest(
            filepath,
            name_col=name_col,
            email_col=email_col,
            interactive=False,
        )
        if matched.empty:
            return "(No rows matched to Berkeley faculty)"
        return _serialize_dataframe(matched, format, limit=limit)
    except Exception as exc:
        return _handle_error(exc)


@mcp.tool()
def propose_spec(
    filepath: str,
    years: Optional[list[str]] = None,
) -> str:
    """Introspect a CSV/Excel file and generate draft YAML specs.

    Copies the file to ``Proposed/`` and writes candidate
    ``sources.yaml`` + ``features_spec.yaml`` stanzas.

    Parameters
    ----------
    filepath : str
        Path to a CSV or Excel file.
    years : list[str] | None
        Academic years this source covers (defaults to current year).

    Returns
    -------
    str
        The generated YAML text for both specs.
    """
    from ucbfaculty.propose_spec import propose_spec as _propose

    p = Path(filepath)
    if not p.exists():
        return f"File not found: {filepath}"

    try:
        result = _propose(filename=p, years=years)
        parts = []
        if result.get("sources_yaml"):
            parts.append("# sources.yaml stanza\n" + result["sources_yaml"])
        if result.get("features_spec_yaml"):
            parts.append("# features_spec.yaml stanza\n" + result["features_spec_yaml"])
        if result.get("data_file"):
            parts.append(f"\nData file copied to: {result['data_file']}")
        return "\n\n".join(parts) if parts else "(No specs generated)"
    except Exception as exc:
        return _handle_error(exc)


# ===================================================================
# 4. Unified query tools (general-purpose SQL + schema introspection)
# ===================================================================

@mcp.tool()
def query(
    sql: str,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    format: Optional[str] = None,
) -> str:
    """Run a read-only SQL query against the ucbfaculty data sources.

    Available views (registered as DuckDB tables):

    - ``vitals`` -- faculty roster with FullName, DeptName, Division,
      EmploymentCategory, etc.; one row per Email.
    - ``committees`` -- long-format committee membership history
      (Committee, Email, Year, Role, ...).
    - One view per feature returned by ``available_features()``, each
      with columns ``Email`` and ``<FeatureName>``.

    Use ``describe()`` to list the views and ``describe("<view>")`` for
    a view's columns and dtypes.  Views register lazily on first SQL
    reference, so include ``FROM <feature_name>`` (or ``JOIN``) and the
    server will load it.

    Parameters
    ----------
    sql : str
        DuckDB-compatible SQL.  Standard ``SELECT`` / ``JOIN`` /
        ``GROUP BY`` / window-function syntax is supported, including
        CTEs and subqueries.  Quote identifiers with double quotes for
        feature names that may shadow keywords.
    output_path : str | None
        When set, write the *full* result to disk and return a one-line
        confirmation plus a 5-row JSON preview, instead of inlining the
        whole table.  Use this when the result is too large to bring
        into context.  Format follows the path extension:

        - ``.csv``: CSV (preserves nothing dtype-wise but is readable
          via ``Read`` for eyeballing).
        - ``.parquet`` or ``.pq``: Parquet (preserves dtypes, smaller,
          and queryable directly via ``SELECT * FROM '<path>'``).
        - any other / no extension: defaults to Parquet, with
          ``.parquet`` appended to the path.

        After spilling to Parquet you can issue further queries against
        the file: ``query("SELECT ... FROM '/tmp/spill.parquet' ...")``.
    limit : int | None
        Cap rows returned (default 100 for json; 25 for org/markdown).
        Ignored when ``output_path`` is set: spills always contain the
        full result.
    format : str | None
        Output format: ``"json"`` (default), ``"org"``, or ``"markdown"``.
    """
    from pathlib import Path as _Path
    from ucbfaculty import queries as _queries
    import duckdb as _duckdb

    try:
        df = _queries.execute(sql)
    except _duckdb.Error as exc:
        return f"SQL error: {exc}"
    except Exception as exc:
        return _handle_error(exc)

    if output_path is not None:
        if df.empty:
            return "(No results -- nothing written)"

        path = _Path(output_path)
        ext = path.suffix.lower()
        try:
            if ext == ".csv":
                df.to_csv(path, index=False)
            elif ext in (".parquet", ".pq"):
                df.to_parquet(path)
            else:
                # Default to Parquet for ambiguous paths; append the
                # extension so downstream tooling can pick the right
                # reader by suffix.
                path = path.with_name(path.name + ".parquet")
                df.to_parquet(path)
        except Exception as exc:
            return _handle_error(exc)

        rows, cols = df.shape
        preview = _serialize_dataframe(df.head(5), "json")
        return (
            f"wrote {rows} rows x {cols} cols to {path}\n"
            f"first 5 rows:\n{preview}"
        )

    return _serialize_dataframe(
        df, format, limit=limit,
        hint=(
            "Add WHERE/LIMIT, aggregate (GROUP BY), or set "
            "output_path to spill the full result to disk."
        ),
    )


@mcp.tool()
def describe(
    view: Optional[str] = None,
    format: Optional[str] = None,
) -> str:
    """Describe the views available to ``query``.

    Parameters
    ----------
    view : str | None
        A view name.  When omitted, returns the lists of static,
        available, and currently-registered views.  When set, returns
        the view's columns (name + DuckDB type) and row count.
    format : str | None
        Output format: ``"json"`` (default), ``"org"``, or ``"markdown"``.
    """
    from ucbfaculty import queries as _queries
    import json as _json
    import pandas as _pd

    fmt = _resolve_format(format)
    try:
        info = _queries.describe(view)
    except KeyError as exc:
        return str(exc)
    except Exception as exc:
        return _handle_error(exc)

    if view is None:
        if fmt == "json":
            return _json.dumps(info)
        lines = []
        for key in ("static", "available", "registered"):
            entries = info.get(key, [])
            lines.append(f"{key}: {', '.join(entries) if entries else '(none)'}")
        return "\n".join(lines)

    if fmt == "json":
        return _json.dumps(info, default=str)

    cols_df = _pd.DataFrame(info["columns"], columns=["column", "type"])
    suffix = f"\nview: {info['view']}, rows: {info['rows']}"
    if fmt == "markdown":
        return cols_df.to_markdown(index=False) + suffix
    from tabulate import tabulate as _tab
    return _tab(cols_df, headers="keys", tablefmt="orgtbl", showindex=False) + suffix


# ===================================================================
# CLI entry point
# ===================================================================

def _build_parser() -> argparse.ArgumentParser:
    from ucbfaculty import __version__

    parser = argparse.ArgumentParser(
        prog="ucbfaculty-mcp",
        description="MCP server for querying UC Berkeley faculty data.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print the installed ucbfaculty version and exit.",
    )
    parser.add_argument(
        "--passphrase",
        help="GPG passphrase to pre-unlock RawData on startup.",
    )
    parser.add_argument(
        "--default-format",
        choices=["org", "markdown", "json"],
        default="json",
        help="Default output format for tabular data (default: json).",
    )
    return parser


def _suppress_pinentry() -> None:
    """Prevent GPG from spawning pinentry, which would hijack stdio.

    Sets environment variables so that ``gpg`` never attempts interactive
    passphrase prompts.  Without a passphrase pre-set via ``--passphrase``
    or gpg-agent, decrypt operations will fail cleanly with a
    ``DecryptionError`` instead of blocking on a terminal prompt.
    """
    import os
    os.environ.pop("GPG_TTY", None)
    os.environ["GPG_AGENT_INFO"] = ""
    # --batch is handled per-call in rawdata._gpg_decrypt, but unsetting
    # GPG_TTY ensures pinentry-curses/tty won't try to grab the terminal.


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and run the MCP server over stdio."""
    global _DEFAULT_FORMAT

    # CRITICAL: suppress pinentry before any GPG operation can fire
    _suppress_pinentry()

    parser = _build_parser()
    args = parser.parse_args(argv)

    _DEFAULT_FORMAT = args.default_format

    if args.passphrase:
        from ucbfaculty.security import unlock_rawdata as _unlock
        success, msg = _unlock(args.passphrase)
        if success:
            logger.info("Pre-unlock succeeded: %s", msg)
        else:
            logger.warning("Pre-unlock failed: %s", msg)

    logger.info(
        "Starting ucbfaculty MCP server (default_format=%s)", _DEFAULT_FORMAT,
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
