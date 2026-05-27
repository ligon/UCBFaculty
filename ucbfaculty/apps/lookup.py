"""
Interactive faculty lookup application built with Panel.

Run locally with:

    poetry run panel serve ucbfaculty/apps/lookup.py --autoreload
"""


from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
from io import BytesIO
import yaml

logger = logging.getLogger(__name__)

DEBUG_MODE = os.environ.get("UCBFACULTY_DEBUG", "").lower() in ("1", "true", "yes")

import pandas as pd
import panel as pn
import pyarrow  # noqa: F401 - ensure pyarrow is available for parquet
import ucbfaculty as uc

PASSGATE_CSS = """
.passphrase-input .bk-input-icon {
    display: none !important;
}
.passphrase-input input {
    padding-right: 10px !important;
}
.bk-password-input input::-ms-reveal,
.bk-password-input input::-ms-clear {
    display: none !important;
}
.bk-password-input input::-webkit-credentials-auto-fill-button,
.bk-password-input input::-webkit-textfield-decoration-container {
    display: none !important;
}
"""

ROOT = Path(__file__).resolve().parents[2]
LICENSE_PATH = ROOT / "LICENSE.org"
LICENSE_HTML_PATH = ROOT / "LICENSE.html"
LICENSE_SESSION_FILE = Path.home() / ".ucbfaculty" / "license_session.json"
LICENSE_SESSION_TTL = timedelta(hours=24)


class _SelectionBus:
    """Lightweight pub/sub helper so tabs can share selected faculty info."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[dict[str, str]], None]] = []
        self._payload: dict[str, str] = {}

    def subscribe(self, callback: Callable[[dict[str, str]], None]) -> None:
        self._listeners.append(callback)

    def publish(self, **payload: Optional[str]) -> None:
        clean: dict[str, str] = {
            key: value.strip()
            for key, value in payload.items()
            if isinstance(value, str) and value.strip()
        }
        if not clean:
            return
        self._payload.update(clean)
        for listener in list(self._listeners):
            listener(dict(self._payload))


def _selection_bus() -> _SelectionBus:
    bus = pn.state.cache.get("ucbfaculty_selection_bus")
    if not isinstance(bus, _SelectionBus):
        bus = _SelectionBus()
        pn.state.cache["ucbfaculty_selection_bus"] = bus
    return bus


def _publish_row_selection(record: Mapping[str, Any]) -> None:
    name_keys = (
        "FullName",
        "Faculty",
        "Member",
        "FacultyDisplay",
    )
    email_keys = ("Email", "email", "Email (feature)")

    def _first(keys: Iterable[str]) -> Optional[str]:
        for key in keys:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    name = _first(name_keys)
    email = _first(email_keys)
    if not name and not email:
        return
    _selection_bus().publish(name=name, email=email)


def _wire_row_click(table: pn.widgets.Tabulator) -> pn.widgets.Tabulator:
    """Attach a cell-click handler so row data updates the shared selection."""

    def _handle_event(event) -> None:
        row_index = getattr(event, "row", None)
        if row_index is None:
            return
        df = getattr(table, "value", None)
        if df is None:
            return
        try:
            record = df.iloc[int(row_index)]
        except Exception as exc:  # noqa: BLE001 - guard against stale indexes
            logger.debug("Row click lookup failed (stale index?): %s", exc)
            return
        _publish_row_selection(record.to_dict())

    table.on_click(_handle_event)
    return table

if __package__ in (None, ""):
    # Allow running via `panel serve ucbfaculty/apps/lookup.py`
    import sys

    sys.path.insert(0, str(ROOT))
    from ucbfaculty.merge_features import build_feature_dataframe  # type: ignore
    from ucbfaculty import security  # type: ignore
    from ucbfaculty.rawdata import DecryptionError  # type: ignore
    from ucbfaculty.faculty import (  # type: ignore
        FACULTY_NAME_DICTIONARY,
        FacultyMember,
        committee_membership,
        committee_names,
        committee_year_range,
        clear_faculty_caches,
    )
    from ucbfaculty.utils import name_variants as _name_variants  # type: ignore
else:
    from ..merge_features import build_feature_dataframe
    from ..faculty import (
        FACULTY_NAME_DICTIONARY,
        FacultyMember,
        committee_membership,
        committee_names,
        committee_year_range,
        clear_faculty_caches,
    )
    from ..utils import name_variants as _name_variants
    from .. import security
    from ..rawdata import DecryptionError

# Clear caches on module reload to pick up dictionary updates
clear_faculty_caches()


@lru_cache
def _faculty_reference_table() -> pd.DataFrame:
    """Return a cached table of core faculty attributes indexed by Email."""
    start = time.perf_counter()
    features = [
        "FullName",
        "FirstName",
        "LastName",
        "Email",
        "DeptName",
        "Division",
        "EmploymentCategory",
        "Affiliations",
        "UID",
    ]
    df = build_feature_dataframe(features, interactive=False)

    # Merge helper may emit lists when duplicate values exist; flatten to strings.
    def _stringify(value):
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return value

    for column in df.columns:
        df[column] = df[column].map(_stringify)

    elapsed = time.perf_counter() - start
    logger.info("_faculty_reference_table completed in %.2f seconds", elapsed)
    return df


@lru_cache
def _available_feature_names() -> tuple[str, ...]:
    """Fetch sorted feature names from the feature spec."""
    try:
        from ..new_feature import available_features
    except ImportError:
        from ucbfaculty.new_feature import available_features
    return tuple(available_features())


def _read_license_session() -> Optional[dict]:
    """Load cached license-acceptance session if still valid."""
    try:
        with LICENSE_SESSION_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        LICENSE_SESSION_FILE.unlink(missing_ok=True)
        return None

    timestamp_raw = payload.get("timestamp")
    if not timestamp_raw:
        LICENSE_SESSION_FILE.unlink(missing_ok=True)
        return None

    try:
        timestamp = datetime.fromisoformat(timestamp_raw)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
    except ValueError:
        LICENSE_SESSION_FILE.unlink(missing_ok=True)
        return None

    if datetime.now(timezone.utc) - timestamp > LICENSE_SESSION_TTL:
        LICENSE_SESSION_FILE.unlink(missing_ok=True)
        return None

    return payload


def _write_license_session(username: str) -> None:
    """Persist the fact that the user accepted the license."""
    LICENSE_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "username": username.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with LICENSE_SESSION_FILE.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)


@lru_cache
def _committee_feature_options() -> tuple[str, ...]:
    """Return curated feature names suitable for committee enrichment.

    This function is cached because it iterates through all available features
    and tests each one for validity, which can be slow (~2-3 seconds).
    """
    start = time.perf_counter()
    available = sorted(_available_feature_names())
    options: list[str] = []
    for name in available:
        # Skip canonical helper features that are implicitly included elsewhere
        if name in {"FullName"}:
            continue
        try:
            frame = _committee_feature_frame((name,))
        except Exception as exc:
            logger.debug("Skipping feature %r for committee options: %s", name, exc)
            continue
        if frame is not None and "Email" in frame.columns:
            options.append(name)
    elapsed = time.perf_counter() - start
    logger.info("_committee_feature_options completed in %.2f seconds", elapsed)
    return tuple(options)


@lru_cache
def _committee_feature_frame(feature_names: tuple[str, ...]) -> Optional[pd.DataFrame]:
    """
    Load feature data used to enrich committee membership rows.
    Returns a DataFrame with Email, FullName, and requested features.
    """
    start = time.perf_counter()
    if not feature_names:
        return None

    requested = set(feature_names)
    standard_features = sorted(requested.difference({"Appointment"}))
    base_feature_list = sorted(set(standard_features) | {"FullName"})

    try:
        base_features = build_feature_dataframe(base_feature_list, interactive=False)
    except (FileNotFoundError, ValueError, KeyError, DecryptionError) as exc:
        if isinstance(exc, DecryptionError):
            reauth = pn.state.cache.get("_reauth_callback")
            if reauth:
                reauth()
        elapsed = time.perf_counter() - start
        logger.info("_committee_feature_frame(%s) failed in %.2f seconds", feature_names, elapsed)
        return None

    feature_df = base_features.reset_index()
    keep = ["Email", "FullName"]
    keep.extend([name for name in standard_features if name in feature_df.columns])
    result = feature_df[keep]

    if "Appointment" in requested:
        current_map = _current_committee_map()
        if not current_map.empty:
            def _to_string(entries):
                display = []
                for committee, role in entries:
                    if role:
                        display.append(f"{committee} ({role})")
                    else:
                        display.append(committee)
                return ", ".join(display)

            current_strings = current_map.map(_to_string).reset_index()
            current_strings.columns = ["Email", "Appointment"]
            result = result.merge(current_strings, on="Email", how="left")

    elapsed = time.perf_counter() - start
    logger.info("_committee_feature_frame(%s) completed in %.2f seconds", feature_names, elapsed)
    return result


@lru_cache
def _current_committee_map() -> pd.Series:
    """Return a Series mapping Email to formatted current committee assignments."""
    try:
        try:
            from ..new_feature import Feature
        except ImportError:
            from ucbfaculty.new_feature import Feature
        from ..new_feature import Features
        current_raw = Features(["Appointment", "Role"])["latest"].reset_index()
    except Exception as exc:
        logger.debug("Failed to load Appointment feature: %s", exc)
        return pd.Series(dtype=object)

    # Email is already in the index from Feature("Appointment")
    current_raw = current_raw.dropna(subset=["Email"])
    current_raw["Entry"] = current_raw.apply(
        lambda row: (
            str(row["Appointment"]),
            str(row["Role"]).strip() if pd.notna(row.get("Role")) else "",
        ),
        axis=1,
    )
    grouped = current_raw.groupby("Email")["Entry"].apply(lambda series: list(dict.fromkeys(series)))
    return grouped



@lru_cache
def _license_text() -> str:
    """Read the full license text from disk (cached)."""
    try:
        return LICENSE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "Full license text is unavailable. Please contact the project maintainer."


@lru_cache
def _license_markdown() -> str:
    """Render the Org-mode license text into coarse Markdown for display."""
    text = _license_text()
    lines: list[str] = []
    in_quote = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#+TITLE:"):
            lines.append(f"# {stripped.split(':', 1)[1].strip()}")
            continue
        if stripped.startswith("#+SUBTITLE:"):
            lines.append(f"**{stripped.split(':', 1)[1].strip()}**")
            continue
        if stripped.startswith("#+AUTHOR:"):
            lines.append(f"*Author:* {stripped.split(':', 1)[1].strip()}")
            continue
        if stripped.startswith("#+BEGIN_QUOTE"):
            in_quote = True
            continue
        if stripped.startswith("#+END_QUOTE"):
            in_quote = False
            lines.append("")
            continue
        if stripped.startswith("#+INCLUDE:"):
            target = stripped.split(None, 1)[1].strip().strip('"')
            lines.append(f"[Full AGPL text]({target})")
            continue
        if stripped.startswith("#+BEGIN_CENTER"):
            lines.append("")
            continue
        if stripped.startswith("#+END_CENTER"):
            lines.append("")
            continue
        if stripped.startswith("*"):
            level = len(stripped) - len(stripped.lstrip("*"))
            heading = stripped.lstrip("*").strip()
            level = max(level, 1) + 1
            lines.append(f"{'#' * level} {heading}")
            continue
        if in_quote:
                lines.append(f"> {raw}")
                continue
        lines.append(raw)
    return "\n".join(lines)


@lru_cache
def _license_html() -> Optional[str]:
    """Return the HTML version of the license if available."""
    if LICENSE_HTML_PATH.exists():
        return LICENSE_HTML_PATH.read_text(encoding="utf-8")
    return None


def _download_buttons(
    df: pd.DataFrame,
    base_filename: str,
    include_index: bool = True,
) -> pn.Column:
    """Create CSV/Parquet download buttons and a clipboard copy button.

    Parquet preserves dtype metadata (Int64, string, category, etc.) and index
    structure, making it ideal for downstream analysis.
    """
    def _csv_callback():
        return BytesIO(df.to_csv(index=include_index).encode("utf-8"))

    def _parquet_callback():
        buf = BytesIO()
        df.to_parquet(buf, index=include_index, engine="pyarrow")
        buf.seek(0)
        return buf

    csv_btn = pn.widgets.FileDownload(
        label="CSV",
        filename=f"{base_filename}.csv",
        button_type="primary",
        callback=_csv_callback,
        width=100,
        height=40,
        css_classes=["download-csv-btn"],
    )
    parquet_btn = pn.widgets.FileDownload(
        label="Parquet",
        filename=f"{base_filename}.parquet",
        button_type="success",
        callback=_parquet_callback,
        width=100,
        height=40,
        css_classes=["download-parquet-btn"],
    )

    # Clipboard button with JavaScript copy action
    csv_text = df.to_csv(index=include_index, sep="\t")  # tab-separated for Sheets
    copy_btn = pn.widgets.Button(
        name="Copy",
        button_type="light",
        width=80,
        height=40,
        description="Copy to clipboard (paste into Google Sheets)",
    )
    copy_btn.js_on_click(
        args={"csv_data": csv_text},
        code="""
            navigator.clipboard.writeText(csv_data).then(function() {
                // Brief visual feedback
                var origLabel = cb_obj.label;
                cb_obj.label = 'Copied!';
                setTimeout(function() { cb_obj.label = origLabel; }, 1500);
            });
        """,
    )

    button_row = pn.Row(csv_btn, parquet_btn, copy_btn, sizing_mode="fixed")

    # Add tooltips via JavaScript since FileDownload doesn't support description tooltips
    # Panel uses shadow DOM, so we must recursively traverse shadow roots to find buttons
    tooltip_script = pn.pane.HTML("""
        <script>
        (function() {
            function addTooltips(root, depth) {
                if (depth > 15) return;
                // Find wrapper divs with our CSS classes
                root.querySelectorAll('.download-csv-btn').forEach(wrapper => {
                    var btn = wrapper.querySelector('button');
                    if (!btn && wrapper.shadowRoot) btn = wrapper.shadowRoot.querySelector('button');
                    if (btn) btn.title = 'Download as CSV file';
                });
                root.querySelectorAll('.download-parquet-btn').forEach(wrapper => {
                    var btn = wrapper.querySelector('button');
                    if (!btn && wrapper.shadowRoot) btn = wrapper.shadowRoot.querySelector('button');
                    if (btn) btn.title = 'Download as Parquet (preserves column types)';
                });
                // Recurse into shadow roots
                root.querySelectorAll('*').forEach(el => {
                    if (el.shadowRoot) addTooltips(el.shadowRoot, depth + 1);
                });
            }
            function run() { addTooltips(document, 0); }
            run();
            setTimeout(run, 500);
            setTimeout(run, 1500);
        })();
        </script>
    """, height=0, width=0, sizing_mode="fixed", margin=0)

    return pn.Column(button_row, tooltip_script, sizing_mode="fixed", margin=0)



def _load_config_payload() -> dict[str, Any]:
    """Return the current config content if available."""
    try:
        return uc.load_config()
    except FileNotFoundError:
        return {}


def _sync_rawdata_directory() -> tuple[bool, str]:
    """Run git pull inside the configured RawData directory."""
    path = uc.datadir
    if not path.exists():
        return False, f"Data directory not found: {path}"
    if not (path / ".git").exists():
        return False, f"No git repository detected at {path}"
    command = ["git", "-C", str(path), "pull", "--ff-only"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return False, "git executable not found on PATH."
    if result.returncode != 0:
        message = result.stderr.strip() or "git pull failed."
        return False, message
    output = result.stdout.strip() or "Already up to date."
    return True, output


def build_splash_view(on_accept: Callable[[dict[str, str]], None]) -> pn.viewable.Viewable:
    """Create a splash screen that captures credentials and confirms license acceptance."""
    username = pn.widgets.TextInput(name="Username", placeholder="Enter your CalNet ID or email")
    existing_passphrase = security.get_gpg_passphrase()
    passphrase = pn.widgets.PasswordInput(
        name="Passphrase",
        placeholder="Enter passphrase (required)",
        value=existing_passphrase or "",
    )
    accept_terms = pn.widgets.Checkbox(name="I have read and accept the license terms", value=False)
    proceed = pn.widgets.Button(name="Enter UCB Faculty Tools", button_type="primary", disabled=True, width=260)

    for widget in (username, passphrase):
        widget.width = 260
        widget.height = 28
    accept_terms.width = 260

    summary = pn.pane.Markdown(
        """### UCB Faculty Tools

Before accessing faculty data tools you must affirm the license terms:

- Use the application only for lawful academic employment and service workflows.
- Comply with nondiscrimination requirements outlined in the ENAFA-CA addendum.
- Preserve academic freedom protections when interpreting results.

Expand the full license below or download a copy for your records.""",
        sizing_mode="stretch_width",
    )

    html_license = _license_html()
    if html_license:
        license_download = pn.widgets.FileDownload(
            label="Download License (HTML)",
            filename="LICENSE.html",
            button_type="default",
            callback=lambda data=html_license: BytesIO(data.encode("utf-8")),
            width=160,
        )
        license_panel = pn.pane.HTML(
            html_license,
            height=360,
            sizing_mode="stretch_width",
            styles={"overflow-y": "auto"},
        )
    else:
        license_download = pn.widgets.FileDownload(
            label="Download License (Org)",
            filename="LICENSE.org",
            button_type="default",
            callback=lambda: BytesIO(_license_text().encode("utf-8")),
            width=160,
        )
        license_panel = pn.pane.Markdown(
            _license_markdown(),
            height=360,
            sizing_mode="stretch_width",
            styles={"overflow-y": "auto", "white-space": "pre-wrap"},
        )
    license_download.sizing_mode = "fixed"
    license_view = pn.Accordion(
        (
            "Full License (click to expand)",
            license_panel,
        ),
        active=[],
        sizing_mode="stretch_width",
    )

    def _refresh_button(*_):
        proceed.disabled = not (
            username.value.strip()
            and passphrase.value.strip()
            and accept_terms.value
        )

    username.param.watch(_refresh_button, "value")
    passphrase.param.watch(_refresh_button, "value")
    accept_terms.param.watch(_refresh_button, "value")
    _refresh_button()

    error_display = pn.pane.Markdown(
        "", visible=False, sizing_mode="stretch_width",
        styles={"color": "#cc3333", "font-size": "0.9em"},
    )

    def _handle_accept(_=None):
        raw_passphrase = passphrase.value
        success, message = security.unlock_rawdata(raw_passphrase)
        if not success:
            error_display.object = f"⚠️ {message}"
            error_display.visible = True
            return
        error_display.visible = False
        credentials = {
            "username": username.value.strip(),
            "passphrase": "",
        }
        pn.state.cache["ucbfaculty_auth"] = credentials
        _write_license_session(credentials["username"])
        on_accept(credentials)

    proceed.on_click(_handle_accept)

    form = pn.WidgetBox(
        pn.pane.Markdown("#### Authenticate to Continue"),
        username,
        passphrase,
        error_display,
        accept_terms,
        proceed,
        width=320,
    )

    left_panel = pn.Column(summary, form, sizing_mode="stretch_height", min_width=500)
    license_header = pn.pane.Markdown("#### License Reference", sizing_mode="stretch_width")
    right_panel = pn.Column(
        license_header,
        license_download,
        license_view,
        sizing_mode="stretch_height",
        min_width=360,
        align="start",
    )

    return pn.Column(
        pn.Spacer(height=30),
        pn.Row(
            pn.Spacer(width=20),
            left_panel,
            pn.Spacer(width=30),
            right_panel,
            pn.Spacer(width=20),
            sizing_mode="stretch_width",
            align="start",
        ),
        pn.Spacer(height=30),
        sizing_mode="stretch_both",
    )


def _filter_faculty(
    last_name: str,
    first_name: str,
    email: str,
    department: str,
    division: str,
    employment_groups: Iterable[str],
    search_fields: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return rows matching the provided search criteria."""
    df = _faculty_reference_table()
    mask = pd.Series(True, index=df.index)

    if email:
        mask &= df.index.str.contains(email.strip(), case=False, na=False)
    if last_name:
        mask &= df["LastName"].astype(str).str.contains(last_name.strip(), case=False, na=False)
    if first_name:
        mask &= df["FirstName"].astype(str).str.contains(first_name.strip(), case=False, na=False)
    if department:
        mask &= df["DeptName"].astype(str).str.contains(department.strip(), case=False, na=False)
    if division:
        mask &= df["Division"].astype(str).str.contains(division.strip(), case=False, na=False)
    if employment_groups:
        groups = {g for g in employment_groups if g}
        if groups:
            def _matches_group(value):
                if isinstance(value, list):
                    return any(v in groups for v in value)
                if isinstance(value, str):
                    tokens = [v.strip() for v in value.split(',') if v.strip()]
                    return any(token in groups for token in tokens)
                return False

            mask &= df["EmploymentCategory"].apply(_matches_group)

    filtered = df[mask]
    if search_fields:
        filtered = filtered[list(search_fields)]
    return filtered.sort_index()


def _render_faculty_details(matches: pd.DataFrame) -> pn.viewable.Viewable:
    """Render details for zero, one, or many matches."""
    if matches.empty:
        return pn.pane.Markdown("⚠️ No matching faculty member found.", sizing_mode="stretch_width")

    if len(matches) > 1:
        sort_columns = [col for col in ("LastName", "FirstName") if col in matches.columns]
        if sort_columns:
            matches = matches.sort_values(sort_columns, kind="stable")
        display_cols = ["FullName", "FirstName", "LastName", "DeptName", "EmploymentCategory"]
        existing_cols = [c for c in display_cols if c in matches.columns]
        df_display = matches[existing_cols].copy()
        index_name = df_display.index.name or "Email"
        df_display = df_display.reset_index().rename(columns={"index": index_name})
        table = pn.widgets.Tabulator(
            df_display,
            sizing_mode="stretch_both",
            min_height=400,
            show_index=False,
            pagination=None,
            theme="simple",
            selectable=1,
        )
        table.editors = {column: None for column in df_display.columns}
        _wire_row_click(table)
        download_row = _download_buttons(
            matches[existing_cols],
            "faculty_matches",
            include_index=True,
        )
        return pn.Column(
            download_row,
            pn.pane.Markdown(
                f"🔍 {len(matches)} matches found. Refine your search or click a row to load that person in the details pane.",
                sizing_mode="stretch_width",
            ),
            table,
        )

    # Exactly one match
    email = matches.index[0]
    row = matches.iloc[0]

    member = FacultyMember(Email=email)

    try:
        service_history = member.service_history(include_current=True, with_year_start=True)
        history_reset = service_history.reset_index() if hasattr(service_history, "reset_index") else service_history
        years_by_committee = {}
        chair_streaks = {}
        if hasattr(history_reset, "empty") and not history_reset.empty:
            # Use streak columns from service_history when available
            history_reset = history_reset.sort_values(["Committee", "YearStart"])
            latest = history_reset.groupby("Committee", observed=True).tail(1)
            years_by_committee = latest.set_index("Committee")["ServiceStreak"].to_dict()
            if "ChairStreak" in latest.columns:
                chair_streaks = latest.set_index("Committee")["ChairStreak"].to_dict()
        service_error = None
    except Exception as exc:  # noqa: BLE001 - guard user display
        service_history = pd.DataFrame()
        years_by_committee = {}
        chair_streaks = {}
        service_error = exc

    assignments = _current_committee_map().get(email, [])
    entries = []
    for committee, role in assignments:
        years_served = None if service_error else max(int(years_by_committee.get(committee, 0)), 1) if hasattr(years_by_committee, "get") else 1
        chair_years = None if service_error else int(chair_streaks.get(committee, 0)) if hasattr(chair_streaks, "get") else 0
        base_label = f"{committee} ({role})" if role else committee
        if years_served is None:
            suffix = " — years unavailable"
        else:
            suffix = f" — {years_served} year" + ("s" if years_served != 1 else "")
            if chair_years:
                suffix += f" (chair streak: {chair_years})"
        entries.append(f"{base_label}{suffix}")
    entries = list(dict.fromkeys(entry for entry in entries if entry))

    if entries:
        current_panel_content = pn.pane.Markdown(
            "**Current Committee Assignments**\n\n" + "\n".join(f"- {entry}" for entry in entries),
            sizing_mode="stretch_width",
        )
    else:
        current_panel_content = None

    header = pn.pane.Markdown(
        f"### {member.FullName or row.get('FullName') or email}\n"
        f"- **Email:** {member.Email or email}\n"
        f"- **UID:** {member.UID or row.get('UID', '—')}\n"
        f"- **Department:** {member.DeptName or row.get('DeptName', '—')}\n"
        f"- **Employment:** {row.get('EmploymentCategory', '—')}\n"
        f"- **Affiliations:** {row.get('Affiliations', '—')}",
        sizing_mode="stretch_width",
    )

    if service_error:
        service_panel = pn.pane.Markdown(f"Service history unavailable: {service_error}", sizing_mode="stretch_width")
    else:
        try:
            history_display = service_history.drop(columns=["YearStart"], errors="ignore")
            service_panel = pn.Accordion(
                ("Service History", pn.pane.DataFrame(history_display, sizing_mode="stretch_width")),
                sizing_mode="stretch_width",
                active=[0] if not history_display.empty else [],
            )
        except Exception as exc:  # noqa: BLE001 - guard user display
            service_panel = pn.pane.Markdown(f"Service history unavailable: {exc}", sizing_mode="stretch_width")

    volunteer_info = member.volunteer_info(as_df=True)
    if volunteer_info is not None and not getattr(volunteer_info, "empty", False):
        volunteer_panel = pn.Accordion(
            ("Volunteer Info", pn.pane.DataFrame(volunteer_info, sizing_mode="stretch_width")),
            sizing_mode="stretch_width",
        )
    else:
        volunteer_panel = pn.pane.Markdown("No volunteer information recorded.", sizing_mode="stretch_width")

    panels = [header]
    if current_panel_content is not None:
        panels.append(current_panel_content)
    panels.extend([service_panel, volunteer_panel])
    return pn.Column(*panels, sizing_mode="stretch_width", scroll=True)


def build_lookup_view() -> pn.viewable.Viewable:
    """Construct the Panel view for faculty lookup."""
    start = time.perf_counter()
    last_name = pn.widgets.TextInput(name="Last Name", placeholder="e.g., Ligon")
    first_name = pn.widgets.TextInput(name="First Name", placeholder="Optional")
    email = pn.widgets.TextInput(name="Email", placeholder="Optional")
    department = pn.widgets.TextInput(name="Department", placeholder="Optional")
    divisions = sorted({
        item
        for values in _faculty_reference_table()["Division"]
        for item in (values if isinstance(values, list) else [values])
        if isinstance(item, str) and item.strip()
    })
    divisions.insert(0, "")
    division_select = pn.widgets.Select(name="Division", options=divisions)

    employment_values = _faculty_reference_table()["EmploymentCategory"].dropna()
    employment_categories = set()
    for value in employment_values:
        if isinstance(value, list):
            employment_categories.update(v for v in value if isinstance(v, str) and v.strip())
        elif isinstance(value, str):
            employment_categories.update(v.strip() for v in value.split(',') if v.strip())
    employment_categories = sorted(employment_categories)
    label_map = {
        "Active Faculty": "Active Faculty",
        "Emeritus (Retired)": "Emeritus (Retired)",
        "Emeritus (Recalled)": "Emeritus (Recalled)",
        "Deans/Admin": "Deans/Admin",
    }
    employment_labels = [label_map.get(cat, cat) for cat in employment_categories]
    label_to_value = dict(zip(employment_labels, employment_categories))
    employment_select = pn.widgets.CheckBoxGroup(
        name="Employment",
        options=employment_labels,
        value=employment_labels,  # All categories selected by default
        inline=False,
    )

    def _apply_selection(payload: dict[str, str]) -> None:
        if not payload:
            return
        incoming_email = payload.get("email")
        incoming_name = payload.get("name")
        if incoming_email and email.value != incoming_email:
            email.value = incoming_email
            first_name.value = ""
            last_name.value = ""
        elif incoming_name and incoming_name != last_name.value:
            email.value = ""
            cleaned = incoming_name.replace("\u2019", "'")
            if "," in cleaned:
                last, first = [part.strip() for part in cleaned.split(",", 1)]
            else:
                parts = cleaned.split()
                last = parts[-1] if parts else ""
                first = " ".join(parts[:-1])
            last_name.value = last
            first_name.value = first
        department.value = ""
        division_select.value = ""
        employment_select.value = employment_labels  # Reset to all selected

    _selection_bus().subscribe(_apply_selection)

    for widget in (last_name, first_name, email):
        widget.width = 260
        widget.height = 28
    department.width = 260
    department.height = 28
    division_select.width = 260
    division_select.height = 28
    employment_select.width = 260

    @pn.depends(last_name, first_name, email, department, division_select, employment_select)
    def details_view(last: str, first: str, mail: str, dept: str, div: str, employment: tuple[str, ...]):
        selected_categories = [label_to_value.get(label, label) for label in employment]
        matches = _filter_faculty(last, first, mail, dept, div, selected_categories)
        return _render_faculty_details(matches)

    sidebar = pn.Column(
        pn.pane.Markdown("#### Search Filters"),
        last_name,
        first_name,
        email,
        department,
        division_select,
        employment_select,
        pn.Spacer(height=15),
        pn.pane.Markdown(
            "Filter by name, email, department, division, or employment category. "
            "Results update automatically when exactly one match remains.",
            sizing_mode="stretch_width",
        ),
        width=280,
        sizing_mode="stretch_height",
    )
    result = pn.Row(sidebar, pn.Column(details_view, sizing_mode="stretch_both"), sizing_mode="stretch_both")
    elapsed = time.perf_counter() - start
    logger.info("build_lookup_view completed in %.2f seconds", elapsed)
    return result


def build_feature_selection_view() -> pn.viewable.Viewable:
    """Construct a tab that lets users assemble merged feature tables."""
    start = time.perf_counter()
    options = list(_available_feature_names())
    default_selection = [f for f in ("FullName", "DeptName") if f in options]
    if not default_selection:
        default_selection = options[:3]

    feature_select = pn.widgets.MultiSelect(
        name="Features",
        options=options,
        value=default_selection,
        size=min(18, max(10, len(options))),
        sizing_mode="stretch_width",
    )
    dropna_checkbox = pn.widgets.Checkbox(name="Drop rows with missing values", value=False)

    @pn.depends(feature_select, dropna_checkbox)
    def render_table(selected: list[str], dropna: bool):
        if not selected:
            return pn.pane.Markdown("Select at least one feature to build a table.", sizing_mode="stretch_width")

        selected = list(dict.fromkeys(selected))
        special_features = {"Appointment"}
        requested_special = special_features.intersection(selected)
        requested_standard = [feature for feature in selected if feature not in special_features]

        # Ensure canonical mapping (FullName) always included for context
        requested = list(dict.fromkeys(["FullName"] + requested_standard))
        try:
            df = build_feature_dataframe(requested, interactive=False)
        except (FileNotFoundError, ValueError, KeyError, DecryptionError) as exc:
            if isinstance(exc, DecryptionError):
                reauth = pn.state.cache.get("_reauth_callback")
                if reauth:
                    reauth()
                    return pn.pane.Markdown("")
            return pn.pane.Markdown(
                f"⚠️ Unable to load requested features because a source dataset could not be read:\n``{exc}``",
                sizing_mode="stretch_width",
            )

        if requested_special:
            enrichment = _committee_feature_frame(tuple(requested_special))
            if enrichment is not None and "Email" in enrichment.columns:
                enrichment = enrichment.set_index("Email")
                columns = [feature for feature in requested_special if feature in enrichment.columns]
                if columns:
                    df = df.join(enrichment[columns], how="left")

        if dropna:
            df = df.dropna()

        if df.empty:
            return pn.pane.Markdown("No rows available after applying the current filters.", sizing_mode="stretch_width")

        df_display = df.copy()
        index_name = df_display.index.name or "index"
        if index_name in df_display.columns:
            df_display = df_display.rename(columns={index_name: f"{index_name} (feature)"})

        df_display = df_display.reset_index().rename(columns={"index": index_name})

        summary = pn.pane.Markdown(
            f"Showing **{len(df_display):,}** rows and **{len(df_display.columns):,}** columns.",
            sizing_mode="stretch_width",
        )
        hint = pn.pane.Markdown(
            "Click a row to jump to that faculty member in the lookup tab.",
            sizing_mode="stretch_width",
            margin=(0, 0, 10, 0),
        )
        table = pn.widgets.Tabulator(
            df_display,
            sizing_mode="stretch_both",
            min_height=600,
            show_index=False,
            pagination=None,
            theme="simple",
            selectable=1,
        )
        table.editors = {column: None for column in df_display.columns}
        _wire_row_click(table)
        download_row = _download_buttons(
            df_display,
            "faculty_features",
            include_index=False,
        )

        return pn.Column(download_row, summary, hint, table, sizing_mode="stretch_both")

    controls = pn.Column(
        pn.pane.Markdown("#### Feature Selection"),
        feature_select,
        dropna_checkbox,
        width=280,
        sizing_mode="stretch_height",
    )
    result = pn.Row(
        controls,
        pn.Column(render_table, sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
    )
    elapsed = time.perf_counter() - start
    logger.info("build_feature_selection_view completed in %.2f seconds", elapsed)
    return result


def build_committee_view() -> pn.viewable.Viewable:
    """Create a committee-centric exploration view."""
    start = time.perf_counter()
    committees = committee_names()
    min_year, max_year = committee_year_range()
    feature_options = _committee_feature_options()

    if not committees or min_year is None or max_year is None:
        elapsed = time.perf_counter() - start
        logger.info("build_committee_view completed in %.2f seconds (no data)", elapsed)
        return pn.pane.Markdown(
            "Committee records are unavailable. Ensure past committee data is present in RawData.",
            sizing_mode="stretch_width",
        )

    default_start = max(min_year, max_year - 9)
    committee_select = pn.widgets.Select(name="Committee", options=committees, value=committees[0], width=260, height=60)
    year_slider = pn.widgets.IntRangeSlider(
        name="Academic Year Range",
        start=min_year,
        end=max_year,
        value=(default_start, max_year),
        step=1,
    )
    default_feature_selection = []
    feature_select = pn.widgets.MultiChoice(
        name="Additional Features",
        options=list(feature_options),
        value=default_feature_selection,
        width=260,
    )

    def render_table(committee: str, year_range: tuple[int, int], features: list[str]):
        start_year, end_year = year_range
        df = committee_membership(committee=committee, start_year=start_year, end_year=end_year)
        df = df.sort_values("YearStart", ascending=False)
        selected_features = tuple(sorted(set(features)))

        if selected_features:
            try:
                feature_frame = _committee_feature_frame(selected_features)
            except DecryptionError as exc:
                reauth = pn.state.cache.get("_reauth_callback")
                if reauth:
                    reauth()
                    return pn.pane.Markdown("")
                return pn.pane.Markdown(str(exc), sizing_mode="stretch_width")
            if feature_frame is None:
                return pn.pane.Markdown(
                    "⚠️ Unable to load the selected features. Ensure RawData assets are available.",
                    sizing_mode="stretch_width",
                )

            feature_frame = feature_frame.copy()

            merge_columns = ["Email"] + [name for name in selected_features if name in feature_frame.columns]
            feature_subset = feature_frame[merge_columns].drop_duplicates(subset="Email")
            df = df.merge(feature_subset, on="Email", how="left")

        base_columns = ["Academic Year", "FullName", "Role", "DeptName", "Email"]
        base_columns = [col for col in base_columns if col in df.columns]
        feature_columns = [name for name in selected_features if name in df.columns]
        columns_for_display = base_columns + [col for col in feature_columns if col not in base_columns]

        if not columns_for_display:
            return pn.Column(
                pn.pane.Markdown(
                    "No displayable columns available for this selection.",
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            )

        df_display = df[columns_for_display].reset_index(drop=True)

        ordered_columns = [
            "Academic Year",
            "FullName",
            "Role",
            "DeptName",
            "Email",
        ]
        ordered_columns.extend([name for name in feature_columns])
        display_columns = [col for col in ordered_columns if col in df_display.columns]
        display_columns.extend([col for col in df_display.columns if col not in display_columns])
        df_display = df_display[display_columns]

        if df_display.empty:
            return pn.Column(
                pn.pane.Markdown(
                    f"No committee membership records found for **{committee}** between {start_year}–{end_year}.",
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            )

        summary = pn.pane.Markdown(
            f"**{committee}**: {len(df_display)} appointments between {start_year}–{end_year}.",
            sizing_mode="stretch_width",
        )
        hint = pn.pane.Markdown(
            "Click a row to open that person in the Faculty Lookup tab.",
            sizing_mode="stretch_width",
            margin=(0, 0, 10, 0),
        )
        table = pn.widgets.Tabulator(
            df_display,
            min_height=600,
            sizing_mode="stretch_both",
            show_index=False,
            pagination=None,
            theme="simple",
            selectable=1,
        )
        table.editors = {column: None for column in df_display.columns}
        _wire_row_click(table)
        download_row = _download_buttons(
            df_display,
            f"{committee.lower().replace(' ', '_')}_membership",
            include_index=False,
        )
        return pn.Column(download_row, summary, hint, table, sizing_mode="stretch_both")

    controls = pn.Column(
        pn.pane.Markdown("#### Committee Selection"),
        committee_select,
        year_slider,
        pn.pane.Markdown("#### Additional Features"),
        feature_select,
        width=280,
        sizing_mode="stretch_height",
    )
    result = pn.Row(
        controls,
        pn.Column(
            pn.bind(
                render_table,
                committee=committee_select,
                year_range=year_slider,
                features=feature_select,
            ),
            sizing_mode="stretch_both",
        ),
        sizing_mode="stretch_both",
    )
    elapsed = time.perf_counter() - start
    logger.info("build_committee_view completed in %.2f seconds", elapsed)
    return result


def build_settings_view() -> pn.viewable.Viewable:
    """Provide helpers for managing configuration and RawData sync."""
    config_path_input = pn.widgets.TextInput(
        name="Config File",
        value=str(uc.config_file),
        sizing_mode="stretch_width",
    )
    data_dir_input = pn.widgets.TextInput(
        name="Data Directory",
        value=str(uc.datadir),
        sizing_mode="stretch_width",
    )
    spelling_input = pn.widgets.TextInput(
        name="Spelling Dictionary",
        value=str(uc.spelling_dictionary),
        sizing_mode="stretch_width",
    )
    faculty_spelling_input = pn.widgets.TextInput(
        name="Faculty Spellings",
        value=str(uc.faculty_spellings_path),
        sizing_mode="stretch_width",
    )
    save_button = pn.widgets.Button(name="Save Configuration", button_type="primary", width=220, height=40)
    save_button.sizing_mode = "fixed"
    save_status = pn.pane.Markdown("", sizing_mode="stretch_width")

    def _refresh_form() -> None:
        payload = _load_config_payload()
        config_path_input.value = str(uc.config_file)
        data_dir_input.value = str(payload.get("data_directory", uc.datadir))
        spelling_input.value = str(payload.get("spelling_dictionary", uc.spelling_dictionary))
        faculty_spelling_input.value = str(payload.get("faculty_spellings", uc.faculty_spellings_path))

    def _handle_save(_=None) -> None:
        target_path = config_path_input.value.strip() or str(uc.config_file)
        data_dir = data_dir_input.value.strip()
        if not data_dir:
            save_status.object = "❌ Data directory is required."
            return
        payload: dict[str, Any] = {"data_directory": data_dir}
        spelling_value = spelling_input.value.strip()
        if spelling_value:
            payload["spelling_dictionary"] = spelling_value
        faculty_value = faculty_spelling_input.value.strip()
        if faculty_value:
            payload["faculty_spellings"] = faculty_value
        try:
            target = Path(target_path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)
            uc.set_config_file(target)
            _faculty_reference_table.cache_clear()
            _committee_feature_frame.cache_clear()
            _current_committee_map.cache_clear()
            _committee_feature_options.cache_clear()
            clear_faculty_caches()
            _refresh_form()
            datadir_summary.object = f"Current data directory: `{uc.datadir}`"
            save_status.object = "✅ Configuration saved. Consider refreshing open tabs."
        except Exception as exc:  # noqa: BLE001
            save_status.object = f"❌ Failed to save config: {exc}"

    save_button.on_click(_handle_save)

    sync_button = pn.widgets.Button(name="Sync Repository", button_type="primary", width=220, height=40)
    sync_button.sizing_mode = "fixed"
    sync_status = pn.pane.Markdown("", sizing_mode="stretch_width")

    def _handle_sync(_=None) -> None:
        sync_button.disabled = True
        success, message = _sync_rawdata_directory()
        sync_button.disabled = False
        prefix = "✅" if success else "❌"
        sync_status.object = f"{prefix} {message}"
        if success:
            try:
                from ucbfaculty.committee import clear_committee_caches
                clear_committee_caches()
            except Exception as exc:
                logger.debug("Failed to clear committee caches after sync: %s", exc)

    sync_button.on_click(_handle_sync)

    _refresh_form()

    config_section = pn.Column(
        pn.pane.Markdown("#### Configuration"),
        config_path_input,
        data_dir_input,
        spelling_input,
        faculty_spelling_input,
        pn.Row(save_button),
        save_status,
        sizing_mode="stretch_width",
    )

    datadir_summary = pn.pane.Markdown(
        f"Current data directory: `{uc.datadir}`",
        sizing_mode="stretch_width",
    )

    sync_section = pn.Column(
        pn.pane.Markdown("#### Sync"),
        datadir_summary,
        sync_button,
        sync_status,
        sizing_mode="stretch_width",
    )

    return pn.Row(
        config_section,
        pn.Spacer(width=20),
        sync_section,
        sizing_mode="stretch_width",
    )


def _build_passphrase_gate(on_accept: Callable[[str], None]) -> pn.viewable.Viewable:
    """Prompt for a GPG passphrase when reusing a cached license session."""
    prompt = pn.widgets.PasswordInput(
        name="GPG Passphrase",
        placeholder="Enter passphrase to decrypt RawData",
        width=360,
        css_classes=["passphrase-input"],
    )
    button = pn.widgets.Button(
        name="Unlock",
        button_type="primary",
        disabled=True,
        width=180,
    )

    def _refresh_button(*_):
        button.disabled = not prompt.value.strip()

    prompt.param.watch(_refresh_button, "value")
    _refresh_button()

    error_display = pn.pane.Markdown(
        "", visible=False, sizing_mode="stretch_width",
        styles={"color": "#cc3333", "font-size": "0.9em"},
    )

    def _handle_accept(_=None):
        success, message = security.unlock_rawdata(prompt.value)
        if not success:
            error_display.object = f"⚠️ {message}"
            error_display.visible = True
            return
        error_display.visible = False
        on_accept(prompt.value)

    button.on_click(_handle_accept)

    form = pn.Column(
        pn.pane.Markdown(
            "Enter your GPG passphrase to unlock RawData.",
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
        ),
        prompt,
        error_display,
        button,
        width=400,
        height=180,
        sizing_mode="fixed",
        margin=(8, 12, 8, 12),
    )
    card = pn.Card(
        form,
        title="Unlock RawData",
        width=440,
        height=240,
        sizing_mode="fixed",
    )
    return pn.Column(
        pn.Spacer(sizing_mode="stretch_both"),
        pn.Row(
            pn.Spacer(sizing_mode="stretch_both"),
            card,
            pn.Spacer(sizing_mode="stretch_both"),
            sizing_mode="stretch_both",
        ),
        pn.Spacer(sizing_mode="stretch_both"),
        sizing_mode="stretch_both",
    )


# Clear lookup.py's own caches that depend on name matching (after functions are defined)
_faculty_reference_table.cache_clear()
_committee_feature_frame.cache_clear()
_current_committee_map.cache_clear()
_committee_feature_options.cache_clear()


def _build_about_card() -> pn.layout.Card:
    """Build the always-visible About card (collapsed by default).

    Displays the installed ucbfaculty version in the card header so
    users can tell at a glance whether they're on a fresh install.
    Expanding the card reveals the full diagnostic blob (install path,
    Python version, platform, config paths) plus a Copy button that
    drops the same text on the clipboard for support tickets.
    """
    from ucbfaculty._diagnostics import get_diagnostics, format_diagnostics

    diag = get_diagnostics()
    md_body = format_diagnostics(diag, style="markdown")
    text_blob = format_diagnostics(diag, style="text")

    copy_btn = pn.widgets.Button(
        name="Copy diagnostic",
        button_type="light",
        width=160,
        height=30,
        description="Copy this diagnostic blob to the clipboard.",
    )
    copy_btn.js_on_click(
        args={"diag_text": text_blob},
        code="""
            navigator.clipboard.writeText(diag_text).then(function() {
                var orig = cb_obj.label;
                cb_obj.label = 'Copied!';
                setTimeout(function() { cb_obj.label = orig; }, 1500);
            });
        """,
    )

    body = pn.Column(
        pn.pane.Markdown(md_body, sizing_mode="stretch_width"),
        copy_btn,
        sizing_mode="stretch_width",
    )

    return pn.Card(
        body,
        title=f"About — ucbfaculty {diag['version']}",
        collapsible=True,
        collapsed=True,
        sizing_mode="stretch_width",
        margin=(15, 5, 5, 5),
        styles={"font-size": "0.85em", "opacity": "0.85"},
    )


def build_app() -> pn.viewable.Viewable:
    """Create the full Panel application with tabs."""
    pn.extension("tabulator", sizing_mode="stretch_both", raw_css=[PASSGATE_CSS])
    container = pn.Column(sizing_mode="stretch_both")

    # Create debugger widget if debug mode is enabled
    debug_widget = None
    if DEBUG_MODE:
        debug_widget = pn.widgets.Debugger(
            name="App Debugger",
            level=logging.DEBUG,  # Capture DEBUG and above
            sizing_mode="stretch_width",
            height=200,
        )
        logger.info("Debug mode enabled - Debugger widget active")

    tabs_cache: dict[str, pn.Tabs] = {}
    # Track which tabs have been loaded (by index)
    loaded_tabs: set[int] = set()

    def _wrap_with_debugger(content: pn.viewable.Viewable) -> list[pn.viewable.Viewable]:
        """Wrap content with debugger widget if debug mode is enabled."""
        if debug_widget is not None:
            return [content, debug_widget]
        return [content]

    def _ensure_tabs() -> pn.Tabs:
        if "tabs" not in tabs_cache:
            start = time.perf_counter()
            # Tab builders and their names - only Faculty Lookup is loaded eagerly
            tab_builders: list[tuple[str, Callable[[], pn.viewable.Viewable]]] = [
                ("Faculty Lookup", build_lookup_view),
                ("Feature Explorer", build_feature_selection_view),
                ("Committee Explorer", build_committee_view),
                ("Settings", build_settings_view),
            ]

            # Create placeholder containers for each tab
            tab_containers: list[pn.Column] = []
            for i, (name, builder) in enumerate(tab_builders):
                tab_container = pn.Column(sizing_mode="stretch_both")
                if i == 0:
                    # Load the first tab (Faculty Lookup) eagerly since it's shown by default
                    tab_container.objects = [builder()]
                    loaded_tabs.add(0)
                else:
                    # Show loading indicator for other tabs
                    tab_container.objects = [
                        pn.indicators.LoadingSpinner(
                            value=True, size=50, name="Loading..."
                        )
                    ]
                tab_containers.append(tab_container)

            tabs_cache["tabs"] = pn.Tabs(
                *[(name, tab_containers[i]) for i, (name, _) in enumerate(tab_builders)],
                sizing_mode="stretch_both",
            )

            def _on_tab_change(event: Any) -> None:
                """Load tab content lazily when tab is first selected."""
                tab_index = event.new
                if tab_index not in loaded_tabs:
                    loaded_tabs.add(tab_index)
                    tab_name, builder = tab_builders[tab_index]
                    logger.info("Lazy loading tab %d: %s", tab_index, tab_name)
                    tab_containers[tab_index].objects = [builder()]

            tabs_cache["tabs"].param.watch(_on_tab_change, "active")

            selection_bus = _selection_bus()

            def _focus_lookup(_payload: dict[str, str]):
                tabs_cache["tabs"].active = 0

            selection_bus.subscribe(_focus_lookup)
            elapsed = time.perf_counter() - start
            logger.info("_ensure_tabs completed in %.2f seconds (lazy loading enabled)", elapsed)
        return tabs_cache["tabs"]

    def _show_application(_credentials: dict[str, str]):
        container.objects = _wrap_with_debugger(_ensure_tabs())

    def _show_reauth_gate():
        """Replace main UI with passphrase gate on DecryptionError."""
        from ucbfaculty.rawdata import _cached_gpg_decrypt
        _cached_gpg_decrypt.cache_clear()

        def _on_reauth(_pass: str):
            tabs_cache.clear()
            loaded_tabs.clear()
            container.objects = _wrap_with_debugger(_ensure_tabs())

        container.objects = _wrap_with_debugger(
            _build_passphrase_gate(_on_reauth)
        )

    pn.state.cache["_reauth_callback"] = _show_reauth_gate

    session_data = _read_license_session()
    if session_data:
        pn.state.cache["ucbfaculty_auth"] = {
            "username": session_data.get("username", ""),
            "passphrase": "",
        }

        def _unlock(_pass: str):
            container.objects = _wrap_with_debugger(_ensure_tabs())

        # Skip passphrase gate if already unlocked via:
        # - CLI passphrase flag
        # - gpg-agent cached passphrase
        if security.is_unlocked():
            container.objects = _wrap_with_debugger(_ensure_tabs())
        else:
            container.objects = _wrap_with_debugger(_build_passphrase_gate(_unlock))
    else:
        splash = build_splash_view(_show_application)
        container.objects = _wrap_with_debugger(splash)
    # Wrap container with a persistent About card at the bottom so the
    # installed version is always visible (collapsed by default).  See
    # _build_about_card above for rationale.
    return pn.Column(
        container,
        _build_about_card(),
        sizing_mode="stretch_both",
    )


APP = build_app()
APP.servable("UCB Faculty Lookup")


if __name__ == "__main__":
    pn.serve(APP, title="UCB Faculty Lookup")
