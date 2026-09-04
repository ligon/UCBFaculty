"""First-run setup wizard.

Checks prerequisites (GPG installed, secret key present), gets the
confidential data archive onto the machine, and writes a minimal
config file so the main lookup app can start.

The archive is hosted in a shared Box folder.  The wizard offers the
three transports :mod:`ucbfaculty.datasync` knows about:

- **Box Drive** (macOS / Windows): the folder is already on disk once
  Box Drive is installed and signed in; the wizard finds it.
- **rclone** (any platform, and the only option on Linux): connect an
  rclone remote to the Box folder, then mirror it locally.
- **git**: for deployments that host the archive in a git repository.
"""

import shutil
import subprocess
from pathlib import Path

import panel as pn
import yaml

from ucbfaculty import datasync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpg_installed() -> bool:
    return shutil.which("gpg") is not None


def _gpg_secret_keys() -> list[str]:
    """Return uid strings for locally available GPG secret keys."""
    if not _gpg_installed():
        return []
    try:
        res = subprocess.run(
            ["gpg", "--list-secret-keys", "--with-colons"],
            capture_output=True, text=True,
        )
        keys: list[str] = []
        for line in res.stdout.splitlines():
            if line.startswith("uid:") or line.startswith("sec:"):
                parts = line.split(":")
                uid = parts[9] if len(parts) > 9 else ""
                if uid:
                    keys.append(uid)
        return keys
    except Exception:
        return []


def _export_public_key(uid: str) -> str:
    """Armored export of the public key matching *uid*."""
    res = subprocess.run(
        ["gpg", "--armor", "--export", uid],
        capture_output=True, text=True,
    )
    return res.stdout if res.returncode == 0 else ""


def _config_payload(target: Path, backend: str, repo_url: str = "") -> dict:
    config = {
        "data_directory": str(target),
        "feature_specs_yaml": [str(target / "features_spec.yaml")],
        "sources_yaml": [str(target / "sources.yaml")],
        "spelling_dictionary": str(target / "spelling_dictionary.json"),
        "faculty_spellings": str(target / "faculty_spellings.json"),
        "email_dictionary": str(target / "email_dictionary.json"),
        "data_backend": backend,
    }
    if backend == "rclone":
        config["rclone_remote"] = datasync.remote_spec()
    if backend == "git" and repo_url:
        config["data_repo_url"] = repo_url
    return config


BACKEND_LABELS = {
    "box-drive": "Box Drive (macOS / Windows)",
    "rclone": "rclone (any platform)",
    "git": "git repository",
}


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------

def setup_view():
    pn.extension(design="material", notifications=True)

    # ---- Tab 1: Prerequisites --------------------------------------------
    gpg_ok = _gpg_installed()
    secret_keys = _gpg_secret_keys()

    gpg_alert = pn.pane.Alert(
        "✅ GPG is installed." if gpg_ok
        else "⚠️ **GPG is not installed.** Please install it before continuing.",
        alert_type="success" if gpg_ok else "danger",
    )
    key_alert = pn.pane.Alert(
        f"✅ Found {len(secret_keys)} secret key(s)." if secret_keys
        else (
            "⚠️ **No GPG secret keys found.** "
            "Generate one with `gpg --full-generate-key` and restart."
        ),
        alert_type="success" if secret_keys else "warning",
    )

    # Show / copy public key for the chair
    pub_key_area = pn.widgets.TextAreaInput(
        name="Your Public Key (send to the committee chair)",
        height=180, disabled=True, visible=bool(secret_keys),
    )
    if secret_keys:
        pub_key_area.value = _export_public_key(secret_keys[0])

    prereq_tab = pn.Column(
        pn.pane.Markdown(
            "### Prerequisites\n\n"
            "The app needs **GPG** (to decrypt confidential data) and a "
            "**secret key** whose public half has been shared with the chair.\n\n"
            "- **macOS:** [GPG Suite](https://gpgtools.org/)  \n"
            "- **Windows:** [Gpg4win](https://gpg4win.org/)\n\n"
            "After installing GPG and generating a key, restart this wizard."
        ),
        gpg_alert,
        key_alert,
        pub_key_area,
    )

    # ---- Tab 2: Get the data & save config -------------------------------
    box_drive_dir = datasync.find_box_drive_data_dir()
    have_box_drive = bool(datasync.box_drive_roots())
    have_rclone = datasync.rclone_path() is not None
    remote_connected = have_rclone and datasync.rclone_remote_exists()

    if box_drive_dir is not None:
        initial_backend = "box-drive"
    elif have_rclone or not have_box_drive:
        initial_backend = "rclone"
    else:
        initial_backend = "box-drive"

    backend_select = pn.widgets.RadioBoxGroup(
        name="How will this machine get the data?",
        options=list(BACKEND_LABELS.values()),
        value=BACKEND_LABELS[initial_backend],
    )

    def _backend() -> str:
        for key, label in BACKEND_LABELS.items():
            if label == backend_select.value:
                return key
        return "rclone"

    path_input = pn.widgets.TextInput(
        name="Local Directory",
        value=str(box_drive_dir or datasync.DEFAULT_LOCAL_DIR),
    )
    repo_input = pn.widgets.TextInput(
        name="Repository URL (git only)",
        value="",
        placeholder="https://... or git@...:org/repo.git",
        visible=False,
    )
    log_widget = pn.widgets.Terminal(
        height=180, sizing_mode="stretch_width", visible=False,
    )

    status_md = pn.pane.Markdown("", sizing_mode="stretch_width")
    connect_btn = pn.widgets.Button(
        name="Connect rclone to Box (sign in with CalNet)", button_type="default",
    )
    fetch_btn = pn.widgets.Button(name="Download / Sync Data", button_type="primary")
    finish_btn = pn.widgets.Button(
        name="Save Config & Finish", button_type="success", disabled=True,
    )

    box_drive_help = pn.pane.Markdown(
        "**Box Drive** keeps the shared folder on your disk automatically.\n\n"
        "1. Install [Box Drive](https://www.box.com/resources/downloads) and sign in "
        "with your `@berkeley.edu` address (CalNet).\n"
        f"2. Open the [data folder in Box]({datasync.BOX_FOLDER_URL}) once, so it appears "
        "under your Box Drive folder.\n"
        "3. Point *Local Directory* at that folder.  If the wizard found it, "
        "the path is already filled in.\n",
        sizing_mode="stretch_width",
    )
    rclone_help = pn.pane.Markdown(
        "**rclone** mirrors the Box folder into a local directory.\n\n"
        "1. Install [rclone](https://rclone.org/install/) if it is missing.\n"
        "2. Click *Connect rclone to Box*.  A browser sign-in opens (or a link "
        "appears in the log below); choose **Sign in with SSO** to use CalNet.\n"
        "3. Click *Download / Sync Data*.  Re-run the same step later from the "
        "app's Settings tab to pick up new data.\n",
        sizing_mode="stretch_width",
    )
    git_help = pn.pane.Markdown(
        "**git**: clone (or pull) an archive hosted in a git repository.  "
        "Enter the repository URL; you may be prompted to authenticate.\n",
        sizing_mode="stretch_width",
    )

    def _refresh_status(*_events) -> None:
        backend = _backend()
        box_drive_help.visible = backend == "box-drive"
        rclone_help.visible = backend == "rclone"
        git_help.visible = backend == "git"
        repo_input.visible = backend == "git"
        connect_btn.visible = backend == "rclone"
        fetch_btn.visible = backend != "box-drive"
        fetch_btn.name = "Clone / Pull Data" if backend == "git" else "Download / Sync Data"
        lines = []
        if backend == "box-drive":
            if not have_box_drive:
                lines.append("⚠️ No Box Drive folder was found on this machine.")
            elif box_drive_dir is None:
                lines.append(
                    "⚠️ Box Drive is installed but the data folder was not found "
                    "inside it yet.  Open the folder in Box once, wait for it to sync, "
                    "then re-check."
                )
            else:
                lines.append(f"✅ Found the data folder at `{box_drive_dir}`.")
        elif backend == "rclone":
            if not have_rclone:
                lines.append("⚠️ rclone is not installed.")
            elif datasync.rclone_remote_exists():
                lines.append(f"✅ rclone remote `{datasync.DEFAULT_REMOTE}` is connected to Box.")
            else:
                lines.append("rclone is installed; connect it to Box to continue.")
        status_md.object = "  \n".join(lines)
        _refresh_finish_state()

    def _refresh_finish_state(*_events) -> None:
        try:
            candidate = Path(path_input.value).expanduser()
        except (TypeError, ValueError):
            finish_btn.disabled = True
            return
        finish_btn.disabled = not (candidate / datasync.MARKER_FILE).exists()

    def _log(line: str) -> None:
        log_widget.visible = True
        log_widget.write(line)

    def _run_connect(event):
        connect_btn.loading = True
        log_widget.clear()
        try:
            ok, message = datasync.create_rclone_remote(log=_log)
            if ok:
                pn.state.notifications.success(message)
            else:
                pn.state.notifications.error("Could not connect rclone to Box. See log.")
                _log(message + "\n")
        finally:
            connect_btn.loading = False
            _refresh_status()

    def _run_fetch(event):
        backend = _backend()
        target = Path(path_input.value).expanduser()
        if backend == "git" and not repo_input.value.strip():
            pn.state.notifications.error("Enter the repository URL first.")
            return
        if (
            target.exists()
            and any(target.iterdir())
            and not (target / datasync.MARKER_FILE).exists()
            and not (target / ".git").exists()
        ):
            pn.state.notifications.error(
                "Directory exists, is not empty, and does not look like the data archive."
            )
            return
        fetch_btn.loading = True
        log_widget.clear()
        config = _config_payload(target, backend, repo_input.value.strip())
        try:
            ok, message = datasync.fetch_data_directory(
                config, target, backend=backend, log=_log,
            )
            if ok:
                pn.state.notifications.success("Data is up to date.")
            else:
                pn.state.notifications.error("Download failed. See log.")
                _log(message + "\n")
        except Exception as exc:  # noqa: BLE001
            _log(f"Error: {exc}\n")
        finally:
            fetch_btn.loading = False
            _refresh_finish_state()

    def _save_config(event):
        target = Path(path_input.value).expanduser()
        config = _config_payload(target, _backend(), repo_input.value.strip())
        config_dir = Path.home() / ".ucbfaculty"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh)
        pn.state.notifications.success(
            f"Config saved to {config_path}. Restart the app to continue."
        )
        finish_btn.name = "Saved — restart the app"
        finish_btn.disabled = True

    connect_btn.on_click(_run_connect)
    fetch_btn.on_click(_run_fetch)
    finish_btn.on_click(_save_config)
    backend_select.param.watch(_refresh_status, "value")
    path_input.param.watch(_refresh_finish_state, "value")
    _refresh_status()

    data_tab = pn.Column(
        pn.pane.Markdown(
            "### Get the Data\n\n"
            "The confidential data lives in a shared "
            f"[Box folder]({datasync.BOX_FOLDER_URL}).  You need to have been "
            "given access to it (ask the committee chair).  Choose how this "
            "machine will keep a copy:\n"
        ),
        backend_select,
        box_drive_help,
        rclone_help,
        git_help,
        status_md,
        repo_input,
        path_input,
        pn.Row(connect_btn, fetch_btn),
        log_widget,
        finish_btn,
    )

    # ---- Layout ----------------------------------------------------------
    # dynamic=False so Tab 2 renders eagerly.  Lazy rendering occasionally
    # leaves the second tab silently empty when activated, which broke a
    # real first-install attempt.
    tabs = pn.Tabs(
        ("1. Prerequisites", prereq_tab),
        ("2. Get Data & Configure", data_tab),
    )

    return pn.template.FastListTemplate(
        site="UCBFaculty",
        title="Setup Assistant",
        main=[tabs],
    )


if __name__.startswith("bokeh"):
    setup_view().servable()
