"""First-run setup wizard.

Checks prerequisites (GPG installed, secret key present), clones or
updates the RawData repository, and writes a minimal config file so the
main lookup app can start.
"""

import shutil
import subprocess
from pathlib import Path

import panel as pn
import yaml


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


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------

def setup_view():
    pn.extension(design="material", notifications=True)

    home = Path.home()
    default_data_dir = home / "RawData"

    # ---- Tab 1: Prerequisites --------------------------------------------
    gpg_ok = _gpg_installed()
    secret_keys = _gpg_secret_keys()

    gpg_alert = pn.pane.Alert(
        "\u2705 GPG is installed." if gpg_ok
        else "\u26a0\ufe0f **GPG is not installed.** Please install it before continuing.",
        alert_type="success" if gpg_ok else "danger",
    )
    key_alert = pn.pane.Alert(
        f"\u2705 Found {len(secret_keys)} secret key(s)." if secret_keys
        else (
            "\u26a0\ufe0f **No GPG secret keys found.** "
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

    # ---- Tab 2: Clone data & save config ---------------------------------
    repo_input = pn.widgets.TextInput(
        name="Repository URL",
        value="https://github.com/ligon/RawData",
    )
    path_input = pn.widgets.TextInput(
        name="Local Directory",
        value=str(default_data_dir),
    )
    clone_log = pn.widgets.Terminal(
        height=150, sizing_mode="stretch_width", visible=False,
    )

    clone_btn = pn.widgets.Button(name="Clone / Sync Data", button_type="primary")
    finish_btn = pn.widgets.Button(
        name="Save Config & Finish", button_type="success", disabled=True,
    )

    def _run_clone(event):
        target = Path(path_input.value).expanduser()

        if target.exists() and any(target.iterdir()) and not (target / ".git").exists():
            pn.state.notifications.error("Directory exists and is not a git repo.")
            return

        clone_btn.loading = True
        clone_log.visible = True
        clone_log.clear()

        if target.exists() and (target / ".git").exists():
            cmd = ["git", "-C", str(target), "pull"]
            clone_log.write("Pulling latest changes...\n")
        else:
            cmd = ["git", "clone", repo_input.value, str(target)]
            clone_log.write(f"Cloning {repo_input.value}...\n")

        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            clone_log.write(res.stdout)
            if res.stderr:
                clone_log.write(res.stderr)
            if res.returncode == 0:
                pn.state.notifications.success("Repository synced.")
                finish_btn.disabled = False
            else:
                pn.state.notifications.error("Git operation failed. See log above.")
        except Exception as exc:
            clone_log.write(f"Error: {exc}\n")
        finally:
            clone_btn.loading = False

    clone_btn.on_click(_run_clone)

    def _save_config(event):
        target = Path(path_input.value).expanduser()
        config = {
            "data_directory": str(target),
            "feature_specs_yaml": [str(target / "features_spec.yaml")],
            "sources_yaml": [str(target / "sources.yaml")],
            "spelling_dictionary": str(target / "spelling_dictionary.json"),
            "faculty_spellings": str(target / "faculty_spellings.json"),
            "email_dictionary": str(target / "email_dictionary.json"),
        }
        config_dir = Path.home() / ".ucbfaculty"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh)
        pn.state.notifications.success(
            f"Config saved to {config_path}. Restart the app to continue."
        )
        finish_btn.name = "Saved \u2014 restart the app"
        finish_btn.disabled = True

    finish_btn.on_click(_save_config)

    # Enable Finish whenever the current path_input value points at a
    # directory that already contains features_spec.yaml.  This lets users
    # who already have a RawData clone (at any location) skip the clone
    # step entirely.
    def _refresh_finish_state(*_events):
        try:
            candidate = Path(path_input.value).expanduser()
        except (TypeError, ValueError):
            finish_btn.disabled = True
            return
        finish_btn.disabled = not (candidate / "features_spec.yaml").exists()

    path_input.param.watch(_refresh_finish_state, "value")
    _refresh_finish_state()

    data_tab = pn.Column(
        pn.pane.Markdown(
            "### Download Data\n\n"
            "Clone (or update) the confidential data repository.  "
            "You may be prompted to sign in to GitHub.\n"
        ),
        repo_input,
        path_input,
        clone_btn,
        clone_log,
        finish_btn,
    )

    # ---- Layout ----------------------------------------------------------
    # dynamic=False so Tab 2 renders eagerly.  Lazy rendering occasionally
    # leaves the second tab silently empty when activated, which broke a
    # real first-install attempt.
    tabs = pn.Tabs(
        ("1. Prerequisites", prereq_tab),
        ("2. Download & Configure", data_tab),
    )

    return pn.template.FastListTemplate(
        site="UCBFaculty",
        title="Setup Assistant",
        main=[tabs],
    )


if __name__.startswith("bokeh"):
    setup_view().servable()
