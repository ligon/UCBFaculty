"""Keep the local data directory in step with its upstream archive.

The confidential data archive (encrypted ``.gpg`` files plus the
``sources.yaml`` / ``features_spec.yaml`` / spelling-dictionary
sidecars) is a plain folder of files.  Where that folder is hosted is
a deployment decision, and this module keeps the package indifferent
to it by putting every transport behind the same two operations:

- :func:`fetch_data_directory` -- first-time acquisition into an
  empty local directory.
- :func:`sync_data_directory` -- bring an existing local copy up to
  date.

Three backends are provided:

``box-drive``
    The Box Drive desktop client (macOS, Windows) mounts the user's
    Box account as a local folder.  ``data_directory`` points inside
    that mount; Box Drive keeps it current, so both operations reduce
    to "check the folder is there".

``rclone``
    ``rclone`` (https://rclone.org) mirrors a remote folder into
    ``data_directory``.  Any rclone remote works (Box, Google Drive,
    Dropbox, S3, ...); the package's default is a Box remote rooted at
    the shared folder by its Box folder id, so users never type a Box
    path.

``git``
    ``data_directory`` is a clone of a git repository.  Fetch is
    ``git clone``; sync is ``git pull --ff-only``.

Berkeley's deployment uses Box, because campus policy classifies the
archive as protection-level 3 data, for which Box is approved and
GitHub is not.  The Box specifics live in :data:`BOX_FOLDER_ID` and
:data:`DEFAULT_REMOTE`; nothing else here knows about Box.

Backend selection is explicit when ``data_backend`` is set in the
config (``box-drive`` / ``rclone`` / ``git``) and otherwise inferred:
see :func:`detect_backend`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

BACKENDS = ("box-drive", "rclone", "git")

#: Box folder id of the shared archive (the trailing number in the
#: folder's web URL).  rclone's ``root_folder_id`` option roots a
#: remote at this folder directly.
BOX_FOLDER_ID = "410604666843"
BOX_FOLDER_URL = f"https://berkeley.app.box.com/folder/{BOX_FOLDER_ID}"

#: Name of the rclone remote the package creates and looks for.
DEFAULT_REMOTE = "ucbfaculty"

#: Where ``rclone sync`` / ``git clone`` put the archive when the user
#: has not chosen somewhere else.
DEFAULT_LOCAL_DIR = Path.home() / ".ucbfaculty" / "data"

#: File whose presence marks a directory as a usable data archive.
MARKER_FILE = "features_spec.yaml"

#: Git upstreams that no longer carry the archive.  A checkout whose
#: origin matches one of these gets a pointer to the Box instructions
#: whenever it is synced, so users of the pre-Box setup are told how
#: to move.
RETIRED_UPSTREAMS = (
    "github.com/ligon/RawData",
    "github.com:ligon/RawData",
)

#: rclone filter rules applied to every transfer.  ``.git`` is excluded
#: so a maintainer's local git checkout can double as a mirror source
#: or target; plaintext spreadsheets are excluded so a stray decrypted
#: file can never travel in either direction; ``Proposed*/`` folders
#: hold work-in-progress specs that are not part of the published
#: archive; Box Notes are Box-side annotations, not data.
RCLONE_FILTERS = (
    "--exclude", ".git/**",
    "--exclude", ".venv/**",
    "--exclude", "__pycache__/**",
    "--exclude", "Proposed*/**",
    "--exclude", "*.csv",
    "--exclude", "*.xls",
    "--exclude", "*.xlsx",
    "--exclude", "*.xlsm",
    "--exclude", "*.boxnote",
)

Log = Optional[Callable[[str], None]]


def _emit(log: Log, text: str) -> None:
    if log is not None and text:
        log(text)


def _run(cmd: list[str], log: Log = None) -> tuple[int, str]:
    """Run *cmd*, echoing the command and its combined output to *log*."""
    _emit(log, "$ " + " ".join(cmd) + "\n")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        _emit(log, f"{exc}\n")
        return 127, str(exc)
    output = (result.stdout + result.stderr).strip()
    _emit(log, output + "\n")
    return result.returncode, output


# ---------------------------------------------------------------------------
# Box Drive
# ---------------------------------------------------------------------------

def box_drive_roots(platform: str | None = None, home: Path | None = None) -> list[Path]:
    """Candidate mount points for the Box Drive desktop client.

    Only locations that exist on disk are returned.  Box Drive is not
    available on Linux, so the list is empty there unless the
    ``UCBFACULTY_BOX_DRIVE`` environment variable names a mount.
    """
    platform = platform or sys.platform
    home = home or Path.home()
    candidates: list[Path] = []
    env_root = os.environ.get("UCBFACULTY_BOX_DRIVE")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    if platform == "darwin":
        # Box Drive 2.24+ uses the File Provider location; older
        # versions mounted at ~/Box.
        candidates.append(home / "Library" / "CloudStorage" / "Box-Box")
        candidates.append(home / "Box")
    elif platform == "win32":
        candidates.append(home / "Box")
    return [c for c in candidates if c.is_dir()]


def is_box_drive_path(path: Path | str, roots: list[Path] | None = None) -> bool:
    """True when *path* lies inside a Box Drive mount."""
    roots = box_drive_roots() if roots is None else roots
    target = Path(path).expanduser()
    for root in roots:
        try:
            target.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def find_box_drive_data_dir(roots: list[Path] | None = None, max_depth: int = 3) -> Path | None:
    """Locate the archive inside a Box Drive mount.

    Searches each mount up to *max_depth* levels for a directory that
    contains :data:`MARKER_FILE`.  Shallow matches win; the first mount
    in :func:`box_drive_roots` order wins ties.
    """
    roots = box_drive_roots() if roots is None else roots
    for root in roots:
        frontier = [root]
        for _ in range(max_depth + 1):
            next_frontier: list[Path] = []
            for folder in frontier:
                if (folder / MARKER_FILE).is_file():
                    return folder
                try:
                    next_frontier.extend(
                        p for p in sorted(folder.iterdir())
                        if p.is_dir() and not p.name.startswith(".")
                    )
                except OSError:
                    continue
            frontier = next_frontier
            if not frontier:
                break
    return None


def box_drive_check(data_dir: Path) -> tuple[bool, str]:
    """Confirm a Box Drive-hosted archive is present (nothing to transfer)."""
    if not (data_dir / MARKER_FILE).exists():
        return False, (
            f"{data_dir} is inside Box Drive but does not contain {MARKER_FILE}.  "
            f"Check that you have been given access to {BOX_FOLDER_URL} and that "
            "the folder has finished syncing."
        )
    return True, f"Box Drive keeps {data_dir} in sync with Box; nothing to do."


# ---------------------------------------------------------------------------
# rclone
# ---------------------------------------------------------------------------

def rclone_path() -> str | None:
    """Absolute path of the ``rclone`` executable, or None."""
    return shutil.which("rclone")


def rclone_remotes() -> list[str]:
    """Names of configured rclone remotes (without the trailing colon)."""
    exe = rclone_path()
    if exe is None:
        return []
    try:
        result = subprocess.run(
            [exe, "listremotes"], capture_output=True, text=True, check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip().rstrip(":") for line in result.stdout.splitlines() if line.strip()]


def rclone_remote_exists(name: str = DEFAULT_REMOTE) -> bool:
    return name in rclone_remotes()


def remote_spec(config: dict | None = None, name: str = DEFAULT_REMOTE) -> str:
    """rclone source spec for the archive: ``<remote>:[path]``.

    Honours ``rclone_remote`` in the config so a deployment can point
    at any remote or sub-folder; otherwise the package-named remote
    rooted at the shared Box folder.
    """
    value = (config or {}).get("rclone_remote")
    if value:
        value = str(value).strip()
        return value if ":" in value else f"{value}:"
    return f"{name}:"


def create_rclone_remote(
    name: str = DEFAULT_REMOTE,
    folder_id: str = BOX_FOLDER_ID,
    log: Log = None,
) -> tuple[bool, str]:
    """Create (or re-authorise) the Box remote.

    Runs ``rclone config create``, which walks the user through Box's
    OAuth flow: rclone prints a ``http://127.0.0.1:53682/auth?...`` link
    and, where it can, opens it in a browser.  Box's sign-in page offers
    "Sign in with SSO", which is CalNet.  Output is streamed to *log*
    line by line so the link is visible while rclone waits.  The
    resulting token is written to rclone's own config file and is never
    echoed to *log*.
    """
    exe = rclone_path()
    if exe is None:
        return False, (
            "rclone is not installed.  Install it from https://rclone.org/install/ "
            "(or install Box Drive and point the data directory at the mounted folder)."
        )
    cmd = [exe, "config", "create", name, "box", "root_folder_id", folder_id]
    _emit(log, "$ " + " ".join(cmd) + "\n")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except OSError as exc:
        return False, f"Could not start rclone: {exc}"
    assert proc.stdout is not None
    tail: list[str] = []
    for line in proc.stdout:
        # rclone echoes the finished remote, token included, on success.
        if "token" in line or "refresh_token" in line:
            continue
        _emit(log, line)
        tail.append(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        return False, "rclone config create failed:\n" + "\n".join(tail[-10:])
    if not rclone_remote_exists(name):
        return False, f"rclone reported success but remote {name!r} is not listed."
    return True, f"rclone remote {name!r} is connected to Box folder {folder_id}."


def rclone_sync(
    source: str,
    dest: Path | str,
    *,
    dry_run: bool = False,
    log: Log = None,
) -> tuple[bool, str]:
    """Mirror *source* (an rclone spec) into *dest*.

    ``rclone sync`` makes *dest* match the remote folder, deleting local
    files that no longer exist upstream, so that the holdings
    invariant (data at rest == declared sources) survives removals.
    :data:`RCLONE_FILTERS` keep ``.git`` and any plaintext spreadsheets
    out of the comparison.
    """
    exe = rclone_path()
    if exe is None:
        return False, "rclone is not installed (https://rclone.org/install/)."
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "sync", source, str(dest), *RCLONE_FILTERS, "--stats-one-line", "-v"]
    if dry_run:
        cmd.append("--dry-run")
    code, output = _run(cmd, log)
    if code != 0:
        hint = ""
        if "didn't find section" in output or "Failed to create file system" in output:
            hint = (
                f"\nThe rclone remote for {source!r} is not configured.  "
                "Run `ucbfaculty sync` to connect, or "
                f"`rclone config create {DEFAULT_REMOTE} box root_folder_id {BOX_FOLDER_ID}`."
            )
        elif "401" in output or "invalid_grant" in output or (
            "token" in output.lower() and "expired" in output.lower()
        ):
            hint = (
                "\nAuthorisation has lapsed.  Reconnect with "
                f"`rclone config reconnect {source.split(':')[0]}:`."
            )
        return False, "rclone sync failed.\n" + output + hint
    summary = output.splitlines()[-1] if output else "Synced."
    return True, f"Synced {source} -> {dest}\n{summary}"


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def git_path() -> str | None:
    return shutil.which("git")


def git_clone(url: str, dest: Path | str, log: Log = None) -> tuple[bool, str]:
    """Clone *url* into *dest* (which must not already be a checkout)."""
    exe = git_path()
    if exe is None:
        return False, "git executable not found on PATH."
    dest = Path(dest).expanduser()
    if (dest / ".git").exists():
        return git_pull(dest, log)
    if dest.exists() and any(dest.iterdir()):
        return False, f"{dest} exists and is not empty; refusing to clone into it."
    code, output = _run([exe, "clone", url, str(dest)], log)
    if code != 0:
        return False, "git clone failed.\n" + output
    return True, f"Cloned {url} -> {dest}"


def git_pull(data_dir: Path | str, log: Log = None) -> tuple[bool, str]:
    """Fast-forward the checkout at *data_dir*."""
    exe = git_path()
    if exe is None:
        return False, "git executable not found on PATH."
    data_dir = Path(data_dir).expanduser()
    if not (data_dir / ".git").exists():
        return False, f"No git repository detected at {data_dir}"
    code, output = _run([exe, "-C", str(data_dir), "pull", "--ff-only"], log)
    if code != 0:
        return False, output or "git pull failed."
    return True, output or "Already up to date."


def git_upstream(data_dir: Path | str) -> str | None:
    """URL of the checkout's ``origin`` remote, or None."""
    exe = git_path()
    data_dir = Path(data_dir).expanduser()
    if exe is None or not (data_dir / ".git").exists():
        return None
    try:
        result = subprocess.run(
            [exe, "-C", str(data_dir), "remote", "get-url", "origin"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def retired_upstream_note(data_dir: Path | str) -> str | None:
    """Migration hint when *data_dir* tracks a retired git upstream."""
    url = git_upstream(data_dir)
    if not url:
        return None
    normalised = url.lower().removesuffix(".git")
    if not any(marker.lower() in normalised for marker in RETIRED_UPSTREAMS):
        return None
    return (
        f"NOTE: this copy of the data comes from {url}, which is being retired.  "
        f"The archive now lives in Box ({BOX_FOLDER_URL}).  See the "
        "'Upgrading from the GitHub-based setup' section of INSTALL.org: "
        "install Box Drive (macOS/Windows) and run `ucbfaculty init --force`, "
        "or run `ucbfaculty sync --backend rclone` on Linux."
    )


# ---------------------------------------------------------------------------
# Backend selection and the two public operations
# ---------------------------------------------------------------------------

def detect_backend(config: dict | None = None, data_dir: Path | str | None = None) -> str:
    """Return ``"box-drive"``, ``"rclone"``, ``"git"``, or ``"none"``.

    An explicit ``data_backend`` in the config wins.  Otherwise, in
    order: ``rclone_remote`` set -> rclone; ``data_repo_url`` set or
    the directory is a git checkout -> git; the directory is inside a
    Box Drive mount -> box-drive; the package-named rclone remote is
    configured on this machine -> rclone.
    """
    config = config or {}
    explicit = str(config.get("data_backend") or "").strip().lower()
    if explicit:
        if explicit not in BACKENDS:
            raise ValueError(
                f"Unknown data_backend {explicit!r}; expected one of {', '.join(BACKENDS)}"
            )
        return explicit
    if data_dir is None:
        data_dir = config.get("data_directory")
    data_dir = Path(data_dir).expanduser() if data_dir else None
    if config.get("rclone_remote"):
        return "rclone"
    if config.get("data_repo_url") or (data_dir is not None and (data_dir / ".git").exists()):
        return "git"
    if data_dir is not None and is_box_drive_path(data_dir):
        return "box-drive"
    if rclone_path() and rclone_remote_exists(DEFAULT_REMOTE):
        return "rclone"
    return "none"


def _no_backend_message(data_dir: Path) -> str:
    return (
        f"No upstream is configured for {data_dir}.  Either install Box Drive and "
        f"point data_directory inside the mounted copy of {BOX_FOLDER_URL}, install "
        "rclone and run `ucbfaculty sync` to connect, or set data_backend / "
        "data_repo_url in config.yaml for a git-hosted archive."
    )


def _resolve(config: dict | None, data_dir: Path | str | None) -> tuple[dict, Path]:
    if config is None:
        try:
            from ucbfaculty import load_config
            config = load_config()
        except FileNotFoundError:
            config = {}
    if data_dir is None:
        data_dir = config.get("data_directory") or DEFAULT_LOCAL_DIR
    return config, Path(data_dir).expanduser()


def fetch_data_directory(
    config: dict | None = None,
    data_dir: Path | str | None = None,
    *,
    backend: str | None = None,
    log: Log = None,
) -> tuple[bool, str]:
    """Acquire the archive into *data_dir* for the first time.

    Returns ``(ok, message)``.  Safe to call on an existing copy: each
    backend degrades to its sync operation.
    """
    config, data_dir = _resolve(config, data_dir)
    backend = backend or detect_backend(config, data_dir)
    if backend == "box-drive":
        return box_drive_check(data_dir)
    if backend == "rclone":
        return rclone_sync(remote_spec(config), data_dir, log=log)
    if backend == "git":
        url = config.get("data_repo_url")
        if (data_dir / ".git").exists():
            return git_pull(data_dir, log)
        if not url:
            return False, "data_repo_url is not set in config.yaml; cannot clone."
        return git_clone(url, data_dir, log)
    return False, _no_backend_message(data_dir)


def sync_data_directory(
    config: dict | None = None,
    data_dir: Path | str | None = None,
    *,
    backend: str | None = None,
    log: Log = None,
) -> tuple[bool, str]:
    """Bring an existing local copy of the archive up to date.

    Returns ``(ok, message)``.  In ``box-drive`` mode there is nothing
    to do beyond confirming the folder is present; ``rclone`` mirrors
    the remote; ``git`` fast-forwards; otherwise the message explains
    how to get connected.
    """
    config, data_dir = _resolve(config, data_dir)
    backend = backend or detect_backend(config, data_dir)
    if backend == "box-drive":
        return box_drive_check(data_dir)
    if backend == "rclone":
        return rclone_sync(remote_spec(config), data_dir, log=log)
    if backend == "git":
        ok, message = git_pull(data_dir, log)
        note = retired_upstream_note(data_dir)
        if note:
            message = f"{message}\n\n{note}"
        return ok, message
    return False, _no_backend_message(data_dir)
