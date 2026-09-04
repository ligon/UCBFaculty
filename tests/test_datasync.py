"""Tests for ucbfaculty.datasync: backend detection and transport commands.

Every external process (rclone, git) is faked, so these run anywhere.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ucbfaculty import datasync


# ---------------------------------------------------------------------------
# Box Drive
# ---------------------------------------------------------------------------

def test_box_drive_roots_macos_prefers_cloudstorage(tmp_path, monkeypatch):
    monkeypatch.delenv("UCBFACULTY_BOX_DRIVE", raising=False)
    cloud = tmp_path / "Library" / "CloudStorage" / "Box-Box"
    cloud.mkdir(parents=True)
    legacy = tmp_path / "Box"
    legacy.mkdir()
    roots = datasync.box_drive_roots(platform="darwin", home=tmp_path)
    assert roots == [cloud, legacy]


def test_box_drive_roots_linux_empty_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("UCBFACULTY_BOX_DRIVE", raising=False)
    assert datasync.box_drive_roots(platform="linux", home=tmp_path) == []


def test_box_drive_roots_env_override(tmp_path, monkeypatch):
    mount = tmp_path / "mnt"
    mount.mkdir()
    monkeypatch.setenv("UCBFACULTY_BOX_DRIVE", str(mount))
    assert datasync.box_drive_roots(platform="linux", home=tmp_path) == [mount]


def test_is_box_drive_path(tmp_path):
    root = tmp_path / "Box"
    root.mkdir()
    assert datasync.is_box_drive_path(root / "UCBFaculty" / "RawData", roots=[root])
    assert not datasync.is_box_drive_path(tmp_path / "elsewhere", roots=[root])


def test_find_box_drive_data_dir_finds_nested_marker(tmp_path):
    root = tmp_path / "Box"
    target = root / "Shared" / "RawData"
    target.mkdir(parents=True)
    (target / datasync.MARKER_FILE).write_text("{}")
    (root / "Other").mkdir()
    assert datasync.find_box_drive_data_dir(roots=[root]) == target


def test_find_box_drive_data_dir_respects_depth(tmp_path):
    root = tmp_path / "Box"
    deep = root / "a" / "b" / "c" / "d" / "RawData"
    deep.mkdir(parents=True)
    (deep / datasync.MARKER_FILE).write_text("{}")
    assert datasync.find_box_drive_data_dir(roots=[root], max_depth=2) is None
    assert datasync.find_box_drive_data_dir(roots=[root], max_depth=5) == deep


# ---------------------------------------------------------------------------
# rclone
# ---------------------------------------------------------------------------

class _FakeRun:
    """Records subprocess.run calls and returns a canned result."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def test_remote_spec_defaults_and_config():
    assert datasync.remote_spec() == f"{datasync.DEFAULT_REMOTE}:"
    assert datasync.remote_spec({"rclone_remote": "mybox"}) == "mybox:"
    assert datasync.remote_spec({"rclone_remote": "mybox:Some/Dir"}) == "mybox:Some/Dir"


def test_rclone_remotes_parses_listremotes(monkeypatch):
    fake = _FakeRun(stdout="dropbox:\nucbfaculty:\n")
    monkeypatch.setattr(datasync.shutil, "which", lambda name: "/usr/bin/rclone")
    monkeypatch.setattr(datasync.subprocess, "run", fake)
    assert datasync.rclone_remotes() == ["dropbox", "ucbfaculty"]
    assert datasync.rclone_remote_exists("ucbfaculty")
    assert not datasync.rclone_remote_exists("nope")


def test_rclone_remotes_empty_without_rclone(monkeypatch):
    monkeypatch.setattr(datasync.shutil, "which", lambda name: None)
    assert datasync.rclone_remotes() == []


def test_rclone_sync_builds_command_with_filters(tmp_path, monkeypatch):
    fake = _FakeRun(stdout="Transferred: 3 / 3, 100%")
    monkeypatch.setattr(datasync.shutil, "which", lambda name: "/usr/bin/rclone")
    monkeypatch.setattr(datasync.subprocess, "run", fake)
    dest = tmp_path / "data"
    lines: list[str] = []
    ok, message = datasync.rclone_sync("ucbfaculty:", dest, log=lines.append)
    assert ok, message
    assert dest.is_dir()
    cmd = fake.calls[0]
    assert cmd[:4] == ["/usr/bin/rclone", "sync", "ucbfaculty:", str(dest)]
    # Plaintext spreadsheets and .git never travel.
    assert "--exclude" in cmd
    excluded = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--exclude"]
    assert ".git/**" in excluded
    assert "*.csv" in excluded and "*.xlsx" in excluded
    assert "Proposed*/**" in excluded
    assert "--dry-run" not in cmd
    assert "Transferred" in message
    assert lines and lines[0].startswith("$ ")


def test_rclone_sync_dry_run_flag(tmp_path, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(datasync.shutil, "which", lambda name: "/usr/bin/rclone")
    monkeypatch.setattr(datasync.subprocess, "run", fake)
    datasync.rclone_sync("ucbfaculty:", tmp_path / "d", dry_run=True)
    assert fake.calls[0][-1] == "--dry-run"


def test_rclone_sync_missing_remote_hint(tmp_path, monkeypatch):
    fake = _FakeRun(returncode=1, stderr="Failed to create file system for \"ucbfaculty:\": didn't find section in config file")
    monkeypatch.setattr(datasync.shutil, "which", lambda name: "/usr/bin/rclone")
    monkeypatch.setattr(datasync.subprocess, "run", fake)
    ok, message = datasync.rclone_sync("ucbfaculty:", tmp_path / "d")
    assert not ok
    assert "ucbfaculty sync" in message
    assert datasync.BOX_FOLDER_ID in message


def test_rclone_sync_without_rclone(tmp_path, monkeypatch):
    monkeypatch.setattr(datasync.shutil, "which", lambda name: None)
    ok, message = datasync.rclone_sync("ucbfaculty:", tmp_path / "d")
    assert not ok and "rclone" in message


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def test_git_pull_requires_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(datasync.shutil, "which", lambda name: "/usr/bin/git")
    ok, message = datasync.git_pull(tmp_path)
    assert not ok and "No git repository" in message


def test_git_pull_command(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    fake = _FakeRun(stdout="Already up to date.")
    monkeypatch.setattr(datasync.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(datasync.subprocess, "run", fake)
    ok, message = datasync.git_pull(tmp_path)
    assert ok and message == "Already up to date."
    assert fake.calls[0] == ["/usr/bin/git", "-C", str(tmp_path), "pull", "--ff-only"]


def test_git_clone_refuses_nonempty_dir(tmp_path, monkeypatch):
    (tmp_path / "stray").write_text("x")
    monkeypatch.setattr(datasync.shutil, "which", lambda name: "/usr/bin/git")
    ok, message = datasync.git_clone("https://example.invalid/repo.git", tmp_path)
    assert not ok and "refusing" in message


def test_git_clone_command(tmp_path, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(datasync.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(datasync.subprocess, "run", fake)
    dest = tmp_path / "clone"
    ok, _ = datasync.git_clone("https://example.invalid/repo.git", dest)
    assert ok
    assert fake.calls[0] == ["/usr/bin/git", "clone", "https://example.invalid/repo.git", str(dest)]


# ---------------------------------------------------------------------------
# Backend detection and dispatch
# ---------------------------------------------------------------------------

@pytest.fixture
def no_rclone_remote(monkeypatch):
    monkeypatch.setattr(datasync, "rclone_path", lambda: None)
    monkeypatch.setattr(datasync, "rclone_remote_exists", lambda name=datasync.DEFAULT_REMOTE: False)
    monkeypatch.setattr(datasync, "box_drive_roots", lambda *a, **k: [])


def test_detect_backend_explicit_wins(tmp_path, no_rclone_remote):
    (tmp_path / ".git").mkdir()
    assert datasync.detect_backend({"data_backend": "rclone"}, tmp_path) == "rclone"
    assert datasync.detect_backend({"data_backend": "Box-Drive"}, tmp_path) == "box-drive"


def test_detect_backend_rejects_unknown(tmp_path, no_rclone_remote):
    with pytest.raises(ValueError):
        datasync.detect_backend({"data_backend": "ftp"}, tmp_path)


def test_detect_backend_inference_order(tmp_path, monkeypatch, no_rclone_remote):
    assert datasync.detect_backend({}, tmp_path) == "none"
    assert datasync.detect_backend({"data_repo_url": "x"}, tmp_path) == "git"
    (tmp_path / ".git").mkdir()
    assert datasync.detect_backend({}, tmp_path) == "git"
    assert datasync.detect_backend({"rclone_remote": "mybox:"}, tmp_path) == "rclone"


def test_detect_backend_box_drive_and_default_remote(tmp_path, monkeypatch):
    root = tmp_path / "Box"
    (root / "RawData").mkdir(parents=True)
    monkeypatch.setattr(datasync, "box_drive_roots", lambda *a, **k: [root])
    monkeypatch.setattr(datasync, "rclone_path", lambda: "/usr/bin/rclone")
    monkeypatch.setattr(datasync, "rclone_remote_exists", lambda name=datasync.DEFAULT_REMOTE: True)
    assert datasync.detect_backend({}, root / "RawData") == "box-drive"
    assert datasync.detect_backend({}, tmp_path / "mirror") == "rclone"


def test_sync_dispatches_box_drive(tmp_path, monkeypatch):
    root = tmp_path / "Box"
    data = root / "RawData"
    data.mkdir(parents=True)
    monkeypatch.setattr(datasync, "box_drive_roots", lambda *a, **k: [root])
    ok, message = datasync.sync_data_directory({}, data)
    assert not ok and datasync.MARKER_FILE in message
    (data / datasync.MARKER_FILE).write_text("{}")
    ok, message = datasync.sync_data_directory({}, data)
    assert ok and "Box Drive" in message


def test_sync_dispatches_rclone_with_config_remote(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        datasync, "rclone_sync",
        lambda source, dest, dry_run=False, log=None: calls.append((source, Path(dest))) or (True, "ok"),
    )
    config = {"rclone_remote": "mybox:Archive", "data_directory": str(tmp_path / "d")}
    ok, _ = datasync.sync_data_directory(config)
    assert ok
    assert calls == [("mybox:Archive", tmp_path / "d")]


def test_sync_dispatches_git(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(datasync, "git_pull", lambda d, log=None: (True, f"pulled {d}"))
    ok, message = datasync.sync_data_directory({}, tmp_path)
    assert ok and message == f"pulled {tmp_path}"


def test_fetch_git_clones_when_missing(tmp_path, monkeypatch, no_rclone_remote):
    seen = {}
    monkeypatch.setattr(
        datasync, "git_clone",
        lambda url, dest, log=None: seen.update(url=url, dest=Path(dest)) or (True, "cloned"),
    )
    config = {"data_repo_url": "https://example.invalid/r.git"}
    ok, _ = datasync.fetch_data_directory(config, tmp_path / "new")
    assert ok
    assert seen == {"url": "https://example.invalid/r.git", "dest": tmp_path / "new"}


def test_fetch_git_without_url_fails(tmp_path, no_rclone_remote):
    ok, message = datasync.fetch_data_directory({"data_backend": "git"}, tmp_path / "new")
    assert not ok and "data_repo_url" in message


def test_sync_without_backend_explains(tmp_path, no_rclone_remote):
    ok, message = datasync.sync_data_directory({}, tmp_path)
    assert not ok
    assert datasync.BOX_FOLDER_URL in message
    assert "ucbfaculty sync" in message


# ---------------------------------------------------------------------------
# CLI integration: init picks up a Box Drive folder and records the backend
# ---------------------------------------------------------------------------

def test_cli_init_detects_box_drive(tmp_path, monkeypatch, capsys):
    from ucbfaculty import cli

    root = tmp_path / "Box"
    data = root / "UCBFaculty Data"
    data.mkdir(parents=True)
    (data / datasync.MARKER_FILE).write_text("{}")
    monkeypatch.setattr(datasync, "box_drive_roots", lambda *a, **k: [root])
    monkeypatch.setattr(datasync, "rclone_path", lambda: None)
    monkeypatch.setattr(datasync, "rclone_remote_exists", lambda name=datasync.DEFAULT_REMOTE: False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(cli.sys, "platform", "linux")

    cli.init_config(data_dir=None, force=True)

    import yaml

    written = yaml.safe_load((tmp_path / ".ucbfaculty" / "config.yaml").read_text())
    assert written["data_directory"] == str(data)
    assert written["data_backend"] == "box-drive"
    assert "rclone_remote" not in written
    assert "box-drive" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Upgrade path: retired git upstream, and `sync --backend` rewriting config
# ---------------------------------------------------------------------------

def test_retired_upstream_note(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(datasync, "git_upstream", lambda d: "git@github.com:ligon/RawData.git")
    note = datasync.retired_upstream_note(tmp_path)
    assert note and datasync.BOX_FOLDER_URL in note and "init --force" in note
    monkeypatch.setattr(datasync, "git_upstream", lambda d: "https://example.edu/org/RawData.git")
    assert datasync.retired_upstream_note(tmp_path) is None
    monkeypatch.setattr(datasync, "git_upstream", lambda d: None)
    assert datasync.retired_upstream_note(tmp_path) is None


def test_git_sync_appends_retired_note(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(datasync, "git_pull", lambda d, log=None: (False, "fatal: repository not found"))
    monkeypatch.setattr(datasync, "git_upstream", lambda d: "https://github.com/ligon/RawData")
    ok, message = datasync.sync_data_directory({}, tmp_path)
    assert not ok
    assert message.startswith("fatal: repository not found")
    assert "NOTE" in message and "Box" in message


def _cli_sync_env(tmp_path, monkeypatch):
    """Point ucbfaculty at a temp config and fake every transport."""
    import ucbfaculty
    from ucbfaculty import cli

    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(ucbfaculty, "config_file", config_path)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(datasync, "box_drive_roots", lambda *a, **k: [])
    monkeypatch.setattr(datasync, "rclone_path", lambda: "/usr/bin/rclone")
    monkeypatch.setattr(datasync, "rclone_remote_exists", lambda name=datasync.DEFAULT_REMOTE: True)

    def fake_fetch(config, data_dir, backend=None, log=None):
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        (Path(data_dir) / datasync.MARKER_FILE).write_text("{}")
        return True, f"fetched via {backend}"

    monkeypatch.setattr(datasync, "fetch_data_directory", fake_fetch)
    return cli, config_path


def test_cli_sync_backend_rewrites_existing_config(tmp_path, monkeypatch, capsys):
    import yaml

    cli, config_path = _cli_sync_env(tmp_path, monkeypatch)
    old_clone = tmp_path / "RawData"
    (old_clone / ".git").mkdir(parents=True)
    (old_clone / datasync.MARKER_FILE).write_text("{}")
    config_path.write_text(yaml.dump({
        "data_directory": str(old_clone),
        "spelling_dictionary": str(old_clone / "spelling_dictionary.json"),
        "coms_sheet_key": "abc123",
        "data_repo_url": "https://github.com/ligon/RawData",
    }))
    mirror = tmp_path / "mirror"
    monkeypatch.setattr(datasync, "DEFAULT_LOCAL_DIR", mirror)

    cli.sync_data(backend="rclone")

    written = yaml.safe_load(config_path.read_text())
    assert written["data_directory"] == str(mirror)
    assert written["data_backend"] == "rclone"
    assert written["rclone_remote"] == f"{datasync.DEFAULT_REMOTE}:"
    assert written["spelling_dictionary"] == str(mirror / "spelling_dictionary.json")
    assert written["coms_sheet_key"] == "abc123"          # unrelated key preserved
    assert "data_repo_url" not in written                  # stale transport key dropped
    assert "Rewrote" in capsys.readouterr().out


def test_cli_sync_without_backend_leaves_existing_config(tmp_path, monkeypatch):
    import yaml

    cli, config_path = _cli_sync_env(tmp_path, monkeypatch)
    data = tmp_path / "data"
    data.mkdir()
    (data / datasync.MARKER_FILE).write_text("{}")
    original = {"data_directory": str(data), "data_backend": "rclone", "rclone_remote": "ucbfaculty:"}
    config_path.write_text(yaml.dump(original))

    cli.sync_data()

    assert yaml.safe_load(config_path.read_text()) == original


def test_cli_sync_box_drive_backend_finds_folder(tmp_path, monkeypatch):
    import yaml

    cli, config_path = _cli_sync_env(tmp_path, monkeypatch)
    root = tmp_path / "Box"
    folder = root / "COMS Faculty Data"
    folder.mkdir(parents=True)
    (folder / datasync.MARKER_FILE).write_text("{}")
    monkeypatch.setattr(datasync, "box_drive_roots", lambda *a, **k: [root])
    config_path.write_text(yaml.dump({"data_directory": str(tmp_path / "RawData")}))

    cli.sync_data(backend="box-drive")

    written = yaml.safe_load(config_path.read_text())
    assert written["data_directory"] == str(folder)
    assert written["data_backend"] == "box-drive"
    assert "rclone_remote" not in written
