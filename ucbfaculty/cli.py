"""Command-line helpers for the ucbfaculty package."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import Sequence

import panel as pn

from ucbfaculty import security, set_config_file

APP_PATH = Path(__file__).resolve().parent / "apps" / "lookup.py"
SETUP_PATH = Path(__file__).resolve().parent / "apps" / "setup.py"


def _serve_lookup(no_autoreload: bool, show: bool, port: int | None, address: str | None, app_path: Path = APP_PATH) -> None:
    """Launch the Panel lookup application."""
    serve_kwargs = {
        "autoreload": not no_autoreload,
        "show": show,
    }
    if port is not None:
        serve_kwargs["port"] = port
    if address is not None:
        serve_kwargs["address"] = address

    pn.serve(str(app_path), **serve_kwargs)


def unlock() -> None:
    """Cache GPG passphrase in gpg-agent by triggering a test decryption.

    Runs ``gpg --decrypt`` on a RawData file, which causes gpg-agent to
    invoke pinentry.  The user types their passphrase into pinentry
    (never into a Python process), and gpg-agent caches it for subsequent
    operations.
    """
    import subprocess
    from ucbfaculty import datadir

    gpg_files = sorted(datadir.glob("*.gpg"))
    if not gpg_files:
        print("No .gpg files found in the data directory.")
        print(f"  Checked: {datadir}")
        sys.exit(1)

    test_file = gpg_files[0]
    print(f"Decrypting {test_file.name} to cache passphrase in gpg-agent...")
    print("(pinentry should prompt you for your passphrase)\n")

    result = subprocess.run(
        ["gpg", "--quiet", "--decrypt", str(test_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        print("Success — passphrase cached in gpg-agent.")
    else:
        stderr = result.stderr.decode(errors="ignore").strip()
        print(f"Decryption failed: {stderr}")
        sys.exit(1)


def init_config(data_dir: Path | None, force: bool = False) -> None:
    """Write ``~/.ucbfaculty/config.yaml`` for an existing RawData directory.

    Non-GUI counterpart of the setup wizard, intended for users who already
    have a RawData clone on disk (cloned manually via SSH, downloaded as a
    tarball, copied from another machine, etc.) and just need the config
    file pointing at it.

    Parameters
    ----------
    data_dir : Path or None
        Path to the existing RawData directory.  If ``None``, the function
        searches a small list of conventional locations: ``~/RawData`` and a
        ``RawData`` sibling of the UCBFaculty package directory.
    force : bool
        Overwrite an existing config.yaml if present.
    """
    import yaml

    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir).expanduser())
    else:
        # Conventional locations: home dir, then a sibling of UCBFaculty.
        candidates.append(Path.home() / "RawData")
        ucbf_pkg_parent = Path(__file__).resolve().parent.parent.parent
        candidates.append(ucbf_pkg_parent / "RawData")

    resolved: Path | None = None
    for candidate in candidates:
        if (candidate / "features_spec.yaml").exists():
            resolved = candidate
            break

    if resolved is None:
        print("Could not locate a RawData directory.")
        print("Looked in:")
        for candidate in candidates:
            marker = "exists" if candidate.exists() else "missing"
            print(f"  {candidate}  ({marker})")
        print("\nPass --data-dir PATH to point at your RawData clone.")
        sys.exit(1)

    config_dir = Path.home() / ".ucbfaculty"
    config_path = config_dir / "config.yaml"
    if config_path.exists() and not force:
        print(f"Config already exists at {config_path}.")
        print("Re-run with --force to overwrite, or edit the file directly.")
        sys.exit(1)

    config = {
        "data_directory": str(resolved),
        "feature_specs_yaml": [str(resolved / "features_spec.yaml")],
        "sources_yaml": [str(resolved / "sources.yaml")],
        "spelling_dictionary": str(resolved / "spelling_dictionary.json"),
        "faculty_spellings": str(resolved / "faculty_spellings.json"),
        "email_dictionary": str(resolved / "email_dictionary.json"),
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.dump(config, handle)
    print(f"Wrote {config_path}")
    print(f"  data_directory: {resolved}")

    # On macOS / Windows, also drop a double-clickable launcher so the
    # user never has to open a terminal again.  Best-effort: failures
    # here don't fail init -- the user can still run `ucbfaculty serve`.
    if sys.platform in ("darwin", "win32"):
        try:
            _install_launcher(dest=None, force=force)
        except (OSError, RuntimeError) as exc:
            print(f"\nNote: could not install launcher ({exc}).")
            print("Run `ucbfaculty install-launcher` manually if you want one.")


def _install_launcher(dest: Path | None, force: bool) -> None:
    """Build a platform-native double-click launcher for the Panel app.

    Dispatches by ``sys.platform``: macOS gets a ``.app`` bundle via
    :func:`ucbfaculty._mac_launcher.write_app_bundle`; other platforms
    currently print a friendly error.  Additional platforms can plug
    into the same dispatcher without changing callers.
    """
    if sys.platform == "darwin":
        from ucbfaculty import __version__
        from ucbfaculty._mac_launcher import write_app_bundle

        try:
            written = write_app_bundle(
                dest=dest, force=force, version=__version__
            )
        except FileExistsError as exc:
            print(str(exc))
            sys.exit(1)
        print(f"Wrote: {written}")
        print("Double-click the app in Finder to launch UCB Faculty Tools.")
        return

    if sys.platform == "win32":
        from ucbfaculty._win_launcher import write_bat_shortcut

        try:
            written = write_bat_shortcut(dest=dest, force=force)
        except FileExistsError as exc:
            print(str(exc))
            sys.exit(1)
        print(f"Wrote: {written}")
        print("Double-click the file on your Desktop to launch UCB Faculty Tools.")
        return

    print(f"install-launcher is not supported on {sys.platform!r}.")
    print("On Linux, launch with: `ucbfaculty serve --show`")
    sys.exit(1)


def print_config_info() -> None:
    """
    Emit diagnostic information about the active configuration and key resources.
    """
    from datetime import datetime
    import json
    from ucbfaculty import (
        config_file,
        datadir,
        feature_specs_yaml,
        sources_yaml,
        spelling_dictionary,
        faculty_spellings_path,
        FACULTY_NAME_DICTIONARY,
    )

    def _describe_path(label: str, path: Path) -> None:
        exists = path.exists()
        print(f"{label}: {path}")
        print(f"  Exists: {'yes' if exists else 'no'}")
        if exists:
            try:
                stat = path.stat()
            except OSError:
                return
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime)
            print(f"  Size: {size} bytes")
            print(f"  Modified: {modified.isoformat()}")

    print("UCB Faculty Lookup Environment")
    print("--------------------------------")
    print(f"Config path: {config_file}")
    _describe_path("Data directory", datadir)
    if isinstance(feature_specs_yaml, (list, tuple)):
        for idx, path in enumerate(feature_specs_yaml, 1):
            _describe_path(f"Feature specs YAML #{idx}", Path(path))
    else:
        _describe_path("Feature specs YAML", Path(feature_specs_yaml))
    if isinstance(sources_yaml, (list, tuple)):
        for idx, path in enumerate(sources_yaml, 1):
            _describe_path(f"Sources YAML #{idx}", Path(path))
    else:
        _describe_path("Sources YAML", Path(sources_yaml))
    _describe_path("Spelling dictionary", spelling_dictionary)
    _describe_path("Faculty spellings", faculty_spellings_path)

    sample_keys = []
    try:
        with open(faculty_spellings_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for canonical, variants in payload.items():
            sample_keys.append((canonical, variants))
            if len(sample_keys) >= 3:
                break
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = None

    print("\nDictionary sample:")
    if not sample_keys:
        print("  (unable to read faculty spellings)")
    else:
        for canonical, variants in sample_keys:
            preview = ", ".join(variants[:3]) if variants else "(no variants)"
            print(f"  {canonical} <- {preview}")

    canonical_example = "Vinod K. Aggarwal"
    try:
        resolved = FACULTY_NAME_DICTIONARY.spell(canonical_example)
        print(f"\nExample lookup: '{canonical_example}' → '{resolved}'")
    except (FileNotFoundError, OSError):
        print("\nExample lookup: (faculty spellings file not available)")


def _doctor() -> None:
    """Check data holdings against declared sources and report.

    Verifies the holdings invariant (every encrypted file at rest is
    declared in ``sources.yaml``) and that every declared source is
    present.  Exits 0 when the invariant holds, 1 on violations,
    2 when the data directory is missing.
    """
    from ucbfaculty import datadir
    from ucbfaculty.feature_spec_loader import load_source_specs_multi
    from ucbfaculty.rawdata import validate_holdings

    print(f"Data directory: {datadir}")
    if not Path(datadir).exists():
        print("  ERROR: data directory does not exist.")
        raise SystemExit(2)

    report = validate_holdings()
    specs = load_source_specs_multi()
    print(f"Declared sources: {len(report['declared'])}")
    for name in report["declared"]:
        spec = specs.get(name)
        acc = (
            f"accessioned {spec.accessioned}"
            if spec is not None and spec.accessioned
            else "accession date unrecorded"
        )
        status = "MISSING" if name in report["missing"] else "ok"
        print(f"  [{status:>7}] {name} ({acc})")
    if report["undeclared"]:
        print("Encrypted files at rest with no sources.yaml stanza:")
        for name in report["undeclared"]:
            print(f"  [VIOLATION] {name}")
        print("Declare each file in sources.yaml, or remove it from the archive.")

    ok = not report["undeclared"] and not report["missing"]
    print(
        "Holdings invariant: "
        + ("OK -- data at rest matches declared sources." if ok else "FAILED")
    )
    raise SystemExit(0 if ok else 1)


def _prompt_passphrase() -> None:
    """Ensure the security module has the desired passphrase configured."""
    prompt = "Enter GPG passphrase (leave blank to skip): "
    try:
        value = getpass.getpass(prompt)
    except (EOFError, KeyboardInterrupt):
        print("\nPassphrase prompt aborted.", file=sys.stderr)
        raise SystemExit(1) from None
    security.set_gpg_passphrase(value if value else None)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``ucbfaculty`` CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser supporting ``serve``, ``panel``, and
        ``info`` sub-commands, plus server and passphrase options.
    """
    from ucbfaculty import __version__

    parser = argparse.ArgumentParser(
        description="Utilities for working with the UCB Faculty service data tools."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print the installed ucbfaculty version and exit.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=("serve", "panel", "info", "unlock", "init", "install-launcher", "doctor"),
        help="Command to execute (default: serve).",
    )
    parser.add_argument(
        "--no-autoreload",
        action="store_true",
        help="Disable autoreload; enabled by default to mirror panel serve --autoreload.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the app in a browser after launching.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5006,
        help="Port for the server (default: 5006).",
    )
    parser.add_argument(
        "--address",
        default=None,
        help="Bind address for the server.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to an alternate config.yaml (overrides UCBFACULTY_CONFIG).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="(init only) Path to an existing RawData directory.  "
        "If omitted, ~/RawData and a RawData sibling of the install "
        "are searched.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="(init, install-launcher) Overwrite existing config.yaml / .app bundle.",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="(install-launcher only) Destination path for the .app bundle.  "
        "Defaults to ~/Applications/UCB Faculty Tools.app (or Desktop if "
        "~/Applications doesn't exist).",
    )
    parser.add_argument(
        "-p",
        "--passphrase",
        default=None,
        help="Passphrase used to decrypt GPG-protected RawData (prompts if omitted).",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt-passphrase",
        action="store_true",
        help="Prompt in the CLI for a GPG passphrase before launching the app.",
    )
    prompt_group.add_argument(
        "--defer-passphrase",
        action="store_true",
        help="Deprecated; passphrase entry is deferred to the app by default.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Panel debugger widget to capture backend errors and logs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``ucbfaculty`` command-line tool.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments.  Defaults to ``sys.argv[1:]``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        set_config_file(args.config)
    if args.command == "info":
        print_config_info()
        return
    if args.command == "unlock":
        unlock()
        return
    if args.command == "init":
        init_config(
            data_dir=Path(args.data_dir).expanduser() if args.data_dir else None,
            force=args.force,
        )
        return
    if args.command == "install-launcher":
        _install_launcher(
            dest=Path(args.dest).expanduser() if args.dest else None,
            force=args.force,
        )
        return
    if args.command == "doctor":
        _doctor()
        return
    # Check for config existence
    from ucbfaculty import config_file
    target_app = APP_PATH
    if not config_file.exists():
        print(f"Config file not found at {config_file}. Launching setup...")
        target_app = SETUP_PATH
    else:
        # Only prompt for passphrase if we are running the real app
        if args.passphrase is not None:
            security.set_gpg_passphrase(args.passphrase if args.passphrase else None)
        elif args.prompt_passphrase:
            _prompt_passphrase()

    if args.debug:
        os.environ["UCBFACULTY_DEBUG"] = "1"

    _serve_lookup(
        no_autoreload=args.no_autoreload,
        show=args.show,
        port=args.port,
        address=args.address,
        app_path=target_app,
    )


if __name__ == "__main__":
    main()
