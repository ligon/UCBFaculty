"""Shared diagnostic blob: install + environment metadata.

Single source of truth for the version / install-path / config-path
info surfaced in three places:

- ``ucbfaculty info`` (CLI)
- ``ucbfaculty-mcp --version`` (CLI)
- The "About" card in the Panel app

If a user reports "I don't see <recent feature>", the output here
should make the root cause obvious (stale install, wrong env,
shadowing checkout, ...).  The string in :func:`format_diagnostics`
is intended to be paste-ready into a support ticket.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

#: Canonical upstream URL.  Used to template the "reinstall from
#: scratch" command in the support blob.
REPO_URL = "https://github.com/ligon/UCBFaculty.git"


def get_diagnostics() -> dict[str, Any]:
    """Collect environment info into a flat dict.

    Returns
    -------
    dict
        Keys: ``version``, ``install_path``, ``python``, ``platform``,
        ``config_path``, ``data_directory``, ``repo_url``.  All values
        are strings (no ``None``); missing/unconfigured values are
        rendered as ``"(not configured)"`` so the dict can be JSON-
        serialised without further massaging.
    """
    import ucbfaculty

    install_path = Path(ucbfaculty.__file__).resolve().parent

    # `config_file` is always defined (default path even if absent),
    # but `datadir` only exists after a config has been loaded.  Treat
    # the latter as optional so a fresh, unconfigured install still
    # produces a usable diagnostic blob.
    try:
        from ucbfaculty import config_file as _config_file
        config_path = str(_config_file)
    except (ImportError, AttributeError):
        config_path = "(not configured)"

    try:
        from ucbfaculty import datadir as _datadir
        data_dir = str(_datadir) if _datadir else "(not configured)"
    except (ImportError, AttributeError):
        data_dir = "(not configured)"

    return {
        "version": getattr(ucbfaculty, "__version__", "unknown"),
        "install_path": str(install_path),
        "python": (
            f"{sys.version_info.major}.{sys.version_info.minor}"
            f".{sys.version_info.micro}"
        ),
        "platform": platform.platform(),
        "config_path": config_path,
        "data_directory": data_dir,
        "repo_url": REPO_URL,
    }


def format_diagnostics(
    d: dict[str, Any] | None = None,
    *,
    style: str = "text",
) -> str:
    """Render a diagnostic dict as plain text or Markdown.

    Parameters
    ----------
    d : dict or None
        Pre-collected diagnostic dict.  If None, :func:`get_diagnostics`
        is called.  Pass an explicit dict when you want to reuse one
        snapshot across multiple surfaces (text + markdown).
    style : {"text", "markdown"}
        ``"text"`` returns a plain monospaced block suitable for the
        terminal or a Copy-to-clipboard payload; ``"markdown"`` returns
        the same content with Markdown formatting for the Panel app's
        About card.
    """
    if d is None:
        d = get_diagnostics()

    if style == "markdown":
        return (
            f"**Version:** `{d['version']}`  \n"
            f"**Install path:** `{d['install_path']}`  \n"
            f"**Python:** {d['python']}  \n"
            f"**Platform:** {d['platform']}  \n"
            f"**Config:** `{d['config_path']}`  \n"
            f"**Data directory:** `{d['data_directory']}`  \n"
            f"\n"
            f"**Looks out of date?** Reinstall from scratch:\n"
            f"```\n"
            f"pip install --force-reinstall --no-cache-dir \\\n"
            f"  git+{d['repo_url']}@main\n"
            f"```\n"
        )

    if style == "text":
        return (
            f"ucbfaculty {d['version']}\n"
            f"  install:    {d['install_path']}\n"
            f"  python:     {d['python']}\n"
            f"  platform:   {d['platform']}\n"
            f"  config:     {d['config_path']}\n"
            f"  data dir:   {d['data_directory']}\n"
            f"\n"
            f"Looks out of date?  Reinstall with:\n"
            f"  pip install --force-reinstall --no-cache-dir \\\n"
            f"    git+{d['repo_url']}@main\n"
        )

    raise ValueError(f"Unknown style: {style!r} (expected 'text' or 'markdown')")
