"""
Security helpers for managing session-specific secrets.

This module provides a layered approach to GPG passphrase management:

1. gpg-agent - Uses system gpg-agent's cached passphrase (preferred)
2. In-memory passphrase - Set via set_gpg_passphrase() with loopback mode

Usage:
    # Option 1: Let gpg-agent handle it (if passphrase is cached)
    # Just use ucbfaculty normally - it will try gpg-agent first

    # Option 2: Unlock with passphrase
    from ucbfaculty.security import unlock_rawdata
    success, msg = unlock_rawdata("my-passphrase")

    # Option 3: Set passphrase for this session only
    from ucbfaculty.security import set_gpg_passphrase
    set_gpg_passphrase("my-passphrase")
"""

from __future__ import annotations

import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_GPG_PASSPHRASE: Optional[str] = None


def set_gpg_passphrase(passphrase: Optional[str]) -> None:
    """Store the GPG passphrase in memory for the current process.

    Parameters
    ----------
    passphrase : str | None
        The passphrase to store.  Passing ``None`` or an empty
        string clears any previously stored passphrase.

    Returns
    -------
    None
    """
    global _GPG_PASSPHRASE
    if passphrase:
        _GPG_PASSPHRASE = passphrase
    else:
        _GPG_PASSPHRASE = None


def get_gpg_passphrase() -> Optional[str]:
    """Return the in-memory GPG passphrase, or ``None`` if unset.

    Returns
    -------
    str | None
        The passphrase previously stored via
        :func:`set_gpg_passphrase`, or ``None``.
    """
    return _GPG_PASSPHRASE


def clear_gpg_passphrase() -> None:
    """Convenience wrapper to remove any stored GPG passphrase."""
    set_gpg_passphrase(None)


def _preset_gpg_agent(passphrase: str, keygrip: Optional[str] = None) -> bool:
    """
    Preset passphrase in gpg-agent using gpg-preset-passphrase.

    This allows subsequent GPG operations to use the cached passphrase
    without requiring loopback mode.

    Parameters
    ----------
    passphrase : str
        The passphrase to preset
    keygrip : str, optional
        The keygrip to preset. If not provided, attempts to find the
        default secret key's keygrip.

    Returns
    -------
    bool
        True if preset succeeded, False otherwise
    """
    if not keygrip:
        # Try to get the keygrip for the default secret key
        try:
            result = subprocess.run(
                ["gpg", "--list-secret-keys", "--with-keygrip"],
                capture_output=True, text=True, check=False
            )
            # Parse keygrip from output (format: "Keygrip = XXXX...")
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("Keygrip = "):
                    keygrip = line.split("=")[1].strip()
                    break
        except Exception as e:
            logger.debug(f"Failed to get keygrip: {e}")
            return False

    if not keygrip:
        logger.debug("No keygrip found")
        return False

    # Preset the passphrase
    try:
        result = subprocess.run(
            ["gpg-preset-passphrase", "--preset", keygrip],
            input=passphrase,
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            logger.debug(f"Preset passphrase for keygrip {keygrip[:8]}...")
            return True
        else:
            logger.debug(f"gpg-preset-passphrase failed: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.debug("gpg-preset-passphrase not found")
        return False
    except Exception as e:
        logger.debug(f"Failed to preset passphrase: {e}")
        return False


def _test_decrypt_with_passphrase(passphrase: str, test_file: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Test if a passphrase can successfully decrypt a GPG file.

    Parameters
    ----------
    passphrase : str
        The passphrase to test
    test_file : Path, optional
        A .gpg file to test with. If not provided, finds one in the data directory.

    Returns
    -------
    tuple[bool, str]
        (success, message) where success is True if decryption worked
    """
    if test_file is None:
        # Find a test file in the data directory
        try:
            from ucbfaculty import datadir
            gpg_files = list(datadir.glob("*.gpg"))
            if not gpg_files:
                return False, "No .gpg files found in data directory"
            test_file = gpg_files[0]
        except Exception as e:
            return False, f"Cannot locate data directory: {e}"

    # Try decryption with loopback mode
    command = [
        "gpg",
        "--batch",
        "--quiet",
        "--pinentry-mode", "loopback",
        "--passphrase-fd", "0",
        "--decrypt",
        str(test_file),
    ]
    try:
        result = subprocess.run(
            command,
            input=passphrase.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return True, "Passphrase verified"
        else:
            stderr = result.stderr.decode(errors="ignore")
            if "BAD_PASSPHRASE" in stderr or "bad passphrase" in stderr.lower():
                return False, "Incorrect passphrase"
            elif "NO_SECKEY" in stderr or "No secret key" in stderr:
                return False, "No secret key available for this file"
            else:
                return False, f"Decryption failed: {stderr.strip()[:100]}"
    except Exception as e:
        return False, f"Error testing passphrase: {e}"


def _test_gpg_agent() -> bool:
    """
    Test if gpg-agent can decrypt without a passphrase prompt.

    Returns True if gpg-agent appears to have the passphrase cached.
    """
    try:
        from ucbfaculty import datadir
        gpg_files = list(datadir.glob("*.gpg"))
        if not gpg_files:
            return False
        test_file = gpg_files[0]
    except Exception as exc:
        logger.debug("Failed to locate gpg files for agent check: %s", exc)
        return False

    # Try decryption without passphrase (relies on gpg-agent)
    command = [
        "gpg",
        "--batch",
        "--quiet",
        "--decrypt",
        str(test_file),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=5,  # Short timeout - if it hangs, agent doesn't have passphrase
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception as exc:
        logger.debug("gpg-agent unlock check failed: %s", exc)
        return False


def unlock_rawdata(
    passphrase: str,
    preset_agent: bool = True,
) -> Tuple[bool, str]:
    """
    Verify a GPG passphrase and configure the session for decryption.

    This function:
    1. Tests that the passphrase works for decrypting RawData files
    2. Stores the passphrase in memory for this session
    3. Optionally presets the passphrase in gpg-agent

    Parameters
    ----------
    passphrase : str
        The GPG passphrase to use
    preset_agent : bool, default True
        If True, attempt to preset the passphrase in gpg-agent

    Returns
    -------
    tuple[bool, str]
        (success, message) tuple indicating result

    Examples
    --------
    >>> from ucbfaculty.security import unlock_rawdata
    >>> success, msg = unlock_rawdata("my-passphrase")
    >>> if success:
    ...     print("Ready to access encrypted data")
    ... else:
    ...     print(f"Failed: {msg}")
    """
    # Test the passphrase
    success, message = _test_decrypt_with_passphrase(passphrase)
    if not success:
        return False, message

    # Store in memory
    set_gpg_passphrase(passphrase)

    # Optionally preset in gpg-agent
    if preset_agent:
        if _preset_gpg_agent(passphrase):
            logger.debug("Passphrase preset in gpg-agent")

    return True, "Passphrase verified"


def get_effective_passphrase() -> Optional[str]:
    """
    Get the passphrase to use, checking all sources in priority order.

    Resolution order:
    1. In-memory passphrase (set via set_gpg_passphrase)
    2. None (gpg-agent will be tried by the decrypt function)

    Returns
    -------
    str or None
        The passphrase if found, None otherwise
    """
    return get_gpg_passphrase()


def is_unlocked() -> bool:
    """Check whether RawData can be decrypted with the current config.

    Returns
    -------
    bool
        ``True`` if a passphrase is available in memory
        or if ``gpg-agent`` has the passphrase cached.
    """
    if get_effective_passphrase():
        return True
    return _test_gpg_agent()


def lock_rawdata() -> None:
    """Clear all cached passphrases, re-locking RawData access.

    Returns
    -------
    None
    """
    clear_gpg_passphrase()
