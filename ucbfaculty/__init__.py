#!/usr/bin/env python
"""ucbfaculty -- Faculty data tools for Berkeley's Academic Senate.

Submodules
----------
ucbfaculty.faculty
    Faculty vitals, committee membership, department lookups.
ucbfaculty.new_feature
    ``Feature`` / ``Features`` classes for loading data from YAML specs.
ucbfaculty.merge_features
    ``build_feature_dataframe()`` to combine features into a DataFrame.
ucbfaculty.security
    GPG passphrase management (``unlock_rawdata``, ``set_gpg_passphrase``).
ucbfaculty.rawdata
    Low-level encrypted-file loading (``get_dataframe``).
ucbfaculty.spelling
    Name resolution via spelling dictionaries.

Quick start
-----------
::

    from ucbfaculty.faculty import faculty_vitals, committee_membership
    from ucbfaculty.new_feature import Feature, Features

CLI
---
``ucbfaculty serve``   -- launch the Panel web app
``ucbfaculty unlock``  -- cache GPG passphrase via pinentry / gpg-agent
``ucbfaculty info``    -- print diagnostic info about paths and config
``ucbfaculty-mcp``     -- start the MCP server for LLM tool use
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("ucbfaculty")
except PackageNotFoundError:
    __version__ = "0.0.0"

import os
from pathlib import Path
import yaml
from typing import Any, Iterable


def _default_config_path() -> Path:
    env_path = os.environ.get("UCBFACULTY_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".ucbfaculty" / "config.yaml"


config_file = _default_config_path()


def _expand_pathlike(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("~"):
        return str(Path(value).expanduser())
    if isinstance(value, (list, tuple)):
        return [_expand_pathlike(v) for v in value]
    return value


def load_config(target: Path | None = None) -> dict:
    """Load configuration from the specified config file."""
    path = Path(target).expanduser() if target else config_file
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. "
            "Please create it with data_directory and spelling_dictionary paths."
        )

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    for key, value in config.items():
        config[key] = _expand_pathlike(value)
    return config


def get_data_dir(target: Path | None = None) -> Path:
    """Get the data directory from config if it exists."""
    try:
        config = load_config(target)
        return Path(config["data_directory"]).expanduser()
    except FileNotFoundError:
        return Path.home() / ".ucbfaculty" / "data"


def _resolve_multi_paths(
    candidates: Any, base_dir: Path, default_filename: str,
) -> Path | tuple[Path, ...]:
    """Resolve one or more paths from a config value.

    When *candidates* is a list, returns a tuple of resolved Paths.
    When it is a single string or missing, returns a single Path.
    """
    if not candidates:
        return base_dir / default_filename
    if isinstance(candidates, Iterable) and not isinstance(candidates, (str, bytes)):
        paths = []
        for candidate in candidates:
            candidate_path = Path(candidate).expanduser()
            if not candidate_path.is_absolute():
                candidate_path = base_dir / candidate_path
            paths.append(candidate_path)
        if len(paths) == 1:
            return paths[0]
        return tuple(paths)
    candidate_path = Path(candidates).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = base_dir / candidate_path
    return candidate_path


def get_spelling_dictionary(key: str = "spelling_dictionary", target: Path | None = None) -> Path:
    """Resolve a path to a spelling dictionary entry.

    Resolution order:

    1. The path under ``key`` in the user's config file, if explicitly
       set.  Returned as-is whether or not it exists on disk, so that
       typos in the config surface as ``FileNotFoundError`` from the
       loader with the user's path in the message.
    2. ``{data_directory}/{key}.json`` as the conventional default
       location under the configured data directory.
    3. For ``key == "spelling_dictionary"`` *only*: when (1) was not
       set and (2) does not exist on disk, fall back to the
       column-header normalisation dictionary shipped with the
       package.  This lets ``pip install ucbfaculty`` give basic name
       normalisation even without ``RawData`` configured.  Sensitive
       dictionaries (``faculty_spellings``, ``email_dictionary``)
       have no packaged fallback and may resolve to a path that does
       not exist; their callers are expected to handle that.
    """
    base_dir = get_data_dir(target)
    try:
        config = load_config(target)
        value = config.get(key)
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                return base_dir / path
            return path
    except FileNotFoundError:
        pass
    default_path = base_dir / f"{key}.json"
    if default_path.exists() or key != "spelling_dictionary":
        return default_path
    packaged = Path(__file__).resolve().parent / "spelling_dictionary.json"
    if packaged.exists():
        return packaged
    return default_path

def _resolve_single_path(candidates: Any, base_dir: Path, default_filename: str) -> Path:
    """Resolve a single path from config (strings may be relative to the data dir)."""
    if not candidates:
        return base_dir / default_filename
    if isinstance(candidates, Iterable) and not isinstance(candidates, (str, bytes)):
        # Prefer the first entry if a list/tuple was supplied
        candidates = next(iter(candidates), None)
        if candidates is None:
            return base_dir / default_filename
    candidate_path = Path(candidates).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = base_dir / candidate_path
    return candidate_path


def get_feature_specs_yaml(target: Path | None = None) -> Path | tuple[Path, ...]:
    """Get the feature_specs.yaml path(s) from config if specified."""
    base_dir = get_data_dir(target)
    try:
        config = load_config(target)
        return _resolve_multi_paths(config.get("feature_specs_yaml"), base_dir, "features_spec.yaml")
    except FileNotFoundError:
        return base_dir / "features_spec.yaml"


def get_sources_yaml(target: Path | None = None) -> Path | tuple[Path, ...]:
    """Get the sources.yaml path(s) from config if specified."""
    base_dir = get_data_dir(target)
    try:
        config = load_config(target)
        return _resolve_multi_paths(config.get("sources_yaml"), base_dir, "sources.yaml")
    except FileNotFoundError:
        return base_dir / "sources.yaml"


def set_config_file(path: Path | str) -> None:
    """Update the active config file location and refresh derived paths."""
    global config_file
    config_file = Path(path).expanduser()
    _refresh_resource_paths()


def _refresh_resource_paths() -> None:
    """Refresh derived paths and dictionaries after config updates."""
    global datadir, feature_specs_yaml, sources_yaml
    global spelling_dictionary, faculty_spellings_path, FACULTY_NAME_DICTIONARY
    global email_dictionary_path, EMAIL_DICTIONARY, current_year
    datadir = get_data_dir(config_file)
    feature_specs_yaml = get_feature_specs_yaml(config_file)
    sources_yaml = get_sources_yaml(config_file)
    spelling_dictionary = get_spelling_dictionary(target=config_file)
    faculty_spellings_path = get_spelling_dictionary(key="faculty_spellings", target=config_file)
    from . import spelling  # Local import to avoid circular dependency

    FACULTY_NAME_DICTIONARY = spelling.Dictionary(faculty_spellings_path)
    email_dictionary_path = get_spelling_dictionary(key="email_dictionary", target=config_file)
    EMAIL_DICTIONARY = spelling.Dictionary(email_dictionary_path)
    current_year = None
    # Pre-load specs if available; fall back quietly if files are missing.
    try:
        from .feature_spec_loader import load_feature_specs_multi, load_source_specs_multi
    except Exception:
        return
    try:
        globals()["FEATURE_SPECS"] = load_feature_specs_multi(feature_specs_yaml)
    except Exception:
        globals()["FEATURE_SPECS"] = {}
    try:
        globals()["SOURCE_SPECS"] = load_source_specs_multi(sources_yaml)
    except Exception:
        globals()["SOURCE_SPECS"] = {}


# Initialize derived resources on import
datadir: Path
feature_specs_yaml: Path | tuple[Path, ...]
sources_yaml: Path | tuple[Path, ...]
spelling_dictionary: Path
faculty_spellings_path: Path
email_dictionary_path: Path
FEATURE_SPECS: dict
SOURCE_SPECS: dict
FACULTY_NAME_DICTIONARY = None  # type: ignore[assignment]
EMAIL_DICTIONARY = None  # type: ignore[assignment]
# Dynamically resolved current academic year (set lazily in faculty.get_current_year()).
current_year: str | None = None
_refresh_resource_paths()

# Convenience setter for current_year (used by faculty.get_current_year)
def set_current_year(value: str) -> None:
    global current_year
    current_year = value
