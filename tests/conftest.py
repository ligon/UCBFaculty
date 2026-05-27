"""Shared test configuration.

When RawData is not available (e.g. CI), reconfigure ucbfaculty to use
the synthetic fixture data in tests/fixtures/.  Tests that need real
encrypted data will still skip gracefully via their own guards.
"""

import importlib
import sys
from pathlib import Path

import yaml

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture_data_available() -> bool:
    """Return True if the fixture directory has the spec files."""
    return (FIXTURES / "features_spec.yaml").exists()


def _rawdata_available() -> bool:
    """Return True if a real RawData directory is configured and present."""
    try:
        from ucbfaculty import datadir
        return (datadir / "features_spec.yaml").exists()
    except Exception:
        return False


def pytest_configure(config):
    """Point ucbfaculty at test fixtures when real data is absent."""
    if _rawdata_available():
        return  # Real data available; nothing to override

    if not _fixture_data_available():
        return  # No fixtures either; tests will skip individually

    import os

    # Build a config pointing at the fixtures directory
    cfg = {
        "data_directory": str(FIXTURES),
        "feature_specs_yaml": [str(FIXTURES / "features_spec.yaml")],
        "sources_yaml": [str(FIXTURES / "sources.yaml")],
        "spelling_dictionary": str(FIXTURES / "spelling_dictionary.json"),
        "faculty_spellings": str(FIXTURES / "faculty_spellings.json"),
        "email_dictionary": str(FIXTURES / "email_dictionary.json"),
    }

    # Write a temporary config file
    config_dir = Path.home() / ".ucbfaculty"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    # Point ucbfaculty at the fixture config via env var
    os.environ["UCBFACULTY_CONFIG"] = str(config_path)

    # Force a full reimport so all submodules pick up fixture paths.
    # Delete all ucbfaculty modules from the cache first.
    to_delete = [key for key in sys.modules if key == "ucbfaculty" or key.startswith("ucbfaculty.")]
    for key in to_delete:
        del sys.modules[key]

    importlib.import_module("ucbfaculty")
