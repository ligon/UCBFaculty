import importlib
import sys
from pathlib import Path

import pytest


_RAWDATA_CANDIDATES = [
    Path("/var/tmp/coder-mirrors/RawData"),
    Path.home() / "RawData",
]


def _find_rawdata() -> Path:
    """Return a local RawData path if one exists, else skip the test."""
    for candidate in _RAWDATA_CANDIDATES:
        if (candidate / "faculty_spellings.json").exists():
            return candidate
    pytest.skip("RawData directory not available in test environment")


@pytest.fixture
def temp_config(monkeypatch, tmp_path):
    data_dir = _find_rawdata()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"data_directory: {data_dir}",
                f"spelling_dictionary: {data_dir / 'spelling_dictionary.json'}",
                f"faculty_spellings: {data_dir / 'faculty_spellings.json'}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("UCBFACULTY_CONFIG", str(config_path))
    # Ensure module picks up new configuration
    if "ucbfaculty" in sys.modules:
        sys.modules.pop("ucbfaculty")
    module = importlib.import_module("ucbfaculty")
    yield config_path, module


def test_info_command_prints_paths(temp_config, capsys):
    _, module = temp_config
    cli = importlib.import_module("ucbfaculty.cli")

    cli.print_config_info()
    out = capsys.readouterr().out

    assert str(module.config_file) in out
    assert "Data directory" in out
    assert "Exists: yes" in out


def test_faculty_spellings_path_exists(temp_config):
    _, module = temp_config
    assert module.faculty_spellings_path.exists()
