"""Tests for the holdings invariant and accession provenance.

Covers ``validate_holdings`` (data at rest == declared sources),
the once-per-process loader warning, the ``_accessioned`` field on
``SourceSpec``, and the ``ucbfaculty doctor`` exit codes.
"""

from pathlib import Path

import pytest

from ucbfaculty import cli, rawdata
from ucbfaculty.feature_spec_loader import load_source_specs
from ucbfaculty.rawdata import validate_holdings


def _write_sources(tmp_path: Path, names, accessioned: str | None = '"2026-01-14"') -> Path:
    """Write a minimal sources.yaml declaring *names* and return its path."""
    lines = []
    for n in names:
        lines.append(f'"{n}":')
        if accessioned is not None:
            lines.append(f"  _accessioned: {accessioned}")
        lines.append("  _authority_for:")
        lines.append("    - Email")
        lines.append('  _years: ["2025-26"]')
    spec = tmp_path / "sources.yaml"
    spec.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return spec


def test_validate_holdings_clean(tmp_path):
    """Declared sources present, nothing extra: empty undeclared/missing."""
    names = ["a.csv.gpg", "b.xlsx.gpg"]
    spec = _write_sources(tmp_path, names)
    for n in names:
        (tmp_path / n).write_bytes(b"ciphertext")

    report = validate_holdings(data_dir=tmp_path, sources=spec)

    assert report["declared"] == sorted(names)
    assert report["at_rest"] == sorted(names)
    assert report["undeclared"] == []
    assert report["missing"] == []


def test_validate_holdings_flags_undeclared(tmp_path):
    """An encrypted file with no stanza is an invariant violation."""
    spec = _write_sources(tmp_path, ["a.csv.gpg"])
    (tmp_path / "a.csv.gpg").write_bytes(b"ciphertext")
    (tmp_path / "stray.xlsx.gpg").write_bytes(b"ciphertext")

    report = validate_holdings(data_dir=tmp_path, sources=spec)

    assert report["undeclared"] == ["stray.xlsx.gpg"]
    assert report["missing"] == []


def test_validate_holdings_flags_missing(tmp_path):
    """A declared source absent from the data directory is reported."""
    spec = _write_sources(tmp_path, ["a.csv.gpg", "gone.csv.gpg"])
    (tmp_path / "a.csv.gpg").write_bytes(b"ciphertext")

    report = validate_holdings(data_dir=tmp_path, sources=spec)

    assert report["missing"] == ["gone.csv.gpg"]
    assert report["undeclared"] == []


def test_sourcespec_accessioned_coerced_to_str(tmp_path):
    """A bare YAML date (parsed as datetime.date) normalises to ISO string."""
    spec = _write_sources(tmp_path, ["a.csv.gpg"], accessioned="2026-01-14")

    specs = load_source_specs(spec)

    assert specs["a.csv.gpg"].accessioned == "2026-01-14"
    assert isinstance(specs["a.csv.gpg"].accessioned, str)


def test_sourcespec_accessioned_defaults_none(tmp_path):
    spec = _write_sources(tmp_path, ["a.csv.gpg"], accessioned=None)

    specs = load_source_specs(spec)

    assert specs["a.csv.gpg"].accessioned is None


def test_loader_warns_once_on_undeclared(monkeypatch):
    """The encrypted-load path warns exactly once per process."""
    monkeypatch.setattr(rawdata, "_HOLDINGS_WARNED", False)
    monkeypatch.setattr(
        rawdata,
        "validate_holdings",
        lambda *a, **k: {
            "declared": [],
            "at_rest": ["stray.csv.gpg"],
            "undeclared": ["stray.csv.gpg"],
            "missing": [],
        },
    )

    with pytest.warns(UserWarning, match="stray.csv.gpg"):
        rawdata._warn_undeclared_holdings_once()

    # Second call: flag set, no further warning.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        rawdata._warn_undeclared_holdings_once()


def test_loader_warning_never_raises(monkeypatch):
    """A broken holdings check must not break data loading."""
    monkeypatch.setattr(rawdata, "_HOLDINGS_WARNED", False)

    def boom(*a, **k):
        raise RuntimeError("no spec available")

    monkeypatch.setattr(rawdata, "validate_holdings", boom)

    rawdata._warn_undeclared_holdings_once()  # must not raise


@pytest.mark.parametrize(
    "undeclared,missing,expected_code",
    [([], [], 0), (["stray.csv.gpg"], [], 1), ([], ["gone.csv.gpg"], 1)],
)
def test_doctor_exit_codes(monkeypatch, tmp_path, capsys, undeclared, missing, expected_code):
    """doctor exits 0 only when the holdings invariant holds.

    Patch via :func:`importlib.import_module` (the live ``sys.modules``
    entries) rather than monkeypatch string paths: other tests reload
    the ``ucbfaculty`` package, which can drop submodule attributes
    from the package namespace and break string-path resolution.
    """
    import importlib

    pkg = importlib.import_module("ucbfaculty")
    rawdata_mod = importlib.import_module("ucbfaculty.rawdata")
    loader_mod = importlib.import_module("ucbfaculty.feature_spec_loader")

    declared = ["a.csv.gpg"] + missing
    monkeypatch.setattr(pkg, "datadir", tmp_path)
    monkeypatch.setattr(
        rawdata_mod,
        "validate_holdings",
        lambda *a, **k: {
            "declared": declared,
            "at_rest": ["a.csv.gpg"] + undeclared,
            "undeclared": undeclared,
            "missing": missing,
        },
    )
    monkeypatch.setattr(loader_mod, "load_source_specs_multi", lambda *a, **k: {})

    with pytest.raises(SystemExit) as excinfo:
        cli._doctor()

    assert excinfo.value.code == expected_code
    out = capsys.readouterr().out
    assert ("OK" in out) == (expected_code == 0)


def test_doctor_missing_datadir_exits_2(monkeypatch, tmp_path):
    import importlib

    pkg = importlib.import_module("ucbfaculty")
    monkeypatch.setattr(pkg, "datadir", tmp_path / "nonexistent")

    with pytest.raises(SystemExit) as excinfo:
        cli._doctor()

    assert excinfo.value.code == 2
