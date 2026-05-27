import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

import ucbfaculty.security as sec


@pytest.fixture(autouse=True)
def _clear_passphrase():
    """Ensure clean state before and after each test."""
    sec.clear_gpg_passphrase()
    yield
    sec.clear_gpg_passphrase()


class TestPassphraseMemory:
    def test_set_get_roundtrip(self):
        sec.set_gpg_passphrase("secret")
        assert sec.get_gpg_passphrase() == "secret"

    def test_clear(self):
        sec.set_gpg_passphrase("secret")
        sec.clear_gpg_passphrase()
        assert sec.get_gpg_passphrase() is None

    def test_set_none_clears(self):
        sec.set_gpg_passphrase("secret")
        sec.set_gpg_passphrase(None)
        assert sec.get_gpg_passphrase() is None

    def test_set_empty_clears(self):
        sec.set_gpg_passphrase("secret")
        sec.set_gpg_passphrase("")
        assert sec.get_gpg_passphrase() is None


class TestGetEffectivePassphrase:
    def test_memory_first(self):
        sec.set_gpg_passphrase("mem")
        assert sec.get_effective_passphrase() == "mem"

    def test_none_when_empty(self):
        assert sec.get_effective_passphrase() is None


class TestLockRawdata:
    def test_clears_memory(self):
        sec.set_gpg_passphrase("secret")
        sec.lock_rawdata()
        assert sec.get_gpg_passphrase() is None

class TestTestDecrypt:
    def test_success(self, tmp_path):
        gpg_file = tmp_path / "test.gpg"
        gpg_file.write_bytes(b"data")
        result = MagicMock(returncode=0)
        with patch("ucbfaculty.security.subprocess.run", return_value=result):
            ok, msg = sec._test_decrypt_with_passphrase("pass", test_file=gpg_file)
        assert ok is True

    def test_bad_passphrase(self, tmp_path):
        gpg_file = tmp_path / "test.gpg"
        gpg_file.write_bytes(b"data")
        result = MagicMock(returncode=2, stderr=b"BAD_PASSPHRASE")
        with patch("ucbfaculty.security.subprocess.run", return_value=result):
            ok, msg = sec._test_decrypt_with_passphrase("wrong", test_file=gpg_file)
        assert ok is False
        assert "Incorrect" in msg


class TestIsUnlocked:
    def test_true_with_passphrase(self):
        sec.set_gpg_passphrase("secret")
        assert sec.is_unlocked() is True

    def test_falls_back_to_agent(self):
        with patch.object(sec, "_test_gpg_agent", return_value=True):
            assert sec.is_unlocked() is True

    def test_false_when_nothing(self):
        with patch.object(sec, "_test_gpg_agent", return_value=False):
            assert sec.is_unlocked() is False
