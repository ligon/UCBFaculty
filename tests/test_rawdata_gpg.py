from pathlib import Path

import pytest

from ucbfaculty import rawdata, security


class DummyResult:
    def __init__(self, returncode=0, stdout=b"data", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_gpg_decrypt_uses_loopback_when_passphrase_set(monkeypatch, tmp_path):
    """When passphrase is set, loopback mode is used directly (no agent attempt)."""
    security.set_gpg_passphrase("secret")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        if "--pinentry-mode" in cmd:
            return DummyResult()  # Success for loopback
        else:
            return DummyResult(returncode=2)  # Fail for agent attempt

    monkeypatch.setattr(rawdata.subprocess, "run", fake_run)
    path = tmp_path / "data.csv.gpg"
    path.write_bytes(b"encrypted")

    rawdata._gpg_decrypt(path)

    security.clear_gpg_passphrase()

    # When passphrase is available, should skip agent and go straight to loopback
    assert len(calls) == 1
    loopback_call = calls[0]
    assert "--pinentry-mode" in loopback_call["cmd"]
    assert "--passphrase-fd" in loopback_call["cmd"]
    assert loopback_call["kwargs"]["input"] == b"secret"


def test_gpg_decrypt_uses_agent_when_available(monkeypatch, tmp_path):
    """When gpg-agent has passphrase cached, it's used directly."""
    security.clear_gpg_passphrase()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        # Agent succeeds on first try
        return DummyResult(stdout=b"decrypted data")

    monkeypatch.setattr(rawdata.subprocess, "run", fake_run)
    path = tmp_path / "data.csv.gpg"
    path.write_bytes(b"encrypted")

    result = rawdata._gpg_decrypt(path)

    assert result == b"decrypted data"
    # Only one call needed (agent worked)
    assert len(calls) == 1
    # Should NOT use loopback mode
    assert "--pinentry-mode" not in calls[0]["cmd"]


def test_gpg_decrypt_without_passphrase_falls_back(monkeypatch, tmp_path):
    """When no passphrase and agent fails, falls back to regular gpg."""
    security.clear_gpg_passphrase()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        if len(calls) == 1:
            # First call: agent attempt fails
            return DummyResult(returncode=2)
        else:
            # Second call: fallback succeeds
            return DummyResult()

    monkeypatch.setattr(rawdata.subprocess, "run", fake_run)
    path = tmp_path / "data.csv.gpg"
    path.write_bytes(b"encrypted")

    rawdata._gpg_decrypt(path)

    # Two calls: agent attempt + fallback
    assert len(calls) == 2
    # Neither should use loopback (no passphrase set)
    for call in calls:
        assert "--pinentry-mode" not in call["cmd"]


def test_gpg_decrypt_failure_raises(monkeypatch, tmp_path):
    """DecryptionError is raised with helpful message on failure."""
    security.set_gpg_passphrase("secret")

    def fake_run(cmd, **kwargs):
        # All attempts fail
        return DummyResult(returncode=2, stderr=b"[GNUPG:] ENC_TO 87654321 1 0\n[GNUPG:] BAD_PASSPHRASE 87654321\n")

    monkeypatch.setattr(rawdata.subprocess, "run", fake_run)
    path = tmp_path / "data.csv.gpg"
    path.write_bytes(b"encrypted")

    with pytest.raises(rawdata.DecryptionError) as excinfo:
        rawdata._gpg_decrypt(path)

    assert "Incorrect passphrase for key 87654321" in str(excinfo.value)
    security.clear_gpg_passphrase()


def test_try_gpg_agent_decrypt_returns_none_on_timeout(monkeypatch, tmp_path):
    """_try_gpg_agent_decrypt returns None when gpg times out."""
    import subprocess

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 5)

    monkeypatch.setattr(rawdata.subprocess, "run", fake_run)
    path = tmp_path / "data.csv.gpg"
    path.write_bytes(b"encrypted")

    result = rawdata._try_gpg_agent_decrypt(path)
    assert result is None


def test_try_gpg_agent_decrypt_returns_none_on_failure(monkeypatch, tmp_path):
    """_try_gpg_agent_decrypt returns None when gpg fails."""
    def fake_run(cmd, **kwargs):
        return DummyResult(returncode=2)

    monkeypatch.setattr(rawdata.subprocess, "run", fake_run)
    path = tmp_path / "data.csv.gpg"
    path.write_bytes(b"encrypted")

    result = rawdata._try_gpg_agent_decrypt(path)
    assert result is None
