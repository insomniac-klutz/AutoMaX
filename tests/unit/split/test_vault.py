from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from amx.split import CertifyCounter, LocalVault, SealedTouchLog, VaultError
from amx.split.vault import LOCAL_DIR_WARNING, ROOT_NOTE
from tests.unit.split.helpers import make_frame


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_default_root_follows_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMX_VAULT", str(tmp_path / "v"))
    assert LocalVault().root == (tmp_path / "v").resolve()
    assert LocalVault(tmp_path / "w").root == (tmp_path / "w").resolve()
    monkeypatch.delenv("AMX_VAULT")
    assert LocalVault().root == (Path.home() / ".amx" / "vault").resolve()
    assert LocalVault.mode == "local_dir"


def test_dir_and_file_modes(tmp_path: Path) -> None:
    old = os.umask(0o002)  # a permissive umask must not loosen vault modes
    try:
        vault = LocalVault(tmp_path / "vault")
        uf = make_frame(20)
        vault.write_fold("run-1", "calib", uf)
        vault.write_fold("run-1", "sealed", uf)
        vault.hmac_key("run-1")
        vault.write_text("run-1", "taskspec.yaml", "name: x\n")
        CertifyCounter(vault, "run-1", 2).acquire("test")
        SealedTouchLog(vault, "run-1").record("simulate")
    finally:
        os.umask(old)
    run = vault.root / "run-1"
    dirs = [vault.root, run, *(p for p in run.rglob("*") if p.is_dir())]
    files = [p for p in run.rglob("*") if p.is_file()]
    assert len(files) >= 7
    for d in dirs:
        assert mode(d) == 0o700, d
    for f in files:
        assert mode(f) == 0o600, f
    assert sorted(p.name for p in files if p.name.endswith(".tmp")) == []


def test_fold_round_trip_and_refusals(tmp_path: Path) -> None:
    vault = LocalVault(tmp_path / "vault")
    uf = make_frame(25, time=list(range(25)), series=["s"] * 25)
    vault.write_fold("r", "sealed", uf)
    assert vault.has_fold("r", "sealed") and not vault.has_fold("r", "calib")
    back = vault.read_fold("r", "sealed")
    assert back.roles == uf.roles
    assert back.content_hash() == uf.content_hash()
    with pytest.raises(VaultError, match="already exists"):
        vault.write_fold("r", "sealed", uf)
    vault.write_fold("r", "sealed", uf.take(range(5)), overwrite=True)
    assert vault.read_fold("r", "sealed").n == 5
    with pytest.raises(VaultError, match="not in the vault"):
        vault.read_fold("r", "calib")
    with pytest.raises(VaultError, match="folds are"):
        vault.write_fold("r", "dev", uf)


@pytest.mark.parametrize("bad", ["../escape", "a/b", ".", "..", "", ".hidden"])
def test_invalid_run_ids_are_refused(tmp_path: Path, bad: str) -> None:
    vault = LocalVault(tmp_path / "vault")
    with pytest.raises(VaultError, match="invalid"):
        vault.run_path(bad)
    with pytest.raises(VaultError, match="invalid"):
        vault.write_text("ok", bad, "x")


def test_hmac_key_is_created_once(tmp_path: Path) -> None:
    vault = LocalVault(tmp_path / "vault")
    k1 = vault.hmac_key("r")
    assert isinstance(k1, bytes) and len(k1) == 32
    assert LocalVault(tmp_path / "vault").hmac_key("r") == k1
    assert vault.hmac_key("other") != k1
    (vault.root / "r" / "hmac.key").write_bytes(b"short")
    with pytest.raises(VaultError, match="corrupt"):
        vault.hmac_key("r")


def test_text_snapshots(tmp_path: Path) -> None:
    vault = LocalVault(tmp_path / "vault")
    vault.write_text("r", "taskspec.yaml", "name: a\n")
    assert vault.read_text("r", "taskspec.yaml") == "name: a\n"
    vault.write_text("r", "taskspec.yaml", "name: b\n")
    assert vault.read_text("r", "taskspec.yaml") == "name: b\n"
    with pytest.raises(VaultError, match="already exists"):
        vault.write_text("r", "taskspec.yaml", "name: c\n", overwrite=False)
    with pytest.raises(VaultError, match="not in the vault"):
        vault.read_text("r", "missing.txt")


def test_isolation_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = LocalVault(tmp_path / "vault")
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    warning = vault.isolation_warning()
    assert warning is not None
    assert "Not safe against a determined agent running as the same user" in warning
    assert warning == LOCAL_DIR_WARNING
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    as_root = vault.isolation_warning()
    assert as_root is not None and ROOT_NOTE in as_root and LOCAL_DIR_WARNING in as_root


def test_existing_root_is_not_chmodded(tmp_path: Path) -> None:
    """An existing root may be a shared directory such as /tmp: amx must not chmod it."""
    root = tmp_path / "shared-root"
    root.mkdir()
    os.chmod(root, 0o1777)
    vault = LocalVault(root)
    vault.write_fold("run-1", "calib", make_frame(20))
    vault.write_text("run-1", "taskspec.yaml", "name: x\n")
    assert mode(root) == 0o1777
    assert mode(root / "run-1") == 0o700  # what amx creates is still private
    assert mode(root / "run-1" / "folds") == 0o700
    warning = vault.isolation_warning()
    assert warning is not None and "1777" in warning


def test_new_root_and_existing_run_dirs_are_forced_private(tmp_path: Path) -> None:
    old = os.umask(0o002)
    try:
        vault = LocalVault(tmp_path / "parent" / "new-root")
        vault.write_text("r", "taskspec.yaml", "name: x\n")
        assert mode(vault.root) == 0o700  # created by amx
        run = vault.root / "r"
        (run / "folds").mkdir()
        os.chmod(run, 0o755)
        os.chmod(run / "folds", 0o777)
        vault.write_fold("r", "sealed", make_frame(20))
    finally:
        os.umask(old)
    assert mode(run) == 0o700 and mode(run / "folds") == 0o700


def test_shared_root_of_another_user_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "shared-root"
    root.mkdir()
    os.chmod(root, 0o755)
    other_uid = root.stat().st_uid + 1
    monkeypatch.setattr(os, "geteuid", lambda: other_uid)
    vault = LocalVault(root)
    with pytest.raises(VaultError, match="not owned by the current user"):
        vault.write_text("r", "taskspec.yaml", "name: x\n")
    with pytest.raises(VaultError, match="not owned by the current user"):
        vault.write_fold("r", "calib", make_frame(20))
    assert mode(root) == 0o755 and list(root.iterdir()) == []
    os.chmod(root, 0o700)  # a private root of another user is left to the OS to refuse
    vault.write_text("r", "taskspec.yaml", "name: x\n")
    assert mode(root / "r") == 0o700


def test_root_that_is_a_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "not-a-dir"
    path.write_text("x")
    with pytest.raises(VaultError, match="not a directory"):
        LocalVault(path).write_text("r", "taskspec.yaml", "name: x\n")
