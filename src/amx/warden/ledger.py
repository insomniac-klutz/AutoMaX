"""Vault-wide ledger of certified calibration partitions (review findings on budget bypass).

The per-run certify counter can be sidestepped by re-splitting the same data into a new run
directory (the split is deterministic), by re-splitting with another seed (overlapping
calibration sets), or by adding an irrelevant column (a new data hash). The ledger therefore
fingerprints the calibration units themselves, by unit id and gold label, and refuses a new
certification whose calibration set overlaps an earlier one by more than ``MAX_OVERLAP``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from amx.split.vault import LocalVault

LEDGER_DIR = "_ledger"
INDEX_FILE = "calib_partitions.jsonl"
MAX_OVERLAP = 0.01


class PartitionUsedError(PermissionError):
    """This calibration data has already been certified (in this or another run)."""


def fingerprints(ids: Iterable[Any], gold: Iterable[Any]) -> list[str]:
    """Sorted 64-bit fingerprints of (unit id, gold label) pairs."""
    out = {
        hashlib.sha256(f"{i}\x00{g!r}".encode()).hexdigest()[:16]
        for i, g in zip(ids, gold, strict=True)
    }
    return sorted(out)


def partition_key(fps: Iterable[str]) -> str:
    return "sha256:" + hashlib.sha256("\n".join(sorted(fps)).encode()).hexdigest()


def _dir(vault: LocalVault) -> Path:
    d = vault.root / LEDGER_DIR
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


@contextmanager
def _locked(vault: LocalVault) -> Iterator[Path]:
    d = _dir(vault)
    fd = os.open(d / (INDEX_FILE + ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield d
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _index(d: Path) -> list[dict[str, Any]]:
    path = d / INDEX_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_private(path: Path, text: str, *, append: bool = False) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "a" if append else "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(path, 0o600)


def claim_partition(vault: LocalVault, fps: list[str], run_id: str) -> str:
    """Record ``run_id``'s calibration set; refuse if it overlaps an earlier claim too much."""
    key = partition_key(fps)
    mine = set(fps)
    with _locked(vault) as d:
        for rec in _index(d):
            if "file" not in rec:  # key-only record from an earlier ledger format
                share = 1.0 if rec.get("key") == key else 0.0
            else:
                prev = set((d / rec["file"]).read_text(encoding="utf-8").split())
                share = len(mine & prev) / max(len(mine), 1)
            if share > MAX_OVERLAP:
                raise PartitionUsedError(
                    f"{100 * share:.1f}% of this calibration data was already certified by run "
                    f"'{rec['run_id']}'; a new certify call needs fresh calibration data (Q1)"
                )
        name = key.split(":", 1)[1][:24] + ".fps"
        _write_private(d / name, "\n".join(fps) + "\n")
        _write_private(
            d / INDEX_FILE,
            json.dumps({"key": key, "run_id": run_id, "file": name}) + "\n",
            append=True,
        )
    return key
