"""Vault-wide ledger of certified calibration partitions (review finding: budget per partition).

The per-run certify counter can be sidestepped by re-splitting the same data into a new run
directory: the split is deterministic, so the new run gets the same calibration fold with a
fresh counter. The ledger keys the budget by the partition itself: sha256 of the data hash and
the sorted calibration unit ids. A partition certified once (in any run) is refused afterwards.
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
LEDGER_FILE = "calib_partitions.jsonl"


class PartitionUsedError(PermissionError):
    """This calibration partition has already been certified."""


def partition_key(data_hash: str, calib_ids: Iterable[str]) -> str:
    payload = data_hash + "\n" + "\n".join(sorted(str(i) for i in calib_ids))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dir(vault: LocalVault) -> Path:
    d = vault.root / LEDGER_DIR
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


@contextmanager
def _locked(vault: LocalVault) -> Iterator[Path]:
    d = _dir(vault)
    lock = d / (LEDGER_FILE + ".lock")
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield d / LEDGER_FILE
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def claim_partition(vault: LocalVault, key: str, run_id: str) -> None:
    """Record that ``run_id`` certifies partition ``key``; refuse if any run already did."""
    with _locked(vault) as path:
        for rec in _records(path):
            if rec["key"] == key:
                raise PartitionUsedError(
                    f"this calibration partition was already certified by run '{rec['run_id']}'; "
                    "a new certify call needs fresh calibration data (OQ Q1)"
                )
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "run_id": run_id}) + "\n")
        os.chmod(path, 0o600)


def partition_claimed(vault: LocalVault, key: str) -> bool:
    with _locked(vault) as path:
        return any(rec["key"] == key for rec in _records(path))
