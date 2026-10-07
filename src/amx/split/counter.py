"""Certify-call counter and sealed-touch log, kept in the vault (HANDOFF 7.11, OQ Q1).

Both are append-only JSONL files in the run's vault directory. Every read-modify-append is
done under an exclusive ``fcntl.flock`` on a sibling lock file, so the limits hold across
concurrent processes on one host (POSIX only). The first ``max_calls`` ever recorded caps
later counters: re-opening the counter with a larger budget does not add calls.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from amx._log import get_logger
from amx.split.errors import CertifyBudgetExhausted, SealedTouchError, VaultError
from amx.split.vault import FILE_MODE, LocalVault

log = get_logger(__name__)

TouchKind = Literal["final_report", "simulate"]
TOUCH_KINDS: tuple[TouchKind, ...] = ("final_report", "simulate")


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, FILE_MODE)
    try:
        os.fchmod(fd, FILE_MODE)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VaultError(f"{path.name} line {lineno} is corrupt; refusing to continue") from exc
        if not isinstance(rec, dict):
            raise VaultError(f"{path.name} line {lineno} is not a record")
        records.append(rec)
    return records


def _append_record(path: Path, record: dict[str, Any]) -> None:
    line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, FILE_MODE)
    try:
        os.fchmod(fd, FILE_MODE)
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class CertifyCounter:
    """At most ``max_calls`` certify calls per run, enforced across processes."""

    LOG_NAME = "certify_calls.jsonl"
    LOCK_NAME = "certify_calls.lock"

    def __init__(self, vault: LocalVault, run_id: str, max_calls: int) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self.vault = vault
        self.run_id = run_id
        self.max_calls = int(max_calls)
        self._log = vault.file_path(run_id, self.LOG_NAME)
        self._lock = vault.file_path(run_id, self.LOCK_NAME)

    def _limit(self, records: list[dict[str, Any]]) -> int:
        recorded = [int(r["max_calls"]) for r in records if isinstance(r.get("max_calls"), int)]
        return min([self.max_calls, *recorded])

    def records(self) -> list[dict[str, Any]]:
        with _locked(self._lock):
            return _read_records(self._log)

    def used(self) -> int:
        return len(self.records())

    def remaining(self) -> int:
        with _locked(self._lock):
            records = _read_records(self._log)
            return max(0, self._limit(records) - len(records))

    def acquire(self, purpose: str) -> int:
        """Record one certify call and return its 0-based index.

        Raises :class:`CertifyBudgetExhausted` once ``used == max_calls``.
        """
        with _locked(self._lock):
            records = _read_records(self._log)
            used = len(records)
            limit = self._limit(records)
            if used >= limit:
                raise CertifyBudgetExhausted(
                    f"run '{self.run_id}' has used {used} of {limit} certify call(s); "
                    "a further call needs fresh calib data (OQ Q1)"
                )
            _append_record(
                self._log,
                {
                    "index": used,
                    "purpose": str(purpose),
                    "max_calls": self.max_calls,
                    "pid": os.getpid(),
                    "ts": _now(),
                },
            )
        log.info("certify call %d of %d acquired for run %s", used + 1, limit, self.run_id)
        return used


class SealedTouchLog:
    """Touches of the sealed fold. ``final_report`` is single-touch per release (HANDOFF 7.11).

    A second ``final_report`` raises :class:`SealedTouchError` unless ``new_sealed=True``
    declares fresh sealed data, which opens a new release. ``simulate`` is recorded only.
    """

    LOG_NAME = "sealed_touches.jsonl"
    LOCK_NAME = "sealed_touches.lock"

    def __init__(self, vault: LocalVault, run_id: str) -> None:
        self.vault = vault
        self.run_id = run_id
        self._log = vault.file_path(run_id, self.LOG_NAME)
        self._lock = vault.file_path(run_id, self.LOCK_NAME)

    def touches(self) -> list[dict[str, Any]]:
        with _locked(self._lock):
            return _read_records(self._log)

    def record(self, kind: TouchKind | str, *, new_sealed: bool = False, note: str = "") -> int:
        """Record a touch of kind ``kind`` and return the release number it belongs to."""
        if kind not in TOUCH_KINDS:
            raise ValueError(f"touch kind must be one of {TOUCH_KINDS}, got {kind!r}")
        if new_sealed and kind != "final_report":
            raise ValueError("new_sealed only applies to a final_report touch")
        with _locked(self._lock):
            records = _read_records(self._log)
            release = max((int(r.get("release", 0)) for r in records), default=0)
            if kind == "final_report":
                if new_sealed:
                    release += 1
                elif any(
                    r.get("kind") == "final_report" and int(r.get("release", 0)) == release
                    for r in records
                ):
                    raise SealedTouchError(
                        f"the sealed fold of run '{self.run_id}' was already used for a final "
                        "report in this release; a second one needs fresh sealed data "
                        "(--new-sealed)"
                    )
            _append_record(
                self._log,
                {
                    "kind": kind,
                    "release": release,
                    "new_sealed": bool(new_sealed),
                    "note": note,
                    "pid": os.getpid(),
                    "ts": _now(),
                },
            )
        log.info("sealed touch %s recorded for run %s (release %d)", kind, self.run_id, release)
        return release
