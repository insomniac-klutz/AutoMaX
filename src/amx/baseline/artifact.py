"""Freezing and loading the A0 trivial-model artifact (HANDOFF 5.1, 5.2; ROLLER step 9).

An artifact directory holds ``model.joblib`` (the fitted predictor) and ``meta.json`` (family,
roles, TaskSpec hash, τ grid hash, ``dev_cov`` on the grid, library versions). Its hash is

    sha256 over the sorted list of (relative path, sha256 of the file)

taken over every file except ``artifact.sha256``, which records the hash at freeze time.
:func:`load_artifact` recomputes it and refuses a mismatch with the recorded value or with
the caller's ``expected_hash``, *before* unpickling anything. It reads each file into memory
once and parses the very bytes it hashed, so a file swapped on disk after hashing is never
the one loaded.

The same fit must give the same hash (ROLLER step 9), so :func:`freeze` refuses meta that
holds a time stamp: a key matching ``time|date|created|stamp`` (case-insensitive, at any
depth) or a string value that parses as an ISO-8601 date or datetime. The ``roles`` block is
exempt: it names columns (one role is called ``time``), not run metadata.

The hash proves integrity, not authenticity: whoever can rewrite the directory can rewrite
the recorded hash too. The warden therefore pins the hash it froze and passes it as
``expected_hash`` (signing arrives in A2). Never load an artifact from an untrusted source:
``joblib.load`` executes pickled code.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from amx._log import get_logger
from amx.baseline.trivial import ScoredPredictor
from amx.cert.grid import TauGrid

log = get_logger(__name__)

MODEL_FILE = "model.joblib"
META_FILE = "meta.json"
HASH_FILE = "artifact.sha256"
ARTIFACT_FORMAT = 1
REQUIRED_META = ("family", "roles", "spec_hash", "grid_hash", "dev_cov")
_VERSIONED = ("automax", "numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "joblib")
_TIME_KEY = re.compile(r"time|date|created|stamp", re.IGNORECASE)
_COLUMN_NAMES = "roles"


class ArtifactError(RuntimeError):
    """The artifact is missing, malformed or does not match its hash."""


@dataclass(frozen=True)
class FrozenArtifact:
    path: Path
    artifact_hash: str
    meta: dict[str, Any]
    predictor: ScoredPredictor

    @property
    def target_name(self) -> str | None:
        target = self.meta["roles"].get("target")
        return None if target is None else str(target)


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in _VERSIONED:
        try:
            out[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - source tree without install
            out[name] = "unknown"
    return out


def artifact_meta(
    predictor: ScoredPredictor,
    *,
    spec_hash: str,
    grid: TauGrid,
    seed: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The standard ``meta.json`` payload; ``dev_cov`` is evaluated on the fixed τ grid."""
    roles = asdict(predictor.roles)
    meta: dict[str, Any] = {
        "format": ARTIFACT_FORMAT,
        "family": predictor.family.value,
        "roles": {k: list(v) if isinstance(v, tuple) else v for k, v in roles.items()},
        "spec_hash": spec_hash,
        "grid_hash": grid.hash,
        "grid": {"size": grid.size, "low": grid.low, "high": grid.high},
        "dev_cov": [float(x) for x in predictor.dev_cov(grid.values)],
        "seed": seed,
        "q13_fit": "bag_of_oof_fold_models",
    }
    meta.update(extra or {})
    return meta


def _is_iso_datetime(text: str) -> bool:
    try:
        datetime.fromisoformat(text.strip())
    except ValueError:
        return False
    return True


def _time_stamps(value: Any, path: str = "") -> list[str]:
    """Paths in ``value`` whose key names a time or whose string value is an ISO-8601 date."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, sub in value.items():
            where = f"{path}.{key}" if path else str(key)
            if _TIME_KEY.search(str(key)):
                found.append(where)
            found.extend(_time_stamps(sub, where))
    elif isinstance(value, list | tuple):
        for i, sub in enumerate(value):
            found.extend(_time_stamps(sub, f"{path}[{i}]"))
    elif isinstance(value, str) and _is_iso_datetime(value):
        found.append(path)
    return found


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hashed_files(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*") if p.is_file() and p.relative_to(root).as_posix() != HASH_FILE
    ]


def _combine(entries: Iterable[tuple[str, str]], root: Path) -> str:
    """The artifact hash from (relative path, file sha256) pairs."""
    ordered = sorted(entries)
    if not ordered:
        raise ArtifactError(f"{root} holds no artifact files")
    payload = "".join(f"{rel}\t{digest}\n" for rel, digest in ordered)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_artifact_hash(directory: str | Path) -> str:
    """sha256 over the sorted (relative path, file sha256) list, excluding the hash file."""
    root = Path(directory)
    return _combine(
        ((p.relative_to(root).as_posix(), _file_sha256(p)) for p in _hashed_files(root)), root
    )


def freeze(predictor: ScoredPredictor, out_dir: str | Path, meta: Mapping[str, Any]) -> str:
    """Write ``model.joblib`` + ``meta.json`` to an empty ``out_dir``; return the artifact hash.

    ``meta`` must carry at least ``family``, ``roles``, ``spec_hash``, ``grid_hash`` and
    ``dev_cov`` (see :func:`artifact_meta`); library versions are added when absent. Meta
    holding a time stamp is refused (see the module docstring), so that the same fit gives
    the same hash.
    """
    root = Path(out_dir)
    if root.exists() and any(root.iterdir()):
        raise ArtifactError(f"refusing to freeze into non-empty directory {root}")
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ArtifactError(f"meta is missing {missing}")
    if meta["family"] != predictor.family.value:
        raise ArtifactError("meta family does not match the predictor")
    if meta["roles"].get("target") != predictor.roles.target:
        raise ArtifactError("meta roles do not match the predictor")
    dev_cov = np.asarray(meta["dev_cov"], dtype=np.float64)
    if dev_cov.ndim != 1 or np.any(np.diff(dev_cov) < 0):
        raise ArtifactError("dev_cov must be a non-decreasing list")
    payload = {"versions": library_versions(), **dict(meta)}
    stamps = _time_stamps({k: v for k, v in payload.items() if k != _COLUMN_NAMES})
    if stamps:
        raise ArtifactError(
            f"meta must hold no time stamp (the same fit must give the same hash): {stamps}"
        )
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, root / MODEL_FILE)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (root / META_FILE).write_text(text, encoding="utf-8")
    digest = compute_artifact_hash(root)
    (root / HASH_FILE).write_text(digest + "\n", encoding="utf-8")
    log.info("froze %s artifact at %s (%s)", predictor.family.value, root, digest)
    return digest


def load_artifact(directory: str | Path, expected_hash: str | None = None) -> FrozenArtifact:
    """Verify the artifact hash, then load it. Refuses on any mismatch.

    Every file is read into memory once; the hash is taken over those bytes and the meta and
    model are parsed from the same buffers (never re-read from disk).
    """
    root = Path(directory)
    for name in (MODEL_FILE, META_FILE, HASH_FILE):
        if not (root / name).is_file():
            raise ArtifactError(f"{root} is not an artifact: {name} is missing")
    try:
        blobs = {p.relative_to(root).as_posix(): p.read_bytes() for p in _hashed_files(root)}
    except OSError as exc:
        raise ArtifactError(f"cannot read artifact {root}: {exc}") from exc
    if MODEL_FILE not in blobs or META_FILE not in blobs:
        raise ArtifactError(f"{root} is not an artifact: {MODEL_FILE} or {META_FILE} vanished")
    digest = _combine(
        ((rel, hashlib.sha256(data).hexdigest()) for rel, data in blobs.items()), root
    )
    recorded = (root / HASH_FILE).read_text(encoding="utf-8").strip()
    if digest != recorded:
        raise ArtifactError(f"artifact files changed since freeze: {digest} != {recorded}")
    if expected_hash is not None and digest != expected_hash:
        raise ArtifactError(f"artifact hash {digest} does not match expected {expected_hash}")
    meta = json.loads(blobs[META_FILE].decode("utf-8"))
    predictor = joblib.load(io.BytesIO(blobs[MODEL_FILE]))
    if not isinstance(predictor, ScoredPredictor):
        raise ArtifactError(f"{MODEL_FILE} does not hold a ScoredPredictor")
    return FrozenArtifact(path=root, artifact_hash=digest, meta=meta, predictor=predictor)
