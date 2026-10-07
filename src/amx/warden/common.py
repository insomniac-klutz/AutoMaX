"""Shared warden plumbing: the run's frozen inputs, read from warden-owned copies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from amx.baseline.artifact import HASH_FILE, META_FILE, compute_artifact_hash
from amx.cert.grid import TauGrid
from amx.spec.hashing import content_hash
from amx.spec.loader import parse_taskspec
from amx.spec.models import TaskSpec
from amx.split.manifest import SplitManifest
from amx.split.run import MANIFEST_NAME, TASKSPEC_NAME
from amx.split.vault import LocalVault

ARTIFACT_DIR = "artifact"


class WardenError(RuntimeError):
    """The run is not in a state the warden can certify."""


@dataclass(frozen=True)
class FrozenRun:
    run_dir: Path
    run_id: str
    spec: TaskSpec
    spec_hash: str
    manifest: SplitManifest
    artifact_dir: Path
    artifact_hash: str
    meta: dict[str, Any]
    grid: TauGrid

    @property
    def dev_cov(self) -> NDArray[np.float64]:
        return np.asarray(self.meta["dev_cov"], dtype=np.float64)


def frozen_run(run_dir: str | Path, vault: LocalVault, run_id: str | None = None) -> FrozenRun:
    """Load the spec and manifest from the VAULT snapshot, never from the agent-writable run dir.

    The artifact is identified by its hash; its model is never unpickled in the warden process
    (it runs in a subprocess, see runner.py).
    """
    rd = Path(run_dir).resolve()
    rid = run_id or rd.name
    if not vault.has_fold(rid, "calib"):
        raise WardenError(f"run '{rid}' has no calibration fold in the vault; run `amx split`")
    spec = parse_taskspec(yaml.safe_load(vault.read_text(rid, TASKSPEC_NAME)))
    manifest = SplitManifest.model_validate_json(vault.read_text(rid, MANIFEST_NAME))
    art = rd / ARTIFACT_DIR
    if not (art / META_FILE).is_file():
        raise WardenError(f"no frozen artifact in {art}; run `amx baseline --trivial` first")
    digest = compute_artifact_hash(art)
    recorded = (art / HASH_FILE).read_text(encoding="utf-8").strip()
    if digest != recorded:
        raise WardenError(f"artifact changed after freeze ({digest} != {recorded})")
    meta = json.loads((art / META_FILE).read_text(encoding="utf-8"))
    spec_hash = content_hash(spec)
    if meta.get("spec_hash") != spec_hash:
        raise WardenError(
            "the artifact was fitted for a different TaskSpec than the vault snapshot"
        )
    grid = TauGrid.from_spec(spec.cert)
    if meta.get("grid_hash") != grid.hash or len(meta.get("dev_cov", [])) != grid.size:
        raise WardenError("the artifact's dev coverage curve is not on this run's τ grid")
    return FrozenRun(rd, rid, spec, spec_hash, manifest, art, digest, meta, grid)
