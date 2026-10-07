"""Shared warden plumbing: the run's frozen inputs, read from warden-owned copies."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from amx.baseline.artifact import HASH_FILE, META_FILE, compute_artifact_hash
from amx.cert.grid import TauGrid
from amx.cert.guarantee import GuaranteeType, effective_guarantee_type, guarantee_type_for
from amx.data.unitframe import UnitFrame
from amx.loss.base import Loss
from amx.loss.custom import load_custom_loss
from amx.loss.registry import build_loss
from amx.spec.enums import LossKind, Regime
from amx.spec.hashing import content_hash
from amx.spec.loader import parse_taskspec
from amx.spec.models import TaskSpec
from amx.split.manifest import SplitManifest, ids_hmac
from amx.split.run import MANIFEST_NAME, TASKSPEC_NAME
from amx.split.vault import LocalVault

ARTIFACT_DIR = "artifact"
PINNED_ARTIFACT = "pinned_artifact"
CERTIFIED_ARTIFACT = "certified_artifact.sha256"
PRE_PROFILE_NAME = "profile_pre.json"
CUSTOM_LOSS_NAME = "custom_loss.py"
CUSTOM_LOSS_HASH = "custom_loss.sha256"
LOCAL_DIR_ASSUMPTION = (
    "vault mode local_dir gives no isolation: validity assumes that neither the agent nor the "
    "artifact could read calibration labels (HANDOFF 8.4)"
)


class WardenError(RuntimeError):
    """The run is not in a state the warden can certify."""


@dataclass
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
    pinned: bool
    _staging: Path | None = field(default=None, repr=False)

    @property
    def dev_cov(self) -> NDArray[np.float64]:
        return np.asarray(self.meta["dev_cov"], dtype=np.float64)

    @property
    def guarantee_type(self) -> GuaranteeType:
        """7.1: the audit, not the user, picks the certifier (see effective_guarantee_type)."""
        return effective_guarantee_type(
            self.manifest.regime,
            self.spec.task.family,
            time_declared=self.spec.data.time_column is not None,
        )

    def regime_warnings(self) -> list[str]:
        declared = guarantee_type_for(self.manifest.regime, self.spec.task.family)
        if self.guarantee_type is declared:
            return []
        return [
            f"a time column is declared but the regime is '{self.manifest.regime.value}', "
            "which differs from the audit's recommendation 'temporal'",
            f"guarantee downgraded to '{self.guarantee_type.value}' (HANDOFF 7.1, 6.5)",
        ]

    def pin(self, vault: LocalVault) -> None:
        """Pin the staged artifact into the vault exactly once (at the certify call)."""
        if self.pinned:
            return
        assert self._staging is not None
        dest = vault.run_path(self.run_id) / PINNED_ARTIFACT
        if dest.exists():
            raise WardenError("an artifact is already pinned for this run")
        shutil.copytree(self.artifact_dir, dest)
        if compute_artifact_hash(dest) != self.artifact_hash:
            raise WardenError("the pinned copy does not match the checked artifact")
        vault.write_text(self.run_id, CERTIFIED_ARTIFACT, self.artifact_hash, overwrite=False)
        self.artifact_dir, self.pinned = dest, True

    def cleanup(self) -> None:
        if self._staging is not None:
            shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = None


def _stage(src: Path) -> tuple[Path, Path, str]:
    """Copy the run-dir artifact to a private temporary directory OUTSIDE the vault and hash the
    copy; its recorded hash must match (accidental edits after freeze are refused)."""
    staging = Path(tempfile.mkdtemp(prefix="amx-stage-"))
    os.chmod(staging, 0o700)
    dest = staging / ARTIFACT_DIR
    shutil.copytree(src, dest)
    digest = compute_artifact_hash(dest)
    recorded_path = dest / HASH_FILE
    recorded = recorded_path.read_text(encoding="utf-8").strip() if recorded_path.is_file() else ""
    if digest != recorded:
        shutil.rmtree(staging, ignore_errors=True)
        raise WardenError(f"artifact changed after freeze ({digest} != {recorded or 'missing'})")
    return staging, dest, digest


def frozen_run(
    run_dir: str | Path,
    vault: LocalVault,
    run_id: str | None = None,
    *,
    use_pin: bool = False,
) -> FrozenRun:
    """Load the spec and manifest from the VAULT snapshot, never from the agent-writable run dir.

    Before the certify call (``use_pin=False``) the run-dir artifact is staged outside the vault
    and hashed; :meth:`FrozenRun.pin` moves it into the vault when the call is spent. After it
    (``use_pin=True``) only the pinned copy is used, and a run-dir artifact that differs from
    the certified one is refused. Models are never unpickled in the warden process.
    """
    rd = Path(run_dir).resolve()
    rid = run_id or rd.name
    if not vault.has_fold(rid, "calib"):
        raise WardenError(f"run '{rid}' has no calibration fold in the vault; run `amx split`")
    spec = parse_taskspec(yaml.safe_load(vault.read_text(rid, TASKSPEC_NAME)))
    manifest = SplitManifest.model_validate_json(vault.read_text(rid, MANIFEST_NAME))
    if manifest.regime is Regime.GROUPED and spec.data.independence_group is None:
        raise WardenError("regime 'grouped' needs a group independence unit (OQ Q2)")
    art = rd / ARTIFACT_DIR
    staging: Path | None = None
    if use_pin:
        pinned_dir = vault.run_path(rid) / PINNED_ARTIFACT
        if not pinned_dir.is_dir():
            raise WardenError("no certified artifact is pinned for this run; certify first")
        digest = compute_artifact_hash(pinned_dir)
        if digest != vault.read_text(rid, CERTIFIED_ARTIFACT).strip():
            raise WardenError("the pinned artifact changed after certification")
        if (art / META_FILE).is_file() and compute_artifact_hash(art) != digest:
            raise WardenError("the run-dir artifact differs from the certified one")
        art_dir, pinned = pinned_dir, True
    else:
        if not (art / META_FILE).is_file():
            raise WardenError(f"no frozen artifact in {art}; run `amx baseline` first")
        staging, art_dir, digest = _stage(art)
        pinned = False
    meta = json.loads((art_dir / META_FILE).read_text(encoding="utf-8"))
    spec_hash = content_hash(spec)
    grid = TauGrid.from_spec(spec.cert)
    problem = None
    if meta.get("spec_hash") != spec_hash:
        problem = "the artifact was fitted for a different TaskSpec than the vault snapshot"
    elif meta.get("grid_hash") != grid.hash or len(meta.get("dev_cov", [])) != grid.size:
        problem = "the artifact's dev coverage curve is not on this run's τ grid"
    if problem is not None:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise WardenError(problem)
    return FrozenRun(
        rd, rid, spec, spec_hash, manifest, art_dir, digest, meta, grid, pinned, staging
    )


def load_calib(vault: LocalVault, run: FrozenRun) -> UnitFrame:
    """The calibration fold, checked against the manifest's keyed digest (C13)."""
    calib = vault.read_fold(run.run_id, "calib")
    digest = ids_hmac(calib.ids.tolist(), vault.hmac_key(run.run_id))
    if not hmac.compare_digest(digest, run.manifest.calib_digest):
        raise WardenError("the calibration fold does not match the split manifest's digest")
    return calib


def warden_loss(
    vault: LocalVault, run: FrozenRun, *, confirm_loss: bool
) -> tuple[Loss, str | None]:
    """The run's loss; a custom loss is loaded ONLY from its vault snapshot, hash-checked."""
    spec = run.spec
    if spec.task.loss.kind is not LossKind.CUSTOM:
        return build_loss(spec, confirmed=confirm_loss), None
    src = vault.file_path(run.run_id, CUSTOM_LOSS_NAME)
    if not src.is_file():
        raise WardenError("the custom loss was not snapshotted into the vault at split")
    digest = "sha256:" + hashlib.sha256(src.read_bytes()).hexdigest()
    if digest != vault.read_text(run.run_id, CUSTOM_LOSS_HASH).strip():
        raise WardenError("the vault copy of the custom loss changed after split")
    assert spec.task.loss.fn is not None
    loss = load_custom_loss(
        src, spec.task.loss.fn, confirmed=confirm_loss, params=dict(spec.task.loss.params)
    )
    return loss, digest
