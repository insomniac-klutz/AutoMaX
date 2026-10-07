"""Shared warden plumbing: the run's frozen inputs, read from warden-owned copies."""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from amx.baseline.artifact import HASH_FILE, META_FILE, compute_artifact_hash
from amx.cert.grid import TauGrid
from amx.cert.guarantee import GuaranteeType, guarantee_type_for
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
PRE_PROFILE_NAME = "profile_pre.json"
CUSTOM_LOSS_NAME = "custom_loss.py"
CUSTOM_LOSS_HASH = "custom_loss.sha256"
LOCAL_DIR_ASSUMPTION = (
    "vault mode local_dir gives no isolation: validity assumes that neither the agent nor the "
    "artifact could read calibration labels (HANDOFF 8.4)"
)


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
    recommended_regime: Regime | None

    @property
    def dev_cov(self) -> NDArray[np.float64]:
        return np.asarray(self.meta["dev_cov"], dtype=np.float64)

    @property
    def guarantee_type(self) -> GuaranteeType:
        """7.1: the profiler, not the user, picks the certifier; a time structure the audit found
        downgrades a declared exchangeable regime to ``holdout_empirical``."""
        gtype = guarantee_type_for(self.manifest.regime, self.spec.task.family)
        if (
            self.recommended_regime is Regime.TEMPORAL
            and self.manifest.regime is not Regime.TEMPORAL
        ):
            return GuaranteeType.HOLDOUT_EMPIRICAL
        return gtype

    def regime_warnings(self) -> list[str]:
        rec = self.recommended_regime
        if rec is None or rec is self.manifest.regime:
            return []
        out = [
            f"declared regime '{self.manifest.regime.value}' differs from the audit's "
            f"recommendation '{rec.value}'"
        ]
        if self.guarantee_type is not guarantee_type_for(
            self.manifest.regime, self.spec.task.family
        ):
            out.append(f"guarantee downgraded to '{self.guarantee_type.value}' (HANDOFF 7.1, 6.5)")
        return out


def _pin_artifact(src: Path, vault: LocalVault, run_id: str) -> tuple[Path, str]:
    """Copy the artifact into the vault and hash the COPY; the warden only uses the copy."""
    dest = vault.run_path(run_id) / PINNED_ARTIFACT
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest, compute_artifact_hash(dest)


def frozen_run(run_dir: str | Path, vault: LocalVault, run_id: str | None = None) -> FrozenRun:
    """Load the spec and manifest from the VAULT snapshot, never from the agent-writable run dir.

    The artifact is copied into the vault and identified by the hash of that copy; its model is
    never unpickled in the warden process (it runs in a subprocess, see runner.py).
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
    if not (art / META_FILE).is_file():
        raise WardenError(f"no frozen artifact in {art}; run `amx baseline` first")
    pinned, digest = _pin_artifact(art, vault, rid)
    recorded_path = pinned / HASH_FILE
    recorded = recorded_path.read_text(encoding="utf-8").strip() if recorded_path.is_file() else ""
    if digest != recorded:
        raise WardenError(f"artifact changed after freeze ({digest} != {recorded or 'missing'})")
    meta = json.loads((pinned / META_FILE).read_text(encoding="utf-8"))
    spec_hash = content_hash(spec)
    if meta.get("spec_hash") != spec_hash:
        raise WardenError(
            "the artifact was fitted for a different TaskSpec than the vault snapshot"
        )
    grid = TauGrid.from_spec(spec.cert)
    if meta.get("grid_hash") != grid.hash or len(meta.get("dev_cov", [])) != grid.size:
        raise WardenError("the artifact's dev coverage curve is not on this run's τ grid")
    rec: Regime | None = None
    try:
        pre = json.loads(vault.read_text(rid, PRE_PROFILE_NAME))
        rec = Regime(pre["recommended_regime"])
    except Exception:
        rec = None
    return FrozenRun(rd, rid, spec, spec_hash, manifest, pinned, digest, meta, grid, rec)


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
