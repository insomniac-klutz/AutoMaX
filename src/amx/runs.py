"""Run-level orchestration behind the CLI (HANDOFF 5.1, 5.3).

These functions are the agent-visible side of a run: they read the spec and data, write the dev
fold, fit and freeze the A0 baseline, profile feasibility on dev, and render reports. Anything
that reads calibration or sealed data lives in :mod:`amx.warden`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from amx._log import get_logger
from amx.baseline.artifact import artifact_meta, freeze, load_artifact
from amx.baseline.trivial import fit_trivial
from amx.cert.certificate import Certificate
from amx.cert.grid import TauGrid
from amx.data.io import load_unitframe
from amx.data.unitframe import UnitFrame
from amx.loss.registry import build_loss
from amx.profile.audit import pre_profile, resolve_regime
from amx.profile.feasibility import Feasibility, feasibility
from amx.report.bands_md import render_bands_md
from amx.report.frontier import plot_frontier
from amx.spec.hashing import content_hash
from amx.spec.loader import load_taskspec
from amx.spec.models import TaskSpec
from amx.split.manifest import SplitManifest, read_manifest
from amx.split.oof import oof_train_mask
from amx.split.run import MANIFEST_NAME, TASKSPEC_NAME, load_dev, split_run
from amx.split.vault import LocalVault
from amx.warden.certify import CERT_RELPATH

log = get_logger(__name__)

ARTIFACT_RELPATH = Path("artifact")
PROFILE_NAME = "profile.json"
REPORT_DIR = Path("report")
SMALL_DATA = 3000


class RunError(RuntimeError):
    """A run step was called in the wrong state or without a required human flag."""


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def run_spec(run_dir: str | Path) -> TaskSpec:
    """The run's TaskSpec copy (agent-visible; the warden uses its own vault snapshot)."""
    return load_taskspec(Path(run_dir) / TASKSPEC_NAME)


def run_manifest(run_dir: str | Path) -> SplitManifest:
    return read_manifest(Path(run_dir) / MANIFEST_NAME)


def profile_pre(spec_path: str | Path) -> dict[str, Any]:
    """Label-free profile of the whole dataset, before any split (C9)."""
    spec = load_taskspec(spec_path)
    uf = load_unitframe(spec, spec_path)
    pre = pre_profile(uf, spec)
    return {"pre": pre.model_dump(mode="json"), "resolved_regime": resolve_regime(spec, pre).value}


def split(
    spec_path: str | Path,
    run_dir: str | Path,
    *,
    vault: LocalVault | None = None,
    run_id: str | None = None,
    allow_small: bool = False,
) -> dict[str, Any]:
    """Profile (label-free), resolve the regime and split; dev goes to ``run_dir``."""
    spec = load_taskspec(spec_path)
    uf = load_unitframe(spec, spec_path)
    if uf.n < SMALL_DATA and not allow_small:
        raise RunError(f"only {uf.n} units; small-data runs need --allow-small (HANDOFF 8.1)")
    pre = pre_profile(uf, spec)
    regime = resolve_regime(spec, pre)
    rd = Path(run_dir).resolve()
    rid = run_id or rd.name
    res = split_run(spec, uf, rd, vault or LocalVault(), rid, regime=regime)
    _write_json(rd / PROFILE_NAME, {"pre": pre.model_dump(mode="json")})
    counts = res.manifest.counts.model_dump()
    log.info("split %s: regime %s, counts %s", rid, regime.value, counts)
    return {
        "run_id": rid,
        "regime": regime.value,
        "counts": counts,
        "manifest": str(res.manifest_path),
        "reproducibility_hash": res.manifest.reproducibility_hash(),
    }


def _dev_groups(spec: TaskSpec, dev: UnitFrame, ids: NDArray[Any]) -> NDArray[Any] | None:
    col = spec.data.independence_group
    if col is None:
        return None
    by_id = dict(zip(dev.ids.tolist(), dev.column(col).tolist(), strict=True))
    return np.asarray([by_id[str(i)] for i in ids], dtype=object)


def baseline(run_dir: str | Path, *, confirm_loss: bool = False) -> dict[str, Any]:
    """Fit and freeze the A0 trivial baseline on dev (ROLLER step 9)."""
    rd = Path(run_dir).resolve()
    art = rd / ARTIFACT_RELPATH
    if art.exists() and any(art.iterdir()):
        raise RunError(f"{art} already holds an artifact; use a new run to refit")
    spec = run_spec(rd)
    manifest = run_manifest(rd)
    dev, oof = load_dev(rd)
    loss = build_loss(spec, confirmed=confirm_loss, base_dir=rd)
    regime = manifest.regime

    def mask_fn(folds: NDArray[np.int64], f: int) -> NDArray[np.bool_]:
        return oof_train_mask(dev, folds, f, spec, regime=regime)

    seed = spec.splits.seed
    predictor = fit_trivial(spec, dev, oof, mask_fn, loss, seed)
    grid = TauGrid.from_spec(spec.cert)
    meta = artifact_meta(predictor, spec_hash=content_hash(spec), grid=grid, seed=seed)
    groups = _dev_groups(spec, dev, predictor.oof_ids())
    if groups is not None:
        meta["dev_cov"] = [float(x) for x in predictor.dev_cov(grid.values, groups=groups)]
        meta["dev_cov_unit"] = "groups"
    digest = freeze(predictor, art, meta)
    return {"artifact": str(art), "artifact_hash": digest, "family": spec.task.family.value}


def profile_feasibility(run_dir: str | Path, *, confirm_loss: bool = False) -> Feasibility:
    """Dev-only feasibility pass (C9, C14), using the frozen baseline when it exists."""
    rd = Path(run_dir).resolve()
    spec = run_spec(rd)
    manifest = run_manifest(rd)
    dev, _ = load_dev(rd)
    loss = build_loss(spec, confirmed=confirm_loss, base_dir=rd)
    grid = TauGrid.from_spec(spec.cert)
    kwargs: dict[str, Any] = {}
    art = rd / ARTIFACT_RELPATH
    if (art / "meta.json").is_file():
        frozen = load_artifact(art)
        kwargs = {
            "tau": grid.values,
            "dev_cov": np.asarray(frozen.meta["dev_cov"], dtype=np.float64),
            "oof_scores": frozen.predictor.oof_scores(),
            "oof_losses": frozen.predictor.oof_losses(),
        }
    feas = feasibility(
        spec,
        regime=manifest.regime,
        dev_target=dev.target,
        n_calib=manifest.counts.calib,
        loss_fn=loss,
        loss_is_binary=loss.is_binary,
        **kwargs,
    )
    path = rd / PROFILE_NAME
    current: dict[str, Any] = {}
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
    current["feasibility"] = feas.model_dump(mode="json")
    _write_json(path, current)
    return feas


def report(run_dir: str | Path) -> dict[str, str]:
    """Render ``report/bands.md`` and ``report/frontier.png`` from the certificate JSON."""
    rd = Path(run_dir).resolve()
    cert_path = rd / CERT_RELPATH
    if not cert_path.is_file():
        raise RunError(f"no certificate at {cert_path}; the warden writes it at freeze")
    cert = Certificate.read(cert_path)
    out = rd / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    md = out / "bands.md"
    md.write_text(render_bands_md(cert), encoding="utf-8")
    png = plot_frontier(cert, out / "frontier.png")
    (out / "bands.json").write_text(cert.to_json(), encoding="utf-8")
    return {"bands_md": str(md), "frontier_png": str(png), "bands_json": str(out / "bands.json")}
