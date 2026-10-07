"""``amx certify --freeze``: one certify call on the calibration fold (HANDOFF 7.2, 8.4).

Order matters: the token is checked and the certify budget is spent BEFORE any calibration
label is read, so an aborted call still counts (7.11, Q1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import amx
from amx._log import get_logger
from amx.cert.budget import DeltaBudget
from amx.cert.certificate import Certificate, build_certificate
from amx.cert.guarantee import guarantee_type_for
from amx.cert.ltt import bonferroni_diagnostic, fixed_sequence_ltt, grid_stats
from amx.cert.slices import slice_risks
from amx.data.unitframe import UnitFrame
from amx.loss.registry import build_loss
from amx.split.counter import CertifyCounter
from amx.split.vault import LocalVault
from amx.warden.common import FrozenRun, frozen_run
from amx.warden.runner import run_resolver
from amx.warden.token import verify_token

log = get_logger(__name__)

CERT_RELPATH = Path("cert") / "certificate.json"


def _slice_labels(units: UnitFrame, by: str) -> np.ndarray:
    return units.target if by == "target" else units.column(by)


def population_assumptions(run: FrozenRun) -> list[str]:
    out: list[str] = []
    dropped = run.manifest.dropped_reasons
    if dropped:
        out.append(
            "the certified population excludes units dropped at split: "
            + ", ".join(f"{getattr(k, 'value', k)}={v}" for k, v in sorted(dropped.items()))
        )
    out.append(
        "exact-duplicate input clusters were kept within one fold, so the certified population "
        "excludes exact duplicates of dev units (Q7)"
    )
    return out


def certify_run(
    run_dir: str | Path,
    *,
    token: str | None,
    vault: LocalVault | None = None,
    run_id: str | None = None,
    confirm_loss: bool = False,
) -> Certificate:
    v = vault or LocalVault()
    rid = run_id or Path(run_dir).resolve().name
    verify_token(v, rid, token)
    run = frozen_run(run_dir, v, rid)
    spec = run.spec
    loss = build_loss(spec, confirmed=confirm_loss)
    counter = CertifyCounter(v, rid, spec.splits.max_certify_calls)
    call = counter.acquire("certify")
    log.info("certify call %d of %d for run %s", call, spec.splits.max_certify_calls, rid)

    calib = v.read_fold(rid, "calib")
    out = run_resolver(
        run.artifact_dir,
        calib.without_target(),
        v.run_path(rid) / "tmp",
        expected_hash=run.artifact_hash,
    )
    values = out["value"].to_numpy()
    scores = out["score"].to_numpy(dtype=np.float64)
    losses = loss(values, calib.target)
    group_col = spec.data.independence_group
    groups = calib.column(group_col) if group_col else None
    stats = grid_stats(scores, losses, run.grid.values, binary=loss.is_binary, groups=groups)
    budget = DeltaBudget(spec.bands.delta, spec.bands.m, spec.cert.call_policy)
    result = fixed_sequence_ltt(
        stats,
        spec.bands.alphas,
        budget.delta_per_band,
        run.dev_cov,
        start_factor=spec.cert.start_factor,
    )
    slices = []
    for band in result.bands:
        tau = None if band.index is None else float(run.grid.values[band.index])
        committed = scores <= tau if tau is not None else np.zeros(scores.shape, bool)
        per = []
        for ws in spec.bands.watch_slices:
            per.extend(
                slice_risks(
                    ws.name,
                    _slice_labels(calib, ws.by),
                    losses,
                    committed,
                    alpha=band.alpha,
                    binary=loss.is_binary,
                )
            )
        slices.append(per)
    regime = run.manifest.regime
    cert = build_certificate(
        result,
        policies=spec.bands.policies,
        budget=budget,
        grid=run.grid,
        regime=regime,
        guarantee_type=guarantee_type_for(regime, spec.task.family),
        run_id=rid,
        taskspec_hash=run.spec_hash,
        artifact_hash=run.artifact_hash,
        amx_version=amx.__version__,
        group_column=group_col,
        extra_assumptions=population_assumptions(run),
        certify_calls_used=counter.used(),
        certify_calls_max=spec.splits.max_certify_calls,
        slices=slices,
        diagnostics={
            "bonferroni_grid": bonferroni_diagnostic(
                stats, spec.bands.alphas, budget.delta_per_band
            ),
            "loss": {"name": loss.name, "binary": loss.is_binary, "confirmed": confirm_loss},
        },
    )
    path = cert.write(run.run_dir / CERT_RELPATH)
    log.info("wrote %s", path)
    return cert
