"""``amx certify --freeze``: one certify call on the calibration fold (HANDOFF 7.2, 8.4).

Order matters. Everything that can refuse the call (token, artifact pin, calibration digest,
feasibility gate, slice columns, loss confirmation) runs first. Then the partition ledger is
claimed and the run's certify budget is spent, and only after that are calibration labels
turned into losses (7.11, Q1). An aborted call after that point still counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import amx
from amx._log import get_logger
from amx.cert.budget import DeltaBudget
from amx.cert.certificate import Certificate, build_certificate
from amx.cert.ltt import bonferroni_diagnostic, fixed_sequence_ltt, grid_stats, start_index
from amx.cert.nmin import n_min
from amx.cert.slices import slice_labels, slice_risks
from amx.data.unitframe import UnitFrame
from amx.split.counter import CertifyCounter
from amx.split.vault import LocalVault
from amx.warden.common import (
    LOCAL_DIR_ASSUMPTION,
    FrozenRun,
    WardenError,
    frozen_run,
    load_calib,
    warden_loss,
)
from amx.warden.ledger import claim_partition, partition_key
from amx.warden.runner import run_resolver
from amx.warden.token import verify_token

log = get_logger(__name__)

CERT_RELPATH = Path("cert") / "certificate.json"


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


def n_independent(run: FrozenRun, calib: UnitFrame) -> int:
    col = run.spec.data.independence_group
    if col is None:
        return calib.n
    return len(set(np.asarray(calib.column(col), dtype=object).astype(str).tolist()))


def infeasible_bands(run: FrozenRun, n_ref: int, delta_j: float) -> list[str]:
    """Bands the dev-side estimates say cannot certify (7.4): n_min shortfall or no start."""
    out = []
    for alpha in run.spec.bands.alphas:
        need = n_min(alpha, delta_j)
        if n_ref < need:
            out.append(f"alpha={alpha:g}: calib has {n_ref} independence units < n_min={need}")
        elif start_index(run.dev_cov, n_ref, need, run.spec.cert.start_factor) is None:
            out.append(f"alpha={alpha:g}: no τ reaches 1.25 x n_min committed units on dev")
    return out


def slice_columns(run: FrozenRun, calib: UnitFrame) -> dict[str, NDArray[Any]]:
    """Resolve every watch slice BEFORE the budget is spent; aggregate-only labels."""
    out: dict[str, NDArray[Any]] = {}
    cols = set(calib.table.column_names)
    for ws in run.spec.bands.watch_slices:
        if ws.by == "target":
            raw = calib.target
        elif ws.by in cols:
            raw = calib.column(ws.by)
        else:
            raise WardenError(f"watch slice '{ws.name}' names unknown column '{ws.by}'")
        out[ws.name] = slice_labels(raw)
    return out


def certify_run(
    run_dir: str | Path,
    *,
    token: str | None,
    vault: LocalVault | None = None,
    run_id: str | None = None,
    confirm_loss: bool = False,
    force: bool = False,
) -> Certificate:
    v = vault or LocalVault()
    rid = run_id or Path(run_dir).resolve().name
    verify_token(v, rid, token)
    run = frozen_run(run_dir, v, rid)
    spec = run.spec
    loss, loss_digest = warden_loss(v, run, confirm_loss=confirm_loss)
    calib = load_calib(v, run)
    budget = DeltaBudget(spec.bands.delta, spec.bands.m, spec.cert.call_policy)
    n_ref = n_independent(run, calib)
    blocked = infeasible_bands(run, n_ref, budget.delta_per_band)
    if blocked and not force:
        raise WardenError(
            "the dev-side feasibility check says these bands cannot certify; the single "
            "certify call would be spent for nothing. Re-run with --force to proceed anyway: "
            + "; ".join(blocked)
        )
    slices_by = slice_columns(run, calib)

    key = partition_key(run.manifest.data_hash, calib.ids.tolist())
    counter = CertifyCounter(v, rid, spec.splits.max_certify_calls)
    if counter.remaining() == 0:
        counter.acquire("certify")  # raises CertifyBudgetExhausted with the counter's message
    claim_partition(v, key, rid)
    call = counter.acquire("certify")
    log.info("certify call %d of %d for run %s", call, spec.splits.max_certify_calls, rid)

    out = run_resolver(run.artifact_dir, calib.without_target(), expected_hash=run.artifact_hash)
    values = out["value"].to_numpy()
    scores = out["score"].to_numpy(dtype=np.float64)
    losses = loss(values, calib.target)
    group_col = spec.data.independence_group
    groups = calib.column(group_col) if group_col else None
    stats = grid_stats(scores, losses, run.grid.values, binary=loss.is_binary, groups=groups)
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
        for name, labels in slices_by.items():
            per.extend(
                slice_risks(
                    name,
                    labels,
                    losses,
                    committed,
                    alpha=band.alpha,
                    binary=loss.is_binary,
                    descriptive=group_col is not None,
                )
            )
        slices.append(per)
    assumptions = [*population_assumptions(run), LOCAL_DIR_ASSUMPTION]
    if loss_digest is not None:
        assumptions.append(f"custom loss source {loss_digest}, confirmed by a human")
    cert = build_certificate(
        result,
        policies=spec.bands.policies,
        budget=budget,
        grid=run.grid,
        regime=run.manifest.regime,
        guarantee_type=run.guarantee_type,
        run_id=rid,
        taskspec_hash=run.spec_hash,
        artifact_hash=run.artifact_hash,
        amx_version=amx.__version__,
        group_column=group_col,
        extra_assumptions=assumptions,
        certify_calls_used=counter.used(),
        certify_calls_max=spec.splits.max_certify_calls,
        slices=slices,
        diagnostics={
            "bonferroni_grid": bonferroni_diagnostic(
                stats, spec.bands.alphas, budget.delta_per_band
            ),
            "loss": {"name": loss.name, "binary": loss.is_binary, "confirmed": confirm_loss},
            "forced_bands": blocked if force else [],
            "calib_partition": key,
        },
    )
    extra_warnings = [*run.regime_warnings(), *(f"forced: {b}" for b in blocked if force)]
    if extra_warnings:
        cert = cert.model_copy(update={"warnings": [*extra_warnings, *cert.warnings]})
    path = cert.write(run.run_dir / CERT_RELPATH)
    log.info("wrote %s", path)
    return cert
