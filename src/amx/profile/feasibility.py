"""Dev-only feasibility pass, after the split (HANDOFF 7.4, 8.2, C9, C14).

Inputs are dev-side only: dev labels, dev OOF losses and scores of the baseline, and the count
of calibration units from the split manifest. The calibration fold itself is never read here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict

from amx.cert.bounds import cp_lower, cp_upper, hb_lower, hb_upper
from amx.cert.budget import DeltaBudget
from amx.cert.guarantee import GuaranteeType, guarantee_type_for
from amx.cert.ltt import start_index
from amx.cert.nmin import n_min
from amx.spec.enums import Family, Regime, TargetKind
from amx.spec.models import TaskSpec

CAP_SHARE_WARN = 0.01
IMBALANCE_WARN = 0.05
FLOOR_BIN = 200  # units in the baseline's most confident dev OOF bin


class BandFeasibility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float
    delta_j: float
    n_min: int
    necessary_ok: bool
    feasible_on_dev: bool | None
    start_tau: float | None
    expected_committed_at_start: float | None
    trivially_met_by_constant: bool
    below_baseline_floor: bool
    notes: list[str]


class Feasibility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regime: Regime
    guarantee_type: GuaranteeType
    n_dev: int
    n_calib: int
    target_summary: dict[str, float | int | str]
    constant_risk: float | None
    constant_risk_upper95: float | None
    baseline_floor_estimate: float | None
    baseline_floor_lower95: float | None
    independence_unit: str
    bands: list[BandFeasibility]
    requires_force: bool
    requires_allow_small: bool
    recommendations: list[str]
    warnings: list[str]


def _target_summary(
    y: NDArray[np.generic], kind: TargetKind
) -> tuple[dict[str, float | int | str], list[str]]:
    warnings: list[str] = []
    if kind is TargetKind.CATEGORICAL:
        vals, counts = np.unique(np.asarray(y, dtype=object).astype(str), return_counts=True)
        share = counts / counts.sum()
        lo, hi = float(np.min(share)), float(np.max(share))
        summary: dict[str, float | int | str] = {
            "n_classes": int(vals.size),
            "majority_share": hi,
            "minority_share": lo,
        }
        if lo < IMBALANCE_WARN:
            warnings.append(f"extreme imbalance: smallest class share {lo:.4f}")
        return summary, warnings
    v = np.asarray(y, dtype=np.float64)
    top = float(np.max(v))
    at_top = float(np.mean(v == top))
    summary = {
        "min": float(np.min(v)),
        "median": float(np.median(v)),
        "max": top,
        "share_at_max": at_top,
    }
    if at_top >= CAP_SHARE_WARN:
        warnings.append(
            f"{100 * at_top:.1f}% of dev targets sit exactly at the maximum {top:g}: "
            "likely a cap (censoring); risk is measured against capped gold"
        )
    return summary, warnings


def _constant_losses(
    y: NDArray[np.generic], spec: TaskSpec, loss_fn: object
) -> NDArray[np.float64] | None:
    """Losses of the best constant answer on dev (majority class, or the median)."""
    if not callable(loss_fn):
        return None
    kind = spec.data.target.kind
    if kind is TargetKind.CATEGORICAL:
        vals, counts = np.unique(np.asarray(y, dtype=object), return_counts=True)
        const = np.full(len(y), vals[int(np.argmax(counts))], dtype=object)
    elif kind in (TargetKind.NUMERIC, TargetKind.SERIES):
        const = np.full(len(y), float(np.median(np.asarray(y, dtype=np.float64))))
    else:
        return None
    return np.asarray(loss_fn(const, y), dtype=np.float64)


def feasibility(
    spec: TaskSpec,
    *,
    regime: Regime,
    dev_target: ArrayLike,
    n_calib: int,
    loss_fn: object,
    loss_is_binary: bool,
    tau: ArrayLike | None = None,
    dev_cov: ArrayLike | None = None,
    oof_scores: ArrayLike | None = None,
    oof_losses: ArrayLike | None = None,
) -> Feasibility:
    """n_min check, dev-estimated start points, constant-predictor and baseline-floor flags.

    ``n_calib`` counts independence units: calibration groups when the spec certifies at group
    level (``independence_unit: group:<col>``), units otherwise (Q2).
    """
    y = np.asarray(dev_target)
    summary, warnings = _target_summary(y, spec.data.target.kind)
    budget = DeltaBudget(spec.bands.delta, spec.bands.m, spec.cert.call_policy)
    dj = budget.delta_per_band
    gtype = guarantee_type_for(regime, spec.task.family)
    recs: list[str] = []

    const = _constant_losses(y, spec, loss_fn)
    c_risk = c_up = None
    if const is not None and const.size:
        c_risk = float(const.mean())
        s = float(const.sum())
        c_up = cp_upper(round(s), const.size) if loss_is_binary else hb_upper(s, const.size)

    floor = floor_lo = None
    if oof_scores is not None and oof_losses is not None:
        sc = np.asarray(oof_scores, dtype=np.float64)
        ls = np.asarray(oof_losses, dtype=np.float64)
        ok = np.isfinite(sc)
        sc, ls = sc[ok], ls[ok]
        if ls.size >= FLOOR_BIN:
            order = np.argsort(sc, kind="stable")[:FLOOR_BIN]
            s = float(ls[order].sum())
            floor = s / FLOOR_BIN
            floor_lo = cp_lower(round(s), FLOOR_BIN) if loss_is_binary else hb_lower(s, FLOOR_BIN)

    t = None if tau is None else np.asarray(tau, dtype=np.float64)
    cov = None if dev_cov is None else np.asarray(dev_cov, dtype=np.float64)
    bands: list[BandFeasibility] = []
    force = False
    for alpha in spec.bands.alphas:
        need = n_min(alpha, dj)
        notes: list[str] = []
        necessary = n_calib >= need
        if not necessary:
            notes.append(f"calib has {n_calib} independence units, fewer than n_min={need}")
        feas: bool | None = None
        start_tau = expected = None
        if t is not None and cov is not None:
            idx = start_index(cov, n_calib, need, spec.cert.start_factor)
            feas = idx is not None
            if idx is not None:
                start_tau = float(t[idx])
                expected = float(cov[idx] * n_calib)
            else:
                notes.append("no grid point reaches 1.25 x n_min committed units on dev estimates")
        trivial = c_up is not None and c_up <= alpha
        if trivial:
            notes.append("a constant predictor already meets this band at full coverage")
        below = floor_lo is not None and floor_lo > alpha
        if below:
            notes.append(
                "the baseline's most confident dev OOF units already exceed alpha (a limit of "
                "this baseline, not necessarily label noise)"
            )
        if not necessary or feas is False:
            force = True
        bands.append(
            BandFeasibility(
                alpha=alpha,
                delta_j=dj,
                n_min=need,
                necessary_ok=necessary,
                feasible_on_dev=feas,
                start_tau=start_tau,
                expected_committed_at_start=expected,
                trivially_met_by_constant=trivial,
                below_baseline_floor=below,
                notes=notes,
            )
        )
    if force:
        recs.append(
            "some bands look infeasible: enlarge the calib fraction, loosen the bands, or use "
            "fewer bands (a smaller m raises delta_j); proceeding needs --force"
        )
    if spec.task.family is Family.FORECASTING:
        recs.append("forecasting selective risk is reported as holdout_empirical (D15)")
    n_dev = int(y.shape[0])
    return Feasibility(
        regime=regime,
        guarantee_type=gtype,
        n_dev=n_dev,
        n_calib=n_calib,
        target_summary=summary,
        constant_risk=c_risk,
        constant_risk_upper95=c_up,
        baseline_floor_estimate=floor,
        baseline_floor_lower95=floor_lo,
        independence_unit="groups" if spec.data.independence_group else "units",
        bands=bands,
        requires_force=force,
        requires_allow_small=(n_dev + n_calib) < 3000,
        recommendations=recs,
        warnings=warnings,
    )
