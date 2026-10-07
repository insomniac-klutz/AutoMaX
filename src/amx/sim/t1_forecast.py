"""T1-forecast on generator (c): ACI interval evaluation and empirical selective risk.

HANDOFF 12 T1-forecast as amended by C12, OQ Q6 (decision D18) and decision D20:

* **Why a stationary gate (D20).** The deterministic ACI bound is one-sided in a way that
  matters: with α_t ≥ −γh, mean(err) − α ≤ (α_0 + γh)/(γ(T − h)) (``bound_over``), about 0.007
  at h = 1 and 0.015 at h = 24 for α_0 = 0.1, γ = 0.005, T = 3000. A regime shift that only
  raises the noise pushes miscoverage UP, onto that side, so on the shifted path the ±0.02
  check cannot fail. Only the under side, (1 − α_0 + γh)/(γ(T − h)) (``bound_under``,
  ≈ 0.06), leaves room for a failure.
* **Gate (stationary path).** The stationary variant of generator (c) (no shift), seed
  :data:`T1_FORECAST_STATIONARY_SEED`. Per horizon h, split-conformal absolute-residual scores
  from a calibration window, then ACI with delayed feedback (γ = 0.005, α_target = 0.1) over at
  least 2000 evaluation steps. Pass iff, for every horizon, |miscoverage − α_target| ≤ 0.02
  AND the infinite + empty interval share ≤ 0.01 (non-degenerate intervals: ACI holds the
  long-run rate even when almost every interval is infinite). The ±0.02 is an EMPIRICAL
  target (``empirical_target``): the two-sided bound needs about 9,050 steps for it.
* **Reported, not gated (shifted path).** Generator (c) with its regime shift, seed
  :data:`T1_FORECAST_SEED`: miscoverage, infinite and empty shares, the 300-step
  local-coverage range and ``implied_by_bound``.
* **implied_by_bound.** Per horizon, True when the observed deviation lies on a side whose
  one-sided bound is already ≤ the tolerance, i.e. the ±tolerance check could not have failed
  there. The bounds cover the first T − h errors; the reported miscoverage averages all T and
  differs by at most h/T, so the flag errs towards calling a pass uninformative.
* **Selective risk.** One commit mechanism (Q6): a label-free scorer
  (:meth:`SeasonalARShift.exceed_scores`), the 200-point τ grid and fixed-sequence LTT on the
  calibration window (band α = 0.02, δ = 0.1, loss 1[|e| > 2.0]), with ``dev_cov`` from an
  earlier dev window. On the evaluation window the selective risk is reported with a
  moving-block bootstrap CI (block 48, B = 1000) and the label ``holdout_empirical`` (7.1:
  valid only if calibration-to-future is stationary), split into pre- and post-shift parts on
  the shifted path, where no guarantee applies (7.5, 7.6). It is reported, never gated.

Layout of one series (time indices): burn-in, dev origins, calibration origins, an embargo
of ``horizon_max``, evaluation origins (``rolling_origins`` with stride 1). On the shifted
path the noise sd doubles ``shift_offset`` steps after the first evaluation target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from amx._log import get_logger
from amx.cert import (
    DeltaBudget,
    TauGrid,
    aci_run,
    block_bootstrap_ratio,
    fixed_sequence_ltt,
    grid_stats,
    rolling_origins,
    selective_risk,
)
from amx.cert.aci import ACIResult
from amx.cert.guarantee import GuaranteeType
from amx.sim.generators import SeasonalARShift

T1_FORECAST_SEED = 140_003
"""Pre-registered seed of the SHIFTED T1-forecast path (reported, not gated; D20)."""

T1_FORECAST_STATIONARY_SEED = 140_013
"""Seed of the STATIONARY T1-forecast path (the gate, D20), fixed in code before its first run."""

T1_FORECAST_HORIZONS: tuple[int, ...] = (1, 24)
T1_FORECAST_GAMMA = 0.005
T1_FORECAST_ALPHA = 0.1
T1_FORECAST_TOLERANCE = 0.02
"""Empirical ±0.02 target on long-run miscoverage (Q6: not implied by the two-sided bound)."""

T1_FORECAST_MAX_DEGENERATE = 0.01
"""Gate ceiling on the infinite + empty interval share per horizon (D20)."""

MIN_EVAL_STEPS = 2000

# The selective commit band below (alpha 0.02, tol 2.0) was chosen AFTER looking at the
# shifted path at seed 140_003, so it is not pre-registered. It only feeds the reported
# holdout_empirical selective risk (never gated) and is to be logged as a decision.
T1_FORECAST_SEL_TOL = 2.0
"""Tolerance of the selective loss 1[|e| > tol] (fixed; pre-shift unconditional rate ≈ 0.024)."""

T1_FORECAST_SEL_ALPHA = 0.02
"""Selective risk band of the forecasting commit; LTT on the calibration window picks τ̂."""

ForecastPath = Literal["stationary", "shifted"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class SelectiveReport:
    """Empirical selective risk on the evaluation window (never a certificate).

    The label is a constant property, ``holdout_empirical`` (D15, Q6): it cannot be set.
    """

    alpha: float
    tol: float
    delta: float
    tau_hat: float | None
    calib_status: str
    coverage_eval: float
    n_committed_eval: int
    risk_est: float
    risk_lo: float
    risk_hi: float
    conf: float
    block_len: int
    risk_pre_shift: float
    risk_post_shift: float

    @property
    def guarantee_type(self) -> GuaranteeType:
        return GuaranteeType.HOLDOUT_EMPIRICAL

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["guarantee_type"] = self.guarantee_type.value
        return d


def implied_by_bound(deviation: float, over: float, under: float, tolerance: float) -> bool:
    """True when the ±``tolerance`` check could not fail on the side of ``deviation`` (D20).

    ``deviation`` is miscoverage − α_target; ``over`` and ``under`` are the one-sided ACI
    bounds of :func:`amx.cert.aci_side_bounds`. A zero deviation lies on both sides.
    """
    if deviation > 0.0:
        return over <= tolerance
    if deviation < 0.0:
        return under <= tolerance
    return min(over, under) <= tolerance


@dataclass(frozen=True)
class T1ForecastHorizon:
    """Interval verdict and selective report of one horizon on one path (D20).

    ``passed`` is the gate criterion (within tolerance AND non-degenerate); it gates the run
    only on the stationary path (:attr:`T1ForecastResult.gated`).
    """

    horizon: int
    aci: ACIResult
    miscoverage: float
    abs_deviation: float
    bound_over: float
    bound_under: float
    implied_by_bound: bool
    degenerate_share: float
    within_tolerance: bool
    non_degenerate: bool
    passed: bool
    selective: SelectiveReport

    @classmethod
    def from_aci(
        cls,
        aci: ACIResult,
        selective: SelectiveReport,
        *,
        tolerance: float,
        max_degenerate: float,
    ) -> T1ForecastHorizon:
        """Judge one ACI evaluation against ±``tolerance`` and the degenerate-share ceiling."""
        dev = aci.miscoverage - aci.alpha_target
        degenerate = aci.infinite_share + aci.empty_share
        within = abs(dev) <= tolerance
        non_degenerate = degenerate <= max_degenerate
        return cls(
            horizon=aci.horizon,
            aci=aci,
            miscoverage=aci.miscoverage,
            abs_deviation=abs(dev),
            bound_over=aci.bound_over,
            bound_under=aci.bound_under,
            implied_by_bound=implied_by_bound(dev, aci.bound_over, aci.bound_under, tolerance),
            degenerate_share=degenerate,
            within_tolerance=within,
            non_degenerate=non_degenerate,
            passed=within and non_degenerate,
            selective=selective,
        )

    def summary(self) -> dict[str, Any]:
        d: dict[str, Any] = dict(self.aci.summary())
        d.update(
            {
                "abs_deviation": self.abs_deviation,
                "implied_by_bound": self.implied_by_bound,
                "degenerate_share": self.degenerate_share,
                "within_tolerance": self.within_tolerance,
                "non_degenerate": self.non_degenerate,
                "passed": self.passed,
                "selective": self.selective.to_dict(),
            }
        )
        return d


@dataclass(frozen=True)
class T1ForecastResult:
    """One T1-forecast path. Only the stationary path is gated (D20)."""

    generator: str
    path: ForecastPath
    seed: int
    alpha_target: float
    gamma: float
    tolerance: float
    max_degenerate: float
    n_calib: int
    n_eval: int
    shift_index: int | None
    horizons: tuple[T1ForecastHorizon, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def target_label(self) -> str:
        """The ±tolerance target is empirical, not implied by the ACI bound at 2000 steps (Q6)."""
        return "empirical_target"

    @property
    def interval_guarantee(self) -> GuaranteeType:
        return GuaranteeType.LONG_RUN_FREQUENCY

    @property
    def gated(self) -> bool:
        """Only the stationary path carries the T1-forecast gate (D20)."""
        return self.path == "stationary"

    @property
    def passed(self) -> bool:
        """Every horizon meets the gate criterion (decides the gate only when ``gated``)."""
        return all(h.passed for h in self.horizons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator,
            "path": self.path,
            "gated": self.gated,
            "seed": self.seed,
            "alpha_target": self.alpha_target,
            "gamma": self.gamma,
            "tolerance": self.tolerance,
            "max_degenerate": self.max_degenerate,
            "target_label": self.target_label,
            "interval_guarantee": self.interval_guarantee.value,
            "n_calib": self.n_calib,
            "n_eval": self.n_eval,
            "shift_index": self.shift_index,
            "passed": self.passed,
            "horizons": [h.summary() for h in self.horizons],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class T1ForecastGate:
    """Both T1-forecast paths (D20): the stationary path is gated, the shifted one reported."""

    stationary: T1ForecastResult
    shifted: T1ForecastResult

    def __post_init__(self) -> None:
        if self.stationary.path != "stationary" or self.shifted.path != "shifted":
            raise ValueError("T1ForecastGate needs a stationary and a shifted path")

    @property
    def passed(self) -> bool:
        return self.stationary.passed

    @property
    def paths(self) -> tuple[T1ForecastResult, T1ForecastResult]:
        return (self.stationary, self.shifted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gated": (
                "stationary path, every horizon: |miscoverage - alpha_target| <= tolerance "
                "and infinite + empty share <= max_degenerate"
            ),
            "reported_only": (
                "shifted path: miscoverage, infinite and empty share, local coverage range, "
                "implied_by_bound; selective risk on both paths"
            ),
            "passed": self.passed,
            "stationary": self.stationary.to_dict(),
            "shifted": self.shifted.to_dict(),
        }


def run_t1_forecast(
    generator: SeasonalARShift | None = None,
    *,
    horizons: tuple[int, ...] = T1_FORECAST_HORIZONS,
    alpha_target: float = T1_FORECAST_ALPHA,
    gamma: float = T1_FORECAST_GAMMA,
    tolerance: float = T1_FORECAST_TOLERANCE,
    max_degenerate: float = T1_FORECAST_MAX_DEGENERATE,
    n_dev: int = 1000,
    n_calib: int = 2000,
    n_eval: int = 3000,
    shift_offset: int | None = None,
    sel_alpha: float = T1_FORECAST_SEL_ALPHA,
    sel_delta: float = 0.1,
    sel_tol: float = T1_FORECAST_SEL_TOL,
    block_len: int = 48,
    n_boot: int = 1000,
    conf: float = 0.95,
    seed: int | None = None,
    allow_short: bool = False,
) -> T1ForecastResult:
    """Run T1-forecast on one path of generator (c). See the module docstring.

    ``generator`` defaults to generator (c) WITH its shift; pass
    ``SeasonalARShift().stationary()`` for the gated path (or use
    :func:`run_t1_forecast_gate`). ``seed`` defaults to the path's pre-registered seed
    (:data:`T1_FORECAST_STATIONARY_SEED` or :data:`T1_FORECAST_SEED`). Per horizon the gate
    criterion is |miscoverage − α_target| ≤ ``tolerance`` (empirical target) and infinite +
    empty share ≤ ``max_degenerate``. ``n_eval`` evaluation origins per horizon (at least 2000
    unless ``allow_short``, which only smoke tests use); on the shifted path ``shift_offset``
    places the noise-sd doubling that many steps after the first evaluation target (default
    ``n_eval // 3``).
    """
    gen = SeasonalARShift() if generator is None else generator
    path: ForecastPath = "shifted" if gen.has_shift else "stationary"
    if seed is None:
        seed = T1_FORECAST_SEED if gen.has_shift else T1_FORECAST_STATIONARY_SEED
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must be >= 1")
    if n_eval < MIN_EVAL_STEPS and not allow_short:
        raise ValueError(f"T1-forecast needs at least {MIN_EVAL_STEPS} evaluation steps")
    h_max = max(horizons)
    rng = np.random.default_rng(seed)

    dev0 = gen.min_origin(h_max)
    dev_origins = np.arange(dev0, dev0 + n_dev, dtype=np.int64)
    cal0 = dev0 + n_dev + h_max
    cal_origins = np.arange(cal0, cal0 + n_calib, dtype=np.int64)
    cal_info_end = int(cal_origins[-1]) + h_max + 1  # every calibration target lies below
    first_origin = cal_info_end - 1 + h_max  # embargo of h_max indices
    n_time = first_origin + n_eval + h_max
    eval_origins = np.fromiter(
        (o for _, o in rolling_origins(n_time, cal_info_end, h_max, 1, h_max)), dtype=np.int64
    )[:n_eval]
    if eval_origins.shape[0] != n_eval:
        raise RuntimeError("evaluation window layout is inconsistent")
    if gen.has_shift:
        offset = n_eval // 3 if shift_offset is None else shift_offset
        shift_at = int(eval_origins[0]) + 1 + offset
    else:
        shift_at = n_time  # no shift: every target is "pre-shift"
    y = gen.series(n_time, shift_at, rng)
    tau = TauGrid().values
    budget = DeltaBudget(sel_delta, 1)

    out: list[T1ForecastHorizon] = []
    for h in horizons:
        cal = gen.forecast(y, cal_origins, h)
        ev = gen.forecast(y, eval_origins, h)
        aci = aci_run(cal.abs_resid, ev.preds, ev.targets, alpha_target, gamma, h)

        s_dev = gen.exceed_scores(y, dev_origins, h, sel_tol)
        s_cal = gen.exceed_scores(y, cal_origins, h, sel_tol)
        s_ev = gen.exceed_scores(y, eval_origins, h, sel_tol)
        dev_cov = np.mean(s_dev[:, None] <= tau[None, :], axis=0)
        loss_cal = (cal.abs_resid > sel_tol).astype(np.float64)
        loss_ev = (ev.abs_resid > sel_tol).astype(np.float64)
        res = fixed_sequence_ltt(
            grid_stats(s_cal, loss_cal, tau, binary=True),
            [sel_alpha],
            budget.delta_per_band,
            dev_cov,
        )
        tau_hat = res.tau_hat(0)
        committed = (
            (s_ev <= tau_hat).astype(np.float64) if tau_hat is not None else np.zeros_like(s_ev)
        )
        ci = block_bootstrap_ratio(
            loss_ev,
            committed,
            block_len=min(block_len, n_eval),
            B=n_boot,
            conf=conf,
            rng=rng,
        )
        post = (eval_origins + h) >= shift_at
        sel = SelectiveReport(
            alpha=sel_alpha,
            tol=sel_tol,
            delta=sel_delta,
            tau_hat=tau_hat,
            calib_status=res.bands[0].stop_reason.value,
            coverage_eval=float(np.mean(committed)),
            n_committed_eval=int(np.sum(committed)),
            risk_est=ci.est,
            risk_lo=ci.lo,
            risk_hi=ci.hi,
            conf=conf,
            block_len=min(block_len, n_eval),
            risk_pre_shift=selective_risk(loss_ev[~post], committed[~post]),
            risk_post_shift=selective_risk(loss_ev[post], committed[post]),
        )
        hz = T1ForecastHorizon.from_aci(
            aci, sel, tolerance=tolerance, max_degenerate=max_degenerate
        )
        out.append(hz)
        _log.info(
            "t1_forecast horizon",
            extra={
                "amx": {
                    "path": path,
                    "horizon": h,
                    "miscoverage": aci.miscoverage,
                    "infinite_share": aci.infinite_share,
                    "empty_share": aci.empty_share,
                    "implied_by_bound": hz.implied_by_bound,
                    "passed": hz.passed,
                }
            },
        )
    if gen.has_shift:
        notes: tuple[str, ...] = (
            "shifted path: reported, not gated (D20); the shift raises miscoverage, the side "
            "whose one-sided ACI bound is already within the tolerance (implied_by_bound)",
            "selective risk is holdout_empirical: the evaluation window contains a regime shift",
        )
    else:
        notes = (
            "stationary path: the T1-forecast gate (D20); the ±tolerance target is empirical, "
            "the ACI guarantee is the deterministic long-run bound reported per horizon",
            "selective risk is holdout_empirical: calibration and evaluation windows share "
            "one stationary law on this path",
        )
    return T1ForecastResult(
        generator=gen.name,
        path=path,
        seed=seed,
        alpha_target=alpha_target,
        gamma=gamma,
        tolerance=tolerance,
        max_degenerate=max_degenerate,
        n_calib=n_calib,
        n_eval=n_eval,
        shift_index=shift_at if gen.has_shift else None,
        horizons=tuple(out),
        notes=notes,
    )


def run_t1_forecast_gate(
    generator: SeasonalARShift | None = None,
    *,
    stationary_seed: int = T1_FORECAST_STATIONARY_SEED,
    shifted_seed: int = T1_FORECAST_SEED,
    **kwargs: Any,
) -> T1ForecastGate:
    """T1-forecast as gated by D20: the stationary path (gate) and the shifted path (reported).

    ``generator`` is generator (c) with its shift (default :class:`SeasonalARShift`); its
    :meth:`~SeasonalARShift.stationary` variant runs the gated path. Both paths use their
    pre-registered seeds; ``kwargs`` go to :func:`run_t1_forecast` for both.
    """
    gen = SeasonalARShift() if generator is None else generator
    if not gen.has_shift:
        raise ValueError("pass generator (c) with its regime shift; the gate derives the rest")
    return T1ForecastGate(
        stationary=run_t1_forecast(gen.stationary(), seed=stationary_seed, **kwargs),
        shifted=run_t1_forecast(gen, seed=shifted_seed, **kwargs),
    )
