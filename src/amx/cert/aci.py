"""Split-conformal intervals and Adaptive Conformal Inference per horizon (HANDOFF 7.5, A.4).

As amended by C12 and OQ Q6 (decision D18):

* ACI is an **evaluation** tool for forecast intervals. It is never shipped as online state in
  a frozen artifact; the selective commit of the forecasting family goes through the scorer,
  the τ grid and LTT on a calibration window and is labelled ``holdout_empirical``.
* Updates are **per horizon with delayed feedback**. At origin t an interval for y_{t+h} is
  issued at level α_t; its error err_t is only observed at origin t + h, so

      α_t = α_{t-1} + γ (α_target − err_{t-h})   if t − h ≥ 0,   else α_t = α_{t-1},

  with α_{-1} = α_0 (the start value, by default α_target). For h = 1 this is the classic
  update α_{t+1} = α_t + γ (α_target − err_t) of Gibbs and Candès.
* α_t ≤ 0 issues the infinite interval (err = 0) and α_t ≥ 1 the empty interval (err = 1).
  Both are counted and reported (``infinite_share``, ``empty_share``).

The guarantee is a long-run frequency statement about **interval miscoverage only**
(``GuaranteeType.LONG_RUN_FREQUENCY``); see :func:`aci_bound`. It says nothing about the
selective risk of committed forecasts.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amx._log import get_logger
from amx.cert.guarantee import GuaranteeType

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

LOCAL_WINDOW = 300
"""Window (in evaluation steps) of the local-coverage trace (HANDOFF 7.5)."""

DEFAULT_GAMMA = 0.005
"""Default ACI step size (HANDOFF 7.5, A.4)."""

_log = get_logger(__name__)


def conformal_rank(n: int, alpha: float) -> int:
    """The rank k = ⌈(n + 1)(1 − α)⌉ of the split-conformal quantile among n scores.

    Computed in exact rational arithmetic on the given float ``alpha``, so the ceiling never
    moves because of rounding in ``(n + 1) * (1 - alpha)``. ``k > n`` means the infinite
    interval; ``k ≤ 0`` (only when α ≥ 1) means the empty interval.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    return math.ceil((n + 1) * (1 - Fraction(alpha)))


def split_conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """Split-conformal threshold: the ⌈(n+1)(1−α)⌉-th smallest calibration score.

    Returns ``+inf`` when that rank exceeds n (too few scores for level 1 − α, including every
    α ≤ 0) and ``-inf`` when the rank is below 1 (α ≥ 1: the empty set). With exchangeable
    scores, P(s_{n+1} ≤ q) ≥ 1 − α.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(s)):
        raise ValueError("conformal scores must be finite")
    k = conformal_rank(int(s.shape[0]), alpha)
    if k > s.shape[0]:
        return math.inf
    if k < 1:
        return -math.inf
    return float(np.partition(s, k - 1)[k - 1])


def aci_bound(alpha0: float, gamma: float, T: int, h: int) -> float:
    """Deterministic bound on |mean(err_0, …, err_{T-h-1}) − α_target| for per-horizon ACI.

    Derivation (Gibbs and Candès 2021, Prop. 4.1, extended to delay h):

    1. **Range.** For α_0 ∈ [0, 1], every α_t lies in [−γh, 1 + γh]. One step moves α by at most
       γ·max(α_target, 1 − α_target) ≤ γ. Take an excursion of α below 0 that starts at t*
       (α_{t*-1} > 0 ≥ α_{t*}), so α_{t*} > −γ. A later step u decreases α only if
       err_{u-h} = 1, which needs α_{u-h} > 0 (α ≤ 0 forces err = 0). Inside the excursion that
       is only possible for u − h < t*, i.e. for at most h − 1 further steps. Hence the
       excursion stays above −γh. The upper side is symmetric (α ≥ 1 forces err = 1).
    2. **Telescoping.** The updates at t = h, …, T − 1 consume err_0, …, err_{T-h-1}, so
       α_{T-1} = α_0 + γ Σ_{s=0}^{T-h-1} (α_target − err_s), i.e.
       mean(err_0..err_{T-h-1}) − α_target = (α_0 − α_{T-1}) / (γ (T − h)).
    3. **Bound.** |α_0 − α_{T-1}| ≤ max(α_0 + γh, 1 + γh − α_0) = max(α_0, 1 − α_0) + γh.

    So |mean − α_target| ≤ (max(α_0, 1 − α_0) + γh) / (γ (T − h)). For h = 1, γ = 0.005,
    T = 2000, α_0 = 0.1 this is ≈ 0.0905, the classic value. Holds for any error sequence,
    including adversarial ones; it is the only guarantee ACI gives (``long_run_frequency``).
    """
    return max(aci_side_bounds(alpha0, gamma, T, h))


def aci_side_bounds(alpha0: float, gamma: float, T: int, h: int) -> tuple[float, float]:
    """One-sided deterministic bounds ``(over, under)`` on the feedback miscoverage (D20).

    From steps 2 and 3 of :func:`aci_bound`, mean(err_0, …, err_{T-h-1}) − α_target =
    (α_0 − α_{T-1}) / (γ (T − h)) with α_{T-1} ∈ [−γh, 1 + γh], so

    * ``over``  = (α_0 + γh) / (γ (T − h)) bounds mean − α_target (miscoverage ABOVE target);
    * ``under`` = (1 − α_0 + γh) / (γ (T − h)) bounds α_target − mean (miscoverage BELOW it).

    ``aci_bound`` is the larger of the two. When one side's bound is already within a
    tolerance, a deviation on that side cannot fail a ±tolerance check: at α_0 = 0.1,
    γ = 0.005, T = 3000 the over side is ≈ 0.007 (h = 1) and ≈ 0.015 (h = 24), so only an
    under-coverage deviation can fail the T1-forecast ±0.02 target (D20).
    """
    if not (0.0 <= alpha0 <= 1.0):
        raise ValueError("alpha0 must lie in [0, 1]")
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")
    if h < 1 or h >= T:
        raise ValueError("need h >= 1 and T > h")
    denom = gamma * (T - h)
    return ((alpha0 + gamma * h) / denom, (1.0 - alpha0 + gamma * h) / denom)


@dataclass(frozen=True)
class ACITrace:
    """Per-step ACI state from the generic driver :func:`aci_alphas`."""

    alpha: FloatArray
    err: IntArray
    alpha_target: float
    gamma: float
    horizon: int

    @property
    def steps(self) -> int:
        return int(self.alpha.shape[0])

    def feedback_miscoverage(self) -> float:
        """mean(err_0..err_{T-h-1}): the errors whose feedback entered the updates."""
        n = self.steps - self.horizon
        return float(np.mean(self.err[:n])) if n > 0 else math.nan


def _check_aci_args(alpha_target: float, gamma: float, horizon: int, alpha0: float) -> None:
    if not (0.0 < alpha_target < 1.0):
        raise ValueError("alpha_target must lie in (0, 1)")
    if not (0.0 <= alpha0 <= 1.0):
        raise ValueError("alpha0 must lie in [0, 1]")
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")


def aci_alphas(
    err_fn: Callable[[int, float], int],
    alpha_target: float,
    gamma: float,
    T: int,
    horizon: int,
    *,
    alpha0: float | None = None,
) -> ACITrace:
    """Generic ACI driver with delayed feedback (C12).

    ``err_fn(t, α_t)`` supplies err_t ∈ {0, 1} only while 0 < α_t < 1; at the boundaries the
    driver forces err_t = 0 (α_t ≤ 0, infinite interval) or err_t = 1 (α_t ≥ 1, empty
    interval) without calling it. Used by the property test of :func:`aci_bound`.
    """
    a0 = alpha_target if alpha0 is None else alpha0
    _check_aci_args(alpha_target, gamma, horizon, a0)
    if T < 1:
        raise ValueError("T must be >= 1")
    alpha = np.empty(T, dtype=np.float64)
    err = np.empty(T, dtype=np.int64)
    a = a0
    for t in range(T):
        if t >= horizon:
            a = a + gamma * (alpha_target - int(err[t - horizon]))
        alpha[t] = a
        if a <= 0.0:
            e = 0
        elif a >= 1.0:
            e = 1
        else:
            e = int(err_fn(t, a))
            if e not in (0, 1):
                raise ValueError(f"err_fn must return 0 or 1, got {e!r}")
        err[t] = e
    return ACITrace(alpha=alpha, err=err, alpha_target=alpha_target, gamma=gamma, horizon=horizon)


def local_coverage(err: ArrayLike, window: int = LOCAL_WINDOW) -> FloatArray:
    """Coverage 1 − mean(err) over each run of ``window`` consecutive steps (HANDOFF 7.5).

    Returns T − window + 1 values (empty when T < window).
    """
    e = np.asarray(err, dtype=np.float64).reshape(-1)
    if window < 1:
        raise ValueError("window must be >= 1")
    if e.shape[0] < window:
        return np.empty(0, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(e)])
    return np.asarray(1.0 - (cs[window:] - cs[:-window]) / window, dtype=np.float64)


@dataclass(frozen=True)
class ACIResult:
    """Interval evaluation of one horizon by ACI over a fixed calibration score set.

    ``half_width`` is +inf for infinite intervals and NaN for empty ones (α_t ≥ 1).
    ``median_finite_width`` is the median full width 2·q over the finite, non-empty intervals.
    ``miscoverage`` is the mean error over all T evaluation steps; ``bound`` is the
    deterministic :func:`aci_bound` on the first T − h of them (``feedback_miscoverage``) and
    ``bound_over`` / ``bound_under`` are its one-sided parts (:func:`aci_side_bounds`, D20).
    The two means differ by at most h / T. The guarantee label is a
    constant property: an ACI evaluation can never be relabelled as anything stronger than
    ``long_run_frequency`` (C12, Q6).
    """

    horizon: int
    alpha_target: float
    gamma: float
    alpha: FloatArray
    half_width: FloatArray
    err: IntArray
    miscoverage: float
    local_window: int
    local_coverage: FloatArray
    infinite_share: float
    empty_share: float
    median_finite_width: float
    bound: float
    bound_over: float
    bound_under: float
    n_calib: int

    @property
    def guarantee_type(self) -> GuaranteeType:
        """Interval miscoverage only, as a long-run frequency; fixed, never upgraded."""
        return GuaranteeType.LONG_RUN_FREQUENCY

    @property
    def steps(self) -> int:
        return int(self.alpha.shape[0])

    @property
    def feedback_miscoverage(self) -> float:
        """mean(err_0..err_{T-h-1}): the errors the deterministic bounds cover."""
        n = self.steps - self.horizon
        return float(np.mean(self.err[:n])) if n > 0 else math.nan

    @property
    def local_coverage_range(self) -> tuple[float, float]:
        if self.local_coverage.size == 0:
            return (math.nan, math.nan)
        return (float(np.min(self.local_coverage)), float(np.max(self.local_coverage)))

    def summary(self) -> dict[str, float | int | str]:
        lo, hi = self.local_coverage_range
        return {
            "horizon": self.horizon,
            "steps": self.steps,
            "alpha_target": self.alpha_target,
            "gamma": self.gamma,
            "miscoverage": self.miscoverage,
            "feedback_miscoverage": self.feedback_miscoverage,
            "bound": self.bound,
            "bound_over": self.bound_over,
            "bound_under": self.bound_under,
            "infinite_share": self.infinite_share,
            "empty_share": self.empty_share,
            "median_finite_width": self.median_finite_width,
            "local_window": self.local_window,
            "local_coverage_min": lo,
            "local_coverage_max": hi,
            "n_calib": self.n_calib,
            "guarantee_type": self.guarantee_type.value,
        }


def aci_run(
    calib_scores: ArrayLike,
    preds: ArrayLike,
    ys: ArrayLike,
    alpha_target: float,
    gamma: float = DEFAULT_GAMMA,
    horizon: int = 1,
    *,
    alpha0: float | None = None,
    window: int = LOCAL_WINDOW,
) -> ACIResult:
    """ACI for absolute-residual scores with a FIXED calibration score set (C12, Q6).

    ``preds[t]`` is the point forecast issued at origin t for y_{t+h} and ``ys[t]`` the
    realised y_{t+h}. The interval at origin t is ``preds[t] ± q(α_t)`` with q the
    split-conformal quantile of ``calib_scores`` (absolute residuals from a calibration window
    that precedes the evaluation window). err_t = 1[|ys[t] − preds[t]| > q(α_t)] and its
    feedback reaches α at origin t + h.
    """
    a0 = alpha_target if alpha0 is None else alpha0
    _check_aci_args(alpha_target, gamma, horizon, a0)
    cal = np.sort(np.asarray(calib_scores, dtype=np.float64).reshape(-1))
    if cal.size == 0 or not np.all(np.isfinite(cal)) or np.any(cal < 0):
        raise ValueError("calib_scores must be a non-empty set of finite absolute residuals")
    p = np.asarray(preds, dtype=np.float64).reshape(-1)
    y = np.asarray(ys, dtype=np.float64).reshape(-1)
    if p.shape != y.shape or p.size <= horizon:
        raise ValueError("preds and ys must have the same length T > horizon")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(y))):
        raise ValueError("preds and ys must be finite")
    resid = np.abs(y - p)
    n = int(cal.shape[0])
    T = int(p.shape[0])

    alpha = np.empty(T, dtype=np.float64)
    half = np.empty(T, dtype=np.float64)
    err = np.empty(T, dtype=np.int64)
    a = a0
    for t in range(T):
        if t >= horizon:
            a = a + gamma * (alpha_target - int(err[t - horizon]))
        alpha[t] = a
        if a >= 1.0:
            half[t], err[t] = math.nan, 1
            continue
        k = conformal_rank(n, a) if a > 0.0 else n + 1
        if k > n:
            half[t], err[t] = math.inf, 0
            continue
        q = float(cal[max(k, 1) - 1])
        half[t] = q
        err[t] = int(resid[t] > q)

    finite = np.isfinite(half)
    infinite_share = float(np.mean(np.isposinf(half)))
    empty_share = float(np.mean(np.isnan(half)))
    med = float(np.median(2.0 * half[finite])) if np.any(finite) else math.nan
    loc = local_coverage(err, window)
    over, under = aci_side_bounds(a0, gamma, T, horizon)
    res = ACIResult(
        horizon=horizon,
        alpha_target=alpha_target,
        gamma=gamma,
        alpha=alpha,
        half_width=half,
        err=err,
        miscoverage=float(np.mean(err)),
        local_window=window,
        local_coverage=loc,
        infinite_share=infinite_share,
        empty_share=empty_share,
        median_finite_width=med,
        bound=max(over, under),
        bound_over=over,
        bound_under=under,
        n_calib=n,
    )
    _log.debug(
        "aci_run",
        extra={
            "amx": {
                "horizon": horizon,
                "steps": T,
                "miscoverage": res.miscoverage,
                "infinite_share": infinite_share,
                "empty_share": empty_share,
            }
        },
    )
    return res
