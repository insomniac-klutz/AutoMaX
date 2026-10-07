"""Monte Carlo checks of the forecasting-interval statistics (C12, Q6, CLAUDE.md MC rule).

* Split conformal: marginal coverage P(s_{n+1} ≤ q) lies in [1 − α, 1 − α + 1/(n + 1)] for
  exchangeable continuous scores (finite-sample guarantee).
* ACI on generator (c) (AR(1) + season 24): long-run miscoverage per horizon over 2000 steps,
  stationary and with the regime shift. The deterministic bound must hold on every path; the
  ±0.02 band is an EMPIRICAL target (Q6) and the infinite-interval share is reported.
* Moving-block bootstrap: the percentile CI of the selective-risk ratio under serial
  dependence has approximately nominal coverage (approximate; labelled ``holdout_empirical``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.stats import multivariate_normal, norm

from amx.cert import aci_bound, aci_run, block_bootstrap_ratio, conformal_rank
from amx.cert import split_conformal_quantile as scq
from amx.sim.generators import SeasonalARShift

SEED = 150_001


@pytest.mark.slow
@pytest.mark.parametrize("n", [19, 99, 500])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_split_conformal_marginal_coverage(n: int, alpha: float) -> None:
    reps = 40_000
    rng = np.random.default_rng(SEED + n)
    s = rng.exponential(size=(reps, n + 1))
    k = conformal_rank(n, alpha)
    if k > n:
        q = np.full(reps, math.inf)
    else:
        q = np.partition(s[:, :n], k - 1, axis=1)[:, k - 1]
        for i in range(5):  # the vectorized rank equals the library function
            assert q[i] == scq(s[i, :n], alpha)
    cover = float(np.mean(s[:, n] <= q))
    se = math.sqrt(0.25 / reps)
    lo, hi = 1 - alpha, 1 - alpha + 1 / (n + 1)
    assert lo - 3 * se <= cover <= hi + 3 * se, (cover, lo, hi)


def _aci_path(
    gen: SeasonalARShift, rng: np.random.Generator, h: int, *, shift: bool
) -> tuple[float, float, float, float, float]:
    """One path: calibration window, embargo, 2000 evaluation origins. Returns
    (miscoverage over all steps, feedback miscoverage, bound, infinite share, empty share)."""
    n_cal, n_eval, h_max = 1000, 2000, 24
    c0 = gen.min_origin(h_max)
    cal_o = np.arange(c0, c0 + n_cal)
    e0 = int(cal_o[-1]) + h_max + h_max
    ev_o = np.arange(e0, e0 + n_eval)
    n_time = int(ev_o[-1]) + h_max + 1
    shift_at = e0 + n_eval // 3 if shift else n_time
    y = gen.series(n_time, shift_at, rng)
    cal = gen.forecast(y, cal_o, h)
    ev = gen.forecast(y, ev_o, h)
    res = aci_run(cal.abs_resid, ev.preds, ev.targets, 0.1, 0.005, h)
    fb = float(np.mean(res.err[: n_eval - h]))
    return res.miscoverage, fb, res.bound, res.infinite_share, res.empty_share


@pytest.mark.slow
@pytest.mark.parametrize("h", [1, 24])
@pytest.mark.parametrize("shift", [False, True])
def test_aci_miscoverage_on_seasonal_ar(h: int, shift: bool) -> None:
    gen = SeasonalARShift()
    paths = 150
    rng = np.random.default_rng(SEED + 10 * h + int(shift))
    rows = np.array([_aci_path(gen, rng, h, shift=shift) for _ in range(paths)])
    mis, fb, bound, inf_share, empty_share = rows.T
    # deterministic guarantee: every path, no exceptions
    assert np.all(np.abs(fb - 0.1) <= bound + 1e-12)
    assert np.all(empty_share == 0.0)  # α never reaches 1 at α_target = 0.1, γ = 0.005
    within = float(np.mean(np.abs(mis - 0.1) <= 0.02))
    report = (
        f"h={h} shift={shift}: mean miscoverage {mis.mean():.4f} (sd {mis.std():.4f}), "
        f"share within ±0.02 {within:.3f}, bound {bound[0]:.4f}, "
        f"infinite share mean {inf_share.mean():.4f} max {inf_share.max():.4f}"
    )
    print(report)
    # the long-run target is met on average (MC SE of the mean ≈ sd / sqrt(paths))
    assert abs(mis.mean() - 0.1) <= 0.01, report
    # the empirical ±0.02 target holds on nearly every path
    assert within >= 0.95, report
    if not shift:
        # stationary: α_t stays near α_target, intervals are essentially never infinite
        assert inf_share.mean() < 0.01, report


def _true_ratio(a: float, b: float, rho: float) -> float:
    """P(u > b | w ≤ a) for standard bivariate normal (u, w) with correlation ρ."""
    joint = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).cdf([b, a])
    return float(1.0 - joint / norm.cdf(a))


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    """Stationary Gaussian AR(1) with unit marginal variance."""
    eps = rng.standard_normal(n) * math.sqrt(1 - phi**2)
    eps[0] = rng.standard_normal()
    return np.asarray(lfilter([1.0], [1.0, -phi], eps))


@pytest.mark.slow
def test_block_bootstrap_ratio_coverage_under_dependence() -> None:
    """Commit c_t = 1[w_t ≤ a], loss ℓ_t = 1[u_t > b], w and u AR(1) with corr ρ.

    The moving-block percentile CI is APPROXIMATE: at n = 2000 (about 95 committed losses)
    its coverage at nominal 0.95 was measured at 0.92 to 0.94 for block lengths 12 to 50
    (the percentile interval of a skewed ratio under-covers slightly). The test asserts at
    least 0.90 and that blocks repair the clear under-coverage of the iid bootstrap.
    """
    a, b, rho, phi = 0.3, 1.0, 0.5, 0.6
    truth = _true_ratio(a, b, rho)
    rng = np.random.default_rng(SEED + 99)
    reps, n = 600, 2000
    hits_block = hits_iid = n_iid = 0
    for r in range(reps):
        w = _ar1(rng, n, phi)
        u = rho * w + math.sqrt(1 - rho**2) * _ar1(rng, n, phi)
        c = (w <= a).astype(float)
        ell = (u > b).astype(float)
        _, lo, hi = block_bootstrap_ratio(ell, c, block_len=25, B=500, rng=rng)
        hits_block += lo <= truth <= hi
        if r % 3 == 0:  # the iid comparison is costly (one block per unit): every third rep
            _, lo1, hi1 = block_bootstrap_ratio(ell, c, block_len=1, B=500, rng=rng)
            hits_iid += lo1 <= truth <= hi1
            n_iid += 1
    cov_block, cov_iid = hits_block / reps, hits_iid / n_iid
    print(f"block-bootstrap coverage {cov_block:.3f}, iid-bootstrap coverage {cov_iid:.3f}")
    assert 0.90 <= cov_block <= 0.99, cov_block
    # serial dependence makes the naive iid bootstrap under-cover; blocks repair most of it
    assert cov_block > cov_iid + 0.03, (cov_block, cov_iid)


def test_aci_bound_needs_about_9050_steps_for_002() -> None:
    """Q6: ±0.02 is not implied by the ACI bound at 2000 steps (needs ≈ 9,050)."""
    assert aci_bound(0.1, 0.005, 2000, 1) > 0.02
    assert aci_bound(0.1, 0.005, 9100, 1) <= 0.02 < aci_bound(0.1, 0.005, 9000, 1)
