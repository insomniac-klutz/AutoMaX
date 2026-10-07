"""Split conformal, per-horizon ACI, block bootstrap and rolling origins (ROLLER step 7, C12)."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from amx.cert import (
    aci_alphas,
    aci_bound,
    aci_run,
    aci_side_bounds,
    block_bootstrap_ratio,
    conformal_rank,
    local_coverage,
    rolling_origins,
    selective_risk,
    split_conformal_quantile,
)
from amx.cert.block_bootstrap import GUARANTEE_TYPE
from amx.cert.guarantee import GuaranteeType

# --- split conformal -------------------------------------------------------------------------


def test_split_conformal_rank_and_value() -> None:
    s = np.arange(1.0, 10.0)  # n = 9 scores 1..9
    # ceil(10 * 0.9) = 9 -> the 9th smallest
    assert split_conformal_quantile(s, 0.1) == 9.0
    # ceil(10 * 0.5) = 5
    assert split_conformal_quantile(s[::-1], 0.5) == 5.0
    # n = 8: ceil(9 * 0.9) = 9 > 8 -> infinite interval
    assert split_conformal_quantile(s[:8], 0.1) == math.inf


def test_split_conformal_boundaries() -> None:
    s = np.array([0.3, 0.1, 0.2])
    assert split_conformal_quantile(s, 0.0) == math.inf
    assert split_conformal_quantile(s, -0.2) == math.inf
    assert split_conformal_quantile(s, 1.0) == -math.inf
    assert split_conformal_quantile(np.array([]), 0.5) == math.inf
    with pytest.raises(ValueError):
        split_conformal_quantile(np.array([0.1, np.nan]), 0.1)


@pytest.mark.parametrize("n", [1, 7, 19, 99, 374, 1999])
@pytest.mark.parametrize("alpha", [0.005, 0.05, 0.1, 0.3, 0.7, 0.9])
def test_conformal_rank_is_exact(n: int, alpha: float) -> None:
    exact = (n + 1) * (1 - Fraction(alpha))
    k = conformal_rank(n, alpha)
    assert k - 1 < exact <= k


def test_conformal_rank_integer_products() -> None:
    # (n + 1)(1 - α) on exactly representable α: no float drift either way
    assert conformal_rank(19, 0.25) == 15
    assert conformal_rank(3, 0.5) == 2
    assert conformal_rank(9, 0.0) == 10
    with pytest.raises(ValueError):
        conformal_rank(-1, 0.1)
    with pytest.raises(ValueError):
        conformal_rank(5, math.nan)


# --- ACI bound -------------------------------------------------------------------------------


def test_aci_bound_classic_value() -> None:
    # Gibbs and Candès: (max(α, 1 − α) + γ) / (γ T) = 0.905 / 10 at h = 1
    b = aci_bound(0.1, 0.005, 2000, 1)
    assert b == pytest.approx(0.905 / (0.005 * 1999))
    assert abs(b - 0.0905) < 1e-3


def test_aci_bound_grows_with_horizon_and_shrinks_with_t() -> None:
    assert aci_bound(0.1, 0.005, 2000, 24) > aci_bound(0.1, 0.005, 2000, 1)
    assert aci_bound(0.1, 0.005, 9050, 1) < 0.0201
    assert aci_bound(0.1, 0.005, 4000, 1) < aci_bound(0.1, 0.005, 2000, 1)


@pytest.mark.parametrize(
    ("a0", "g", "T", "h"),
    [(-0.1, 0.005, 100, 1), (1.1, 0.005, 100, 1), (0.1, 0.0, 100, 1), (0.1, 0.01, 5, 5)],
)
def test_aci_bound_rejects_bad_args(a0: float, g: float, T: int, h: int) -> None:
    with pytest.raises(ValueError):
        aci_bound(a0, g, T, h)


def test_aci_side_bounds_formula() -> None:
    """D20: over = (α_0 + γh)/(γ(T − h)), under = (1 − α_0 + γh)/(γ(T − h))."""
    over, under = aci_side_bounds(0.1, 0.005, 3000, 1)
    assert over == pytest.approx(0.105 / (0.005 * 2999))
    assert under == pytest.approx(0.905 / (0.005 * 2999))
    over24, under24 = aci_side_bounds(0.1, 0.005, 3000, 24)
    assert over24 == pytest.approx((0.1 + 0.12) / (0.005 * 2976))
    assert under24 == pytest.approx((0.9 + 0.12) / (0.005 * 2976))
    # T1-forecast at α_0 = 0.1, γ = 0.005, T = 3000: the over side is already within ±0.02
    # for both horizons, so a miscoverage ABOVE target cannot fail the ±0.02 check there
    assert over < 0.02 and over24 < 0.02
    assert under > 0.02 and under24 > 0.02
    for a0, g, T, h in [(0.1, 0.005, 2000, 1), (0.9, 0.02, 500, 3), (0.5, 0.1, 50, 24)]:
        assert max(aci_side_bounds(a0, g, T, h)) == pytest.approx(aci_bound(a0, g, T, h))
    with pytest.raises(ValueError):
        aci_side_bounds(0.1, 0.005, 5, 5)


def test_aci_side_bounds_hold_one_sided() -> None:
    """Always missing sits near the over bound, always covering near the under bound."""
    T, g = 400, 0.05
    miss = aci_alphas(lambda t, a: 1, 0.5, g, T, 1, alpha0=0.5)
    cover = aci_alphas(lambda t, a: 0, 0.5, g, T, 1, alpha0=0.5)
    over, under = aci_side_bounds(0.5, g, T, 1)
    up = miss.feedback_miscoverage() - 0.5
    down = 0.5 - cover.feedback_miscoverage()
    assert 0.0 < up <= over + 1e-12 and 0.0 < down <= under + 1e-12
    rng = np.random.default_rng(5)
    for h in (1, 4, 24):
        bits = rng.uniform(size=T) < rng.uniform()
        tr = aci_alphas(lambda t, a, b=bits: int(b[t]), 0.1, g, T, h)
        o, u = aci_side_bounds(0.1, g, T, h)
        d = tr.feedback_miscoverage() - 0.1
        assert -u - 1e-12 <= d <= o + 1e-12


# --- generic driver --------------------------------------------------------------------------


def test_aci_alphas_classic_update_at_h1() -> None:
    errs = [1, 0, 0, 1, 1, 0, 0, 0]
    tr = aci_alphas(lambda t, a: errs[t], 0.1, 0.05, len(errs), 1)
    a = 0.1
    expected = [a]
    for e in errs[:-1]:
        a = a + 0.05 * (0.1 - e)
        expected.append(a)
    np.testing.assert_allclose(tr.alpha, expected)
    np.testing.assert_array_equal(tr.err, errs)


def test_aci_alphas_delayed_feedback() -> None:
    h = 3
    errs = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
    tr = aci_alphas(lambda t, a: errs[t], 0.2, 0.01, len(errs), h)
    # α stays at α_0 until the first feedback arrives at t = h
    assert np.all(tr.alpha[:h] == 0.2)
    for t in range(h, len(errs)):
        assert tr.alpha[t] == pytest.approx(tr.alpha[t - 1] + 0.01 * (0.2 - errs[t - h]))


def test_aci_alphas_forces_errors_at_boundaries() -> None:
    calls: list[float] = []

    def always_miss(t: int, a: float) -> int:
        calls.append(a)
        assert 0.0 < a < 1.0
        return 1

    tr = aci_alphas(always_miss, 0.1, 0.2, 200, 1)
    assert np.any(tr.alpha <= 0.0)
    # every α ≤ 0 step is an infinite interval: error forced to 0, err_fn not called
    assert np.all(tr.err[tr.alpha <= 0.0] == 0)
    assert len(calls) == int(np.sum(tr.alpha > 0.0))

    tr2 = aci_alphas(lambda t, a: 0, 0.9, 0.2, 200, 2)
    assert np.any(tr2.alpha >= 1.0)
    assert np.all(tr2.err[tr2.alpha >= 1.0] == 1)


def test_aci_alphas_telescoping_identity() -> None:
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, 500)
    for h in (1, 4):
        tr = aci_alphas(lambda t, a: int(bits[t]), 0.1, 0.02, 500, h)
        lhs = tr.feedback_miscoverage() - 0.1
        rhs = (tr.alpha[0] - tr.alpha[-1]) / (0.02 * (500 - h))
        assert lhs == pytest.approx(rhs, abs=1e-12)
        assert abs(lhs) <= aci_bound(0.1, 0.02, 500, h) + 1e-12


def test_aci_alphas_rejects_bad_returns_and_args() -> None:
    with pytest.raises(ValueError):
        aci_alphas(lambda t, a: 2, 0.1, 0.01, 10, 1)
    with pytest.raises(ValueError):
        aci_alphas(lambda t, a: 0, 0.0, 0.01, 10, 1)
    with pytest.raises(ValueError):
        aci_alphas(lambda t, a: 0, 0.1, 0.01, 10, 0)
    with pytest.raises(ValueError):
        aci_alphas(lambda t, a: 0, 0.1, 0.01, 0, 1)


# --- aci_run ---------------------------------------------------------------------------------


def test_aci_run_matches_split_conformal_per_step() -> None:
    rng = np.random.default_rng(0)
    cal = np.abs(rng.standard_normal(500))
    T = 1500
    preds = rng.standard_normal(T)
    ys = preds + rng.standard_normal(T) * np.where(np.arange(T) < 700, 1.0, 1.6)
    res = aci_run(cal, preds, ys, 0.1, 0.005, 2)
    assert res.steps == T and res.horizon == 2
    for t in range(T):
        a = float(res.alpha[t])
        if a >= 1.0:
            assert math.isnan(res.half_width[t]) and res.err[t] == 1
            continue
        q = split_conformal_quantile(cal, a)
        if q == math.inf:
            assert res.half_width[t] == math.inf and res.err[t] == 0
        else:
            assert res.half_width[t] == q
            assert res.err[t] == int(abs(ys[t] - preds[t]) > q)
    assert res.miscoverage == pytest.approx(float(np.mean(res.err)))
    assert res.local_coverage.shape == (T - 300 + 1,)
    assert res.guarantee_type is GuaranteeType.LONG_RUN_FREQUENCY
    assert res.bound == pytest.approx(aci_bound(0.1, 0.005, T, 2))
    first = res.err[: T - 2].mean()
    assert abs(first - 0.1) <= res.bound + 1e-12
    assert res.feedback_miscoverage == pytest.approx(first)
    assert (res.bound_over, res.bound_under) == pytest.approx(aci_side_bounds(0.1, 0.005, T, 2))
    assert -res.bound_under - 1e-12 <= first - 0.1 <= res.bound_over + 1e-12
    s = res.summary()
    assert s["bound_over"] == res.bound_over and s["bound_under"] == res.bound_under
    assert s["feedback_miscoverage"] == res.feedback_miscoverage


def test_aci_run_counts_infinite_intervals() -> None:
    # every residual exceeds every calibration score: α is driven to ≤ 0 (infinite intervals)
    cal = np.linspace(0.0, 1.0, 50)
    T = 600
    preds = np.zeros(T)
    ys = np.full(T, 10.0)
    res = aci_run(cal, preds, ys, 0.1, 0.05, 1)
    assert res.infinite_share > 0.0
    assert res.empty_share == 0.0
    inf_steps = np.isposinf(res.half_width)
    assert np.all(res.err[inf_steps] == 0)
    # α in (0, 1/(n+1)) is infinite too: the rank exceeds n
    assert np.all(inf_steps[(res.alpha > 0) & (res.alpha < 1 / 51)])
    s = res.summary()
    assert s["infinite_share"] == res.infinite_share


def test_aci_run_counts_empty_intervals() -> None:
    # perfect forecasts never miss: α climbs to ≥ 1 (empty intervals, err = 1)
    cal = np.linspace(0.1, 1.0, 50)
    T = 600
    preds = np.arange(T, dtype=float)
    res = aci_run(cal, preds, preds.copy(), 0.5, 0.05, 1)
    assert res.empty_share > 0.0
    empty = np.isnan(res.half_width)
    assert np.all(res.alpha[empty] >= 1.0)
    assert np.all(res.err[empty] == 1)
    assert math.isfinite(res.median_finite_width)


def test_aci_label_is_never_upgraded() -> None:
    """ROLLER step 7: the ACI label is fixed; no constructor argument can change it."""
    res = aci_run(np.ones(20), np.zeros(30), np.zeros(30), 0.1)
    assert res.guarantee_type is GuaranteeType.LONG_RUN_FREQUENCY
    assert res.summary()["guarantee_type"] == "long_run_frequency"
    kwargs = {f: getattr(res, f) for f in res.__dataclass_fields__}
    with pytest.raises(TypeError):
        type(res)(**kwargs, guarantee_type=GuaranteeType.PAC_HIGH_PROB)  # type: ignore[call-arg]
    with pytest.raises(AttributeError):
        res.guarantee_type = GuaranteeType.PAC_HIGH_PROB  # type: ignore[misc]


def test_aci_run_validates_inputs() -> None:
    with pytest.raises(ValueError):
        aci_run(np.array([]), np.zeros(5), np.zeros(5), 0.1)
    with pytest.raises(ValueError):
        aci_run(np.array([-1.0]), np.zeros(5), np.zeros(5), 0.1)
    with pytest.raises(ValueError):
        aci_run(np.ones(3), np.zeros(5), np.zeros(4), 0.1)
    with pytest.raises(ValueError):
        aci_run(np.ones(3), np.zeros(2), np.zeros(2), 0.1, horizon=2)


def test_local_coverage_window() -> None:
    err = np.array([0, 1, 0, 0, 1, 1])
    np.testing.assert_allclose(local_coverage(err, 3), [2 / 3, 2 / 3, 2 / 3, 1 / 3])
    assert local_coverage(err, 10).size == 0
    with pytest.raises(ValueError):
        local_coverage(err, 0)


# --- block bootstrap -------------------------------------------------------------------------


def test_selective_risk_ratio() -> None:
    assert selective_risk([1, 0, 1, 1], [1, 1, 0, 1]) == pytest.approx(2 / 3)
    assert math.isnan(selective_risk([1, 0], [0, 0]))


def test_block_bootstrap_basic_properties() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    c = (rng.uniform(size=n) < 0.6).astype(float)
    ell = (rng.uniform(size=n) < 0.1).astype(float)
    est, lo, hi = block_bootstrap_ratio(ell, c, block_len=40, B=400, rng=np.random.default_rng(2))
    assert est == pytest.approx(selective_risk(ell, c))
    assert lo <= est <= hi
    assert lo >= 0.0 and hi <= 1.0
    again = block_bootstrap_ratio(ell, c, block_len=40, B=400, rng=np.random.default_rng(2))
    assert again == (est, lo, hi)
    assert GUARANTEE_TYPE is GuaranteeType.HOLDOUT_EMPIRICAL


def test_block_bootstrap_full_block_is_degenerate() -> None:
    ell = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    c = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    est, lo, hi = block_bootstrap_ratio(ell, c, block_len=5, B=50, rng=np.random.default_rng(0))
    assert lo == hi == est == pytest.approx(0.75)


def test_block_bootstrap_matches_naive_resampling() -> None:
    # the prefix-sum implementation equals an explicit concatenation of drawn blocks
    rng = np.random.default_rng(5)
    n, b = 103, 10
    ell = rng.uniform(size=n)
    c = (rng.uniform(size=n) < 0.5).astype(float)
    B = 30
    lo_hi = block_bootstrap_ratio(ell, c, block_len=b, B=B, conf=0.5, rng=np.random.default_rng(9))
    starts = np.random.default_rng(9).integers(0, n - b + 1, size=(B, -(-n // b)))
    reps = []
    for row in starts:
        idx = np.concatenate([np.arange(s, s + b) for s in row])[:n]
        if c[idx].sum() > 0:
            reps.append(float(np.sum(ell[idx] * c[idx]) / np.sum(c[idx])))
    lo, hi = np.quantile(reps, [0.25, 0.75])
    assert lo_hi.lo == pytest.approx(lo) and lo_hi.hi == pytest.approx(hi)


def test_block_bootstrap_nothing_committed_and_errors() -> None:
    r = block_bootstrap_ratio(
        np.ones(10), np.zeros(10), block_len=2, B=10, rng=np.random.default_rng(0)
    )
    assert math.isnan(r.est) and math.isnan(r.lo) and math.isnan(r.hi)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        block_bootstrap_ratio(np.ones(5), np.ones(4), block_len=2, rng=rng)
    with pytest.raises(ValueError):
        block_bootstrap_ratio(np.ones(5), np.full(5, 0.5), block_len=2, rng=rng)
    with pytest.raises(ValueError):
        block_bootstrap_ratio(np.ones(5), np.ones(5), block_len=6, rng=rng)
    with pytest.raises(ValueError):
        block_bootstrap_ratio(np.ones(5), np.ones(5), block_len=2, conf=1.0, rng=rng)


# --- rolling origins -------------------------------------------------------------------------


def test_rolling_origins_example() -> None:
    assert list(rolling_origins(10, 5, 2, stride=2)) == [(5, 4), (7, 6)]
    assert list(rolling_origins(10, 5, 2, stride=1, embargo=2)) == [(5, 6), (6, 7)]


@pytest.mark.parametrize("embargo", [0, 1, 24])
@pytest.mark.parametrize("stride", [1, 3])
def test_rolling_origins_invariants(embargo: int, stride: int) -> None:
    n, initial, hmax = 300, 50, 24
    pairs = list(rolling_origins(n, initial, hmax, stride, embargo))
    assert pairs, "some folds must fit"
    for train_end, origin in pairs:
        assert origin == train_end - 1 + embargo
        assert origin + hmax <= n - 1
        # first target lies `embargo` indices after the fitting window
        assert origin + 1 - train_end == embargo
    ends = [p[0] for p in pairs]
    assert ends == list(range(initial, initial + stride * len(pairs), stride))
    # the next fold would not fit
    nxt = ends[-1] + stride - 1 + embargo
    assert nxt + hmax > n - 1


def test_rolling_origins_errors_and_empty() -> None:
    assert list(rolling_origins(10, 9, 3)) == []
    with pytest.raises(ValueError):
        list(rolling_origins(10, 0, 1))
    with pytest.raises(ValueError):
        list(rolling_origins(10, 2, 1, embargo=-1))
