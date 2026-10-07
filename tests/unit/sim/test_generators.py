"""Generators of amx.sim: shapes, dtypes, determinism and documented formulas (ROLLER step 8)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import norm

from amx.cert import TauGrid
from amx.sim import (
    ClusteredUnits,
    GaussianMixture,
    HeteroscedasticRegression,
    SeasonalARShift,
    SelectiveGenerator,
    selective_generator,
)

TAU = TauGrid().values


@pytest.mark.parametrize("key", ["a", "b", "d"])
def test_sample_shapes_and_dtypes(key: str) -> None:
    gen = selective_generator(key)
    assert isinstance(gen, SelectiveGenerator)
    b = gen.sample(300, np.random.default_rng(0))
    n = b.n_units
    for arr in (b.scores, b.losses, b.cond_risk):
        assert arr.dtype == np.float64 and arr.shape == (n,)
    assert np.all((b.losses == 0.0) | (b.losses == 1.0))
    assert np.all((b.cond_risk >= 0.0) & (b.cond_risk <= 1.0))
    assert np.all((b.scores >= 0.0) & (b.scores <= 1.0))
    if key == "d":
        assert gen.grouped
        assert b.groups is not None and b.groups.dtype == np.int64
        assert np.unique(b.groups).shape[0] == 300  # n counts groups
        assert n >= 300
    else:
        assert not gen.grouped
        assert b.groups is None and n == 300


@pytest.mark.parametrize("key", ["a", "b", "d"])
def test_sample_is_deterministic(key: str) -> None:
    gen = selective_generator(key)
    b1 = gen.sample(200, np.random.default_rng(42))
    b2 = gen.sample(200, np.random.default_rng(42))
    np.testing.assert_array_equal(b1.scores, b2.scores)
    np.testing.assert_array_equal(b1.losses, b2.losses)


def test_generator_c_is_not_selective() -> None:
    with pytest.raises(ValueError, match="forecasting-only"):
        selective_generator("c")


def test_gaussian_mixture_plugin_is_fixed_by_seed() -> None:
    g1, g2 = GaussianMixture(), GaussianMixture()
    np.testing.assert_array_equal(g1.plugin_means, g2.plugin_means)
    assert not np.array_equal(g1.plugin_means, GaussianMixture(seed=1).plugin_means)
    # perturbation of the documented size around means on a circle of radius 1.5
    np.testing.assert_allclose(np.linalg.norm(g1.means, axis=1), 1.5)
    assert np.max(np.abs(g1.plugin_means - g1.means)) < 0.15 * 4


def test_gaussian_mixture_score_and_risk() -> None:
    g = GaussianMixture()
    x = np.random.default_rng(1).standard_normal((1000, 2)) * 2
    yhat, s, r = g.predict(x)
    ph, pt = g.plugin_posterior(x), g.true_posterior(x)
    np.testing.assert_allclose(ph.sum(axis=1), 1.0)
    np.testing.assert_array_equal(yhat, np.argmax(ph, axis=1))
    np.testing.assert_allclose(s, 1.0 - ph.max(axis=1))
    np.testing.assert_allclose(r, 1.0 - pt[np.arange(1000), yhat])
    assert np.all(s <= 1.0 - 1.0 / 4 + 1e-12)
    # temperature 1.6 > 1 makes the plug-in under-confident relative to its own logits
    cold = GaussianMixture(temperature=1.0).plugin_posterior(x).max(axis=1)
    assert np.all(ph.max(axis=1) <= cold + 1e-12)


def test_heteroscedastic_formulas() -> None:
    g = HeteroscedasticRegression()
    x = np.linspace(0.0, 1.0, 101)
    sg = 0.1 + 0.4 * x
    b = 0.05 * np.cos(3 * x)
    r = 1 - (norm.cdf((0.3 - b) / sg) - norm.cdf((-0.3 - b) / sg))
    np.testing.assert_allclose(g.cond_risk(x), r, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(g.score(x), 2 * norm.cdf(-0.3 / (0.8 * sg)), rtol=1e-12)
    assert float(np.min(g.cond_risk(x))) > 0.005  # the 0.5% band is infeasible by design


def test_clustered_formulas_and_sizes() -> None:
    g = ClusteredUnits()
    s, r, gid = g.draw(20_000, np.random.default_rng(3))
    sizes = np.bincount(gid)
    assert sizes.min() >= 1
    assert abs(sizes.mean() - 5.0) < 0.05
    x = np.log(s / (1 - s)) + 3.2  # invert the score: s = expit(-3.2 + x)
    u = np.log(r / (1 - r)) + 3.0 - 1.2 * x  # r = expit(-3 + 1.2 x + u)
    # u is constant within each group and has sd 0.8
    first = np.r_[0, np.cumsum(sizes)[:-1]]
    np.testing.assert_allclose(u, np.repeat(u[first], sizes), atol=1e-8)
    assert abs(float(np.std(u[first])) - 0.8) < 0.02
    np.testing.assert_allclose(expit(-3.2 + x), s)


@pytest.mark.parametrize("key", ["a", "b", "d"])
def test_dev_cov_comes_from_the_oracle(key: str) -> None:
    gen = selective_generator(key)
    n_mc = 20_000 if key == "d" else 100_000
    o = gen.oracle(TAU, n_mc, np.random.default_rng(5))
    cov = gen.dev_cov(TAU, o)
    np.testing.assert_array_equal(cov, o.cov)
    assert np.all(np.diff(cov) >= 0) and cov[0] >= 0.0 and cov[-1] <= 1.0
    with pytest.raises(ValueError):
        gen.dev_cov(TAU[:-1], o)


# --- generator (c) -----------------------------------------------------------------------------


def test_seasonal_series_and_shift() -> None:
    g = SeasonalARShift()
    y = g.series(20_000, 10_000, np.random.default_rng(0))
    assert y.shape == (20_000,) and y.dtype == np.float64
    z = y - g.season(np.arange(20_000))
    innov = z[1:] - 0.6 * z[:-1]
    sd_pre, sd_post = innov[:9_998].std(), innov[10_000:].std()
    assert abs(sd_pre - 0.5) < 0.02 and abs(sd_post - 1.0) < 0.04


def test_stationary_variant_has_no_shift() -> None:
    """D20: the gated T1-forecast path runs the same law without the regime shift."""
    g = SeasonalARShift()
    st = g.stationary()
    assert g.has_shift and not st.has_shift
    assert st.shift_factor == 1.0 and st.name != g.name
    assert (st.phi, st.period, st.amplitude, st.noise_sd) == (g.phi, g.period, g.amplitude, 0.5)
    y1 = st.series(5_000, 1_000, np.random.default_rng(3))
    y2 = st.series(5_000, 5_000, np.random.default_rng(3))
    np.testing.assert_array_equal(y1, y2)  # the shift index is irrelevant without a shift
    z = y1 - st.season(np.arange(5_000))
    innov = z[1:] - 0.6 * z[:-1]
    assert abs(innov[2_000:].std() - 0.5) < 0.03


def test_seasonal_naive_forecast() -> None:
    g = SeasonalARShift()
    y = np.arange(200, dtype=float)
    for h in (1, 24, 25):
        L = g.lag(h)
        assert L % 24 == 0 and h <= L
        fs = g.forecast(y, np.array([100, 120]), h)
        np.testing.assert_array_equal(fs.preds, y[np.array([100, 120]) + h - L])
        np.testing.assert_array_equal(fs.targets, y[np.array([100, 120]) + h])
    with pytest.raises(ValueError):
        g.forecast(y, np.array([10]), 1)  # needs y[10 + 1 - 24]
    with pytest.raises(ValueError):
        g.forecast(y, np.array([199]), 1)


def test_exceed_scores_use_no_future_values() -> None:
    g = SeasonalARShift()
    rng = np.random.default_rng(2)
    y = g.series(2_000, 2_000, rng)
    origins = np.arange(g.min_origin(24), 1_500, 7)
    for h in (1, 24):
        s = g.exceed_scores(y, origins, h, 2.0)
        assert np.all((s >= 0) & (s <= 1))
        for o in origins[::40]:
            y2 = y.copy()
            y2[o + 1 :] = rng.standard_normal(y2.shape[0] - o - 1) * 50
            s2 = g.exceed_scores(y2, np.array([o]), h, 2.0)
            assert s2[0] == pytest.approx(s[np.searchsorted(origins, o)])


def test_exceed_scores_are_calibrated_before_the_shift() -> None:
    """The scorer knows the first regime: mean predicted exceedance ≈ observed rate."""
    g = SeasonalARShift()
    y = g.series(60_000, 60_000, np.random.default_rng(4))
    o = np.arange(g.min_origin(24), 59_000)
    for h in (1, 24):
        s = g.exceed_scores(y, o, h, 2.0)
        miss = g.forecast(y, o, h).abs_resid > 2.0
        assert abs(float(s.mean()) - float(miss.mean())) < 0.004
