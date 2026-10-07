"""Monte Carlo validity of the p-values at the null boundary R = α (C5).

Each case draws ``REPS`` calibration samples whose true mean loss is exactly α and counts how
often H0: R ≥ α is rejected at level δ. Exact tests sit on the boundary, so the bound is
δ + 3 MC standard errors, not δ. Two-point losses {0, c} catch floor-instead-of-ceil and
missing-e bugs; Bernoulli losses catch float-count bugs; Beta losses check conservativeness.
"""

from __future__ import annotations

import numpy as np
import pytest

from amx.cert import n_min, p_binomial, p_hoeffding_bentkus

REPS = 40_000
DELTAS = (0.1, 0.025, 0.1 / 12)
ALPHAS = (0.01, 0.05)


def _ceiling(delta: float, reps: int = REPS) -> float:
    return delta + 3.0 * np.sqrt(delta * (1 - delta) / reps)


def _ns(alpha: float, delta: float) -> list[int]:
    return sorted({n_min(alpha, delta), 400, 2000})


@pytest.mark.slow
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("delta", DELTAS)
def test_binomial_bernoulli_valid_and_tight(alpha: float, delta: float) -> None:
    rng = np.random.default_rng(1)
    for n in _ns(alpha, delta):
        k = rng.binomial(n, alpha, size=REPS).astype(np.int64)
        rate = float(np.mean(p_binomial(k, np.full(REPS, n), alpha) <= delta))
        assert rate <= _ceiling(delta), (n, rate)
        if n == 2000:
            assert rate >= delta / 10, f"binomial test too conservative: {rate}"


@pytest.mark.slow
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("delta", DELTAS)
@pytest.mark.parametrize("c", [0.5, 0.99])
def test_hb_two_point_valid(alpha: float, delta: float, c: float) -> None:
    rng = np.random.default_rng(2)
    for n in _ns(alpha, delta):
        sums = c * rng.binomial(n, alpha / c, size=REPS)
        rate = float(np.mean(p_hoeffding_bentkus(sums, np.full(REPS, n), alpha) <= delta))
        assert rate <= _ceiling(delta), (n, c, rate)


@pytest.mark.slow
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("delta", DELTAS)
def test_hb_bernoulli_valid(alpha: float, delta: float) -> None:
    rng = np.random.default_rng(3)
    for n in _ns(alpha, delta):
        sums = rng.binomial(n, alpha, size=REPS).astype(np.float64)
        rate = float(np.mean(p_hoeffding_bentkus(sums, np.full(REPS, n), alpha) <= delta))
        assert rate <= _ceiling(delta), (n, rate)


@pytest.mark.slow
@pytest.mark.parametrize("alpha", ALPHAS)
def test_hb_beta_valid(alpha: float) -> None:
    rng = np.random.default_rng(4)
    reps = 10_000
    kappa = 2.0
    for delta in DELTAS:
        for n in _ns(alpha, delta):
            sums = np.zeros(reps)
            for start in range(0, reps, 1000):
                block = rng.beta(alpha * kappa, (1 - alpha) * kappa, size=(1000, n))
                sums[start : start + 1000] = block.sum(axis=1)
            rate = float(np.mean(p_hoeffding_bentkus(sums, np.full(reps, n), alpha) <= delta))
            assert rate <= _ceiling(delta, reps), (n, delta, rate)
