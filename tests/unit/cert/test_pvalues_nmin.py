from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binom

from amx.cert import h1, n_min, p_binomial, p_hoeffding_bentkus, p_value


@pytest.mark.parametrize(
    ("alpha", "delta", "expected"),
    [
        # HANDOFF 7.4 table
        (0.005, 0.100, 460),
        (0.005, 0.025, 736),
        (0.01, 0.100, 230),
        (0.01, 0.025, 368),
        (0.02, 0.100, 114),
        (0.02, 0.025, 183),
        (0.05, 0.100, 45),
        (0.05, 0.025, 72),
        # 7.2 Bonferroni-over-grid figure (delta_j / G, G = 200)
        (0.01, 0.025 / 200, 895),
        # delta / (m * 3 calls), recomputed in the review
        (0.005, 0.1 / 12, 956),
        (0.01, 0.1 / 12, 477),
        (0.02, 0.1 / 12, 237),
        (0.05, 0.1 / 12, 94),
    ],
)
def test_n_min_table(alpha: float, delta: float, expected: int) -> None:
    n = n_min(alpha, delta)
    assert n == expected
    assert (1 - alpha) ** n <= delta < (1 - alpha) ** (n - 1)


def test_binomial_requires_integer_counts() -> None:
    with pytest.raises(TypeError, match="integer"):
        p_binomial(np.array([1.0]), np.array([10]), 0.1)


def test_binomial_regression_float_floor_c1() -> None:
    # n * mean can come out as 0.9999999999999999 for k = 1; flooring it would certify.
    n, alpha = 374, 0.01
    losses = np.zeros(n)
    losses[0] = 1.0
    p = float(p_binomial(int(losses.astype(np.int64).sum()), n, alpha))
    assert p == pytest.approx(0.1110, abs=5e-4)
    assert p > 0.025
    assert float(binom.cdf(0, n, alpha)) < 0.025  # what the floored value would have given


def test_binomial_edge_cases() -> None:
    assert float(p_binomial(0, 0, 0.1)) == 1.0
    assert float(p_binomial(5, 5, 0.1)) == 1.0
    with pytest.raises(ValueError):
        p_binomial(3, 2, 0.1)
    with pytest.raises(ValueError):
        p_binomial(1, 2, 1.0)


def _reference_hb(r_hat: float, n: int, alpha: float) -> float:
    """aangelopoulos/ltt core/bounds.py hb_p_value, re-derived (not vendored)."""

    def kl(a: float, b: float) -> float:
        return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))

    bentkus = math.e * binom.cdf(math.ceil(n * r_hat), n, alpha)
    hoeffding = math.exp(-n * kl(min(r_hat, alpha), alpha))
    return min(bentkus, hoeffding)


@pytest.mark.parametrize("seed", range(5))
def test_hb_matches_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(200):
        n = int(rng.integers(10, 5000))
        alpha = float(rng.uniform(0.002, 0.3))
        r_hat = float(rng.uniform(1e-4, alpha * 0.999))
        s = r_hat * n
        ours = float(p_hoeffding_bentkus(s, n, alpha))
        ref = min(1.0, _reference_hb(s / n, n, alpha))
        assert ours == pytest.approx(ref, rel=1e-9, abs=1e-15)


def test_hb_edge_cases() -> None:
    assert float(p_hoeffding_bentkus(0.0, 0, 0.1)) == 1.0
    assert float(p_hoeffding_bentkus(20.0, 100, 0.1)) == 1.0  # R_hat >= alpha
    n, alpha = 300, 0.02
    assert float(p_hoeffding_bentkus(0.0, n, alpha)) == pytest.approx((1 - alpha) ** n)
    assert float(h1(0.0, 0.1)) == pytest.approx(-math.log(0.9))


def test_dispatch() -> None:
    assert float(p_value(np.int64(2), 100, 0.05, binary=True)) == pytest.approx(
        binom.cdf(2, 100, 0.05)
    )
    assert float(p_value(2.0, 100, 0.05, binary=False)) >= float(
        p_value(np.int64(2), 100, 0.05, binary=True)
    )
