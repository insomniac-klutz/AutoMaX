from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from amx.cert import TauGrid, fixed_sequence_ltt, grid_stats, p_binomial, p_hoeffding_bentkus

GRID = TauGrid(size=40, low=1e-3).values


@settings(max_examples=200, deadline=None)
@given(st.integers(1, 3000), st.data())
def test_pvalues_monotone(n: int, data: st.DataObject) -> None:
    k = data.draw(st.integers(0, n))
    a1 = data.draw(st.floats(1e-4, 0.5))
    a2 = data.draw(st.floats(a1, 0.6))
    assert float(p_binomial(k, n, a2)) <= float(p_binomial(k, n, a1)) + 1e-12
    if k < n:
        assert float(p_binomial(k, n, a1)) <= float(p_binomial(k + 1, n, a1)) + 1e-12
    s = data.draw(st.floats(0, n))
    assert float(p_hoeffding_bentkus(s, n, a2)) <= float(p_hoeffding_bentkus(s, n, a1)) + 1e-12


@settings(max_examples=60, deadline=None)
@given(st.integers(0, 2**32 - 1), st.integers(200, 3000))
def test_bands_nest_and_tau_nondecreasing(seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, n)
    L = (rng.uniform(size=n) < rng.uniform(0, 0.3) * s).astype(float)
    alphas = np.sort(rng.choice(np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2]), 4, replace=False))
    dev_cov = np.clip(GRID * rng.uniform(0.5, 2), 0, 1)
    res = fixed_sequence_ltt(grid_stats(s, L, GRID, binary=True), alphas, 0.025, dev_cov)
    idx = [-1 if b.index is None else b.index for b in res.bands]
    assert idx == sorted(idx)
    for b in res.bands:
        if b.raw_index is not None:
            assert b.index is not None and b.index >= b.raw_index
