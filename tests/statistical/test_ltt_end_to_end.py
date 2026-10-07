"""End-to-end validity of fixed-sequence LTT on a model with exactly known risk (C2, C5).

Scores s ~ U(0, 1); loss ~ Bernoulli(s/2) given s. Then cov(τ) = τ and R(τ) = τ/4 exactly, so
a band α is truly met for τ ≤ 4α. The test gates raw per-band rates at δ_j and the family-wise
rate after monotonisation at δ.
"""

from __future__ import annotations

import numpy as np
import pytest

from amx.cert import DeltaBudget, TauGrid, fixed_sequence_ltt, grid_stats

REPS = 2000


@pytest.mark.slow
@pytest.mark.parametrize("n_calib", [500, 2000, 10_000])
def test_fixed_sequence_ltt_validity(n_calib: int) -> None:
    rng = np.random.default_rng(n_calib)
    grid = TauGrid()
    tau = grid.values
    alphas = np.array([0.005, 0.01, 0.02, 0.05])
    budget = DeltaBudget(0.1, len(alphas))
    dj = budget.delta_per_band
    raw_viol = np.zeros(len(alphas))
    any_viol = 0
    certified = np.zeros(len(alphas))
    for _ in range(REPS):
        s = rng.uniform(0, 1, n_calib)
        L = (rng.uniform(size=n_calib) < s / 2).astype(np.float64)
        res = fixed_sequence_ltt(grid_stats(s, L, tau, binary=True), alphas, dj, tau)
        bad = False
        for j, b in enumerate(res.bands):
            if b.raw_index is not None:
                certified[j] += 1
                raw_viol[j] += tau[b.raw_index] / 4 > alphas[j]
            if b.index is not None and tau[b.index] / 4 > alphas[j]:
                bad = True
        any_viol += bad
    slack = 3 * np.sqrt(dj * (1 - dj) / REPS)
    assert np.all(raw_viol / REPS <= dj + slack), raw_viol / REPS
    fwer_slack = 3 * np.sqrt(budget.delta * (1 - budget.delta) / REPS)
    assert any_viol / REPS <= budget.delta + fwer_slack
    if n_calib == 10_000:
        assert certified[-1] > 0, "the loosest band should certify at n_calib = 10000"
