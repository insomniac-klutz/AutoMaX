"""Rao–Blackwell oracles (C6): agreement with labelled Monte Carlo, group sums, semantics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from amx.cert import TauGrid, grid_stats
from amx.sim import ClusteredUnits, GaussianMixture, HeteroscedasticRegression, OracleCurve
from amx.sim.oracle import CurveSums, group_sums, unit_sums

TAU = TauGrid().values
N_LABELLED = 1_000_000


def _probe_indices(cov: np.ndarray, targets: tuple[float, ...]) -> list[int]:
    return sorted({int(np.argmin(np.abs(cov - t))) for t in targets})


@pytest.mark.parametrize("gen", [GaussianMixture(), HeteroscedasticRegression()])
def test_rao_blackwell_matches_labelled_monte_carlo(
    gen: GaussianMixture | HeteroscedasticRegression,
) -> None:
    """R(τ) from known r(x) agrees with 1e6 fresh labelled samples within 4 combined SE."""
    oracle = gen.oracle(TAU, N_LABELLED, np.random.default_rng(11))
    batch = gen.sample(N_LABELLED, np.random.default_rng(12))
    stats = grid_stats(batch.scores, batch.losses, TAU, binary=True)
    emp = stats.risk()
    for g in _probe_indices(oracle.cov, (0.02, 0.1, 0.3, 0.6, 0.95)):
        n_g = int(stats.n_units[g])
        assert n_g > 5_000
        se_emp = math.sqrt(emp[g] * (1 - emp[g]) / n_g)
        se = math.hypot(se_emp, float(oracle.risk_se[g]))
        assert abs(emp[g] - oracle.risk[g]) < 4 * se, (g, emp[g], oracle.risk[g], se)
        # Rao–Blackwellisation never increases the variance of the risk estimate
        assert oracle.risk_se[g] <= se_emp * 1.01
        # coverage agrees too (binomial SE)
        p = oracle.cov[g]
        assert abs(stats.coverage()[g] - p) < 4 * math.sqrt(2 * p * (1 - p) / N_LABELLED)


def test_group_oracle_matches_labelled_group_estimate() -> None:
    gen = ClusteredUnits()
    oracle = gen.oracle(TAU, 200_000, np.random.default_rng(21))
    assert oracle.estimand == "group_weighted"
    batch = gen.sample(200_000, np.random.default_rng(22))
    assert batch.groups is not None
    for g in _probe_indices(oracle.cov, (0.1, 0.5, 0.9)):
        mask = batch.scores <= TAU[g]
        cnt = np.bincount(batch.groups[mask], minlength=200_000)
        lsum = np.bincount(batch.groups[mask], weights=batch.losses[mask], minlength=200_000)
        has = cnt > 0
        lg = lsum[has] / cnt[has]
        emp, se_emp = float(lg.mean()), float(lg.std() / math.sqrt(lg.size))
        se = math.hypot(se_emp, float(oracle.risk_se[g]))
        assert abs(emp - oracle.risk[g]) < 4 * se, (g, emp, oracle.risk[g], se)
        assert abs(has.mean() - oracle.cov[g]) < 4 * math.sqrt(2 * 0.25 / 200_000)


def test_group_oracle_reports_unit_weighted_curve() -> None:
    o = ClusteredUnits().oracle(TAU, 20_000, np.random.default_rng(0))
    assert o.unit_cov is not None and o.unit_risk is not None and o.unit_risk_se is not None
    # a group counts as covered once any unit commits: group coverage ≥ unit coverage
    assert np.all(o.cov >= o.unit_cov - 1e-12)
    assert np.all(np.diff(o.cov) >= 0) and np.all(np.diff(o.unit_cov) >= 0)


def test_group_sums_against_brute_force() -> None:
    rng = np.random.default_rng(7)
    n_groups = 300
    sizes = 1 + rng.poisson(3, n_groups)
    gid = np.repeat(np.arange(n_groups), sizes)
    perm = rng.permutation(gid.size)  # group ids need not be contiguous
    gid = gid[perm]
    s = rng.uniform(size=gid.size)
    r = rng.uniform(size=gid.size)
    tau = np.array([0.05, 0.2, 0.5, 0.9, 1.0])
    out = group_sums(s, r, gid, tau)
    for j, t in enumerate(tau):
        vals = []
        for g in range(n_groups):
            m = (gid == g) & (s <= t)
            if m.any():
                vals.append(r[m].mean())
        v = np.array(vals)
        assert out.count[j] == pytest.approx(v.size)
        assert out.total[j] == pytest.approx(v.sum())
        assert out.total_sq[j] == pytest.approx(np.sum(v**2))


def test_unit_sums_and_ratio_se() -> None:
    rng = np.random.default_rng(1)
    s, r = rng.uniform(size=1000), rng.uniform(size=1000)
    tau = np.array([0.0, 0.3, 1.0])
    sums = unit_sums(s, r, tau)
    mean, se = sums.ratio()
    assert sums.count[0] == 0 and math.isnan(mean[0]) and math.isnan(se[0])
    m = s <= 0.3
    assert mean[1] == pytest.approx(r[m].mean())
    assert se[1] == pytest.approx(math.sqrt(np.sum((r[m] - r[m].mean()) ** 2)) / m.sum())
    doubled = sums + sums
    np.testing.assert_allclose(doubled.ratio()[0][1:], mean[1:])
    assert CurveSums.zeros(3).count.sum() == 0


def _curve(risk: list[float], se: list[float], cov: list[float]) -> OracleCurve:
    n = len(risk)
    return OracleCurve(
        tau=np.linspace(0.1, 1, n),
        cov=np.array(cov),
        risk=np.array(risk),
        risk_se=np.array(se),
        method="test",
        n_mc=1,
    )


def test_oracle_curve_semantics() -> None:
    o = _curve([math.nan, 0.004, 0.009, 0.02], [math.nan, 1e-4, 1e-3, 1e-4], [0, 0.1, 0.3, 0.6])
    assert not o.indeterminate(0, 0.01) and not o.violates(0, 0.01)  # nothing committed
    assert not o.indeterminate(1, 0.01) and not o.violates(1, 0.01)
    assert o.indeterminate(2, 0.01)  # |0.009 − 0.01| < 4 · 1e-3
    assert not o.indeterminate(2, 0.01, k=0.5)
    assert o.violates(3, 0.01) and not o.indeterminate(3, 0.01)
    assert o.coverage_at(0.01) == pytest.approx(0.3)
    assert o.coverage_at(0.001) == 0.0
    assert o.coverage_at(0.05) == pytest.approx(0.6)
    # the grid point that attains the oracle coverage (None when no point has R ≤ α)
    assert o.coverage_index(0.01) == 2 and o.coverage_index(0.05) == 3
    assert o.coverage_index(0.001) is None
    # ties in coverage resolve to the most liberal τ, as the fixed-sequence walk would
    tied = _curve([0.004, 0.005, 0.006, 0.02], [1e-4] * 4, [0.1, 0.3, 0.3, 0.6])
    assert tied.coverage_index(0.01) == 2
    with pytest.raises(ValueError):
        o.check_grid(np.linspace(0.1, 1, 5))
