"""Monte Carlo check of the oracle's delta-method SE (C6, CLAUDE.md Monte Carlo rule).

The C6 indeterminacy rule trusts :attr:`amx.sim.OracleCurve.risk_se`: a selected τ̂ with
|R(τ̂) − α| < 4 SE is not judged. The SE is the delta-method value
SE = sqrt(Σ c (v − R̂)²) / Σ c of the Rao–Blackwell ratio estimate R̂ = Σ c v / Σ c, with v the
known conditional risk r(x) for generators (a) and (b) and the per-group mean of r over the
committed units for the group-level generator (d).

Here each oracle is recomputed at a small ``n_mc`` over many independent, pre-registered
seeds. At grid points with enough committed mass the empirical sd of ``risk[g]`` across seeds,
divided by the mean reported ``risk_se[g]``, must lie in [0.85, 1.15]. With 600 seeds the
relative Monte Carlo SE of an sd is about 1/sqrt(2·599) ≈ 2.9%, so the band is about ±5 MC SE.
"""

from __future__ import annotations

import numpy as np
import pytest

from amx.cert import TauGrid
from amx.sim import selective_generator

TAU = TauGrid().values
ORACLE_SE_SEED = 160_001
"""Pre-registered base seed of the SE Monte Carlo (one child sequence per oracle run)."""

N_SEEDS = 600
SMALL_N_MC = {"a": 20_000, "b": 20_000, "d": 5_000}
"""Small oracle sizes (inputs for (a) and (b), groups for (d)) so the SE is visible."""

COVERAGE_TARGETS = (0.1, 0.3, 0.6, 0.9)
MIN_COMMITTED = 500
"""A probed grid point needs at least this many committed units (groups) per oracle run."""

RATIO_BAND = (0.85, 1.15)


@pytest.mark.slow
@pytest.mark.parametrize("key", ["a", "b", "d"])
def test_oracle_delta_method_se_matches_monte_carlo_sd(key: str) -> None:
    gen = selective_generator(key)
    n_mc = SMALL_N_MC[key]
    children = np.random.SeedSequence([ORACLE_SE_SEED, ord(key)]).spawn(N_SEEDS)
    risk = np.empty((N_SEEDS, TAU.shape[0]))
    se = np.empty_like(risk)
    cov = np.empty_like(risk)
    for i, child in enumerate(children):
        o = gen.oracle(TAU, n_mc, np.random.default_rng(child))
        risk[i], se[i], cov[i] = o.risk, o.risk_se, o.cov
    mean_cov = cov.mean(axis=0)
    probes = sorted({int(np.argmin(np.abs(mean_cov - t))) for t in COVERAGE_TARGETS})
    checked: list[tuple[int, float, float, float]] = []
    for g in probes:
        if np.min(cov[:, g]) * n_mc < MIN_COMMITTED or not np.all(np.isfinite(risk[:, g])):
            continue
        sd_emp = float(np.std(risk[:, g], ddof=1))
        se_mean = float(np.mean(se[:, g]))
        ratio = sd_emp / se_mean
        checked.append((g, float(mean_cov[g]), float(np.mean(risk[:, g])), ratio))
        assert RATIO_BAND[0] <= ratio <= RATIO_BAND[1], (key, g, mean_cov[g], sd_emp, se_mean)
    print(f"{gen.name}: (grid index, coverage, mean risk, sd/SE) {checked}")
    assert len(checked) >= 3, (key, checked)
