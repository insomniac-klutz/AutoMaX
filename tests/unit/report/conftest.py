"""A fixture certificate built with amx.cert on synthetic data (fixed seed).

Scores are uniform; units with s < 0.01 lose half the time, all others 0.1% of the time.
``dev_cov`` stalls between τ = 0.03 and τ = 0.9, so the tight band starts late and fails, the
middle band starts late and certifies to the end of the grid, and the loose band starts early,
meets the high-loss head and stops: it inherits the middle band's τ̂ (C3). Every band status
and both slice flags occur.
"""

from __future__ import annotations

import numpy as np
import pytest

from amx.cert import (
    BandStatus,
    Certificate,
    DeltaBudget,
    GuaranteeType,
    TauGrid,
    build_certificate,
    fixed_sequence_ltt,
    grid_stats,
    slice_risks,
)
from amx.cert.certificate import Interval, RiskEstimate
from amx.spec.enums import Policy, Regime

SEED = 20261007
ALPHAS = (0.001, 0.02, 0.05)


def make_certificate(*, grouped: bool = False) -> Certificate:
    rng = np.random.default_rng(SEED)
    n = 6000
    s = rng.uniform(0.0, 1.0, n)
    losses = (rng.uniform(size=n) < np.where(s < 0.01, 0.5, 0.001)).astype(np.float64)
    groups = rng.integers(0, 2500, n).astype(str) if grouped else None
    grid = TauGrid(size=120)
    tau = grid.values
    dev_cov = np.where(tau < 0.03, tau, np.where(tau < 0.9, 0.03, tau))
    budget = DeltaBudget(0.1, len(ALPHAS))
    stats = grid_stats(s, losses, tau, binary=not grouped, groups=groups)
    result = fixed_sequence_ltt(stats, ALPHAS, budget.delta_per_band, dev_cov)
    part = np.where(s < 0.02, "head", np.where(s < 0.5, "mid", "tail"))
    part[:15] = "rare"
    slices = []
    for b in result.bands:
        committed = s <= (tau[b.index] if b.index is not None else -1.0)
        slices.append(
            slice_risks("part", part, losses, committed, alpha=b.alpha, binary=not grouped)
        )
    return build_certificate(
        result,
        policies=[Policy.AUTO, Policy.AUTO, Policy.AUDIT],
        budget=budget,
        grid=grid,
        regime=Regime.GROUPED if grouped else Regime.IID,
        guarantee_type=GuaranteeType.PAC_HIGH_PROB,
        run_id="run-fixture",
        taskspec_hash="sha256:" + "ab12" * 16,
        artifact_hash="sha256:" + "cd34" * 16,
        amx_version="0.1.0.dev0",
        group_column="cluster" if grouped else None,
        slices=slices,
    )


def with_type(cert: Certificate, gtype: GuaranteeType) -> Certificate:
    return cert.model_copy(update={"guarantee": cert.guarantee.model_copy(update={"type": gtype})})


def with_sealed(cert: Certificate) -> Certificate:
    bands = [
        b.model_copy(
            update={
                "risk_sealed": RiskEstimate(est=0.0071, upper95=0.0123, bound="clopper_pearson"),
                "coverage_sealed": Interval(est=0.881, ci95=(0.862, 0.9), method="dkw"),
            }
        )
        for b in cert.bands
    ]
    guarantee = cert.guarantee.model_copy(update={"sealed_n": 5987})
    return cert.model_copy(update={"bands": bands, "guarantee": guarantee})


@pytest.fixture(scope="session")
def cert() -> Certificate:
    c = make_certificate()
    assert [b.status for b in c.bands] == [
        BandStatus.UNCERTIFIED,
        BandStatus.CERTIFIED,
        BandStatus.INHERITED,
    ]
    return c


@pytest.fixture(scope="session")
def grouped_cert() -> Certificate:
    return make_certificate(grouped=True)
