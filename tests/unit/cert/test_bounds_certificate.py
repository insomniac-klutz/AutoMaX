from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from amx.cert import (
    Certificate,
    DeltaBudget,
    GuaranteeType,
    TauGrid,
    build_certificate,
    cp_lower,
    cp_upper,
    dkw_halfwidth,
    fixed_sequence_ltt,
    grid_stats,
    guarantee_type_for,
    hb_lower,
    hb_upper,
    slice_risks,
)
from amx.spec.enums import Family, Policy, Regime


def test_cp_bounds_known_values() -> None:
    assert cp_upper(0, 299) == pytest.approx(1 - 0.05 ** (1 / 299), rel=1e-6)
    assert cp_lower(0, 100) == 0.0
    assert cp_upper(10, 10) == 1.0
    assert cp_lower(5, 100) < 0.05 < cp_upper(5, 100)


def test_hb_bounds_bracket_and_dominate_cp() -> None:
    for k, n in [(0, 500), (3, 400), (40, 1000)]:
        u = hb_upper(float(k), n)
        assert u >= k / n
        assert u >= cp_upper(k, n) - 1e-9
        assert hb_lower(float(k), n) <= k / n


def test_dkw() -> None:
    assert dkw_halfwidth(1000) == pytest.approx(np.sqrt(np.log(40) / 2000))


def test_slice_flags() -> None:
    labels = np.array(["a"] * 2000 + ["b"] * 30)
    losses = np.r_[np.zeros(2000), np.ones(3), np.zeros(27)]
    out = slice_risks("grp", labels, losses, np.ones(2030, bool), alpha=0.01, binary=True)
    by = {s.value: s for s in out}
    assert by["a"].flag is None
    assert by["b"].flag == "insufficient_n"


def test_guarantee_type_selection() -> None:
    assert guarantee_type_for(Regime.IID, Family.CLASSIFICATION) is GuaranteeType.PAC_HIGH_PROB
    assert guarantee_type_for(Regime.TEMPORAL, Family.REGRESSION) is GuaranteeType.HOLDOUT_EMPIRICAL
    assert (
        guarantee_type_for(Regime.TEMPORAL, Family.FORECASTING) is GuaranteeType.HOLDOUT_EMPIRICAL
    )
    with pytest.raises(ValueError):
        guarantee_type_for(Regime.AUTO, Family.CLASSIFICATION)


def test_certificate_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 6000
    s = rng.uniform(0, 1, n)
    L = (rng.uniform(size=n) < 0.05 * s).astype(float)
    grid = TauGrid(size=80)
    st = grid_stats(s, L, grid.values, binary=True)
    budget = DeltaBudget(0.1, 3)
    res = fixed_sequence_ltt(
        st, [0.005, 0.01, 0.03], budget.delta_per_band, np.clip(grid.values, 0, 1)
    )
    labels = np.where(s > 0.5, "hi", "lo")
    slices = []
    for b in res.bands:
        committed = s <= (grid.values[b.index] if b.index is not None else -1)
        slices.append(slice_risks("half", labels, L, committed, alpha=b.alpha, binary=True))
    cert = build_certificate(
        res,
        policies=[Policy.AUTO, Policy.AUTO, Policy.AUDIT],
        budget=budget,
        grid=grid,
        regime=Regime.IID,
        guarantee_type=GuaranteeType.PAC_HIGH_PROB,
        run_id="r1",
        taskspec_hash="sha256:x",
        artifact_hash="sha256:y",
        amx_version="0.1",
        slices=slices,
    )
    path = cert.write(tmp_path / "certificate.json")
    again = Certificate.read(path)
    assert again == cert
    assert again.guarantee.delta_per_band == pytest.approx(0.1 / 3)
    assert again.guarantee.p_value_family == "binomial"
    assert len(again.frontier_descriptive) == 80
    taus = [b.tau_hat if b.tau_hat is not None else -1 for b in again.bands]
    assert taus == sorted(taus)
    assert "risk is measured against the provided gold labels" in again.guarantee.assumptions
