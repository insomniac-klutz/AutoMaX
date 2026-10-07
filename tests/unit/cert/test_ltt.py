from __future__ import annotations

import numpy as np
import pytest

from amx.cert import (
    BandStatus,
    DeltaBudget,
    StopReason,
    TauGrid,
    bonferroni_diagnostic,
    fixed_sequence_ltt,
    grid_stats,
    monotonize,
    n_min,
    p_value,
)
from amx.spec.enums import CallPolicy

GRID = TauGrid(size=60, low=1e-3, high=1.0).values


def naive_certify(s, L, alphas, dj, grid, dev_cov, factor=1.25, groups=None, binary=True):
    """Appendix A.2 written out literally, with C1 (integer k), C2/C3 (statuses)."""
    n_cal = len(s) if groups is None else len(set(groups))
    raw = []
    for a in alphas:
        need = n_min(a, dj)
        start = next((g for g in range(len(grid)) if dev_cov[g] * n_cal >= factor * need), None)
        last, reason = None, "infeasible_on_dev" if start is None else "end_of_grid"
        if start is not None:
            for g in range(start, len(grid)):
                mask = s <= grid[g]
                col = L if L.ndim == 1 else L[:, g]
                if groups is None:
                    n = int(mask.sum())
                    stat = int(col[mask].astype(int).sum()) if binary else float(col[mask].sum())
                    bin_ = binary
                else:
                    gs = {}
                    for gi, m, v in zip(groups, mask, col, strict=True):
                        if m:
                            gs.setdefault(gi, []).append(v)
                    n = len(gs)
                    stat = float(sum(np.mean(v) for v in gs.values()))
                    bin_ = False
                if n < need:
                    reason = "sample_size_limited"
                    break
                if float(p_value(stat, n, a, binary=bin_)) <= dj:
                    last = g
                else:
                    reason = "risk_limited"
                    break
        raw.append((last, reason))
    out, best = [], None
    for last, reason in raw:
        if last is not None:
            best = last if best is None else max(best, last)
        out.append((last, reason, best))
    return out


def _case(seed: int, n: int = 3000, two_d: bool = False):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, n) ** 2
    p_err = 0.2 * s
    if two_d:
        L = (rng.uniform(size=(n, len(GRID))) < p_err[:, None]).astype(float)
    else:
        L = (rng.uniform(size=n) < p_err).astype(float)
    dev_cov = np.searchsorted(np.sort(rng.uniform(0, 1, 5000) ** 2), GRID, side="right") / 5000
    return s, L, dev_cov


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("two_d", [False, True])
def test_vectorized_matches_naive(seed: int, two_d: bool) -> None:
    s, L, dev_cov = _case(seed, two_d=two_d)
    alphas, dj = [0.01, 0.02, 0.05, 0.1], 0.025
    res = fixed_sequence_ltt(grid_stats(s, L, GRID, binary=True), alphas, dj, dev_cov)
    ref = naive_certify(s, L, alphas, dj, GRID, dev_cov)
    for b, (last, reason, best) in zip(res.bands, ref, strict=True):
        assert b.raw_index == last
        assert b.stop_reason.value == reason
        assert b.index == best


@pytest.mark.parametrize("seed", range(6))
def test_group_mode_matches_naive(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    n = 1500
    groups = rng.integers(0, 300, n)
    s = rng.uniform(0, 1, n)
    L = (rng.uniform(size=n) < 0.15 * s).astype(float)
    dev_cov = np.clip(GRID * 1.5, 0, 1)
    alphas, dj = [0.02, 0.05, 0.1], 0.05
    st = grid_stats(s, L, GRID, binary=True, groups=groups)
    assert not st.binary and st.estimand == "group_weighted"
    assert st.n_total_indep == len(np.unique(groups))
    res = fixed_sequence_ltt(st, alphas, dj, dev_cov)
    ref = naive_certify(s, L, alphas, dj, GRID, dev_cov, groups=groups.tolist(), binary=False)
    for b, (last, reason, best) in zip(res.bands, ref, strict=True):
        assert (b.raw_index, b.stop_reason.value, b.index) == (last, reason, best)


def test_binary_declared_but_fractional_rejected() -> None:
    with pytest.raises(ValueError, match="C1"):
        grid_stats([0.1, 0.2], [0.0, 0.5], GRID, binary=True)


def test_inherited_status_and_nesting() -> None:
    # Band 0 certifies far out; band 1 starts later and fails immediately -> inherits band 0.
    n = 4000
    s = np.linspace(0, 1, n)
    L = np.zeros(n)
    L[(s > 0.5) & (np.arange(n) % 10 == 0)] = 1.0  # risk jumps above 0.5
    st = grid_stats(s, L, GRID, binary=True)
    dev_cov = np.clip(GRID, 0, 1)
    cov_override = dev_cov.copy()
    res = fixed_sequence_ltt(st, [0.01, 0.02], 0.05, cov_override)
    b0, b1 = res.bands
    assert b0.status is BandStatus.CERTIFIED
    assert b1.index is not None and b1.index >= b0.index
    assert res.tau_hat(1) >= res.tau_hat(0)


def test_monotonize_statuses() -> None:
    out = monotonize([5, 3, None, 7, 7])
    assert out == [
        (5, BandStatus.CERTIFIED, 0),
        (5, BandStatus.INHERITED, 0),
        (5, BandStatus.INHERITED, 0),
        (7, BandStatus.CERTIFIED, 3),
        (7, BandStatus.CERTIFIED, 4),
    ]
    assert monotonize([None, None]) == [(None, BandStatus.UNCERTIFIED, None)] * 2


def test_infeasible_on_dev() -> None:
    rng = np.random.default_rng(7)
    s = rng.uniform(0, 1, 5000)
    L = (rng.uniform(size=5000) < 0.004).astype(float)
    st = grid_stats(s, L, GRID, binary=True)
    res = fixed_sequence_ltt(st, [0.02, 0.05], 0.05, np.zeros_like(GRID))
    assert all(b.stop_reason is StopReason.INFEASIBLE_ON_DEV for b in res.bands)
    assert all(b.status is BandStatus.UNCERTIFIED for b in res.bands)


def test_bonferroni_diagnostic_runs() -> None:
    s, L, _ = _case(3)
    st = grid_stats(s, L, GRID, binary=True)
    diag = bonferroni_diagnostic(st, [0.02, 0.05], 0.025)
    assert diag[0]["per_test_level"] == pytest.approx(0.025 / len(GRID))


def test_budget_policies() -> None:
    assert DeltaBudget(0.1, 4).delta_per_band == pytest.approx(0.025)
    assert DeltaBudget(0.1, 4, CallPolicy.PREREGISTERED, k=3).delta_per_band == pytest.approx(
        0.1 / 12
    )
    sliced = DeltaBudget(0.1, 4, CallPolicy.SLICED, k=3, release_fixed_in_advance=False)
    assert sliced.delta_per_band == pytest.approx(0.1 / 12)
    assert DeltaBudget(0.1, 4).simultaneous_level == 0.1
    with pytest.raises(ValueError):
        DeltaBudget(0.1, 4, CallPolicy.SINGLE, k=2)


def test_inherited_through_the_walk_when_a_looser_band_is_sample_size_limited() -> None:
    """The looser band starts earlier (smaller n_min) where calib has too few units; the
    tighter band starts later and certifies; monotonisation lends its τ̂ to the looser band."""
    s = np.r_[np.full(100, 0.02), np.linspace(0.5, 1.0, 1900)]
    L = np.r_[np.zeros(100), (np.arange(1900) % 10 == 0).astype(float)]
    tau = TauGrid(size=200).values
    res = fixed_sequence_ltt(grid_stats(s, L, tau, binary=True), [0.05, 0.2], 0.05, tau)
    tight, loose = res.bands
    assert tight.status is BandStatus.CERTIFIED and tight.raw_index is not None
    assert loose.stop_reason is StopReason.SAMPLE_SIZE_LIMITED and loose.raw_index is None
    assert loose.status is BandStatus.INHERITED and loose.source_band == 0
    assert loose.index == tight.raw_index
