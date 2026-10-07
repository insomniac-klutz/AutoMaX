"""Fixed-sequence Learn-then-Test along τ (HANDOFF 7.2, Appendix A.2, corrected by C1-C3).

The commit rule is ``s(x) ≤ τ``. For every grid τ_g the certifier needs the number of committed
independence units and the sum of their losses. Unit mode uses each committed unit's loss;
group mode (``independence_unit: group:<col>``) uses, per group with at least one committed
unit, the mean loss over its committed units, so the estimand is the group-weighted selective
risk and only the Hoeffding–Bentkus p-value is valid (OQ Q2).

Validity needs the scores, the per-τ answers and ``dev_cov`` to be fixed before calibration
data is seen. This module only computes; the warden enforces where the inputs come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amx.cert.nmin import n_min
from amx.cert.pvalues import is_binary_losses, p_value

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Estimand = Literal["unit_weighted", "group_weighted"]


class StopReason(StrEnum):
    END_OF_GRID = "end_of_grid"
    RISK_LIMITED = "risk_limited"
    SAMPLE_SIZE_LIMITED = "sample_size_limited"
    INFEASIBLE_ON_DEV = "infeasible_on_dev"


class BandStatus(StrEnum):
    CERTIFIED = "certified"
    INHERITED = "inherited"
    UNCERTIFIED = "uncertified"


@dataclass(frozen=True)
class GridStats:
    """Committed counts and loss sums at every grid point of one calibration sample."""

    tau: FloatArray
    n_units: IntArray
    n_indep: IntArray
    loss_sum: FloatArray
    unit_loss_sum: FloatArray
    k_int: IntArray | None
    binary: bool
    estimand: Estimand
    n_total_units: int
    n_total_indep: int

    @property
    def size(self) -> int:
        return int(self.tau.shape[0])

    def risk(self) -> FloatArray:
        """Empirical selective risk in the certified estimand (NaN where nothing commits)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n_indep > 0, self.loss_sum / np.maximum(self.n_indep, 1), np.nan)

    def unit_risk(self) -> FloatArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(
                self.n_units > 0, self.unit_loss_sum / np.maximum(self.n_units, 1), np.nan
            )

    def coverage(self) -> FloatArray:
        return np.asarray(self.n_units / max(self.n_total_units, 1), dtype=np.float64)

    def statistic(self) -> FloatArray | IntArray:
        """What the p-value consumes: the integer loss count (binary) or the loss sum."""
        if self.binary:
            assert self.k_int is not None
            return self.k_int
        return self.loss_sum


def _factorize(groups: ArrayLike) -> tuple[IntArray, int]:
    _, inverse = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)
    inv = inverse.astype(np.int64).reshape(-1)
    return inv, int(np.max(inv)) + 1 if inv.size else 0


def grid_stats(
    scores: ArrayLike,
    losses: ArrayLike,
    tau: ArrayLike,
    *,
    binary: bool,
    groups: ArrayLike | None = None,
) -> GridStats:
    """Committed counts and loss sums along the grid.

    ``losses`` is (n,) when the committed answer does not depend on τ, or (n, G) when it does
    (a cascade answers from the first stage that commits at τ). ``binary`` comes from the
    loss declaration; it is checked against the data before the binomial path is used (C1).
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    t = np.asarray(tau, dtype=np.float64).reshape(-1)
    L = np.asarray(losses, dtype=np.float64)
    n = s.shape[0]
    if not np.all(np.isfinite(s)):
        raise ValueError("commit scores must be finite")
    if np.any(np.diff(t) <= 0):
        raise ValueError("the τ grid must be strictly increasing")
    if L.shape not in ((n,), (n, t.shape[0])):
        raise ValueError(f"losses must have shape ({n},) or ({n}, {t.shape[0]}), got {L.shape}")
    if L.size and (float(np.min(L)) < 0.0 or float(np.max(L)) > 1.0):
        raise ValueError("losses must lie in [0, 1]")
    if binary and not is_binary_losses(L):
        raise ValueError("loss declared binary but calibration losses are not all 0 or 1 (C1)")

    order = np.argsort(s, kind="stable")
    n_units = np.searchsorted(s[order], t, side="right").astype(np.int64)
    G = t.shape[0]

    if groups is None:
        Ls = L[order]
        if L.ndim == 1:
            if binary:
                cum_i = np.concatenate([[0], np.cumsum(Ls.astype(np.int64))])
                k_vals: IntArray = cum_i[n_units].astype(np.int64)
                k_int: IntArray | None = k_vals
                loss_sum = k_vals.astype(np.float64)
            else:
                cum = np.concatenate([[0.0], np.cumsum(Ls)])
                loss_sum, k_int = cum[n_units], None
        else:
            cols = np.arange(G)
            if binary:
                C_i = np.vstack([np.zeros((1, G), np.int64), np.cumsum(Ls.astype(np.int64), 0)])
                k_vals = C_i[n_units, cols].astype(np.int64)
                k_int = k_vals
                loss_sum = k_vals.astype(np.float64)
            else:
                C = np.vstack([np.zeros((1, G)), np.cumsum(Ls, axis=0)])
                loss_sum, k_int = C[n_units, cols], None
        return GridStats(
            tau=t,
            n_units=n_units,
            n_indep=n_units.copy(),
            loss_sum=np.asarray(loss_sum, dtype=np.float64),
            unit_loss_sum=np.asarray(loss_sum, dtype=np.float64),
            k_int=k_int,
            binary=binary,
            estimand="unit_weighted",
            n_total_units=n,
            n_total_indep=n,
        )

    gid, n_groups = _factorize(groups)
    if gid.shape[0] != n:
        raise ValueError("groups must have one entry per unit")
    n_indep = np.zeros(G, dtype=np.int64)
    g_loss = np.zeros(G, dtype=np.float64)
    u_loss = np.zeros(G, dtype=np.float64)
    for g in range(G):
        idx = order[: n_units[g]]
        if idx.size == 0:
            continue
        col = L[idx] if L.ndim == 1 else L[idx, g]
        cnt = np.bincount(gid[idx], minlength=n_groups)
        lsum = np.bincount(gid[idx], weights=col, minlength=n_groups)
        has = cnt > 0
        n_indep[g] = int(has.sum())
        g_loss[g] = float(np.sum(lsum[has] / cnt[has]))
        u_loss[g] = float(lsum.sum())
    return GridStats(
        tau=t,
        n_units=n_units,
        n_indep=n_indep,
        loss_sum=g_loss,
        unit_loss_sum=u_loss,
        k_int=None,
        binary=False,  # group means are fractional even for 0/1 unit losses (Q2)
        estimand="group_weighted",
        n_total_units=n,
        n_total_indep=n_groups,
    )


@dataclass(frozen=True)
class BandResult:
    band: int
    alpha: float
    delta_j: float
    need: int
    start_index: int | None
    raw_index: int | None
    stop_reason: StopReason
    tests_run: int
    p_value_raw: float | None
    index: int | None
    status: BandStatus
    source_band: int | None


@dataclass(frozen=True)
class LTTResult:
    bands: tuple[BandResult, ...]
    stats: GridStats

    def tau_hat(self, j: int) -> float | None:
        idx = self.bands[j].index
        return None if idx is None else float(self.stats.tau[idx])

    def tau_hat_raw(self, j: int) -> float | None:
        idx = self.bands[j].raw_index
        return None if idx is None else float(self.stats.tau[idx])


def start_index(dev_cov: FloatArray, n_ref: int, need: int, factor: float) -> int | None:
    """First grid index whose dev-estimated committed count reaches ``factor · need`` (7.2.3)."""
    hits = np.nonzero(np.asarray(dev_cov, dtype=np.float64) * n_ref >= factor * need)[0]
    return int(hits[0]) if hits.size else None


def fixed_sequence_ltt(
    stats: GridStats,
    alphas: ArrayLike,
    delta_j: float,
    dev_cov: ArrayLike,
    *,
    start_factor: float = 1.25,
) -> LTTResult:
    """Certify every band, then monotonise across bands (7.2 steps 3-6, C2, C3).

    ``dev_cov[g]`` is the coverage at τ_g estimated on dev, in the certified independence units
    (share of units, or share of groups with at least one committed unit).
    """
    a = np.asarray(alphas, dtype=np.float64).reshape(-1)
    cov = np.asarray(dev_cov, dtype=np.float64).reshape(-1)
    if cov.shape[0] != stats.size:
        raise ValueError("dev_cov must have one value per grid point")
    if np.any(np.diff(a) <= 0):
        raise ValueError("alphas must be strictly increasing")
    if not (0.0 < delta_j < 1.0):
        raise ValueError("delta_j must be in (0, 1)")
    stat = stats.statistic()
    raw: list[tuple[int | None, StopReason, int, float | None, int, int | None]] = []
    for alpha in a:
        need = n_min(float(alpha), delta_j)
        start = start_index(cov, stats.n_total_indep, need, start_factor)
        if start is None:
            raw.append((None, StopReason.INFEASIBLE_ON_DEV, 0, None, need, None))
            continue
        p = p_value(stat, stats.n_indep, float(alpha), binary=stats.binary)
        last: int | None = None
        reason = StopReason.END_OF_GRID
        tests = 0
        for g in range(start, stats.size):
            if stats.n_indep[g] < need:
                reason = StopReason.SAMPLE_SIZE_LIMITED
                break
            tests += 1
            if p[g] <= delta_j:
                last = g
            else:
                reason = StopReason.RISK_LIMITED
                break
        p_raw = None if last is None else float(p[last])
        raw.append((last, reason, tests, p_raw, need, start))

    bands: list[BandResult] = []
    merged = monotonize([r[0] for r in raw])
    for j, ((last, reason, tests, p_raw, need, start), (index, status, source)) in enumerate(
        zip(raw, merged, strict=True)
    ):
        bands.append(
            BandResult(
                band=j,
                alpha=float(a[j]),
                delta_j=delta_j,
                need=need,
                start_index=start,
                raw_index=last,
                stop_reason=reason,
                tests_run=tests,
                p_value_raw=p_raw,
                index=index,
                status=status,
                source_band=source,
            )
        )
    indices = [-1 if b.index is None else b.index for b in bands]
    assert all(x <= y for x, y in pairwise(indices)), "bands must nest"
    return LTTResult(bands=tuple(bands), stats=stats)


def monotonize(
    raw_indices: list[int | None],
) -> list[tuple[int | None, BandStatus, int | None]]:
    """Cross-band monotonisation τ̂'_j = max_{k≤j} τ̂_k (7.2 step 6) with C3 statuses.

    Returns (index, status, source band) per band. A band whose own walk certified at least
    as far as every tighter band keeps ``certified``; otherwise it is ``inherited`` from the
    tightest band that reached the running maximum.
    """
    out: list[tuple[int | None, BandStatus, int | None]] = []
    best: int | None = None
    best_band: int | None = None
    for j, last in enumerate(raw_indices):
        own = last is not None and (best is None or last >= best)
        if own:
            best, best_band = last, j
        if best is None:
            out.append((None, BandStatus.UNCERTIFIED, None))
        elif own:
            out.append((best, BandStatus.CERTIFIED, j))
        else:
            out.append((best, BandStatus.INHERITED, best_band))
    return out


def bonferroni_diagnostic(
    stats: GridStats, alphas: ArrayLike, delta_j: float
) -> list[dict[str, Any]]:
    """Bonferroni over the whole grid (δ_j / G per test). Diagnostic only; never released (C15)."""
    out: list[dict[str, Any]] = []
    stat = stats.statistic()
    for alpha in np.asarray(alphas, dtype=np.float64).reshape(-1):
        need = n_min(float(alpha), delta_j)
        p = p_value(stat, stats.n_indep, float(alpha), binary=stats.binary)
        ok = np.nonzero((stats.n_indep >= need) & (p <= delta_j / stats.size))[0]
        idx = int(ok[-1]) if ok.size else None
        out.append(
            {
                "alpha": float(alpha),
                "tau_hat": None if idx is None else float(stats.tau[idx]),
                "index": idx,
                "per_test_level": delta_j / stats.size,
            }
        )
    return out
