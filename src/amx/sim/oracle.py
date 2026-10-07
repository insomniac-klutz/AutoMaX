"""Population risk-coverage curves of synthetic generators (HANDOFF 12 T1-synth, C6).

The oracle never uses sampled labels. It averages the KNOWN conditional risk r(x) of the
committed answer over a large Monte Carlo sample of inputs (Rao–Blackwellisation): with
c(x) = 1[s(x) ≤ τ],

    cov(τ) = E[c(x)],    R(τ) = E[c(x) r(x)] / E[c(x)],

estimated by ratio means over n_mc inputs, with the delta-method standard error
SE = sqrt(Σ c (r − R̂)²) / Σ c. For group-level certification (OQ Q2) the same ratio is taken
over groups, with the per-group mean of r over committed units as the value.

A selected τ̂ whose oracle risk is within k·SE of α cannot be judged (C6): it is
*indeterminate* and is excluded from violation counts and reported instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Estimand = Literal["unit_weighted", "group_weighted"]

RAO_BLACKWELL_MC = "rao_blackwell_mc"
INDETERMINATE_K = 4.0
"""C6: |R(τ) − α| < 4 SE at a selected τ marks the selection indeterminate."""


@dataclass(frozen=True)
class CurveSums:
    """Running sums along the grid: committed count, Σ value and Σ value² over committed."""

    count: FloatArray
    total: FloatArray
    total_sq: FloatArray

    def __add__(self, other: CurveSums) -> CurveSums:
        return CurveSums(
            self.count + other.count, self.total + other.total, self.total_sq + other.total_sq
        )

    @classmethod
    def zeros(cls, size: int) -> CurveSums:
        z = np.zeros(size, dtype=np.float64)
        return cls(z, z.copy(), z.copy())

    def ratio(self) -> tuple[FloatArray, FloatArray]:
        """Ratio mean Σ v / count and its delta-method SE (NaN where count = 0)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self.count > 0, self.total / np.maximum(self.count, 1.0), np.nan)
            ss = np.maximum(self.total_sq - self.count * np.nan_to_num(mean) ** 2, 0.0)
            se = np.where(self.count > 0, np.sqrt(ss) / np.maximum(self.count, 1.0), np.nan)
        return np.asarray(mean, dtype=np.float64), np.asarray(se, dtype=np.float64)


def event_sums(
    times: ArrayLike, d_count: ArrayLike, d_total: ArrayLike, d_total_sq: ArrayLike, tau: ArrayLike
) -> CurveSums:
    """Sum increments whose event time is ≤ τ_g, for every grid point g.

    A unit-level curve uses one event per input (time s(x), increments 1, r, r²); the group
    curve uses one event per unit with the increments of its group's running mean.
    """
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    order = np.argsort(t, kind="stable")
    pos = np.searchsorted(t[order], np.asarray(tau, dtype=np.float64), side="right")

    def cum(x: ArrayLike) -> FloatArray:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)[order]
        return np.asarray(np.concatenate([[0.0], np.cumsum(arr)])[pos], dtype=np.float64)

    return CurveSums(cum(d_count), cum(d_total), cum(d_total_sq))


def unit_sums(scores: ArrayLike, cond_risk: ArrayLike, tau: ArrayLike) -> CurveSums:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    r = np.asarray(cond_risk, dtype=np.float64).reshape(-1)
    return event_sums(s, np.ones_like(s), r, r * r, tau)


def group_sums(
    scores: ArrayLike, cond_risk: ArrayLike, groups: ArrayLike, tau: ArrayLike
) -> CurveSums:
    """Group-level sums: count = groups with ≥ 1 committed unit, value = their mean r.

    Within a group the committed set at τ is the prefix of its units sorted by score, so the
    group's value changes only at its own unit scores. Each unit contributes the change of its
    group's running mean (and of the indicator, for the group's first unit) at its score.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    r = np.asarray(cond_risk, dtype=np.float64).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    if not (s.shape == r.shape == g.shape):
        raise ValueError("scores, cond_risk and groups must align")
    order = np.lexsort((s, g))
    gs, ss, rs = g[order], s[order], r[order]
    n = gs.shape[0]
    first = np.ones(n, dtype=bool)
    first[1:] = gs[1:] != gs[:-1]
    starts = np.flatnonzero(first)
    sizes = np.diff(np.append(starts, n))
    pos = np.arange(n) - np.repeat(starts, sizes)
    cr = np.cumsum(rs)
    within = cr - np.repeat(cr[starts] - rs[starts], sizes)
    mean = within / (pos + 1.0)
    prev = np.where(first, 0.0, np.roll(mean, 1))
    return event_sums(ss, first.astype(np.float64), mean - prev, mean * mean - prev * prev, tau)


@dataclass(frozen=True)
class OracleCurve:
    """Population coverage and selective risk of a generator along a τ grid.

    ``cov`` and ``risk`` are in the certified independence unit: units, or groups with at least
    one committed unit (then ``estimand`` is ``group_weighted`` and the unit-weighted curve is
    kept in ``unit_cov`` / ``unit_risk``).
    """

    tau: FloatArray
    cov: FloatArray
    risk: FloatArray
    risk_se: FloatArray
    method: str
    n_mc: int
    estimand: Estimand = "unit_weighted"
    unit_cov: FloatArray | None = None
    unit_risk: FloatArray | None = None
    unit_risk_se: FloatArray | None = None

    @property
    def size(self) -> int:
        return int(self.tau.shape[0])

    def indeterminate(self, index: int, alpha: float, k: float = INDETERMINATE_K) -> bool:
        """|R(τ_index) − α| < k · SE(τ_index) (C6). An empty committed set is never so."""
        r, se = float(self.risk[index]), float(self.risk_se[index])
        if math.isnan(r):
            return False
        return abs(r - alpha) < k * se

    def violates(self, index: int, alpha: float) -> bool:
        """R(τ_index) > α. An empty committed population has no risk and never violates."""
        r = float(self.risk[index])
        return (not math.isnan(r)) and r > alpha

    def coverage_index(self, alpha: float) -> int | None:
        """Grid index of the oracle-coverage point at band α (None if no point has R ≤ α).

        Among grid points with R ≤ α it has the largest coverage; ties go to the most liberal
        τ, as the fixed-sequence walk would release.
        """
        ok = np.flatnonzero(np.nan_to_num(self.risk, nan=np.inf) <= alpha)
        if ok.size == 0:
            return None
        best = self.cov[ok] == np.max(self.cov[ok])
        return int(ok[best][-1])

    def coverage_at(self, alpha: float) -> float:
        """Oracle coverage at band α: max cov over grid points with R ≤ α (0 if none)."""
        g = self.coverage_index(alpha)
        return float(self.cov[g]) if g is not None else 0.0

    def check_grid(self, tau: ArrayLike) -> None:
        t = np.asarray(tau, dtype=np.float64).reshape(-1)
        if not np.array_equal(t, self.tau):
            raise ValueError("oracle was computed on a different τ grid")


def curve_from_sums(
    tau: ArrayLike,
    sums: CurveSums,
    n_indep: int,
    *,
    n_mc: int,
    estimand: Estimand = "unit_weighted",
    unit: CurveSums | None = None,
    n_units: int | None = None,
) -> OracleCurve:
    t = np.asarray(tau, dtype=np.float64).reshape(-1)
    risk, se = sums.ratio()
    unit_cov = unit_risk = unit_se = None
    if unit is not None:
        if n_units is None:
            raise ValueError("n_units is required with unit sums")
        unit_risk, unit_se = unit.ratio()
        unit_cov = unit.count / n_units
    return OracleCurve(
        tau=t,
        cov=sums.count / n_indep,
        risk=risk,
        risk_se=se,
        method=RAO_BLACKWELL_MC,
        n_mc=n_mc,
        estimand=estimand,
        unit_cov=unit_cov,
        unit_risk=unit_risk,
        unit_risk_se=unit_se,
    )
