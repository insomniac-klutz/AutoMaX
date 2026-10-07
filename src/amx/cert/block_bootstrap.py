"""Moving-block bootstrap CI for a selective-risk ratio over time-ordered units (HANDOFF 7.5).

Forecasting selective risk is reported, never certified: the estimate and its interval are
labelled ``holdout_empirical`` (decision D15, OQ Q6). The moving-block bootstrap (Künsch 1989)
resamples overlapping blocks of consecutive units so that short-range serial dependence
survives in every replicate. The percentile interval is approximate; its coverage is checked
by Monte Carlo in ``tests/statistical/test_aci_mc.py``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amx.cert.guarantee import GuaranteeType

FloatArray = NDArray[np.float64]

GUARANTEE_TYPE = GuaranteeType.HOLDOUT_EMPIRICAL
"""The only label a block-bootstrap selective-risk interval may carry (D15, Q6)."""


class RatioCI(NamedTuple):
    """Point estimate and percentile interval of sum(ℓ·c) / sum(c)."""

    est: float
    lo: float
    hi: float


def selective_risk(loss: ArrayLike, committed: ArrayLike) -> float:
    """sum(ℓ · c) / sum(c); NaN when nothing is committed."""
    ell = np.asarray(loss, dtype=np.float64).reshape(-1)
    c = np.asarray(committed, dtype=np.float64).reshape(-1)
    den = float(np.sum(c))
    return float(np.sum(ell * c) / den) if den > 0 else math.nan


def block_bootstrap_ratio(
    loss: ArrayLike,
    committed: ArrayLike,
    *,
    block_len: int,
    B: int = 1000,
    conf: float = 0.95,
    rng: np.random.Generator,
) -> RatioCI:
    """Moving-block bootstrap percentile CI for the selective-risk ratio.

    Units must be in time order. Each replicate concatenates ⌈n / block_len⌉ blocks of
    ``block_len`` consecutive units, with start points drawn uniformly from the n − block_len + 1
    possible ones, and truncates to n units. Replicates that commit nothing are dropped; the
    interval is NaN when no replicate commits. Block sums come from prefix sums, so memory is
    O(B · n / block_len).
    """
    ell = np.asarray(loss, dtype=np.float64).reshape(-1)
    c = np.asarray(committed, dtype=np.float64).reshape(-1)
    n = int(ell.shape[0])
    if c.shape[0] != n or n == 0:
        raise ValueError("loss and committed must be non-empty and of equal length")
    if not (np.all(np.isfinite(ell)) and np.all(np.isfinite(c))):
        raise ValueError("loss and committed must be finite")
    if np.any((c != 0.0) & (c != 1.0)):
        raise ValueError("committed must be 0/1")
    if not (1 <= block_len <= n):
        raise ValueError("block_len must lie in [1, n]")
    if B < 1 or not (0.0 < conf < 1.0):
        raise ValueError("need B >= 1 and 0 < conf < 1")

    est = selective_risk(ell, c)
    num_cs = np.concatenate([[0.0], np.cumsum(ell * c)])
    den_cs = np.concatenate([[0.0], np.cumsum(c)])
    n_blocks = -(-n // block_len)
    last_len = n - (n_blocks - 1) * block_len
    starts = rng.integers(0, n - block_len + 1, size=(B, n_blocks))
    full = starts[:, :-1]
    tail = starts[:, -1]
    num = np.sum(num_cs[full + block_len] - num_cs[full], axis=1)
    num = num + num_cs[tail + last_len] - num_cs[tail]
    den = np.sum(den_cs[full + block_len] - den_cs[full], axis=1)
    den = den + den_cs[tail + last_len] - den_cs[tail]
    ok = den > 0
    if not np.any(ok):
        return RatioCI(est, math.nan, math.nan)
    reps = num[ok] / den[ok]
    lo, hi = np.quantile(reps, [(1.0 - conf) / 2.0, (1.0 + conf) / 2.0])
    return RatioCI(est, float(lo), float(hi))
