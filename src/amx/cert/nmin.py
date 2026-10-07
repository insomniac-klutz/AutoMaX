"""Minimum committed calibration units (HANDOFF 7.4)."""

from __future__ import annotations

import math


def n_min(alpha: float, delta: float) -> int:
    """Smallest n with (1 − α)^n ≤ δ: the count needed to certify R ≤ α with zero losses."""
    if not (0.0 < alpha < 1.0) or not (0.0 < delta < 1.0):
        raise ValueError("alpha and delta must lie in (0, 1)")
    log_d, log_q = math.log(delta), math.log1p(-alpha)
    n = max(1, math.ceil(log_d / log_q))
    # Guard the float ceiling in both directions.
    while n > 1 and (n - 1) * log_q <= log_d:
        n -= 1
    while n * log_q > log_d:
        n += 1
    return n
