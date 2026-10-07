"""Confidence bounds for reported (not certified) quantities (C7, C17)."""

from __future__ import annotations

import math

from scipy.stats import beta

from amx.cert.pvalues import p_hoeffding_bentkus


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper–Pearson upper bound on a binomial proportion."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def cp_lower(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper–Pearson lower bound on a binomial proportion."""
    if n <= 0 or k <= 0:
        return 0.0
    return float(beta.ppf(1.0 - conf, k, n - k + 1))


def hb_upper(loss_sum: float, n: int, conf: float = 0.95, tol: float = 1e-10) -> float:
    """Upper bound on the mean of [0, 1] losses: inf{a : p_HB(R̂; a) ≤ 1 − conf}."""
    if n <= 0:
        return 1.0
    level = 1.0 - conf
    r_hat = loss_sum / n
    lo, hi = min(max(r_hat, 0.0), 1.0 - 1e-12), 1.0 - 1e-12
    if float(p_hoeffding_bentkus(loss_sum, n, hi)) > level:
        return 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if float(p_hoeffding_bentkus(loss_sum, n, mid)) <= level:
            hi = mid
        else:
            lo = mid
    return hi


def hb_lower(loss_sum: float, n: int, conf: float = 0.95) -> float:
    """Lower bound on the mean of [0, 1] losses, by symmetry on 1 − ℓ."""
    if n <= 0:
        return 0.0
    return max(0.0, 1.0 - hb_upper(n - loss_sum, n, conf))


def risk_upper(loss_sum: float, n: int, *, binary: bool, conf: float = 0.95) -> float:
    return cp_upper(round(loss_sum), n, conf) if binary else hb_upper(loss_sum, n, conf)


def risk_lower(loss_sum: float, n: int, *, binary: bool, conf: float = 0.95) -> float:
    return cp_lower(round(loss_sum), n, conf) if binary else hb_lower(loss_sum, n, conf)


def bound_name(binary: bool) -> str:
    return "clopper_pearson" if binary else "hoeffding_bentkus"


def dkw_halfwidth(n: int, conf: float = 0.95) -> float:
    """Half-width of the DKW band, uniform over τ, for an empirical coverage curve (C17)."""
    if n <= 0:
        return 1.0
    return math.sqrt(math.log(2.0 / (1.0 - conf)) / (2.0 * n))
