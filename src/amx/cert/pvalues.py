"""p-values for H0: R ≥ α (HANDOFF 7.3, corrected by C1).

Binary losses use the exact binomial tail on the INTEGER loss count. Bounded fractional losses
use the Hoeffding–Bentkus combination from Learn-then-Test / RCPS, which matches the authors'
reference implementation (``aangelopoulos/ltt``, ``core/bounds.py: hb_p_value``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import xlogy
from scipy.stats import binom

FloatArray = NDArray[np.float64]


def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def _as_int_array(x: ArrayLike, what: str) -> NDArray[np.int64]:
    arr = np.asarray(x)
    if arr.dtype.kind not in "iu":
        raise TypeError(
            f"{what} must be an integer count, not {arr.dtype}; "
            "pass the summed 0/1 losses, never n * mean (C1)"
        )
    if np.any(arr < 0):
        raise ValueError(f"{what} must be >= 0")
    return arr.astype(np.int64)


def p_binomial(k: ArrayLike, n: ArrayLike, alpha: float) -> FloatArray:
    """P(Binomial(n, α) ≤ k), vectorized. ``n = 0`` gives 1."""
    _check_alpha(alpha)
    k_arr, n_arr = _as_int_array(k, "k"), _as_int_array(n, "n")
    if np.any(k_arr > n_arr):
        raise ValueError("k cannot exceed n")
    p = np.asarray(binom.cdf(k_arr, n_arr, alpha), dtype=np.float64)
    return np.where(n_arr == 0, 1.0, np.minimum(p, 1.0))


def h1(a: ArrayLike, b: float) -> FloatArray:
    """Bernoulli KL divergence kl(a ‖ b) = a ln(a/b) + (1−a) ln((1−a)/(1−b)), with 0·ln 0 = 0."""
    a_arr = np.asarray(a, dtype=np.float64)
    return np.asarray(xlogy(a_arr, a_arr / b) + xlogy(1.0 - a_arr, (1.0 - a_arr) / (1.0 - b)))


def p_hoeffding_bentkus(loss_sum: ArrayLike, n: ArrayLike, alpha: float) -> FloatArray:
    """Hoeffding–Bentkus p-value for a mean of [0, 1] losses, vectorized.

    p = min( exp(−n · h1(min(R̂, α), α)),  e · P(Binomial(n, α) ≤ ⌈n R̂⌉) ),  R̂ = loss_sum / n.
    ``n = 0`` or ``R̂ ≥ α`` gives 1. ``⌈loss_sum⌉`` errs only upward under floating error,
    which is the conservative direction.
    """
    _check_alpha(alpha)
    s = np.asarray(loss_sum, dtype=np.float64)
    n_arr = _as_int_array(n, "n")
    if np.any(s < 0) or np.any(s > n_arr + 1e-9):
        raise ValueError("loss_sum must lie in [0, n]")
    with np.errstate(divide="ignore", invalid="ignore"):
        r_hat = np.where(n_arr > 0, s / np.maximum(n_arr, 1), 1.0)
    hoeff = np.exp(-n_arr * h1(np.minimum(r_hat, alpha), alpha))
    k = np.minimum(np.ceil(s), n_arr).astype(np.int64)
    bent = math.e * np.asarray(binom.cdf(k, n_arr, alpha), dtype=np.float64)
    p = np.minimum(np.minimum(hoeff, bent), 1.0)
    return np.where((n_arr == 0) | (r_hat >= alpha), 1.0, p)


def p_value(loss_sum: ArrayLike, n: ArrayLike, alpha: float, *, binary: bool) -> FloatArray:
    """Dispatch: exact binomial for binary losses (integer sums), HB otherwise."""
    if binary:
        return p_binomial(loss_sum, n, alpha)
    return p_hoeffding_bentkus(loss_sum, n, alpha)


def is_binary_losses(losses: Any) -> bool:
    """True when every observed loss is exactly 0 or 1 (C1 assertion before the binomial path)."""
    arr = np.asarray(losses, dtype=np.float64)
    return bool(np.all((arr == 0.0) | (arr == 1.0)))
