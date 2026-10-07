"""Loss validation and the loss-distribution report (HANDOFF 7.9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from amx.loss.base import FloatArray, Loss, LossError

MASS_AT_ENDS_WARN = 0.95
NEAR_CONSTANT_SHARE = 0.99


@dataclass
class LossCheck:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LossDistribution:
    n: int
    mean: float
    mass_at_zero: float
    mass_at_one: float
    histogram: list[int]
    bin_edges: list[float]
    warnings: list[str]


def check_loss(loss: Loss, gold: Any, *, seed: int = 0, n_pairs: int = 2000) -> LossCheck:
    """Range, identity and binarity checks on gold-vs-gold and random gold pairs."""
    res = LossCheck(ok=True)
    g = np.asarray(gold, dtype=object).reshape(-1)
    if g.size == 0:
        return LossCheck(ok=False, errors=["no gold values to validate against"])
    try:
        same = loss(g, g)
        if np.any(same != 0.0):
            res.errors.append(f"ℓ(y, y) must be 0; got max {float(np.max(same))!r}")
    except LossError as exc:
        res.errors.append(f"gold-vs-gold: {exc}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, g.size, size=(min(n_pairs, max(g.size, 2)), 2))
    try:
        vals = loss(g[idx[:, 0]], g[idx[:, 1]])
        if loss.is_binary and not np.all((vals == 0.0) | (vals == 1.0)):
            res.errors.append("declared binary but produced values outside {0, 1}")
    except LossError as exc:
        res.errors.append(f"random pairs: {exc}")
    res.ok = not res.errors
    return res


def loss_distribution(values: FloatArray, *, is_binary: bool, bins: int = 10) -> LossDistribution:
    """Histogram of ℓ on baseline predictions with the 7.9 warnings."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise LossError("no loss values")
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    at0, at1 = float(np.mean(v == 0.0)), float(np.mean(v == 1.0))
    warnings: list[str] = []
    if not is_binary and at0 + at1 > MASS_AT_ENDS_WARN:
        warnings.append(
            f"fractional loss has {100 * (at0 + at1):.1f}% of its mass at {{0, 1}}; "
            "a binary loss may be the honest choice"
        )
    _, counts = np.unique(v, return_counts=True)
    if int(np.max(counts)) / v.size >= NEAR_CONSTANT_SHARE:
        warnings.append(
            "ℓ is nearly constant on these predictions; check the tolerance or loss parameters"
        )
    return LossDistribution(
        n=int(v.size),
        mean=float(v.mean()),
        mass_at_zero=at0,
        mass_at_one=at1,
        histogram=[int(c) for c in hist],
        bin_edges=[float(e) for e in edges],
        warnings=warnings,
    )
