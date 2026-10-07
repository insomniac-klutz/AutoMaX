"""Monotone squashing of unbounded non-negative losses into [0, 1] (HANDOFF 7.9).

A band on a squashed loss is in squashed units; reports must say so.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from amx.loss.base import FloatArray, Loss, LossError


def arctan_squash(x: Any, scale: float = 1.0) -> FloatArray:
    """Map [0, ∞) onto [0, 1) by (2/π)·arctan(x/scale); strictly increasing."""
    if scale <= 0:
        raise LossError("squash scale must be > 0")
    arr = np.asarray(x, dtype=np.float64)
    if np.any(arr < 0):
        raise LossError("only non-negative raw losses can be squashed")
    return np.asarray((2.0 / np.pi) * np.arctan(arr / scale), dtype=np.float64)


def squashed(raw: Any, name: str, scale: float = 1.0) -> Loss:
    """Wrap a raw non-negative loss callable as a bounded Loss."""

    def fn(pred: Any, gold: Any) -> FloatArray:
        return arctan_squash(raw(pred, gold), scale)

    return Loss(name, fn, is_binary=False, params={"squash": "arctan", "scale": scale})
