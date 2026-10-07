"""Loss contract: ℓ(pred, gold) ∈ [0, 1] per unit (HANDOFF 2.1, 7.9)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
LossFn = Callable[[Any, Any], Any]


class LossError(ValueError):
    """A loss was misconfigured or produced values outside [0, 1]."""


@dataclass(frozen=True)
class Loss:
    """A vectorized, bounded loss.

    ``is_binary`` selects the exact binomial p-value at certification; the certifier still
    checks that every observed value is in {0, 1} before using it (C1).
    """

    name: str
    fn: LossFn
    is_binary: bool
    params: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, pred: Any, gold: Any) -> FloatArray:
        n_pred, n_gold = _length(pred), _length(gold)
        if n_pred != n_gold:
            raise LossError(f"{self.name}: pred has {n_pred} units, gold has {n_gold}")
        try:
            out = np.asarray(self.fn(pred, gold), dtype=np.float64).reshape(-1)
        except LossError:
            raise
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise LossError(f"{self.name}: cannot score these values: {exc}") from exc
        if out.shape[0] != n_gold:
            raise LossError(f"{self.name}: returned {out.shape[0]} values for {n_gold} units")
        if not np.all(np.isfinite(out)):
            raise LossError(f"{self.name}: returned non-finite values")
        lo, hi = (float(np.min(out)), float(np.max(out))) if out.size else (0.0, 0.0)
        if lo < 0.0 or hi > 1.0:
            raise LossError(f"{self.name}: values must lie in [0, 1] (got {lo!r}..{hi!r})")
        return out


def _length(x: Any) -> int:
    if isinstance(x, np.ndarray):
        return int(x.shape[0]) if x.ndim else 1
    return len(x)
