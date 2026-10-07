"""The τ grid, fixed before calibration (HANDOFF 7.2 step 1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from amx.spec.hashing import content_hash
from amx.spec.models import CertSpec


@dataclass(frozen=True)
class TauGrid:
    """Log-spaced thresholds from conservative to liberal."""

    size: int = 200
    low: float = 1e-4
    high: float = 1.0

    def __post_init__(self) -> None:
        if self.size < 2 or not (0.0 < self.low < self.high <= 1.0):
            raise ValueError("grid needs size >= 2 and 0 < low < high <= 1")

    @property
    def values(self) -> NDArray[np.float64]:
        return np.geomspace(self.low, self.high, self.size)

    @property
    def hash(self) -> str:
        return content_hash(
            {"kind": "geomspace", "size": self.size, "low": self.low, "high": self.high}
        )

    @classmethod
    def from_spec(cls, cert: CertSpec) -> TauGrid:
        return cls(size=cert.grid_size, low=cert.grid_min, high=cert.grid_max)
