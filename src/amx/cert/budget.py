"""Splitting the failure budget δ across bands and certify calls (OQ Q1, decision D18)."""

from __future__ import annotations

from dataclasses import dataclass

from amx.spec.enums import CallPolicy


@dataclass(frozen=True)
class DeltaBudget:
    """δ accounting.

    * ``single`` (default): one certify call per calib fold, δ_j = δ / m.
    * ``preregistered``: K graphs fixed before the call, certified together, δ_j = δ / (m K).
    * ``sliced``: K adaptive calls on disjoint calib slices; δ_j = δ / m per call if the released
      call is fixed in advance, else δ / (m K).
    """

    delta: float
    m: int
    policy: CallPolicy = CallPolicy.SINGLE
    k: int = 1
    release_fixed_in_advance: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.delta < 1.0) or self.m < 1 or self.k < 1:
            raise ValueError("need 0 < delta < 1, m >= 1, k >= 1")
        if self.policy is CallPolicy.SINGLE and self.k != 1:
            raise ValueError("policy 'single' has exactly one call")

    @property
    def divisor(self) -> int:
        if self.policy is CallPolicy.PREREGISTERED:
            return self.m * self.k
        if self.policy is CallPolicy.SLICED and not self.release_fixed_in_advance:
            return self.m * self.k
        return self.m

    @property
    def delta_per_band(self) -> float:
        return self.delta / self.divisor

    @property
    def simultaneous_level(self) -> float:
        """Bound on P(some band of the released certificate is violated).

        After cross-band monotonisation only this simultaneous statement holds; a single band
        j is bounded by the sum of δ_k over k ≤ j, not by δ_j (C2).
        """
        return self.delta
