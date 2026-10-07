"""Guarantee objects: what a certificate claims and under which assumptions (HANDOFF 7.1, 7.6)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from amx.spec.enums import CallPolicy, Family, Regime


class GuaranteeType(StrEnum):
    PAC_HIGH_PROB = "pac_high_prob"
    EXPECTATION = "expectation"
    LONG_RUN_FREQUENCY = "long_run_frequency"
    HOLDOUT_EMPIRICAL = "holdout_empirical"
    NONE = "none"


NEVER_CLAIMED: tuple[str, ...] = (
    "per-class or per-slice risk bounds (slice numbers are descriptive)",
    "performance under distribution shift",
    "any statement about Phase B (LLM) outcomes",
    "coverage: it is estimated at the certified threshold, not guaranteed",
)


def guarantee_type_for(regime: Regime, family: Family) -> GuaranteeType:
    """Certifier selection for selective commits (7.1, Q6). The profiler resolves ``auto``."""
    if regime is Regime.AUTO:
        raise ValueError("resolve regime 'auto' before choosing a guarantee type")
    if family is Family.FORECASTING or regime in (Regime.TEMPORAL, Regime.BLOCKED):
        return GuaranteeType.HOLDOUT_EMPIRICAL
    return GuaranteeType.PAC_HIGH_PROB


def base_assumptions(regime: Regime, *, group_column: str | None) -> list[str]:
    out = [
        f"calibration units and future units are exchangeable under regime={regime.value}",
        "the commit rule is a fixed function of the inputs, set before calibration",
        "risk is measured against the provided gold labels",
    ]
    if group_column is not None:
        out.append(
            f"groups defined by '{group_column}' are independent; the certified quantity is "
            "the group-weighted selective risk"
        )
    if regime is Regime.TEMPORAL:
        out.append("the calibration window is representative of the future (stationarity)")
    return out


class Guarantee(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: GuaranteeType
    delta: float
    delta_per_band: float
    simultaneous_level: float
    regime: Regime
    estimand: Literal["unit_weighted", "group_weighted"]
    p_value_family: Literal["binomial", "hoeffding_bentkus"]
    cert_mode: Literal["fixed_sequence"] = "fixed_sequence"
    call_policy: CallPolicy = CallPolicy.SINGLE
    certify_calls_used: int
    certify_calls_max: int
    calib_n: int
    calib_independent_n: int
    sealed_n: int | None = None
    grid_hash: str
    assumptions: list[str]
