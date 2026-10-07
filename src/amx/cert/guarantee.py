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


CLAIMING_TYPES = frozenset({GuaranteeType.PAC_HIGH_PROB, GuaranteeType.EXPECTATION})


def claims_certification(gtype: GuaranteeType) -> bool:
    """Whether a guarantee type may be described as a certified selective-risk bound (D23)."""
    return gtype in CLAIMING_TYPES


def guarantee_type_for(regime: Regime, family: Family) -> GuaranteeType:
    """Certifier selection for selective commits (7.1, Q6). The profiler resolves ``auto``."""
    if regime is Regime.AUTO:
        raise ValueError("resolve regime 'auto' before choosing a guarantee type")
    if family is Family.FORECASTING or regime in (Regime.TEMPORAL, Regime.BLOCKED):
        return GuaranteeType.HOLDOUT_EMPIRICAL
    return GuaranteeType.PAC_HIGH_PROB


def effective_guarantee_type(
    regime: Regime, family: Family, *, time_declared: bool
) -> GuaranteeType:
    """7.1 with the exchangeability audit applied: a declared time column means future units
    come after past ones, so an exchangeable (iid/grouped) regime is downgraded to
    ``holdout_empirical`` (6.5: warnings must include any regime downgrade)."""
    gtype = guarantee_type_for(regime, family)
    if time_declared and regime not in (Regime.TEMPORAL, Regime.BLOCKED):
        return GuaranteeType.HOLDOUT_EMPIRICAL
    return gtype


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
