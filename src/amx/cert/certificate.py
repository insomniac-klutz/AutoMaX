"""Certificate and report payload (HANDOFF 6.5 as amended by C2, C3, C7, C17)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from amx.cert.bounds import dkw_halfwidth
from amx.cert.budget import DeltaBudget
from amx.cert.grid import TauGrid
from amx.cert.guarantee import (
    NEVER_CLAIMED,
    Guarantee,
    GuaranteeType,
    base_assumptions,
    claims_certification,
)
from amx.cert.ltt import BandStatus, LTTResult, StopReason
from amx.cert.slices import SliceRisk
from amx.spec.enums import Policy, Regime


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Interval(_M):
    est: float | None
    ci95: tuple[float, float] | None
    method: str


class RiskEstimate(_M):
    est: float | None
    upper95: float | None = None
    bound: str | None = None


class SliceEntry(_M):
    slice: str
    value: str
    n: int
    risk: float | None
    upper95: float | None
    bound: str
    flag: Literal["insufficient_n", "upper_gt_2alpha", "lower_gt_2alpha"] | None


class BandEntry(_M):
    alpha: float
    policy: Policy
    delta_j: float
    status: BandStatus
    tau_hat: float | None
    tau_hat_raw: float | None
    source_band: int | None
    stop_reason: StopReason
    n_min_required: int
    start_index: int | None
    tests_run: int
    p_value_raw: float | None
    n_committed_calib: int
    n_independent_calib: int
    coverage_at_certified_tau: Interval
    risk_calib: RiskEstimate
    risk_calib_unit_weighted: float | None
    risk_sealed: RiskEstimate | None = None
    coverage_sealed: Interval | None = None
    per_slice: list[SliceEntry] = []


class FrontierPoint(_M):
    tau: float
    coverage: float
    risk: float | None


class Certificate(_M):
    amx_version: str
    claims_certification: bool
    run_id: str
    taskspec_hash: str
    artifact_hash: str
    guarantee: Guarantee
    bands: list[BandEntry]
    frontier_descriptive: list[FrontierPoint]
    warnings: list[str]
    never_claimed: list[str]
    diagnostics: dict[str, Any] = {}

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=False) + "\n"

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def read(cls, path: str | Path) -> Certificate:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _f(x: float) -> float | None:
    return None if x != x else float(x)  # NaN -> None


def build_certificate(
    result: LTTResult,
    *,
    policies: Sequence[Policy],
    budget: DeltaBudget,
    grid: TauGrid,
    regime: Regime,
    guarantee_type: GuaranteeType,
    run_id: str,
    taskspec_hash: str,
    artifact_hash: str,
    amx_version: str,
    group_column: str | None = None,
    extra_assumptions: Sequence[str] = (),
    certify_calls_used: int = 1,
    certify_calls_max: int = 1,
    slices: Sequence[Sequence[SliceRisk]] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> Certificate:
    """Assemble the certificate from one LTT run on calibration data."""
    st = result.stats
    if len(policies) != len(result.bands):
        raise ValueError("one policy per band")
    risk = st.risk()
    unit_risk = st.unit_risk()
    cov = st.coverage()
    # DKW needs independent draws: only unit-level, exchangeable (pac) calibration sets get an
    # interval; group-level and temporal coverage is reported as a descriptive estimate (C17).
    use_dkw = st.estimand == "unit_weighted" and guarantee_type is GuaranteeType.PAC_HIGH_PROB
    half = dkw_halfwidth(st.n_total_units)

    def coverage_interval(c: float) -> Interval:
        if not use_dkw:
            return Interval(est=c, ci95=None, method="descriptive")
        return Interval(
            est=c, ci95=(max(0.0, c - half), min(1.0, c + half)), method="dkw_uniform_in_tau"
        )

    bands: list[BandEntry] = []
    warnings: list[str] = []
    for j, b in enumerate(result.bands):
        idx = b.index
        if idx is None:
            n_c = n_i = 0
            cov_iv = coverage_interval(0.0)
            r_est, r_unit = None, None
        else:
            n_c, n_i = int(st.n_units[idx]), int(st.n_indep[idx])
            c = float(cov[idx])
            cov_iv = coverage_interval(c)
            r_est, r_unit = _f(float(risk[idx])), _f(float(unit_risk[idx]))
        if b.status is BandStatus.UNCERTIFIED:
            warnings.append(
                f"band alpha={b.alpha:g}: not certifiable ({b.stop_reason.value}); coverage 0"
            )
        elif b.stop_reason is StopReason.SAMPLE_SIZE_LIMITED:
            warnings.append(f"band alpha={b.alpha:g}: walk stopped by n_min shortfall")
        entries = [] if slices is None else [SliceEntry(**vars(s)) for s in slices[j]]
        for s in entries:
            if s.flag == "lower_gt_2alpha":
                warnings.append(
                    f"band alpha={b.alpha:g}: small slice {s.slice}={s.value} (n={s.n}) has a "
                    "lower bound above 2*alpha"
                )
            if s.flag == "upper_gt_2alpha":
                warnings.append(
                    f"band alpha={b.alpha:g}: slice {s.slice}={s.value} upper bound "
                    f"{s.upper95:.4g} exceeds 2*alpha"
                )
        bands.append(
            BandEntry(
                alpha=b.alpha,
                policy=policies[j],
                delta_j=b.delta_j,
                status=b.status,
                tau_hat=result.tau_hat(j),
                tau_hat_raw=result.tau_hat_raw(j),
                source_band=b.source_band,
                stop_reason=b.stop_reason,
                n_min_required=b.need,
                start_index=b.start_index,
                tests_run=b.tests_run,
                p_value_raw=b.p_value_raw,
                n_committed_calib=n_c,
                n_independent_calib=n_i,
                coverage_at_certified_tau=cov_iv,
                risk_calib=RiskEstimate(est=r_est),
                risk_calib_unit_weighted=r_unit,
                per_slice=entries,
            )
        )
    frontier = [
        FrontierPoint(tau=float(t), coverage=float(c), risk=_f(float(r)))
        for t, c, r in zip(st.tau, cov, risk, strict=True)
    ]
    guarantee = Guarantee(
        type=guarantee_type,
        delta=budget.delta,
        delta_per_band=budget.delta_per_band,
        simultaneous_level=budget.simultaneous_level,
        regime=regime,
        estimand=st.estimand,
        p_value_family="binomial" if st.binary else "hoeffding_bentkus",
        call_policy=budget.policy,
        certify_calls_used=certify_calls_used,
        certify_calls_max=certify_calls_max,
        calib_n=st.n_total_units,
        calib_independent_n=st.n_total_indep,
        grid_hash=grid.hash,
        assumptions=[*base_assumptions(regime, group_column=group_column), *extra_assumptions],
    )
    return Certificate(
        amx_version=amx_version,
        claims_certification=claims_certification(guarantee_type),
        run_id=run_id,
        taskspec_hash=taskspec_hash,
        artifact_hash=artifact_hash,
        guarantee=guarantee,
        bands=bands,
        frontier_descriptive=frontier,
        warnings=warnings,
        never_claimed=list(NEVER_CLAIMED),
        diagnostics=diagnostics or {},
    )
