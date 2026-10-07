"""TaskSpec models (HANDOFF 6.1 as amended by C8, C15 and decisions D18/D19).

The TaskSpec is the only place where a run's domain enters: data, loss, constraints, priors.
Validation here is structural. Checks that need data (the n_min feasibility check, loss
range checks on real labels) run later in ``amx profile`` and ``amx.loss.validate`` (C9).
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from amx.spec.enums import (
    FAMILY_BUILTIN_LOSSES,
    FAMILY_COMMIT_UNITS,
    FAMILY_DEFAULT_LOSS,
    R1_FAMILIES,
    BuiltinLoss,
    CallPolicy,
    CertMode,
    CommitUnit,
    DataFormat,
    Family,
    ForecastOrigins,
    InputKind,
    LossKind,
    Policy,
    Regime,
    TargetKind,
)

FRACTION_TOLERANCE = 1e-9
NAME_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class InputSpec(_Model):
    name: str = Field(min_length=1)
    kind: InputKind


class TargetSpec(_Model):
    name: str = Field(min_length=1)
    kind: TargetKind


class DataSpec(_Model):
    uri: str = Field(min_length=1, description="file, directory, or loader module (format custom)")
    format: DataFormat
    unit_id: str = Field(min_length=1)
    inputs: list[InputSpec] | Literal["infer"] = "infer"
    target: TargetSpec
    time_column: str | None = None
    group_columns: list[str] = Field(
        default_factory=list, description="entity ids that must not straddle folds"
    )
    series_columns: list[str] = Field(
        default_factory=list,
        description="temporal entity ids; a series may span folds in time (D19)",
    )
    independence_unit: str = Field(default="unit", description="unit | group:<col> (see 7.5)")

    @field_validator("independence_unit")
    @classmethod
    def _parse_independence(cls, v: str) -> str:
        if v == "unit":
            return v
        if v.startswith("group:") and len(v) > len("group:"):
            return v
        raise ValueError("independence_unit must be 'unit' or 'group:<col>'")

    @model_validator(mode="after")
    def _check_columns(self) -> DataSpec:
        col = self.independence_group
        if col is not None and col not in (*self.group_columns, *self.series_columns):
            raise ValueError(
                f"independence_unit names '{col}', which is in neither group_columns "
                "nor series_columns"
            )
        if isinstance(self.inputs, list):
            names = [i.name for i in self.inputs]
            if len(set(names)) != len(names):
                raise ValueError("input names must be unique")
            reserved = {self.unit_id, self.target.name}
            clash = reserved.intersection(names)
            if clash:
                raise ValueError(f"unit_id and target cannot be inputs: {sorted(clash)}")
        if self.unit_id == self.target.name:
            raise ValueError("unit_id and target must be different columns")
        overlap = set(self.group_columns) & set(self.series_columns)
        if overlap:
            raise ValueError(f"columns cannot be both group and series columns: {sorted(overlap)}")
        return self

    @property
    def independence_group(self) -> str | None:
        """Column that defines the independence unit, or None for unit-level."""
        if self.independence_unit == "unit":
            return None
        return self.independence_unit.split(":", 1)[1]

    @property
    def input_names(self) -> list[str] | None:
        return None if self.inputs == "infer" else [i.name for i in self.inputs]


class LossSpec(_Model):
    kind: LossKind = LossKind.BUILTIN
    name: BuiltinLoss | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    path: str | None = None
    fn: str | None = None

    @model_validator(mode="after")
    def _check_kind(self) -> LossSpec:
        if self.kind is LossKind.CUSTOM:
            if not self.path or not self.fn:
                raise ValueError("a custom loss needs both 'path' and 'fn'")
            if self.name is not None:
                raise ValueError("a custom loss must not set a builtin 'name'")
        elif self.path is not None or self.fn is not None:
            raise ValueError("'path' and 'fn' are only valid for kind: custom")
        return self


class ForecastSpec(_Model):
    horizons: list[int] = Field(min_length=1)
    origins: ForecastOrigins = ForecastOrigins.ROLLING
    origin_stride: int = Field(default=1, ge=1)
    max_lag: int = Field(default=0, ge=0)

    @field_validator("horizons")
    @classmethod
    def _horizons(cls, v: list[int]) -> list[int]:
        if any(h < 1 for h in v):
            raise ValueError("horizons must be >= 1")
        if any(b <= a for a, b in pairwise(v)):
            raise ValueError("horizons must be strictly increasing")
        return v


class TaskSection(_Model):
    family: Family
    commit_unit: CommitUnit | None = None
    loss: LossSpec = Field(default_factory=LossSpec)
    forecast: ForecastSpec | None = None

    @model_validator(mode="after")
    def _check_family(self) -> TaskSection:
        if self.family not in R1_FAMILIES:
            raise ValueError(
                f"family '{self.family.value}' is designed for but not built in release R1 (D7)"
            )
        if (
            self.commit_unit is not None
            and self.commit_unit not in FAMILY_COMMIT_UNITS[self.family]
        ):
            allowed = sorted(u.value for u in FAMILY_COMMIT_UNITS[self.family])
            raise ValueError(f"commit_unit for {self.family.value} must be one of {allowed}")
        if self.loss.kind is LossKind.BUILTIN:
            name = self.loss.name or FAMILY_DEFAULT_LOSS[self.family]
            if name is None:
                raise ValueError(
                    f"family '{self.family.value}' has no default loss; name one of "
                    f"{sorted(n.value for n in FAMILY_BUILTIN_LOSSES[self.family])} (OQ Q8)"
                )
            if name not in FAMILY_BUILTIN_LOSSES[self.family]:
                raise ValueError(
                    f"builtin loss '{name.value}' does not fit family '{self.family.value}'"
                )
            if (
                name in (BuiltinLoss.MISSED_ANOMALY, BuiltinLoss.ANOMALY_COST)
                and "normal_label" not in self.loss.params
            ):
                raise ValueError(f"loss '{name.value}' needs params.normal_label")
            if name is BuiltinLoss.ERR_GT_TOL and "tol" not in self.loss.params:
                raise ValueError("loss 'err_gt_tol' needs params.tol")
        if self.family is Family.FORECASTING and self.forecast is None:
            raise ValueError("family forecasting needs a 'forecast' block (horizons, max_lag)")
        if self.family is not Family.FORECASTING and self.forecast is not None:
            raise ValueError("'forecast' is only valid for family forecasting")
        return self

    @property
    def loss_name(self) -> BuiltinLoss | None:
        """Resolved builtin loss name, or None for a custom loss."""
        if self.loss.kind is LossKind.CUSTOM:
            return None
        return self.loss.name or FAMILY_DEFAULT_LOSS[self.family]

    @property
    def resolved_commit_unit(self) -> CommitUnit:
        if self.commit_unit is not None:
            return self.commit_unit
        return sorted(FAMILY_COMMIT_UNITS[self.family], key=lambda u: u.value)[0]


class WatchSlice(_Model):
    name: str = Field(min_length=1)
    by: str = Field(min_length=1, description="a data column, or 'target'")


class BandsSpec(_Model):
    alphas: list[float] = Field(min_length=1)
    policies: list[Policy] = Field(min_length=1)
    delta: float = 0.10
    watch_slices: list[WatchSlice] = Field(default_factory=list)

    @field_validator("alphas")
    @classmethod
    def _alphas(cls, v: list[float]) -> list[float]:
        if any(not (0.0 < a < 1.0) or not math.isfinite(a) for a in v):
            raise ValueError("every alpha must be in (0, 1)")
        if any(b <= a for a, b in pairwise(v)):
            raise ValueError("alphas must be strictly increasing")
        return v

    @field_validator("delta")
    @classmethod
    def _delta(cls, v: float) -> float:
        if not (0.0 < v <= 0.5):
            raise ValueError("delta must be in (0, 0.5]")
        return v

    @model_validator(mode="after")
    def _policies(self) -> BandsSpec:
        if len(self.policies) != len(self.alphas):
            raise ValueError("policies and alphas must have the same length")
        seen_audit = False
        for p in self.policies:
            if p is Policy.AUDIT:
                seen_audit = True
            elif seen_audit:
                raise ValueError("policies must be auto bands first, then audit bands (C8)")
        if Policy.AUTO not in self.policies:
            raise ValueError("at least one band must have policy 'auto'")
        names = [s.name for s in self.watch_slices]
        if len(set(names)) != len(names):
            raise ValueError("watch_slices names must be unique")
        return self

    @property
    def m(self) -> int:
        return len(self.alphas)


class Fractions(_Model):
    """All three keys are required when fractions are given (C8)."""

    dev: float
    calib: float
    sealed: float

    @model_validator(mode="after")
    def _sum(self) -> Fractions:
        for key in ("dev", "calib", "sealed"):
            v = getattr(self, key)
            if not (0.0 < v < 1.0):
                raise ValueError(f"fraction '{key}' must be in (0, 1)")
        total = self.dev + self.calib + self.sealed
        if abs(total - 1.0) > FRACTION_TOLERANCE:
            raise ValueError(f"fractions must sum to 1 (got {total!r})")
        return self


class SplitsSpec(_Model):
    regime: Regime = Regime.AUTO
    fractions: Fractions = Field(default_factory=lambda: Fractions(dev=0.6, calib=0.2, sealed=0.2))
    seed: int = 1337
    max_certify_calls: int = Field(default=1, ge=1)
    oof_folds: int = Field(default=5, ge=2, le=20)
    embargo: int | None = Field(
        default=None,
        ge=0,
        description="distinct time steps dropped between temporal folds; default derives "
        "from forecast horizons and max_lag",
    )


class BudgetSpec(_Model):
    wall_clock_hours: float = Field(default=4.0, gt=0)
    cpu_cores: int = Field(default=16, ge=1)
    gpus: int = Field(default=0, ge=0)
    max_experiments: int = Field(default=300, ge=1)
    agent_max_usd: float = Field(default=150.0, ge=0)


class ConstraintsSpec(_Model):
    max_apply_ms_per_unit: float = Field(default=50.0, gt=0)
    max_model_mb: float = Field(default=500.0, gt=0)
    allow_pretrained_encoders: bool = True
    allow_gpu: bool = False
    interpretable_only: bool = False
    forbidden_inputs: list[str] = Field(default_factory=list)


class PriorsSpec(_Model):
    label_hierarchy: dict[str, Any] | None = None
    invariants: list[dict[str, Any]] = Field(default_factory=list)


class CertSpec(_Model):
    """Certification parameters, fixed before calibration and hashed at split (C15)."""

    grid_size: int = Field(default=200, ge=10, le=5000)
    grid_min: float = Field(default=1e-4, gt=0)
    grid_max: float = Field(default=1.0, le=1.0)
    start_factor: float = Field(default=1.25, ge=1.0)
    mode: CertMode = CertMode.FIXED_SEQUENCE
    call_policy: CallPolicy = CallPolicy.SINGLE
    tau0: float | None = Field(default=None, gt=0, le=1)
    aci_gamma: float = Field(default=0.005, gt=0, lt=1)

    @model_validator(mode="after")
    def _grid(self) -> CertSpec:
        if self.grid_min >= self.grid_max:
            raise ValueError("grid_min must be below grid_max")
        return self


class TaskSpec(_Model):
    spec_version: Literal[1] = 1
    name: str = Field(pattern=NAME_PATTERN)
    data: DataSpec
    task: TaskSection
    bands: BandsSpec
    splits: SplitsSpec = Field(default_factory=SplitsSpec)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    constraints: ConstraintsSpec = Field(default_factory=ConstraintsSpec)
    priors: PriorsSpec = Field(default_factory=PriorsSpec)
    cert: CertSpec = Field(default_factory=CertSpec)

    @model_validator(mode="after")
    def _cross(self) -> TaskSpec:
        regime = self.splits.regime
        data = self.data
        if regime is Regime.GROUPED and data.independence_unit == "unit":
            raise ValueError(
                "regime 'grouped' requires independence_unit: group:<col> (OQ Q2): "
                "units inside a group are not independent"
            )
        if regime is Regime.IID and data.group_columns:
            raise ValueError("regime 'iid' cannot have group_columns; use 'grouped'")
        if regime is Regime.TEMPORAL and data.time_column is None:
            raise ValueError("regime 'temporal' needs data.time_column")
        if self.task.family is Family.FORECASTING:
            if regime not in (Regime.AUTO, Regime.TEMPORAL):
                raise ValueError("family forecasting needs regime 'temporal' (or 'auto')")
            if data.time_column is None:
                raise ValueError("family forecasting needs data.time_column")
        if self.cert.call_policy is CallPolicy.SINGLE and self.splits.max_certify_calls != 1:
            raise ValueError(
                "call_policy 'single' allows exactly one certify call per calib fold (OQ Q1)"
            )
        names = data.input_names
        if names is not None:
            known = {*names, data.target.name, data.unit_id, *data.group_columns}
            known.update(data.series_columns)
            if data.time_column is not None:
                known.add(data.time_column)
            for s in self.bands.watch_slices:
                if s.by != "target" and s.by not in known:
                    raise ValueError(f"watch_slice '{s.name}' uses unknown column '{s.by}'")
            unknown = set(self.constraints.forbidden_inputs) - set(names)
            if unknown:
                raise ValueError(f"forbidden_inputs are not inputs: {sorted(unknown)}")
        return self

    @property
    def embargo_steps(self) -> int:
        """Distinct time steps dropped between temporal folds."""
        if self.splits.embargo is not None:
            return self.splits.embargo
        if self.task.forecast is not None:
            return max(self.task.forecast.horizons) + self.task.forecast.max_lag
        return 0
