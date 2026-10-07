"""Closed vocabularies used by the TaskSpec (HANDOFF 6.1, 3)."""

from __future__ import annotations

from enum import StrEnum


class InputKind(StrEnum):
    TEXT = "text"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TIMESTAMP = "timestamp"
    IMAGE = "image"
    AUDIO = "audio"
    SERIES = "series"
    INFER = "infer"


class TargetKind(StrEnum):
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    SPANS = "spans"
    SET = "set"
    SERIES = "series"


class DataFormat(StrEnum):
    PARQUET = "parquet"
    CSV = "csv"
    JSONL = "jsonl"
    IMAGE_FOLDER = "image_folder"
    CUSTOM = "custom"


class Family(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    EXTRACTION = "extraction"
    ANOMALY = "anomaly"
    MULTILABEL = "multilabel"
    RANKING = "ranking"
    DETECTION = "detection"


R1_FAMILIES = frozenset(
    {
        Family.CLASSIFICATION,
        Family.REGRESSION,
        Family.FORECASTING,
        Family.EXTRACTION,
        Family.ANOMALY,
    }
)


class CommitUnit(StrEnum):
    ROW = "row"
    SERIES_HORIZON = "series_horizon"
    FIELD = "field"
    SPAN = "span"
    QUERY = "query"
    OBJECT = "object"


# Commit units each family may use (decision D19 / review C-forecast-horizons).
FAMILY_COMMIT_UNITS: dict[Family, frozenset[CommitUnit]] = {
    Family.CLASSIFICATION: frozenset({CommitUnit.ROW}),
    Family.REGRESSION: frozenset({CommitUnit.ROW}),
    Family.FORECASTING: frozenset({CommitUnit.SERIES_HORIZON}),
    Family.EXTRACTION: frozenset({CommitUnit.FIELD, CommitUnit.SPAN}),
    Family.ANOMALY: frozenset({CommitUnit.ROW}),
    Family.MULTILABEL: frozenset({CommitUnit.ROW}),
    Family.RANKING: frozenset({CommitUnit.QUERY}),
    Family.DETECTION: frozenset({CommitUnit.OBJECT}),
}


class BuiltinLoss(StrEnum):
    ZERO_ONE = "zero_one"
    ERR_GT_TOL = "err_gt_tol"
    ONE_MINUS_F1 = "one_minus_f1"
    MISSED_ANOMALY = "missed_anomaly"
    ANOMALY_COST = "anomaly_cost"


# Builtin losses each R1 family may use. Anything else needs a custom loss.
FAMILY_BUILTIN_LOSSES: dict[Family, frozenset[BuiltinLoss]] = {
    Family.CLASSIFICATION: frozenset({BuiltinLoss.ZERO_ONE}),
    Family.REGRESSION: frozenset({BuiltinLoss.ERR_GT_TOL}),
    Family.FORECASTING: frozenset({BuiltinLoss.ERR_GT_TOL}),
    Family.EXTRACTION: frozenset({BuiltinLoss.ONE_MINUS_F1, BuiltinLoss.ZERO_ONE}),
    Family.ANOMALY: frozenset({BuiltinLoss.MISSED_ANOMALY, BuiltinLoss.ANOMALY_COST}),
}

# The family default loss (HANDOFF section 3). Anomaly has none: a spec must name one (Q8).
FAMILY_DEFAULT_LOSS: dict[Family, BuiltinLoss | None] = {
    Family.CLASSIFICATION: BuiltinLoss.ZERO_ONE,
    Family.REGRESSION: BuiltinLoss.ERR_GT_TOL,
    Family.FORECASTING: BuiltinLoss.ERR_GT_TOL,
    Family.EXTRACTION: BuiltinLoss.ONE_MINUS_F1,
    Family.ANOMALY: None,
}


class LossKind(StrEnum):
    BUILTIN = "builtin"
    CUSTOM = "custom"


class Policy(StrEnum):
    AUTO = "auto"
    AUDIT = "audit"


class Regime(StrEnum):
    AUTO = "auto"
    IID = "iid"
    GROUPED = "grouped"
    TEMPORAL = "temporal"
    BLOCKED = "blocked"


class CertMode(StrEnum):
    FIXED_SEQUENCE = "fixed_sequence"


class CallPolicy(StrEnum):
    """How certify calls share the failure budget (OQ Q1)."""

    SINGLE = "single"
    PREREGISTERED = "preregistered"
    SLICED = "sliced"


class ForecastOrigins(StrEnum):
    ROLLING = "rolling"
    FIXED = "fixed"
