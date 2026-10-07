"""A0 trivial-model shim: dev-fitted scored predictors and their frozen artifact.

Provisional (ROLLER step 9). Replaced by the A1 operator library and learned scorer. The
resolver entry point is ``python -m amx.baseline.resolve`` (not re-exported, so that running
it as a module stays clean).
"""

from amx.baseline.artifact import (
    ArtifactError,
    FrozenArtifact,
    artifact_meta,
    compute_artifact_hash,
    freeze,
    library_versions,
    load_artifact,
)
from amx.baseline.trivial import (
    FORECAST_INPUTS,
    TIE_EPS,
    Design,
    NotFittedError,
    ScoredPredictor,
    SignalScorer,
    TrainMaskFn,
    TrivialClassifier,
    TrivialForecaster,
    TrivialRegressor,
    fit_trivial,
    forecast_lookback,
    forecast_roles,
    kfold_train_mask,
    make_forecast_units,
)

__all__ = [
    "FORECAST_INPUTS",
    "TIE_EPS",
    "ArtifactError",
    "Design",
    "FrozenArtifact",
    "NotFittedError",
    "ScoredPredictor",
    "SignalScorer",
    "TrainMaskFn",
    "TrivialClassifier",
    "TrivialForecaster",
    "TrivialRegressor",
    "artifact_meta",
    "compute_artifact_hash",
    "fit_trivial",
    "forecast_lookback",
    "forecast_roles",
    "freeze",
    "kfold_train_mask",
    "library_versions",
    "load_artifact",
    "make_forecast_units",
]
