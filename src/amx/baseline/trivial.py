"""A0 trivial-model shim: scored predictors fitted on dev only (ROLLER step 9).

Provisional. The A1 operator library and learned scorer (HANDOFF 7.7) replace this module.
It exists so that A0 can run T1-real-cheap and the profiler's feasibility pass on real data
with the simplest models that still emit an uncertainty signal.

Every predictor follows the same recipe:

1. **OOF on dev.** For each OOF fold ``f`` a model is fitted on ``train_mask_fn(folds, f)``
   and predicts fold ``f``. Units with fold ``-1`` (temporal training-only blocks) are never
   predicted. ``fit`` receives the dev :class:`~amx.data.UnitFrame` only, so no calibration or
   sealed unit can reach it (HANDOFF 6.2 rule 1, 8.1).
2. **Raw signal.** A per-unit uncertainty signal ``u ≥ 0`` (1 − max probability, interval
   width, volatility × √horizon).
3. **Scorer.** Isotonic regression from the OOF signal to the OOF loss (HANDOFF 7.7, A0 form).
   Ties are broken strictly by the raw signal (review finding "scorer ties"):
   ``s(u) = clip((1 − ε)·iso(u) + ε·u / (u + c), 0, 1)`` with ``ε = 1e-9`` and ``c`` the
   median of the positive OOF signals, fixed at fit time (1 when none is positive). The
   bounded map ``u / (u + c)`` keeps the tie-breaker below ε for unbounded signals (interval
   widths, vol·√h), and the ``(1 − ε)`` factor keeps ``s < 1`` so clipping cannot create new
   ties at the top. Scaling by ``c`` keeps the map steep on the signal's own scale: with
   ``c = 1`` its slope ``1/(1 + u)²`` falls below float resolution once ``u ≫ 1``.
4. **Artifact (OQ Q13 default (a)).** The deployed model is the *bag* of the K fold models;
   predictions average them. Nothing is refitted on all of dev, so the deployed signal is
   close to the OOF signal the scorer was trained on.

``dev_cov(τ)`` is the share of dev OOF units with ``s_oof ≤ τ``; the certifier uses it to
fix each band's start index before calibration (HANDOFF 7.2 step 3).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from amx._log import get_logger
from amx.data.unitframe import Roles, UnitFrame
from amx.loss.base import Loss
from amx.spec.enums import Family
from amx.spec.models import TaskSpec

log = get_logger(__name__)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
TrainMaskFn = Callable[[IntArray, int], BoolArray]

TIE_EPS = 1e-9
MISSING_CATEGORY = "__missing__"
MAX_CATEGORIES = 100
RELATIVE_FLOOR = 1e-12
QUANTILES = (0.1, 0.9)
FORECAST_INPUTS = ("anchor", "vol", "horizon")


class NotFittedError(RuntimeError, AttributeError):
    """The predictor has not been fitted (an AttributeError too, for protocol checks)."""


# protocol ----------------------------------------------------------------------------------


@runtime_checkable
class ScoredPredictor(Protocol):
    """A dev-fitted predictor with a commit score s(x) ∈ [0, 1] (HANDOFF 2.3, A0 shim)."""

    family: Family

    @property
    def roles(self) -> Roles: ...

    def fit(
        self,
        dev: UnitFrame,
        oof_folds: ArrayLike,
        train_mask_fn: TrainMaskFn,
        loss: Loss,
        seed: int,
    ) -> ScoredPredictor: ...

    def predict(self, uf: UnitFrame) -> NDArray[Any]: ...

    def signal(self, uf: UnitFrame) -> FloatArray: ...

    def commit_score(self, uf: UnitFrame) -> FloatArray: ...

    def dev_cov(self, tau: ArrayLike, groups: ArrayLike | None = None) -> FloatArray: ...

    def oof_ids(self) -> NDArray[Any]: ...

    def oof_scores(self) -> FloatArray: ...

    def oof_losses(self) -> FloatArray: ...


def kfold_train_mask(folds: IntArray, f: int) -> BoolArray:
    """The iid/grouped rule of ``amx.split.oof.train_mask``: every unit outside fold ``f``."""
    return np.asarray(np.asarray(folds) != f, dtype=np.bool_)


# scorer ------------------------------------------------------------------------------------


def _signal_array(u: ArrayLike) -> FloatArray:
    arr = np.asarray(u, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("raw uncertainty signals must be finite")
    if arr.size and float(np.min(arr)) < 0.0:
        raise ValueError("raw uncertainty signals must be >= 0")
    return arr


class SignalScorer:
    """Isotonic map from a raw signal u ≥ 0 to the expected loss, strictly increasing in u.

    ``scale_`` is the tie-breaker scale ``c`` of ``ε·u/(u + c)``: the median of the positive
    fitting signals, or 1 when none is positive. It is fixed at fit time, so ``s`` is a
    function of ``u`` alone (batch invariant).
    """

    def __init__(self) -> None:
        self.iso_: IsotonicRegression | None = None
        self.scale_ = 1.0

    def fit(self, u: ArrayLike, losses: ArrayLike) -> SignalScorer:
        uu = _signal_array(u)
        ll = np.asarray(losses, dtype=np.float64).reshape(-1)
        if uu.shape != ll.shape:
            raise ValueError("one loss per signal value is required")
        if uu.size < 2:
            raise ValueError("the scorer needs at least two OOF units")
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso_ = iso.fit(uu, ll)
        positive = uu[uu > 0.0]
        self.scale_ = float(np.median(positive)) if positive.size else 1.0
        return self

    def __call__(self, u: ArrayLike) -> FloatArray:
        if self.iso_ is None:
            raise NotFittedError("the scorer is not fitted")
        uu = _signal_array(u)
        if uu.size == 0:  # an empty batch (sklearn refuses 0 samples)
            return np.empty(0, dtype=np.float64)
        g = np.asarray(self.iso_.predict(uu), dtype=np.float64)
        s = (1.0 - TIE_EPS) * g + TIE_EPS * (uu / (uu + self.scale_))
        return np.asarray(np.clip(s, 0.0, 1.0), dtype=np.float64)


# design matrix -----------------------------------------------------------------------------


def _is_numeric(col: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(col.dtype):
        return False
    return bool(
        pd.api.types.is_numeric_dtype(col.dtype) or pd.api.types.is_datetime64_any_dtype(col.dtype)
    )


def _as_float(col: pd.Series) -> FloatArray:
    if pd.api.types.is_datetime64_any_dtype(col.dtype):
        ts = pd.to_datetime(col, utc=True).dt.tz_localize(None)
        raw = ts.to_numpy(dtype="datetime64[ns]")
        out = raw.astype(np.int64).astype(np.float64)
        out[np.isnat(raw)] = np.nan
        return out
    return np.asarray(pd.to_numeric(col, errors="coerce"), dtype=np.float64)


@dataclass(frozen=True)
class Design:
    """Input typing fixed on dev: numeric columns are median-imputed and standardised, the
    rest one-hot encoded (unknown categories ignored)."""

    columns: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> Design:
        cols = tuple(str(c) for c in df.columns)
        if not cols:
            raise ValueError("the trivial models need at least one input column")
        numeric = tuple(c for c in cols if _is_numeric(df[c]))
        return cls(cols, numeric, tuple(c for c in cols if c not in numeric))

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"inputs are missing columns: {missing}")
        data: dict[str, Any] = {}
        for c in self.columns:
            col = df[c]
            if c in self.numeric:
                data[c] = _as_float(col)
            else:
                obj = col.astype(object)
                absent = obj.isna().to_numpy()
                text = obj.astype(str).to_numpy(dtype=object)
                text[absent] = MISSING_CATEGORY
                data[c] = text
        return pd.DataFrame(data, index=pd.RangeIndex(len(df)))

    def transformer(self, *, dense: bool) -> ColumnTransformer:
        parts: list[tuple[str, Any, list[str]]] = []
        if self.numeric:
            num = Pipeline(
                [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
            )
            parts.append(("num", num, list(self.numeric)))
        if self.categorical:
            enc = OneHotEncoder(
                handle_unknown="ignore", max_categories=MAX_CATEGORIES, sparse_output=not dense
            )
            parts.append(("cat", enc, list(self.categorical)))
        return ColumnTransformer(parts, sparse_threshold=0.0 if dense else 0.3)


# shared OOF machinery ----------------------------------------------------------------------


def _check_folds(oof_folds: ArrayLike, n: int) -> IntArray:
    folds = np.asarray(oof_folds).reshape(-1)
    if folds.shape != (n,):
        raise ValueError(f"oof_folds must have one entry per dev unit ({n}), got {folds.shape}")
    if folds.dtype.kind not in "iu":
        raise TypeError("oof_folds must be integers")
    folds = folds.astype(np.int64)
    if np.any(folds < -1):
        raise ValueError("oof fold ids must be >= -1 (-1 = never predicted)")
    if not np.any(folds >= 0):
        raise ValueError("no OOF fold to predict")
    return folds


def _train_rows(train_mask_fn: TrainMaskFn, folds: IntArray, f: int) -> IntArray:
    mask = np.asarray(train_mask_fn(folds, f), dtype=np.bool_).reshape(-1)
    if mask.shape != folds.shape:
        raise ValueError("train_mask_fn must return one boolean per dev unit")
    if np.any(mask & (folds == f)):
        raise ValueError(f"train mask for fold {f} contains units of fold {f}")
    rows = np.flatnonzero(mask).astype(np.int64)
    if rows.size == 0:
        raise ValueError(f"fold {f} has no training units")
    return rows


def _dev_target(dev: UnitFrame) -> NDArray[Any]:
    if not dev.has_target:
        raise ValueError("fit needs a dev UnitFrame with its target")
    return dev.target


class _OOFPredictor:
    """Shared state and scoring for the trivial predictors."""

    family: Family

    def __init__(self) -> None:
        self.roles_: Roles | None = None
        self.seed_: int | None = None
        self.scorer_ = SignalScorer()
        self.oof_ids_: NDArray[Any] = np.empty(0, dtype=object)
        self.oof_signal_: FloatArray = np.empty(0, dtype=np.float64)
        self.oof_scores_: FloatArray = np.empty(0, dtype=np.float64)
        self.oof_losses_: FloatArray = np.empty(0, dtype=np.float64)

    @property
    def roles(self) -> Roles:
        if self.roles_ is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted")
        return self.roles_

    def signal(self, uf: UnitFrame) -> FloatArray:
        raise NotImplementedError

    def commit_score(self, uf: UnitFrame) -> FloatArray:
        return self.scorer_(self.signal(uf))

    def oof_ids(self) -> NDArray[Any]:
        return self.oof_ids_.copy()

    def oof_scores(self) -> FloatArray:
        return self.oof_scores_.copy()

    def oof_losses(self) -> FloatArray:
        return self.oof_losses_.copy()

    def dev_cov(self, tau: ArrayLike, groups: ArrayLike | None = None) -> FloatArray:
        """Share of dev OOF units with ``s_oof ≤ τ`` at each τ.

        With ``groups`` (aligned with :meth:`oof_ids`) the share is over groups: a group counts
        as committed when at least one of its units is, matching the group-level certifier.
        """
        t = np.asarray(tau, dtype=np.float64).reshape(-1)
        s = self.oof_scores_
        if s.size == 0:
            raise NotFittedError(f"{type(self).__name__} is not fitted")
        if groups is None:
            ref = np.sort(s)
        else:
            g = np.asarray(groups, dtype=object).reshape(-1)
            if g.shape != s.shape:
                raise ValueError("groups must be aligned with oof_ids()")
            _, gid = np.unique(g.astype(str), return_inverse=True)
            gid = gid.reshape(-1)
            gmin = np.full(int(np.max(gid)) + 1, np.inf)
            np.minimum.at(gmin, gid, s)
            ref = np.sort(gmin)
        return np.asarray(np.searchsorted(ref, t, side="right") / ref.size, dtype=np.float64)

    def _finish(
        self, dev: UnitFrame, rows: IntArray, signal: FloatArray, losses: FloatArray, seed: int
    ) -> None:
        self.scorer_ = SignalScorer().fit(signal, losses)
        self.roles_ = dev.roles
        self.seed_ = seed
        self.oof_ids_ = np.asarray(dev.ids[rows], dtype=object)
        self.oof_signal_ = np.asarray(signal, dtype=np.float64)
        self.oof_scores_ = self.scorer_(signal)
        self.oof_losses_ = np.asarray(losses, dtype=np.float64)
        log.info(
            "fitted %s on %d dev units (%d OOF units), mean OOF loss %.4g",
            type(self).__name__,
            dev.n,
            rows.size,
            float(np.mean(losses)),
        )


# classification ----------------------------------------------------------------------------


class TrivialClassifier(_OOFPredictor):
    """One-hot/standardise + logistic regression; raw signal u = 1 − max probability.

    A fold whose training units hold a single class (possible under forward chaining) fits no
    model: its held-out units get the uniform distribution over the dev classes, i.e. the
    maximal signal ``u = 1 − 1/C``, and the deployed bag leaves that fold out. A prior-only
    model would instead give those units ``u = 0``, the most confident score.
    """

    family = Family.CLASSIFICATION

    def __init__(self) -> None:
        super().__init__()
        self.design_: Design | None = None
        self.classes_: NDArray[Any] = np.empty(0, dtype=object)
        self.models_: list[Any] = []

    def fit(
        self,
        dev: UnitFrame,
        oof_folds: ArrayLike,
        train_mask_fn: TrainMaskFn,
        loss: Loss,
        seed: int,
    ) -> Self:
        y = _dev_target(dev)
        folds = _check_folds(oof_folds, dev.n)
        design = Design.from_frame(dev.inputs_pandas())
        X = design.frame(dev.inputs_pandas())
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("classification needs at least two classes on dev")
        self.design_, self.classes_, self.models_ = design, classes, []
        proba = np.zeros((dev.n, classes.size), dtype=np.float64)
        for f in np.unique(folds[folds >= 0]).tolist():
            train = _train_rows(train_mask_fn, folds, f)
            held = np.flatnonzero(folds == f)
            if np.unique(y[train]).size < 2:
                log.warning(
                    "OOF fold %d trains on a single class; its %d held-out units get the "
                    "uniform distribution over the %d dev classes and the fold is left out "
                    "of the deployed bag",
                    f,
                    held.size,
                    classes.size,
                )
                proba[held] = 1.0 / classes.size
                continue
            model = self._fit_one(X.iloc[train], y[train], seed)
            proba[held] = self._proba([model], X.iloc[held])
            self.models_.append(model)
        if not self.models_:
            raise ValueError("no OOF fold has training units of at least two classes")
        rows = np.flatnonzero(folds >= 0).astype(np.int64)
        p = proba[rows]
        values = classes[np.argmax(p, axis=1)]
        signal = 1.0 - np.max(p, axis=1)
        self._finish(dev, rows, signal, loss(values, y[rows]), seed)
        return self

    def _fit_one(self, X: pd.DataFrame, y: NDArray[Any], seed: int) -> Any:
        assert self.design_ is not None
        est = LogisticRegression(max_iter=1000, random_state=seed)
        pipe = Pipeline([("prep", self.design_.transformer(dense=False)), ("model", est)])
        return pipe.fit(X, y)

    def _proba(self, models: Sequence[Any], X: pd.DataFrame) -> FloatArray:
        index = {c: i for i, c in enumerate(self.classes_.tolist())}
        out = np.zeros((len(X), self.classes_.size), dtype=np.float64)
        if len(X) == 0:  # an empty batch (sklearn refuses 0 samples)
            return out
        for m in models:
            cols = [index[c] for c in m.classes_.tolist()]
            out[:, cols] += np.asarray(m.predict_proba(X), dtype=np.float64)
        return out / len(models)

    def _bag_proba(self, uf: UnitFrame) -> FloatArray:
        if self.design_ is None or not self.models_:
            raise NotFittedError("TrivialClassifier is not fitted")
        return self._proba(self.models_, self.design_.frame(uf.inputs_pandas()))

    def predict(self, uf: UnitFrame) -> NDArray[Any]:
        return np.asarray(self.classes_[np.argmax(self._bag_proba(uf), axis=1)])

    def signal(self, uf: UnitFrame) -> FloatArray:
        return np.asarray(1.0 - np.max(self._bag_proba(uf), axis=1), dtype=np.float64)


# regression --------------------------------------------------------------------------------


def _host_independent(est: HistGradientBoostingRegressor) -> HistGradientBoostingRegressor:
    """Drop the fit-time OpenMP thread count from a fitted gradient-boosting model.

    sklearn's bin mapper keeps the number of threads it used during ``fit``. Pickled as is, it
    makes the artifact hash depend on the host (ROLLER step 9: the same seed gives the same
    artifact hash). ``None`` is the bin mapper's default: the thread count is chosen again
    whenever it transforms, so predictions are unchanged.
    """
    mapper = getattr(est, "_bin_mapper", None)
    if mapper is not None:
        mapper.n_threads = None
    return est


@dataclass(frozen=True)
class _RegressionBag:
    """One fold's preprocessing plus point and quantile models."""

    prep: Any
    point: Any
    lower: Any
    upper: Any


class TrivialRegressor(_OOFPredictor):
    """Gradient-boosted point model plus 0.1/0.9 quantile models; raw signal = interval width.

    For a relative tolerance loss the width is divided by ``max(|ŷ|, abs_floor)`` (with a
    1e-12 floor so ŷ = 0 stays finite), read from ``loss.params``.
    """

    family = Family.REGRESSION

    def __init__(self) -> None:
        super().__init__()
        self.design_: Design | None = None
        self.models_: list[_RegressionBag] = []
        self.relative_ = False
        self.abs_floor_ = 0.0

    def fit(
        self,
        dev: UnitFrame,
        oof_folds: ArrayLike,
        train_mask_fn: TrainMaskFn,
        loss: Loss,
        seed: int,
    ) -> Self:
        y = np.asarray(_dev_target(dev), dtype=np.float64)
        if not np.all(np.isfinite(y)):
            raise ValueError("regression targets on dev must be finite")
        folds = _check_folds(oof_folds, dev.n)
        self.relative_ = bool(loss.params.get("relative", False))
        self.abs_floor_ = float(loss.params.get("abs_floor", 0.0))
        design = Design.from_frame(dev.inputs_pandas())
        X = design.frame(dev.inputs_pandas())
        self.design_, self.models_ = design, []
        point = np.zeros(dev.n, dtype=np.float64)
        signal = np.zeros(dev.n, dtype=np.float64)
        for f in np.unique(folds[folds >= 0]).tolist():
            train = _train_rows(train_mask_fn, folds, f)
            held = np.flatnonzero(folds == f)
            bag = self._fit_one(X.iloc[train], y[train], seed)
            point[held], signal[held] = self._point_signal([bag], X.iloc[held])
            self.models_.append(bag)
        rows = np.flatnonzero(folds >= 0).astype(np.int64)
        self._finish(dev, rows, signal[rows], loss(point[rows], y[rows]), seed)
        return self

    def _fit_one(self, X: pd.DataFrame, y: FloatArray, seed: int) -> _RegressionBag:
        assert self.design_ is not None
        prep = self.design_.transformer(dense=True)
        Xt = prep.fit_transform(X)
        point = HistGradientBoostingRegressor(loss="squared_error", random_state=seed)
        lower = HistGradientBoostingRegressor(
            loss="quantile", quantile=QUANTILES[0], random_state=seed
        )
        upper = HistGradientBoostingRegressor(
            loss="quantile", quantile=QUANTILES[1], random_state=seed
        )
        point, lower, upper = (_host_independent(m.fit(Xt, y)) for m in (point, lower, upper))
        return _RegressionBag(prep, point, lower, upper)

    def _point_signal(
        self, bags: Sequence[_RegressionBag], X: pd.DataFrame
    ) -> tuple[FloatArray, FloatArray]:
        n = len(X)
        point, lo, hi = np.zeros(n), np.zeros(n), np.zeros(n)
        if n == 0:  # an empty batch (sklearn refuses 0 samples)
            return point, hi
        for b in bags:
            Xt = b.prep.transform(X)
            point += np.asarray(b.point.predict(Xt), dtype=np.float64)
            lo += np.asarray(b.lower.predict(Xt), dtype=np.float64)
            hi += np.asarray(b.upper.predict(Xt), dtype=np.float64)
        k = float(len(bags))
        point, lo, hi = point / k, lo / k, hi / k
        width = np.abs(hi - lo)
        if self.relative_:
            width = width / np.maximum(np.maximum(np.abs(point), self.abs_floor_), RELATIVE_FLOOR)
        return point, np.asarray(width, dtype=np.float64)

    def _bag(self, uf: UnitFrame) -> tuple[FloatArray, FloatArray]:
        if self.design_ is None or not self.models_:
            raise NotFittedError("TrivialRegressor is not fitted")
        return self._point_signal(self.models_, self.design_.frame(uf.inputs_pandas()))

    def predict(self, uf: UnitFrame) -> NDArray[Any]:
        return self._bag(uf)[0]

    def signal(self, uf: UnitFrame) -> FloatArray:
        return self._bag(uf)[1]


# forecasting -------------------------------------------------------------------------------


def forecast_lookback(horizons: Sequence[int], season: int) -> int:
    """Steps before the origin that the forecast features read (anchor and volatility)."""
    anchor = max(season * math.ceil(h / season) - h for h in horizons)
    return max(season, anchor)


def make_forecast_units(
    values: ArrayLike,
    times: ArrayLike,
    horizons: Sequence[int],
    season: int,
    max_lag: int,
    *,
    stride: int = 1,
) -> pd.DataFrame:
    """(origin, horizon) units of one regularly spaced series (HANDOFF 3, OQ Q6).

    For origin ``o`` (an integer position) and horizon ``h``:

    * ``anchor`` = ``y[o + h − season·⌈h/season⌉]``, the seasonal-naive forecast;
    * ``vol`` = standard deviation of the last ``season`` first differences up to ``o``;
    * ``target`` = ``y[o + h]``.

    Both features read only ``y[o − max_lag .. o]``: ``max_lag`` must cover
    :func:`forecast_lookback`, and origins start at ``max_lag`` so every unit has the same
    lookback. Only units whose target is observed are produced. Columns: ``unit_id``
    (``o{origin}_h{h}``), ``origin``, ``origin_time``, ``horizon``, ``anchor``, ``vol``,
    ``target``, sorted by origin, then horizon.
    """
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    t = np.asarray(times).reshape(-1)
    hs = [int(h) for h in horizons]
    if t.shape != y.shape:
        raise ValueError("times must have one entry per value")
    if not np.all(np.isfinite(y)):
        raise ValueError("series values must be finite")
    if not hs or any(h < 1 for h in hs) or any(b <= a for a, b in pairwise(hs)):
        raise ValueError("horizons must be positive and strictly increasing")
    if season < 2:
        raise ValueError("season must be >= 2 (the volatility needs two differences)")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    need = forecast_lookback(hs, season)
    if max_lag < need:
        raise ValueError(f"max_lag={max_lag} is below the features' lookback of {need} steps")
    n = y.size
    origins = np.arange(max_lag, n, stride, dtype=np.int64)
    diffs = np.diff(y)
    # vol at o: std of y[i] - y[i-1] for i = o-season+1 .. o, i.e. diffs[o-season : o]
    if origins.size:
        windows = np.lib.stride_tricks.sliding_window_view(diffs, season)
        vol_at = np.std(windows[origins - season], axis=1)
    else:
        vol_at = np.empty(0, dtype=np.float64)
    frames = []
    for h in hs:
        ok = origins + h <= n - 1
        o = origins[ok]
        anchor_idx = o + h - season * math.ceil(h / season)
        frames.append(
            pd.DataFrame(
                {
                    "unit_id": [f"o{int(i)}_h{h}" for i in o],
                    "origin": o,
                    "origin_time": t[o],
                    "horizon": np.full(o.size, h, dtype=np.int64),
                    "anchor": y[anchor_idx],
                    "vol": vol_at[ok],
                    "target": y[o + h],
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["origin", "horizon"], kind="stable").reset_index(drop=True)


def forecast_roles() -> Roles:
    """UnitFrame roles of :func:`make_forecast_units` output."""
    return Roles(unit_id="unit_id", target="target", inputs=FORECAST_INPUTS, time="origin_time")


class TrivialForecaster(_OOFPredictor):
    """Seasonal-naive point forecast; raw signal = vol · √horizon.

    It has no learned parameters besides the isotonic scorer, which is fitted **in-sample on
    all dev units** (``oof_folds`` and ``train_mask_fn`` are accepted and ignored). That only
    affects how well ``dev_cov`` predicts calibration coverage, hence the efficiency of each
    band's start index, never validity: the scorer is still fixed before calibration.
    """

    family = Family.FORECASTING

    def fit(
        self,
        dev: UnitFrame,
        oof_folds: ArrayLike,
        train_mask_fn: TrainMaskFn,
        loss: Loss,
        seed: int,
    ) -> Self:
        y = np.asarray(_dev_target(dev), dtype=np.float64)
        anchor, signal = self._anchor_signal(dev)
        rows = np.arange(dev.n, dtype=np.int64)
        self._finish(dev, rows, signal, loss(anchor, y), seed)
        return self

    @staticmethod
    def _anchor_signal(uf: UnitFrame) -> tuple[FloatArray, FloatArray]:
        missing = [c for c in FORECAST_INPUTS if c not in uf.roles.inputs]
        if missing:
            raise ValueError(f"forecast units need inputs {list(FORECAST_INPUTS)}: {missing}")
        anchor = np.asarray(uf.column("anchor"), dtype=np.float64)
        vol = np.asarray(uf.column("vol"), dtype=np.float64)
        horizon = np.asarray(uf.column("horizon"), dtype=np.float64)
        return anchor, np.asarray(vol * np.sqrt(horizon), dtype=np.float64)

    def predict(self, uf: UnitFrame) -> NDArray[Any]:
        return self._anchor_signal(uf)[0]

    def signal(self, uf: UnitFrame) -> FloatArray:
        return self._anchor_signal(uf)[1]


# dispatch ----------------------------------------------------------------------------------

_BY_FAMILY: dict[Family, type[_OOFPredictor]] = {
    Family.CLASSIFICATION: TrivialClassifier,
    Family.REGRESSION: TrivialRegressor,
    Family.FORECASTING: TrivialForecaster,
}


def fit_trivial(
    spec: TaskSpec,
    dev: UnitFrame,
    oof_folds: ArrayLike,
    train_mask_fn: TrainMaskFn,
    loss: Loss,
    seed: int,
) -> ScoredPredictor:
    """Fit the A0 trivial predictor for the spec's family on dev."""
    family = spec.task.family
    cls = _BY_FAMILY.get(family)
    if cls is None:
        raise NotImplementedError(
            f"no A0 trivial model for family '{family.value}'; it arrives with the A1 "
            "operator library"
        )
    predictor: Any = cls()
    fitted: ScoredPredictor = predictor.fit(dev, oof_folds, train_mask_fn, loss, seed)
    return fitted
