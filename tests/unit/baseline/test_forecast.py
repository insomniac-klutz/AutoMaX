from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from amx.baseline import (
    TrivialForecaster,
    forecast_lookback,
    forecast_roles,
    kfold_train_mask,
    make_forecast_units,
)
from amx.data import Roles, UnitFrame
from amx.loss import err_gt_tol

HORIZONS = (1, 5, 12, 13)
SEASON = 12


def series(n: int = 400, seed: int = 3) -> tuple[np.ndarray, pd.DatetimeIndex]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 20 + 4 * np.sin(2 * np.pi * t / SEASON) + rng.normal(size=n) * (1 + (t % SEASON > 6))
    return y, pd.date_range("2021-03-01", periods=n, freq="D")


def test_shapes_and_columns() -> None:
    y, t = series()
    max_lag = forecast_lookback(HORIZONS, SEASON)
    assert max_lag == 12  # h = 13 reads y[o - 11]; the volatility reads y[o - 12]
    units = make_forecast_units(y, t, HORIZONS, SEASON, max_lag)
    assert list(units.columns) == [
        "unit_id",
        "origin",
        "origin_time",
        "horizon",
        "anchor",
        "vol",
        "target",
    ]
    expected = sum(len(y) - h - max_lag for h in HORIZONS)
    assert len(units) == expected
    assert units["unit_id"].is_unique
    first = units.iloc[0]
    assert first["unit_id"] == f"o{max_lag}_h1"
    assert first["origin_time"] == t[max_lag]
    for h in HORIZONS:
        sub = units[units["horizon"] == h]
        assert sub["origin"].min() == max_lag
        assert sub["origin"].max() == len(y) - 1 - h
    assert list(units["origin"]) == sorted(units["origin"])


def test_feature_values() -> None:
    y, t = series()
    units = make_forecast_units(y, t, HORIZONS, SEASON, 20, stride=3)
    assert set((units["origin"] - 20) % 3) == {0}
    for row in units.sample(40, random_state=0).itertuples():
        o, h = int(row.origin), int(row.horizon)
        assert row.anchor == y[o + h - SEASON * math.ceil(h / SEASON)]
        assert row.target == y[o + h]
        diffs = y[o - SEASON + 1 : o + 1] - y[o - SEASON : o]
        assert row.vol == pytest.approx(np.std(diffs), rel=1e-12)


def test_no_look_ahead() -> None:
    y, t = series()
    base = make_forecast_units(y, t, HORIZONS, SEASON, 12)
    assert np.all(
        base["origin"] + base["horizon"] - SEASON * np.ceil(base["horizon"] / SEASON)
        <= base["origin"]
    )
    for cut in (40, 151, 333):
        z = y.copy()
        z[cut:] = 1e6 + np.arange(z.size - cut)
        changed = make_forecast_units(z, t, HORIZONS, SEASON, 12)
        before = base["origin"] < cut
        cols = ["unit_id", "origin", "horizon", "anchor", "vol"]
        pd.testing.assert_frame_equal(base.loc[before, cols], changed.loc[before, cols])
        # Only the targets of earlier origins may change; a unit whose origin is the cut
        # itself does read the changed value, which shows the test can detect a change.
        at_cut = base["origin"] == cut
        assert not np.allclose(base.loc[at_cut, "vol"], changed.loc[at_cut, "vol"])


def test_refusals() -> None:
    y, t = series(50)
    with pytest.raises(ValueError, match="lookback"):
        make_forecast_units(y, t, HORIZONS, SEASON, 11)
    with pytest.raises(ValueError, match="season"):
        make_forecast_units(y, t, [1], 1, 5)
    with pytest.raises(ValueError, match="horizons"):
        make_forecast_units(y, t, [5, 5], SEASON, 12)
    bad = y.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        make_forecast_units(bad, t, [1], SEASON, 12)


def test_forecaster_point_signal_and_scores() -> None:
    y, t = series(600)
    units = make_forecast_units(y, t, HORIZONS, SEASON, 12)
    uf = UnitFrame.from_pandas(units, forecast_roles())
    folds = np.full(uf.n, -1, dtype=np.int64)  # ignored by design
    model = TrivialForecaster().fit(uf, folds, kfold_train_mask, err_gt_tol(2.0), 0)
    assert np.array_equal(model.predict(uf), units["anchor"].to_numpy())
    expected = units["vol"].to_numpy() * np.sqrt(units["horizon"].to_numpy())
    assert np.allclose(model.signal(uf), expected)
    assert model.oof_ids().tolist() == uf.ids.tolist()
    u, s = model.signal(uf), model.commit_score(uf)
    order = np.argsort(u, kind="stable")
    assert np.all(np.diff(s[order]) >= 0.0)
    # The tie-breaker scale is fixed in-sample on dev, from the positive signals only.
    assert model.scorer_.scale_ == float(np.median(u[u > 0.0]))
    empty = uf.take(np.empty(0, dtype=np.int64))
    for out in (model.predict(empty), model.signal(empty), model.commit_score(empty)):
        assert out.shape == (0,) and out.dtype == np.float64
    cov = model.dev_cov(np.geomspace(1e-4, 1.0, 50))
    assert np.all(np.diff(cov) >= 0.0) and cov[-1] == 1.0
    no_vol = UnitFrame.from_pandas(
        units.drop(columns=["vol"]),
        Roles(unit_id="unit_id", target="target", inputs=("anchor", "horizon")),
    )
    with pytest.raises(ValueError, match="vol"):
        model.predict(no_vol)
