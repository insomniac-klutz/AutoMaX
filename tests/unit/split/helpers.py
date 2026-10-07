"""Synthetic frames and specs shared by the split tests (no real data)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from amx.data import Roles, UnitFrame
from amx.spec import TaskSpec, parse_taskspec


def unit_id(i: int) -> str:
    """Distinctive ids: no id is a substring of another, none looks like hex."""
    return f"unit-{i:06d}-z"


def make_spec(
    regime: str = "iid",
    *,
    kind: str = "categorical",
    seed: int = 7,
    fractions: Sequence[float] = (0.6, 0.2, 0.2),
    embargo: int | None = None,
    oof_folds: int = 5,
    time_column: str | None = None,
    group_columns: Sequence[str] = (),
    series_columns: Sequence[str] = (),
) -> TaskSpec:
    data: dict[str, Any] = {
        "uri": "data/units.parquet",
        "format": "parquet",
        "unit_id": "id",
        "target": {"name": "y", "kind": kind},
        "group_columns": list(group_columns),
        "series_columns": list(series_columns),
    }
    if time_column is not None:
        data["time_column"] = time_column
    if regime == "grouped" and group_columns:
        data["independence_unit"] = f"group:{group_columns[0]}"
    task: dict[str, Any] = {"family": "classification"}
    if kind == "numeric":
        task = {
            "family": "regression",
            "loss": {"kind": "builtin", "name": "err_gt_tol", "params": {"tol": 0.1}},
        }
    splits: dict[str, Any] = {
        "regime": regime,
        "seed": seed,
        "fractions": dict(zip(("dev", "calib", "sealed"), fractions, strict=True)),
        "oof_folds": oof_folds,
    }
    if embargo is not None:
        splits["embargo"] = embargo
    return parse_taskspec(
        {
            "spec_version": 1,
            "name": "split-test",
            "data": data,
            "task": task,
            "bands": {"alphas": [0.05, 0.1], "policies": ["auto", "audit"]},
            "splits": splits,
        }
    )


def make_forecast_spec(
    *,
    embargo: int | None = None,
    horizons: Sequence[int] = (1, 3),
    max_lag: int = 2,
    oof_folds: int = 5,
    seed: int = 7,
) -> TaskSpec:
    """A temporal forecasting spec on ``make_frame(..., time=...)`` frames (time column ``t``)."""
    raw = make_spec(
        "temporal", time_column="t", embargo=embargo, oof_folds=oof_folds, seed=seed
    ).model_dump(mode="json")
    raw["task"] = {
        "family": "forecasting",
        "commit_unit": "series_horizon",
        "loss": {"kind": "builtin", "name": "err_gt_tol", "params": {"tol": 0.1}},
        "forecast": {"horizons": list(horizons), "max_lag": max_lag},
    }
    raw["data"]["target"]["kind"] = "series"
    return parse_taskspec(raw)


def make_frame(
    n: int,
    *,
    seed: int = 0,
    classes: Sequence[str] = ("class-alpha", "class-beta", "class-gamma"),
    class_probs: Sequence[float] | None = None,
    numeric: bool = False,
    dup_of: dict[int, int] | None = None,
    groups: Sequence[Any] | None = None,
    time: Sequence[Any] | None = None,
    series: Sequence[Any] | None = None,
) -> UnitFrame:
    """Units with continuous inputs (so no accidental duplicates).

    ``dup_of`` maps a row to the row whose inputs it copies (exact duplicates).
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.integers(0, 1_000_000, n)
    for row, src in (dup_of or {}).items():
        a[row] = a[src]
        b[row] = b[src]
    y: Any
    if numeric:
        y = np.round(rng.gamma(2.0, 50.0, n) + 0.123456, 6)
    else:
        y = rng.choice(np.asarray(classes, dtype=object), size=n, p=class_probs)
    cols: dict[str, Any] = {"id": [unit_id(i) for i in range(n)], "a": a, "b": b, "y": y}
    group_cols: tuple[str, ...] = ()
    series_cols: tuple[str, ...] = ()
    time_col = None
    if groups is not None:
        cols["g"] = list(groups)
        group_cols = ("g",)
    if series is not None:
        cols["s"] = list(series)
        series_cols = ("s",)
    if time is not None:
        cols["t"] = list(time)
        time_col = "t"
    roles = Roles(
        unit_id="id",
        target="y",
        inputs=("a", "b"),
        time=time_col,
        group_columns=group_cols,
        series_columns=series_cols,
    )
    return UnitFrame.from_pandas(pd.DataFrame(cols), roles)


def fold_of_id(uf: UnitFrame, folds: np.ndarray[Any, Any]) -> dict[str, str]:
    return dict(zip(uf.ids.tolist(), [str(f) for f in folds], strict=True))
