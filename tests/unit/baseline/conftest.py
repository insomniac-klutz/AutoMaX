"""Synthetic data for the A0 trivial-model tests (no network, fixed seeds).

Every frame carries a numeric ``tag`` input equal to the row number, so a spy on the fit
calls can tell exactly which rows a model was fitted on. The full frame stands for the whole
dataset; ``outer_split`` keeps 60% as dev and the rest stands in for calibration units that
``fit`` must never see.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from amx.data import Roles, UnitFrame

INPUTS = ("x1", "x2", "cat", "tag")


@dataclass(frozen=True)
class Split:
    full: UnitFrame
    dev: UnitFrame
    rest: UnitFrame
    folds: np.ndarray


def _base(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"u{i:05d}" for i in range(n)],
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.choice(np.array(["a", "b", "c", None], dtype=object), size=n),
            "tag": np.arange(n, dtype=np.float64),
        }
    )


def classification_frame(n: int, seed: int) -> UnitFrame:
    rng = np.random.default_rng(seed)
    df = _base(n, rng)
    logit = 2.0 * df["x1"] - df["x2"] + (df["cat"] == "a").astype(float)
    df["y"] = np.where(rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit)), "pos", "neg")
    return UnitFrame.from_pandas(df, Roles("id", "y", INPUTS))


def regression_frame(n: int, seed: int) -> UnitFrame:
    rng = np.random.default_rng(seed)
    df = _base(n, rng)
    noise = rng.normal(size=n) * (0.2 + np.abs(df["x2"].to_numpy()))
    df["y"] = 3.0 * df["x1"] + 5.0 + noise
    return UnitFrame.from_pandas(df, Roles("id", "y", INPUTS))


def outer_split(full: UnitFrame, k: int, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    order = rng.permutation(full.n)
    n_dev = int(0.6 * full.n)
    dev = full.take(np.sort(order[:n_dev]))
    rest = full.take(np.sort(order[n_dev:]))
    folds = rng.permutation(np.arange(dev.n) % k).astype(np.int64)
    return Split(full, dev, rest, folds)


@pytest.fixture(scope="module", autouse=True)
def _one_thread() -> Iterator[None]:
    """Small fits run faster without OpenMP thread start-up."""
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def clf_split() -> Split:
    return outer_split(classification_frame(1500, seed=11), k=5, seed=12)


@pytest.fixture(scope="module")
def reg_split() -> Split:
    return outer_split(regression_frame(700, seed=21), k=3, seed=22)
