from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from amx.data import Roles, UnitFrame
from amx.loss import err_gt_tol, zero_one
from amx.profile import feasibility, pre_profile, resolve_regime
from amx.spec import Regime, parse_taskspec


def spec(**over: Any):
    raw: dict[str, Any] = {
        "spec_version": 1,
        "name": "p",
        "data": {
            "uri": "x",
            "format": "csv",
            "unit_id": "id",
            "target": {"name": "y", "kind": "categorical"},
        },
        "task": {"family": "classification"},
        "bands": {"alphas": [0.005, 0.05], "policies": ["auto", "audit"]},
    }
    for k, v in over.items():
        raw[k] = {**raw.get(k, {}), **v}
    return parse_taskspec(raw)


def frame(n: int, extra: dict[str, Any], seed: int = 0) -> UnitFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"id": [f"u{i}" for i in range(n)], "a": rng.normal(size=n), **extra})
    df["y"] = rng.integers(0, 2, n)
    inputs = tuple(c for c in df.columns if c not in ("id", "y"))
    return UnitFrame.from_pandas(df, Roles("id", "y", inputs))


def test_iid_recommended_without_structure() -> None:
    pre = pre_profile(frame(5000, {"b": np.arange(5000) % 3}), spec())
    assert pre.recommended_regime is Regime.IID
    assert resolve_regime(spec(), pre) is Regime.IID
    assert pre.modality == "tabular"


def test_group_like_column_is_only_a_warning() -> None:
    """Undeclared id-like columns never switch the regime; the user must declare groups."""
    pre = pre_profile(frame(5000, {"entity": [f"e{i % 400}" for i in range(5000)]}), spec())
    assert "entity" in pre.group_candidates
    assert pre.recommended_regime is Regime.IID
    assert any("group_columns" in w for w in pre.warnings)


def test_time_like_column_warns() -> None:
    times = pd.date_range("2021-01-01", periods=4000, freq="min")
    pre = pre_profile(frame(4000, {"when": times}), spec())
    assert "when" in pre.time_candidates
    assert any("temporal" in w for w in pre.warnings)


def test_duplicates_and_small_n_reported() -> None:
    uf = frame(100, {"a": np.zeros(100)})
    pre = pre_profile(uf, spec())
    assert pre.exact_duplicate_units == 100
    assert any("allow-small" in w for w in pre.warnings)


def test_feasibility_flags_infeasible_band_and_constant() -> None:
    s = spec()
    y = np.array(["n"] * 990 + ["a"] * 10, dtype=object)
    tau = np.geomspace(1e-4, 1, 50)
    f = feasibility(
        s,
        regime=Regime.IID,
        dev_target=y,
        n_calib=300,
        loss_fn=zero_one(),
        loss_is_binary=True,
        tau=tau,
        dev_cov=np.clip(tau, 0, 1),
    )
    tight, loose = f.bands
    assert not tight.necessary_ok and f.requires_force
    assert loose.trivially_met_by_constant  # majority class error 1% < 5% band
    assert f.constant_risk == 0.01


def test_feasibility_detects_cap_and_floor() -> None:
    s = parse_taskspec(
        {
            "spec_version": 1,
            "name": "r",
            "data": {
                "uri": "x",
                "format": "csv",
                "unit_id": "id",
                "target": {"name": "y", "kind": "numeric"},
            },
            "task": {"family": "regression", "loss": {"params": {"tol": 0.1, "relative": True}}},
            "bands": {"alphas": [0.01], "policies": ["auto"]},
        }
    )
    rng = np.random.default_rng(0)
    y = np.r_[rng.uniform(1, 10, 950), np.full(50, 10.0)]
    f = feasibility(
        s,
        regime=Regime.IID,
        dev_target=y,
        n_calib=5000,
        loss_fn=err_gt_tol(0.1, relative=True),
        loss_is_binary=True,
        oof_scores=rng.uniform(size=1000),
        oof_losses=(rng.uniform(size=1000) < 0.3).astype(float),
    )
    assert any("cap" in w for w in f.warnings)
    assert f.bands[0].below_baseline_floor
    assert not f.requires_force  # a baseline limit is a warning, not infeasibility
