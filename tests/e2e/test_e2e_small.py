"""Offline end-to-end runs on generated toy data through the real CLI (make e2e-small).

Covers the human gates: certify without --freeze or without the token is refused, the certify
budget is one call per calib fold (Q1), and nothing from calib or sealed lands in the run dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from amx.baseline.trivial import make_forecast_units
from amx.cli import app
from amx.split.vault import LocalVault

runner = CliRunner()


def _invoke(*args: str, ok: bool = True) -> Any:
    res = runner.invoke(app, [str(a) for a in args])
    if ok:
        assert res.exit_code == 0, res.output
    return res


def _write_spec(path: Path, raw: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def toy_classification(tmp: Path, n: int = 6000) -> Path:
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["p", "q", "r"], size=n)
    logit = 2.5 * x1 - 1.5 * x2 + (cat == "r") * 1.0
    y = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    df = pd.DataFrame(
        {"uid": [f"u{i:05d}" for i in range(n)], "x1": x1, "x2": x2, "c": cat, "y": y}
    )
    df.to_parquet(tmp / "toy.parquet", index=False)
    return _write_spec(
        tmp / "spec.yaml",
        {
            "spec_version": 1,
            "name": "toy-cls",
            "data": {
                "uri": "toy.parquet",
                "format": "parquet",
                "unit_id": "uid",
                "target": {"name": "y", "kind": "categorical"},
            },
            "task": {"family": "classification"},
            "bands": {
                "alphas": [0.02, 0.05],
                "policies": ["auto", "audit"],
                "delta": 0.1,
                "watch_slices": [{"name": "by_c", "by": "c"}],
            },
            "splits": {"regime": "iid", "seed": 11},
        },
    )


def toy_grouped_regression(tmp: Path, n_groups: int = 1500) -> Path:
    rng = np.random.default_rng(1)
    sizes = 1 + rng.poisson(3, n_groups)
    gid = np.repeat(np.arange(n_groups), sizes)
    n = gid.size
    x = rng.uniform(0, 1, n)
    effect = rng.normal(0, 0.1, n_groups)[gid]
    y = 2 + np.sin(4 * x) + effect + rng.normal(0, 0.05 + 0.3 * x, n)
    df = pd.DataFrame(
        {"uid": [f"r{i:06d}" for i in range(n)], "g": [f"g{k}" for k in gid], "x": x, "y": y}
    )
    df.to_csv(tmp / "reg.csv", index=False)
    return _write_spec(
        tmp / "spec_reg.yaml",
        {
            "spec_version": 1,
            "name": "toy-reg",
            "data": {
                "uri": "reg.csv",
                "format": "csv",
                "unit_id": "uid",
                "inputs": [{"name": "x", "kind": "numeric"}],
                "target": {"name": "y", "kind": "numeric"},
                "group_columns": ["g"],
                "independence_unit": "group:g",
            },
            "task": {"family": "regression", "loss": {"params": {"tol": 0.25, "relative": True}}},
            "bands": {"alphas": [0.05, 0.2], "policies": ["auto", "audit"], "delta": 0.1},
            "splits": {"regime": "grouped", "seed": 12},
        },
    )


def toy_forecast(tmp: Path, n: int = 4000) -> Path:
    rng = np.random.default_rng(2)
    t = np.arange(n)
    y = 50 + 10 * np.sin(2 * np.pi * t / 24) + np.cumsum(rng.normal(0, 0.3, n)) * 0.1
    y = y + rng.normal(0, 1.0, n)
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    units = make_forecast_units(y, times, [1, 6], 24, 24)
    units.to_parquet(tmp / "fc.parquet", index=False)
    return _write_spec(
        tmp / "spec_fc.yaml",
        {
            "spec_version": 1,
            "name": "toy-fc",
            "data": {
                "uri": "fc.parquet",
                "format": "parquet",
                "unit_id": "unit_id",
                "inputs": [
                    {"name": "anchor", "kind": "numeric"},
                    {"name": "vol", "kind": "numeric"},
                    {"name": "horizon", "kind": "numeric"},
                ],
                "target": {"name": "target", "kind": "series"},
                "time_column": "origin_time",
            },
            "task": {
                "family": "forecasting",
                "loss": {"params": {"tol": 0.1, "relative": True, "abs_floor": 1.0}},
                "forecast": {"horizons": [1, 6], "max_lag": 24},
            },
            "bands": {"alphas": [0.05, 0.1], "policies": ["auto", "audit"], "delta": 0.1},
            "splits": {"regime": "temporal", "seed": 13},
        },
    )


def _full_run(spec: Path, run: Path) -> dict[str, Any]:
    _invoke("profile", "--spec", spec)
    split = json.loads(_invoke("split", "--spec", spec, "--run", run).stdout)
    _invoke("baseline", "--run", run)
    feas = json.loads(_invoke("profile", "--run", run).stdout)
    assert feas["n_calib"] == split["counts"]["calib"]

    refused = _invoke("certify", "--run", run, ok=False)
    assert refused.exit_code == 2  # no --freeze
    refused = _invoke("certify", "--run", run, "--freeze", ok=False)
    assert refused.exit_code == 2 and "token" in refused.output  # no token
    token = _invoke("warden", "issue-token", "--run", run).stdout.strip()
    bad = _invoke("certify", "--run", run, "--freeze", "--token", "wrong", ok=False)
    assert bad.exit_code == 2

    cert = json.loads(_invoke("certify", "--run", run, "--freeze", "--token", token).stdout)
    again = _invoke("certify", "--run", run, "--freeze", "--token", token, ok=False)
    assert again.exit_code == 2  # one certify call per calib fold (Q1)

    rep = json.loads(_invoke("report", "--run", run).stdout)
    assert Path(rep["bands_md"]).read_text(encoding="utf-8").startswith("#")
    assert Path(rep["frontier_png"]).stat().st_size > 0

    sim = _invoke(
        "simulate", "--t1", "--real", "--run", run, "--token", token, "--resplits", "20", ok=False
    )
    assert sim.exit_code in (0, 1), sim.output
    payload = json.loads(sim.stdout)
    assert payload["kind"] == "t1_real_cheap" and payload["resplits"] == 20
    return {"split": split, "cert": cert, "t1": payload}


def _no_held_out_ids_in_run_dir(run: Path, vault_root: Path) -> None:
    held: set[str] = set()
    vault = LocalVault(vault_root)
    rid = run.resolve().name
    for fold in ("calib", "sealed"):
        held |= set(vault.read_fold(rid, fold).ids.tolist())
    dev_ids = set(pq.read_table(run / "data" / "dev.parquet").column(0).to_pylist())
    assert not held & dev_ids
    for path in run.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".yaml"}:
            text = path.read_text(encoding="utf-8")
            leaked = [i for i in list(held)[:200] if f'"{i}"' in text]
            assert not leaked, (path, leaked[:3])


@pytest.mark.parametrize("make", [toy_classification, toy_grouped_regression, toy_forecast])
def test_end_to_end(make: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    spec = make(data_dir)
    run = tmp_path / "runs" / f"run-{make.__name__}"
    out = _full_run(spec, run)
    cert = json.loads((run / "cert" / "certificate.json").read_text(encoding="utf-8"))
    assert cert["guarantee"]["certify_calls_used"] == 1
    if make is toy_forecast:
        assert cert["guarantee"]["type"] == "holdout_empirical"
        md = (run / "report" / "bands.md").read_text(encoding="utf-8").lower()
        assert "certified" not in md
    else:
        assert cert["guarantee"]["type"] == "pac_high_prob"
    if make is toy_grouped_regression:
        assert cert["guarantee"]["estimand"] == "group_weighted"
        assert cert["guarantee"]["p_value_family"] == "hoeffding_bentkus"
    import os

    _no_held_out_ids_in_run_dir(run, Path(os.environ["AMX_VAULT"]))
    assert out["split"]["counts"]["calib"] > 0
