"""Offline end-to-end runs on generated toy data through the real CLI (make e2e-small).

Covers the human gates: certify without --freeze or without the token is refused, the certify
budget is one call per calib fold (Q1), and nothing from calib or sealed lands in the run dir.
"""

from __future__ import annotations

import json
import os
import re
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
            "bands": {
                "alphas": [0.05, 0.2],
                "policies": ["auto", "audit"],
                "delta": 0.1,
                "watch_slices": [{"name": "by_target", "by": "target"}],
            },
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
    indep = split["independent_counts"]
    assert feas["n_calib"] == (indep["calib"] if indep else split["counts"]["calib"])

    refused = _invoke("certify", "--run", run, ok=False)
    assert refused.exit_code == 2  # no --freeze
    refused = _invoke("certify", "--run", run, "--freeze", ok=False)
    assert refused.exit_code == 2 and "token" in refused.output  # no token
    token = _invoke("warden", "issue-token", "--run", run).stdout.strip()
    again_token = _invoke("warden", "issue-token", "--run", run, ok=False)
    assert again_token.exit_code == 2  # re-issue needs --rotate
    bad = _invoke("certify", "--run", run, "--freeze", "--token", "wrong", ok=False)
    assert bad.exit_code == 2
    early = _invoke(
        "simulate", "--t1", "--real", "--run", run, "--token", token, "--resplits", "5", ok=False
    )
    assert early.exit_code == 2 and "budget" in early.output  # calib budget still open

    cert = json.loads(_invoke("certify", "--run", run, "--freeze", "--token", token).stdout)
    again = _invoke("certify", "--run", run, "--freeze", "--token", token, ok=False)
    assert again.exit_code == 2  # one certify call per calib fold (Q1)
    full = json.loads((run / "cert" / "certificate.json").read_text(encoding="utf-8"))
    for band in full["bands"]:
        if band["tau_hat"] is not None and band["status"] == "certified":
            # rejecting H0: R >= alpha at level delta_j implies an empirical risk below alpha
            assert band["risk_calib"]["est"] < band["alpha"]

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


def _no_held_out_values_in_run_dir(run: Path, vault_root: Path) -> None:
    """No calib/sealed unit id, and no full-precision calib/sealed numeric gold value, in any
    text file of the run directory (HANDOFF 8.4: only aggregates leave the warden)."""
    vault = LocalVault(vault_root)
    rid = run.resolve().name
    held_ids: set[str] = set()
    held_vals: set[str] = set()
    for fold in ("calib", "sealed"):
        uf = vault.read_fold(rid, fold)
        held_ids |= set(uf.ids.tolist())
        y = uf.target
        if y.dtype.kind == "f":
            held_vals |= {repr(float(v)) for v in y if len(repr(float(v))) >= 7}
    dev_ids = set(pq.read_table(run / "data" / "dev.parquet").column(0).to_pylist())
    assert not held_ids & dev_ids
    token = re.compile(r"[A-Za-z0-9_.+-]+")
    for path in run.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".yaml", ".txt"}:
            words = set(token.findall(path.read_text(encoding="utf-8")))
            assert not words & held_ids, (path, sorted(words & held_ids)[:3])
            assert not words & held_vals, (path, sorted(words & held_vals)[:3])


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
    _no_held_out_values_in_run_dir(run, Path(os.environ["AMX_VAULT"]))
    assert out["split"]["counts"]["calib"] > 0


def _ready(spec: Path, run: Path, *extra: str) -> str:
    _invoke("split", "--spec", spec, "--run", run)
    _invoke("baseline", "--run", run, *extra)
    return str(_invoke("warden", "issue-token", "--run", run).stdout.strip())


def test_resplit_into_a_new_run_cannot_recertify_the_same_partition(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    spec = toy_classification(data, n=4000)
    first, second = tmp_path / "runs" / "first", tmp_path / "runs" / "second"
    tok = _ready(spec, first)
    _invoke("certify", "--run", first, "--freeze", "--token", tok)
    tok2 = _ready(spec, second)  # deterministic split: same calibration partition
    res = _invoke("certify", "--run", second, "--freeze", "--token", tok2, ok=False)
    assert res.exit_code == 2 and "already certified" in res.output


def test_infeasible_bands_need_force(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    spec_path = toy_classification(data, n=4000)
    raw = yaml.safe_load(spec_path.read_text())
    raw["bands"]["alphas"], raw["bands"]["policies"] = [0.0005, 0.05], ["auto", "audit"]
    spec_path.write_text(yaml.safe_dump(raw))
    run = tmp_path / "runs" / "tight"
    tok = _ready(spec_path, run)
    res = _invoke("certify", "--run", run, "--freeze", "--token", tok, ok=False)
    assert res.exit_code == 2 and "--force" in res.output
    forced = json.loads(
        _invoke("certify", "--run", run, "--freeze", "--token", tok, "--force").stdout
    )
    assert any(w.startswith("forced:") for w in forced["warnings"])


def test_declared_time_column_downgrades_an_iid_certificate(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    spec_path = toy_classification(data, n=4000)
    df = pd.read_parquet(data / "toy.parquet")
    df["when"] = pd.date_range("2022-01-01", periods=len(df), freq="h")
    df.to_parquet(data / "toy.parquet", index=False)
    raw = yaml.safe_load(spec_path.read_text())
    raw["data"]["time_column"] = "when"
    spec_path.write_text(yaml.safe_dump(raw))  # regime stays iid as declared
    run = tmp_path / "runs" / "timed"
    tok = _ready(spec_path, run)
    out = json.loads(_invoke("certify", "--run", run, "--freeze", "--token", tok).stdout)
    assert out["guarantee"] == "holdout_empirical" and not out["claims_certification"]
    assert any("differs from the audit" in w for w in out["warnings"])
    assert all(b["status"] != "certified" for b in out["bands"])


def test_custom_loss_is_pinned_in_the_vault(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    spec_path = toy_grouped_regression(data, n_groups=900)
    (data / "losses").mkdir()
    (data / "losses" / "my_loss.py").write_text(
        "import numpy as np\n"
        "def loss(p, g):\n"
        "    return np.minimum(np.abs(np.asarray(p, float) - np.asarray(g, float)), 1.0)\n"
    )
    raw = yaml.safe_load(spec_path.read_text())
    raw["task"]["loss"] = {"kind": "custom", "path": "losses/my_loss.py", "fn": "loss"}
    raw["bands"]["watch_slices"] = []
    spec_path.write_text(yaml.safe_dump(raw))
    run = tmp_path / "runs" / "custom"
    refused = _invoke("split", "--spec", spec_path, "--run", run)  # split needs no confirmation
    assert refused.exit_code == 0
    assert _invoke("baseline", "--run", run, ok=False).exit_code == 2  # needs --confirm-loss
    _invoke("baseline", "--run", run, "--confirm-loss")
    tok = str(_invoke("warden", "issue-token", "--run", run).stdout.strip())
    # editing the run-dir copy after split does not change what the warden loads
    (run / "losses" / "my_loss.py").write_text("def loss(p, g):\n    return [0.0] * len(g)\n")
    vault = LocalVault()
    rid = run.resolve().name
    vault_copy = vault.file_path(rid, "custom_loss.py")
    original = vault_copy.read_text()
    vault_copy.write_text(original + "\n# tampered\n")
    res = _invoke(
        "certify", "--run", run, "--freeze", "--token", tok, "--confirm-loss", "--force", ok=False
    )
    assert res.exit_code == 2 and "changed after split" in res.output
    vault_copy.write_text(original)
    cert = json.loads(
        _invoke(
            "certify", "--run", run, "--freeze", "--token", tok, "--confirm-loss", "--force"
        ).stdout
    )
    full = json.loads((run / "cert" / "certificate.json").read_text())
    assert any("custom loss source sha256:" in a for a in full["guarantee"]["assumptions"])
    assert cert["guarantee"] == "pac_high_prob"
