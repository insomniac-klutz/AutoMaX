"""Regression tests for the fix-verification findings (ledger, pin, resolver, digest, slices)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import yaml
from typer.testing import CliRunner

from amx.cert import DeltaBudget, GuaranteeType, TauGrid, build_certificate
from amx.cert.ltt import fixed_sequence_ltt, grid_stats
from amx.cert.slices import slice_risks
from amx.cli import app
from amx.data.unitframe import Roles, UnitFrame
from amx.spec.enums import Policy, Regime
from amx.split.vault import LocalVault
from amx.warden import WardenError, frozen_run
from amx.warden.ledger import MAX_OVERLAP, PartitionUsedError, claim_partition, fingerprints
from amx.warden.runner import run_resolver

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "e2e"))
from test_e2e_small import toy_classification

runner = CliRunner()


def _ok(*args: Any) -> str:
    res = runner.invoke(app, [str(a) for a in args])
    assert res.exit_code == 0, res.output
    return str(res.stdout)


def _run(tmp_path: Path, name: str, n: int = 4000, mutate: Any = None) -> tuple[Path, Path]:
    d = tmp_path / f"data-{name}"
    d.mkdir()
    spec = toy_classification(d, n=n)
    if mutate is not None:
        mutate(d, spec)
    run = tmp_path / "runs" / name
    _ok("split", "--spec", spec, "--run", run)
    _ok("baseline", "--run", run)
    return spec, run


def test_ledger_refuses_overlapping_calibration_data(tmp_path: Path) -> None:
    vault = LocalVault(tmp_path / "v")
    ids = [f"u{i}" for i in range(1000)]
    gold = [i % 2 for i in range(1000)]
    claim_partition(vault, fingerprints(ids, gold), "r1")
    shifted = ids[5:] + [f"x{i}" for i in range(5)]  # 99.5% overlap
    with pytest.raises(PartitionUsedError, match="already certified"):
        claim_partition(vault, fingerprints(shifted, gold[5:] + [0] * 5), "r2")
    fresh = [f"z{i}" for i in range(1000)]
    claim_partition(vault, fingerprints(fresh, gold), "r3")
    assert MAX_OVERLAP < 0.05


def test_added_column_does_not_dodge_the_ledger(tmp_path: Path) -> None:
    def add_const(d: Path, spec: Path) -> None:
        df = pd.read_parquet(d / "toy.parquet")
        df["zz"] = 1.0
        df.to_parquet(d / "toy.parquet", index=False)

    _, first = _run(tmp_path, "first")
    tok = _ok("warden", "issue-token", "--run", first).strip()
    _ok("certify", "--run", first, "--freeze", "--token", tok)
    _, second = _run(tmp_path, "second", mutate=add_const)
    tok2 = _ok("warden", "issue-token", "--run", second).strip()
    res = runner.invoke(app, ["certify", "--run", str(second), "--freeze", "--token", tok2])
    assert res.exit_code == 2 and "already certified" in res.output


def test_pin_is_kept_and_a_swapped_artifact_is_refused_after_certify(tmp_path: Path) -> None:
    _, run = _run(tmp_path, "pin")
    tok = _ok("warden", "issue-token", "--run", run).strip()
    cert = json.loads(_ok("certify", "--run", run, "--freeze", "--token", tok))
    assert cert["bands"]
    vault = LocalVault()
    fr = frozen_run(run, vault, use_pin=True)
    assert fr.pinned and str(fr.artifact_dir).startswith(str(vault.root))
    meta = run / "artifact" / "meta.json"
    payload = json.loads(meta.read_text())
    payload["seed"] = 999
    meta.write_text(json.dumps(payload))
    from amx.baseline.artifact import HASH_FILE, compute_artifact_hash

    (run / "artifact" / HASH_FILE).write_text(compute_artifact_hash(run / "artifact") + "\n")
    with pytest.raises(WardenError, match="differs from the certified"):
        frozen_run(run, vault, use_pin=True)
    res = runner.invoke(
        app, ["simulate", "--t1", "--real", "--run", str(run), "--token", tok, "--resplits", "5"]
    )
    assert res.exit_code == 2 and "certified" in res.output


def test_resolver_never_sees_a_vault_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        seen["cmd"], seen["cwd"], seen["env"] = cmd, kw.get("cwd"), kw.get("env")
        raise RuntimeError("stop")

    monkeypatch.setattr(subprocess, "run", fake_run)
    vault = LocalVault(tmp_path / "vault")
    art = vault.run_path("r") / "pinned_artifact"
    art.mkdir(parents=True)
    (art / "meta.json").write_text("{}")
    uf = UnitFrame(pa.table({"id": ["a"], "x": [1.0]}), Roles("id", None, ("x",)))
    with pytest.raises(RuntimeError, match="stop"):
        run_resolver(art, uf, expected_hash="sha256:x")
    root = str(vault.root)
    assert "-I" in seen["cmd"]
    assert not any(root in str(c) for c in seen["cmd"])
    assert root not in str(seen["cwd"]) and root not in seen["env"]["HOME"]
    assert "AMX_VAULT" not in seen["env"]


def test_calib_digest_mismatch_is_refused(tmp_path: Path) -> None:
    from amx.warden.common import load_calib

    _, run = _run(tmp_path, "digest")
    vault = LocalVault()
    fr = frozen_run(run, vault)
    try:
        bad = fr.manifest.model_copy(update={"calib_digest": "hmac-sha256:" + "0" * 64})
        fr.manifest = bad
        with pytest.raises(WardenError, match="digest"):
            load_calib(vault, fr)
    finally:
        fr.cleanup()


def test_auto_regime_with_undeclared_id_columns_fails_closed(tmp_path: Path) -> None:
    def add_entity(d: Path, spec: Path) -> None:
        df = pd.read_parquet(d / "toy.parquet")
        df["entity"] = [f"e{i % 300}" for i in range(len(df))]
        df.to_parquet(d / "toy.parquet", index=False)
        raw = yaml.safe_load(spec.read_text())
        raw["splits"]["regime"] = "auto"
        spec.write_text(yaml.safe_dump(raw))

    d = tmp_path / "data"
    d.mkdir()
    spec = toy_classification(d, n=4000)
    add_entity(d, spec)
    res = runner.invoke(app, ["split", "--spec", str(spec), "--run", str(tmp_path / "r")])
    assert res.exit_code == 2 and "group_columns" in res.output


def test_group_mode_slices_have_group_level_bounds_and_flags() -> None:
    labels = np.array(["a"] * 400 + ["b"] * 400)
    groups = np.repeat(np.arange(160), 5).astype(str)
    losses = np.r_[np.zeros(400), (np.arange(400) % 2).astype(float)]
    out = slice_risks(
        "s", labels, losses, np.ones(800, bool), alpha=0.05, binary=True, groups=groups
    )
    by = {s.value: s for s in out}
    assert by["a"].n == 80 and by["a"].bound == "hoeffding_bentkus_groups"
    assert by["b"].flag == "upper_gt_2alpha"


def test_coverage_interval_is_dkw_only_for_unit_pac_certificates() -> None:
    rng = np.random.default_rng(0)
    s = rng.uniform(size=3000)
    L = (rng.uniform(size=3000) < 0.02 * s).astype(float)
    grid = TauGrid(size=50)
    budget = DeltaBudget(0.1, 1)

    def cert(gtype: GuaranteeType, groups: Any = None) -> str:
        st = grid_stats(s, L, grid.values, binary=groups is None, groups=groups)
        res = fixed_sequence_ltt(st, [0.05], budget.delta_per_band, grid.values)
        c = build_certificate(
            res,
            policies=[Policy.AUTO],
            budget=budget,
            grid=grid,
            regime=Regime.IID if groups is None else Regime.GROUPED,
            guarantee_type=gtype,
            run_id="r",
            taskspec_hash="h",
            artifact_hash="a",
            amx_version="0",
        )
        return c.bands[0].coverage_at_certified_tau.method

    assert cert(GuaranteeType.PAC_HIGH_PROB) == "dkw_uniform_in_tau"
    assert cert(GuaranteeType.HOLDOUT_EMPIRICAL) == "descriptive"
    assert cert(GuaranteeType.PAC_HIGH_PROB, groups=np.arange(3000) // 3) == "descriptive"


def test_ledger_reads_key_only_records(tmp_path: Path) -> None:
    from amx.warden.ledger import INDEX_FILE, LEDGER_DIR, partition_key

    vault = LocalVault(tmp_path / "v")
    fps = fingerprints(["a", "b"], [0, 1])
    d = vault.root / LEDGER_DIR
    d.mkdir(parents=True)
    (d / INDEX_FILE).write_text(json.dumps({"key": "sha256:other", "run_id": "old"}) + "\n")
    claim_partition(vault, fps, "new")  # unrelated old record: accepted
    (d / INDEX_FILE).write_text(json.dumps({"key": partition_key(fps), "run_id": "old"}) + "\n")
    with pytest.raises(PartitionUsedError):
        claim_partition(vault, fps, "again")
