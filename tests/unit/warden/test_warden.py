from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from amx.cli import app
from amx.data.unitframe import Roles, UnitFrame
from amx.split.vault import LocalVault
from amx.warden import (
    ResolverError,
    TokenError,
    WardenError,
    frozen_run,
    issue_token,
    run_resolver,
    verify_token,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "e2e"))
from test_e2e_small import toy_classification

runner = CliRunner()


def test_token_round_trip(tmp_path: Path) -> None:
    vault = LocalVault(tmp_path / "v")
    with pytest.raises(TokenError, match="no freeze token"):
        verify_token(vault, "r1", "anything")
    tok = issue_token(vault, "r1")
    verify_token(vault, "r1", tok)
    with pytest.raises(TokenError):
        verify_token(vault, "r1", tok + "x")
    with pytest.raises(TokenError):
        verify_token(vault, "r1", None)
    assert tok not in vault.read_text("r1", "freeze_token.sha256")


def test_resolver_refuses_a_target_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0], "y": [0, 1]})
    uf = UnitFrame.from_pandas(df, Roles("id", "y", ("x",)))
    with pytest.raises(ResolverError, match="target"):
        run_resolver(tmp_path, uf, tmp_path / "w", expected_hash="sha256:x")


@pytest.fixture
def frozen(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    spec = toy_classification(d, n=4000)
    run = tmp_path / "runs" / "rw"
    for args in (
        ["split", "--spec", str(spec), "--run", str(run)],
        ["baseline", "--run", str(run)],
    ):
        res = runner.invoke(app, args)
        assert res.exit_code == 0, res.output
    return run


def test_frozen_run_reads_vault_snapshot_and_detects_tampering(frozen: Path) -> None:
    vault = LocalVault()
    fr = frozen_run(frozen, vault)
    assert fr.grid.size == len(fr.dev_cov)
    # editing the agent-visible spec copy does not change what the warden certifies
    (frozen / "taskspec.yaml").write_text("garbage: [", encoding="utf-8")
    assert frozen_run(frozen, vault).spec_hash == fr.spec_hash
    meta = frozen / "artifact" / "meta.json"
    payload = json.loads(meta.read_text())
    payload["dev_cov"] = list(np.minimum(np.asarray(payload["dev_cov"]) * 2, 1.0))
    meta.write_text(json.dumps(payload))
    with pytest.raises(WardenError, match="changed after freeze"):
        frozen_run(frozen, vault)


def test_doctor_json() -> None:
    res = runner.invoke(app, ["doctor", "--json"])
    payload = json.loads(res.stdout)
    assert payload["vault"]["mode"] == "local_dir"
    assert any("Not safe" in w for w in payload["warnings"])
