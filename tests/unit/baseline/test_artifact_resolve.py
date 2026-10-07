from __future__ import annotations

import copy as copy_module
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from amx.baseline import (
    ArtifactError,
    TrivialClassifier,
    TrivialForecaster,
    TrivialRegressor,
    artifact_meta,
    compute_artifact_hash,
    forecast_roles,
    freeze,
    kfold_train_mask,
    load_artifact,
    make_forecast_units,
)
from amx.baseline import resolve as resolve_mod
from amx.cert import TauGrid
from amx.data import UnitFrame
from amx.loss import err_gt_tol, zero_one
from tests.unit.baseline.conftest import Split

REPO = Path(__file__).resolve().parents[3]
GRID = TauGrid(size=60)


def fit_clf(split: Split, seed: int) -> TrivialClassifier:
    return TrivialClassifier().fit(split.dev, split.folds, kfold_train_mask, zero_one(), seed)


def frozen(split: Split, out: Path, seed: int = 3) -> str:
    model = fit_clf(split, seed)
    meta = artifact_meta(model, spec_hash="sha256:spec", grid=GRID, seed=seed)
    return freeze(model, out, meta)


@pytest.fixture(scope="module")
def clf_artifact(clf_split: Split, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    out = tmp_path_factory.mktemp("art") / "artifact"
    return out, frozen(clf_split, out)


def test_same_seed_same_artifact_hash(
    clf_split: Split, clf_artifact: tuple[Path, str], tmp_path: Path
) -> None:
    _, digest = clf_artifact
    assert frozen(clf_split, tmp_path / "again") == digest
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64


def test_same_seed_same_regressor_artifact_hash(reg_split: Split, tmp_path: Path) -> None:
    hashes = []
    for name in ("a", "b"):
        model = TrivialRegressor().fit(
            reg_split.dev, reg_split.folds, kfold_train_mask, err_gt_tol(0.5), 9
        )
        meta = artifact_meta(model, spec_hash="sha256:spec", grid=GRID, seed=9)
        hashes.append(freeze(model, tmp_path / name, meta))
    assert hashes[0] == hashes[1]


_FREEZE_REGRESSOR = """
import sys
from pathlib import Path

from amx.baseline import TrivialRegressor, artifact_meta, freeze, kfold_train_mask
from amx.cert import TauGrid
from amx.loss import err_gt_tol
from tests.unit.baseline.conftest import outer_split, regression_frame

split = outer_split(regression_frame(700, seed=21), k=3, seed=22)
model = TrivialRegressor().fit(split.dev, split.folds, kfold_train_mask, err_gt_tol(0.5), 9)
meta = artifact_meta(model, spec_hash="sha256:spec", grid=TauGrid(size=60), seed=9)
sys.stdout.write(freeze(model, Path(sys.argv[1]), meta))
"""


def test_regressor_artifact_hash_does_not_depend_on_thread_count(tmp_path: Path) -> None:
    """The same fit frozen on hosts with different OpenMP thread counts has one hash."""
    hashes = []
    for threads in ("1", "4"):
        pythonpath = os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH")) if p)
        proc = subprocess.run(
            [sys.executable, "-c", _FREEZE_REGRESSOR, str(tmp_path / f"threads{threads}")],
            capture_output=True,
            text=True,
            cwd=REPO,
            env={**os.environ, "OMP_NUM_THREADS": threads, "PYTHONPATH": pythonpath},
            timeout=300,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        hashes.append(proc.stdout.strip())
    assert hashes[0].startswith("sha256:")
    assert hashes[0] == hashes[1]


def test_meta_contents(clf_artifact: tuple[Path, str], clf_split: Split) -> None:
    path, digest = clf_artifact
    meta = json.loads((path / "meta.json").read_text())
    assert meta["family"] == "classification"
    assert meta["roles"]["target"] == "y" and meta["roles"]["unit_id"] == "id"
    assert meta["grid_hash"] == GRID.hash
    assert len(meta["dev_cov"]) == GRID.size
    assert np.all(np.diff(meta["dev_cov"]) >= 0)
    assert {"python", "numpy", "scikit-learn", "joblib"} <= set(meta["versions"])
    assert meta["q13_fit"] == "bag_of_oof_fold_models"
    assert sorted(p.name for p in path.iterdir()) == [
        "artifact.sha256",
        "meta.json",
        "model.joblib",
    ]
    art = load_artifact(path, expected_hash=digest)
    model = fit_clf(clf_split, 3)
    assert np.array_equal(art.predictor.predict(clf_split.rest), model.predict(clf_split.rest))
    assert np.array_equal(
        art.predictor.commit_score(clf_split.rest), model.commit_score(clf_split.rest)
    )


def test_load_refuses_hash_mismatch(clf_artifact: tuple[Path, str], tmp_path: Path) -> None:
    path, digest = clf_artifact
    with pytest.raises(ArtifactError, match="expected"):
        load_artifact(path, expected_hash="sha256:" + "0" * 64)
    copy = tmp_path / "copy"
    copy.mkdir()
    for p in path.iterdir():
        (copy / p.name).write_bytes(p.read_bytes())
    assert compute_artifact_hash(copy) == digest
    meta = json.loads((copy / "meta.json").read_text())
    meta["spec_hash"] = "sha256:other"
    (copy / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactError, match="changed since freeze"):
        load_artifact(copy)
    (copy / "extra.txt").write_text("x")
    assert compute_artifact_hash(copy) != digest
    (copy / "meta.json").unlink()
    with pytest.raises(ArtifactError, match="missing"):
        load_artifact(copy)


@pytest.mark.parametrize("name", ["model.joblib", "meta.json"])
def test_load_uses_the_bytes_it_hashed(
    name: str,
    clf_split: Split,
    clf_artifact: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file swapped on disk right after it was hashed must not be the one loaded."""
    path, digest = clf_artifact
    copy = tmp_path / "copy"
    copy.mkdir()
    for p in path.iterdir():
        (copy / p.name).write_bytes(p.read_bytes())
    original = load_artifact(copy, expected_hash=digest)
    assert isinstance(original.predictor, TrivialClassifier)
    if name == "model.joblib":
        evil_model = copy_module.deepcopy(original.predictor)
        for pipe in evil_model.models_:  # every fold model now predicts one class, surely
            pipe.named_steps["model"].coef_[:] = 0.0
            pipe.named_steps["model"].intercept_[:] = 50.0
        buf = io.BytesIO()
        joblib.dump(evil_model, buf)
        evil = buf.getvalue()
    else:
        evil = json.dumps({**original.meta, "spec_hash": "sha256:evil"}).encode()
    real_open = Path.open
    swapped: list[Path] = []

    def swap_after_first_read(self: Path, *args: Any, **kwargs: Any) -> Any:
        mode = kwargs.get("mode", args[0] if args else "r")
        if self.name == name and mode == "rb" and not swapped:
            with real_open(self, "rb") as fh:
                data = fh.read()
            with real_open(self, "wb") as fh:
                fh.write(evil)
            swapped.append(self)
            return io.BytesIO(data)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_after_first_read)
    art = load_artifact(copy, expected_hash=digest)
    assert swapped, "the loader never read the file"
    assert (copy / name).read_bytes() == evil
    assert art.meta == original.meta
    rest = clf_split.rest
    assert np.array_equal(art.predictor.predict(rest), original.predictor.predict(rest))
    assert np.array_equal(art.predictor.commit_score(rest), original.predictor.commit_score(rest))
    assert len(set(original.predictor.predict(rest).tolist())) == 2


def test_freeze_refusals(clf_split: Split, clf_artifact: tuple[Path, str], tmp_path: Path) -> None:
    path, _ = clf_artifact
    model = fit_clf(clf_split, 3)
    meta = artifact_meta(model, spec_hash="sha256:spec", grid=GRID, seed=3)
    with pytest.raises(ArtifactError, match="non-empty"):
        freeze(model, path, meta)
    with pytest.raises(ArtifactError, match="missing"):
        freeze(model, tmp_path / "a", {k: v for k, v in meta.items() if k != "dev_cov"})
    with pytest.raises(ArtifactError, match="family"):
        freeze(model, tmp_path / "b", {**meta, "family": "regression"})
    with pytest.raises(ArtifactError, match="non-decreasing"):
        freeze(model, tmp_path / "c", {**meta, "dev_cov": [0.5, 0.2]})


@pytest.mark.parametrize(
    "extra",
    [
        {"created_at": "run-7"},
        {"frozen": "2026-10-07T12:30:00+00:00"},
        {"frozen": "2026-10-07"},
        {"run": {"Timestamp": 1760000000}},
        {"run": {"notes": ["ok", "20261007T123000"]}},
        {"lastUpdate": 3},
        {"versions": {"numpy": "2026-01-01"}},
    ],
)
def test_freeze_refuses_time_stamps_in_meta(
    clf_split: Split, tmp_path: Path, extra: dict[str, object]
) -> None:
    model = fit_clf(clf_split, 3)
    meta = artifact_meta(model, spec_hash="sha256:spec", grid=GRID, seed=3, extra=extra)
    with pytest.raises(ArtifactError, match="time stamp"):
        freeze(model, tmp_path / "a", meta)
    assert not (tmp_path / "a").exists()


def test_freeze_accepts_time_role_and_date_like_column_names(tmp_path: Path) -> None:
    """Roles name columns, not run metadata: a time role or a date-named column is fine."""
    rng = np.random.default_rng(5)
    y = 10.0 + rng.normal(size=200).cumsum()
    times = pd.date_range("2021-03-01", periods=y.size, freq="D")
    units = make_forecast_units(y, times, [1, 2], season=4, max_lag=4)
    uf = UnitFrame.from_pandas(units, forecast_roles())
    model = TrivialForecaster().fit(uf, np.zeros(uf.n, np.int64), kfold_train_mask, zero_one(), 0)
    meta = artifact_meta(model, spec_hash="sha256:spec", grid=GRID, seed=0)
    assert meta["roles"]["time"] == "origin_time"
    meta["roles"]["inputs"] = [*meta["roles"]["inputs"], "2021-03-01"]
    digest = freeze(model, tmp_path / "f", meta)
    assert load_artifact(tmp_path / "f", expected_hash=digest).meta["roles"]["time"] == (
        "origin_time"
    )


def write_inputs(split: Split, path: Path, *, with_target: bool = False) -> pd.DataFrame:
    df = split.rest.table.to_pandas()
    cols = ["id", "x1", "x2", "cat", "tag"] + (["y"] if with_target else [])
    df[cols].to_parquet(path, index=False)
    return df


def run_resolver(*args: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "amx.baseline.resolve", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ},
        timeout=120,
        check=False,
    )


def test_resolve_subprocess_round_trip(
    clf_split: Split, clf_artifact: tuple[Path, str], tmp_path: Path
) -> None:
    path, digest = clf_artifact
    inputs = tmp_path / "inputs.parquet"
    write_inputs(clf_split, inputs)
    out = tmp_path / "out" / "resolved.parquet"
    proc = run_resolver(path, inputs, out, "--expected-hash", digest)
    assert proc.returncode == 0, proc.stderr
    got = pq.read_table(out).to_pandas()
    assert list(got.columns) == ["unit_id", "value", "score"]
    art = load_artifact(path)
    assert got["unit_id"].tolist() == clf_split.rest.ids.tolist()
    assert got["value"].tolist() == art.predictor.predict(clf_split.rest).tolist()
    assert np.array_equal(got["score"].to_numpy(), art.predictor.commit_score(clf_split.rest))


def test_resolve_subprocess_empty_inputs_write_empty_output(
    clf_split: Split, clf_artifact: tuple[Path, str], tmp_path: Path
) -> None:
    path, digest = clf_artifact
    inputs = tmp_path / "inputs.parquet"
    empty = clf_split.rest.table.select(["id", "x1", "x2", "cat", "tag"]).slice(0, 0)
    pq.write_table(empty, inputs)
    out = tmp_path / "out" / "resolved.parquet"
    proc = run_resolver(path, inputs, out, "--expected-hash", digest)
    assert proc.returncode == resolve_mod.EXIT_OK, proc.stderr
    got = pq.read_table(out)
    assert got.num_rows == 0
    assert got.column_names == ["unit_id", "value", "score"]
    assert got.schema.field("unit_id").type == pa.string()
    assert got.schema.field("score").type == pa.float64()


def test_resolve_output_types_match_for_empty_and_full_batches(
    reg_split: Split, tmp_path: Path
) -> None:
    model = TrivialRegressor().fit(
        reg_split.dev, reg_split.folds, kfold_train_mask, err_gt_tol(0.5), 1
    )
    path = tmp_path / "reg"
    freeze(model, path, artifact_meta(model, spec_hash="sha256:spec", grid=GRID, seed=1))
    cols = ["id", "x1", "x2", "cat", "tag"]
    schemas = []
    for name, rows in (("full", reg_split.rest.n), ("empty", 0)):
        inputs, out = tmp_path / f"{name}.parquet", tmp_path / f"{name}_out.parquet"
        pq.write_table(reg_split.rest.table.select(cols).slice(0, rows), inputs)
        assert resolve_mod.main([str(path), str(inputs), str(out)]) == resolve_mod.EXIT_OK
        got = pq.read_table(out)
        assert got.num_rows == rows
        schemas.append(got.schema.remove_metadata())
    assert schemas[0] == schemas[1]


def test_resolve_subprocess_refuses_target_column(
    clf_split: Split, clf_artifact: tuple[Path, str], tmp_path: Path
) -> None:
    path, _ = clf_artifact
    inputs = tmp_path / "inputs.parquet"
    write_inputs(clf_split, inputs, with_target=True)
    out = tmp_path / "out.parquet"
    proc = run_resolver(path, inputs, out)
    assert proc.returncode == resolve_mod.EXIT_REFUSED
    assert "'y'" in proc.stderr and "target" in proc.stderr
    assert not out.exists()


def test_resolve_refuses_missing_inputs_and_wrong_hash(
    clf_split: Split, clf_artifact: tuple[Path, str], tmp_path: Path
) -> None:
    path, _ = clf_artifact
    inputs = tmp_path / "inputs.parquet"
    df = clf_split.rest.table.to_pandas()
    df[["id", "x1", "x2", "tag"]].to_parquet(inputs, index=False)
    out = tmp_path / "out.parquet"
    assert resolve_mod.main([str(path), str(inputs), str(out)]) == resolve_mod.EXIT_REFUSED
    write_inputs(clf_split, inputs)
    code = resolve_mod.main(
        [str(path), str(inputs), str(out), "--expected-hash", "sha256:" + "f" * 64]
    )
    assert code == resolve_mod.EXIT_ARTIFACT
    assert not out.exists()
    assert resolve_mod.main([str(path), str(inputs), str(out)]) == resolve_mod.EXIT_OK
    assert out.exists()
