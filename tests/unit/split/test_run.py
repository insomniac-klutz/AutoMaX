from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from amx.data import Roles, UnitFrame
from amx.spec import TaskSpec, load_taskspec, parse_taskspec
from amx.split import (
    OOF_COLUMN,
    TRAIN_ONLY,
    Fold,
    LocalVault,
    SplitError,
    SplitManifest,
    load_dev,
    oof_folds,
    read_manifest,
    split_run,
)
from tests.unit.split.helpers import make_frame, make_spec

KEYED = {"calib_digest", "sealed_digest", "digest_key_id"}


@pytest.fixture
def vault(tmp_path: Path) -> LocalVault:
    return LocalVault(tmp_path / "vault")


def test_split_run_layout_and_load_dev(tmp_path: Path, vault: LocalVault) -> None:
    uf = make_frame(400, dup_of={11: 2, 12: 2})
    spec = make_spec(seed=3)
    rd = tmp_path / "runs" / "r1"
    res = split_run(spec, uf, rd, vault, "r1")

    files = sorted(str(p.relative_to(rd)) for p in rd.rglob("*") if p.is_file())
    assert files == ["data/dev.parquet", "splits.manifest.json", "taskspec.yaml"]
    assert res.dev_path == rd / "data" / "dev.parquet"
    assert load_taskspec(rd / "taskspec.yaml") == spec
    assert vault.read_text("r1", "taskspec.yaml") == (rd / "taskspec.yaml").read_text()
    assert read_manifest(rd / "splits.manifest.json") == res.manifest
    assert SplitManifest.model_validate_json(vault.read_text("r1", "splits.manifest.json"))

    dev, oof = load_dev(rd)
    assert dev.roles == uf.roles
    assert dev.n == res.manifest.counts.dev == res.assignment.counts["dev"]
    assert np.array_equal(oof, res.oof)
    assert OOF_COLUMN not in dev.table.column_names
    assert set(oof.tolist()) == set(range(spec.splits.oof_folds))
    recomputed = oof_folds(dev, spec.splits.oof_folds, "iid", spec.splits.seed)
    assert np.array_equal(recomputed, oof)

    calib, sealed = vault.read_fold("r1", "calib"), vault.read_fold("r1", "sealed")
    assert calib.n == res.manifest.counts.calib and sealed.n == res.manifest.counts.sealed
    assert calib.has_target and calib.roles == uf.roles
    all_ids = set(dev.ids) | set(calib.ids) | set(sealed.ids)
    assert len(all_ids) == uf.n


def test_split_run_writes_nothing_from_calib_or_sealed_under_run_dir(
    tmp_path: Path, vault: LocalVault
) -> None:
    uf = make_frame(500, seed=4)
    rd = tmp_path / "run"
    res = split_run(make_spec(seed=9), uf, rd, vault, "r2")
    held_mask = res.assignment.mask(Fold.CALIB) | res.assignment.mask(Fold.SEALED)
    held_ids = set(uf.ids[held_mask].tolist())
    assert len(held_ids) == 200
    dev_ids = set(uf.ids[res.assignment.mask(Fold.DEV)].tolist())
    for path in rd.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".parquet":
            ids = set(pq.read_table(path).column("id").to_pylist())
            assert ids == dev_ids
            assert not ids & held_ids
        else:
            text = path.read_text()
            assert not any(i in text for i in held_ids), path
            assert not any(i in text for i in dev_ids), path
    # held-out ids live in the vault only
    assert held_ids == set(vault.read_fold("r2", "calib").ids) | set(
        vault.read_fold("r2", "sealed").ids
    )


def test_same_seed_reproduces_split_and_different_seed_does_not(
    tmp_path: Path, vault: LocalVault
) -> None:
    uf = make_frame(300, seed=1)
    a = split_run(make_spec(seed=21), uf, tmp_path / "a", vault, "a")
    b = split_run(make_spec(seed=21), uf.take(np.arange(uf.n)[::-1]), tmp_path / "b", vault, "b")
    c = split_run(make_spec(seed=22), uf, tmp_path / "c", vault, "c")
    da, db, dc = (m.manifest.model_dump(mode="json") for m in (a, b, c))
    # the HMAC key is per run (C13), so only the keyed digests differ between runs
    assert {k for k in da if da[k] != db[k]} == KEYED
    assert {k: v for k, v in da.items() if k not in KEYED} == {
        k: v for k, v in db.items() if k not in KEYED
    }
    assert da["dev_ids_sha256"] != dc["dev_ids_sha256"]
    dev_a, oof_a = load_dev(tmp_path / "a")
    dev_b, oof_b = load_dev(tmp_path / "b")
    assert dict(zip(dev_a.ids, oof_a, strict=True)) == dict(zip(dev_b.ids, oof_b, strict=True))


def test_reproducibility_hash_is_equal_for_identical_splits(
    tmp_path: Path, vault: LocalVault
) -> None:
    """Decision D20: two split_run calls with the same spec, data and seed agree."""
    uf = make_frame(300, seed=1, dup_of={5: 2})
    a = split_run(make_spec(seed=21), uf, tmp_path / "a", vault, "a")
    b = split_run(make_spec(seed=21), uf.take(np.arange(uf.n)[::-1]), tmp_path / "b", vault, "b")
    c = split_run(make_spec(seed=22), uf, tmp_path / "c", vault, "c")
    assert a.manifest.content_hash() != b.manifest.content_hash()  # per-run HMAC keys
    assert a.manifest.reproducibility_hash() == b.manifest.reproducibility_hash()
    assert a.manifest.reproducibility_hash() != c.manifest.reproducibility_hash()
    on_disk = read_manifest(tmp_path / "b" / "splits.manifest.json")
    assert on_disk.reproducibility_hash() == a.manifest.reproducibility_hash()


def test_temporal_split_run_has_training_only_block(tmp_path: Path, vault: LocalVault) -> None:
    t = np.repeat(np.arange(150), 2)
    uf = make_frame(t.size, time=t.tolist(), series=[f"s{i % 3}" for i in range(t.size)])
    spec = make_spec("temporal", time_column="t", series_columns=["s"], embargo=2)
    res = split_run(spec, uf, tmp_path / "run", vault, "rt")
    assert res.manifest.oof_scheme == "forward_chaining"
    assert res.manifest.embargo_steps == 2 and res.manifest.counts.dropped == 8
    dev, oof = load_dev(tmp_path / "run")
    assert TRAIN_ONLY in oof.tolist()
    t_dev = dev.time
    assert t_dev is not None
    assert t_dev[oof == TRAIN_ONLY].max() < t_dev[oof == 0].min()
    calib = vault.read_fold("rt", "calib")
    assert calib.time is not None and t_dev.max() < calib.time.min()


def test_split_run_refusals(tmp_path: Path, vault: LocalVault) -> None:
    uf = make_frame(200)
    spec = make_spec()
    split_run(spec, uf, tmp_path / "r", vault, "r")
    with pytest.raises(SplitError, match="already split"):
        split_run(spec, uf, tmp_path / "r", vault, "r-new")
    with pytest.raises(SplitError, match="vault already holds"):
        split_run(spec, uf, tmp_path / "other", vault, "r")
    with pytest.raises(SplitError, match="contain each other"):
        split_run(spec, uf, vault.root / "inside", vault, "x")
    with pytest.raises(SplitError, match="contain each other"):
        split_run(spec, uf, tmp_path, vault, "x")
    with pytest.raises(SplitError, match="profiler"):
        split_run(make_spec("auto"), uf, tmp_path / "auto", vault, "auto")
    with pytest.raises(SplitError, match="roles"):
        split_run(spec, uf.without_target(), tmp_path / "nt", vault, "nt")


def explicit_inputs_spec(*names: str) -> TaskSpec:
    raw = make_spec().model_dump(mode="json")
    raw["data"]["inputs"] = [{"name": n, "kind": "numeric"} for n in names]
    return parse_taskspec(raw)


@pytest.mark.parametrize(
    ("frame_kw", "spec", "field"),
    [
        ({"time": list(range(200))}, make_spec(), "time"),  # frame has a time role, spec none
        ({"series": [f"s{i % 4}" for i in range(200)]}, make_spec(), "series_columns"),
        ({}, explicit_inputs_spec("a"), "inputs"),  # frame inputs (a, b), spec inputs (a,)
        ({}, explicit_inputs_spec("b", "a"), "inputs"),  # same names, other order
    ],
    ids=["time", "series", "inputs", "input-order"],
)
def test_split_run_requires_the_roles_of_spec_data(
    tmp_path: Path, vault: LocalVault, frame_kw: dict[str, Any], spec: TaskSpec, field: str
) -> None:
    uf = make_frame(200, **frame_kw)
    with pytest.raises(SplitError, match=f"roles do not match spec.data.*{field}"):
        split_run(spec, uf, tmp_path / "run", vault, "roles")
    assert not (tmp_path / "run").exists()
    assert not vault.has_fold("roles", "calib")


def test_regime_override_resolves_auto(tmp_path: Path, vault: LocalVault) -> None:
    res = split_run(make_spec("auto"), make_frame(200), tmp_path / "r", vault, "r", regime="iid")
    assert res.manifest.regime.value == "iid"


def test_load_dev_checks_the_hash_lock(tmp_path: Path, vault: LocalVault) -> None:
    rd = tmp_path / "r"
    split_run(make_spec(), make_frame(200), rd, vault, "r")
    path = rd / "splits.manifest.json"
    raw = json.loads(path.read_text())
    raw["dev_ids_sha256"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(SplitError, match="hash lock"):
        load_dev(rd)
    dev, _ = load_dev(rd, verify=False)
    assert dev.n == 120


def values_in_file(col: pa.ChunkedArray) -> set[object]:
    """Every value a parquet column carries: its rows plus any dictionary entries."""
    found = set(col.to_pylist())
    for chunk in col.chunks:
        if pa.types.is_dictionary(chunk.type):
            found.update(chunk.dictionary.to_pylist())
    return found


def test_dev_parquet_carries_no_held_out_dictionary_values(
    tmp_path: Path, vault: LocalVault
) -> None:
    """Arrow dictionaries survive ``take``; dev.parquet must not carry calib/sealed values."""
    n = 300
    rng = np.random.default_rng(8)
    df = pd.DataFrame(
        {
            "id": [f"unit-{i:06d}-z" for i in range(n)],
            "a": rng.normal(size=n),
            "c": pd.Categorical([f"level-{i:05d}-q" for i in range(n)]),  # one value per unit
            "y": pd.Categorical(rng.choice(["class-p", "class-q"], size=n)),
        }
    )
    uf = UnitFrame.from_pandas(df, Roles(unit_id="id", target="y", inputs=("a", "c")))
    # UnitFrame decodes dictionaries at construction (defence in depth with split_run).
    assert not pa.types.is_dictionary(uf.table.schema.field("c").type)
    res = split_run(make_spec(seed=5), uf, tmp_path / "run", vault, "rdict")
    dev_mask = res.assignment.mask(Fold.DEV)
    held_levels = set(df["c"].astype(str)[~dev_mask])
    assert held_levels

    raw = pq.read_table(res.dev_path)
    assert raw.num_rows == int(dev_mask.sum())
    for name in raw.column_names:
        col = raw.column(name)
        assert values_in_file(col) == set(col.to_pylist()), name
    assert not values_in_file(raw.column("c")) & held_levels
    pandas_meta = (raw.schema.metadata or {}).get(b"pandas", b"")
    assert b"num_categories" not in pandas_meta  # the category count spans every fold
    dev, _ = load_dev(tmp_path / "run")
    assert set(dev.column("c").tolist()) == set(df["c"].astype(str)[dev_mask])
