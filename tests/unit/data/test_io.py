from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from amx.data import (
    ROLES_METADATA_KEY,
    DataLoadError,
    Roles,
    UnitFrame,
    cache_dir,
    datasets_dir,
    load_unitframe,
    read_table,
    read_unitframe,
    roles_from_schema,
    write_unitframe,
)
from amx.spec import DataFormat, TaskSpec, parse_taskspec

ROLES = Roles(
    unit_id="id",
    target="y",
    inputs=("a", "b"),
    time="t",
    group_columns=("g",),
    series_columns=("s",),
)


def frame(n: int = 30) -> UnitFrame:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "id": [f"u{i:03d}" for i in range(n)],
            "a": rng.normal(size=n),
            "b": rng.choice(["p", "q", None], size=n),
            "t": pd.date_range("2021-01-01", periods=n, freq="D"),
            "g": rng.integers(0, 4, n),
            "s": rng.choice(["s1", "s2"], size=n),
            "y": rng.integers(0, 2, n),
        }
    )
    return UnitFrame.from_pandas(df, ROLES)


def make_spec(uri: str, fmt: str, **data: Any) -> TaskSpec:
    d: dict[str, Any] = {
        "uri": uri,
        "format": fmt,
        "unit_id": "id",
        "target": {"name": "y", "kind": "categorical"},
    }
    d.update(data)
    return parse_taskspec(
        {
            "spec_version": 1,
            "name": "io-test",
            "data": d,
            "task": {"family": "classification"},
            "bands": {"alphas": [0.05], "policies": ["auto"]},
        }
    )


# cache ------------------------------------------------------------------------------------


def test_cache_dir_env_and_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMX_CACHE", str(tmp_path / "c"))
    assert cache_dir() == tmp_path / "c"
    assert not (tmp_path / "c").exists()
    d = datasets_dir(create=True)
    assert d == tmp_path / "c" / "datasets" and d.is_dir()
    monkeypatch.delenv("AMX_CACHE")
    assert cache_dir() == Path.home() / ".cache" / "amx"


# read_table ---------------------------------------------------------------------------------

DF = pd.DataFrame({"id": ["a", "b", "c"], "x": [1.5, 2.5, 3.5], "y": [0, 1, 0]})


@pytest.mark.parametrize("fmt", ["parquet", "csv", "jsonl"])
def test_read_table_formats(tmp_path: Path, fmt: str) -> None:
    path = tmp_path / f"d.{fmt}"
    if fmt == "parquet":
        DF.to_parquet(path, index=False)
    elif fmt == "csv":
        DF.to_csv(path, index=False)
    else:
        DF.to_json(path, orient="records", lines=True)
    table = read_table(path, DataFormat(fmt))
    assert table.num_rows == 3
    assert table.column("id").to_pylist() == ["a", "b", "c"]
    assert table.column("x").to_pylist() == [1.5, 2.5, 3.5]


def test_read_table_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "missing.parquet", DataFormat.PARQUET)
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not parquet")
    with pytest.raises(DataLoadError):
        read_table(bad, DataFormat.PARQUET)
    with pytest.raises(NotImplementedError, match="A1"):
        read_table(tmp_path, DataFormat.IMAGE_FOLDER)


LOADER_ARROW = """
from pathlib import Path
import pyarrow as pa

def load(cache_dir: Path) -> pa.Table:
    (cache_dir / "touched").write_text("1")
    return pa.table({"id": ["a", "b"], "x": [1, 2], "y": [0, 1]})
"""

LOADER_PANDAS = """
import pandas as pd

def load(cache_dir):
    return pd.DataFrame({"id": ["a", "b"], "x": [1, 2], "y": [0, 1]})
"""


def test_custom_loader_relative_to_spec(tmp_path: Path) -> None:
    sub = tmp_path / "dsets" / "toy"
    sub.mkdir(parents=True)
    (sub / "loader.py").write_text(LOADER_ARROW)
    spec_path = tmp_path / "dsets" / "spec.yaml"
    spec = make_spec("toy/loader.py", "custom")
    cache = tmp_path / "cache"
    uf = load_unitframe(spec, spec_path, cache_dir=cache)
    assert uf.ids.tolist() == ["a", "b"]
    assert uf.roles.inputs == ("x",)
    assert (cache / "touched").read_text() == "1"


def test_custom_loader_pandas_and_default_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AMX_CACHE", str(tmp_path / "cache"))
    path = tmp_path / "loader_pd.py"
    path.write_text(LOADER_PANDAS)
    table = read_table(path, DataFormat.CUSTOM)
    assert isinstance(table, pa.Table) and table.num_rows == 2
    assert (tmp_path / "cache" / "datasets").is_dir()


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("x = 1\n", "must define load"),
        ("def load(cache_dir):\n    return [1, 2]\n", "expected pyarrow.Table"),
    ],
)
def test_custom_loader_protocol_errors(tmp_path: Path, source: str, match: str) -> None:
    path = tmp_path / "loader_bad.py"
    path.write_text(source)
    with pytest.raises(DataLoadError, match=match):
        read_table(path, DataFormat.CUSTOM, cache_dir=tmp_path / "c")
    with pytest.raises(DataLoadError, match="Python file"):
        read_table(tmp_path / "data.csv", DataFormat.CUSTOM)


def test_load_unitframe_parquet(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    DF.to_parquet(tmp_path / "data" / "d.parquet", index=False)
    spec = make_spec("data/d.parquet", "parquet")
    uf = load_unitframe(spec, tmp_path / "spec.yaml")
    assert uf.n == 3 and uf.roles.target == "y" and uf.roles.inputs == ("x",)


# persistence --------------------------------------------------------------------------------


def test_unitframe_round_trip_restores_roles(tmp_path: Path) -> None:
    uf = frame()
    path = write_unitframe(uf, tmp_path / "sub" / "uf.parquet")
    meta = pq.read_schema(path).metadata
    assert meta is not None and ROLES_METADATA_KEY in meta
    assert json.loads(meta[ROLES_METADATA_KEY])["unit_id"] == "id"
    back = read_unitframe(path)
    assert back.roles == uf.roles
    assert back.content_hash() == uf.content_hash()
    assert np.array_equal(back.input_row_keys(), uf.input_row_keys())
    assert list(tmp_path.joinpath("sub").iterdir()) == [path]  # no temp files left


def test_round_trip_without_target(tmp_path: Path) -> None:
    uf = frame().without_target()
    back = read_unitframe(write_unitframe(uf, tmp_path / "x.parquet"))
    assert back.roles.target is None and not back.has_target
    assert "y" not in back.table.column_names


def test_extra_columns_and_file_mode(tmp_path: Path) -> None:
    uf = frame(10)
    path = write_unitframe(
        uf, tmp_path / "e.parquet", extra_columns={"extra": np.arange(10)}, file_mode=0o600
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert pq.read_table(path).column("extra").to_pylist() == list(range(10))
    assert "extra" not in read_unitframe(path).table.column_names
    with pytest.raises(DataLoadError, match="clashes"):
        write_unitframe(uf, tmp_path / "f.parquet", extra_columns={"a": np.arange(10)})
    with pytest.raises(DataLoadError, match="needs 10 values"):
        write_unitframe(uf, tmp_path / "f.parquet", extra_columns={"z": np.arange(3)})


def test_read_unitframe_needs_roles(tmp_path: Path) -> None:
    path = tmp_path / "plain.parquet"
    DF.to_parquet(path, index=False)
    with pytest.raises(DataLoadError, match="no amx roles"):
        read_unitframe(path)
    schema = pa.schema([("id", pa.string())], metadata={ROLES_METADATA_KEY: b"{not json"})
    with pytest.raises(DataLoadError, match="invalid roles"):
        roles_from_schema(schema)
    future = json.dumps(
        {
            "format": 99,
            "unit_id": "id",
            "target": None,
            "inputs": [],
            "time": None,
            "group_columns": [],
            "series_columns": [],
        }
    ).encode()
    with pytest.raises(DataLoadError, match="unsupported"):
        roles_from_schema(pa.schema([("id", pa.string())], metadata={ROLES_METADATA_KEY: future}))


# dictionary columns -------------------------------------------------------------------------


def has_dictionary(t: pa.DataType) -> bool:
    """True if ``t`` or any type nested in it is dictionary-encoded."""
    if pa.types.is_dictionary(t):
        return True
    return any(has_dictionary(t.field(i).type) for i in range(t.num_fields))


CATEGORICAL_DF = pd.DataFrame(
    {
        "id": ["a", "b", "c"],
        "x": pd.Categorical(["lvl-1", "lvl-2", "lvl-1"], categories=["lvl-1", "lvl-2", "lvl-9"]),
        "y": pd.Categorical(["p", "q", "p"]),
    }
)

LOADER_CATEGORICAL = """
import pandas as pd
import pyarrow as pa

def load(cache_dir):
    word_type = pa.list_(pa.dictionary(pa.int32(), pa.string()))
    words = pa.array([["w1", "w2"], ["w3"], []]).cast(word_type)
    x = pd.Categorical(["p", "q", "p"])
    df = pd.DataFrame({"id": ["a", "b", "c"], "x": x, "y": [0, 1, 0]})
    return pa.Table.from_pandas(df, preserve_index=False).append_column("w", words)
"""


def test_read_table_decodes_dictionary_columns(tmp_path: Path) -> None:
    """Dictionary columns keep their whole dictionary through ``take``; readers decode them."""
    path = tmp_path / "cat.parquet"
    CATEGORICAL_DF.to_parquet(path, index=False)
    assert has_dictionary(pq.read_schema(path).field("x").type)  # the leak precondition
    table = read_table(path, DataFormat.PARQUET)
    assert not any(has_dictionary(f.type) for f in table.schema)
    assert table.column("x").to_pylist() == ["lvl-1", "lvl-2", "lvl-1"]
    # the stale pandas metadata (it records the category count) is dropped with the encoding
    assert b"num_categories" not in (table.schema.metadata or {}).get(b"pandas", b"")
    taken = table.take([0])  # a plain column carries the taken rows' values and nothing else
    assert not has_dictionary(taken.schema.field("x").type)
    assert taken.column("x").to_pylist() == ["lvl-1"]

    (tmp_path / "data").mkdir()
    CATEGORICAL_DF.to_parquet(tmp_path / "data" / "d.parquet", index=False)
    uf = load_unitframe(make_spec("data/d.parquet", "parquet"), tmp_path / "spec.yaml")
    assert not any(has_dictionary(f.type) for f in uf.table.schema)
    assert uf.take([0]).table.column("x").to_pylist() == ["lvl-1"]


def test_custom_loader_dictionary_columns_are_decoded(tmp_path: Path) -> None:
    path = tmp_path / "loader_cat.py"
    path.write_text(LOADER_CATEGORICAL)
    table = read_table(path, DataFormat.CUSTOM, cache_dir=tmp_path / "c")
    assert not any(has_dictionary(f.type) for f in table.schema)
    assert table.column("w").to_pylist() == [["w1", "w2"], ["w3"], []]
    assert table.column("x").to_pylist() == ["p", "q", "p"]
