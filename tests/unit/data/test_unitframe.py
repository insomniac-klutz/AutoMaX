from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from amx.data import Roles, UnitFrame, UnitFrameError
from amx.spec import parse_taskspec

ROLES = Roles(unit_id="id", target="y", inputs=("a", "b"), time="t", group_columns=("g",))


def frame(n: int = 50, seed: int = 0) -> UnitFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": [f"u{i}" for i in range(n)],
            "a": rng.normal(size=n),
            "b": rng.choice(["x", "y", None], size=n),
            "t": pd.date_range("2020-01-01", periods=n, freq="h"),
            "g": rng.integers(0, 5, n),
            "y": rng.integers(0, 2, n),
            "unused": 1,
        }
    )
    return UnitFrame.from_pandas(df, ROLES)


def test_roles_and_accessors() -> None:
    uf = frame()
    assert uf.n == 50
    assert uf.table.column_names == ["id", "a", "b", "g", "y", "t"]
    assert uf.inputs().column_names == ["a", "b"]
    assert uf.groups is not None and set(uf.groups) <= {"0", "1", "2", "3", "4"}
    t = uf.time
    assert t is not None and t.dtype == np.int64 and np.all(np.diff(t) > 0)


def test_duplicate_or_null_ids_rejected() -> None:
    df = pd.DataFrame({"id": ["a", "a"], "a": [1, 2], "b": ["x", "y"], "y": [0, 1]})
    with pytest.raises(UnitFrameError, match="duplicate"):
        UnitFrame.from_pandas(df, Roles("id", "y", ("a", "b")))
    df2 = pd.DataFrame({"id": ["a", None], "a": [1, 2], "b": ["x", "y"], "y": [0, 1]})
    with pytest.raises(UnitFrameError, match="nulls"):
        UnitFrame.from_pandas(df2, Roles("id", "y", ("a", "b")))


def test_missing_column_rejected() -> None:
    df = pd.DataFrame({"id": ["a"], "y": [0]})
    with pytest.raises(UnitFrameError, match="missing"):
        UnitFrame.from_pandas(df, Roles("id", "y", ("a",)))


def test_hash_invariant_to_row_order_and_sensitive_to_values() -> None:
    uf = frame()
    perm = np.random.default_rng(1).permutation(uf.n)
    assert uf.take(perm).content_hash() == uf.content_hash()
    df = uf.table.to_pandas()
    df.loc[3, "a"] = df.loc[3, "a"] + 1e-9
    assert UnitFrame.from_pandas(df, ROLES).content_hash() != uf.content_hash()


def test_without_target_drops_buffers() -> None:
    uf = frame().without_target()
    assert "y" not in uf.table.column_names
    assert not uf.has_target
    with pytest.raises(UnitFrameError):
        _ = uf.target


def test_select_ids_and_take() -> None:
    uf = frame()
    sub = uf.select_ids(["u3", "u1"])
    assert sub.ids.tolist() == ["u3", "u1"]
    with pytest.raises(UnitFrameError, match="unknown"):
        uf.select_ids(["nope"])


def test_from_spec_infers_inputs() -> None:
    spec = parse_taskspec(
        {
            "spec_version": 1,
            "name": "t",
            "data": {
                "uri": "x",
                "format": "csv",
                "unit_id": "id",
                "time_column": "t",
                "target": {"name": "y", "kind": "categorical"},
            },
            "task": {"family": "classification"},
            "bands": {"alphas": [0.05], "policies": ["auto"]},
        }
    )
    tbl = pa.table({"id": ["a", "b"], "t": [1, 2], "y": [0, 1], "f1": [0.1, 0.2], "f2": ["p", "q"]})
    uf = UnitFrame.from_spec(tbl, spec.data)
    assert uf.roles.inputs == ("f1", "f2")


def test_input_row_keys_detect_duplicates() -> None:
    df = pd.DataFrame(
        {"id": ["a", "b", "c"], "a": [1.0, 1.0, 2.0], "b": ["x", "x", "x"], "y": [0, 1, 0]}
    )
    keys = UnitFrame.from_pandas(df, Roles("id", "y", ("a", "b"))).input_row_keys()
    assert keys[0] == keys[1] != keys[2]
