"""UnitFrame: an immutable table of units with named roles (ids, inputs, target, groups, time).

HANDOFF uses UnitFrame without defining it; this is the A0 definition (decision D19). The
canonical storage is an Arrow table. Content hashes are computed from Arrow values, never
from pandas dtypes (C11), and do not depend on row order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from numpy.typing import ArrayLike, NDArray

from amx.spec.models import DataSpec

GROUP_KEY_SEP = "\x1f"


class UnitFrameError(ValueError):
    """The table does not satisfy the roles it was given."""


@dataclass(frozen=True)
class Roles:
    unit_id: str
    target: str | None
    inputs: tuple[str, ...]
    time: str | None = None
    group_columns: tuple[str, ...] = ()
    series_columns: tuple[str, ...] = ()

    @property
    def role_columns(self) -> tuple[str, ...]:
        cols = [self.unit_id, *self.inputs, *self.group_columns, *self.series_columns]
        if self.target is not None:
            cols.append(self.target)
        if self.time is not None and self.time not in cols:
            cols.append(self.time)
        return tuple(dict.fromkeys(cols))

    @classmethod
    def from_spec(
        cls, data: DataSpec, columns: Sequence[str], forbidden: Sequence[str] = ()
    ) -> Roles:
        """Roles from a DataSpec.

        ``inputs: infer`` takes every column not used by another role and not listed in
        ``forbidden`` (the spec's ``constraints.forbidden_inputs``).
        """
        reserved = {data.unit_id, data.target.name, *data.group_columns, *data.series_columns}
        reserved.update(forbidden)
        if data.time_column is not None:
            reserved.add(data.time_column)
        names = data.input_names
        if names is None:
            names = [c for c in columns if c not in reserved]
        return cls(
            unit_id=data.unit_id,
            target=data.target.name,
            inputs=tuple(names),
            time=data.time_column,
            group_columns=tuple(data.group_columns),
            series_columns=tuple(data.series_columns),
        )


class UnitFrame:
    """Units of one dataset (or one fold of it)."""

    def __init__(self, table: pa.Table, roles: Roles) -> None:
        missing = [c for c in roles.role_columns if c not in table.column_names]
        if missing:
            raise UnitFrameError(f"missing columns: {missing}")
        ids = table.column(roles.unit_id)
        if ids.null_count:
            raise UnitFrameError(f"unit id column '{roles.unit_id}' has nulls")
        if pc.count_distinct(ids).as_py() != table.num_rows:
            raise UnitFrameError(f"unit id column '{roles.unit_id}' has duplicate values")
        self._table = _decode_dictionaries(table.select(list(roles.role_columns))).combine_chunks()
        self._roles = roles

    # construction -----------------------------------------------------------------------

    @classmethod
    def from_spec(cls, table: pa.Table, data: DataSpec, forbidden: Sequence[str] = ()) -> UnitFrame:
        return cls(table, Roles.from_spec(data, table.column_names, forbidden))

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, roles: Roles) -> UnitFrame:
        return cls(pa.Table.from_pandas(df, preserve_index=False), roles)

    # accessors --------------------------------------------------------------------------

    @property
    def roles(self) -> Roles:
        return self._roles

    @property
    def table(self) -> pa.Table:
        return self._table

    @property
    def n(self) -> int:
        return int(self._table.num_rows)

    def __len__(self) -> int:
        return self.n

    def column(self, name: str) -> NDArray[Any]:
        return np.asarray(self._table.column(name).to_numpy(zero_copy_only=False))

    @property
    def ids(self) -> NDArray[Any]:
        return np.asarray(
            self._table.column(self._roles.unit_id)
            .cast(pa.string())
            .to_numpy(zero_copy_only=False),
            dtype=object,
        )

    @property
    def has_target(self) -> bool:
        return self._roles.target is not None

    @property
    def target(self) -> NDArray[Any]:
        if self._roles.target is None:
            raise UnitFrameError("this UnitFrame carries no target")
        return self.column(self._roles.target)

    def inputs(self) -> pa.Table:
        return self._table.select(list(self._roles.inputs))

    def inputs_pandas(self) -> pd.DataFrame:
        df: pd.DataFrame = self.inputs().to_pandas()
        return df

    def _key(self, cols: tuple[str, ...]) -> NDArray[Any] | None:
        if not cols:
            return None
        parts = [self._table.column(c).cast(pa.string()).fill_null("") for c in cols]
        if len(parts) == 1:
            return np.asarray(parts[0].to_numpy(zero_copy_only=False), dtype=object)
        # JSON of the value tuple: unambiguous whatever characters the values contain
        cols_py = [p.to_pylist() for p in parts]
        return np.asarray(
            [json.dumps(list(row)) for row in zip(*cols_py, strict=True)], dtype=object
        )

    @property
    def groups(self) -> NDArray[Any] | None:
        """Combined group key per unit (group_columns), or None."""
        return self._key(self._roles.group_columns)

    @property
    def series(self) -> NDArray[Any] | None:
        """Combined series key per unit (series_columns), or None."""
        return self._key(self._roles.series_columns)

    @property
    def time(self) -> NDArray[Any] | None:
        """Time values as a sortable numpy array (timestamps become int64 nanoseconds)."""
        if self._roles.time is None:
            return None
        col = self._table.column(self._roles.time)
        if pa.types.is_timestamp(col.type) or pa.types.is_date(col.type):
            col = col.cast(pa.timestamp("ns")).cast(pa.int64())
        return np.asarray(col.to_numpy(zero_copy_only=False))

    # transforms -------------------------------------------------------------------------

    def take(self, indices: ArrayLike) -> UnitFrame:
        idx = pa.array(np.asarray(indices, dtype=np.int64))
        return UnitFrame(self._table.take(idx), self._roles)

    def select_ids(self, ids: Sequence[str] | NDArray[Any]) -> UnitFrame:
        pos = {v: i for i, v in enumerate(self.ids.tolist())}
        try:
            return self.take([pos[str(i)] for i in ids])
        except KeyError as exc:
            raise UnitFrameError(f"unknown unit id {exc.args[0]!r}") from exc

    def without_target(self) -> UnitFrame:
        """A copy whose Arrow buffers contain no target column."""
        roles = replace(self._roles, target=None)
        cols = [c for c in roles.role_columns]
        return UnitFrame(self._table.select(cols), roles)

    def sorted_by_id(self) -> UnitFrame:
        order = np.argsort(self.ids.astype(str), kind="stable")
        return self.take(order)

    # hashing ----------------------------------------------------------------------------

    def content_hash(self) -> str:
        """sha256 over (role names, column names, types, values) with rows sorted by unit id."""
        t = self.sorted_by_id().table
        h = hashlib.sha256()
        h.update(repr(self._roles).encode())
        for name in sorted(t.column_names):
            col = t.column(name)
            h.update(name.encode() + b"\x00" + str(col.type).encode() + b"\x00")
            h.update(_column_bytes(col))
        return "sha256:" + h.hexdigest()

    def input_row_keys(self) -> NDArray[np.uint64]:
        """64-bit hash of each unit's input values, for exact-duplicate detection."""
        df = self.inputs_pandas()
        if df.shape[1] == 0:
            return np.zeros(self.n, dtype=np.uint64)
        return np.asarray(pd.util.hash_pandas_object(df, index=False).to_numpy(), dtype=np.uint64)


def _decode_dictionaries(table: pa.Table) -> pa.Table:
    """Replace dictionary-encoded columns by their plain values.

    A dictionary column keeps its whole dictionary through ``take``, so a fold written from it
    would carry values of units outside the fold. Decoding at construction closes that leak.
    """
    cols = []
    for name in table.column_names:
        col = table.column(name)
        if pa.types.is_dictionary(col.type):
            col = col.cast(col.type.value_type)
        cols.append(col)
    return pa.table(cols, names=table.column_names)


def _column_bytes(col: pa.ChunkedArray) -> bytes:
    arr = col.combine_chunks()
    nulls = np.asarray(arr.is_null().to_numpy(zero_copy_only=False), dtype=np.bool_)
    t = arr.type
    if pa.types.is_timestamp(t) or pa.types.is_date(t) or pa.types.is_time(t):
        arr = arr.cast(pa.int64())
        t = arr.type
    if pa.types.is_integer(t) or pa.types.is_boolean(t):
        vals = np.asarray(arr.fill_null(0).to_numpy(zero_copy_only=False)).astype(np.int64)
        body = vals.tobytes()
    elif pa.types.is_floating(t):
        vals = np.asarray(arr.fill_null(0.0).to_numpy(zero_copy_only=False)).astype(np.float64)
        body = vals.tobytes()
    else:
        items = arr.cast(pa.string()).to_pylist() if _castable(arr) else arr.to_pylist()
        # length-prefixed so that different value lists can never serialise identically
        body = b"".join(
            len(e).to_bytes(8, "little") + e
            for e in (("" if v is None else str(v)).encode("utf-8") for v in items)
        )
    return nulls.tobytes() + b"\x00" + body


def _castable(arr: pa.Array) -> bool:
    try:
        arr.cast(pa.string())
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return False
    return True
