"""Table readers and UnitFrame persistence (HANDOFF 6.1 ``data.format``, ROLLER step 4).

Readers return Arrow tables (C11: content hashes come from Arrow values, not pandas dtypes).
``format: custom`` points ``data.uri`` at a Python file defining ``load(cache_dir) -> table``;
this is the ``datasets/<name>/loader.py`` protocol. UnitFrames are persisted as parquet with
their Roles stored as JSON in the schema metadata, so a round trip restores the roles.

Readers decode dictionary-encoded columns (:func:`decode_dictionaries`): an Arrow dictionary
keeps every value of the full table through ``take``, so a fold cut from it would carry the
other folds' values (calib/sealed leakage into ``dev.parquet``, HANDOFF 8.1).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.json as pa_json
import pyarrow.parquet as pq
from numpy.typing import ArrayLike

from amx._log import get_logger
from amx.data.cache import datasets_dir
from amx.data.unitframe import Roles, UnitFrame
from amx.spec.enums import DataFormat
from amx.spec.loader import resolve_uri
from amx.spec.models import TaskSpec

log = get_logger(__name__)

ROLES_METADATA_KEY = b"amx.roles"
ROLES_FORMAT_VERSION = 1
LOADER_FUNCTION = "load"
PANDAS_METADATA_KEY = b"pandas"


class DataLoadError(ValueError):
    """A data source could not be read or does not follow its loader protocol."""


# dictionary decoding ---------------------------------------------------------------------


def _plain_type(t: pa.DataType) -> pa.DataType:
    """``t`` with every dictionary type (also nested in lists, structs, maps) replaced by its
    value type."""
    if pa.types.is_dictionary(t):
        return _plain_type(t.value_type)
    if pa.types.is_list(t):
        return pa.list_(t.value_field.with_type(_plain_type(t.value_type)))
    if pa.types.is_large_list(t):
        return pa.large_list(t.value_field.with_type(_plain_type(t.value_type)))
    if pa.types.is_fixed_size_list(t):
        return pa.list_(t.value_field.with_type(_plain_type(t.value_type)), t.list_size)
    if pa.types.is_map(t):
        return pa.map_(
            t.key_field.with_type(_plain_type(t.key_type)),
            t.item_field.with_type(_plain_type(t.item_type)),
            t.keys_sorted,
        )
    if pa.types.is_struct(t):
        return pa.struct(
            [t.field(i).with_type(_plain_type(t.field(i).type)) for i in range(t.num_fields)]
        )
    return t


def decode_dictionaries(table: pa.Table) -> pa.Table:
    """``table`` with every dictionary-encoded column cast to its value type.

    A dictionary column keeps its whole dictionary through ``take``/``filter`` and parquet
    writes it as is, so the rows of one fold would carry the values of every other fold. The
    pandas schema metadata is dropped when a column changes: it is stale afterwards and records
    the category count of the full table. Tables without dictionary columns are returned as is.
    """
    plain = [_plain_type(f.type) for f in table.schema]
    if all(p.equals(f.type) for p, f in zip(plain, table.schema, strict=True)):
        return table
    columns = [
        col if p.equals(col.type) else col.cast(p)
        for col, p in zip(table.columns, plain, strict=True)
    ]
    meta = {k: v for k, v in (table.schema.metadata or {}).items() if k != PANDAS_METADATA_KEY}
    fields = [f.with_type(p) for f, p in zip(table.schema, plain, strict=True)]
    decoded = pa.Table.from_arrays(columns, schema=pa.schema(fields, metadata=meta or None))
    log.debug(
        "decoded dictionary columns %s",
        [f.name for f, p in zip(table.schema, plain, strict=True) if not p.equals(f.type)],
    )
    return decoded


# reading ---------------------------------------------------------------------------------


def read_table(
    path: str | Path, fmt: DataFormat | str, *, cache_dir: Path | None = None
) -> pa.Table:
    """Read ``path`` in format ``fmt`` into an Arrow table.

    ``parquet`` accepts a file or a dataset directory; ``csv`` and ``jsonl`` (newline-delimited
    JSON objects) accept a file. ``custom`` imports ``path`` as a Python file and calls its
    ``load(cache_dir)``, which returns a ``pyarrow.Table`` or a ``pandas.DataFrame``; ``cache_dir``
    defaults to :func:`amx.data.cache.datasets_dir`. Dictionary-encoded columns (pandas
    categoricals, parquet dictionaries) are decoded to their value type
    (:func:`decode_dictionaries`).
    """
    fmt = DataFormat(fmt)
    p = Path(path)
    if fmt is DataFormat.IMAGE_FOLDER:
        raise NotImplementedError(
            "data.format 'image_folder' arrives in milestone A1 (operators and encoders)"
        )
    if fmt is DataFormat.CUSTOM:
        return decode_dictionaries(_read_custom(p, cache_dir))
    if not p.exists():
        raise FileNotFoundError(f"data file not found: {p}")
    try:
        if fmt is DataFormat.PARQUET:
            table = pq.read_table(p)
        elif fmt is DataFormat.CSV:
            table = pa_csv.read_csv(p)
        else:
            table = pa_json.read_json(p)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, OSError) as exc:
        raise DataLoadError(f"cannot read {p} as {fmt.value}: {exc}") from exc
    log.debug("read %s rows from %s (%s)", table.num_rows, p, fmt.value)
    return decode_dictionaries(table)


def _read_custom(path: Path, cache_dir: Path | None) -> pa.Table:
    if path.suffix != ".py" or not path.is_file():
        raise DataLoadError(f"format 'custom' needs data.uri to be a Python file, got {path}")
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    module_name = f"amx_data_loader_{digest}"
    mod_spec = importlib.util.spec_from_file_location(module_name, path)
    if mod_spec is None or mod_spec.loader is None:
        raise DataLoadError(f"cannot import loader {path}")
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules[module_name] = module
    try:
        mod_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    load = getattr(module, LOADER_FUNCTION, None)
    if not callable(load):
        raise DataLoadError(f"{path} must define {LOADER_FUNCTION}(cache_dir: Path) -> table")
    if cache_dir is None:
        cache_dir = datasets_dir(create=True)
    else:
        cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    result = load(cache_dir)
    if isinstance(result, pa.Table):
        table = result
    elif isinstance(result, pd.DataFrame):
        table = pa.Table.from_pandas(result, preserve_index=False)
    else:
        raise DataLoadError(
            f"{path}:{LOADER_FUNCTION} returned {type(result).__name__}; "
            "expected pyarrow.Table or pandas.DataFrame"
        )
    log.debug("custom loader %s returned %s rows", path, table.num_rows)
    return table


def load_unitframe(
    spec: TaskSpec, spec_path: str | Path, *, cache_dir: Path | None = None
) -> UnitFrame:
    """Read ``spec.data`` (``uri`` resolved relative to the spec file) into a UnitFrame.

    The table comes from :func:`read_table`, so it holds no dictionary-encoded columns.
    """
    path = resolve_uri(spec, spec_path)
    table = read_table(path, spec.data.format, cache_dir=cache_dir)
    return UnitFrame.from_spec(table, spec.data)


# persistence -----------------------------------------------------------------------------


def roles_to_json(roles: Roles) -> str:
    payload: dict[str, Any] = {
        "format": ROLES_FORMAT_VERSION,
        "unit_id": roles.unit_id,
        "target": roles.target,
        "inputs": list(roles.inputs),
        "time": roles.time,
        "group_columns": list(roles.group_columns),
        "series_columns": list(roles.series_columns),
    }
    return json.dumps(payload, sort_keys=True)


def roles_from_json(text: str | bytes) -> Roles:
    try:
        raw = json.loads(text)
        version = raw.get("format")
        roles = Roles(
            unit_id=str(raw["unit_id"]),
            target=None if raw["target"] is None else str(raw["target"]),
            inputs=tuple(str(c) for c in raw["inputs"]),
            time=None if raw["time"] is None else str(raw["time"]),
            group_columns=tuple(str(c) for c in raw["group_columns"]),
            series_columns=tuple(str(c) for c in raw["series_columns"]),
        )
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise DataLoadError(f"invalid roles metadata: {exc}") from exc
    if version != ROLES_FORMAT_VERSION:
        raise DataLoadError(f"unsupported roles metadata format {version!r}")
    return roles


def roles_from_schema(schema: pa.Schema) -> Roles:
    """Roles stored by :func:`write_unitframe` in a parquet/Arrow schema."""
    meta = schema.metadata or {}
    if ROLES_METADATA_KEY not in meta:
        raise DataLoadError("no amx roles in the schema metadata (not written by amx)")
    return roles_from_json(meta[ROLES_METADATA_KEY])


def write_unitframe(
    uf: UnitFrame,
    path: str | Path,
    *,
    extra_columns: Mapping[str, ArrayLike] | None = None,
    file_mode: int | None = None,
) -> Path:
    """Write ``uf`` as parquet with its Roles in the schema metadata (key ``amx.roles``).

    ``extra_columns`` are appended (one value per row); they must not clash with role columns
    and are dropped again by :func:`read_unitframe`. The file is written to a temporary name
    created with ``file_mode`` (default: 0o666 minus the umask) and renamed into place.
    """
    p = Path(path)
    table = uf.table
    for name, values in (extra_columns or {}).items():
        if name in table.column_names:
            raise DataLoadError(f"extra column '{name}' clashes with an existing column")
        arr = np.asarray(values)
        if arr.shape != (uf.n,):
            raise DataLoadError(f"extra column '{name}' needs {uf.n} values, got {arr.shape}")
        table = table.append_column(name, pa.array(arr))
    meta = dict(table.schema.metadata or {})
    meta[ROLES_METADATA_KEY] = roles_to_json(uf.roles).encode("utf-8")
    table = table.replace_schema_metadata(meta)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    mode = 0o666 if file_mode is None else file_mode
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            if file_mode is not None:
                os.fchmod(fh.fileno(), file_mode)
            pq.write_table(table, fh)
        os.replace(tmp, p)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return p


def read_unitframe(path: str | Path) -> UnitFrame:
    """Read a UnitFrame written by :func:`write_unitframe`; extra columns are dropped."""
    table = pq.read_table(Path(path))
    return UnitFrame(table, roles_from_schema(table.schema))
