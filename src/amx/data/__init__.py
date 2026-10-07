"""Data adapters and the UnitFrame."""

from amx.data.cache import cache_dir, datasets_dir
from amx.data.fetch import FetchAttempt, FetchError, fetch
from amx.data.io import (
    ROLES_METADATA_KEY,
    DataLoadError,
    decode_dictionaries,
    load_unitframe,
    read_table,
    read_unitframe,
    roles_from_schema,
    write_unitframe,
)
from amx.data.unitframe import Roles, UnitFrame, UnitFrameError

__all__ = [
    "ROLES_METADATA_KEY",
    "DataLoadError",
    "FetchAttempt",
    "FetchError",
    "Roles",
    "UnitFrame",
    "UnitFrameError",
    "cache_dir",
    "datasets_dir",
    "decode_dictionaries",
    "fetch",
    "load_unitframe",
    "read_table",
    "read_unitframe",
    "roles_from_schema",
    "write_unitframe",
]
