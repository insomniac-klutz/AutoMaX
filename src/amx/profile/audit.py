"""Label-free profile pass on all data, before the split (HANDOFF 8.2, C9).

Nothing here reads the target. The pass detects modality, time structure, group-like id columns
and exact duplicates, and recommends a split regime.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from amx.data.unitframe import UnitFrame
from amx.spec.enums import Family, InputKind, Regime
from amx.spec.models import TaskSpec

TIME_NAME_HINTS = ("date", "time", "timestamp")
GROUP_MIN_UNIQUE = 50
GROUP_MIN_SHARE = 0.01
GROUP_MAX_SHARE = 0.5
TEXT_MEAN_CHARS = 40


class PreProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_units: int
    modality: str
    column_kinds: dict[str, str]
    time_candidates: list[str]
    group_candidates: list[str]
    exact_duplicate_units: int
    exact_duplicate_share: float
    declared_regime: Regime
    recommended_regime: Regime
    reasons: list[str]
    warnings: list[str]


def _kind(col: pa.ChunkedArray) -> str:
    t = col.type
    if pa.types.is_timestamp(t) or pa.types.is_date(t):
        return InputKind.TIMESTAMP.value
    if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t):
        return InputKind.NUMERIC.value
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_dictionary(t):
        vals = [v for v in col.cast(pa.string()).to_pylist()[:2000] if v is not None]
        mean_len = float(np.mean([len(v) for v in vals])) if vals else 0.0
        return InputKind.TEXT.value if mean_len >= TEXT_MEAN_CHARS else InputKind.CATEGORICAL.value
    return "other"


def column_kinds(uf: UnitFrame) -> dict[str, str]:
    declared: dict[str, str] = {}
    table = uf.inputs()
    for name in table.column_names:
        declared[name] = _kind(table.column(name))
    return declared


def _modality(kinds: dict[str, str], family: Family) -> str:
    if family is Family.FORECASTING:
        return "series"
    present = set(kinds.values())
    if InputKind.TEXT.value in present and present <= {InputKind.TEXT.value}:
        return "text"
    if InputKind.TEXT.value in present:
        return "mixed"
    return "tabular"


def _group_candidates(uf: UnitFrame) -> list[str]:
    n = uf.n
    out: list[str] = []
    table = uf.table
    skip = {uf.roles.unit_id, uf.roles.target, uf.roles.time}
    for name in table.column_names:
        if name in skip:
            continue
        col = table.column(name)
        if pa.types.is_floating(col.type):
            continue
        uniq = len(set(col.cast(pa.string()).to_pylist()))
        if uniq >= max(GROUP_MIN_UNIQUE, GROUP_MIN_SHARE * n) and uniq <= GROUP_MAX_SHARE * n:
            out.append(name)
    return out


def _time_candidates(uf: UnitFrame) -> list[str]:
    out = [] if uf.roles.time is None else [uf.roles.time]
    for name in uf.inputs().column_names:
        col = uf.table.column(name)
        temporal_type = pa.types.is_timestamp(col.type) or pa.types.is_date(col.type)
        if temporal_type or any(h in name.lower() for h in TIME_NAME_HINTS):
            out.append(name)
    return list(dict.fromkeys(out))


def pre_profile(uf: UnitFrame, spec: TaskSpec) -> PreProfile:
    """Label-free audit and regime recommendation (8.2)."""
    kinds = column_kinds(uf)
    keys = uf.input_row_keys()
    _, counts = np.unique(keys, return_counts=True)
    dup_units = int(np.sum(counts[counts > 1]))
    times = _time_candidates(uf)
    groups = _group_candidates(uf)
    reasons: list[str] = []
    warnings: list[str] = []
    family = spec.task.family
    if family is Family.FORECASTING or spec.data.time_column is not None:
        rec = Regime.TEMPORAL
        reasons.append("a time column is declared: future units come after past ones")
    elif spec.data.group_columns:
        rec = Regime.GROUPED
        reasons.append("group columns are declared: units in a group are not independent")
    elif groups:
        rec = Regime.GROUPED
        reasons.append(f"repeated high-cardinality id-like columns: {groups}")
    else:
        rec = Regime.IID
        reasons.append("no time or group structure detected")
    if times and rec is not Regime.TEMPORAL:
        warnings.append(f"time-like columns {times} found; consider regime temporal")
    if dup_units:
        warnings.append(
            f"{dup_units} units share their inputs with another unit; duplicate clusters are "
            "kept in one fold and the certified population excludes duplicates of dev (Q7)"
        )
    declared = spec.splits.regime
    if declared is not Regime.AUTO and declared is not rec:
        warnings.append(
            f"declared regime '{declared.value}' differs from the recommendation '{rec.value}'"
        )
    if uf.n < 3000:
        warnings.append(f"only {uf.n} units: small-data mode needs --allow-small (8.1)")
    return PreProfile(
        n_units=uf.n,
        modality=_modality(kinds, family),
        column_kinds=kinds,
        time_candidates=times,
        group_candidates=groups,
        exact_duplicate_units=dup_units,
        exact_duplicate_share=dup_units / max(uf.n, 1),
        declared_regime=declared,
        recommended_regime=rec,
        reasons=reasons,
        warnings=warnings,
    )


def resolve_regime(spec: TaskSpec, pre: PreProfile) -> Regime:
    """The regime the split will use: the declared one, or the recommendation for 'auto'."""
    return pre.recommended_regime if spec.splits.regime is Regime.AUTO else spec.splits.regime


def as_dict(pre: PreProfile) -> dict[str, Any]:
    return pre.model_dump(mode="json")
