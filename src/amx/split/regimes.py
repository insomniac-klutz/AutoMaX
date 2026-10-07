"""Fold assignment per regime (HANDOFF 7.5, 8.1, 8.2; ROLLER step 5; OQ Q7).

Every unit ends in ``dev``, ``calib``, ``sealed`` or ``dropped`` (with a reason).

* ``iid``: exact-duplicate input clusters stay in one fold. Clusters are stratified by the
  target of their first unit (categorical value, or one of 10 quantile bins of a numeric
  target), shuffled with ``np.random.default_rng(seed)`` and cut by cumulative unit counts.
  Fold sizes per stratum come from :func:`allocate_strata`: floors of the quotas plus the
  leftover units to the folds with the largest global deficit, so with singleton clusters
  every stratum's fold sizes are within ±1 of its quotas and every fold total is within ±1
  of ``n * fraction``, however many small strata there are.
* ``grouped``: clusters are the union of group keys and duplicate clusters; no cluster
  straddles folds.
* ``temporal``: the time column must be integer, floating, timestamp or date (strings are
  refused: they sort lexicographically). Folds are contiguous in time with cuts at time-value
  boundaries, then ``embargo_steps`` distinct time steps after the dev block and after the
  calib block are dropped (reason ``embargo``); a forecasting spec needs an embargo of at least
  ``max(horizons) + max_lag`` (HANDOFF 7.5). Calib/sealed units whose inputs equal a dev
  unit's inputs are dropped (reason ``duplicate_of_dev``).

``auto`` must be resolved by the profiler first; ``blocked`` is deferred (ROLLER step 5).
All computation runs in unit-id order, so the assignment does not depend on row order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import ArrayLike, NDArray

from amx._log import get_logger
from amx.data.unitframe import UnitFrame
from amx.spec.enums import Regime, TargetKind
from amx.spec.models import TaskSpec
from amx.split.dedupe import canonical_clusters, canonical_order, duplicate_keys
from amx.split.errors import SplitError

log = get_logger(__name__)

N_QUANTILE_BINS = 10
_EPS = 1e-9


class Fold(StrEnum):
    DEV = "dev"
    CALIB = "calib"
    SEALED = "sealed"
    DROPPED = "dropped"


class DropReason(StrEnum):
    EMBARGO = "embargo"
    DUPLICATE_OF_DEV = "duplicate_of_dev"


KEPT_FOLDS: tuple[Fold, Fold, Fold] = (Fold.DEV, Fold.CALIB, Fold.SEALED)
_FOLD_CODES: tuple[Fold, ...] = (*KEPT_FOLDS, Fold.DROPPED)
_DROPPED = 3


@dataclass(frozen=True)
class FoldAssignment:
    """Fold of every unit, aligned with the rows of the UnitFrame that was split."""

    regime: Regime
    folds: NDArray[np.str_]
    drop_reasons: NDArray[np.str_]
    clusters: NDArray[np.int64]
    embargo_steps: int = 0

    def __post_init__(self) -> None:
        n = self.folds.shape[0]
        if self.drop_reasons.shape[0] != n or self.clusters.shape[0] != n:
            raise ValueError("folds, drop_reasons and clusters must have the same length")

    @property
    def n(self) -> int:
        return int(self.folds.shape[0])

    def mask(self, fold: Fold | str) -> NDArray[np.bool_]:
        return np.asarray(self.folds == Fold(fold).value, dtype=np.bool_)

    def indices(self, fold: Fold | str) -> NDArray[np.int64]:
        return np.flatnonzero(self.mask(fold)).astype(np.int64)

    @property
    def counts(self) -> dict[str, int]:
        return {f.value: int(np.count_nonzero(self.folds == f.value)) for f in _FOLD_CODES}

    @property
    def dropped_reasons(self) -> dict[str, int]:
        reasons = self.drop_reasons[self.folds == Fold.DROPPED.value]
        values, counts = np.unique(reasons, return_counts=True)
        return {str(v): int(c) for v, c in zip(values, counts, strict=True)}


# public entry point ----------------------------------------------------------------------


def assign_folds(
    uf: UnitFrame, spec: TaskSpec, regime: Regime | str | None = None
) -> FoldAssignment:
    """Assign every unit of ``uf`` to a fold under ``regime`` (default ``spec.splits.regime``)."""
    regime = spec.splits.regime if regime is None else Regime(regime)
    if regime is Regime.AUTO:
        raise SplitError(
            "regime 'auto' must be resolved by the profiler (amx profile) before splitting"
        )
    if regime is Regime.BLOCKED:
        raise NotImplementedError("regime 'blocked' is deferred (ROLLER step 5)")
    if uf.n == 0:
        raise SplitError("cannot split an empty UnitFrame")
    fr = spec.splits.fractions
    fractions = (fr.dev, fr.calib, fr.sealed)
    order = canonical_order(uf)
    seed = spec.splits.seed
    if regime is Regime.IID:
        codes_c, reasons_c, clusters_c = _assign_iid(uf, spec, order, fractions, seed)
    elif regime is Regime.GROUPED:
        codes_c, reasons_c, clusters_c = _assign_grouped(uf, order, fractions, seed)
    else:
        codes_c, reasons_c, clusters_c = _assign_temporal(uf, spec, order, fractions)

    names = np.array([f.value for f in _FOLD_CODES])
    folds = np.empty(uf.n, dtype=names.dtype)
    folds[order] = names[codes_c]
    reasons = np.empty(uf.n, dtype=np.dtype("<U32"))
    reasons[order] = reasons_c
    clusters = np.empty(uf.n, dtype=np.int64)
    clusters[order] = clusters_c
    out = FoldAssignment(
        regime=regime,
        folds=folds,
        drop_reasons=reasons,
        clusters=clusters,
        embargo_steps=spec.embargo_steps if regime is Regime.TEMPORAL else 0,
    )
    counts = out.counts
    empty = [f.value for f in KEPT_FOLDS if counts[f.value] == 0]
    if empty:
        raise SplitError(
            f"regime '{regime.value}' leaves fold(s) {empty} empty (counts {counts}); "
            "more units, fewer or smaller clusters, or other fractions are needed"
        )
    log.info(
        "split: regime=%s counts=%s dropped=%s",
        regime.value,
        counts,
        out.dropped_reasons,
        extra={"amx": {"regime": regime.value, "counts": counts}},
    )
    return out


# allocation helpers ----------------------------------------------------------------------


def largest_remainder(
    n: int, fractions: Sequence[float], rng: np.random.Generator | None = None
) -> NDArray[np.int64]:
    """Integer sizes summing to ``n`` with ``|size_f - n * frac_f| < 1`` (Hamilton rounding).

    Ties between equal remainders are broken by ``rng`` (so no fold is systematically
    favoured across strata), or by fold order when ``rng`` is None.
    """
    fr = np.asarray(fractions, dtype=np.float64)
    if fr.ndim != 1 or not np.isfinite(fr).all() or np.any(fr < 0) or fr.sum() <= 0:
        raise ValueError(f"fractions must be a non-empty vector of non-negative numbers: {fr}")
    fr = np.asarray(fr / fr.sum(), dtype=np.float64)  # sum is 1 within 1e-9; renormalise
    quotas = n * fr
    base = np.floor(quotas + _EPS).astype(np.int64)
    remainder = np.round(quotas - base, 9)
    left = int(n - int(base.sum()))
    if left > 0:
        tie = rng.random(fr.size) if rng is not None else np.arange(fr.size, dtype=np.float64)
        order = np.lexsort((tie, -remainder))
        base[order[:left]] += 1
    return base


def allocate_strata(
    sizes: ArrayLike, fractions: Sequence[float], rng: np.random.Generator | None = None
) -> NDArray[np.int64]:
    """Fold sizes per stratum (shape ``(len(sizes), len(fractions))``).

    Each stratum first gets the floors of its quotas ``m_s * frac_f``. Its leftover units go,
    at most one per fold, to the folds with the largest *global* deficit: the cumulative target
    over the strata so far minus the units allocated so far. Ties go to the larger remainder,
    then to ``rng`` (fold order when ``rng`` is None). Every stratum stays within ±1 of its
    quotas and, with three folds, every running fold total stays within ±1 of its target.
    Largest-remainder rounding per stratum alone lets the totals drift by up to one unit per
    stratum: with singleton strata every unit would land in the largest fold.
    """
    m = np.asarray(sizes, dtype=np.int64).reshape(-1)
    fr = np.asarray(fractions, dtype=np.float64)
    if fr.ndim != 1 or not np.isfinite(fr).all() or np.any(fr < 0) or fr.sum() <= 0:
        raise ValueError(f"fractions must be a non-empty vector of non-negative numbers: {fr}")
    if np.any(m < 0):
        raise ValueError("stratum sizes must be non-negative")
    fr = np.asarray(fr / fr.sum(), dtype=np.float64)
    out = np.zeros((m.size, fr.size), dtype=np.int64)
    allocated = np.zeros(fr.size, dtype=np.int64)
    seen = 0
    for s, size in enumerate(m.tolist()):
        quotas = size * fr
        base = np.floor(quotas + _EPS).astype(np.int64)
        left = int(size - int(base.sum()))
        seen += size
        if left > 0:
            remainder = np.round(quotas - base, 9)
            deficit = np.round(seen * fr - (allocated + base), 9)
            tie = rng.random(fr.size) if rng is not None else np.arange(fr.size, dtype=np.float64)
            order = np.lexsort((tie, -remainder, -deficit))
            base[order[:left]] += 1
        out[s] = base
        allocated += base
    return out


def cut_by_cumulative(sizes: NDArray[np.int64], targets: NDArray[np.int64]) -> NDArray[np.int64]:
    """Fold index per item, walking items in order and cutting at cumulative unit targets.

    An item goes to the fold whose cumulative target interval contains its midpoint; with
    unit-size items fold f receives exactly ``targets[f]`` items.
    """
    sizes = np.asarray(sizes, dtype=np.int64)
    mid = np.cumsum(sizes) - sizes / 2.0
    bounds = np.cumsum(np.asarray(targets, dtype=np.int64))[:-1].astype(np.float64)
    return np.asarray(np.searchsorted(bounds, mid, side="right"), dtype=np.int64)


def stratum_codes(values: NDArray[Any], kind: TargetKind | None) -> NDArray[np.int64]:
    """Stratum per value: the value itself (categorical), 10 quantile bins (numeric), else one.

    Missing values form their own stratum. Targets that are not scalars (spans, sets) are
    not stratified.
    """
    n = len(values)
    if kind is None or n == 0:
        return np.zeros(n, dtype=np.int64)
    if kind is TargetKind.CATEGORICAL:
        s = pd.Series(np.asarray(values, dtype=object), dtype=object)
        null = s.isna().to_numpy(dtype=np.bool_)
        keys = s.astype(str).to_numpy(dtype=object)
        keys[null] = "\x00<null>"
        _, inverse = np.unique(keys, return_inverse=True)
        return np.asarray(inverse, dtype=np.int64).reshape(-1)
    if kind in (TargetKind.NUMERIC, TargetKind.SERIES):
        try:
            x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
        except (TypeError, ValueError):
            return np.zeros(n, dtype=np.int64)
        finite = np.isfinite(x)
        bins = np.full(n, N_QUANTILE_BINS, dtype=np.int64)
        if finite.any():
            qs = np.linspace(0.0, 1.0, N_QUANTILE_BINS + 1)[1:-1]
            edges = np.quantile(x[finite], qs)
            bins[finite] = np.searchsorted(edges, x[finite], side="right")
        _, inverse = np.unique(bins, return_inverse=True)
        return np.asarray(inverse, dtype=np.int64).reshape(-1)
    return np.zeros(n, dtype=np.int64)


def first_index(clusters: NDArray[np.int64], n_clusters: int) -> NDArray[np.int64]:
    """Position of the first unit of each dense cluster id 0..n_clusters-1."""
    values, first = np.unique(clusters, return_index=True)
    if values.size != n_clusters or (n_clusters and int(values[-1]) != n_clusters - 1):
        raise ValueError("cluster ids must be dense 0..n_clusters-1")
    return np.asarray(first, dtype=np.int64)


def _is_time_type(t: pa.DataType) -> bool:
    return bool(
        pa.types.is_integer(t)
        or pa.types.is_floating(t)
        or pa.types.is_timestamp(t)
        or pa.types.is_date(t)
    )


def time_values(uf: UnitFrame) -> NDArray[Any]:
    """Sortable time value per unit (HANDOFF 7.5: dev < calib < sealed in time).

    The time column must have an integer, floating, timestamp or date Arrow type, whose sort
    order is temporal order. Strings, binary, booleans and other types are refused: strings
    sort lexicographically, so dates such as ``dd.mm.yyyy`` would be cut in the wrong order
    without any error. Also refuses a missing time column, nulls and NaN.
    """
    time_col = uf.roles.time
    t = uf.time
    if time_col is None or t is None:
        raise SplitError("regime 'temporal' needs a time column (data.time_column)")
    col_type = uf.table.column(time_col).type
    if not _is_time_type(col_type):
        raise SplitError(
            f"time column '{time_col}' has Arrow type {col_type}; temporal order needs integer, "
            "floating, timestamp or date values (strings sort lexicographically, so dates "
            "like dd.mm.yyyy would be ordered wrongly). Please declare it as a timestamp: "
            "parse it into a timestamp or date column (for example in a format: custom loader) "
            "or use numeric time steps"
        )
    if uf.table.column(time_col).null_count:
        raise SplitError(f"time column '{time_col}' has nulls; temporal order is undefined")
    arr = np.asarray(t)
    if np.issubdtype(arr.dtype, np.floating) and bool(np.isnan(arr).any()):
        raise SplitError(f"time column '{time_col}' has NaN values; temporal order is undefined")
    return arr


def min_embargo_steps(spec: TaskSpec) -> int:
    """Smallest embargo the spec allows: ``max(horizons) + max_lag`` for forecasting, else 0.

    HANDOFF 7.5: temporal folds are separated by at least the maximum horizon or lag, so no
    target window or lagged input reaches across a fold boundary.
    """
    forecast = spec.task.forecast
    if forecast is None:
        return 0
    return max(forecast.horizons) + forecast.max_lag


def check_embargo(spec: TaskSpec) -> int:
    """``spec.embargo_steps`` after checking it against :func:`min_embargo_steps`."""
    embargo = spec.embargo_steps
    minimum = min_embargo_steps(spec)
    if embargo < minimum:
        raise SplitError(
            f"forecasting needs an embargo of {minimum} steps or more (max(horizons) + max_lag, "
            f"HANDOFF 7.5), but splits.embargo is {embargo}; raise it or leave it unset to use "
            "the minimum"
        )
    return embargo


def make_rng(seed: int, *stream: int) -> np.random.Generator:
    """Seeded generator; negative seeds are mapped into the unsigned 64-bit range."""
    base = seed % (1 << 64)
    if not stream:
        return np.random.default_rng(base)
    return np.random.default_rng([base, *stream])


# regimes ---------------------------------------------------------------------------------

_Arrays = tuple[NDArray[np.int64], NDArray[np.str_], NDArray[np.int64]]


def _assign_iid(
    uf: UnitFrame,
    spec: TaskSpec,
    order: NDArray[np.int64],
    fractions: Sequence[float],
    seed: int,
) -> _Arrays:
    if uf.roles.group_columns or spec.data.group_columns:
        raise SplitError("regime 'iid' cannot be used with group_columns; use 'grouped'")
    clusters = canonical_clusters(uf, Regime.IID, order)
    n_clusters = int(np.max(clusters)) + 1
    sizes = np.bincount(clusters, minlength=n_clusters).astype(np.int64)
    first = first_index(clusters, n_clusters)
    if uf.has_target:
        target = uf.target[order]
        strata = stratum_codes(target[first], spec.data.target.kind)
    else:
        log.warning("iid split without a target: no stratification")
        strata = np.zeros(n_clusters, dtype=np.int64)
    rng = make_rng(seed)
    n_strata = int(np.max(strata)) + 1
    units = np.bincount(strata, weights=sizes, minlength=n_strata).astype(np.int64)
    targets = allocate_strata(units, fractions, rng)
    fold_of_cluster = np.empty(n_clusters, dtype=np.int64)
    for s in range(n_strata):
        members = np.flatnonzero(strata == s)
        if members.size == 0:
            continue
        perm = members[rng.permutation(members.size)]
        fold_of_cluster[perm] = cut_by_cumulative(sizes[perm], targets[s])
    codes = fold_of_cluster[clusters]
    reasons = np.full(uf.n, "", dtype=np.dtype("<U32"))
    return codes, reasons, clusters


def _assign_grouped(
    uf: UnitFrame, order: NDArray[np.int64], fractions: Sequence[float], seed: int
) -> _Arrays:
    clusters = canonical_clusters(uf, Regime.GROUPED, order)
    n_clusters = int(np.max(clusters)) + 1
    sizes = np.bincount(clusters, minlength=n_clusters).astype(np.int64)
    largest = int(np.max(sizes))
    if largest > fractions[1] * uf.n:
        log.warning(
            "grouped split: the largest cluster has %d units, more than the calib target", largest
        )
    rng = make_rng(seed)
    perm = rng.permutation(n_clusters)
    targets = largest_remainder(uf.n, fractions, rng)
    fold_of_cluster = np.empty(n_clusters, dtype=np.int64)
    fold_of_cluster[perm] = cut_by_cumulative(sizes[perm], targets)
    codes = fold_of_cluster[clusters]
    reasons = np.full(uf.n, "", dtype=np.dtype("<U32"))
    return codes, reasons, clusters


def _assign_temporal(
    uf: UnitFrame, spec: TaskSpec, order: NDArray[np.int64], fractions: Sequence[float]
) -> _Arrays:
    if uf.roles.group_columns or spec.data.group_columns:
        raise SplitError(
            "regime 'temporal' with group_columns is not supported in A0: a group would "
            "straddle folds in time. Use series_columns for entities that may span folds, "
            "or regime 'grouped'."
        )
    embargo = check_embargo(spec)
    t_c = time_values(uf)[order]
    steps, step_of_unit, step_counts = np.unique(t_c, return_inverse=True, return_counts=True)
    step_of_unit = np.asarray(step_of_unit).reshape(-1)
    n_steps = steps.size
    targets = largest_remainder(uf.n, fractions, None)
    block = cut_by_cumulative(step_counts.astype(np.int64), targets)
    block_steps = [int(np.count_nonzero(block == b)) for b in range(3)]
    if block_steps[0] == 0 or block_steps[1] <= embargo or block_steps[2] <= embargo:
        raise SplitError(
            f"temporal split: {n_steps} distinct time steps cut into blocks of "
            f"{block_steps} steps cannot hold an embargo of {embargo} steps after dev and "
            "after calib"
        )
    step_reason = np.full(n_steps, "", dtype=np.dtype("<U32"))
    step_code = block.copy()
    for b in (0, 1):
        last = int(np.max(np.flatnonzero(block == b)))
        drop = np.arange(last + 1, last + 1 + embargo)
        step_code[drop] = _DROPPED
        step_reason[drop] = DropReason.EMBARGO.value
    codes = step_code[step_of_unit]
    reasons = step_reason[step_of_unit]
    clusters = canonical_clusters(uf, Regime.TEMPORAL, order)
    keys = duplicate_keys(uf)
    if keys is not None:
        keys_c = keys[order]
        dev_keys = keys_c[codes == 0]
        dup = ((codes == 1) | (codes == 2)) & np.isin(keys_c, dev_keys)
        codes[dup] = _DROPPED
        reasons[dup] = DropReason.DUPLICATE_OF_DEV.value
    return codes, reasons, clusters
