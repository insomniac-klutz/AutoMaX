"""Clusters that must stay inside one fold: exact duplicates and groups (HANDOFF 8.2, OQ Q7).

Every function here works on a *canonical* unit order (units sorted by id), so cluster ids
and everything derived from them depend on the data content and the seed only, never on the
row order of the input table (the data hash is row-order independent too).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from amx._log import get_logger
from amx.data.unitframe import UnitFrame
from amx.spec.enums import Regime
from amx.split.errors import SplitError

log = get_logger(__name__)


def canonical_order(uf: UnitFrame) -> NDArray[np.int64]:
    """Row positions of ``uf`` sorted by unit id (ids are unique, so the order is total)."""
    ids = uf.ids.astype(str)
    return np.asarray(np.argsort(ids, kind="stable"), dtype=np.int64)


def dense_by_first(labels: NDArray[Any]) -> NDArray[np.int64]:
    """Relabel to 0..C-1, numbering labels by their first appearance."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros(0, dtype=np.int64)
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size, dtype=np.int64)
    return rank[np.asarray(inverse).reshape(-1)]


def duplicate_keys(uf: UnitFrame) -> NDArray[np.uint64] | None:
    """Per-unit hash of the input values, or None when the frame has no input columns.

    With no inputs every unit would be a "duplicate" of every other, which says nothing
    about leakage; duplicate handling is skipped in that case.
    """
    if not uf.roles.inputs:
        return None
    return uf.input_row_keys()


def merge_labels(*labelings: NDArray[Any] | None) -> NDArray[np.int64]:
    """Connected components of units linked by sharing any label (union-find over labelings).

    Returns dense ids numbered by first appearance in the given unit order.
    """
    present = [np.asarray(lab) for lab in labelings if lab is not None]
    if not present:
        raise ValueError("merge_labels needs at least one labeling")
    n = present[0].shape[0]
    if any(lab.shape[0] != n for lab in present):
        raise ValueError("labelings must have the same length")
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    offset = n
    units = np.arange(n, dtype=np.int64)
    for lab in present:
        codes = dense_by_first(lab)
        rows.append(units)
        cols.append(offset + codes)
        offset += int(np.max(codes)) + 1
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    graph = coo_matrix((np.ones(r.size, dtype=np.int8), (r, c)), shape=(offset, offset)).tocsr()
    _, comp = connected_components(graph, directed=False)
    return dense_by_first(np.asarray(comp)[:n])


def canonical_clusters(
    uf: UnitFrame, regime: Regime, order: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Cluster id per unit in canonical order (``order`` from :func:`canonical_order`).

    * iid and temporal: exact-duplicate input clusters.
    * grouped: union of group keys and duplicate clusters, so neither a group nor a duplicate
      cluster can straddle folds.
    """
    keys = duplicate_keys(uf)
    dup = None if keys is None else keys[order]
    if regime is Regime.GROUPED:
        cols = uf.roles.group_columns
        if not cols:
            raise SplitError("regime 'grouped' needs data.group_columns")
        # every group column separately: no id of ANY listed column may straddle folds (6.1)
        per_column = [np.asarray(uf.column(c), dtype=object).astype(str)[order] for c in cols]
        return merge_labels(*per_column, dup)
    if dup is None:
        return np.arange(uf.n, dtype=np.int64)
    return dense_by_first(dup)


def cluster_ids(uf: UnitFrame, regime: Regime) -> NDArray[np.int64]:
    """Cluster id per unit, aligned with the rows of ``uf``.

    Ids are numbered by first appearance in unit-id order, so a unit keeps its cluster id
    when the rows of ``uf`` are permuted.
    """
    order = canonical_order(uf)
    out = np.empty(uf.n, dtype=np.int64)
    out[order] = canonical_clusters(uf, Regime(regime), order)
    return out
