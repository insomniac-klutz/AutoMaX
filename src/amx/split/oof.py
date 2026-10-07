"""Out-of-fold (OOF) folds on dev that follow the outer regime (HANDOFF 8.1, ROLLER C16).

* ``iid``: K folds stratified like the outer split; whole duplicate clusters in one fold.
* ``grouped``: whole clusters (groups merged with duplicate clusters) per fold, balanced by
  unit count (GroupKFold).
* ``temporal``: forward chaining. Distinct time steps are cut into K + 1 contiguous blocks;
  block 0 is training only (fold -1, ``TRAIN_ONLY``, never predicted) and blocks 1..K are
  folds 0..K-1. Fold f trains on everything earlier, minus an embargo of ``embargo_steps``
  distinct time steps just before the fold (:func:`oof_train_mask` reads it from the spec).
  Mirroring the outer ``duplicate_of_dev`` rule, a unit whose inputs exactly equal those of a
  unit in an earlier block is never predicted: it gets fold -2 (``NOT_PREDICTED``). It still
  trains the folds that come after it in time, so no predicted unit has an exact duplicate in
  its training set.

Fold assignment always refines the cluster ids: a cluster never spans two OOF folds
(iid/grouped). Computation runs in unit-id order, so results do not depend on row order.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amx._log import get_logger
from amx.data.unitframe import UnitFrame
from amx.spec.enums import Regime, TargetKind
from amx.spec.models import TaskSpec
from amx.split.dedupe import canonical_clusters, canonical_order, dense_by_first, duplicate_keys
from amx.split.errors import SplitError
from amx.split.regimes import check_embargo, first_index, make_rng, stratum_codes, time_values

log = get_logger(__name__)

TRAIN_ONLY = -1
NOT_PREDICTED = -2
_OOF_STREAM = 0x00F
OofScheme = Literal["stratified_cluster_kfold", "group_kfold", "forward_chaining"]


def oof_scheme(regime: Regime | str) -> OofScheme:
    """Name of the OOF scheme used for ``regime`` (recorded in the split manifest)."""
    r = Regime(regime)
    if r is Regime.IID:
        return "stratified_cluster_kfold"
    if r is Regime.GROUPED:
        return "group_kfold"
    if r is Regime.TEMPORAL:
        return "forward_chaining"
    raise SplitError(f"no OOF scheme for regime '{r.value}'")


def oof_folds(
    dev: UnitFrame,
    k: int,
    regime: Regime | str,
    seed: int,
    clusters: ArrayLike | None = None,
    *,
    target_kind: TargetKind | None = None,
) -> NDArray[np.int64]:
    """OOF fold per dev unit: 0..k-1, or for ``temporal`` -1 (``TRAIN_ONLY``, the first block)
    and -2 (``NOT_PREDICTED``, an exact duplicate of a unit in an earlier block).

    ``clusters`` (aligned with the rows of ``dev``) are the outer split's cluster ids; when
    omitted they are recomputed from ``dev``. ``target_kind`` selects the iid stratification;
    when omitted, a floating-point target is binned and anything else is treated as
    categorical.
    """
    r = Regime(regime)
    if k < 2:
        raise SplitError(f"OOF needs k >= 2 folds, got {k}")
    if dev.n == 0:
        raise SplitError("cannot build OOF folds on an empty dev fold")
    if r is Regime.TEMPORAL:
        return _forward_chaining(dev, k)
    if r not in (Regime.IID, Regime.GROUPED):
        raise SplitError(f"no OOF scheme for regime '{r.value}'")
    order = canonical_order(dev)
    if clusters is None:
        cl = canonical_clusters(dev, r, order)
    else:
        given = np.asarray(clusters)
        if given.shape != (dev.n,):
            raise SplitError(f"clusters must have one id per dev unit ({dev.n}), got {given.shape}")
        cl = dense_by_first(given[order])
    n_clusters = int(np.max(cl)) + 1
    if n_clusters < k:
        raise SplitError(f"dev has {n_clusters} clusters, fewer than k = {k} OOF folds")
    sizes = np.bincount(cl, minlength=n_clusters).astype(np.int64)
    if r is Regime.IID and dev.has_target:
        first = first_index(cl, n_clusters)
        target = dev.target[order]
        kind = target_kind if target_kind is not None else _infer_kind(target)
        strata = stratum_codes(target[first], kind)
    else:
        strata = np.zeros(n_clusters, dtype=np.int64)
    rng = make_rng(seed, _OOF_STREAM)
    fold_of_cluster = _balanced_assign(sizes, strata, k, rng)
    out = np.empty(dev.n, dtype=np.int64)
    out[order] = fold_of_cluster[cl]
    return out


def train_mask(
    folds: ArrayLike,
    f: int,
    regime: Regime | str,
    *,
    time: ArrayLike | None = None,
    embargo_steps: int | None = None,
) -> NDArray[np.bool_]:
    """Units that may be used to fit the model that predicts OOF fold ``f``.

    iid/grouped: ``folds != f`` (``time`` and ``embargo_steps`` are ignored).

    temporal: ``time`` (one value per unit) and ``embargo_steps`` are required; use
    :func:`oof_train_mask`, which reads both from the dev frame and the spec. Training units
    are the units earlier in time than fold ``f``: ``TRAIN_ONLY`` units, folds ``< f`` and
    ``NOT_PREDICTED`` units before the fold, minus the last ``embargo_steps`` distinct time
    steps before it.
    """
    fo = np.asarray(folds, dtype=np.int64)
    r = Regime(regime)
    k = int(np.max(fo)) + 1 if fo.size else 0
    if not (0 <= f < k):
        raise ValueError(f"fold {f} is not in 0..{k - 1}")
    if r in (Regime.IID, Regime.GROUPED):
        return np.asarray(fo != f, dtype=np.bool_)
    if r is not Regime.TEMPORAL:
        raise SplitError(f"no OOF scheme for regime '{r.value}'")
    if time is None or embargo_steps is None:
        raise SplitError(
            "a temporal train mask needs time and embargo_steps (HANDOFF 7.5); use "
            "oof_train_mask(dev, folds, f, spec), which reads spec.embargo_steps"
        )
    if embargo_steps < 0:
        raise ValueError(f"embargo_steps must be >= 0, got {embargo_steps}")
    t = np.asarray(time)
    if t.shape != fo.shape:
        raise ValueError("time must have one value per unit")
    t0 = t[fo == f].min()
    earlier = t < t0
    mask = np.asarray(
        ((fo == TRAIN_ONLY) | (fo == NOT_PREDICTED) | ((fo >= 0) & (fo < f))) & earlier,
        dtype=np.bool_,
    )
    if embargo_steps > 0:
        before = np.unique(t[earlier])
        if before.size <= embargo_steps:
            mask[:] = False
        else:
            mask &= t < before[-embargo_steps]
    return mask


def oof_train_mask(
    dev: UnitFrame,
    folds: ArrayLike,
    f: int,
    spec: TaskSpec,
    *,
    regime: Regime | str | None = None,
) -> NDArray[np.bool_]:
    """Training units for OOF fold ``f`` of ``dev``: the public entry point.

    ``regime`` defaults to ``spec.splits.regime``; pass the resolved regime (the manifest's
    ``regime``) when the spec says ``auto``. For ``temporal`` the time values come from
    ``dev`` and the embargo is ``spec.embargo_steps``, which must reach
    ``max(horizons) + max_lag`` for forecasting (HANDOFF 7.5).
    """
    r = spec.splits.regime if regime is None else Regime(regime)
    if r is Regime.AUTO:
        raise SplitError(
            "regime 'auto' is ambiguous here: pass the resolved regime "
            "(the 'regime' field of splits.manifest.json)"
        )
    fo = np.asarray(folds, dtype=np.int64)
    if fo.shape != (dev.n,):
        raise SplitError(f"folds must hold one fold per dev unit ({dev.n}), got {fo.shape}")
    if r is Regime.TEMPORAL:
        return train_mask(fo, f, r, time=time_values(dev), embargo_steps=check_embargo(spec))
    return train_mask(fo, f, r)


# helpers ---------------------------------------------------------------------------------


def _infer_kind(target: NDArray[Any]) -> TargetKind:
    if np.issubdtype(np.asarray(target).dtype, np.floating):
        return TargetKind.NUMERIC
    return TargetKind.CATEGORICAL


def _balanced_assign(
    sizes: NDArray[np.int64], strata: NDArray[np.int64], k: int, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Assign clusters to k folds, balancing unit counts within each stratum and overall.

    Within a stratum, clusters are shuffled, ordered by size (largest first) and each goes to
    the fold with the fewest units of that stratum, ties broken by the fewest units overall.
    Singleton-only strata use the equivalent round-robin.
    """
    n_clusters = sizes.size
    fold = np.empty(n_clusters, dtype=np.int64)
    total = np.zeros(k, dtype=np.int64)
    for s in range(int(np.max(strata)) + 1 if n_clusters else 0):
        members = np.flatnonzero(strata == s)
        if members.size == 0:
            continue
        members = members[rng.permutation(members.size)]
        members = members[np.argsort(-sizes[members], kind="stable")]
        if int(np.max(sizes[members])) == 1:
            ring = np.argsort(total, kind="stable")
            assigned = ring[np.arange(members.size) % k]
            fold[members] = assigned
            total += np.bincount(assigned, minlength=k)
            continue
        local = np.zeros(k, dtype=np.int64)
        for c in members:
            key = local * (int(total.sum()) + 1) + total
            j = int(np.argmin(key))
            fold[c] = j
            local[j] += sizes[c]
            total[j] += sizes[c]
    return fold


def _forward_chaining(dev: UnitFrame, k: int) -> NDArray[np.int64]:
    steps, inverse = np.unique(time_values(dev), return_inverse=True)
    if steps.size < k + 1:
        raise SplitError(
            f"forward chaining with k = {k} needs at least {k + 1} distinct dev time steps, "
            f"got {steps.size}"
        )
    block_of_step = np.empty(steps.size, dtype=np.int64)
    for b, chunk in enumerate(np.array_split(np.arange(steps.size), k + 1)):
        block_of_step[chunk] = b
    block = block_of_step[np.asarray(inverse).reshape(-1)]
    folds = block - 1
    keys = duplicate_keys(dev)
    if keys is not None:
        unique_keys, inverse_keys = np.unique(keys, return_inverse=True)
        key_of_unit = np.asarray(inverse_keys, dtype=np.int64).reshape(-1)
        first_block = np.full(unique_keys.size, k + 1, dtype=np.int64)
        np.minimum.at(first_block, key_of_unit, block)
        repeat = (block >= 1) & (block > first_block[key_of_unit])
        folds[repeat] = NOT_PREDICTED
        if repeat.any():
            log.info(
                "forward chaining: %d dev units duplicate an earlier block and are not predicted",
                int(repeat.sum()),
            )
    empty = [f for f in range(k) if not np.any(folds == f)]
    if empty:
        raise SplitError(
            f"forward chaining: OOF fold(s) {empty} have no units left to predict after "
            "removing exact duplicates of earlier blocks; use fewer OOF folds or more "
            "distinct inputs"
        )
    return np.asarray(folds, dtype=np.int64)
