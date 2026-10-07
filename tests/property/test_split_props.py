"""Property tests for amx.split (ROLLER step 5): synthetic frames only."""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from amx.data import UnitFrame
from amx.spec import Regime
from amx.split import (
    NOT_PREDICTED,
    Fold,
    SplitError,
    assign_folds,
    cluster_ids,
    oof_folds,
    train_mask,
)
from tests.unit.split.helpers import make_frame, make_spec

FRACTIONS = st.sampled_from([(0.6, 0.2, 0.2), (0.5, 0.25, 0.25), (0.4, 0.3, 0.3), (0.7, 0.2, 0.1)])
SEEDS = st.integers(0, 2**31 - 1)


def one_fold_per(labels: np.ndarray, folds: np.ndarray) -> bool:
    return all(np.unique(folds[labels == g]).size == 1 for g in np.unique(labels))


@settings(max_examples=40, deadline=None)
@given(
    sizes=st.lists(st.integers(1, 8), min_size=12, max_size=80),
    n_dups=st.integers(0, 10),
    fractions=FRACTIONS,
    seed=SEEDS,
)
def test_no_group_straddles_folds(
    sizes: list[int], n_dups: int, fractions: tuple[float, ...], seed: int
) -> None:
    groups = np.repeat(np.arange(len(sizes)), sizes)
    n = groups.size
    rng = np.random.default_rng(seed)
    dup_of = {int(a): int(b) for a, b in rng.integers(0, n, (n_dups, 2)) if a != b}
    uf = make_frame(n, seed=seed % 1000, groups=[f"grp{g}" for g in groups], dup_of=dup_of)
    spec = make_spec("grouped", group_columns=["g"], seed=seed, fractions=fractions)
    try:
        fa = assign_folds(uf, spec)
    except SplitError:
        assume(False)  # too few clusters for three non-empty folds
        return
    assert one_fold_per(groups, fa.folds)
    assert one_fold_per(uf.input_row_keys(), fa.folds)  # duplicates never straddle either
    assert one_fold_per(fa.clusters, fa.folds)
    assert fa.counts["dropped"] == 0 and sum(fa.counts.values()) == n
    dev_idx = fa.indices(Fold.DEV)
    if np.unique(fa.clusters[dev_idx]).size >= 3:
        oof = oof_folds(uf.take(dev_idx), 3, Regime.GROUPED, seed, fa.clusters[dev_idx])
        assert one_fold_per(fa.clusters[dev_idx], oof)  # OOF folds refine the clusters


@settings(max_examples=40, deadline=None)
@given(
    class_counts=st.lists(st.integers(1, 60), min_size=1, max_size=6),
    fractions=FRACTIONS,
    seed=SEEDS,
)
def test_iid_singleton_strata_within_one(
    class_counts: list[int], fractions: tuple[float, ...], seed: int
) -> None:
    labels = np.repeat([f"class-{i}" for i in range(len(class_counts))], class_counts)
    labels = labels[np.random.default_rng(seed).permutation(labels.size)]
    n = labels.size
    uf = make_frame(n, seed=seed % 1000)
    table = uf.table.set_column(uf.table.column_names.index("y"), "y", [labels.tolist()])
    uf = UnitFrame(table, uf.roles)
    spec = make_spec(seed=seed, fractions=fractions)
    try:
        fa = assign_folds(uf, spec)
    except SplitError:
        assume(False)
        return
    for cls in np.unique(labels):
        mask = labels == cls
        m = int(mask.sum())
        for f, frac in zip(("dev", "calib", "sealed"), fractions, strict=True):
            size = int(np.count_nonzero(fa.folds[mask] == f))
            assert abs(size - m * frac) < 1.0 + 1e-9
    k = 3
    dev_idx = fa.indices(Fold.DEV)
    if dev_idx.size >= k:
        oof = oof_folds(uf.take(dev_idx), k, "iid", seed, fa.clusters[dev_idx])
        counts = np.bincount(oof, minlength=k)
        assert counts.max() - counts.min() <= 1


@settings(max_examples=60, deadline=None)
@given(
    stratum_sizes=st.lists(st.integers(1, 4), min_size=20, max_size=150),
    fractions=FRACTIONS,
    seed=SEEDS,
)
def test_iid_many_small_strata_keep_totals_near_fractions(
    stratum_sizes: list[int], fractions: tuple[float, ...], seed: int
) -> None:
    """Per-stratum rounding must not drift the fold totals (each stratum within ±1 and every
    total within ±1 of n * fraction)."""
    labels = np.repeat([f"class-{i}" for i in range(len(stratum_sizes))], stratum_sizes)
    labels = labels[np.random.default_rng(seed).permutation(labels.size)]
    n = labels.size
    uf = make_frame(n, seed=seed % 1000)
    table = uf.table.set_column(uf.table.column_names.index("y"), "y", [labels.tolist()])
    uf = UnitFrame(table, uf.roles)
    fa = assign_folds(uf, make_spec(seed=seed, fractions=fractions))
    names = ("dev", "calib", "sealed")
    for cls in np.unique(labels):
        mask = labels == cls
        for f, frac in zip(names, fractions, strict=True):
            size = int(np.count_nonzero(fa.folds[mask] == f))
            assert abs(size - int(mask.sum()) * frac) < 1.0 + 1e-9
    for f, frac in zip(names, fractions, strict=True):
        assert abs(fa.counts[f] - n * frac) < 1.0 + 1e-9, (fa.counts, n, fractions)


@settings(max_examples=40, deadline=None)
@given(
    per_step=st.lists(st.integers(1, 5), min_size=20, max_size=120),
    embargo=st.integers(0, 3),
    fractions=FRACTIONS,
    seed=SEEDS,
)
def test_temporal_order_and_exact_embargo(
    per_step: list[int], embargo: int, fractions: tuple[float, ...], seed: int
) -> None:
    t = np.repeat(np.arange(len(per_step)) * 10, per_step)  # gaps in time values are fine
    uf = make_frame(t.size, seed=seed % 1000, time=t.tolist())
    spec = make_spec("temporal", time_column="t", embargo=embargo, seed=seed, fractions=fractions)
    try:
        fa = assign_folds(uf, spec)
    except SplitError:
        assume(False)
        return
    dev_t, cal_t, sea_t = (t[fa.mask(f)] for f in (Fold.DEV, Fold.CALIB, Fold.SEALED))
    assert dev_t.max() < cal_t.min() and cal_t.max() < sea_t.min()
    steps = np.unique(t)
    assert ((steps > dev_t.max()) & (steps < cal_t.min())).sum() == embargo
    assert ((steps > cal_t.max()) & (steps < sea_t.min())).sum() == embargo
    assert one_fold_per(t, fa.folds)
    if steps[steps <= dev_t.max()].size >= 4:
        oof = oof_folds(uf.take(fa.indices(Fold.DEV)), 3, "temporal", seed)
        assert one_fold_per(dev_t, oof)
        for f in (-1, 0, 1):
            assert dev_t[oof == f].max() < dev_t[oof == f + 1].min()


@settings(max_examples=40, deadline=None)
@given(
    per_step=st.lists(st.integers(1, 4), min_size=12, max_size=80),
    n_dups=st.integers(0, 40),
    embargo=st.integers(0, 2),
    seed=SEEDS,
)
def test_temporal_oof_predicts_no_duplicate_of_its_training_units(
    per_step: list[int], n_dups: int, embargo: int, seed: int
) -> None:
    t = np.repeat(np.arange(len(per_step)), per_step)
    rng = np.random.default_rng(seed)
    dup_of = {int(a): int(b) for a, b in rng.integers(0, t.size, (n_dups, 2)) if a != b}
    uf = make_frame(t.size, seed=seed % 1000, time=t.tolist(), dup_of=dup_of)
    try:
        folds = oof_folds(uf, 3, "temporal", seed)
    except SplitError:
        assume(False)  # a fold left without units to predict
        return
    keys = uf.input_row_keys()
    for f in range(3):
        train = train_mask(folds, f, "temporal", time=t, embargo_steps=embargo)
        predicted = folds == f
        assert predicted.any()
        assert not set(keys[predicted].tolist()) & set(keys[train].tolist())
        if train.any():
            assert t[train].max() < t[predicted].min()
    # a unit is held back only when its inputs occur at an earlier time
    for row in np.flatnonzero(folds == NOT_PREDICTED):
        assert np.any((keys == keys[row]) & (t < t[row]))


@settings(max_examples=25, deadline=None)
@given(
    n=st.integers(30, 200),
    regime=st.sampled_from(["iid", "grouped", "temporal"]),
    seed=SEEDS,
)
def test_assignment_ignores_row_order(n: int, regime: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    kw: dict[str, Any] = {"dup_of": {1: 0, 2: 0}}
    spec_kw: dict[str, Any] = {"seed": seed}
    if regime == "grouped":
        kw["groups"] = rng.integers(0, max(4, n // 3), n).tolist()
        spec_kw["group_columns"] = ["g"]
    if regime == "temporal":
        kw["time"] = rng.integers(0, max(10, n // 2), n).tolist()
        spec_kw["time_column"] = "t"
    uf = make_frame(n, seed=seed % 1000, **kw)
    spec = make_spec(regime, **spec_kw)
    try:
        a = assign_folds(uf, spec)
    except SplitError:
        assume(False)
        return
    perm = rng.permutation(n)
    b = assign_folds(uf.take(perm), spec)
    assert np.array_equal(a.folds[perm], b.folds)
    assert np.array_equal(cluster_ids(uf, Regime(regime))[perm], b.clusters)
