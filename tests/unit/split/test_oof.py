from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from amx.spec import Regime, TargetKind
from amx.split import (
    NOT_PREDICTED,
    TRAIN_ONLY,
    Fold,
    SplitError,
    assign_folds,
    cluster_ids,
    oof_folds,
    oof_scheme,
    oof_train_mask,
    train_mask,
)
from tests.unit.split.helpers import fold_of_id, make_forecast_spec, make_frame, make_spec


def refines(folds: np.ndarray, clusters: np.ndarray) -> bool:
    """Every cluster lies inside one fold."""
    return all(np.unique(folds[clusters == c]).size == 1 for c in np.unique(clusters))


def test_oof_scheme_names() -> None:
    assert oof_scheme("iid") == "stratified_cluster_kfold"
    assert oof_scheme(Regime.GROUPED) == "group_kfold"
    assert oof_scheme("temporal") == "forward_chaining"
    with pytest.raises(SplitError):
        oof_scheme("blocked")


def test_iid_oof_is_stratified_and_balanced() -> None:
    uf = make_frame(1003, classes=("class-a", "class-b"), class_probs=(0.9, 0.1))
    folds = oof_folds(uf, 5, "iid", seed=3, target_kind=TargetKind.CATEGORICAL)
    assert set(folds.tolist()) == set(range(5))
    counts = np.bincount(folds)
    assert counts.max() - counts.min() <= 1
    y = uf.target
    for cls in np.unique(y):
        per = np.bincount(folds[y == cls], minlength=5)
        assert per.max() - per.min() <= 1


def test_iid_oof_keeps_duplicate_clusters_whole() -> None:
    dup_of = {200 + i: i % 40 for i in range(120)}  # 40 clusters of size 4
    uf = make_frame(500, dup_of=dup_of)
    clusters = cluster_ids(uf, Regime.IID)
    for given in (clusters, None):
        folds = oof_folds(uf, 5, Regime.IID, seed=1, clusters=given)
        assert refines(folds, clusters)
        counts = np.bincount(folds)
        assert counts.max() - counts.min() <= 4


def test_iid_oof_numeric_target_inferred() -> None:
    uf = make_frame(400, numeric=True)
    a = oof_folds(uf, 4, "iid", seed=0)
    b = oof_folds(uf, 4, "iid", seed=0, target_kind=TargetKind.NUMERIC)
    assert np.array_equal(a, b)


def test_grouped_oof_refines_groups_and_balances_units() -> None:
    rng = np.random.default_rng(4)
    sizes = rng.integers(1, 9, 200)
    groups = np.repeat([f"grp{i}" for i in range(sizes.size)], sizes)
    uf = make_frame(groups.size, groups=groups.tolist(), dup_of={1: 50, 2: 300})
    clusters = cluster_ids(uf, Regime.GROUPED)
    folds = oof_folds(uf, 5, Regime.GROUPED, seed=9, clusters=clusters)
    assert refines(folds, clusters)
    assert refines(folds, np.unique(groups, return_inverse=True)[1])
    counts = np.bincount(folds, minlength=5)
    assert counts.max() - counts.min() <= int(sizes.max())


def test_oof_refines_outer_split_clusters() -> None:
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 6, 250)
    groups = np.repeat([f"grp{i}" for i in range(sizes.size)], sizes)
    uf = make_frame(groups.size, groups=groups.tolist())
    fa = assign_folds(uf, make_spec("grouped", group_columns=["g"]))
    dev_idx = fa.indices(Fold.DEV)
    dev = uf.take(dev_idx)
    folds = oof_folds(dev, 5, fa.regime, seed=7, clusters=fa.clusters[dev_idx])
    assert refines(folds, fa.clusters[dev_idx])


def test_oof_is_row_order_invariant_and_seeded() -> None:
    uf = make_frame(300, dup_of={10: 0, 11: 0})
    perm = np.random.default_rng(2).permutation(uf.n)
    shuffled = uf.take(perm)
    a = oof_folds(uf, 5, "iid", seed=1)
    b = oof_folds(shuffled, 5, "iid", seed=1)
    assert fold_of_id(uf, a) == fold_of_id(shuffled, b)
    assert not np.array_equal(a, oof_folds(uf, 5, "iid", seed=2))


def test_oof_refusals() -> None:
    uf = make_frame(10)
    with pytest.raises(SplitError, match="k >= 2"):
        oof_folds(uf, 1, "iid", seed=0)
    with pytest.raises(SplitError, match="fewer than k"):
        oof_folds(uf.take(range(3)), 5, "iid", seed=0)
    with pytest.raises(SplitError, match="one id per dev unit"):
        oof_folds(uf, 2, "iid", seed=0, clusters=[0, 1])
    with pytest.raises(SplitError, match="empty"):
        oof_folds(uf.take([]), 2, "iid", seed=0)
    with pytest.raises(SplitError, match="time column"):
        oof_folds(uf, 2, "temporal", seed=0)


# temporal -----------------------------------------------------------------------------------


def test_temporal_oof_is_forward_chaining() -> None:
    t = np.repeat(np.arange(60), 3)
    uf = make_frame(t.size, time=t.tolist())
    k = 5
    folds = oof_folds(uf, k, "temporal", seed=0)
    assert set(folds.tolist()) == {TRAIN_ONLY, *range(k)}
    # k + 1 contiguous blocks of distinct time steps; block 0 is training only
    order = [TRAIN_ONLY, *range(k)]
    for prev, nxt in pairwise(order):
        assert t[folds == prev].max() < t[folds == nxt].min()
    steps_per_block = [np.unique(t[folds == f]).size for f in order]
    assert steps_per_block == [10] * 6
    for step in np.unique(t):
        assert np.unique(folds[t == step]).size == 1
    for f in range(k):
        mask = train_mask(folds, f, "temporal", time=t, embargo_steps=0)
        assert np.array_equal(mask, (folds == TRAIN_ONLY) | (folds < f))
        assert t[mask].max() < t[folds == f].min()


def test_temporal_oof_needs_enough_steps() -> None:
    uf = make_frame(30, time=[i % 4 for i in range(30)])
    with pytest.raises(SplitError, match="distinct dev time steps"):
        oof_folds(uf, 5, "temporal", seed=0)


def test_temporal_train_mask_with_embargo() -> None:
    t = np.arange(60)
    folds = np.repeat(np.array([-1, 0, 1, 2, 3, 4]), 10)
    mask = train_mask(folds, 2, "temporal", time=t, embargo_steps=3)
    assert np.flatnonzero(mask).tolist() == list(range(27))
    assert not train_mask(folds, 0, "temporal", time=t, embargo_steps=10).any()
    with pytest.raises(ValueError, match="time"):
        train_mask(folds, 1, "temporal", embargo_steps=1)


def test_iid_train_mask_and_bad_fold() -> None:
    folds = np.array([0, 1, 2, 0, 1, 2])
    assert train_mask(folds, 1, "iid").tolist() == [True, False, True, True, False, True]
    assert np.array_equal(train_mask(folds, 0, Regime.GROUPED), folds != 0)
    with pytest.raises(ValueError, match="not in"):
        train_mask(folds, 3, "iid")


def test_temporal_oof_never_predicts_a_duplicate_of_its_training_units() -> None:
    """Mirror of the outer duplicate_of_dev rule inside forward chaining."""
    t = np.repeat(np.arange(60), 2)  # k = 5: six blocks of 10 steps (20 rows each)
    # row 45 (block 2) copies row 3 (block 0); row 70 (block 3) copies row 50 (block 2);
    # row 110 (block 5) copies row 30 (block 1); rows 100 and 101 are duplicates in one block
    dup_of = {45: 3, 70: 50, 110: 30, 101: 100}
    uf = make_frame(t.size, time=t.tolist(), dup_of=dup_of)
    folds = oof_folds(uf, 5, "temporal", seed=0)
    assert [folds[r] for r in (45, 70, 110)] == [NOT_PREDICTED] * 3
    assert folds[3] == TRAIN_ONLY and folds[50] == 1 and folds[30] == 0
    assert folds[100] == folds[101] == 4  # a duplicate inside the predicted block is no leak
    keys = uf.input_row_keys()
    for f in range(5):
        train = train_mask(folds, f, "temporal", time=t, embargo_steps=0)
        predicted = folds == f
        assert predicted.any() and not (train & predicted).any()
        assert not set(keys[predicted].tolist()) & set(keys[train].tolist()), f
        assert t[train].max() < t[predicted].min()
    # a not-predicted unit still trains every later fold, and never an earlier one
    assert train_mask(folds, 4, "temporal", time=t, embargo_steps=0)[[45, 70]].all()
    assert not train_mask(folds, 1, "temporal", time=t, embargo_steps=0)[45]


def test_temporal_oof_refuses_a_fold_left_without_predicted_units() -> None:
    t = np.arange(12)  # k = 2: blocks of 4 steps; block 2 copies block 0 entirely
    uf = make_frame(t.size, time=t.tolist(), dup_of={8: 0, 9: 1, 10: 2, 11: 3})
    with pytest.raises(SplitError, match="no units left to predict"):
        oof_folds(uf, 2, "temporal", seed=0)


def test_temporal_train_mask_needs_time_and_embargo() -> None:
    t = np.arange(60)
    folds = np.repeat(np.array([-1, 0, 1, 2, 3, 4]), 10)
    with pytest.raises(SplitError, match="time and embargo_steps"):
        train_mask(folds, 1, "temporal")
    with pytest.raises(SplitError, match="time and embargo_steps"):
        train_mask(folds, 1, "temporal", time=t)
    with pytest.raises(SplitError, match="time and embargo_steps"):
        train_mask(folds, 1, "temporal", embargo_steps=0)
    assert train_mask(folds, 1, "iid").sum() == 50  # iid/grouped need neither


def test_oof_train_mask_reads_the_forecasting_embargo_from_the_spec() -> None:
    spec = make_forecast_spec(horizons=(1, 3), max_lag=2)  # embargo defaults to 3 + 2 = 5
    assert spec.embargo_steps == 5
    t = np.repeat(np.arange(120), 2)
    dev = make_frame(t.size, time=t.tolist())
    folds = oof_folds(dev, 5, "temporal", seed=0)
    for f in range(5):
        mask = oof_train_mask(dev, folds, f, spec)
        expected = train_mask(folds, f, "temporal", time=t, embargo_steps=5)
        assert np.array_equal(mask, expected)
        steps = np.unique(t)
        gap = steps[(steps > t[mask].max()) & (steps < t[folds == f].min())]
        assert gap.size == 5  # exactly max(horizons) + max_lag steps are held back
    explicit = make_forecast_spec(embargo=8)
    m8 = oof_train_mask(dev, folds, 2, explicit)
    assert np.array_equal(m8, train_mask(folds, 2, "temporal", time=t, embargo_steps=8))
    with pytest.raises(SplitError, match="max_lag"):
        oof_train_mask(dev, folds, 2, make_forecast_spec(embargo=1))
    # iid ignores time and embargo; auto must be resolved first
    iid = oof_folds(dev, 5, "iid", seed=0)
    assert np.array_equal(oof_train_mask(dev, iid, 0, make_spec()), iid != 0)
    with pytest.raises(SplitError, match="resolved regime"):
        oof_train_mask(dev, iid, 0, make_spec("auto"))
    assert np.array_equal(oof_train_mask(dev, iid, 0, make_spec("auto"), regime="iid"), iid != 0)
    with pytest.raises(SplitError, match="one fold per dev unit"):
        oof_train_mask(dev, folds[:-1], 0, spec)
