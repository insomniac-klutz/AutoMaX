from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from amx.data import UnitFrame
from amx.data.unitframe import Roles
from amx.spec import Regime, TargetKind, parse_taskspec
from amx.split import (
    DropReason,
    Fold,
    SplitError,
    assign_folds,
    cluster_ids,
    largest_remainder,
    oof_folds,
    stratum_codes,
)
from amx.split.regimes import allocate_strata, cut_by_cumulative
from tests.unit.split.helpers import fold_of_id, make_forecast_spec, make_frame, make_spec

FRACTION_SETS = [(0.6, 0.2, 0.2), (0.5, 0.3, 0.2), (0.7, 0.15, 0.15), (1 / 3, 1 / 3, 1 / 3)]


def assert_within_one(sizes: dict[str, int], n: int, fractions: tuple[float, ...]) -> None:
    for f, frac in zip(("dev", "calib", "sealed"), fractions, strict=True):
        assert abs(sizes.get(f, 0) - n * frac) < 1.0 + 1e-9, (sizes, n, fractions)


# entry point ------------------------------------------------------------------------------


def test_auto_and_blocked_are_refused() -> None:
    uf = make_frame(50)
    with pytest.raises(SplitError, match="profiler"):
        assign_folds(uf, make_spec("auto"))
    with pytest.raises(NotImplementedError, match="deferred"):
        assign_folds(uf, make_spec("iid"), Regime.BLOCKED)


def test_empty_frame_and_tiny_frame_are_refused() -> None:
    uf = make_frame(1)
    with pytest.raises(SplitError):
        assign_folds(uf.take([]), make_spec())
    with pytest.raises(SplitError, match="empty"):
        assign_folds(uf, make_spec())


def test_regime_argument_overrides_spec() -> None:
    uf = make_frame(100)
    fa = assign_folds(uf, make_spec("auto"), "iid")
    assert fa.regime is Regime.IID
    assert sum(fa.counts.values()) == 100 and fa.counts["dropped"] == 0
    # every stratum within ±1 of its quota and the totals within ±1 (here: exact)
    assert {f: fa.counts[f] for f in ("dev", "calib", "sealed")} == {
        "dev": 60,
        "calib": 20,
        "sealed": 20,
    }


# iid ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("fractions", FRACTION_SETS)
@pytest.mark.parametrize("seed", [0, 1, 99])
def test_iid_categorical_strata_within_one(fractions: tuple[float, ...], seed: int) -> None:
    uf = make_frame(
        997,
        seed=seed,
        classes=("class-alpha", "class-beta", "class-gamma", "class-delta"),
        class_probs=(0.7, 0.2, 0.08, 0.02),
    )
    fa = assign_folds(uf, make_spec(seed=seed, fractions=fractions))
    y = uf.target
    for cls in np.unique(y):
        mask = y == cls
        sizes = {f: int(np.count_nonzero(fa.folds[mask] == f)) for f in ("dev", "calib", "sealed")}
        assert_within_one(sizes, int(mask.sum()), fractions)
    assert fa.counts["dropped"] == 0
    assert set(fa.drop_reasons.tolist()) == {""}


@pytest.mark.parametrize("seed", [3, 4])
def test_iid_numeric_target_uses_ten_quantile_bins(seed: int) -> None:
    uf = make_frame(1500, seed=seed, numeric=True)
    fractions = (0.6, 0.2, 0.2)
    fa = assign_folds(uf, make_spec(kind="numeric", seed=seed))
    strata = stratum_codes(uf.target, TargetKind.NUMERIC)
    assert np.unique(strata).size == 10
    for s in np.unique(strata):
        mask = strata == s
        sizes = {f: int(np.count_nonzero(fa.folds[mask] == f)) for f in ("dev", "calib", "sealed")}
        assert_within_one(sizes, int(mask.sum()), fractions)


def test_iid_duplicates_never_straddle_folds() -> None:
    n = 600
    # clusters of size 2, 3 and 4 built from rows 0..59
    dup_of = {}
    for i in range(60):
        for c in range(i % 3 + 1):
            dup_of[100 + 3 * i + c] = i
    uf = make_frame(n, seed=5, dup_of=dup_of)
    fa = assign_folds(uf, make_spec())
    keys = uf.input_row_keys()
    by_key: dict[int, set[str]] = defaultdict(set)
    for k, f in zip(keys.tolist(), fa.folds.tolist(), strict=True):
        by_key[k].add(f)
    assert all(len(v) == 1 for v in by_key.values())
    # the cluster ids returned are the duplicate clusters
    assert np.unique(fa.clusters).size == np.unique(keys).size
    assert np.array_equal(fa.clusters, cluster_ids(uf, Regime.IID))
    dev_keys = set(keys[fa.mask(Fold.DEV)].tolist())
    held = keys[fa.mask(Fold.CALIB) | fa.mask(Fold.SEALED)]
    assert not dev_keys.intersection(held.tolist())


def test_iid_refuses_group_columns() -> None:
    uf = make_frame(50, groups=[i % 5 for i in range(50)])
    with pytest.raises(SplitError, match="grouped"):
        assign_folds(uf, make_spec("auto"), "iid")


def test_iid_seed_controls_assignment_and_row_order_does_not() -> None:
    uf = make_frame(400, seed=1)
    a = assign_folds(uf, make_spec(seed=11))
    b = assign_folds(uf, make_spec(seed=11))
    c = assign_folds(uf, make_spec(seed=12))
    assert np.array_equal(a.folds, b.folds)
    assert not np.array_equal(a.folds, c.folds)
    perm = np.random.default_rng(0).permutation(uf.n)
    shuffled = uf.take(perm)
    d = assign_folds(shuffled, make_spec(seed=11))
    assert fold_of_id(shuffled, d.folds) == fold_of_id(uf, a.folds)


# grouped ------------------------------------------------------------------------------------


def test_grouped_no_group_straddles_and_sizes_track_fractions() -> None:
    rng = np.random.default_rng(2)
    sizes = rng.integers(1, 12, 300)
    groups = np.repeat([f"grp{i}" for i in range(sizes.size)], sizes)
    uf = make_frame(groups.size, groups=groups.tolist())
    spec = make_spec("grouped", group_columns=["g"])
    fa = assign_folds(uf, spec)
    for g in np.unique(groups):
        assert np.unique(fa.folds[groups == g]).size == 1
    largest = int(sizes.max())
    for f, frac in zip(("dev", "calib", "sealed"), (0.6, 0.2, 0.2), strict=True):
        assert abs(fa.counts[f] - frac * uf.n) <= largest


def test_grouped_merges_groups_linked_by_duplicates() -> None:
    n = 400
    groups = [f"grp{i // 4}" for i in range(n)]
    # unit 5 (group 1) duplicates unit 0 (group 0); unit 13 (group 3) duplicates unit 9 (group 2)
    uf = make_frame(n, groups=groups, dup_of={5: 0, 13: 9})
    fa = assign_folds(uf, make_spec("grouped", group_columns=["g"]))
    g = np.asarray(groups)
    assert np.unique(fa.folds[np.isin(g, ["grp0", "grp1"])]).size == 1
    assert np.unique(fa.folds[np.isin(g, ["grp2", "grp3"])]).size == 1
    assert np.unique(fa.clusters[np.isin(g, ["grp0", "grp1"])]).size == 1


def test_grouped_needs_groups_and_raises_on_empty_fold() -> None:
    uf = make_frame(30)
    with pytest.raises(SplitError, match="group_columns"):
        assign_folds(uf, make_spec("auto"), "grouped")
    one_group = make_frame(30, groups=["only"] * 30)
    with pytest.raises(SplitError, match="empty"):
        assign_folds(one_group, make_spec("grouped", group_columns=["g"]))


def test_grouped_null_group_keys_form_one_group() -> None:
    groups: list[str | None] = [None if i % 3 == 0 else f"grp{i // 3}" for i in range(300)]
    uf = make_frame(300, groups=groups)
    fa = assign_folds(uf, make_spec("grouped", group_columns=["g"]))
    nulls = np.asarray([g is None for g in groups])
    assert np.unique(fa.folds[nulls]).size == 1


# temporal -----------------------------------------------------------------------------------


@pytest.mark.parametrize("embargo", [0, 1, 4])
def test_temporal_blocks_are_ordered_with_exact_embargo(embargo: int) -> None:
    t = np.repeat(np.arange(200), np.random.default_rng(0).integers(1, 6, 200))
    uf = make_frame(t.size, time=t.tolist(), series=[f"s{i % 4}" for i in range(t.size)])
    spec = make_spec("temporal", time_column="t", series_columns=["s"], embargo=embargo)
    assert spec.embargo_steps == embargo
    fa = assign_folds(uf, spec)
    assert fa.embargo_steps == embargo
    dev_t, cal_t, sea_t = (t[fa.mask(f)] for f in (Fold.DEV, Fold.CALIB, Fold.SEALED))
    assert dev_t.max() < cal_t.min() and cal_t.max() < sea_t.min()
    steps = np.unique(t)
    gap1 = steps[(steps > dev_t.max()) & (steps < cal_t.min())]
    gap2 = steps[(steps > cal_t.max()) & (steps < sea_t.min())]
    assert gap1.size == embargo and gap2.size == embargo
    dropped = fa.mask(Fold.DROPPED)
    assert set(np.unique(t[dropped]).tolist()) == {*gap1.tolist(), *gap2.tolist()}
    assert set(fa.drop_reasons[dropped].tolist()) <= {DropReason.EMBARGO.value}
    assert fa.counts["dropped"] == int(np.isin(t, np.concatenate([gap1, gap2])).sum())
    if embargo:
        assert fa.dropped_reasons == {"embargo": fa.counts["dropped"]}
    # a time value never straddles folds
    for step in steps:
        assert np.unique(fa.folds[t == step]).size == 1
    # series span folds in time
    s = uf.series
    assert s is not None
    assert np.unique(fa.folds[s == "s0"]).size >= 3


def test_temporal_embargo_defaults_to_horizon_plus_lag() -> None:
    raw = make_spec("temporal", time_column="t").model_dump(mode="json")
    raw["task"] = {
        "family": "forecasting",
        "commit_unit": "series_horizon",
        "loss": {"kind": "builtin", "name": "err_gt_tol", "params": {"tol": 0.1}},
        "forecast": {"horizons": [1, 3], "max_lag": 2},
    }
    raw["data"]["target"]["kind"] = "series"
    spec = parse_taskspec(raw)
    assert spec.embargo_steps == 5
    t = np.arange(300)
    uf = make_frame(t.size, time=t.tolist())
    fa = assign_folds(uf, spec)
    assert fa.counts["dropped"] == 10 and fa.embargo_steps == 5


def test_forecasting_embargo_below_horizon_plus_lag_is_refused() -> None:
    """HANDOFF 7.5: the embargo is at least the maximum horizon plus the maximum lag."""
    uf = make_frame(300, time=list(range(300)))
    for embargo in (0, 1, 4):
        spec = make_forecast_spec(embargo=embargo, horizons=(1, 3), max_lag=2)
        assert spec.embargo_steps == embargo
        with pytest.raises(SplitError, match=r"embargo of 5 .*max\(horizons\) \+ max_lag"):
            assign_folds(uf, spec)
    for embargo in (5, 7):
        fa = assign_folds(uf, make_forecast_spec(embargo=embargo, horizons=(1, 3), max_lag=2))
        assert fa.embargo_steps == embargo and fa.counts["dropped"] == 2 * embargo
    # without a forecast block an explicit embargo has no minimum
    assert assign_folds(uf, make_spec("temporal", time_column="t", embargo=0)).embargo_steps == 0


def test_temporal_cuts_at_time_boundaries_near_fractions() -> None:
    t = np.repeat(np.arange(500), 2)
    uf = make_frame(t.size, time=t.tolist())
    fa = assign_folds(uf, make_spec("temporal", time_column="t", embargo=0))
    assert fa.counts == {"dev": 600, "calib": 200, "sealed": 200, "dropped": 0}


def test_temporal_drops_calib_and_sealed_duplicates_of_dev() -> None:
    t = np.arange(300)
    dup_of = {250: 3, 200: 10, 280: 220}  # 280 (sealed) copies calib unit 220: kept
    uf = make_frame(t.size, time=t.tolist(), dup_of=dup_of)
    fa = assign_folds(uf, make_spec("temporal", time_column="t", embargo=0))
    assert fa.folds[250] == "dropped" and fa.drop_reasons[250] == "duplicate_of_dev"
    assert fa.folds[200] == "dropped" and fa.drop_reasons[200] == "duplicate_of_dev"
    assert fa.folds[3] == "dev" and fa.folds[10] == "dev"
    assert fa.folds[280] == "sealed"
    assert fa.dropped_reasons == {"duplicate_of_dev": 2}


def test_temporal_with_timestamps() -> None:
    t = pd.date_range("2022-01-01", periods=240, freq="h")
    uf = make_frame(t.size, time=list(t))
    fa = assign_folds(uf, make_spec("temporal", time_column="t", embargo=2))
    times = np.asarray(t)
    assert times[fa.mask(Fold.DEV)].max() < times[fa.mask(Fold.CALIB)].min()
    assert fa.counts["dropped"] == 4


def test_temporal_refusals() -> None:
    t = list(range(100))
    with pytest.raises(SplitError, match="not supported in A0"):
        uf = make_frame(100, time=t, groups=[i % 7 for i in range(100)])
        assign_folds(uf, make_spec("auto", time_column="t", group_columns=["g"]), "temporal")
    with pytest.raises(SplitError, match="time column"):
        assign_folds(make_frame(100), make_spec("auto"), "temporal")
    with pytest.raises(SplitError, match="embargo"):
        assign_folds(make_frame(100, time=t), make_spec("temporal", time_column="t", embargo=20))
    with pytest.raises(SplitError, match="nulls"):
        uf = make_frame(100, time=[None, *t[1:]])
        assign_folds(uf, make_spec("temporal", time_column="t"))
    uf = make_frame(100, time=[float(x) for x in t])
    nan_time = pa.array([float("nan"), *map(float, t[1:])], from_pandas=False)
    table = uf.table.set_column(uf.table.column_names.index("t"), "t", nan_time)
    with pytest.raises(SplitError, match="NaN"):
        assign_folds(UnitFrame(table, uf.roles), make_spec("temporal", time_column="t"))
    with pytest.raises(SplitError, match="NaN"):
        oof_folds(UnitFrame(table, uf.roles), 3, "temporal", seed=0)


@pytest.mark.parametrize(
    "values",
    [
        # dd.mm.yyyy sorts lexicographically by day, not by date
        [f"{d:02d}.{m:02d}.2023" for m in range(1, 11) for d in range(1, 11)],
        [f"t{i}".encode() for i in range(100)],
        [bool(i % 2) for i in range(100)],
    ],
    ids=["dd.mm.yyyy-strings", "binary", "boolean"],
)
def test_temporal_refuses_time_values_without_a_temporal_order(values: list[object]) -> None:
    uf = make_frame(len(values), time=values)
    spec = make_spec("temporal", time_column="t", embargo=0)
    with pytest.raises(SplitError, match="declare it as a timestamp"):
        assign_folds(uf, spec)
    with pytest.raises(SplitError, match="declare it as a timestamp"):
        oof_folds(uf, 3, "temporal", seed=0)


@pytest.mark.parametrize(
    "values",
    [
        list(range(100)),
        [float(i) / 3 for i in range(100)],
        list(pd.date_range("2023-01-01", periods=100, freq="D")),
        list(pd.date_range("2023-01-01", periods=100, freq="D").date),
    ],
    ids=["int", "float", "timestamp", "date"],
)
def test_temporal_accepts_numeric_timestamp_and_date_time_values(values: list[object]) -> None:
    uf = make_frame(len(values), time=values)
    fa = assign_folds(uf, make_spec("temporal", time_column="t", embargo=0))
    order = np.asarray(uf.time)
    assert order[fa.mask(Fold.DEV)].max() < order[fa.mask(Fold.CALIB)].min()


def test_temporal_is_row_order_invariant() -> None:
    t = np.repeat(np.arange(120), 2)
    uf = make_frame(t.size, time=t.tolist())
    spec = make_spec("temporal", time_column="t", embargo=1)
    perm = np.random.default_rng(1).permutation(uf.n)
    shuffled = uf.take(perm)
    a, b = assign_folds(uf, spec), assign_folds(shuffled, spec)
    assert fold_of_id(uf, a.folds) == fold_of_id(shuffled, b.folds)


# helpers ------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 7, 10, 101, 1000])
def test_largest_remainder(n: int) -> None:
    rng = np.random.default_rng(n)
    for fr in [(0.6, 0.2, 0.2), (0.5, 0.25, 0.25), (1 / 3, 1 / 3, 1 / 3), (0.9, 0.05, 0.05)]:
        sizes = largest_remainder(n, fr, rng)
        assert sizes.sum() == n
        assert np.all(np.abs(sizes - n * np.asarray(fr)) < 1)
    with pytest.raises(ValueError):
        largest_remainder(5, (0.0, 0.0, 0.0))


def test_largest_remainder_breaks_ties_with_rng() -> None:
    rng = np.random.default_rng(0)
    winners = [int(np.argmax(largest_remainder(1, (1 / 3, 1 / 3, 1 / 3), rng))) for _ in range(90)]
    assert set(winners) == {0, 1, 2}


def test_iid_singleton_strata_do_not_all_land_in_dev() -> None:
    """Per-stratum largest remainder alone sent every singleton stratum to dev."""
    n = 100
    uf = make_frame(n, classes=[f"class-{i:03d}" for i in range(n)])
    labels = np.asarray([f"class-{i:03d}" for i in range(n)], dtype=object)
    table = uf.table.set_column(uf.table.column_names.index("y"), "y", [labels.tolist()])
    fa = assign_folds(UnitFrame(table, uf.roles), make_spec(seed=4))
    assert {f: fa.counts[f] for f in ("dev", "calib", "sealed")} == {
        "dev": 60,
        "calib": 20,
        "sealed": 20,
    }


@pytest.mark.parametrize("fractions", FRACTION_SETS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_allocate_strata_bounds(fractions: tuple[float, ...], seed: int) -> None:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 6, 400)
    alloc = allocate_strata(sizes, fractions, np.random.default_rng(seed))
    fr = np.asarray(fractions)
    assert alloc.shape == (sizes.size, 3) and alloc.dtype == np.int64
    assert np.array_equal(alloc.sum(axis=1), sizes)
    assert np.all(np.abs(alloc - sizes[:, None] * fr) < 1.0 + 1e-9)  # every stratum ±1
    totals = alloc.sum(axis=0)
    assert np.all(np.abs(totals - sizes.sum() * fr) < 1.0 + 1e-9)  # and the totals ±1
    cumulative = np.cumsum(alloc, axis=0)  # ... at every prefix of the strata
    assert np.all(np.abs(cumulative - np.cumsum(sizes)[:, None] * fr) < 1.0 + 1e-9)
    again = allocate_strata(sizes, fractions, np.random.default_rng(seed))
    assert np.array_equal(alloc, again)
    assert allocate_strata(np.zeros(0, dtype=np.int64), fractions, rng).shape == (0, 3)


def test_cut_by_cumulative() -> None:
    sizes = np.ones(10, dtype=np.int64)
    out = cut_by_cumulative(sizes, np.array([6, 2, 2]))
    assert out.tolist() == [0] * 6 + [1] * 2 + [2] * 2
    big = np.array([5, 1, 1, 3], dtype=np.int64)
    assert cut_by_cumulative(big, np.array([6, 2, 2])).tolist() == [0, 0, 1, 2]


def test_stratum_codes() -> None:
    cats = np.array(["b", "a", None, "b"], dtype=object)
    codes = stratum_codes(cats, TargetKind.CATEGORICAL)
    assert codes[0] == codes[3] and len({codes[0], codes[1], codes[2]}) == 3
    x = np.concatenate([np.arange(100.0), [np.nan]])
    bins = stratum_codes(x, TargetKind.NUMERIC)
    assert np.unique(bins).size == 11  # 10 quantile bins + missing
    assert np.all(np.bincount(bins)[:10] == 10)
    assert np.all(stratum_codes(cats, TargetKind.SPANS) == 0)
    assert np.all(stratum_codes(cats, None) == 0)


def test_grouped_keeps_every_group_column_whole() -> None:
    """With group_columns [a, b], neither an 'a' nor a 'b' value may straddle folds."""
    from tests.unit.split.helpers import make_spec as _spec

    n = 3000
    rng = np.random.default_rng(3)
    a = rng.integers(0, 600, n)
    b = a // 2  # b links pairs of a-groups
    df = pd.DataFrame(
        {
            "id": [f"u{i}" for i in range(n)],
            "x": rng.normal(size=n),
            "a": a,
            "b": b,
            "y": rng.integers(0, 2, n),
        }
    )
    uf = UnitFrame.from_pandas(df, Roles("id", "y", ("x",), group_columns=("a", "b")))
    spec = _spec("grouped", group_columns=["a", "b"])
    res = assign_folds(uf, spec, "grouped")
    folds = np.asarray(res.folds)
    for col in ("a", "b"):
        per = pd.DataFrame({"v": df[col], "f": folds}).groupby("v")["f"].nunique()
        assert int(per.max()) == 1, col
