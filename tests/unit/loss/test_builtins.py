from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from amx.loss import (
    LossConfirmationError,
    LossError,
    anomaly_cost,
    build_loss,
    check_loss,
    err_gt_tol,
    load_custom_loss,
    loss_distribution,
    missed_anomaly,
    needs_confirmation,
    one_minus_f1,
    zero_one,
)
from amx.loss.base import Loss
from amx.spec import parse_taskspec


def test_zero_one() -> None:
    out = zero_one()(["a", "b", 1], ["a", "c", 1])
    assert out.tolist() == [0.0, 1.0, 0.0]


def test_zero_one_rejects_missing_gold() -> None:
    with pytest.raises(LossError, match="missing"):
        zero_one()(["a"], [None])


def test_err_gt_tol_absolute_and_relative() -> None:
    assert err_gt_tol(0.5)([1.0, 1.0], [1.4, 1.6]).tolist() == [0.0, 1.0]
    rel = err_gt_tol(0.1, relative=True)
    assert rel([110.0, 89.0], [100.0, 100.0]).tolist() == [0.0, 1.0]


def test_err_gt_tol_relative_at_zero_is_finite() -> None:
    rel = err_gt_tol(0.1, relative=True)
    assert rel([0.0, 0.01], [0.0, 0.0]).tolist() == [0.0, 1.0]
    floored = err_gt_tol(0.1, relative=True, abs_floor=1.0)
    assert floored([0.05], [0.0]).tolist() == [0.0]


def test_err_gt_tol_rejects_nan_and_bad_tol() -> None:
    with pytest.raises(LossError, match="NaN"):
        err_gt_tol(1.0)([1.0], [np.nan])
    with pytest.raises(LossError):
        err_gt_tol(0.0)


def test_one_minus_f1_token_overlap_and_absent() -> None:
    loss = one_minus_f1()
    out = loss(["a b c", None, None, "x"], ["a b d", None, "y", "x"])
    assert out[0] == pytest.approx(1 - 2 / 3)
    assert out[1:].tolist() == [0.0, 1.0, 0.0]
    assert not loss.is_binary
    exact = one_minus_f1(exact=True)
    assert exact.is_binary
    assert exact(["a b", "a b", None], ["a b", "a c", None]).tolist() == [0.0, 1.0, 0.0]


def test_missed_anomaly_one_sided() -> None:
    loss = missed_anomaly(normal_label=0)
    # committed normal + gold anomaly -> 1; everything else 0
    assert loss([0, 0, 1, 1], [0, 1, 0, 1]).tolist() == [0.0, 1.0, 0.0, 0.0]


def test_anomaly_cost_two_sided_normalised() -> None:
    loss = anomaly_cost(normal_label=0, cost_miss=1.0, cost_false_alarm=0.25)
    assert loss([0, 0, 1, 1], [0, 1, 0, 1]).tolist() == [0.0, 1.0, 0.25, 0.0]
    assert not loss.is_binary
    assert anomaly_cost(normal_label=0).is_binary


def test_length_mismatch_and_range_enforced() -> None:
    with pytest.raises(LossError, match="units"):
        zero_one()(["a"], ["a", "b"])
    bad = Loss("bad", lambda p, g: np.full(len(g), 1.5), is_binary=False)
    with pytest.raises(LossError, match=r"\[0, 1\]"):
        bad([0], [0])


def test_check_loss_flags_identity_and_binarity() -> None:
    assert check_loss(zero_one(), ["a", "b", "c"]).ok
    shifted = Loss("shifted", lambda p, g: np.full(len(g), 0.2), is_binary=True)
    res = check_loss(shifted, [1, 2, 3])
    assert not res.ok
    assert any("ℓ(y, y)" in e for e in res.errors)
    assert any("binary" in e for e in res.errors)


def test_loss_distribution_warnings() -> None:
    frac = loss_distribution(np.r_[np.zeros(97), np.ones(2), [0.5]], is_binary=False)
    assert any("mass" in w for w in frac.warnings)
    const = loss_distribution(np.zeros(200), is_binary=True)
    assert any("nearly constant" in w for w in const.warnings)
    fine = loss_distribution(np.r_[np.zeros(80), np.ones(20)], is_binary=True)
    assert fine.warnings == []
    assert sum(fine.histogram) == 100


SPEC = {
    "spec_version": 1,
    "name": "t",
    "data": {
        "uri": "x",
        "format": "csv",
        "unit_id": "id",
        "target": {"name": "y", "kind": "numeric"},
    },
    "task": {"family": "regression", "loss": {"params": {"tol": 1.0, "relative": False}}},
    "bands": {"alphas": [0.05], "policies": ["auto"]},
}


def test_build_loss_default_needs_no_confirmation() -> None:
    spec = parse_taskspec(SPEC)
    assert not needs_confirmation(spec)
    assert build_loss(spec).name == "err_gt_tol"


def test_build_loss_bad_params() -> None:
    raw = {**SPEC, "task": {"family": "regression", "loss": {"params": {"tol": 1, "nope": 2}}}}
    with pytest.raises(LossError, match="bad params"):
        build_loss(parse_taskspec(raw))


def test_custom_loss_requires_confirmation(tmp_path: Path) -> None:
    f = tmp_path / "my_loss.py"
    f.write_text(
        "import numpy as np\n"
        "def loss(p, g):\n    return np.abs(np.asarray(p) - np.asarray(g))\n"
        "loss.is_binary = False\n"
    )
    raw = {
        **SPEC,
        "task": {"family": "regression", "loss": {"kind": "custom", "path": str(f), "fn": "loss"}},
    }
    spec = parse_taskspec(raw)
    assert needs_confirmation(spec)
    with pytest.raises(LossConfirmationError):
        build_loss(spec)
    loss = build_loss(spec, confirmed=True)
    assert loss([0.5], [0.0]).tolist() == [0.5]
    squashed = load_custom_loss(f, "loss", confirmed=True, params={"squash": 1.0})
    assert 0 < squashed([10.0], [0.0])[0] < 1
