"""Builtin losses (HANDOFF section 3, decision D17, OQ Q8)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from amx.loss.base import FloatArray, Loss, LossError


def _as_numeric(x: Any, what: str) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if np.isnan(arr).any():
        raise LossError(f"{what} contains NaN")
    return arr


def _as_objects(x: Any) -> list[Any]:
    return list(np.asarray(x, dtype=object).reshape(-1))


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _comparable(v: Any) -> Any:
    """Hashable, order-preserving form of a label (sequences become tuples)."""
    if isinstance(v, np.ndarray):
        return tuple(_comparable(x) for x in v.tolist())
    if isinstance(v, list | tuple):
        return tuple(_comparable(x) for x in v)
    return v


def zero_one() -> Loss:
    """1 if the committed label differs from gold, else 0 (sequence labels compare as tuples)."""

    def fn(pred: Any, gold: Any) -> FloatArray:
        p, g = _as_objects(pred), _as_objects(gold)
        if any(_is_missing(v) for v in g):
            raise LossError("gold contains missing values")
        return np.fromiter(
            (float(_comparable(a) != _comparable(b)) for a, b in zip(p, g, strict=True)),
            np.float64,
            len(g),
        )

    return Loss("zero_one", fn, is_binary=True)


def err_gt_tol(tol: float, relative: bool = False, abs_floor: float = 0.0) -> Loss:
    """1 if |pred - gold| exceeds the tolerance, else 0.

    Relative mode compares against ``tol * max(|gold|, abs_floor)``. With ``abs_floor = 0`` and
    gold exactly 0, any nonzero error counts as a loss (finite and documented, C8).
    """
    if not (tol > 0 and np.isfinite(tol)):
        raise LossError("err_gt_tol: tol must be a positive finite number")
    if abs_floor < 0:
        raise LossError("err_gt_tol: abs_floor must be >= 0")

    def fn(pred: Any, gold: Any) -> FloatArray:
        p, g = _as_numeric(pred, "pred"), _as_numeric(gold, "gold")
        err = np.abs(p - g)
        bound = tol * np.maximum(np.abs(g), abs_floor) if relative else np.full_like(g, tol)
        return (err > bound).astype(np.float64)

    return Loss(
        "err_gt_tol",
        fn,
        is_binary=True,
        params={"tol": tol, "relative": relative, "abs_floor": abs_floor},
    )


def _tokens(v: Any) -> list[str]:
    if isinstance(v, str):
        return v.split()
    if isinstance(v, Sequence | np.ndarray):
        return [str(t) for t in v]
    return str(v).split()


def _f1(pred: Any, gold: Any) -> float:
    pm, gm = _is_missing(pred), _is_missing(gold)
    if pm and gm:
        return 1.0
    if pm or gm:
        return 0.0
    pt, gt = _tokens(pred), _tokens(gold)
    if not pt and not gt:
        return 1.0
    common = sum((Counter(pt) & Counter(gt)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pt), common / len(gt)
    return 2 * precision * recall / (precision + recall)


def _exact_mismatch(pred: Any, gold: Any) -> float:
    pm, gm = _is_missing(pred), _is_missing(gold)
    if pm or gm:
        return 0.0 if pm and gm else 1.0
    return 0.0 if _tokens(pred) == _tokens(gold) else 1.0


def one_minus_f1(exact: bool = False) -> Loss:
    """1 − token F1 (bag of tokens). ``exact`` scores exact match only (binary).

    A missing value (None) means "field absent": both absent scores 0, one absent scores 1
    (OQ Q21).
    """

    def fn(pred: Any, gold: Any) -> FloatArray:
        p, g = _as_objects(pred), _as_objects(gold)
        if exact:
            return np.fromiter(
                (_exact_mismatch(a, b) for a, b in zip(p, g, strict=True)), np.float64, len(g)
            )
        return np.fromiter((1.0 - _f1(a, b) for a, b in zip(p, g, strict=True)), np.float64, len(g))

    return Loss("one_minus_f1", fn, is_binary=exact, params={"exact": exact})


def missed_anomaly(normal_label: Any) -> Loss:
    """One-sided: 1 if the unit was committed as normal but gold is an anomaly.

    Committing a unit as anomalous is outside this loss's design (such units are escalated,
    not committed); it scores 0 here, and the commit rule must not commit them.
    """

    def fn(pred: Any, gold: Any) -> FloatArray:
        p, g = _as_objects(pred), _as_objects(gold)
        if any(_is_missing(v) for v in g):
            raise LossError("gold contains missing values")
        return np.fromiter(
            (float(a == normal_label and b != normal_label) for a, b in zip(p, g, strict=True)),
            np.float64,
            len(g),
        )

    return Loss("missed_anomaly", fn, is_binary=True, params={"normal_label": normal_label})


def anomaly_cost(normal_label: Any, cost_miss: float = 1.0, cost_false_alarm: float = 1.0) -> Loss:
    """Two-sided: a missed anomaly costs ``cost_miss``, a false alarm ``cost_false_alarm``.

    Costs are normalised by their maximum so ℓ ∈ [0, 1]; the loss is binary when they are equal.
    """
    if cost_miss < 0 or cost_false_alarm < 0 or max(cost_miss, cost_false_alarm) == 0:
        raise LossError("anomaly_cost: costs must be >= 0 and not both 0")
    top = max(cost_miss, cost_false_alarm)
    c_miss, c_fa = cost_miss / top, cost_false_alarm / top

    def fn(pred: Any, gold: Any) -> FloatArray:
        p, g = _as_objects(pred), _as_objects(gold)
        if any(_is_missing(v) for v in g):
            raise LossError("gold contains missing values")
        out = np.zeros(len(g), dtype=np.float64)
        for i, (a, b) in enumerate(zip(p, g, strict=True)):
            pred_normal, gold_normal = a == normal_label, b == normal_label
            if pred_normal and not gold_normal:
                out[i] = c_miss
            elif not pred_normal and gold_normal:
                out[i] = c_fa
        return out

    return Loss(
        "anomaly_cost",
        fn,
        is_binary=c_miss in (0.0, 1.0) and c_fa in (0.0, 1.0),
        params={
            "normal_label": normal_label,
            "cost_miss": cost_miss,
            "cost_false_alarm": cost_false_alarm,
        },
    )
