"""Per-slice selective risk with honest bounds (HANDOFF 7.10, C7). Never a guarantee.

Slice labels are computed so that no raw calibration value reaches the certificate (HANDOFF
8.4: only aggregates leave the warden):

* a numeric column with more than ``MAX_LEVELS`` distinct values is cut into calibration
  deciles labelled ``q01`` ... ``q10`` (no bin edges are emitted);
* any other column keeps its values, but values held by fewer than ``MIN_CELL`` calibration
  units, and every value beyond the ``MAX_LEVELS - 1`` most frequent, are pooled into
  ``(other)``;
* a slice value with no committed unit is not reported.

In group mode (``independence_unit: group:<col>``) unit-level bounds would assume independent
units, so slice numbers are descriptive only (no bound, no 2α flag).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amx.cert.bounds import bound_name, risk_lower, risk_upper
from amx.cert.nmin import n_min

SliceFlag = Literal["insufficient_n", "upper_gt_2alpha", "lower_gt_2alpha"]

MIN_CELL = 30
MAX_LEVELS = 20
OTHER = "(other)"
N_QUANTILES = 10


@dataclass(frozen=True)
class SliceRisk:
    slice: str
    value: str
    n: int
    risk: float | None
    upper95: float | None
    bound: str
    flag: SliceFlag | None


def _is_numeric(values: NDArray[Any]) -> bool:
    if values.dtype.kind in "iuf":
        return True
    try:
        np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return all(not isinstance(v, str) for v in values[:100])


def slice_labels(
    values: ArrayLike, *, max_levels: int = MAX_LEVELS, min_cell: int = MIN_CELL
) -> NDArray[np.str_]:
    """Aggregate-only slice labels for one column of the calibration fold."""
    v = np.asarray(values, dtype=object).reshape(-1)
    if v.size == 0:
        return np.asarray([], dtype=str)
    if _is_numeric(v):
        num = np.asarray(v, dtype=np.float64)
        finite = np.isfinite(num)
        if len(np.unique(num[finite])) > max_levels:
            ranks = np.full(num.shape, -1, dtype=np.int64)
            order = np.argsort(num[finite], kind="stable")
            dec = np.empty(order.size, dtype=np.int64)
            dec[order] = (np.arange(order.size) * N_QUANTILES) // max(order.size, 1)
            ranks[finite] = dec
            return np.asarray(
                [f"q{r + 1:02d}" if r >= 0 else OTHER for r in ranks.tolist()], dtype=str
            )
    lab = np.asarray([OTHER if x is None else str(x) for x in v.tolist()], dtype=object)
    uniq, counts = np.unique(lab.astype(str), return_counts=True)
    order = np.argsort(-counts, kind="stable")
    keep = {str(uniq[i]) for i in order[: max_levels - 1] if counts[i] >= min_cell}
    return np.asarray([x if x in keep else OTHER for x in lab.astype(str)], dtype=str)


def slice_risks(
    name: str,
    labels: ArrayLike,
    losses: ArrayLike,
    committed: ArrayLike,
    *,
    alpha: float,
    binary: bool,
    descriptive: bool = False,
) -> list[SliceRisk]:
    """Empirical selective risk per slice label among committed units.

    ``labels`` should come from :func:`slice_labels`. Slices with fewer than
    n_min(2α, 0.05) committed units are labelled ``insufficient_n``, escalated to
    ``lower_gt_2alpha`` when even their one-sided 95% lower bound exceeds 2α; otherwise a 95%
    upper bound above 2α is flagged ``upper_gt_2alpha`` (6.5).
    """
    lab = np.asarray(labels, dtype=object).astype(str).reshape(-1)
    loss = np.asarray(losses, dtype=np.float64).reshape(-1)
    com = np.asarray(committed, dtype=bool).reshape(-1)
    small = n_min(min(2 * alpha, 0.99), 0.05)
    bname = "descriptive" if descriptive else bound_name(binary)
    out: list[SliceRisk] = []
    for value in sorted(set(lab.tolist())):
        sel = com & (lab == value)
        n = int(sel.sum())
        if n == 0:
            continue
        total = float(loss[sel].sum())
        if descriptive:
            out.append(SliceRisk(name, value, n, total / n, None, bname, None))
            continue
        upper = risk_upper(total, n, binary=binary)
        flag: SliceFlag | None
        if n < small:
            lower = risk_lower(total, n, binary=binary)
            flag = "lower_gt_2alpha" if lower > 2 * alpha else "insufficient_n"
        elif upper > 2 * alpha:
            flag = "upper_gt_2alpha"
        else:
            flag = None
        out.append(SliceRisk(name, value, n, total / n, upper, bname, flag))
    return out
