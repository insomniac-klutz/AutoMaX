"""Per-slice selective risk with honest bounds (HANDOFF 7.10, C7). Never a guarantee."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from amx.cert.bounds import bound_name, risk_upper
from amx.cert.nmin import n_min

SliceFlag = Literal["insufficient_n", "upper_gt_2alpha"]


@dataclass(frozen=True)
class SliceRisk:
    slice: str
    value: str
    n: int
    risk: float | None
    upper95: float | None
    bound: str
    flag: SliceFlag | None


def slice_risks(
    name: str,
    labels: ArrayLike,
    losses: ArrayLike,
    committed: ArrayLike,
    *,
    alpha: float,
    binary: bool,
) -> list[SliceRisk]:
    """Empirical selective risk per slice value among committed units.

    Slices with fewer than n_min(2α, 0.05) committed units are labelled ``insufficient_n``
    instead of flagged; otherwise a one-sided 95% upper bound above 2α is flagged (6.5).
    """
    lab = np.asarray(labels, dtype=object).astype(str).reshape(-1)
    loss = np.asarray(losses, dtype=np.float64).reshape(-1)
    com = np.asarray(committed, dtype=bool).reshape(-1)
    small = n_min(min(2 * alpha, 0.99), 0.05)
    out: list[SliceRisk] = []
    for value in sorted(set(lab.tolist())):
        sel = com & (lab == value)
        n = int(sel.sum())
        if n == 0:
            out.append(SliceRisk(name, value, 0, None, None, bound_name(binary), "insufficient_n"))
            continue
        total = float(loss[sel].sum())
        upper = risk_upper(total, n, binary=binary)
        flag: SliceFlag | None
        if n < small:
            flag = "insufficient_n"
        elif upper > 2 * alpha:
            flag = "upper_gt_2alpha"
        else:
            flag = None
        out.append(SliceRisk(name, value, n, total / n, upper, bound_name(binary), flag))
    return out
