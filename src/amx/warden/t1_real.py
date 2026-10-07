"""T1-real-cheap: resplits of the calibration fold only (OQ Q3, D18, D24).

Descriptive check on real data, run AFTER the run's certify call (the calib budget must be
closed: Q1 forbids calib-derived files before that). The frozen artifact is run once on the
calibration inputs; each of ``resplits`` regime-respecting halvings certifies on one half (C')
and checks the other (T'): a band "violates" when the one-sided 95% LOWER bound of the T'
risk at the released τ exceeds α. In group mode both halves use the group-weighted estimand.
That proxy fires up to ~5% of the time even for a perfect certifier when R(τ̂) ≈ α, and the
resplits share data, so the result gates only on gross failure. Sealed is never read.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from amx._log import get_logger
from amx.cert.bounds import hb_lower, risk_lower
from amx.cert.budget import DeltaBudget
from amx.cert.guarantee import claims_certification
from amx.cert.ltt import fixed_sequence_ltt, grid_stats
from amx.spec.enums import Regime
from amx.split.counter import CertifyCounter
from amx.split.vault import LocalVault
from amx.warden.common import WardenError, frozen_run, load_calib, warden_loss
from amx.warden.runner import run_resolver
from amx.warden.token import verify_token

log = get_logger(__name__)

OUT_RELPATH = Path("sim") / "t1_real_cheap.json"
PROXY_FALSE_ALARM = 0.05  # one-sided 95% lower bound
TEMPORAL_BLOCKS = 20
RELEASED_SHARE_MIN = 0.10


def _halves(
    n: int,
    regime: Regime,
    rng: np.random.Generator,
    groups: NDArray[Any] | None,
    time: NDArray[Any] | None,
) -> NDArray[np.bool_]:
    """Boolean mask of C' (the certifying half), respecting the regime."""
    if regime is Regime.TEMPORAL and time is not None:
        order = np.argsort(time, kind="stable")
        blocks = np.array_split(order, TEMPORAL_BLOCKS)
        pick = rng.permutation(TEMPORAL_BLOCKS)[: TEMPORAL_BLOCKS // 2]
        mask = np.zeros(n, dtype=bool)
        for b in pick:
            mask[blocks[b]] = True
        return mask
    if groups is not None:
        uniq, inv = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)
        chosen = rng.permutation(uniq.size)[: uniq.size // 2]
        return np.isin(inv, chosen)
    mask = np.zeros(n, dtype=bool)
    mask[rng.permutation(n)[: n // 2]] = True
    return mask


def _test_lower(
    losses: NDArray[np.float64],
    mask: NDArray[np.bool_],
    groups: NDArray[Any] | None,
    binary: bool,
) -> tuple[float, int]:
    """One-sided 95% lower bound of the T' risk in the certified estimand, and its n."""
    if groups is None:
        n = int(mask.sum())
        return (risk_lower(float(losses[mask].sum()), n, binary=binary) if n else 0.0), n
    g = np.asarray(groups, dtype=object).astype(str)[mask]
    _, inv = np.unique(g, return_inverse=True)
    if inv.size == 0:
        return 0.0, 0
    cnt = np.bincount(inv)
    means = np.bincount(inv, weights=losses[mask]) / cnt
    return hb_lower(float(means.sum()), int(means.size)), int(means.size)


def t1_real_cheap(
    run_dir: str | Path,
    *,
    token: str | None,
    resplits: int = 200,
    seed: int = 20261007,
    vault: LocalVault | None = None,
    run_id: str | None = None,
    confirm_loss: bool = False,
) -> dict[str, Any]:
    v = vault or LocalVault()
    rid = run_id or Path(run_dir).resolve().name
    verify_token(v, rid, token)
    run = frozen_run(run_dir, v, rid)
    spec = run.spec
    if CertifyCounter(v, rid, spec.splits.max_certify_calls).remaining() > 0:
        raise WardenError(
            "T1-real-cheap reads calibration labels; run it only after the run's certify call "
            "has closed the calibration budget (OQ Q1, Q3)"
        )
    loss, _ = warden_loss(v, run, confirm_loss=confirm_loss)
    calib = load_calib(v, run)
    out = run_resolver(run.artifact_dir, calib.without_target(), expected_hash=run.artifact_hash)
    scores = out["score"].to_numpy(dtype=np.float64)
    losses = loss(out["value"].to_numpy(), calib.target)
    group_col = spec.data.independence_group
    groups = calib.column(group_col) if group_col else None
    regime = run.manifest.regime
    budget = DeltaBudget(spec.bands.delta, spec.bands.m, spec.cert.call_policy)
    dj = budget.delta_per_band
    alphas = list(spec.bands.alphas)
    tau = run.grid.values
    rng = np.random.default_rng(seed)

    m = len(alphas)
    viol = np.zeros(m)
    released = np.zeros(m)
    cov_sum = np.zeros(m)
    any_viol = 0
    for _ in range(resplits):
        c_mask = _halves(calib.n, regime, rng, groups, calib.time)
        t_mask = ~c_mask
        g_c = None if groups is None else groups[c_mask]
        stats = grid_stats(scores[c_mask], losses[c_mask], tau, binary=loss.is_binary, groups=g_c)
        res = fixed_sequence_ltt(
            stats, alphas, dj, run.dev_cov, start_factor=spec.cert.start_factor
        )
        bad = False
        for j, b in enumerate(res.bands):
            if b.index is None:
                continue
            released[j] += 1
            com = t_mask & (scores <= tau[b.index])
            cov_sum[j] += int(com.sum()) / max(int(t_mask.sum()), 1)
            lo, n_t = _test_lower(losses, com, groups, loss.is_binary)
            if n_t and lo > b.alpha:
                viol[j] += 1
                bad = True
        any_viol += bad

    bands = []
    gross = False
    for j, a in enumerate(alphas):
        rate = float(viol[j] / resplits)
        limit = PROXY_FALSE_ALARM + dj + 3 * math.sqrt(dj * (1 - dj) / resplits)
        gross_j = bool(rate > limit)
        gross = gross or gross_j
        bands.append(
            {
                "alpha": float(a),
                "released_share": float(released[j] / resplits),
                "proxy_violation_rate": rate,
                "gross_failure_limit": limit,
                "gross_failure": gross_j,
                "mean_test_coverage_when_released": (
                    float(cov_sum[j] / released[j]) if released[j] else None
                ),
            }
        )
    some_band = bool(np.any(released / resplits >= RELEASED_SHARE_MIN))
    gtype = run.guarantee_type
    payload: dict[str, Any] = {
        "kind": "t1_real_cheap",
        "run_id": rid,
        "guarantee_type": gtype.value,
        "claims_certification": claims_certification(gtype),
        "regime": regime.value,
        "calib_n": int(calib.n),
        "resplits": resplits,
        "seed": seed,
        "delta_per_band": dj,
        "estimand": "group_weighted" if group_col else "unit_weighted",
        "family_wise_proxy_rate": float(any_viol / resplits),
        "bands": bands,
        "gross_failure": gross,
        "at_least_one_band_released": some_band,
        "passed": (not gross) and some_band,
        "note": "descriptive; calib-only resplits after the certify call; sealed never read (Q3)",
    }
    path = run.run_dir / OUT_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("T1-real-cheap for %s: passed=%s", rid, payload["passed"])
    return payload
