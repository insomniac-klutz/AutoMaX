"""T1-real-cheap: resplits of the calibration fold only (OQ Q3, D18).

Descriptive check on real data. The frozen artifact is run once on the calibration inputs; each
of ``resplits`` regime-respecting halvings certifies on one half (C') and checks the other (T'):
a band "violates" when the one-sided 95% LOWER bound of T' risk at the released τ exceeds α.
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
from amx.cert.bounds import risk_lower
from amx.cert.budget import DeltaBudget
from amx.cert.guarantee import guarantee_type_for
from amx.cert.ltt import fixed_sequence_ltt, grid_stats
from amx.loss.registry import build_loss
from amx.spec.enums import Regime
from amx.split.vault import LocalVault
from amx.warden.common import frozen_run
from amx.warden.runner import run_resolver
from amx.warden.token import verify_token

log = get_logger(__name__)

OUT_RELPATH = Path("sim") / "t1_real_cheap.json"
PROXY_FALSE_ALARM = 0.05  # one-sided 95% lower bound
TEMPORAL_BLOCKS = 20


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
    loss = build_loss(spec, confirmed=confirm_loss)

    calib = v.read_fold(rid, "calib")
    out = run_resolver(
        run.artifact_dir,
        calib.without_target(),
        v.run_path(rid) / "tmp",
        expected_hash=run.artifact_hash,
    )
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
    cert_count = np.zeros(m)
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
            cert_count[j] += 1
            com = t_mask & (scores <= tau[b.index])
            n_t = int(com.sum())
            cov_sum[j] += n_t / max(int(t_mask.sum()), 1)
            if n_t == 0:
                continue
            lo = risk_lower(float(losses[com].sum()), n_t, binary=loss.is_binary)
            if lo > b.alpha:
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
                "certified_share": float(cert_count[j] / resplits),
                "proxy_violation_rate": rate,
                "gross_failure_limit": limit,
                "gross_failure": gross_j,
                "mean_test_coverage_when_certified": (
                    float(cov_sum[j] / cert_count[j]) if cert_count[j] else None
                ),
            }
        )
    any_certified = bool(np.any(cert_count / resplits >= 0.10))
    payload: dict[str, Any] = {
        "kind": "t1_real_cheap",
        "run_id": rid,
        "guarantee_type": guarantee_type_for(regime, spec.task.family).value,
        "regime": regime.value,
        "calib_n": int(calib.n),
        "resplits": resplits,
        "seed": seed,
        "delta_per_band": dj,
        "estimand": "group_weighted" if group_col else "unit_weighted",
        "family_wise_proxy_rate": float(any_viol / resplits),
        "bands": bands,
        "gross_failure": gross,
        "at_least_one_band_certified": any_certified,
        "passed": (not gross) and any_certified,
        "note": "descriptive; calib-only resplits; sealed never read (Q3)",
    }
    path = run.run_dir / OUT_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("T1-real-cheap for %s: passed=%s", rid, payload["passed"])
    return payload
