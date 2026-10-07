"""T1-synth: does the certificate hold on generators with known risk? (HANDOFF 12, C2, C6).

For one generator and one calibration size the runner draws ``reps`` independent calibration
sets, certifies each with :func:`amx.cert.grid_stats` and :func:`amx.cert.fixed_sequence_ltt`
under ``DeltaBudget(delta, m)`` (policy ``single``, δ_j = δ / m), and judges every selected τ̂
against the oracle curve of the generator:

* **raw per-band rate** (gated, C2): share of reps whose raw τ̂_j (the band's own walk,
  before cross-band monotonisation) has R(τ̂_j) > α_j;
* **monotonised per-band rate** (reported, not gated): the same for the released τ̂'_j;
  after monotonisation a band is only bounded by Σ_{k≤j} δ_k;
* **family-wise rate** (gated): share of reps where ANY band is violated after monotonisation;
* **indeterminate** (C6): a τ̂ with |R(τ̂) − α_j| < 4 SE(oracle) cannot be judged; it is excluded
  from the violation count and reported, and a band with more than 1% such reps is labelled
  ``indeterminate``;
* **vacuous**: a band whose released certificate is never certified;
* **tightness**: mean over reps of the oracle coverage at the released τ̂'_j (0 when
  uncertified) divided by the oracle coverage at α_j (max coverage over grid points with
  R ≤ α_j).
* **unit-weighted risk** (group-weighted cells only, HANDOFF 7.5 "report both"): the mean
  over certified reps of the oracle UNIT-weighted risk at the released τ̂'_j, and the oracle
  unit-weighted risk at the oracle-coverage point of α_j. Reported, never gated: the
  certificate covers the group-weighted estimand only (OQ Q2).

Rates within bound (:attr:`T1Cell.rates_within_bound`) iff every raw rate ≤
δ_j + 3·sqrt(δ_j(1 − δ_j)/reps) and the family-wise rate ≤ δ + 3·sqrt(δ(1 − δ)/reps). The
gate verdict (:attr:`T1Cell.gate_pass`) also needs the cell label ``pass``: vacuous and
indeterminate cells are reported, never counted as passes (Gate A0 as amended). A vacuous
band inside a cell whose other bands certify is labelled and does not fail the cell.

Seeds are pre-registered below (decision log); each rep draws from its own child of the
cell's :class:`numpy.random.SeedSequence`, so results do not depend on ``n_jobs``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from amx._log import get_logger
from amx.cert import DeltaBudget, TauGrid, fixed_sequence_ltt, grid_stats
from amx.sim.generators import SelectiveGenerator, selective_generator
from amx.sim.oracle import INDETERMINATE_K, OracleCurve

FloatArray = NDArray[np.float64]
BandLabel = Literal["pass", "fail", "vacuous", "indeterminate"]

T1_SEEDS: dict[str, int] = {
    "gaussian_mixture": 120_001,
    "heteroscedastic": 120_002,
    "clustered": 120_004,
}
"""Pre-registered base seeds of the T1-synth cells, per generator (offset by n_calib)."""

ORACLE_SEEDS: dict[str, int] = {
    "gaussian_mixture": 130_001,
    "heteroscedastic": 130_002,
    "clustered": 130_004,
}
"""Pre-registered seeds of the T1 oracle runs, per generator."""

T1_N_MC: dict[str, int] = {
    "gaussian_mixture": 20_000_000,
    "heteroscedastic": 20_000_000,
    "clustered": 2_000_000,
}
"""Oracle Monte Carlo sizes for the gate (inputs; groups for the clustered generator)."""

T1_ALPHAS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
T1_N_CALIB: tuple[int, ...] = (500, 2000, 10_000)
T1_DELTA = 0.1
T1_REPS = 2000
INDETERMINATE_SHARE = 0.01
"""A band with more than this share of indeterminate reps is labelled ``indeterminate``."""

_log = get_logger(__name__)


def slack_bound(level: float, reps: int) -> float:
    """level + 3·sqrt(level(1 − level)/reps): the T1 pass threshold for a violation rate."""
    return level + 3.0 * math.sqrt(level * (1.0 - level) / reps)


def cell_seed(generator: str, n_calib: int) -> int:
    """Pre-registered seed of one T1 cell."""
    return T1_SEEDS[generator] * 100_003 + n_calib


@dataclass(frozen=True)
class T1Band:
    alpha: float
    delta_j: float
    bound: float
    raw_violations: int
    raw_rate: float
    raw_indeterminate: int
    raw_certified: int
    mono_violations: int
    mono_rate: float
    mono_indeterminate: int
    mono_certified: int
    certified_share: float
    mono_certified_share: float
    indeterminate_share: float
    oracle_coverage: float
    mean_certified_coverage: float
    tightness: float | None
    passed: bool
    label: BandLabel
    unit_risk_at_certified: float | None = None
    """Group-weighted cells: mean oracle unit-weighted risk at the released τ̂'_j over
    certified reps (None for unit-weighted cells or when never certified)."""
    unit_risk_at_oracle_cov: float | None = None
    """Group-weighted cells: oracle unit-weighted risk at the oracle-coverage point of α_j."""


@dataclass(frozen=True)
class T1Cell:
    generator: str
    estimand: str
    n_calib: int
    reps: int
    delta: float
    delta_j: float
    seed: int
    grid_size: int
    oracle_method: str
    oracle_n_mc: int
    bands: tuple[T1Band, ...]
    fw_violations: int
    fw_rate: float
    fw_bound: float
    fw_indeterminate: int
    rates_within_bound: bool
    runtime_s: float

    @property
    def label(self) -> str:
        """``fail`` | ``indeterminate`` | ``vacuous`` (all bands) | ``pass``."""
        if not self.rates_within_bound:
            return "fail"
        if any(b.label == "indeterminate" for b in self.bands):
            return "indeterminate"
        if all(b.label == "vacuous" for b in self.bands):
            return "vacuous"
        return "pass"

    @property
    def gate_pass(self) -> bool:
        """The Gate A0 verdict: rates within bound AND label ``pass``.

        An ``indeterminate`` cell (some band with > 1% of τ̂ within 4 oracle SE of α, C6) and a
        ``vacuous`` cell (no band ever certified) are not passes, even with every rate in bound.
        """
        return self.rates_within_bound and self.label == "pass"

    @property
    def certified_bands(self) -> tuple[float, ...]:
        """Bands certified (released) in at least one rep."""
        return tuple(b.alpha for b in self.bands if b.mono_certified > 0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label
        d["gate_pass"] = self.gate_pass
        return d


@dataclass(frozen=True)
class _RepOutcome:
    raw_index: tuple[int | None, ...]
    mono_index: tuple[int | None, ...]


def _one_rep(
    gen: SelectiveGenerator,
    n_calib: int,
    tau: FloatArray,
    alphas: FloatArray,
    delta_j: float,
    dev_cov: FloatArray,
    seed: np.random.SeedSequence,
) -> _RepOutcome:
    batch = gen.sample(n_calib, np.random.default_rng(seed))
    stats = grid_stats(
        batch.scores, batch.losses, tau, binary=batch.groups is None, groups=batch.groups
    )
    res = fixed_sequence_ltt(stats, alphas, delta_j, dev_cov)
    return _RepOutcome(
        raw_index=tuple(b.raw_index for b in res.bands),
        mono_index=tuple(b.index for b in res.bands),
    )


def _chunk(
    gen: SelectiveGenerator,
    n_calib: int,
    tau: FloatArray,
    alphas: FloatArray,
    delta_j: float,
    dev_cov: FloatArray,
    seeds: Sequence[np.random.SeedSequence],
) -> list[_RepOutcome]:
    return [_one_rep(gen, n_calib, tau, alphas, delta_j, dev_cov, s) for s in seeds]


def run_t1_synth(
    generator: SelectiveGenerator | str,
    n_calib: int,
    reps: int,
    alphas: ArrayLike = T1_ALPHAS,
    delta: float = T1_DELTA,
    grid: ArrayLike | None = None,
    seed: int | None = None,
    *,
    oracle: OracleCurve | None = None,
    n_mc: int | None = None,
    n_jobs: int = 1,
) -> T1Cell:
    """Run one T1-synth cell (one generator, one n_calib, all bands).

    ``generator`` is a :class:`SelectiveGenerator` or its letter ``"a"``, ``"b"`` or ``"d"``.
    ``seed`` overrides the pre-registered cell seed (:func:`cell_seed`). ``grid`` defaults to
    the 200-point log grid of 7.2. ``oracle`` may be passed in to reuse one oracle across
    cells; otherwise it is computed with the generator's pre-registered oracle seed and
    ``n_mc`` (default: the generator's ``default_n_mc``). ``dev_cov`` always comes from the
    oracle, never from a calibration draw. For the clustered generator ``n_calib`` counts
    groups. ``n_jobs`` > 1 runs reps in joblib workers; results do not depend on it.
    """
    t_start = time.perf_counter()
    if isinstance(generator, str):
        generator = selective_generator(generator)
    tau = TauGrid().values if grid is None else np.asarray(grid, dtype=np.float64).reshape(-1)
    a = np.asarray(alphas, dtype=np.float64).reshape(-1)
    if reps < 1 or n_calib < 1:
        raise ValueError("need reps >= 1 and n_calib >= 1")
    name = generator.name
    if oracle is None:
        o_seed = ORACLE_SEEDS.get(name, 0)
        oracle = generator.oracle(
            tau, n_mc or generator.default_n_mc, np.random.default_rng(o_seed)
        )
    else:
        oracle.check_grid(tau)
    dev_cov = generator.dev_cov(tau, oracle)
    budget = DeltaBudget(delta, int(a.shape[0]))
    dj = budget.delta_per_band
    base = (cell_seed(name, n_calib) if name in T1_SEEDS else n_calib) if seed is None else seed
    children = np.random.SeedSequence(base).spawn(reps)

    if n_jobs == 1:
        outcomes = _chunk(generator, n_calib, tau, a, dj, dev_cov, children)
    else:
        n_chunks = min(reps, 32)
        parts = [children[i::n_chunks] for i in range(n_chunks)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_chunk)(generator, n_calib, tau, a, dj, dev_cov, p) for p in parts
        )
        outcomes = [out for part in results for out in part]  # tallies are order-free

    m = int(a.shape[0])
    raw_v = np.zeros(m, np.int64)
    raw_i = np.zeros(m, np.int64)
    raw_c = np.zeros(m, np.int64)
    mono_v = np.zeros(m, np.int64)
    mono_i = np.zeros(m, np.int64)
    mono_c = np.zeros(m, np.int64)
    cov_sum = np.zeros(m, np.float64)
    unit_risk = oracle.unit_risk if oracle.estimand == "group_weighted" else None
    unit_sum = np.zeros(m, np.float64)
    unit_n = np.zeros(m, np.int64)
    fw_v = 0
    fw_i = 0
    for out in outcomes:
        any_v = False
        any_i = False
        for j in range(m):
            alpha = float(a[j])
            ri = out.raw_index[j]
            if ri is not None:
                raw_c[j] += 1
                if oracle.indeterminate(ri, alpha, INDETERMINATE_K):
                    raw_i[j] += 1
                elif oracle.violates(ri, alpha):
                    raw_v[j] += 1
            mi = out.mono_index[j]
            if mi is not None:
                mono_c[j] += 1
                cov_sum[j] += float(oracle.cov[mi])
                if unit_risk is not None and np.isfinite(unit_risk[mi]):
                    unit_sum[j] += float(unit_risk[mi])
                    unit_n[j] += 1
                if oracle.indeterminate(mi, alpha, INDETERMINATE_K):
                    mono_i[j] += 1
                    any_i = True
                elif oracle.violates(mi, alpha):
                    mono_v[j] += 1
                    any_v = True
        fw_v += int(any_v)
        fw_i += int(any_i and not any_v)

    bound_j = slack_bound(dj, reps)
    bands: list[T1Band] = []
    for j in range(m):
        alpha = float(a[j])
        oc = oracle.coverage_at(alpha)
        mean_cov = float(cov_sum[j] / reps)
        raw_rate = float(raw_v[j] / reps)
        passed = raw_rate <= bound_j
        ind_share = float(max(raw_i[j], mono_i[j]) / reps)
        label: BandLabel
        if not passed:
            label = "fail"
        elif mono_c[j] == 0:
            label = "vacuous"
        elif ind_share > INDETERMINATE_SHARE:
            label = "indeterminate"
        else:
            label = "pass"
        u_cert: float | None = None
        u_oracle: float | None = None
        if unit_risk is not None:
            if unit_n[j] > 0:
                u_cert = float(unit_sum[j] / unit_n[j])
            g_oc = oracle.coverage_index(alpha)
            if g_oc is not None and np.isfinite(unit_risk[g_oc]):
                u_oracle = float(unit_risk[g_oc])
        bands.append(
            T1Band(
                alpha=alpha,
                delta_j=dj,
                bound=bound_j,
                raw_violations=int(raw_v[j]),
                raw_rate=raw_rate,
                raw_indeterminate=int(raw_i[j]),
                raw_certified=int(raw_c[j]),
                mono_violations=int(mono_v[j]),
                mono_rate=float(mono_v[j] / reps),
                mono_indeterminate=int(mono_i[j]),
                mono_certified=int(mono_c[j]),
                certified_share=float(raw_c[j] / reps),
                mono_certified_share=float(mono_c[j] / reps),
                indeterminate_share=ind_share,
                oracle_coverage=oc,
                mean_certified_coverage=mean_cov,
                tightness=(mean_cov / oc) if oc > 0 else None,
                passed=passed,
                label=label,
                unit_risk_at_certified=u_cert,
                unit_risk_at_oracle_cov=u_oracle,
            )
        )
    fw_rate = fw_v / reps
    fw_bound = slack_bound(budget.simultaneous_level, reps)
    cell = T1Cell(
        generator=name,
        estimand=oracle.estimand,
        n_calib=n_calib,
        reps=reps,
        delta=delta,
        delta_j=dj,
        seed=base,
        grid_size=int(tau.shape[0]),
        oracle_method=oracle.method,
        oracle_n_mc=oracle.n_mc,
        bands=tuple(bands),
        fw_violations=fw_v,
        fw_rate=fw_rate,
        fw_bound=fw_bound,
        fw_indeterminate=fw_i,
        rates_within_bound=all(b.passed for b in bands) and fw_rate <= fw_bound,
        runtime_s=time.perf_counter() - t_start,
    )
    _log.info(
        "t1_synth cell",
        extra={
            "amx": {
                "generator": name,
                "n_calib": n_calib,
                "reps": reps,
                "rates_within_bound": cell.rates_within_bound,
                "gate_pass": cell.gate_pass,
                "label": cell.label,
                "fw_rate": fw_rate,
                "runtime_s": cell.runtime_s,
            }
        },
    )
    return cell
