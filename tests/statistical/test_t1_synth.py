"""T1-synth and T1-forecast as amended for Gate A0 (HANDOFF 12 T1, ROLLER C2, C6, Q2, Q6).

Grid: generators (a), (b), (d) × n_calib {500, 2000, 10000} × bands α {0.5%, 1%, 2%, 5%},
δ = 0.1 split by DeltaBudget (policy single, δ_j = 0.025), 2000 reps per cell (all three
generators; the reduced 1000 allowed for (d) was not needed). Seeds and oracle sizes are the
pre-registered constants of :mod:`amx.sim.t1_synth`.

Pass criterion per cell (``T1Cell.gate_pass``): every RAW per-band violation rate ≤
δ_j + 3·sqrt(δ_j(1 − δ_j)/R), the family-wise rate after monotonisation ≤
δ + 3·sqrt(δ(1 − δ)/R), AND the cell label is ``pass``. Vacuous bands are labelled and do not
fail a cell whose other bands certify; an indeterminate or all-vacuous cell is not a pass. At
n_calib = 10000 every generator must certify at least one band in at least 10% of reps.

Generator (c) is gated on T1-forecast only, as amended by D20: on its STATIONARY variant
(seed fixed in code before the first run), every horizon must have |ACI miscoverage − 0.1| ≤
0.02 over at least 2000 steps (an EMPIRICAL target, Q6) AND an infinite + empty interval
share ≤ 0.01. The shifted path is reported (miscoverage, infinite and empty shares, local
coverage range, implied one-sided bounds and ``implied_by_bound``), never gated: the shift
raises miscoverage, the side whose deterministic ACI bound already forces ±0.02.

Set ``AMX_T1_REPORT_DIR`` to also write ``t1.md`` and ``t1.json`` there.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from amx.cert import TauGrid
from amx.sim import (
    OracleCurve,
    T1Cell,
    T1ForecastGate,
    render_t1_json,
    render_t1_markdown,
    run_t1_forecast_gate,
    run_t1_synth,
    selective_generator,
)
from amx.sim.t1_forecast import (
    T1_FORECAST_MAX_DEGENERATE,
    T1_FORECAST_SEED,
    T1_FORECAST_STATIONARY_SEED,
)
from amx.sim.t1_synth import (
    ORACLE_SEEDS,
    T1_ALPHAS,
    T1_DELTA,
    T1_N_CALIB,
    T1_N_MC,
    T1_REPS,
)
from amx.sim.tables import render_t1_forecast_markdown, render_t1_synth_markdown

GENERATORS = ("a", "b", "d")
N_JOBS = max(1, min(4, os.cpu_count() or 1))
MIN_CERTIFIED_SHARE = 0.10
"""At n_calib = 10000 some band must be certified (released) in at least this share of reps."""

_log = logging.getLogger(__name__)


class _T1Cache:
    def __init__(self) -> None:
        self.tau = TauGrid().values
        self.oracles: dict[str, OracleCurve] = {}
        self.cells: dict[tuple[str, int], T1Cell] = {}
        self._forecast: T1ForecastGate | None = None

    def oracle(self, key: str) -> OracleCurve:
        if key not in self.oracles:
            gen = selective_generator(key)
            rng = np.random.default_rng(ORACLE_SEEDS[gen.name])
            self.oracles[key] = gen.oracle(self.tau, T1_N_MC[gen.name], rng)
        return self.oracles[key]

    def cell(self, key: str, n_calib: int) -> T1Cell:
        if (key, n_calib) not in self.cells:
            self.cells[(key, n_calib)] = run_t1_synth(
                key,
                n_calib,
                T1_REPS,
                T1_ALPHAS,
                T1_DELTA,
                oracle=self.oracle(key),
                n_jobs=N_JOBS,
            )
        return self.cells[(key, n_calib)]

    def forecast(self) -> T1ForecastGate:
        """Both T1-forecast paths, run once on first use (no dependence on test order)."""
        if self._forecast is None:
            self._forecast = run_t1_forecast_gate()
        return self._forecast


@pytest.fixture(scope="module")
def t1() -> _T1Cache:
    return _T1Cache()


@pytest.mark.slow
@pytest.mark.parametrize("n_calib", T1_N_CALIB)
@pytest.mark.parametrize("key", GENERATORS)
def test_t1_synth_cell(t1: _T1Cache, key: str, n_calib: int) -> None:
    cell = t1.cell(key, n_calib)
    summary = [
        (b.alpha, b.label, b.raw_rate, round(b.bound, 4), b.mono_rate, b.certified_share)
        for b in cell.bands
    ]
    assert cell.reps == T1_REPS
    for b in cell.bands:
        assert b.raw_rate <= b.bound, (key, n_calib, summary)
        if b.mono_certified == 0:
            assert b.label == "vacuous"
    assert cell.fw_rate <= cell.fw_bound, (key, n_calib, cell.fw_rate, summary)
    assert cell.rates_within_bound
    assert cell.gate_pass, "\n" + render_t1_synth_markdown([cell])
    if key == "b":
        # min r(x) ≈ 0.0064 > 0.005: the tightest band is infeasible and must read vacuous
        assert cell.bands[0].label == "vacuous"


@pytest.mark.slow
@pytest.mark.parametrize("key", GENERATORS)
def test_some_band_certifies_at_10000(t1: _T1Cache, key: str) -> None:
    cell = t1.cell(key, 10_000)
    best = max(b.mono_certified_share for b in cell.bands)
    assert best >= MIN_CERTIFIED_SHARE, [(b.alpha, b.mono_certified_share) for b in cell.bands]
    assert cell.certified_bands


@pytest.mark.slow
def test_t1_forecast_gate(t1: _T1Cache) -> None:
    """D20 gate: stationary path, ±0.02 AND infinite + empty share ≤ 0.01 per horizon."""
    gate = t1.forecast()
    st = gate.stationary
    report = render_t1_forecast_markdown(gate)
    assert st.path == "stationary" and st.gated and st.shift_index is None
    assert st.seed == T1_FORECAST_STATIONARY_SEED
    assert st.n_eval >= 2000 and st.target_label == "empirical_target"
    assert st.max_degenerate == T1_FORECAST_MAX_DEGENERATE == 0.01
    for h in st.horizons:
        assert h.aci.steps >= 2000
        assert abs(h.miscoverage - st.alpha_target) <= st.tolerance, report
        assert h.aci.infinite_share + h.aci.empty_share <= T1_FORECAST_MAX_DEGENERATE, report
        assert h.selective.guarantee_type.value == "holdout_empirical"
        assert h.passed, report
    assert gate.passed, report


@pytest.mark.slow
def test_t1_forecast_shifted_path_is_reported(t1: _T1Cache) -> None:
    """The shifted path is reported with its implied bounds; it never decides the gate."""
    gate = t1.forecast()
    sh = gate.shifted
    assert sh.path == "shifted" and not sh.gated and sh.shift_index is not None
    assert sh.seed == T1_FORECAST_SEED
    assert gate.passed == gate.stationary.passed
    for h in sh.horizons:
        assert h.aci.steps >= 2000
        assert 0.0 <= h.aci.infinite_share <= 1.0 and 0.0 <= h.aci.empty_share <= 1.0
        lo, hi = h.aci.local_coverage_range
        assert 0.0 <= lo <= hi <= 1.0
        assert 0.0 < h.bound_over < h.bound_under
        assert h.selective.guarantee_type.value == "holdout_empirical"


@pytest.mark.slow
def test_t1_report(t1: _T1Cache) -> None:
    """Render the full table (all cells, both forecast paths) and optionally write it to disk."""
    cells = [t1.cell(k, n) for k in GENERATORS for n in T1_N_CALIB]
    forecast = t1.forecast()
    md = render_t1_markdown(cells, forecast)
    js = render_t1_json(cells, forecast)
    print("\n" + md)
    out = os.environ.get("AMX_T1_REPORT_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "t1.md").write_text(md, encoding="utf-8")
        (Path(out) / "t1.json").write_text(js, encoding="utf-8")
    runtime = sum(c.runtime_s for c in cells)
    _log.info("T1-synth runtime %.1f s over %d cells", runtime, len(cells))
    # the forecast section with both paths and their infinite-interval shares
    assert "### T1-forecast" in md
    assert "Stationary path (gated)" in md and "Shifted path (reported, not gated)" in md
    assert "infinite share" in md
    fc = json.loads(js)["t1_forecast"]
    assert fc["stationary"]["path"] == "stationary" and fc["shifted"]["path"] == "shifted"
    for path in ("stationary", "shifted"):
        assert len(fc[path]["horizons"]) == len(forecast.stationary.horizons)
        assert all("infinite_share" in h for h in fc[path]["horizons"])
    assert fc["passed"] == forecast.passed
    assert all(c.gate_pass for c in cells), md
