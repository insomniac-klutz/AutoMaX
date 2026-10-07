"""Smoke runs of T1-synth (reps = 40) and T1-forecast, plus the labelling logic (C2, C6, Q6)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from amx.cert import TauGrid, aci_run, aci_side_bounds
from amx.cert.guarantee import GuaranteeType
from amx.sim import (
    OracleCurve,
    SeasonalARShift,
    T1Cell,
    T1ForecastHorizon,
    T1ForecastResult,
    run_t1_forecast,
    run_t1_forecast_gate,
    run_t1_synth,
    selective_generator,
)
from amx.sim.t1_forecast import (
    T1_FORECAST_MAX_DEGENERATE,
    T1_FORECAST_SEED,
    T1_FORECAST_STATIONARY_SEED,
    implied_by_bound,
)
from amx.sim.t1_synth import (
    ORACLE_SEEDS,
    T1_ALPHAS,
    T1_SEEDS,
    cell_seed,
    slack_bound,
)
from amx.sim.tables import render_t1_synth_markdown, t1_payload

TAU = TauGrid().values
SMOKE_N_MC = {"a": 300_000, "b": 300_000, "d": 40_000}
LABELS = {"pass", "fail", "vacuous", "indeterminate"}


@pytest.fixture(scope="module")
def oracles() -> dict[str, OracleCurve]:
    out: dict[str, OracleCurve] = {}
    for key, n_mc in SMOKE_N_MC.items():
        gen = selective_generator(key)
        out[key] = gen.oracle(TAU, n_mc, np.random.default_rng(ORACLE_SEEDS[gen.name]))
    return out


def _check_cell(cell: T1Cell, reps: int) -> None:
    assert cell.reps == reps and len(cell.bands) == len(T1_ALPHAS)
    assert cell.delta_j == pytest.approx(0.1 / 4)
    assert cell.fw_bound == pytest.approx(slack_bound(0.1, reps))
    assert 0.0 <= cell.fw_rate <= 1.0
    assert cell.label in {"pass", "fail", "vacuous", "indeterminate"}
    for b in cell.bands:
        assert b.bound == pytest.approx(0.025 + 3 * math.sqrt(0.025 * 0.975 / reps))
        assert 0.0 <= b.raw_rate <= b.certified_share <= 1.0
        # monotonisation only moves τ̂ liberal-ward: at least as many reps certified
        assert b.mono_certified >= b.raw_certified
        assert b.label in LABELS
        assert b.passed == (b.raw_rate <= b.bound)
        if b.mono_certified == 0 and b.passed:
            assert b.label == "vacuous"
    assert cell.rates_within_bound == (
        all(b.passed for b in cell.bands) and cell.fw_rate <= cell.fw_bound
    )
    # the gate verdict: rates within bound AND the cell label is 'pass' (not indeterminate,
    # not vacuous in every band); vacuous bands inside a passing cell do not fail it
    assert cell.gate_pass == (cell.rates_within_bound and cell.label == "pass")
    d = cell.to_dict()
    assert d["label"] == cell.label and len(d["bands"]) == 4
    assert d["gate_pass"] == cell.gate_pass
    assert d["rates_within_bound"] == cell.rates_within_bound


@pytest.mark.parametrize("key", ["a", "b", "d"])
def test_t1_synth_smoke(key: str, oracles: dict[str, OracleCurve]) -> None:
    cell = run_t1_synth(key, 2000, 40, oracle=oracles[key])
    _check_cell(cell, 40)
    gen = selective_generator(key)
    assert cell.seed == cell_seed(gen.name, 2000)
    assert cell.estimand == ("group_weighted" if key == "d" else "unit_weighted")
    assert cell.rates_within_bound and cell.gate_pass
    # the loosest band certifies in some reps for every generator at n_calib = 2000
    assert cell.bands[-1].mono_certified > 0
    # tightness is certified coverage over oracle coverage at the same risk
    b = cell.bands[-1]
    assert b.tightness is not None and 0.0 < b.tightness <= 1.0 + 1e-9
    assert b.oracle_coverage == pytest.approx(oracles[key].coverage_at(0.05))


def test_t1_synth_is_deterministic_and_independent_of_n_jobs(
    oracles: dict[str, OracleCurve],
) -> None:
    c1 = run_t1_synth("b", 2000, 24, oracle=oracles["b"])
    c2 = run_t1_synth("b", 2000, 24, oracle=oracles["b"])
    c3 = run_t1_synth("b", 2000, 24, oracle=oracles["b"], n_jobs=2)
    strip = {"runtime_s"}
    d1, d2, d3 = ({k: v for k, v in c.to_dict().items() if k not in strip} for c in (c1, c2, c3))
    assert d1 == d2 == d3
    c4 = run_t1_synth("b", 2000, 24, oracle=oracles["b"], seed=7)
    assert c4.seed == 7


def test_pre_registered_seeds_are_distinct() -> None:
    seeds = [cell_seed(g, n) for g in T1_SEEDS for n in (500, 2000, 10_000)]
    assert len(set(seeds)) == len(seeds)
    assert len(set(ORACLE_SEEDS.values())) == len(ORACLE_SEEDS)


def test_violations_are_detected(oracles: dict[str, OracleCurve]) -> None:
    """An oracle whose risk sits above every band turns every certification into a violation."""
    o = oracles["b"]
    bad = dataclasses.replace(o, risk=o.risk + 0.5)
    cell = run_t1_synth("b", 10_000, 40, oracle=bad)
    loose = cell.bands[-1]
    assert loose.raw_violations == loose.raw_certified > 0
    assert not loose.passed and loose.label == "fail"
    assert not cell.rates_within_bound and cell.label == "fail" and not cell.gate_pass
    assert cell.fw_violations == max(b.mono_certified for b in cell.bands)


def test_indeterminate_selections_are_excluded_and_flagged(
    oracles: dict[str, OracleCurve],
) -> None:
    o = oracles["b"]
    vague = dataclasses.replace(o, risk_se=np.full_like(o.risk_se, 1.0))
    cell = run_t1_synth("b", 10_000, 40, oracle=vague)
    loose = cell.bands[-1]
    assert loose.raw_violations == 0
    assert loose.raw_indeterminate == loose.raw_certified > 0
    assert loose.label == "indeterminate" and cell.label == "indeterminate"
    assert cell.rates_within_bound  # indeterminate is not counted as a violation ...
    assert not cell.gate_pass  # ... and not counted as a pass either (Gate A0)
    assert cell.fw_violations == 0 and cell.fw_indeterminate > 0
    payload = t1_payload([cell])["t1_synth"]
    assert payload["passed"] is False and payload["cells"][0]["gate_pass"] is False
    fw_row = [
        ln
        for ln in render_t1_synth_markdown([cell]).splitlines()
        if ln.startswith("| heteroscedastic | 10000 |")
    ]
    assert len(fw_row) == 1 and "| NO | indeterminate |" in fw_row[0]


def test_group_cells_report_unit_weighted_risk(oracles: dict[str, OracleCurve]) -> None:
    """HANDOFF 7.5 'report both': group-weighted cells carry the oracle UNIT-weighted risk."""
    o = oracles["d"]
    assert o.unit_risk is not None
    cell = run_t1_synth("d", 2000, 40, oracle=o)
    assert cell.estimand == "group_weighted"
    for b in cell.bands:
        idx = o.coverage_index(b.alpha)
        if idx is None:
            assert b.unit_risk_at_oracle_cov is None
        else:
            assert b.unit_risk_at_oracle_cov == pytest.approx(float(o.unit_risk[idx]))
        if b.mono_certified == 0:
            assert b.unit_risk_at_certified is None
        else:
            assert b.unit_risk_at_certified is not None
            assert 0.0 <= b.unit_risk_at_certified <= 1.0
    loose = cell.bands[-1]
    assert loose.unit_risk_at_certified is not None and loose.unit_risk_at_oracle_cov is not None
    # the mean is over certified reps: a constant unit-risk curve returns that constant
    flat = dataclasses.replace(o, unit_risk=np.full_like(o.unit_risk, 0.123))
    fc = run_t1_synth("d", 2000, 40, oracle=flat)
    assert fc.bands[-1].unit_risk_at_certified == pytest.approx(0.123)
    assert fc.bands[-1].unit_risk_at_oracle_cov == pytest.approx(0.123)
    # rendered in the per-band table and in the JSON
    d = cell.to_dict()["bands"][-1]
    assert d["unit_risk_at_certified"] == loose.unit_risk_at_certified
    assert d["unit_risk_at_oracle_cov"] == loose.unit_risk_at_oracle_cov
    md = render_t1_synth_markdown([cell])
    assert "unit risk @ τ̂' (mean)" in md and "unit risk @ oracle cov" in md
    row = [ln for ln in md.splitlines() if ln.startswith("| clustered | group_weighted | 2000 |")]
    assert (
        f"| {loose.unit_risk_at_certified:.4f} | {loose.unit_risk_at_oracle_cov:.4f} |" in row[-1]
    )


def test_unit_weighted_cells_have_no_separate_unit_risk(oracles: dict[str, OracleCurve]) -> None:
    cell = run_t1_synth("b", 2000, 10, oracle=oracles["b"])
    assert all(
        b.unit_risk_at_certified is None and b.unit_risk_at_oracle_cov is None for b in cell.bands
    )


def test_vacuous_band_is_labelled() -> None:
    cell = run_t1_synth("b", 500, 10, n_mc=100_000)
    tight = cell.bands[0]  # α = 0.5%: below min r(x), infeasible by design
    assert tight.mono_certified == 0 and tight.label == "vacuous"
    assert tight.tightness is None  # oracle coverage at α is 0


def test_vacuous_cell_is_not_a_gate_pass(oracles: dict[str, OracleCurve]) -> None:
    """A cell whose every band is vacuous certifies nothing: not a pass (Gate A0)."""
    cell = run_t1_synth("b", 2000, 10, alphas=(0.005,), oracle=oracles["b"])
    assert cell.bands[0].label == "vacuous" and cell.label == "vacuous"
    assert cell.rates_within_bound and not cell.gate_pass
    # a vacuous band inside a cell whose other bands certify does not fail the cell
    mixed = run_t1_synth("b", 2000, 10, oracle=oracles["b"])
    assert mixed.bands[0].label == "vacuous" and mixed.label == "pass" and mixed.gate_pass


def test_t1_synth_rejects_bad_args(oracles: dict[str, OracleCurve]) -> None:
    with pytest.raises(ValueError):
        run_t1_synth("b", 500, 0, oracle=oracles["b"])
    with pytest.raises(ValueError):
        run_t1_synth("b", 500, 5, grid=TAU[:50], oracle=oracles["b"])
    with pytest.raises(ValueError):
        run_t1_synth("c", 500, 5)


# --- T1-forecast -------------------------------------------------------------------------------


def _check_forecast_path(res: T1ForecastResult) -> None:
    assert res.n_eval >= 2000
    assert res.target_label == "empirical_target"
    assert res.interval_guarantee is GuaranteeType.LONG_RUN_FREQUENCY
    assert [h.horizon for h in res.horizons] == [1, 24]
    assert res.max_degenerate == T1_FORECAST_MAX_DEGENERATE == 0.01
    for h in res.horizons:
        a = h.aci
        assert a.steps == res.n_eval and a.gamma == 0.005 and a.alpha_target == 0.1
        assert h.abs_deviation == pytest.approx(abs(a.miscoverage - 0.1))
        assert h.within_tolerance == (h.abs_deviation <= 0.02)
        # D20: the implied one-sided bounds and the flag that says the check could not fail
        over, under = aci_side_bounds(0.1, 0.005, a.steps, h.horizon)
        assert (h.bound_over, h.bound_under) == pytest.approx((over, under))
        side = over if h.miscoverage >= 0.1 else under
        assert h.implied_by_bound == (side <= 0.02)
        assert h.degenerate_share == pytest.approx(a.infinite_share + a.empty_share)
        assert h.non_degenerate == (h.degenerate_share <= 0.01)
        assert h.passed == (h.within_tolerance and h.non_degenerate)
        assert 0.0 <= a.infinite_share <= 1.0 and 0.0 <= a.empty_share <= 1.0
        lo, hi = a.local_coverage_range
        assert 0.0 <= lo <= hi <= 1.0
        assert math.isfinite(a.median_finite_width)
        s = h.selective
        assert s.guarantee_type is GuaranteeType.HOLDOUT_EMPIRICAL
        if s.n_committed_eval:
            assert s.risk_lo <= s.risk_est <= s.risk_hi
    d = res.to_dict()
    assert d["target_label"] == "empirical_target" and d["path"] == res.path
    assert d["horizons"][0]["selective"]["guarantee_type"] == "holdout_empirical"
    for key in ("bound_over", "bound_under", "implied_by_bound", "degenerate_share"):
        assert key in d["horizons"][0]


def test_t1_forecast_smoke() -> None:
    """D20: the stationary path is the gate; the shifted path is reported, not gated."""
    gate = run_t1_forecast_gate()
    st, sh = gate.stationary, gate.shifted
    _check_forecast_path(st)
    _check_forecast_path(sh)
    assert st.path == "stationary" and st.gated and st.shift_index is None
    assert st.seed == T1_FORECAST_STATIONARY_SEED
    assert st.generator == SeasonalARShift().stationary().name
    assert sh.path == "shifted" and not sh.gated and sh.shift_index is not None
    assert sh.seed == T1_FORECAST_SEED
    assert gate.passed == st.passed
    assert st.passed  # the gate holds on the pre-registered stationary seed
    # the shift pushes miscoverage up, the side whose bound already forces ±0.02 (D20)
    for h in sh.horizons:
        assert h.miscoverage > 0.1 and h.implied_by_bound and h.within_tolerance
    for h in st.horizons:
        assert math.isnan(h.selective.risk_post_shift)
    d = gate.to_dict()
    assert d["passed"] == gate.passed
    assert d["stationary"]["gated"] is True and d["shifted"]["gated"] is False


def test_run_t1_forecast_defaults_to_the_shifted_path() -> None:
    res = run_t1_forecast()
    assert res.path == "shifted" and res.seed == T1_FORECAST_SEED
    stat = run_t1_forecast(SeasonalARShift().stationary())
    assert stat.path == "stationary" and stat.seed == T1_FORECAST_STATIONARY_SEED


def test_implied_by_bound_picks_the_side_of_the_deviation() -> None:
    assert implied_by_bound(0.01, over=0.007, under=0.06, tolerance=0.02)
    assert not implied_by_bound(-0.01, over=0.007, under=0.06, tolerance=0.02)
    assert implied_by_bound(-0.01, over=0.06, under=0.007, tolerance=0.02)
    assert not implied_by_bound(0.01, over=0.06, under=0.007, tolerance=0.02)
    # a zero deviation lies on both sides: forced if either side is
    assert implied_by_bound(0.0, over=0.06, under=0.007, tolerance=0.02)
    assert not implied_by_bound(0.0, over=0.06, under=0.06, tolerance=0.02)


def test_degenerate_intervals_fail_the_forecast_criterion() -> None:
    """ACI holds long-run miscoverage near α even when almost every interval is infinite.

    Every residual exceeds every calibration score, so α_t is driven to the infinite-interval
    region and oscillates there. The ±0.02 check passes (forced by the over-side bound); only
    the infinite + empty share (D20) exposes the degenerate intervals.
    """
    T = 3000
    aci = aci_run(np.linspace(0.0, 1.0, 50), np.zeros(T), np.full(T, 10.0), 0.1, 0.005, 1)
    short = run_t1_forecast(n_eval=400, n_calib=400, n_dev=300, allow_short=True, horizons=(1,))
    hz = T1ForecastHorizon.from_aci(
        aci, short.horizons[0].selective, tolerance=0.02, max_degenerate=0.01
    )
    assert hz.within_tolerance and hz.implied_by_bound
    assert hz.degenerate_share > 0.5
    assert not hz.non_degenerate and not hz.passed


def test_t1_forecast_labels_are_never_upgraded() -> None:
    res = run_t1_forecast(n_eval=400, n_calib=400, n_dev=300, allow_short=True, horizons=(1,))
    sel = res.horizons[0].selective
    kwargs = {f: getattr(sel, f) for f in sel.__dataclass_fields__}
    with pytest.raises(TypeError):
        type(sel)(**kwargs, guarantee_type=GuaranteeType.PAC_HIGH_PROB)  # type: ignore[call-arg]
    with pytest.raises(AttributeError):
        sel.guarantee_type = GuaranteeType.PAC_HIGH_PROB  # type: ignore[misc]
    with pytest.raises(AttributeError):
        res.target_label = "certified"  # type: ignore[misc]
    assert sel.to_dict()["guarantee_type"] == "holdout_empirical"


def test_t1_forecast_requires_2000_steps() -> None:
    with pytest.raises(ValueError):
        run_t1_forecast(n_eval=500)
    short = run_t1_forecast(n_eval=500, n_calib=500, n_dev=300, allow_short=True, horizons=(1,))
    assert short.horizons[0].aci.steps == 500
