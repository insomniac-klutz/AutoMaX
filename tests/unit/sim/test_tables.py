"""Markdown and JSON rendering of T1 results."""

from __future__ import annotations

import json

import pytest

from amx.sim import (
    T1Cell,
    T1ForecastGate,
    render_t1_json,
    render_t1_markdown,
    run_t1_forecast_gate,
    run_t1_synth,
    t1_payload,
)
from amx.sim.tables import render_t1_forecast_markdown, render_t1_synth_markdown


@pytest.fixture(scope="module")
def cells() -> list[T1Cell]:
    return [run_t1_synth("b", n, 10, n_mc=100_000) for n in (500, 2000)]


@pytest.fixture(scope="module")
def forecast() -> T1ForecastGate:
    return run_t1_forecast_gate(n_eval=600, n_calib=600, n_dev=300, allow_short=True)


def test_json_is_strict_and_complete(cells: list[T1Cell], forecast: T1ForecastGate) -> None:
    text = render_t1_json(cells, forecast)
    data = json.loads(text)  # strict: NaN would have raised in json.dumps(allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert data["schema"] == "amx.t1/1"
    synth = data["t1_synth"]
    assert synth["passed"] == all(c.gate_pass for c in cells)
    for c, cd in zip(cells, synth["cells"], strict=True):
        assert cd["gate_pass"] == c.gate_pass and cd["rates_within_bound"] == c.rates_within_bound
    assert len(synth["cells"]) == 2
    band = synth["cells"][0]["bands"][0]
    for key in (
        "raw_rate",
        "mono_rate",
        "bound",
        "certified_share",
        "tightness",
        "label",
        "unit_risk_at_certified",
        "unit_risk_at_oracle_cov",
    ):
        assert key in band
    # unit-weighted cells have no separate unit-weighted risk
    assert band["unit_risk_at_certified"] is None and band["unit_risk_at_oracle_cov"] is None
    assert band["label"] == "vacuous" and band["tightness"] is None
    fc = data["t1_forecast"]
    assert fc["passed"] == forecast.passed == forecast.stationary.passed
    assert fc["stationary"]["gated"] is True and fc["shifted"]["gated"] is False
    for path in ("stationary", "shifted"):
        p = fc[path]
        assert p["path"] == path
        assert p["target_label"] == "empirical_target"
        assert p["interval_guarantee"] == "long_run_frequency"
        assert {h["selective"]["guarantee_type"] for h in p["horizons"]} == {"holdout_empirical"}
        for h in p["horizons"]:
            for key in (
                "bound_over",
                "bound_under",
                "implied_by_bound",
                "infinite_share",
                "empty_share",
                "degenerate_share",
                "local_coverage_min",
                "local_coverage_max",
            ):
                assert key in h, key
    assert fc["stationary"]["shift_index"] is None and fc["shifted"]["shift_index"] is not None
    assert t1_payload(cells)["t1_synth"]["cells"][1]["n_calib"] == 2000
    assert "t1_forecast" not in t1_payload(cells)


def test_markdown_tables(cells: list[T1Cell], forecast: T1ForecastGate) -> None:
    md = render_t1_markdown(cells, forecast)
    assert md.startswith("## T1 results")
    synth = render_t1_synth_markdown(cells)
    band_rows = [ln for ln in synth.splitlines() if ln.startswith("| heteroscedastic |")]
    assert len(band_rows) == 2 * 4 + 2  # 4 bands per cell, plus 2 family-wise rows
    assert "raw rate" in synth and "mono rate" in synth and "FW rate" in synth
    assert "vacuous" in synth
    fmd = render_t1_forecast_markdown(forecast)
    assert "EMPIRICAL target" in fmd and "holdout_empirical" in fmd
    assert "infinite share" in fmd and "local coverage range" in fmd
    assert "bound over" in fmd and "bound under" in fmd and "implied by bound" in fmd
    # both paths, the stationary one gated and the shifted one reported only (D20)
    assert "Stationary path (gated)" in fmd and "Shifted path (reported, not gated)" in fmd
    assert fmd.index("Stationary path") < fmd.index("Shifted path")
    for res in forecast.paths:
        for h in res.horizons:
            assert f"| {h.horizon} | {h.aci.steps} | {h.miscoverage:.4f} |" in fmd
    assert f"seed {forecast.stationary.seed}" in fmd and f"seed {forecast.shifted.seed}" in fmd
    # every table row has the same number of cells as its header
    for block in md.split("\n\n"):
        rows = [ln for ln in block.splitlines() if ln.startswith("|")]
        if rows:
            assert len({ln.count("|") for ln in rows}) == 1, block
