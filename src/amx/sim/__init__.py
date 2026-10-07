"""Synthetic generators with known risk and the T1 simulators (HANDOFF 12, ROLLER step 8).

* :mod:`amx.sim.generators`: generators (a), (b), (d) for T1-synth and (c) for T1-forecast;
* :mod:`amx.sim.oracle`: Rao–Blackwell population curves and the C6 indeterminacy rule;
* :mod:`amx.sim.t1_synth`: the T1-synth runner (raw, monotonised and family-wise rates);
* :mod:`amx.sim.t1_forecast`: the T1-forecast runner (stationary gate and shifted report, D20);
* :mod:`amx.sim.tables`: Markdown and JSON rendering.
"""

from amx.sim.generators import (
    ClusteredUnits,
    ForecastStream,
    GaussianMixture,
    HeteroscedasticRegression,
    SeasonalARShift,
    SelectiveGenerator,
    SimBatch,
    selective_generator,
)
from amx.sim.oracle import OracleCurve
from amx.sim.t1_forecast import (
    T1ForecastGate,
    T1ForecastHorizon,
    T1ForecastResult,
    run_t1_forecast,
    run_t1_forecast_gate,
)
from amx.sim.t1_synth import T1Band, T1Cell, run_t1_synth
from amx.sim.tables import render_t1_json, render_t1_markdown, t1_payload

__all__ = [
    "ClusteredUnits",
    "ForecastStream",
    "GaussianMixture",
    "HeteroscedasticRegression",
    "OracleCurve",
    "SeasonalARShift",
    "SelectiveGenerator",
    "SimBatch",
    "T1Band",
    "T1Cell",
    "T1ForecastGate",
    "T1ForecastHorizon",
    "T1ForecastResult",
    "render_t1_json",
    "render_t1_markdown",
    "run_t1_forecast",
    "run_t1_forecast_gate",
    "run_t1_synth",
    "selective_generator",
    "t1_payload",
]
