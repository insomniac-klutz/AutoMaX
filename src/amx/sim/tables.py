"""Markdown and JSON rendering of T1 results (HANDOFF 12 T1, ROLLER step 8, Gate A0).

The renderers only format what :class:`~amx.sim.t1_synth.T1Cell` and
:class:`~amx.sim.t1_forecast.T1ForecastGate` already hold; they compute nothing new. Every
number in the Markdown appears in the JSON. Labels follow the amended Gate A0 (C2, C6, Q6,
D20):

* T1-synth gates on the **raw** per-band rate (before monotonisation) and the
  **family-wise** rate; the monotonised per-band rate is reported, not gated;
* ``vacuous`` (the released certificate never certified the band) and ``indeterminate``
  (more than 1% of selected τ̂ within 4 oracle SE of α) are flagged, never counted as passes:
  the cell verdict is :attr:`~amx.sim.t1_synth.T1Cell.gate_pass` (rates within bound AND cell
  label ``pass``);
* T1-forecast's ±0.02 is an **empirical target**; the interval guarantee is
  ``long_run_frequency`` and the selective risk is ``holdout_empirical``. The stationary path
  is gated (±0.02 and infinite + empty share ≤ 0.01); the shifted path is reported with its
  one-sided bounds and ``implied_by_bound`` (D20).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from amx.sim.t1_forecast import T1ForecastGate, T1ForecastResult
from amx.sim.t1_synth import T1Cell

SCHEMA = "amx.t1/1"


def _json_safe(obj: Any) -> Any:
    """NaN and ±inf become None so the output is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    return obj


def _num(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def t1_payload(cells: Sequence[T1Cell], forecast: T1ForecastGate | None = None) -> dict[str, Any]:
    """JSON-ready payload: every cell with its bands, flags and the overall gate verdicts."""
    synth_pass = all(c.gate_pass for c in cells)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "t1_synth": {
            "gated": [
                "raw_rate per band",
                "family-wise rate",
                "cell label pass (not indeterminate, not vacuous)",
            ],
            "reported_only": ["mono_rate per band", "tightness", "certified_share"],
            "passed": synth_pass,
            "cells": [c.to_dict() for c in cells],
        },
    }
    if forecast is not None:
        payload["t1_forecast"] = forecast.to_dict()
    return _json_safe(payload)  # type: ignore[no-any-return]


def render_t1_json(
    cells: Sequence[T1Cell], forecast: T1ForecastGate | None = None, *, indent: int = 2
) -> str:
    """Strict JSON (no NaN) of :func:`t1_payload`."""
    return json.dumps(t1_payload(cells, forecast), indent=indent, allow_nan=False)


def _yes(flag: bool) -> str:
    return "yes" if flag else "NO"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return out


def render_t1_synth_markdown(cells: Sequence[T1Cell]) -> str:
    """Per-band table (raw, monotonised, flags, tightness) and per-cell family-wise table."""
    lines = [
        "### T1-synth: per band",
        "",
        "Gated: raw rate ≤ bound (before monotonisation, C2). Reported only: monotonised rate,",
        "certified share, tightness. Indeterminate τ̂ (|R − α| < 4 SE, C6) are excluded from",
        "the violation counts. The family-wise 'pass' is the cell verdict: rates within bound",
        "AND cell label 'pass' (an indeterminate or all-vacuous cell is not a pass).",
        "Group-weighted cells also report the oracle UNIT-weighted risk (HANDOFF 7.5): its mean",
        "at the released τ̂' over certified reps and its value at the oracle-coverage point.",
        "",
    ]
    band_rows: list[list[str]] = []
    for c in cells:
        for b in c.bands:
            band_rows.append(
                [
                    c.generator,
                    c.estimand,
                    str(c.n_calib),
                    str(c.reps),
                    f"{b.alpha:g}",
                    _num(b.delta_j),
                    _num(b.raw_rate),
                    _num(b.bound),
                    _num(b.mono_rate),
                    _num(b.certified_share, 3),
                    _num(b.mono_certified_share, 3),
                    f"{b.raw_indeterminate}/{b.mono_indeterminate}",
                    _num(b.oracle_coverage),
                    _num(b.tightness, 3),
                    _num(b.unit_risk_at_certified),
                    _num(b.unit_risk_at_oracle_cov),
                    b.label,
                ]
            )
    lines += _table(
        [
            "generator",
            "estimand",
            "n_calib",
            "reps",
            "α_j",
            "δ_j",
            "raw rate",
            "bound",
            "mono rate",
            "cert share (raw)",
            "cert share (released)",
            "indeterminate raw/mono",
            "oracle cov",
            "tightness",
            "unit risk @ τ̂' (mean)",
            "unit risk @ oracle cov",
            "label",
        ],
        band_rows,
    )
    lines += ["", "### T1-synth: family-wise", ""]
    fw_rows = [
        [
            c.generator,
            str(c.n_calib),
            str(c.reps),
            _num(c.delta),
            _num(c.fw_rate),
            _num(c.fw_bound),
            str(c.fw_indeterminate),
            _yes(c.gate_pass),
            c.label,
            f"{c.runtime_s:.1f}",
        ]
        for c in cells
    ]
    lines += _table(
        [
            "generator",
            "n_calib",
            "reps",
            "δ",
            "FW rate",
            "FW bound",
            "FW indeterminate",
            "pass",
            "label",
            "runtime s",
        ],
        fw_rows,
    )
    return "\n".join(lines) + "\n"


def _forecast_path_markdown(result: T1ForecastResult) -> list[str]:
    """Interval table and selective-risk table of one T1-forecast path."""
    role = "gated" if result.gated else "reported, not gated"
    shift = (
        f"shift at index {result.shift_index}"
        if result.shift_index is not None
        else "no regime shift"
    )
    lines = [
        f"#### {result.path.capitalize()} path ({role})",
        "",
        f"Generator {result.generator}, seed {result.seed}, α_target {result.alpha_target:g}, "
        f"γ {result.gamma:g}, n_calib {result.n_calib}, evaluation steps {result.n_eval}, "
        f"{shift}.",
        "",
    ]
    rows: list[list[str]] = []
    for h in result.horizons:
        a = h.aci
        lo, hi = a.local_coverage_range
        rows.append(
            [
                str(h.horizon),
                str(a.steps),
                _num(h.miscoverage),
                _num(h.abs_deviation),
                _num(h.bound_over),
                _num(h.bound_under),
                "yes" if h.implied_by_bound else "no",
                _num(a.infinite_share),
                _num(a.empty_share),
                _num(a.median_finite_width, 3),
                f"[{_num(lo, 3)}, {_num(hi, 3)}]",
                _yes(h.passed),
            ]
        )
    lines += _table(
        [
            "h",
            "steps",
            "miscoverage",
            "abs deviation",
            "bound over",
            "bound under",
            "implied by bound",
            "infinite share",
            "empty share",
            "median width",
            "local coverage range",
            "pass" if result.gated else "criterion met (not gated)",
        ],
        rows,
    )
    lines += [
        "",
        "Selective risk on the evaluation window (`holdout_empirical`, not certified):",
        "",
    ]
    sel_rows: list[list[str]] = []
    for h in result.horizons:
        s = h.selective
        sel_rows.append(
            [
                str(h.horizon),
                f"{s.alpha:g}",
                f"{s.tol:g}",
                _num(s.tau_hat),
                s.calib_status,
                _num(s.coverage_eval, 3),
                _num(s.risk_est),
                f"[{_num(s.risk_lo)}, {_num(s.risk_hi)}]",
                _num(s.risk_pre_shift),
                _num(s.risk_post_shift),
                s.guarantee_type.value,
            ]
        )
    lines += _table(
        [
            "h",
            "α",
            "tol",
            "τ̂ (calib window)",
            "calib stop",
            "eval coverage",
            "risk",
            f"block-bootstrap {round(100 * result.horizons[0].selective.conf)}% CI"
            if result.horizons
            else "CI",
            "risk pre-shift",
            "risk post-shift",
            "label",
        ],
        sel_rows,
    )
    if result.notes:
        lines += [""] + [f"- {n}" for n in result.notes]
    return lines


def render_t1_forecast_markdown(gate: T1ForecastGate) -> str:
    """Both T1-forecast paths (D20): the gated stationary path, then the shifted path."""
    st = gate.stationary
    lines = [
        "### T1-forecast",
        "",
        f"Gate (D20): on the stationary path, every horizon has |miscoverage − α| ≤ "
        f"{st.tolerance:g} and infinite + empty share ≤ {st.max_degenerate:g}. "
        f"Verdict: {'PASS' if gate.passed else 'FAIL'}.",
        f"Target |miscoverage − α| ≤ {st.tolerance:g} is an EMPIRICAL target "
        f"({st.target_label}); the interval guarantee is `{st.interval_guarantee.value}`.",
        "Bound over / under are the one-sided deterministic ACI bounds on the miscoverage of "
        "the first T − h errors; 'implied by bound' means the observed deviation lies on a side "
        "whose bound is already within the tolerance, so the check could not fail there.",
        "",
    ]
    lines += _forecast_path_markdown(gate.stationary)
    lines += [""]
    lines += _forecast_path_markdown(gate.shifted)
    return "\n".join(lines) + "\n"


def render_t1_markdown(cells: Sequence[T1Cell], forecast: T1ForecastGate | None = None) -> str:
    """The full T1 report: synth tables, then the forecast section when given."""
    parts = ["## T1 results", "", render_t1_synth_markdown(cells)]
    if forecast is not None:
        parts += ["", render_t1_forecast_markdown(forecast)]
    return "\n".join(parts)
