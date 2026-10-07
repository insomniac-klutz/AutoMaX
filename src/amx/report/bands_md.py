"""``bands.md``: the human-readable report, rendered from the certificate only (HANDOFF 6.5).

ROLLER step 11. The renderer reads nothing but a :class:`amx.cert.Certificate`. It shows the
guarantee (type, δ, regime, estimand, p-value family, certify calls, calibration counts, grid
hash), its assumptions, one row per band, per-slice tables with their flags (7.10, C7), the
warnings and the never-claimed list (7.6).

Two rules make the document auditable:

* **Honesty (HANDOFF 0.3).** For guarantee types that do not bound the selective risk
  (``none``, ``holdout_empirical``, ``long_run_frequency``) the word "certified" never appears;
  thresholds are "selected" and risks "estimated" (:mod:`amx.report.wording`). The rewrite
  runs over the whole document, so text copied from the certificate (assumptions, warnings,
  the never-claimed list, even a run id) cannot reintroduce the word.
* **No invented numbers.** Every number is a value of the certificate formatted by
  :func:`amx.report.wording.fmt_num`, or a number already inside one of its strings. The
  confidence level appears through the certificate's own field names (``ci95``, ``upper95``).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from amx.cert.certificate import BandEntry, Certificate, Interval, RiskEstimate
from amx.cert.guarantee import Guarantee, GuaranteeType
from amx.cert.ltt import BandStatus
from amx.report.wording import MISSING, claims_certification, fmt_num, honest_text

_STATEMENTS: dict[GuaranteeType, str] = {
    GuaranteeType.PAC_HIGH_PROB: (
        "Except with probability at most δ over the draw of the calibration set, the selective "
        "risk at τ̂ is at most α for every band at once, under the assumptions below. Coverage "
        "is estimated, not guaranteed."
    ),
    GuaranteeType.EXPECTATION: (
        "In expectation over the draw of the calibration set, the selective risk at τ̂ is at "
        "most α for each band, under the assumptions below. Coverage is estimated, not "
        "guaranteed."
    ),
    GuaranteeType.LONG_RUN_FREQUENCY: (
        "Long-run frequency only: interval miscoverage is controlled on average over a long run "
        "of future time steps. The thresholds below are selected, and their selective risks "
        "are estimates without a finite-sample bound."
    ),
    GuaranteeType.HOLDOUT_EMPIRICAL: (
        "Empirical estimates on a holdout calibration window. They describe that window and "
        "carry over to future units only if the data are stationary; no finite-sample "
        "guarantee is claimed."
    ),
    GuaranteeType.NONE: "Descriptive numbers only. No guarantee of any kind is claimed.",
}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(_cell(h) for h in header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    out.extend("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows)
    return out


def _code(text: str) -> str:
    return f"`{text}`"


def _interval(iv: Interval | None) -> str:
    if iv is None or iv.est is None:
        return MISSING
    if iv.ci95 is None:
        return fmt_num(iv.est)
    lo, hi = iv.ci95
    return f"{fmt_num(iv.est)} [{fmt_num(lo)}, {fmt_num(hi)}]"


def _risk(r: RiskEstimate | None) -> str:
    if r is None or r.est is None:
        return MISSING
    if r.upper95 is None:
        return fmt_num(r.est)
    bound = f" {r.bound}" if r.bound else ""
    return f"{fmt_num(r.est)} (upper95 {fmt_num(r.upper95)}{bound})"


def _status(band: BandEntry, bands: Sequence[BandEntry], certifying: bool) -> str:
    if band.status is BandStatus.INHERITED:
        src = band.source_band
        if src is not None and 0 <= src < len(bands):
            return f"inherited from α = {fmt_num(bands[src].alpha)}"
        return "inherited"
    if band.status is BandStatus.CERTIFIED:
        return "certified" if certifying else "selected"
    return "uncertified" if certifying else "not selected"


def _guarantee_section(g: Guarantee, certifying: bool) -> list[str]:
    calls = f"{fmt_num(g.certify_calls_used)} / {fmt_num(g.certify_calls_max)}"
    lines = [
        "## Guarantee",
        "",
        f"- Type: {_code(g.type.value)}",
        f"- Statement: {_STATEMENTS[g.type]}",
        f"- δ (total failure budget): {fmt_num(g.delta)}",
        f"- δ per band (δ_j): {fmt_num(g.delta_per_band)}",
        f"- Simultaneous level over all bands: {fmt_num(g.simultaneous_level)}",
        f"- Regime: {_code(g.regime.value)}",
        f"- Estimand: {_code(g.estimand)}",
        f"- p-value family: {_code(g.p_value_family)}",
        f"- Test mode: {_code(g.cert_mode)}; call policy: {_code(g.call_policy.value)}",
        f"- Certify calls used / max: {calls}",
        f"- Calibration units (calib_n): {fmt_num(g.calib_n)}",
        f"- Independent calibration units: {fmt_num(g.calib_independent_n)}",
    ]
    if g.sealed_n is not None:
        lines.append(f"- Sealed units: {fmt_num(g.sealed_n)}")
    lines.append(f"- τ grid hash: {_code(g.grid_hash)}")
    if not certifying:
        lines.append(
            "- The thresholds below are selected on calibration data; the risk figures are "
            "estimates."
        )
    return lines


def _band_section(cert: Certificate, certifying: bool) -> list[str]:
    bands = cert.bands
    group = cert.guarantee.estimand == "group_weighted"
    tau_word = "τ̂" if certifying else "τ̂ (selected)"
    header = [
        "α",
        "policy",
        "status",
        tau_word,
        "raw τ̂",
        "stop reason",
        "n_min",
        "committed n",
        "independent n",
        "coverage est [ci95]",
        "calib risk",
    ]
    if group:
        header.append("calib risk, unit-weighted")
    rows: list[list[str]] = []
    for b in bands:
        row = [
            fmt_num(b.alpha),
            b.policy.value,
            _status(b, bands, certifying),
            fmt_num(b.tau_hat),
            fmt_num(b.tau_hat_raw),
            b.stop_reason.value,
            fmt_num(b.n_min_required),
            fmt_num(b.n_committed_calib),
            fmt_num(b.n_independent_calib),
            _interval(b.coverage_at_certified_tau),
            _risk(b.risk_calib),
        ]
        if group:
            row.append(fmt_num(b.risk_calib_unit_weighted))
        rows.append(row)
    methods = sorted({b.coverage_at_certified_tau.method for b in bands})
    lines = ["## Bands", ""]
    lines.extend(_table(header, rows))
    lines += [
        "",
        "Coverage is an estimate at τ̂; its interval method: "
        + ", ".join(_code(m) for m in methods)
        + ". Raw τ̂ is the band's own walk before cross-band monotonisation.",
        "",
        "### Walk details",
        "",
    ]
    walk_rows = [
        [
            fmt_num(b.alpha),
            fmt_num(b.delta_j),
            fmt_num(b.start_index),
            fmt_num(b.tests_run),
            fmt_num(b.p_value_raw),
        ]
        for b in bands
    ]
    lines.extend(_table(["α", "δ_j", "start index", "tests run", "p-value at raw τ̂"], walk_rows))
    if any(b.risk_sealed is not None or b.coverage_sealed is not None for b in bands):
        lines += ["", "### Sealed fold", ""]
        sealed_rows = [
            [fmt_num(b.alpha), _interval(b.coverage_sealed), _risk(b.risk_sealed)] for b in bands
        ]
        lines.extend(_table(["α", "coverage est [ci95]", "risk"], sealed_rows))
    return lines


def _slice_section(cert: Certificate) -> list[str]:
    lines = [
        "## Slices",
        "",
        "Per-slice selective risk among committed calibration units. Descriptive only: no "
        "per-slice guarantee is claimed. Flag `insufficient_n`: too few "
        "committed units for a meaningful bound. Flag `upper_gt_2alpha`: the upper bound "
        "exceeds twice the band's α.",
    ]
    any_slice = False
    for b in cert.bands:
        if not b.per_slice:
            continue
        any_slice = True
        lines += ["", f"### Band α = {fmt_num(b.alpha)}", ""]
        rows = [
            [
                s.slice,
                s.value,
                fmt_num(s.n),
                fmt_num(s.risk),
                fmt_num(s.upper95),
                s.bound,
                s.flag or "",
            ]
            for s in b.per_slice
        ]
        lines.extend(_table(["slice", "value", "n", "risk", "upper95", "bound", "flag"], rows))
    if not any_slice:
        lines += ["", "No watch slices were evaluated."]
    return lines


def _bullets(items: Sequence[str], empty: str) -> list[str]:
    if not items:
        return [empty]
    return [f"- {item}" for item in items]


def render_bands_md(cert: Certificate) -> str:
    """Render ``bands.md`` from a certificate. Pure: no I/O, no numbers outside the input."""
    g = cert.guarantee
    certifying = claims_certification(g.type)
    title = "Risk-coverage certificate" if certifying else "Risk-coverage report (estimated)"
    lines = [
        f"# {title}",
        "",
        f"- Run: {_code(cert.run_id)}",
        f"- amx version: {_code(cert.amx_version)}",
        f"- TaskSpec hash: {_code(cert.taskspec_hash)}",
        f"- Artifact hash: {_code(cert.artifact_hash)}",
        "",
    ]
    lines += _guarantee_section(g, certifying)
    lines += ["", "## Assumptions", ""]
    lines += _bullets(g.assumptions, "None stated.")
    lines += [""]
    lines += _band_section(cert, certifying)
    lines += [""]
    lines += _slice_section(cert)
    lines += ["", "## Warnings", ""]
    lines += _bullets(cert.warnings, "None.")
    lines += ["", "## Never claimed", ""]
    lines += _bullets(cert.never_claimed, "None listed.")
    lines += [
        "",
        "The risk-coverage frontier (`frontier.png`) is descriptive.",
        "",
    ]
    return honest_text("\n".join(lines), certifying)
