from __future__ import annotations

import re
from typing import Any

import pytest

from amx.cert import Certificate, GuaranteeType
from amx.report import claims_certification, fmt_num, honest_text, render_bands_md
from tests.unit.report.conftest import with_sealed, with_type

NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w|\.\d)")
SECTIONS = [
    "## Guarantee",
    "## Assumptions",
    "## Bands",
    "### Walk details",
    "## Slices",
    "## Warnings",
    "## Never claimed",
]
BAND_HEADER = (
    "| α | policy | status | τ̂ | raw τ̂ | stop reason | n_min | committed n | independent n | "
    "coverage est [ci95] | calib risk |"
)
NON_CERTIFYING = [
    GuaranteeType.NONE,
    GuaranteeType.HOLDOUT_EMPIRICAL,
    GuaranteeType.LONG_RUN_FREQUENCY,
]


def allowed_numbers(cert: Certificate) -> set[str]:
    """Every number the report may show: certificate values through fmt_num, plus numbers
    already written inside the certificate's strings (warnings, slice values, assumptions)."""
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, int | float):
            out.add(fmt_num(node))
        elif isinstance(node, str):
            out.update(NUMBER.findall(node))
        elif isinstance(node, dict):
            for key, value in node.items():
                out.update(NUMBER.findall(str(key)))
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(cert.model_dump(mode="json"))
    return out


def band_rows(md: str) -> list[list[str]]:
    lines = md.splitlines()
    start = lines.index(next(line for line in lines if line.startswith("| α | policy |")))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip("|").split(" | ")])
    return rows


def test_golden_structure(cert: Certificate) -> None:
    md = render_bands_md(cert)
    assert md.startswith("# Risk-coverage certificate\n")
    positions = [md.index(h + "\n") for h in SECTIONS]
    assert positions == sorted(positions)
    assert BAND_HEADER in md
    rows = band_rows(md)
    assert len(rows) == len(cert.bands)
    for row, band in zip(rows, cert.bands, strict=True):
        assert len(row) == 11
        assert row[0] == fmt_num(band.alpha)
        assert row[1] == band.policy.value
        assert row[3] == fmt_num(band.tau_hat)
        assert row[4] == fmt_num(band.tau_hat_raw)
        assert row[5] == band.stop_reason.value
        assert row[6] == str(band.n_min_required)
        assert row[7] == str(band.n_committed_calib)
    assert [r[2] for r in rows] == ["uncertified", "certified", "inherited from α = 0.02"]
    g = cert.guarantee
    assert f"- Type: `{g.type.value}`" in md
    assert f"- δ per band (δ_j): {fmt_num(g.delta_per_band)}" in md
    assert f"- Simultaneous level over all bands: {fmt_num(g.simultaneous_level)}" in md
    assert f"- Certify calls used / max: {g.certify_calls_used} / {g.certify_calls_max}" in md
    assert f"- Calibration units (calib_n): {g.calib_n}" in md
    assert f"- Independent calibration units: {g.calib_independent_n}" in md
    assert f"`{g.grid_hash}`" in md
    for a in g.assumptions:
        assert f"- {a}" in md
    for w in cert.warnings:
        assert f"- {w}" in md
    for item in cert.never_claimed:
        assert f"- {item}" in md
    assert "### Band α = 0.02" in md
    assert "upper_gt_2alpha" in md and "insufficient_n" in md
    assert "### Sealed fold" not in md
    covered = cert.bands[1].coverage_at_certified_tau
    assert covered.ci95 is not None
    lo, hi = covered.ci95
    assert f"{fmt_num(covered.est)} [{fmt_num(lo)}, {fmt_num(hi)}]" in md


def test_grouped_and_sealed_sections(grouped_cert: Certificate) -> None:
    md = render_bands_md(with_sealed(grouped_cert))
    assert "calib risk, unit-weighted" in md
    assert "`group_weighted`" in md and "`hoeffding_bentkus`" in md
    assert "### Sealed fold" in md
    assert "- Sealed units: 5987" in md
    assert "0.0071 (upper95 0.0123 clopper_pearson)" in md


@pytest.mark.parametrize("gtype", list(GuaranteeType), ids=lambda t: t.value)
@pytest.mark.parametrize("variant", ["plain", "grouped", "sealed"])
def test_honesty_rule(
    gtype: GuaranteeType, variant: str, cert: Certificate, grouped_cert: Certificate
) -> None:
    base = {"plain": cert, "grouped": grouped_cert, "sealed": with_sealed(cert)}[variant]
    md = render_bands_md(with_type(base, gtype))
    if gtype in (GuaranteeType.NONE, GuaranteeType.HOLDOUT_EMPIRICAL):
        assert not claims_certification(gtype)
    if claims_certification(gtype):
        assert md.startswith("# Risk-coverage certificate")
        assert "| certified |" in md
    else:
        assert "certified" not in md.lower()
        assert md.startswith("# Risk-coverage report (estimated)")
        assert "| selected |" in md and "| not selected |" in md
        assert "τ̂ (selected)" in md


@pytest.mark.parametrize("gtype", list(GuaranteeType), ids=lambda t: t.value)
@pytest.mark.parametrize("variant", ["plain", "grouped", "sealed"])
def test_every_number_comes_from_the_certificate(
    gtype: GuaranteeType, variant: str, cert: Certificate, grouped_cert: Certificate
) -> None:
    base = {"plain": cert, "grouped": grouped_cert, "sealed": with_sealed(cert)}[variant]
    c = with_type(base, gtype)
    md = render_bands_md(c)
    tokens = NUMBER.findall(md)
    assert len(tokens) > 40
    unknown = sorted(set(tokens) - allowed_numbers(c))
    assert unknown == [], f"numbers not traceable to the certificate: {unknown}"


def test_render_is_pure(cert: Certificate) -> None:
    before = cert.model_dump(mode="json")
    assert render_bands_md(cert) == render_bands_md(cert)
    assert cert.model_dump(mode="json") == before


def test_honest_text_rewrites() -> None:
    text = "Certified band; uncertified; CERTIFIED; not certifiable; recertified."
    out = honest_text(text, certifying=False)
    assert "certified" not in out.lower()
    assert out == "Selected band; not selected; SELECTED; not selectable; reselected."
    assert honest_text(text, certifying=True) == text


def test_fmt_num() -> None:
    assert fmt_num(12000) == "12000"
    assert fmt_num(0.012345678) == "0.01235"
    assert fmt_num(1e-4) == "0.0001"
    assert fmt_num(1.23456e-5) == "1.235e-05"
    assert fmt_num(1.0) == "1"
    assert fmt_num(None) == "–"
    assert fmt_num(float("nan")) == "–"
    with pytest.raises(TypeError):
        fmt_num(True)
