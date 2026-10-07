"""``frontier.png``: the descriptive risk-coverage curve (HANDOFF 5.1, 6.5; ROLLER step 11).

The curve is the certificate's ``frontier_descriptive``: empirical selective risk against
coverage on the calibration sample, one point per grid τ. It carries no guarantee, and the
title says so. Horizontal lines mark each band's α; a marker sits at each band's τ̂.

Rendering uses matplotlib's Agg canvas directly (no pyplot state, no display needed). Colours
follow the reference categorical order of the dataviz palette: slot 1 for the curve, the
following slots for the bands in band order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from amx._log import get_logger
from amx.cert.certificate import Certificate
from amx.report.wording import claims_certification, fmt_num, honest_text

log = get_logger(__name__)

CURVE_COLOR = "#2a78d6"
BAND_COLORS = ("#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
ZOOM = 2.5  # the risk axis shows up to ZOOM × the loosest α when the curve rises above it


def _band_point(cert: Certificate, tau_hat: float) -> tuple[float, float] | None:
    for p in cert.frontier_descriptive:
        if np.isclose(p.tau, tau_hat, rtol=1e-12, atol=0.0) and p.risk is not None:
            return p.coverage, p.risk
    return None


def frontier_title(cert: Certificate) -> str:
    """Plot title: always says "descriptive", and never "certified" unless the type allows."""
    text = f"Risk-coverage frontier (descriptive) - {cert.guarantee.type.value}"
    return honest_text(text, claims_certification(cert.guarantee.type))


def plot_frontier(cert: Certificate, path: str | Path) -> Path:
    """Write the descriptive coverage-vs-risk plot of ``cert`` to ``path`` (PNG)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pts = [(p.coverage, p.risk) for p in cert.frontier_descriptive if p.risk is not None]
    cov = np.asarray([c for c, _ in pts], dtype=np.float64)
    risk = np.asarray([r for _, r in pts], dtype=np.float64)

    fig = Figure(figsize=(7.5, 4.8), dpi=120, facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1, facecolor=SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    if cov.size:
        ax.plot(
            cov,
            risk,
            color=CURVE_COLOR,
            linewidth=2.0,
            solid_joinstyle="round",
            solid_capstyle="round",
            label="empirical selective risk (calibration)",
        )
    alphas = [b.alpha for b in cert.bands]
    drawn: list[float] = []
    for j, b in enumerate(cert.bands):
        color = BAND_COLORS[j % len(BAND_COLORS)]
        ax.axhline(b.alpha, color=color, linewidth=1.0, alpha=0.9)
        ax.annotate(
            rf"$\alpha$ = {fmt_num(b.alpha)}",
            xy=(1.0, b.alpha),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_MUTED,
        )
        if b.tau_hat is None:
            continue
        point = _band_point(cert, b.tau_hat)
        if point is None:
            log.warning("band alpha=%s: tau_hat not on the descriptive frontier", b.alpha)
            continue
        # Bands that share a τ̂ (inherited thresholds) are drawn as growing rings, so that
        # every band's marker stays visible.
        stacked = sum(1 for t in drawn if t == b.tau_hat)
        drawn.append(b.tau_hat)
        ax.plot(
            [point[0]],
            [point[1]],
            marker="o",
            markersize=8 + 5 * stacked,
            zorder=3 - 0.01 * stacked,
            clip_on=False,
            markerfacecolor=color,
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
            linestyle="none",
            label=rf"$\hat{{\tau}}$ = {fmt_num(b.tau_hat)} at $\alpha$ = {fmt_num(b.alpha)}",
        )

    top = max([*alphas, float(np.max(risk)) if risk.size else 0.0, 1e-12])
    if alphas and top > ZOOM * max(alphas):
        top = ZOOM * max(alphas)
    ax.set_ylim(0.0, min(1.0, top * 1.15))
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("coverage (share of calibration units committed)", color=INK, fontsize=10)
    ax.set_ylabel("selective risk", color=INK, fontsize=10)
    ax.set_title(frontier_title(cert), color=INK, fontsize=11, loc="left")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(handles, labels, loc="best", fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, format="png", facecolor=SURFACE, metadata={"Software": None})
    return out
