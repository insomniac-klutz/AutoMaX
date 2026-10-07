"""Reports rendered from a certificate: ``bands.md``, ``frontier.png`` (ROLLER step 11).

The bands JSON schema lives in :mod:`amx.report.schema` (not re-exported, so that
``python -m amx.report.schema`` runs cleanly).
"""

from amx.report.bands_md import render_bands_md
from amx.report.frontier import plot_frontier
from amx.report.wording import CERTIFYING_TYPES, claims_certification, fmt_num, honest_text

__all__ = [
    "CERTIFYING_TYPES",
    "claims_certification",
    "fmt_num",
    "honest_text",
    "plot_frontier",
    "render_bands_md",
]
