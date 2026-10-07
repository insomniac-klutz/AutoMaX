"""Honest wording and number formatting shared by the report renderers (HANDOFF 0.3, 7.1).

HANDOFF 0.3: never emit the word "certified" for a number whose assumptions were not audited.
A report whose guarantee type is ``none`` or ``holdout_empirical`` (and, by the same logic,
``long_run_frequency``, whose guarantee covers interval miscoverage only, D15) describes
*selected* thresholds and *estimated* risks. :func:`honest_text` rewrites claim words for such
reports; the renderers apply it to every piece of text they emit.

Every number a report shows goes through :func:`fmt_num`, so the rendered document contains
no number that cannot be traced back to the certificate.
"""

from __future__ import annotations

import math
import re

from amx.cert.guarantee import GuaranteeType

SIGNIFICANT_DIGITS = 4
MISSING = "–"

# Guarantee types under which a band threshold may be called "certified". LTT gives a
# high-probability bound and CRC an expectation bound on the selective risk; ACI's long-run
# frequency covers interval miscoverage only, never the selective risk at τ̂ (7.1, D15).
CERTIFYING_TYPES: frozenset[GuaranteeType] = frozenset(
    {GuaranteeType.PAC_HIGH_PROB, GuaranteeType.EXPECTATION}
)

# No word boundaries: "recertified" or "pre-certified" must be caught as well.
_CLAIM_WORDS = re.compile(r"(un)?(certified|certifiable)", re.IGNORECASE)
_REPLACEMENT = {
    ("un", "certified"): "not selected",
    ("", "certified"): "selected",
    ("un", "certifiable"): "not selectable",
    ("", "certifiable"): "selectable",
}


def claims_certification(gtype: GuaranteeType) -> bool:
    """True when the report may use the word "certified" for its band thresholds."""
    return gtype in CERTIFYING_TYPES


def _swap(match: re.Match[str]) -> str:
    word = match.group(0)
    key = ((match.group(1) or "").lower(), match.group(2).lower())
    out = _REPLACEMENT[key]
    if word.isupper():
        return out.upper()
    if word[0].isupper():
        return out[0].upper() + out[1:]
    return out


def honest_text(text: str, certifying: bool) -> str:
    """Rewrite claim words ("certified", "uncertified", "certifiable") when not certifying."""
    if certifying:
        return text
    return _CLAIM_WORDS.sub(_swap, text)


def fmt_num(x: float | int | None) -> str:
    """The one number format of the report: integers exactly, floats to 4 significant digits."""
    if x is None:
        return MISSING
    if isinstance(x, bool):
        raise TypeError("booleans are not numbers in a report")
    if isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        return MISSING
    return f"{x:.{SIGNIFICANT_DIGITS}g}"
