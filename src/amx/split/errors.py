"""Errors raised by ``amx.split`` (protected core, HANDOFF 8.5)."""

from __future__ import annotations


class SplitError(ValueError):
    """The data or spec cannot be split as requested."""


class VaultError(RuntimeError):
    """A vault operation was refused or the vault is in an unexpected state."""


class CertifyBudgetExhausted(VaultError):  # noqa: N818 - name fixed by the A0 plan
    """Every certify call allowed for this run has been used (HANDOFF 7.11, OQ Q1)."""


class SealedTouchError(VaultError):
    """The sealed fold was already used for a final report in this release (HANDOFF 7.11)."""
