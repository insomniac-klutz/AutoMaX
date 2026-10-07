"""Custom losses loaded from a file. Using one needs a human ``--confirm-loss`` (HANDOFF 7.9)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from amx.loss.base import Loss, LossError
from amx.loss.squash import squashed


class LossConfirmationError(LossError):
    """A custom loss or overridden default was used without human confirmation."""


def load_custom_loss(
    path: str | Path,
    fn: str,
    *,
    confirmed: bool,
    params: dict[str, Any] | None = None,
) -> Loss:
    """Import ``fn`` from ``path``.

    The function takes (pred, gold) and returns per-unit losses. Set an attribute
    ``is_binary = True`` on it to declare a 0/1 loss. ``params.squash`` (scale) wraps an
    unbounded non-negative loss with the arctan squash.
    """
    if not confirmed:
        raise LossConfirmationError(
            "custom losses change what the certificate means; re-run with --confirm-loss"
        )
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"amx_custom_loss_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise LossError(f"cannot import custom loss from {p}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, fn, None)
    if not callable(func):
        raise LossError(f"{p} has no callable '{fn}'")
    params = dict(params or {})
    if "squash" in params:
        return squashed(func, name=f"custom:{p.name}:{fn}", scale=float(params["squash"]))
    return Loss(
        f"custom:{p.name}:{fn}",
        func,
        is_binary=bool(getattr(func, "is_binary", False)),
        params=params,
    )
