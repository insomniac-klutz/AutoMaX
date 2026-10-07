"""Build the run's loss from its TaskSpec."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from amx.loss import builtins
from amx.loss.base import Loss, LossError
from amx.loss.custom import LossConfirmationError, load_custom_loss
from amx.spec.enums import FAMILY_DEFAULT_LOSS, BuiltinLoss, LossKind
from amx.spec.models import TaskSpec

BUILTINS: dict[BuiltinLoss, Callable[..., Loss]] = {
    BuiltinLoss.ZERO_ONE: builtins.zero_one,
    BuiltinLoss.ERR_GT_TOL: builtins.err_gt_tol,
    BuiltinLoss.ONE_MINUS_F1: builtins.one_minus_f1,
    BuiltinLoss.MISSED_ANOMALY: builtins.missed_anomaly,
    BuiltinLoss.ANOMALY_COST: builtins.anomaly_cost,
}


def needs_confirmation(spec: TaskSpec) -> bool:
    """True when the loss is custom or overrides the family default (HANDOFF 7.9)."""
    if spec.task.loss.kind is LossKind.CUSTOM:
        return True
    return spec.task.loss.name is not None and (
        spec.task.loss.name is not FAMILY_DEFAULT_LOSS[spec.task.family]
    )


def build_loss(spec: TaskSpec, *, confirmed: bool = False, base_dir: str | Path = ".") -> Loss:
    """Instantiate the spec's loss, checking its parameters."""
    ls = spec.task.loss
    if needs_confirmation(spec) and not confirmed:
        raise LossConfirmationError(
            "this TaskSpec uses a custom or non-default loss; confirm it with --confirm-loss"
        )
    if ls.kind is LossKind.CUSTOM:
        assert ls.path is not None and ls.fn is not None
        path = Path(ls.path)
        if not path.is_absolute():
            path = Path(base_dir) / path
        return load_custom_loss(path, ls.fn, confirmed=True, params=dict(ls.params))
    name = spec.task.loss_name
    assert name is not None
    return build_builtin(name, ls.params)


def build_builtin(name: BuiltinLoss, params: dict[str, Any]) -> Loss:
    try:
        return BUILTINS[name](**params)
    except TypeError as exc:
        raise LossError(f"bad params for loss '{name.value}': {exc}") from exc
