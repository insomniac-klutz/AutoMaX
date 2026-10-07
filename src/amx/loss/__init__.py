"""Losses: builtins, custom loading, validation (protected core, HANDOFF 8.5)."""

from amx.loss.base import Loss, LossError
from amx.loss.builtins import anomaly_cost, err_gt_tol, missed_anomaly, one_minus_f1, zero_one
from amx.loss.custom import LossConfirmationError, load_custom_loss
from amx.loss.registry import build_builtin, build_loss, needs_confirmation
from amx.loss.squash import arctan_squash, squashed
from amx.loss.validate import LossCheck, LossDistribution, check_loss, loss_distribution

__all__ = [
    "Loss",
    "LossCheck",
    "LossConfirmationError",
    "LossDistribution",
    "LossError",
    "anomaly_cost",
    "arctan_squash",
    "build_builtin",
    "build_loss",
    "check_loss",
    "err_gt_tol",
    "load_custom_loss",
    "loss_distribution",
    "missed_anomaly",
    "needs_confirmation",
    "one_minus_f1",
    "squashed",
    "zero_one",
]
