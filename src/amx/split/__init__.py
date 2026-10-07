"""Splits, OOF folds, manifests, the local vault and the certify counter (protected core)."""

from amx.split.counter import TOUCH_KINDS, CertifyCounter, SealedTouchLog
from amx.split.dedupe import cluster_ids
from amx.split.errors import (
    CertifyBudgetExhausted,
    SealedTouchError,
    SplitError,
    VaultError,
)
from amx.split.manifest import (
    FoldCounts,
    SplitManifest,
    build_manifest,
    ids_hmac,
    ids_sha256,
    read_manifest,
    write_manifest,
)
from amx.split.oof import (
    NOT_PREDICTED,
    TRAIN_ONLY,
    oof_folds,
    oof_scheme,
    oof_train_mask,
    train_mask,
)
from amx.split.regimes import (
    KEPT_FOLDS,
    DropReason,
    Fold,
    FoldAssignment,
    assign_folds,
    largest_remainder,
    min_embargo_steps,
    stratum_codes,
)
from amx.split.run import OOF_COLUMN, SplitResult, load_dev, split_run
from amx.split.vault import LocalVault

__all__ = [
    "KEPT_FOLDS",
    "NOT_PREDICTED",
    "OOF_COLUMN",
    "TOUCH_KINDS",
    "TRAIN_ONLY",
    "CertifyBudgetExhausted",
    "CertifyCounter",
    "DropReason",
    "Fold",
    "FoldAssignment",
    "FoldCounts",
    "LocalVault",
    "SealedTouchError",
    "SealedTouchLog",
    "SplitError",
    "SplitManifest",
    "SplitResult",
    "VaultError",
    "assign_folds",
    "build_manifest",
    "cluster_ids",
    "ids_hmac",
    "ids_sha256",
    "largest_remainder",
    "load_dev",
    "min_embargo_steps",
    "oof_folds",
    "oof_scheme",
    "oof_train_mask",
    "read_manifest",
    "split_run",
    "stratum_codes",
    "train_mask",
    "write_manifest",
]
