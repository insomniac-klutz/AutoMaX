"""``amx split``: create hash-locked folds for a run (HANDOFF 5.1, 5.3, 8.1-8.4; C13, C16).

Writes, in this order:

1. ``calib`` and ``sealed`` (inputs and labels) to the vault, never under the run tree;
2. the TaskSpec snapshot to the vault and to ``runs/<id>/taskspec.yaml``;
3. dev to ``runs/<id>/data/dev.parquet``, with its OOF fold in column ``__amx_oof_fold``;
4. ``runs/<id>/splits.manifest.json`` (hashes and counts only), plus a copy in the vault.

Every fold is written with its dictionary columns decoded: an Arrow dictionary keeps the
values of all units through ``take``, so ``dev.parquet`` would otherwise carry calib and sealed
values (:func:`amx.data.io.decode_dictionaries`).

A run directory that already holds a manifest is never split again (hash lock).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from amx._log import get_logger
from amx.data.io import decode_dictionaries, roles_from_schema, write_unitframe
from amx.data.unitframe import Roles, UnitFrame
from amx.spec.enums import Regime
from amx.spec.loader import dump_taskspec
from amx.spec.models import TaskSpec
from amx.split.errors import SplitError
from amx.split.manifest import (
    SplitManifest,
    build_manifest,
    ids_sha256,
    manifest_json,
    read_manifest,
    write_manifest,
)
from amx.split.oof import oof_folds, oof_scheme
from amx.split.regimes import Fold, FoldAssignment, assign_folds
from amx.split.vault import LocalVault

log = get_logger(__name__)

OOF_COLUMN = "__amx_oof_fold"
DEV_RELPATH = Path("data") / "dev.parquet"
MANIFEST_NAME = "splits.manifest.json"
TASKSPEC_NAME = "taskspec.yaml"


@dataclass(frozen=True)
class SplitResult:
    run_id: str
    run_dir: Path
    manifest: SplitManifest
    manifest_path: Path
    dev_path: Path
    assignment: FoldAssignment
    oof: NDArray[np.int64]


def _within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _check_roles(spec: TaskSpec, uf: UnitFrame) -> None:
    """Refuse a UnitFrame whose roles are not the ones ``spec.data`` defines.

    Every role steers the split (time and series for ``temporal``, groups for ``grouped``,
    inputs for duplicate clusters, the target for stratification), so all of them must match
    :meth:`Roles.from_spec`, not only ``unit_id`` and ``target``.
    """
    expected = Roles.from_spec(spec.data, uf.table.column_names, spec.constraints.forbidden_inputs)
    if uf.roles == expected:
        return
    diffs = [
        f"{f.name}: frame {getattr(uf.roles, f.name)!r}, spec {getattr(expected, f.name)!r}"
        for f in fields(Roles)
        if getattr(uf.roles, f.name) != getattr(expected, f.name)
    ]
    raise SplitError("the UnitFrame roles do not match spec.data: " + "; ".join(diffs))


def _fold_frame(uf: UnitFrame, idx: NDArray[np.int64]) -> UnitFrame:
    """Units ``idx`` of ``uf`` with dictionary columns decoded, so no other fold's values ride
    along in a dictionary."""
    part = uf.take(idx)
    return UnitFrame(decode_dictionaries(part.table), part.roles)


def split_run(
    spec: TaskSpec,
    uf: UnitFrame,
    run_dir: str | Path,
    vault: LocalVault,
    run_id: str,
    *,
    regime: Regime | str | None = None,
) -> SplitResult:
    """Split ``uf`` for run ``run_id``: dev into ``run_dir``, calib and sealed into ``vault``.

    ``uf`` must carry exactly the roles of ``spec.data`` (``Roles.from_spec``).
    ``regime`` overrides ``spec.splits.regime`` (the profiler's resolution of ``auto``).
    """
    rd = Path(run_dir).expanduser().resolve()
    if _within(rd, vault.root) or _within(vault.root, rd):
        raise SplitError(f"run directory {rd} and vault {vault.root} must not contain each other")
    if (rd / MANIFEST_NAME).exists():
        raise SplitError(f"{rd} is already split ({MANIFEST_NAME} exists); use a new run")
    for name in ("calib", "sealed"):
        if vault.has_fold(run_id, name):
            raise SplitError(f"vault already holds fold '{name}' for run '{run_id}'")
    _check_roles(spec, uf)
    if OOF_COLUMN in uf.table.column_names:
        raise SplitError(f"column name '{OOF_COLUMN}' is reserved")

    assignment = assign_folds(uf, spec, regime)
    dev_idx = assignment.indices(Fold.DEV)
    dev = _fold_frame(uf, dev_idx)
    k = spec.splits.oof_folds
    oof = oof_folds(
        dev,
        k,
        assignment.regime,
        spec.splits.seed,
        assignment.clusters[dev_idx],
        target_kind=spec.data.target.kind,
    )

    vault.write_fold(run_id, "calib", _fold_frame(uf, assignment.indices(Fold.CALIB)))
    vault.write_fold(run_id, "sealed", _fold_frame(uf, assignment.indices(Fold.SEALED)))
    spec_yaml = dump_taskspec(spec)
    vault.write_text(run_id, TASKSPEC_NAME, spec_yaml)

    manifest = build_manifest(
        spec,
        uf,
        assignment,
        hmac_key=vault.hmac_key(run_id),
        oof_k=k,
        oof_scheme=oof_scheme(assignment.regime),
    )
    rd.mkdir(parents=True, exist_ok=True)
    (rd / TASKSPEC_NAME).write_text(spec_yaml, encoding="utf-8")
    dev_path = write_unitframe(dev, rd / DEV_RELPATH, extra_columns={OOF_COLUMN: oof})
    vault.write_text(run_id, MANIFEST_NAME, manifest_json(manifest))
    manifest_path = write_manifest(manifest, rd / MANIFEST_NAME)
    log.info(
        "split run %s: %s (manifest %s, reproducibility %s)",
        run_id,
        manifest.counts.model_dump(),
        manifest.content_hash(),
        manifest.reproducibility_hash(),
    )
    return SplitResult(
        run_id=run_id,
        run_dir=rd,
        manifest=manifest,
        manifest_path=manifest_path,
        dev_path=dev_path,
        assignment=assignment,
        oof=oof,
    )


def load_dev(run_dir: str | Path, *, verify: bool = True) -> tuple[UnitFrame, NDArray[np.int64]]:
    """Dev UnitFrame and its OOF folds from ``run_dir``.

    With ``verify`` the dev ids must match the manifest's ``dev_ids_sha256`` (hash lock).
    """
    rd = Path(run_dir)
    table = pq.read_table(rd / DEV_RELPATH)
    if OOF_COLUMN not in table.column_names:
        raise SplitError(f"{rd / DEV_RELPATH} has no '{OOF_COLUMN}' column")
    oof = np.asarray(table.column(OOF_COLUMN).to_numpy(), dtype=np.int64)
    dev = UnitFrame(table, roles_from_schema(table.schema))
    if verify:
        manifest = read_manifest(rd / MANIFEST_NAME)
        if ids_sha256(dev.ids) != manifest.dev_ids_sha256:
            raise SplitError(f"dev ids in {rd} do not match {MANIFEST_NAME} (hash lock broken)")
        if dev.n != manifest.counts.dev:
            raise SplitError(f"dev count in {rd} does not match {MANIFEST_NAME}")
    return dev, oof
