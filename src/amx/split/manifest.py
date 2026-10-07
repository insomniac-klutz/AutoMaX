"""The hash-locked split manifest (HANDOFF 5.1, 8.2 as amended by C13).

``runs/<id>/splits.manifest.json`` is agent-visible. It holds hashes and counts only:
the spec and data hashes, the sha256 of the sorted dev ids, and for calib and sealed only
counts plus HMAC-SHA256 digests keyed by a secret held in the vault, so the calib and sealed
id sets cannot be confirmed by guessing. It never holds target values or calib/sealed ids;
its fields are closed vocabularies, hashes and integers by construction.

The HMAC key is random per run, so :meth:`SplitManifest.content_hash` differs between two
identical splits. :meth:`SplitManifest.reproducibility_hash` (decision D20) covers every field
except the keyed digests and their key id, so it is equal whenever spec, data and seed are.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from amx import __version__ as amx_version
from amx.data.unitframe import UnitFrame
from amx.spec.enums import Regime
from amx.spec.hashing import content_hash, sha256_hex
from amx.spec.models import TaskSpec
from amx.split.oof import OofScheme
from amx.split.regimes import DropReason, Fold, FoldAssignment

HMAC_PREFIX = "hmac-sha256:"
KEYED_FIELDS = frozenset({"calib_digest", "sealed_digest", "digest_key_id"})


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FoldCounts(_M):
    dev: int = Field(ge=0)
    calib: int = Field(ge=0)
    sealed: int = Field(ge=0)
    dropped: int = Field(ge=0)


class SplitManifest(_M):
    manifest_version: Literal[1] = 1
    spec_hash: str
    data_hash: str
    seed: int
    regime: Regime
    embargo_steps: int = Field(ge=0)
    counts: FoldCounts
    dropped_reasons: dict[DropReason, int]
    dev_ids_sha256: str
    calib_digest: str
    sealed_digest: str
    digest_key_id: str
    oof_k: int = Field(ge=2)
    oof_scheme: OofScheme
    amx_version: str
    independence_group: str | None = None
    independent_counts: FoldCounts | None = Field(
        default=None,
        description="counts in independence units (groups) when independence_unit is group:<col>",
    )

    def content_hash(self) -> str:
        """Hash of the canonical JSON form of the manifest (includes the per-run keyed digests)."""
        return content_hash(self)

    def reproducibility_hash(self) -> str:
        """Hash of every field except ``calib_digest``, ``sealed_digest`` and ``digest_key_id``.

        Decision D20: two splits of the same data with the same spec and seed have equal
        reproducibility hashes although their HMAC keys differ. The calib/sealed partition is
        pinned only indirectly here (the split is deterministic given the spec and data hashes);
        the warden checks it with the keyed digests.
        """
        return content_hash(self.model_dump(mode="json", exclude=set(KEYED_FIELDS)))


def _joined(ids: Iterable[str]) -> bytes:
    return "\n".join(sorted(str(i) for i in ids)).encode("utf-8")


def ids_sha256(ids: Iterable[str]) -> str:
    """sha256 of the sorted ids joined by newlines."""
    return sha256_hex(_joined(ids))


def ids_hmac(ids: Iterable[str], key: bytes) -> str:
    """HMAC-SHA256 of the sorted ids joined by newlines, keyed by a vault-held secret (C13)."""
    return HMAC_PREFIX + hmac.new(key, _joined(ids), hashlib.sha256).hexdigest()


def key_id(key: bytes) -> str:
    """Public fingerprint of an HMAC key (identifies the key without revealing it)."""
    return "sha256:" + hashlib.sha256(b"amx.split.key-id\x00" + key).hexdigest()[:16]


def build_manifest(
    spec: TaskSpec,
    uf: UnitFrame,
    assignment: FoldAssignment,
    *,
    hmac_key: bytes,
    oof_k: int,
    oof_scheme: OofScheme,
) -> SplitManifest:
    """Manifest for ``assignment`` of ``uf`` (deterministic given its inputs)."""
    if assignment.n != uf.n:
        raise ValueError("assignment does not match the UnitFrame")
    ids = uf.ids
    counts = assignment.counts
    group_col = spec.data.independence_group
    indep: FoldCounts | None = None
    if group_col is not None:
        g = np.asarray(uf.column(group_col), dtype=object).astype(str)

        def n_groups(fold: Fold) -> int:
            return len(set(g[assignment.mask(fold)].tolist()))

        indep = FoldCounts(
            dev=n_groups(Fold.DEV),
            calib=n_groups(Fold.CALIB),
            sealed=n_groups(Fold.SEALED),
            dropped=n_groups(Fold.DROPPED),
        )
    reasons = {DropReason(k): v for k, v in sorted(assignment.dropped_reasons.items())}
    return SplitManifest(
        spec_hash=content_hash(spec),
        data_hash=uf.content_hash(),
        seed=spec.splits.seed,
        regime=assignment.regime,
        embargo_steps=assignment.embargo_steps,
        counts=FoldCounts(**counts),
        dropped_reasons=reasons,
        dev_ids_sha256=ids_sha256(ids[assignment.mask(Fold.DEV)]),
        calib_digest=ids_hmac(ids[assignment.mask(Fold.CALIB)], hmac_key),
        sealed_digest=ids_hmac(ids[assignment.mask(Fold.SEALED)], hmac_key),
        digest_key_id=key_id(hmac_key),
        oof_k=oof_k,
        oof_scheme=oof_scheme,
        amx_version=amx_version,
        independence_group=group_col,
        independent_counts=indep,
    )


def manifest_json(manifest: SplitManifest) -> str:
    return manifest.model_dump_json(indent=2) + "\n"


def write_manifest(manifest: SplitManifest, path: str | Path) -> Path:
    """Write the manifest as JSON (atomic rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(manifest_json(manifest), encoding="utf-8")
        os.replace(tmp, p)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return p


def read_manifest(path: str | Path) -> SplitManifest:
    return SplitManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
