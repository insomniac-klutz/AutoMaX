from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pydantic
import pytest

from amx.spec import content_hash
from amx.split import (
    Fold,
    SplitManifest,
    assign_folds,
    build_manifest,
    ids_hmac,
    ids_sha256,
    read_manifest,
    write_manifest,
)
from amx.split.manifest import manifest_json
from tests.unit.split.helpers import make_frame, make_spec

KEY = bytes(range(32))
KEYED = {"calib_digest", "sealed_digest", "digest_key_id"}


def manifest_for(seed: int, key: bytes = KEY, *, numeric: bool = False) -> SplitManifest:
    uf = make_frame(300, seed=0, numeric=numeric, dup_of={7: 1})
    spec = make_spec(seed=seed, kind="numeric" if numeric else "categorical")
    fa = assign_folds(uf, spec)
    return build_manifest(
        spec,
        uf,
        fa,
        hmac_key=key,
        oof_k=spec.splits.oof_folds,
        oof_scheme="stratified_cluster_kfold",
    )


def test_same_seed_same_manifest_different_seed_different() -> None:
    a, b, c = manifest_for(5), manifest_for(5), manifest_for(6)
    assert manifest_json(a) == manifest_json(b)
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()
    assert a.dev_ids_sha256 != c.dev_ids_sha256
    assert a.calib_digest != c.calib_digest and a.sealed_digest != c.sealed_digest


def test_key_changes_only_the_keyed_digests() -> None:
    a, b = manifest_for(5), manifest_for(5, key=b"\x01" * 32)
    da, db = a.model_dump(), b.model_dump()
    assert {k for k in da if da[k] != db[k]} == KEYED


def test_reproducibility_hash_ignores_only_the_keyed_digests() -> None:
    """Decision D20: equal for identical splits although the HMAC key is random per run."""
    a, b, c = manifest_for(5), manifest_for(5, key=b"\x01" * 32), manifest_for(6)
    assert a.content_hash() != b.content_hash()
    assert a.reproducibility_hash() == b.reproducibility_hash()
    assert a.reproducibility_hash() != c.reproducibility_hash()
    assert a.reproducibility_hash() != a.content_hash()
    unkeyed = set(SplitManifest.model_fields) - KEYED
    assert unkeyed and set(json.loads(manifest_json(a))) - KEYED == unkeyed
    changes = {
        "seed": 99,
        "data_hash": "sha256:" + "1" * 64,
        "spec_hash": "sha256:" + "2" * 64,
        "dev_ids_sha256": "sha256:" + "3" * 64,
        "counts": a.counts.model_copy(update={"dropped": a.counts.dropped + 1}),
        "embargo_steps": a.embargo_steps + 1,
        "oof_k": a.oof_k + 1,
        "amx_version": a.amx_version + "+other",
    }
    for field, value in changes.items():
        changed = a.model_copy(update={field: value})
        assert changed.reproducibility_hash() != a.reproducibility_hash(), field
    for field in sorted(KEYED):
        changed = a.model_copy(update={field: "hmac-sha256:" + "f" * 64})
        assert changed.reproducibility_hash() == a.reproducibility_hash(), field


def test_manifest_fields_and_hashes() -> None:
    uf = make_frame(300, seed=0, dup_of={7: 1})
    spec = make_spec(seed=5)
    fa = assign_folds(uf, spec)
    m = build_manifest(spec, uf, fa, hmac_key=KEY, oof_k=5, oof_scheme="stratified_cluster_kfold")
    assert m.spec_hash == content_hash(spec)
    assert m.data_hash == uf.content_hash()
    assert m.seed == 5 and m.regime.value == "iid" and m.embargo_steps == 0
    assert m.counts.model_dump() == fa.counts
    assert m.dropped_reasons == {}
    ids = uf.ids
    dev = sorted(ids[fa.mask(Fold.DEV)].tolist())
    assert m.dev_ids_sha256 == "sha256:" + hashlib.sha256("\n".join(dev).encode()).hexdigest()
    calib = sorted(ids[fa.mask(Fold.CALIB)].tolist())
    expected = hmac.new(KEY, "\n".join(calib).encode(), hashlib.sha256).hexdigest()
    assert m.calib_digest == "hmac-sha256:" + expected
    assert m.sealed_digest == ids_hmac(ids[fa.mask(Fold.SEALED)], KEY)
    assert ids_sha256(["b", "a"]) == ids_sha256(["a", "b"])
    assert m.oof_k == 5 and m.oof_scheme == "stratified_cluster_kfold"
    assert m.amx_version


@pytest.mark.parametrize("numeric", [False, True])
def test_manifest_json_has_no_target_values_and_no_ids(numeric: bool) -> None:
    uf = make_frame(300, seed=0, numeric=numeric, dup_of={7: 1})
    spec = make_spec(seed=5, kind="numeric" if numeric else "categorical")
    fa = assign_folds(uf, spec)
    text = manifest_json(
        build_manifest(spec, uf, fa, hmac_key=KEY, oof_k=5, oof_scheme="stratified_cluster_kfold")
    )
    for uid in uf.ids.tolist():  # no calib/sealed ids, and no dev ids either
        assert uid not in text
    for value in {str(v) for v in uf.target.tolist()}:
        assert value not in text, value
    assert KEY.hex() not in text
    keys = set(json.loads(text))
    assert keys == set(SplitManifest.model_fields)


def test_temporal_manifest_records_drops_and_embargo() -> None:
    uf = make_frame(300, time=list(range(300)), dup_of={250: 3})
    spec = make_spec("temporal", time_column="t", embargo=2)
    fa = assign_folds(uf, spec)
    m = build_manifest(spec, uf, fa, hmac_key=KEY, oof_k=5, oof_scheme="forward_chaining")
    assert m.embargo_steps == 2
    assert {k.value: v for k, v in m.dropped_reasons.items()} == {
        "embargo": 4,
        "duplicate_of_dev": 1,
    }
    assert m.counts.dropped == 5


def test_round_trip_and_strictness(tmp_path: Path) -> None:
    m = manifest_for(5)
    path = write_manifest(m, tmp_path / "a" / "splits.manifest.json")
    assert read_manifest(path) == m
    assert [p.name for p in path.parent.iterdir()] == ["splits.manifest.json"]
    raw = json.loads(path.read_text())
    raw["calib_ids"] = ["unit-000001-z"]
    with pytest.raises(pydantic.ValidationError):
        SplitManifest.model_validate(raw)


def test_assignment_must_match_frame() -> None:
    uf = make_frame(100)
    spec = make_spec()
    fa = assign_folds(uf, spec)
    with pytest.raises(ValueError, match="does not match"):
        build_manifest(
            spec,
            uf.take(range(50)),
            fa,
            hmac_key=KEY,
            oof_k=5,
            oof_scheme="stratified_cluster_kfold",
        )
