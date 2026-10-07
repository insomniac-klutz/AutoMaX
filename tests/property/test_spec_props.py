from __future__ import annotations

import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from amx.spec import content_hash, parse_taskspec
from amx.spec.loader import dump_taskspec
from amx.spec.models import TaskSpec


@st.composite
def specs(draw: st.DrawFn) -> dict[str, object]:
    m = draw(st.integers(1, 5))
    alphas = sorted(draw(st.sets(st.floats(1e-4, 0.5, allow_nan=False), min_size=m, max_size=m)))
    n_auto = draw(st.integers(1, m))
    dev = draw(st.floats(0.3, 0.8))
    calib = draw(st.floats(0.05, (1 - dev) * 0.9))
    return {
        "spec_version": 1,
        "name": "p",
        "data": {
            "uri": "x.parquet",
            "format": "csv",
            "unit_id": "id",
            "target": {"name": "y", "kind": "numeric"},
        },
        "task": {"family": "regression", "loss": {"params": {"tol": draw(st.floats(0.01, 5))}}},
        "bands": {
            "alphas": alphas,
            "policies": ["auto"] * n_auto + ["audit"] * (m - n_auto),
            "delta": draw(st.floats(0.01, 0.5)),
        },
        "splits": {"fractions": {"dev": dev, "calib": calib, "sealed": 1.0 - dev - calib}},
    }


@settings(max_examples=60, deadline=None)
@given(specs())
def test_yaml_round_trip_is_identity(raw: dict[str, object]) -> None:
    spec = parse_taskspec(raw)
    again = parse_taskspec(yaml.safe_load(dump_taskspec(spec)))
    assert again == spec
    assert content_hash(again) == content_hash(spec)


@settings(max_examples=30, deadline=None)
@given(specs())
def test_hash_ignores_key_order(raw: dict[str, object]) -> None:
    spec = parse_taskspec(raw)
    reordered = dict(reversed(list(spec.model_dump(mode="json").items())))
    assert content_hash(TaskSpec.model_validate(reordered)) == content_hash(spec)
