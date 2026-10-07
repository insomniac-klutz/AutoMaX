from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from amx.spec import (
    BuiltinLoss,
    SpecError,
    TaskSpec,
    content_hash,
    dump_taskspec,
    load_taskspec,
    parse_taskspec,
    resolve_uri,
)
from amx.spec.schema import SCHEMA_PATH, taskspec_schema_text

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "specs"

BASE: dict[str, Any] = {
    "spec_version": 1,
    "name": "toy",
    "data": {
        "uri": "data.parquet",
        "format": "parquet",
        "unit_id": "id",
        "inputs": [{"name": "a", "kind": "numeric"}, {"name": "b", "kind": "categorical"}],
        "target": {"name": "y", "kind": "categorical"},
    },
    "task": {"family": "classification"},
    "bands": {"alphas": [0.01, 0.05], "policies": ["auto", "audit"], "delta": 0.1},
}


def make(**patch: Any) -> dict[str, Any]:
    raw = copy.deepcopy(BASE)
    for dotted, value in patch.items():
        node = raw
        keys = dotted.split("__")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return raw


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.yaml")), ids=lambda p: p.stem)
def test_example_specs_load(path: Path) -> None:
    spec = load_taskspec(path)
    assert spec.spec_version == 1
    again = parse_taskspec(TaskSpec.model_validate(spec.model_dump(mode="json")).model_dump())
    assert content_hash(again) == content_hash(spec)


def test_defaults_resolved() -> None:
    spec = parse_taskspec(make())
    assert spec.task.loss_name is BuiltinLoss.ZERO_ONE
    assert spec.splits.max_certify_calls == 1
    assert spec.cert.grid_size == 200 and spec.cert.start_factor == 1.25
    assert spec.embargo_steps == 0


def test_forecast_embargo_from_horizons() -> None:
    spec = load_taskspec(FIXTURES / "forecasting.yaml")
    assert spec.embargo_steps == 28 + 28


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"bands__alphas": [0.05, 0.01]}, "strictly increasing"),
        ({"bands__alphas": [0.0, 0.05]}, "in (0, 1)"),
        ({"bands__alphas": [0.01, 1.0]}, "in (0, 1)"),
        ({"bands__policies": ["auto"]}, "same length"),
        ({"bands__policies": ["audit", "auto"]}, "auto bands first"),
        ({"bands__policies": ["audit", "audit"]}, "at least one band"),
        ({"bands__delta": 0.0}, "delta"),
        ({"bands__delta": 0.6}, "delta"),
        ({"splits__fractions": {"dev": 0.6, "calib": 0.2, "sealed": 0.3}}, "sum to 1"),
        ({"splits__fractions": {"dev": 1.0, "calib": 0.0, "sealed": 0.0}}, "must be in (0, 1)"),
        ({"splits__fractions": {"dev": 0.6, "calib": 0.4}}, "Field required"),
        ({"splits__max_certify_calls": 3}, "exactly one certify call"),
        ({"splits__regime": "grouped"}, "independence_unit"),
        ({"splits__regime": "temporal"}, "time_column"),
        ({"task__family": "multilabel"}, "not built in release R1"),
        ({"task__commit_unit": "field"}, "commit_unit"),
        ({"task__loss": {"name": "err_gt_tol", "params": {"tol": 1}}}, "does not fit"),
        ({"task__loss": {"kind": "custom", "path": "l.py"}}, "both 'path' and 'fn'"),
        ({"task__loss": {"path": "l.py", "fn": "f"}}, "only valid for kind: custom"),
        ({"data__independence_unit": "group:doc"}, "neither group_columns"),
        ({"data__independence_unit": "groups"}, "'unit' or 'group:<col>'"),
        ({"data__unit_id": "y"}, "unit_id and target"),
        ({"bands__watch_slices": [{"name": "s", "by": "nope"}]}, "unknown column"),
        ({"constraints__forbidden_inputs": ["a"]}, "listed as inputs"),
        ({"cert__call_policy": "sliced"}, "not implemented in A0"),
        ({"data__group_columns": ["b"]}, "independence_unit: group:<col>"),
        ({"bands__watch_slices": [{"name": "s", "by": "id"}]}, "unit id"),
        ({"name": "Has Spaces"}, "should match pattern"),
        ({"extra_key": 1}, "Extra inputs"),
    ],
)
def test_validation_rejects(patch: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(message)):
        parse_taskspec(make(**patch))


def test_fraction_sum_tolerance() -> None:
    spec = parse_taskspec(make(splits__fractions={"dev": 0.7, "calib": 0.2, "sealed": 0.1}))
    assert spec.splits.fractions.sealed == 0.1


def test_anomaly_needs_named_loss_and_normal_label() -> None:
    raw = make(task__family="anomaly", data__target={"name": "y", "kind": "categorical"})
    with pytest.raises(ValidationError, match="no default loss"):
        parse_taskspec(raw)
    raw["task"]["loss"] = {"name": "missed_anomaly"}
    with pytest.raises(ValidationError, match="normal_label"):
        parse_taskspec(raw)
    raw["task"]["loss"]["params"] = {"normal_label": "ok"}
    assert parse_taskspec(raw).task.loss_name is BuiltinLoss.MISSED_ANOMALY


def test_grouped_and_iid_group_rules() -> None:
    raw = make(splits__regime="iid", data__group_columns=["b"], data__independence_unit="group:b")
    with pytest.raises(ValidationError, match="cannot have group_columns"):
        parse_taskspec(raw)
    raw["splits"]["regime"] = "grouped"
    assert parse_taskspec(raw).data.independence_group == "b"
    raw["splits"]["regime"] = "auto"  # auto + groups is fine once the unit is declared
    assert parse_taskspec(raw).data.independence_group == "b"


def test_target_cannot_be_a_role_column() -> None:
    with pytest.raises(ValidationError, match="target cannot also be"):
        parse_taskspec(make(data__time_column="y"))


def test_old_version_key_rejected() -> None:
    raw = make()
    raw["amx_version"] = raw.pop("spec_version")
    with pytest.raises(SpecError, match="spec_version"):
        parse_taskspec(raw)


def test_yaml_round_trip(tmp_path: Path) -> None:
    spec = load_taskspec(FIXTURES / "forecasting.yaml")
    out = tmp_path / "s.yaml"
    out.write_text(dump_taskspec(spec))
    assert content_hash(load_taskspec(out)) == content_hash(spec)


def test_resolve_uri_relative_to_spec(tmp_path: Path) -> None:
    spec = parse_taskspec(make())
    assert resolve_uri(spec, tmp_path / "spec.yaml") == (tmp_path / "data.parquet").resolve()


def test_spec_is_frozen() -> None:
    spec = parse_taskspec(make())
    with pytest.raises(ValidationError):
        spec.name = "other"  # type: ignore[misc]


def test_committed_schema_is_current() -> None:
    assert SCHEMA_PATH.read_text(encoding="utf-8") == taskspec_schema_text(), "run `make schemas`"
