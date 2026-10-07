"""Load and resolve TaskSpecs from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from amx.spec.models import TaskSpec


class SpecError(ValueError):
    """A TaskSpec file could not be read or validated."""


def parse_taskspec(raw: dict[str, Any]) -> TaskSpec:
    if not isinstance(raw, dict):
        raise SpecError("a TaskSpec must be a YAML mapping")
    if "amx_version" in raw:
        raise SpecError("'amx_version' was renamed 'spec_version' (decision D19)")
    return TaskSpec.model_validate(raw)


def load_taskspec(path: str | Path) -> TaskSpec:
    """Read and validate a TaskSpec YAML file."""
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SpecError(f"cannot read {p}: {exc}") from exc
    return parse_taskspec(raw)


def dump_taskspec(spec: TaskSpec) -> str:
    """YAML form of a spec (round-trips through ``parse_taskspec``)."""
    return yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False)


def resolve_uri(spec: TaskSpec, spec_path: str | Path) -> Path:
    """Resolve ``data.uri`` relative to the spec file's directory."""
    uri = Path(spec.data.uri).expanduser()
    if uri.is_absolute():
        return uri
    return (Path(spec_path).resolve().parent / uri).resolve()
