"""Gate dataset loaders against their pinned mirrors (network; make gate-data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from amx.data.io import load_unitframe
from amx.spec import load_taskspec

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.network


@pytest.mark.parametrize(
    ("name", "n_units"),
    [("adult", 48_842), ("california_housing", 20_640)],
)
def test_tabular_loaders(name: str, n_units: int) -> None:
    spec_path = ROOT / "datasets" / name / "spec.yaml"
    uf = load_unitframe(load_taskspec(spec_path), spec_path)
    assert uf.n == n_units


def test_california_cap_present() -> None:
    spec_path = ROOT / "datasets" / "california_housing" / "spec.yaml"
    uf = load_unitframe(load_taskspec(spec_path), spec_path)
    assert int(np.sum(uf.target == 500_001.0)) == 965


def test_series_loader_builds_forecast_units() -> None:
    spec_path = ROOT / "datasets" / "electricity_client" / "spec.yaml"
    uf = load_unitframe(load_taskspec(spec_path), spec_path)
    horizons = set(uf.column("horizon").tolist())
    assert horizons == {1, 24}
    assert uf.n > 50_000
