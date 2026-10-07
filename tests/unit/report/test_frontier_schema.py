from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from amx.cert import Certificate, GuaranteeType
from amx.report import plot_frontier
from amx.report import schema as report_schema
from amx.report.frontier import frontier_title
from tests.unit.report.conftest import with_type

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize(
    "gtype", [GuaranteeType.PAC_HIGH_PROB, GuaranteeType.NONE], ids=lambda t: t.value
)
def test_frontier_png_written(gtype: GuaranteeType, cert: Certificate, tmp_path: Path) -> None:
    out = plot_frontier(with_type(cert, gtype), tmp_path / "report" / "frontier.png")
    assert out == tmp_path / "report" / "frontier.png"
    data = out.read_bytes()
    assert data.startswith(PNG_MAGIC)
    assert len(data) > 10_000


def test_frontier_without_any_threshold(cert: Certificate, tmp_path: Path) -> None:
    bands = [b.model_copy(update={"tau_hat": None}) for b in cert.bands]
    out = plot_frontier(cert.model_copy(update={"bands": bands}), tmp_path / "f.png")
    assert out.read_bytes().startswith(PNG_MAGIC)
    empty = cert.model_copy(update={"bands": bands, "frontier_descriptive": []})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert plot_frontier(empty, tmp_path / "e.png").read_bytes().startswith(PNG_MAGIC)


@pytest.mark.parametrize("gtype", list(GuaranteeType), ids=lambda t: t.value)
def test_frontier_title_is_descriptive(gtype: GuaranteeType, cert: Certificate) -> None:
    title = frontier_title(with_type(cert, gtype))
    assert "descriptive" in title
    assert "certified" not in title.lower()


def test_committed_bands_schema_is_current() -> None:
    assert report_schema.SCHEMA_PATH.name == "bands.schema.json"
    committed = report_schema.SCHEMA_PATH.read_text(encoding="utf-8")
    assert committed == report_schema.bands_schema_text(), (
        "run `python -m amx.report.schema --write`"
    )
    assert report_schema.main(["--check"]) == 0


def test_schema_check_detects_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "bands.schema.json"
    target.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(report_schema, "SCHEMA_PATH", target)
    assert report_schema.main(["--check"]) == 1
    assert report_schema.main(["--write"]) == 0
    assert report_schema.main(["--check"]) == 0
    assert target.read_text(encoding="utf-8") == report_schema.bands_schema_text()
