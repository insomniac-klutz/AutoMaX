"""Domain-blind lint (HANDOFF 12 T4, D12, D17).

No dataset name, dataset column name or listed industry word may appear in src/. Dataset terms
come from datasets/*/spec.yaml and the loaders' column constants; amx's own generic column names
(the forecast-unit columns it builds itself) are allowed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCANNED = [ROOT / "src"]
# Generic names amx itself defines or statistics vocabulary that datasets happen to reuse.
ALLOWED = {
    "class",
    "target",
    "value",
    "row_id",
    "unit_id",
    "anchor",
    "vol",
    "horizon",
    "origin_time",
    "origin",
    "population",
    "by_target",
    # ordinary English that coincides with a dataset term: HTTP client, race condition
    "client",
    "race",
}


def _blocklist() -> set[str]:
    words = set()
    for line in (ROOT / "docs" / "domain_blocklist.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line.lower())
    return words


def _dataset_terms() -> set[str]:
    terms: set[str] = set()
    for spec_path in sorted((ROOT / "datasets").glob("*/spec.yaml")):
        terms.update(spec_path.parent.name.lower().split("_"))
        raw = yaml.safe_load(spec_path.read_text())
        terms.update(raw["name"].lower().split("-"))
        data = raw["data"]
        terms.add(data["target"]["name"].lower())
        for inp in data.get("inputs", []) if isinstance(data.get("inputs"), list) else []:
            terms.add(inp["name"].lower())
        for ws in raw.get("bands", {}).get("watch_slices", []):
            terms.add(ws["by"].lower())
            terms.add(ws["name"].lower())
        loader = spec_path.parent / "loader.py"
        tree = ast.parse(loader.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "COLUMNS" for t in node.targets
            ):
                terms.update(ast.literal_eval(node.value))
    return {t.lower() for t in terms if len(t) >= 3} - ALLOWED


def test_terms_are_collected() -> None:
    terms = _dataset_terms()
    assert {"adult", "median_house_value", "median_income"} <= terms


def test_src_is_domain_blind() -> None:
    terms = _dataset_terms() | _blocklist()
    hits: list[str] = []
    for base in SCANNED:
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for term in terms:
                if re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", text):
                    hits.append(f"{path.relative_to(ROOT)}: {term}")
    assert not hits, "domain strings in src/:\n" + "\n".join(sorted(hits))
