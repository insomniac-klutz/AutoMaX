"""JSON-schema export of the TaskSpec (``python -m amx.spec.schema --write|--check``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from amx.spec.models import TaskSpec

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "taskspec.schema.json"


def taskspec_schema_text() -> str:
    return json.dumps(TaskSpec.model_json_schema(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    text = taskspec_schema_text()
    if "--write" in argv:
        SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCHEMA_PATH.write_text(text, encoding="utf-8")
        return 0
    if "--check" in argv:
        current = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else ""
        return 0 if current == text else 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
