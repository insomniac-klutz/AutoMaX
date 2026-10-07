"""JSON-schema export of the certificate (``python -m amx.report.schema --write|--check``).

``docs/schemas/bands.schema.json`` is generated from :class:`amx.cert.Certificate`
(HANDOFF 6.5 as amended by C2, C3, C7, C17). ``--check`` exits 1 when the committed file has
drifted from the model; a unit test runs the same comparison.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from amx._log import configure, get_logger
from amx.cert.certificate import Certificate

log = get_logger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "bands.schema.json"


def bands_schema_text() -> str:
    return json.dumps(Certificate.model_json_schema(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    text = bands_schema_text()
    if "--write" in argv:
        SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCHEMA_PATH.write_text(text, encoding="utf-8")
        return 0
    if "--check" in argv:
        current = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.exists() else ""
        if current != text:
            configure()
            log.error("%s is out of date; run: python -m amx.report.schema --write", SCHEMA_PATH)
            return 1
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
