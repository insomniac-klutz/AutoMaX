"""Resolver subprocess for the A0 trivial artifact (HANDOFF 5.2, 8.4 step 1; ROLLER step 12).

    python -m amx.baseline.resolve ARTIFACT_DIR INPUTS_PARQUET OUT_PARQUET [--expected-hash H]

The warden runs this as a subprocess that receives **inputs only**. It loads the artifact
(verifying its hash), reads the inputs parquet, and writes ``unit_id`` (string), ``value``
and ``score`` (float64, the commit score s(x), C4). An inputs file with no rows gives an
output file with no rows. It refuses, with exit code 2, any input file that has a column
named like the artifact's target, or that lacks the unit id or an input column.

Exit codes: 0 success; 2 refused inputs; 3 artifact missing or hash mismatch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from amx._log import configure, get_logger
from amx.baseline.artifact import ArtifactError, FrozenArtifact, load_artifact
from amx.data.unitframe import Roles, UnitFrame

log = get_logger(__name__)

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_ARTIFACT = 3
OUTPUT_COLUMNS = ("unit_id", "value", "score")


class InputRefusedError(ValueError):
    """The inputs table carries a target column or lacks a required column."""


def check_inputs(table: pa.Table, art: FrozenArtifact) -> Roles:
    """Roles for resolving ``table``; refuses a target column or a missing input."""
    roles = art.meta["roles"]
    target = art.target_name
    if target is not None and target in table.column_names:
        raise InputRefusedError(
            f"the inputs contain a column named '{target}', the artifact's target; "
            "the resolver accepts inputs only"
        )
    unit_id = str(roles["unit_id"])
    inputs = tuple(str(c) for c in roles["inputs"])
    missing = [c for c in (unit_id, *inputs) if c not in table.column_names]
    if missing:
        raise InputRefusedError(f"the inputs lack required columns: {missing}")
    return Roles(unit_id=unit_id, target=None, inputs=inputs)


def resolve_table(art: FrozenArtifact, table: pa.Table) -> pd.DataFrame:
    """Values and commit scores for every unit of an inputs-only table."""
    roles = check_inputs(table, art)
    uf = UnitFrame(table, roles)
    return pd.DataFrame(
        {
            "unit_id": uf.ids,
            "value": art.predictor.predict(uf),
            "score": art.predictor.commit_score(uf),
        }
    )


def output_table(resolved: pd.DataFrame) -> pa.Table:
    """The output parquet table: ``unit_id`` as string and ``score`` as float64, also when
    the batch is empty. ``value`` keeps the type of the predictions; an empty batch of object
    predictions (e.g. string class labels) has no value to infer it from and gets Arrow's
    null type."""
    return pa.table(
        {
            "unit_id": pa.array(resolved["unit_id"].to_numpy(dtype=object), type=pa.string()),
            "value": pa.array(resolved["value"].to_numpy()),
            "score": pa.array(resolved["score"].to_numpy(dtype=float), type=pa.float64()),
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m amx.baseline.resolve",
        description="Resolve an inputs-only parquet with a frozen A0 trivial artifact.",
    )
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("inputs", type=Path, help="inputs-only parquet (no target column)")
    parser.add_argument("out", type=Path, help="output parquet: unit_id, value, score")
    parser.add_argument("--expected-hash", default=None, help="refuse any other artifact hash")
    args = parser.parse_args(argv)
    configure()
    try:
        art = load_artifact(args.artifact_dir, expected_hash=args.expected_hash)
    except ArtifactError as exc:
        log.error("artifact refused: %s", exc)
        return EXIT_ARTIFACT
    table = pq.read_table(args.inputs)
    try:
        out = resolve_table(art, table)
    except InputRefusedError as exc:
        log.error("inputs refused: %s", exc)
        return EXIT_REFUSED
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(output_table(out), args.out)
    log.info("resolved %d units with artifact %s", len(out), art.artifact_hash)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
