"""Run a frozen artifact on inputs only, in a subprocess (HANDOFF 8.4 step 1).

The subprocess receives a parquet file with the unit id and input columns, never a label. It
runs in a fresh temporary directory outside the vault (also its working directory and HOME),
in isolated mode (``python -I``: no user site, no PYTHON* variables, no script or current
directory on ``sys.path``), with a scrubbed environment (no freeze token, no vault path) and
single-threaded math libraries so the commit scores match a batch run bit for bit. In
``local_dir`` vault mode this is hygiene, not isolation: the process still runs as the same
user (HANDOFF 8.4).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from amx.data.unitframe import UnitFrame

KEEP_ENV = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT")
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


class ResolverError(RuntimeError):
    """The resolver subprocess failed or returned malformed output."""


def scrubbed_env(home: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in KEEP_ENV if k in os.environ}
    env.update(THREAD_ENV)
    env["HOME"] = str(home)
    env["TMPDIR"] = str(home)
    return env


def run_resolver(
    artifact_dir: Path,
    units: UnitFrame,
    *,
    expected_hash: str,
    timeout_s: float = 3600.0,
) -> pd.DataFrame:
    """Return a frame with columns unit_id, value, score in the order of ``units``."""
    roles = units.roles
    table = units.table.select([roles.unit_id, *roles.inputs])
    if roles.target is not None:
        raise ResolverError(
            "refusing to resolve units that carry a target; pass units.without_target()"
        )
    tmp = Path(tempfile.mkdtemp(prefix="amx-resolve-"))
    try:
        os.chmod(tmp, 0o700)
        # the resolver gets its own copy of the artifact, so no path it sees points into the vault
        art = tmp / "artifact"
        shutil.copytree(artifact_dir, art)
        in_path, out_path = tmp / "inputs.parquet", tmp / "out.parquet"
        pq.write_table(table, in_path)
        cmd = [
            sys.executable,
            "-I",
            "-m",
            "amx.baseline.resolve",
            str(art),
            str(in_path),
            str(out_path),
            "--expected-hash",
            expected_hash,
        ]
        proc = subprocess.run(
            cmd,
            env=scrubbed_env(tmp),
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            raise ResolverError(f"resolver failed ({proc.returncode}): {proc.stderr[-2000:]}")
        out: pd.DataFrame = pq.read_table(out_path).to_pandas()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    missing = {"unit_id", "value", "score"} - set(out.columns)
    if missing:
        raise ResolverError(f"resolver output lacks columns {sorted(missing)}")
    ids = units.ids.astype(str)
    out["unit_id"] = out["unit_id"].astype(str)
    if len(out) != len(ids) or set(out["unit_id"]) != set(ids):
        raise ResolverError("resolver output ids do not match the inputs")
    out = out.set_index("unit_id").loc[ids].reset_index()
    if not np.all(np.isfinite(out["score"].to_numpy(dtype=np.float64))):
        raise ResolverError("resolver returned non-finite commit scores")
    return out
