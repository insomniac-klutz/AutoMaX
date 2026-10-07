"""Run the A0 gate (ROLLER "Gate A0") and write a summary for docs/gates/A0.md.

    uv run python scripts/gate_a0.py --out runs/gate-a0 [--reps 2000] [--resplits 200]

1. T1-synth (generators a, b, d x n_calib x bands) and T1-forecast (generator c) on synthetic
   oracles: the statistical gate.
2. For each gate dataset (benchmark split seeds, OQ Q3): split, baseline, dev feasibility,
   certify (one call), report, and T1-real-cheap on calib-only resplits.

The warden commands here use a freeze token issued inside this process: the builder runs the
benchmark check and sees aggregates only (Q3 default). Release runs need a human-held token.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from amx import runs
from amx._log import configure
from amx.sim import render_t1_json, render_t1_markdown, run_t1_forecast_gate, run_t1_synth
from amx.split.vault import LocalVault
from amx.warden import certify_run, issue_token, t1_real_cheap

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("adult", "california_housing", "electricity_client")


def synthetic(out: Path, reps: int, jobs: int) -> dict[str, Any]:
    cells = [
        run_t1_synth(g, n, reps, n_jobs=jobs) for g in ("a", "b", "d") for n in (500, 2000, 10_000)
    ]
    gate = run_t1_forecast_gate()
    (out / "t1.md").write_text(render_t1_markdown(cells, gate), encoding="utf-8")
    (out / "t1.json").write_text(render_t1_json(cells, gate) + "\n", encoding="utf-8")
    return {
        "t1_synth_passed": all(c.gate_pass for c in cells),
        "t1_forecast_passed": gate.passed,
        "certified_band_at_10000": {
            c.generator: any(b.certified_share >= 0.10 for b in c.bands)
            for c in cells
            if c.n_calib == 10_000
        },
    }


def dataset(
    name: str,
    out: Path,
    tag: str,
    resplits: int,
    *,
    spec_file: str = "spec.yaml",
    force: bool = False,
) -> dict[str, Any]:
    spec = ROOT / "datasets" / name / spec_file
    run_dir = out / "runs" / f"{name}-{tag}"
    t0 = time.time()
    split = runs.split(spec, run_dir)
    base = runs.baseline(run_dir)
    feas = runs.profile_feasibility(run_dir)
    vault = LocalVault()
    token = issue_token(vault, run_dir.resolve().name)
    cert = certify_run(run_dir, token=token, force=force)
    rep = runs.report(run_dir)
    t1 = t1_real_cheap(run_dir, token=token, resplits=resplits)
    return {
        "dataset": name,
        "run_dir": str(run_dir),
        "seconds": round(time.time() - t0, 1),
        "split": split,
        "artifact_hash": base["artifact_hash"],
        "feasibility": {
            "requires_force": feas.requires_force,
            "constant_risk": feas.constant_risk,
            "baseline_floor_estimate": feas.baseline_floor_estimate,
            "warnings": feas.warnings,
            "bands": [b.model_dump(mode="json") for b in feas.bands],
        },
        "certificate": {
            "guarantee": cert.guarantee.type.value,
            "claims_certification": cert.claims_certification,
            "estimand": cert.guarantee.estimand,
            "p_value_family": cert.guarantee.p_value_family,
            "delta_per_band": cert.guarantee.delta_per_band,
            "calib_n": cert.guarantee.calib_n,
            "bands": [
                {
                    "alpha": b.alpha,
                    "status": b.status.value,
                    "stop_reason": b.stop_reason.value,
                    "tau_hat": b.tau_hat,
                    "n_min": b.n_min_required,
                    "n_committed": b.n_committed_calib,
                    "coverage": b.coverage_at_certified_tau.est,
                    "risk_calib": b.risk_calib.est,
                }
                for b in cert.bands
            ],
            "warnings": cert.warnings,
        },
        "report": rep,
        "t1_real_cheap": t1,
    }


def summary_names(datasets: str) -> list[str]:
    return [n.strip() for n in datasets.split(",") if n.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "gate-a0")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--resplits", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-synthetic", action="store_true", help="partial run; exits 2")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument(
        "--force",
        action="store_true",
        help="certify bands the dev profile marks infeasible (recorded as warnings)",
    )
    args = ap.parse_args()
    configure()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d%H%M%S")
    summary: dict[str, Any] = {"tag": tag}
    if not args.skip_synthetic:
        summary["synthetic"] = synthetic(out, args.reps, args.jobs)
    summary["datasets"] = [
        dataset(n.strip(), out, tag, args.resplits, force=args.force)
        for n in args.datasets.split(",")
        if n.strip()
    ]
    real_ok = all(
        d["t1_real_cheap"]["passed"] and not d["t1_real_cheap"]["gross_failure"]
        for d in summary["datasets"]
    )
    summary["t1_real_cheap_passed"] = real_ok
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if args.skip_synthetic or set(summary_names(args.datasets)) != set(DATASETS):
        return 2  # incomplete: not a gate verdict
    syn = summary["synthetic"]
    ok = (
        real_ok
        and syn["t1_synth_passed"]
        and syn["t1_forecast_passed"]
        and all(syn["certified_band_at_10000"].values())
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
