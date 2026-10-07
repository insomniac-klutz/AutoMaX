"""Command-line interface (HANDOFF 5.3). Warden commands need a human-held freeze token.

Exit codes: 0 success, 1 a check or gate failed, 2 refused (bad state, missing human flag or
token, invalid spec).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from amx._log import configure

app = typer.Typer(no_args_is_help=True, add_completion=False, help="AutoMaX (amx)")
warden_app = typer.Typer(no_args_is_help=True, help="Human-side warden helpers.")
app.add_typer(warden_app, name="warden")

EXIT_FAILED = 1
EXIT_REFUSED = 2

RunOpt = Annotated[Path, typer.Option("--run", "-r", help="run directory (runs/<id>)")]
SpecOpt = Annotated[Path, typer.Option("--spec", "-s", help="TaskSpec YAML")]
TokenOpt = Annotated[
    str | None,
    typer.Option("--token", help="freeze token (or AMX_FREEZE_TOKEN); never give it to an agent"),
]
ConfirmLossOpt = Annotated[
    bool, typer.Option("--confirm-loss", help="human confirmation of a custom or non-default loss")
]


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def _refuse(exc: Exception) -> typer.Exit:
    typer.echo(f"refused: {exc}", err=True)
    return typer.Exit(EXIT_REFUSED)


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    json_logs: Annotated[bool, typer.Option("--json-logs")] = False,
) -> None:
    """AutoMaX: design and certify LLM-free pipelines."""
    configure(logging.DEBUG if verbose else logging.INFO, json_lines=json_logs)


@app.command()
def doctor(as_json: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Environment check: versions, extras, vault mode, token hygiene."""
    from amx.doctor import diagnose

    report = diagnose()
    if as_json:
        _emit(report)
    else:
        typer.echo(f"amx {report['amx_version']} on Python {report['python']}")
        typer.echo(f"vault: {report['vault']['mode']} at {report['vault']['root']}")
        for w in report["warnings"]:
            typer.echo(f"warning: {w}")
        for p in report["problems"]:
            typer.echo(f"PROBLEM: {p}")
    if not report["ok"]:
        raise typer.Exit(EXIT_FAILED)


@app.command()
def profile(
    spec: Annotated[Path | None, typer.Option("--spec", "-s")] = None,
    run: Annotated[Path | None, typer.Option("--run", "-r")] = None,
    confirm_loss: ConfirmLossOpt = False,
) -> None:
    """Label-free profile of a dataset (--spec), or dev-only feasibility of a split run (--run)."""
    from amx import runs
    from amx.loss.base import LossError
    from amx.spec.loader import SpecError

    if (spec is None) == (run is None):
        raise _refuse(ValueError("pass exactly one of --spec or --run"))
    try:
        if spec is not None:
            _emit(runs.profile_pre(spec))
            return
        assert run is not None
        feas = runs.profile_feasibility(run, confirm_loss=confirm_loss)
    except (SpecError, LossError, runs.RunError, FileNotFoundError) as exc:
        raise _refuse(exc) from exc
    _emit(feas.model_dump(mode="json"))
    if feas.requires_force:
        typer.echo(
            "some bands look infeasible; see recommendations (needs --force later)", err=True
        )


@app.command()
def split(
    spec: SpecOpt,
    run: RunOpt,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    allow_small: Annotated[bool, typer.Option("--allow-small")] = False,
) -> None:
    """Hash-locked split: dev into the run dir, calib and sealed into the vault."""
    from amx import runs
    from amx.spec.loader import SpecError
    from amx.split.errors import SplitError, VaultError

    try:
        _emit(runs.split(spec, run, run_id=run_id, allow_small=allow_small))
    except (SpecError, SplitError, VaultError, runs.RunError, ValueError) as exc:
        raise _refuse(exc) from exc


@app.command()
def baseline(run: RunOpt, confirm_loss: ConfirmLossOpt = False) -> None:
    """Fit and freeze the A0 trivial baseline on dev (replaced by the operator library in A1)."""
    from amx import runs
    from amx.baseline.artifact import ArtifactError
    from amx.loss.base import LossError

    try:
        _emit(runs.baseline(run, confirm_loss=confirm_loss))
    except (runs.RunError, LossError, ArtifactError, FileNotFoundError) as exc:
        raise _refuse(exc) from exc


@app.command()
def certify(
    run: RunOpt,
    freeze: Annotated[
        bool, typer.Option("--freeze", help="required: spends a certify call")
    ] = False,
    token: TokenOpt = None,
    confirm_loss: ConfirmLossOpt = False,
) -> None:
    """WARDEN: certify the frozen artifact on the calibration fold (spends one certify call)."""
    from amx.loss.base import LossError
    from amx.split.errors import VaultError
    from amx.warden import TokenError, WardenError, certify_run, resolve_token

    if not freeze:
        raise _refuse(ValueError("certify spends a budgeted call; pass --freeze to confirm"))
    try:
        cert = certify_run(run, token=resolve_token(token), confirm_loss=confirm_loss)
    except (TokenError, WardenError, VaultError, LossError, FileNotFoundError) as exc:
        raise _refuse(exc) from exc
    _emit(
        {
            "guarantee": cert.guarantee.type.value,
            "bands": [
                {"alpha": b.alpha, "status": b.status.value, "tau_hat": b.tau_hat}
                for b in cert.bands
            ],
            "warnings": cert.warnings,
        }
    )


@app.command()
def report(run: RunOpt) -> None:
    """Render report/bands.md and report/frontier.png from the certificate (no new numbers)."""
    from amx import runs

    try:
        _emit(runs.report(run))
    except runs.RunError as exc:
        raise _refuse(exc) from exc


@app.command()
def simulate(
    t1: Annotated[
        bool, typer.Option("--t1", help="run the T1 certificate-validity checks")
    ] = False,
    synthetic: Annotated[bool, typer.Option("--synthetic")] = False,
    real: Annotated[
        bool, typer.Option("--real", help="WARDEN: calib-only resplits of a run")
    ] = False,
    generators: Annotated[str, typer.Option(help="comma list of a,b,d")] = "a,b,d",
    n_calib: Annotated[str, typer.Option(help="comma list")] = "500,2000,10000",
    reps: Annotated[int, typer.Option()] = 2000,
    forecast: Annotated[bool, typer.Option("--forecast/--no-forecast")] = True,
    jobs: Annotated[int, typer.Option(help="worker processes")] = 1,
    out: Annotated[Path | None, typer.Option(help="directory for t1.md and t1.json")] = None,
    run: Annotated[Path | None, typer.Option("--run", "-r")] = None,
    token: TokenOpt = None,
    resplits: Annotated[int, typer.Option()] = 200,
    seed: Annotated[int, typer.Option()] = 20261007,
    confirm_loss: ConfirmLossOpt = False,
) -> None:
    """T1: synthetic oracle (--synthetic) or calib-only real-data resplits (--real, warden)."""
    if not t1 or synthetic == real:
        raise _refuse(ValueError("use --t1 with exactly one of --synthetic or --real"))
    if real:
        from amx.split.errors import VaultError
        from amx.warden import TokenError, WardenError, resolve_token, t1_real_cheap

        if run is None:
            raise _refuse(ValueError("--real needs --run"))
        try:
            payload = t1_real_cheap(
                run,
                token=resolve_token(token),
                resplits=resplits,
                seed=seed,
                confirm_loss=confirm_loss,
            )
        except (TokenError, WardenError, VaultError, FileNotFoundError) as exc:
            raise _refuse(exc) from exc
        _emit(payload)
        if not payload["passed"]:
            raise typer.Exit(EXIT_FAILED)
        return

    from amx.sim import render_t1_json, render_t1_markdown, run_t1_forecast_gate, run_t1_synth

    cells = [
        run_t1_synth(g.strip(), int(n), reps, n_jobs=jobs)
        for g in generators.split(",")
        if g.strip()
        for n in n_calib.split(",")
        if n.strip()
    ]
    gate = run_t1_forecast_gate() if forecast else None
    md, js = render_t1_markdown(cells, gate), render_t1_json(cells, gate)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "t1.md").write_text(md, encoding="utf-8")
        (out / "t1.json").write_text(js + "\n", encoding="utf-8")
    typer.echo(md)
    ok = all(c.gate_pass for c in cells) and (gate is None or gate.passed)
    if not ok:
        raise typer.Exit(EXIT_FAILED)


@warden_app.command("issue-token")
def issue_token_cmd(run: RunOpt) -> None:
    """HUMAN: issue the freeze token for a run. Shown once; keep it out of agent environments."""
    from amx.split.vault import LocalVault
    from amx.warden import issue_token

    rid = run.resolve().name
    typer.echo(issue_token(LocalVault(), rid))


def main() -> None:
    app()
