"""Command-line interface (typer)."""

from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="AutoMaX (amx)")


@app.callback()
def _root() -> None:
    """AutoMaX: design and certify LLM-free pipelines."""


def main() -> None:
    app()
