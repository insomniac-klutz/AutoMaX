from typer.testing import CliRunner

import amx
from amx.cli import app


def test_version_string() -> None:
    assert isinstance(amx.__version__, str)


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
