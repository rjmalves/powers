"""CLI skeleton verification tests."""

from typer.testing import CliRunner

from powers_profile.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["run", "compare", "history", "summary", "dashboard", "scaling"]:
        assert cmd in result.stdout


def test_run_command_help() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "collectors" in result.stdout.lower()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_command_execution_placeholders() -> None:
    result = runner.invoke(app, ["summary"])
    assert result.exit_code == 0
    assert "Not yet implemented" in result.stdout
