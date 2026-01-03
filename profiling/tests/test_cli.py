"""CLI verification tests."""

import json
from pathlib import Path

from typer.testing import CliRunner

from powers_profile.cli import app

runner = CliRunner()


def _make_fake_binary(tmp_path: Path) -> Path:
    script = tmp_path / "fake_binary.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "print('[TIMING] total=5ms')\n",
    )
    script.chmod(0o755)
    return script


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
    result = runner.invoke(app, ["dashboard"])
    assert result.exit_code == 0
    assert "Not yet implemented" in result.stdout


def test_run_history_and_summary_flow(tmp_path: Path) -> None:
    binary = _make_fake_binary(tmp_path)
    output_dir = tmp_path / "profiling_results"

    run_result = runner.invoke(
        app,
        [
            "run",
            "--collectors",
            "timing",
            "--binary",
            str(binary),
            "--output",
            str(output_dir),
        ],
    )
    assert run_result.exit_code == 0

    history_path = output_dir / "history.json"
    history = json.loads(history_path.read_text())
    run_entry = history[0]
    run_id = run_entry["run_id"]
    run_path = Path(run_entry["path"])
    run_data = json.loads(run_path.read_text())
    assert run_data["results"]["timing"]["success"] is True

    history_result = runner.invoke(
        app,
        [
            "history",
            "--output",
            str(output_dir),
            "--format",
            "json",
        ],
    )
    assert history_result.exit_code == 0
    assert run_id in history_result.stdout

    summary_result = runner.invoke(
        app,
        [
            "summary",
            run_id,
            "--output",
            str(output_dir),
        ],
    )
    assert summary_result.exit_code == 0
    assert "Profiling Run Summary" in summary_result.stdout
