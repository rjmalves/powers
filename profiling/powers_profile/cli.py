"""Typer-based CLI skeleton for the profiling framework."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from . import __version__

app = typer.Typer(
    name="powers-profile",
    help="Performance evaluation infrastructure for POWE.RS",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _render_placeholder(title: str, lines: List[str]) -> None:
    """Render placeholder output for commands."""
    console.print(
        Panel.fit(
            "\n".join([*lines, "[yellow]Not yet implemented[/yellow]"]),
            title=title,
        )
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
    ),
) -> None:
    """
    Display version or help when no subcommand is provided.

    The main callback is kept lightweight; subcommands own business logic.
    """
    if version:
        console.print(f"powers-profile {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@app.command()
def run(
    collectors: List[str] = typer.Option(
        ["all"],
        "--collectors",
        "-c",
        help="Collectors to run (cpu, memory, parallel, timing, io, all)",
        show_default=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for profiling results",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-C",
        help="Path to profiling configuration TOML",
    ),
    binary: Optional[Path] = typer.Option(
        None,
        "--binary",
        "-b",
        help="Binary to profile",
    ),
    args: Optional[List[str]] = typer.Argument(
        None,
        help="Arguments to pass to the profiled binary",
        show_default=False,
    ),
) -> None:
    """Execute a profiling run."""
    args_list = args or []
    _render_placeholder(
        "powers-profile run",
        [
            f"Collectors: {collectors}",
            f"Output: {output or 'profiling_results'}",
            f"Config: {config or 'profiling/config/default.toml'}",
            f"Binary: {binary or 'target/release/powers'}",
            f"Args: {args_list}",
        ],
    )


@app.command()
def compare(
    baseline: str = typer.Argument(..., help="Baseline version/run ID"),
    target: str = typer.Argument("HEAD", help="Target version/run ID"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for comparison report",
    ),
) -> None:
    """Compare two profiling runs."""
    _render_placeholder(
        "powers-profile compare",
        [
            f"Baseline: {baseline}",
            f"Target: {target}",
            f"Output: {output or 'stdout'}",
        ],
    )


@app.command()
def history(
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Number of runs to show",
        show_default=True,
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json",
        show_default=True,
    ),
) -> None:
    """View profiling history."""
    _render_placeholder(
        "powers-profile history",
        [
            f"Limit: {limit}",
            f"Format: {format}",
        ],
    )


@app.command()
def summary(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID (default: latest)",
        show_default=False,
    ),
) -> None:
    """Show quick summary of a profiling run."""
    _render_placeholder(
        "powers-profile summary",
        [
            f"Run ID: {run_id or 'latest'}",
        ],
    )


@app.command()
def dashboard(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID (default: latest)",
        show_default=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output HTML file",
    ),
) -> None:
    """Generate interactive dashboard."""
    _render_placeholder(
        "powers-profile dashboard",
        [
            f"Run ID: {run_id or 'latest'}",
            f"Output: {output or 'dashboard.html'}",
        ],
    )


@app.command()
def scaling(
    threads: str = typer.Option(
        "1,2,4,8",
        "--threads",
        "-t",
        help="Comma-separated thread counts to test",
        show_default=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file",
    ),
) -> None:
    """Run parallel scaling analysis."""
    _render_placeholder(
        "powers-profile scaling",
        [
            f"Threads: {threads}",
            f"Output: {output or 'scaling.json'}",
        ],
    )
