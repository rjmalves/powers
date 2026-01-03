"""Typer-based CLI skeleton for the profiling framework."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .collectors import REGISTRY, resolve_collectors
from .collectors.cpu import generate_differential_flamegraph
from .config import load_config
from .runtime import (
    append_history,
    find_run_path,
    generate_run_id,
    history_entry_from_run,
    latest_history_entry,
    load_history,
    load_run,
    save_run,
    status_from_results,
)
from .schemas import CollectorResult, ProfilingRun
from .utils import detect_git_info, detect_system_info

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


def _cli_overrides(output: Optional[Path], binary: Optional[Path]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if output:
        overrides.setdefault("general", {})["output_dir"] = str(output)
    if binary:
        overrides.setdefault("general", {})["binary"] = str(binary)
    return overrides


def _collector_names(
    requested: List[str],
    defaults: List[str],
    implemented: List[str],
) -> List[str]:
    normalized = [c.lower() for c in requested]
    if "all" in normalized:
        return implemented
    if "default" in normalized or not normalized:
        chosen = [c for c in defaults if c in implemented]
        return chosen or implemented
    return normalized


def _print_run_summary(run: ProfilingRun, run_path: Path) -> None:
    table = Table(title="Profiling Run Summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Run ID", run.run_id)
    table.add_row("Status", run.status)
    table.add_row("Collectors", ", ".join(run.collectors_run))
    table.add_row("Binary", run.binary_path)
    table.add_row("Args", " ".join(run.binary_args))
    table.add_row("Duration (s)", f"{run.total_duration_seconds:.2f}")
    table.add_row("Saved at", str(run_path))
    console.print(table)


def _load_run_by_id(run_id: str, output_dir: Path) -> tuple[ProfilingRun, Path]:
    try:
        run_path = find_run_path(output_dir, run_id)
        return load_run(run_path), run_path
    except FileNotFoundError:
        latest = latest_history_entry(output_dir)
        if run_id in {"latest", "HEAD"} and latest is not None:
            run_path = Path(latest.path)
            return load_run(run_path), run_path
        raise


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
    config_obj = load_config(config_path=config, cli_overrides=_cli_overrides(output, binary))

    git_info = detect_git_info()
    run_id = generate_run_id(git_info)

    collector_list = _collector_names(
        collectors,
        config_obj.default_collectors,
        list(REGISTRY.keys()),
    )
    resolved_collectors = resolve_collectors(collector_list, REGISTRY)
    missing_collectors = [c for c in collector_list if c not in REGISTRY]

    run_dir = config_obj.output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    system_info = detect_system_info(config_obj.binary)

    results: Dict[str, CollectorResult] = {}
    start = time.perf_counter()
    for name in resolved_collectors:
        collector = REGISTRY[name]
        result = collector.collect(
            binary=config_obj.binary,
            args=args_list,
            config=config_obj,
            run_dir=run_dir,
        )
        results[name] = result

    for name in missing_collectors:
        results[name] = CollectorResult(
            collector_name=name,
            success=False,
            duration_seconds=0.0,
            data={},
            errors=[f"Collector '{name}' is not implemented."],
            warnings=[],
        )

    total_duration = time.perf_counter() - start
    run_status = status_from_results(results)

    profiling_run = ProfilingRun(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        system_info=system_info,
        git_info=git_info,
        config=config_obj.raw,
        binary_path=str(config_obj.binary),
        binary_args=args_list,
        collectors_run=list(results.keys()),
        results=results,
        total_duration_seconds=total_duration,
        status=run_status,
    )

    run_path = save_run(profiling_run, config_obj.output_dir)
    history_entry = history_entry_from_run(profiling_run, run_path)
    append_history(config_obj.output_dir, history_entry)

    _print_run_summary(profiling_run, run_path)
    if run_status != "success":
        raise typer.Exit(code=1)


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
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Directory containing profiling_results (defaults to config general.output_dir)",
    ),
) -> None:
    """Compare two profiling runs."""
    config_obj = load_config(cli_overrides=_cli_overrides(data_dir, None))
    try:
        baseline_run, baseline_path = _load_run_by_id(baseline, config_obj.output_dir)
        target_run, target_path = _load_run_by_id(target, config_obj.output_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    diff_path: Optional[Path] = None
    cpu_result_base = baseline_run.results.get("cpu")
    cpu_result_target = target_run.results.get("cpu")

    if cpu_result_base and cpu_result_target:
        folded_base = Path(cpu_result_base.data.get("folded_path", ""))
        folded_target = Path(cpu_result_target.data.get("folded_path", ""))
        if folded_base.exists() and folded_target.exists() and config_obj.flamegraph_path:
            diff_path = output or (config_obj.output_dir / f"diff-{baseline_run.run_id}-{target_run.run_id}.svg")
            try:
                generate_differential_flamegraph(
                    baseline_folded=folded_base,
                    target_folded=folded_target,
                    flamegraph_dir=config_obj.flamegraph_path,
                    output_path=diff_path,
                    color=config_obj.flamegraph_colors,
                )
            except Exception as exc:  # pragma: no cover - defensive
                console.print(f"[yellow]Failed to generate differential flamegraph: {exc}[/yellow]")
                diff_path = None
        else:
            console.print(
                "[yellow]CPU folded stacks missing or FlameGraph path unset; skipping differential flamegraph.[/yellow]"
            )

    table = Table(title="Comparison")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Baseline", f"{baseline_run.run_id} ({baseline_run.git_info.commit_short})")
    table.add_row("Target", f"{target_run.run_id} ({target_run.git_info.commit_short})")
    if diff_path:
        table.add_row("Differential FlameGraph", str(diff_path))
    else:
        table.add_row("Differential FlameGraph", "not generated")
    console.print(table)


@app.command()
def history(
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Number of runs to show",
        show_default=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory containing run history",
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
    config_obj = load_config(cli_overrides=_cli_overrides(output, None))
    entries = load_history(config_obj.output_dir)[:limit]

    if not entries:
        console.print("[yellow]No profiling runs found.[/yellow]")
        return

    if format.lower() == "json":
        console.print_json(data=[entry.to_dict() for entry in entries])
        return

    table = Table(title=f"Last {len(entries)} profiling runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Timestamp", style="white")
    table.add_column("Git", style="white")
    table.add_column("Collectors", style="white")
    table.add_column("Status", style="white")

    for entry in entries:
        table.add_row(
            entry.run_id,
            entry.timestamp,
            f"{entry.git_branch}@{entry.git_commit[:7]}",
            ", ".join(entry.collectors_run),
            entry.status,
        )
    console.print(table)


@app.command()
def summary(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID (default: latest)",
        show_default=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory containing run history",
    ),
) -> None:
    """Show quick summary of a profiling run."""
    config_obj = load_config(cli_overrides=_cli_overrides(output, None))
    target_run_id = run_id

    if target_run_id is None:
        latest = latest_history_entry(config_obj.output_dir)
        if latest is None:
            console.print("[yellow]No profiling runs recorded yet.[/yellow]")
            raise typer.Exit(code=1)
        target_run_id = latest.run_id

    try:
        run_path = find_run_path(config_obj.output_dir, target_run_id)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    run = load_run(run_path)
    _print_run_summary(run, run_path)


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
