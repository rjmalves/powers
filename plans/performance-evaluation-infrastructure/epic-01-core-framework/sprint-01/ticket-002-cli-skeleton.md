# T-002: Implement CLI Skeleton with Typer

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Foundation](./00-sprint-overview.md)
> **Dependencies**: T-001
> **Blocks**: T-007, T-008

---

## Context

### Background

The CLI is the primary interface for the profiling framework. This ticket implements the full CLI skeleton with all planned subcommands as placeholders, ready to be filled in by subsequent tickets.

### Current State

T-001 created a minimal CLI with just `--version`. This ticket expands it to the full command structure.

---

## Specification

### CLI Structure

```
powers-profile
├── run           # Execute profiling run
├── compare       # Compare two runs/versions
├── history       # View profiling history
├── summary       # Quick summary of last run
├── dashboard     # Generate interactive dashboard
└── scaling       # Run scaling analysis
```

### Command Signatures

```python
@app.command()
def run(
    collectors: List[str] = typer.Option(["all"], help="Collectors to run"),
    output: Path = typer.Option(None, help="Output directory"),
    config: Path = typer.Option(None, help="Config file path"),
    binary: Path = typer.Option(None, help="Binary to profile"),
    args: List[str] = typer.Argument(None, help="Arguments to pass to binary"),
):
    """Execute a profiling run."""
    pass

@app.command()
def compare(
    baseline: str = typer.Argument(..., help="Baseline version/run ID"),
    target: str = typer.Argument("HEAD", help="Target version/run ID"),
    output: Path = typer.Option(None, help="Output file"),
):
    """Compare two profiling runs."""
    pass

@app.command()
def history(
    limit: int = typer.Option(20, help="Number of runs to show"),
    format: str = typer.Option("table", help="Output format: table, json"),
):
    """View profiling history."""
    pass

@app.command()
def summary(
    run_id: str = typer.Argument(None, help="Run ID (default: latest)"),
):
    """Show quick summary of a profiling run."""
    pass

@app.command()
def dashboard(
    run_id: str = typer.Argument(None, help="Run ID (default: latest)"),
    output: Path = typer.Option(None, help="Output HTML file"),
):
    """Generate interactive dashboard."""
    pass

@app.command()
def scaling(
    threads: str = typer.Option("1,2,4,8", help="Thread counts to test"),
    output: Path = typer.Option(None, help="Output file"),
):
    """Run parallel scaling analysis."""
    pass
```

### Output Behavior

Each command should:
1. Print a placeholder message indicating the command was recognized
2. Echo back the provided options (for verification)
3. Exit cleanly with code 0

Example:
```bash
$ powers-profile run --collectors cpu,memory
[powers-profile] Command: run
[powers-profile] Collectors: ['cpu', 'memory']
[powers-profile] Output: None
[powers-profile] (Not yet implemented)
```

---

## Acceptance Criteria

- [ ] All 6 subcommands registered: run, compare, history, summary, dashboard, scaling
- [ ] `powers-profile --help` shows all subcommands with descriptions
- [ ] `powers-profile run --help` shows all options
- [ ] Each command accepts its specified arguments
- [ ] Invalid commands produce helpful error messages
- [ ] Rich formatting used for output (colors, panels)

---

## Implementation Guide

### Suggested Approach

1. Import typer and rich in `cli.py`
2. Define the main Typer app with metadata
3. Add each command as a decorated function
4. Use `typer.echo()` with Rich formatting for output
5. Implement placeholder logic that echoes inputs

### Key Files to Modify

- `profiling/powers_profile/cli.py` - All CLI logic

### Patterns to Follow

```python
from typing import List, Optional
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="powers-profile",
    help="Performance evaluation infrastructure for POWE.RS",
    add_completion=False,
)
console = Console()

@app.command()
def run(
    collectors: List[str] = typer.Option(
        ["all"],
        "--collectors", "-c",
        help="Collectors to run (cpu, memory, parallel, timing, io, all)"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for profiling results"
    ),
):
    """Execute a profiling run with specified collectors."""
    console.print(Panel.fit(
        f"[bold]Profiling Run[/bold]\n"
        f"Collectors: {collectors}\n"
        f"Output: {output or 'default'}",
        title="powers-profile run"
    ))
    console.print("[yellow]Not yet implemented[/yellow]")
```

### Pitfalls to Avoid

- ⚠️ Use `List[str]` not `list[str]` for Python 3.10 compatibility
- ⚠️ Typer Option defaults must be provided for optional args
- ⚠️ Use `Optional[Path]` not `Path | None` for older Python support

---

## Testing Requirements

### Unit Tests

- [ ] Test each command can be invoked programmatically
- [ ] Test `--help` output contains expected text
- [ ] Test invalid subcommand produces error

### Integration Tests

- [ ] Run each command from shell, verify exit code 0
- [ ] Verify `--help` for each subcommand

```python
# tests/test_cli.py
from typer.testing import CliRunner
from powers_profile.cli import app

runner = CliRunner()

def test_run_command_exists():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "collectors" in result.stdout.lower()

def test_all_commands_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["run", "compare", "history", "summary", "dashboard", "scaling"]:
        assert cmd in result.stdout
```

---

## Documentation Requirements

- [ ] Docstrings on all command functions
- [ ] Update README with command overview

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Typer CLI boilerplate, well-documented library

---

## Definition of Done

- [ ] All 6 commands implemented as placeholders
- [ ] Help text complete
- [ ] Tests pass
- [ ] Code reviewed
