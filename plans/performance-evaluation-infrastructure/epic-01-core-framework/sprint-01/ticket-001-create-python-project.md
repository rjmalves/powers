# T-001: Create Python Project Structure

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Foundation](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-002, T-003, T-004, T-005, T-006

---

## Context

### Background

The performance evaluation infrastructure needs a well-structured Python project that can be installed and run as a CLI tool. This ticket creates the foundational directory structure, pyproject.toml for packaging, and initial module scaffolding.

### Current State

No Python profiling framework exists. There are standalone scripts in `scripts/` that will be replaced by this new framework.

---

## Specification

### Outputs

Create the following directory structure:

```
profiling/
├── powers_profile/              # Main package
│   ├── __init__.py             # Package init with version
│   ├── __main__.py             # Entry point for `python -m powers_profile`
│   ├── cli.py                  # CLI placeholder
│   ├── config.py               # Config placeholder
│   ├── schemas/
│   │   └── __init__.py
│   ├── collectors/
│   │   └── __init__.py
│   ├── analyzers/
│   │   └── __init__.py
│   ├── reporters/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── config/
│   └── default.toml            # Empty config template
├── tests/
│   ├── __init__.py
│   └── test_imports.py         # Verify all modules import
├── pyproject.toml              # Package configuration
└── README.md                   # Basic readme
```

### pyproject.toml Requirements

```toml
[project]
name = "powers-profile"
version = "0.1.0"
description = "Performance evaluation infrastructure for POWE.RS"
requires-python = ">=3.10"
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "toml>=0.10.0",
    "plotly>=5.18.0",
    "pandas>=2.0.0",
]

[project.scripts]
powers-profile = "powers_profile.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["powers_profile"]
```

### Behavior

- `pip install -e .` from `profiling/` directory installs the package
- `powers-profile` command becomes available (placeholder output ok)
- `python -m powers_profile` also works
- All subpackage imports work without errors

---

## Acceptance Criteria

- [ ] Directory structure created as specified
- [ ] `pip install -e .` succeeds from `profiling/` directory
- [ ] `powers-profile --version` prints version (0.1.0)
- [ ] All packages importable: `from powers_profile import *`
- [ ] `test_imports.py` passes
- [ ] README.md exists with basic description

---

## Implementation Guide

### Suggested Approach

1. Create `profiling/` directory in repository root
2. Create all subdirectories and `__init__.py` files
3. Create `pyproject.toml` with dependencies
4. Create `__main__.py` with minimal entry point
5. Create `cli.py` with placeholder Typer app
6. Create `test_imports.py` to verify structure
7. Test installation with `pip install -e .`

### Key Files to Create

- `profiling/pyproject.toml` - Package definition
- `profiling/powers_profile/__init__.py` - Version export
- `profiling/powers_profile/__main__.py` - Entry point
- `profiling/powers_profile/cli.py` - Typer app placeholder
- `profiling/tests/test_imports.py` - Import verification

### Patterns to Follow

```python
# powers_profile/__init__.py
__version__ = "0.1.0"

# powers_profile/__main__.py
from powers_profile.cli import app

if __name__ == "__main__":
    app()

# powers_profile/cli.py
import typer
from powers_profile import __version__

app = typer.Typer(
    name="powers-profile",
    help="Performance evaluation infrastructure for POWE.RS"
)

@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version")
):
    if version:
        typer.echo(f"powers-profile {__version__}")
        raise typer.Exit()
```

### Pitfalls to Avoid

- ⚠️ Don't forget `__init__.py` files in all packages
- ⚠️ Ensure `packages` in pyproject.toml matches actual package name
- ⚠️ Use `>=` for dependency versions, not exact pins

---

## Testing Requirements

### Unit Tests

- [ ] `test_imports.py`: Import all subpackages successfully
- [ ] Verify `__version__` is accessible

### Integration Tests

- [ ] `pip install -e .` completes without errors
- [ ] `powers-profile --version` outputs correct version

---

## Documentation Requirements

- [ ] Basic README.md with:
  - Project description (1-2 sentences)
  - Installation instructions
  - Basic usage placeholder

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Standard Python project scaffolding, well-understood pattern

---

## Definition of Done

- [ ] All files created
- [ ] Package installable
- [ ] Tests pass
- [ ] README exists
- [ ] Code reviewed
