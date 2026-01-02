# Epic 1: Core Profiling Framework

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 3 weeks (2 sprints)
> **Status**: ⬜ Not Started

---

## Summary

This epic establishes the foundational infrastructure for the performance evaluation suite. It creates the Python-based CLI framework, defines data schemas for all profiling output, implements the core collector/analyzer/reporter abstractions, and delivers the first working collector (basic timing extraction).

By the end of this epic, developers can run `powers-profile run --collectors timing` and get a structured JSON output with system information, git context, and timing data.

---

## Scope

### Included

1. **CLI Framework**
   - Python CLI using Typer
   - Subcommands: `run`, `compare`, `history`, `summary`, `dashboard`
   - Configuration file support (TOML)
   - Rich terminal output formatting

2. **Data Schemas**
   - `ProfilingRun` - Complete profiling session
   - `SystemInfo` - Hardware/OS detection
   - `CollectorResult` - Per-collector output
   - `AnalysisResult` - Analysis output
   - JSON schema definitions

3. **Core Abstractions**
   - `Collector` base class and interface
   - `Analyzer` base class and interface
   - `Reporter` base class and interface
   - Plugin discovery mechanism

4. **Utility Modules**
   - Git information extraction
   - System hardware detection
   - Timestamp and versioning

5. **First Collector: Timing**
   - Extract timing data from POWE.RS output
   - Parse log output for timing information
   - Structure as JSON

6. **Output Directory Structure**
   - `profiling_results/runs/` organization
   - `profiling_results/baselines/` for committed baselines
   - `history.json` index file

### Excluded

- CPU profiling (Epic 2)
- Memory profiling (Epic 3)
- Parallelism analysis (Epic 4)
- Visualization dashboards (Epic 5)
- Comprehensive documentation (Epic 6)

---

## Dependencies

- **Requires**: None (first epic)
- **Enables**: All subsequent epics

---

## Acceptance Criteria

- [ ] `powers-profile --help` shows all subcommands
- [ ] `powers-profile run --collectors timing` executes successfully
- [ ] Output JSON matches defined schema
- [ ] System info correctly detected (CPU, RAM, OS, cores)
- [ ] Git commit/branch correctly extracted
- [ ] Runs are stored in `profiling_results/runs/`
- [ ] `powers-profile summary` shows last run
- [ ] `powers-profile history` lists past runs
- [ ] Unit tests for all core modules
- [ ] README with quick start instructions

---

## Technical Approach

### Project Structure

```
profiling/
├── powers_profile/              # Python package
│   ├── __init__.py
│   ├── __main__.py             # Entry point
│   ├── cli.py                  # Typer CLI definitions
│   ├── config.py               # Configuration loading
│   ├── schemas/                # Data schemas
│   │   ├── __init__.py
│   │   ├── run.py              # ProfilingRun
│   │   ├── system.py           # SystemInfo
│   │   └── results.py          # CollectorResult, AnalysisResult
│   ├── collectors/             # Collector implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Collector ABC
│   │   └── timing.py           # First collector
│   ├── analyzers/              # Analyzer implementations
│   │   ├── __init__.py
│   │   └── base.py             # Analyzer ABC
│   ├── reporters/              # Reporter implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Reporter ABC
│   │   └── json_export.py      # JSON export
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── git_info.py
│       ├── system_info.py
│       └── paths.py
├── config/
│   └── default.toml
├── tests/
│   └── ...
├── pyproject.toml
└── README.md
```

### Key Decisions

1. **Package Structure**: Standard Python package with `pyproject.toml`
2. **CLI Framework**: Typer for modern CLI with type hints
3. **Output Format**: JSON with optional pretty-printing
4. **Discovery**: Collectors auto-discovered from `collectors/` directory

---

## Sprints

### [Sprint 1: Foundation](./sprint-01/00-sprint-overview.md)

| ID | Title | Points | Status |
|----|-------|--------|--------|
| T-001 | Create Python project structure | 3 | ⬜ |
| T-002 | Implement CLI skeleton with Typer | 3 | ⬜ |
| T-003 | Define core data schemas | 5 | ⬜ |
| T-004 | Implement system info detection | 3 | ⬜ |
| T-005 | Implement git info extraction | 2 | ⬜ |
| T-006 | Create configuration loading | 3 | ⬜ |

**Sprint Points**: 19

### [Sprint 2: First Collector](./sprint-02/00-sprint-overview.md)

| ID | Title | Points | Status |
|----|-------|--------|--------|
| T-007 | Implement Collector base class | 3 | ⬜ |
| T-008 | Implement timing collector | 5 | ⬜ |
| T-009 | Implement JSON reporter | 3 | ⬜ |
| T-010 | Implement run storage and history | 3 | ⬜ |
| T-011 | Implement summary command | 2 | ⬜ |
| T-012 | Add unit tests for core modules | 5 | ⬜ |
| T-013 | Create quick start README | 2 | ⬜ |

**Sprint Points**: 23

---

## Estimated Effort

- **Duration**: 2 sprints (3 weeks)
- **Story Points**: 42
- **Risk Level**: Low (greenfield Python project)

---

## Definition of Done

- [ ] All tickets complete
- [ ] `powers-profile run` works end-to-end
- [ ] JSON output validated against schema
- [ ] Unit test coverage > 80%
- [ ] README with installation and usage
- [ ] Code reviewed and merged
