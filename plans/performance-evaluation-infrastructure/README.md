# Performance Evaluation Infrastructure

> Enterprise-grade profiling suite for POWE.RS HPC application

## Overview

This plan establishes a comprehensive performance evaluation infrastructure covering CPU, memory, parallelism, and I/O profiling. The suite provides machine-readable output, interactive dashboards, and CLI tools for deep performance investigation.

## Status

| Epic | Name | Duration | Status | Points |
|------|------|----------|--------|--------|
| 1 | [Core Framework](./epic-01-core-framework/00-epic-overview.md) | 3 weeks | ⬜ | 42 |
| 2 | [CPU Profiling](./epic-02-cpu-profiling/00-epic-overview.md) | 2 weeks | ⬜ | 23 |
| 3 | [Memory Profiling](./epic-03-memory-profiling/00-epic-overview.md) | 2 weeks | ⬜ | 24 |
| 4 | [Parallelism Analysis](./epic-04-parallelism-analysis/00-epic-overview.md) | 2 weeks | ⬜ | 23 |
| 5 | [Visualization Dashboard](./epic-05-visualization-dashboard/00-epic-overview.md) | 2 weeks | ⬜ | 28 |
| 6 | [Integration & Docs](./epic-06-integration-docs/00-epic-overview.md) | 1 week | ⬜ | 18 |

**Total**: ~12 weeks, 158 story points

## Quick Links

### Master Plan
- [00-master-plan.md](./00-master-plan.md) - Architecture, goals, and strategy

### Epics
- [Epic 1: Core Framework](./epic-01-core-framework/00-epic-overview.md) - CLI, schemas, first collector
- [Epic 2: CPU Profiling](./epic-02-cpu-profiling/00-epic-overview.md) - perf, FlameGraph
- [Epic 3: Memory Profiling](./epic-03-memory-profiling/00-epic-overview.md) - DHAT, Massif, RSS
- [Epic 4: Parallelism Analysis](./epic-04-parallelism-analysis/00-epic-overview.md) - Thread scaling
- [Epic 5: Visualization Dashboard](./epic-05-visualization-dashboard/00-epic-overview.md) - Plotly dashboards
- [Epic 6: Integration & Docs](./epic-06-integration-docs/00-epic-overview.md) - Documentation, cleanup

### Sprints (Epic 1 Detail)
- [Sprint 1: Foundation](./epic-01-core-framework/sprint-01/00-sprint-overview.md)
  - [T-001: Create Python project](./epic-01-core-framework/sprint-01/ticket-001-create-python-project.md)
  - [T-002: CLI skeleton](./epic-01-core-framework/sprint-01/ticket-002-cli-skeleton.md)
  - [T-003: Data schemas](./epic-01-core-framework/sprint-01/ticket-003-data-schemas.md)
  - [T-004: System info](./epic-01-core-framework/sprint-01/ticket-004-system-info.md)
  - [T-005: Git info](./epic-01-core-framework/sprint-01/ticket-005-git-info.md)
  - [T-006: Config loading](./epic-01-core-framework/sprint-01/ticket-006-config-loading.md)
- [Sprint 2: First Collector](./epic-01-core-framework/sprint-02/00-sprint-overview.md)

## Relationship to Clean Code Refactoring

This plan creates a **new Epic 6** in the overall clean-code-refactoring effort:

| Original | New Structure |
|----------|---------------|
| Epic 5: Memory Optimization | ✅ Complete |
| Epic 6: Test Modernization | → **Epic 7** |
| Epic 7: Performance Validation | → **Epic 8** |
| *(new)* | **Epic 6: Performance Evaluation Infrastructure** |

## Key Deliverables

### CLI Tool
```bash
powers-profile run --suite full          # Full profiling
powers-profile run --collectors cpu      # CPU only
powers-profile compare v0.2.0 HEAD       # Compare versions
powers-profile summary                   # Quick summary
powers-profile dashboard                 # Interactive HTML
powers-profile scaling --threads 1,2,4,8 # Scaling analysis
```

### Output Formats
- **JSON**: Machine-readable, schema-versioned
- **Markdown**: Human-readable reports
- **HTML**: Interactive Plotly dashboards
- **SVG**: FlameGraphs

### Profiling Domains
- **CPU**: perf, FlameGraph, hotspot analysis
- **Memory**: DHAT, Massif, Cachegrind, RSS
- **Parallel**: Thread scaling, efficiency, Amdahl estimation
- **Timing**: Internal POWE.RS timing extraction

## Prerequisites

| Tool | Purpose | Installation |
|------|---------|--------------|
| Python 3.10+ | Framework | System/pyenv |
| valgrind | Memory profiling | `apt install valgrind` |
| perf | CPU profiling | `apt install linux-tools-$(uname -r)` |
| FlameGraph | SVG generation | Clone from GitHub |

## Getting Started

After Epic 1 is complete:

```bash
# Install the profiling framework
cd profiling
pip install -e .

# Verify installation
powers-profile --version

# Run first profile
powers-profile run --collectors timing
```

## Ticket Summary

| Epic | Sprint | Tickets | Points |
|------|--------|---------|--------|
| 1 | 1 | T-001 to T-006 | 19 |
| 1 | 2 | T-007 to T-013 | 23 |
| 2 | 1 | T-014 to T-019 | 23 |
| 3 | 1 | T-020 to T-025 | 24 |
| 4 | 1 | T-026 to T-032 | 23 |
| 5 | 1 | T-033 to T-040 | 28 |
| 6 | 1 | T-041 to T-047 | 18 |
| **Total** | **7 sprints** | **47 tickets** | **158 points** |

---

*Created: 2026-01-02*
*Last Updated: 2026-01-02*
