# Performance Evaluation Infrastructure

> Enterprise-grade profiling suite for POWE.RS HPC application

## Overview

This plan establishes a comprehensive performance evaluation infrastructure covering CPU, memory, parallelism, and I/O profiling. The suite provides machine-readable output, interactive dashboards, and CLI tools for deep performance investigation.

## Status

| Epic | Name | Duration | Status | Points |
|------|------|----------|--------|--------|
| 1 | [Core Framework](./epic-01-core-framework/00-epic-overview.md) | 3 weeks | 🟢 | 42 |
| 2 | [CPU Profiling](./epic-02-cpu-profiling/00-epic-overview.md) | 2 weeks | 🟢 | 23 |
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

### Sprints

- **Epic 1: Core Framework**
  - [Sprint 1: Foundation](./epic-01-core-framework/sprint-01/00-sprint-overview.md)
    - [T-001: Create Python project](./epic-01-core-framework/sprint-01/ticket-001-create-python-project.md)
    - [T-002: CLI skeleton](./epic-01-core-framework/sprint-01/ticket-002-cli-skeleton.md)
    - [T-003: Data schemas](./epic-01-core-framework/sprint-01/ticket-003-data-schemas.md)
    - [T-004: System info](./epic-01-core-framework/sprint-01/ticket-004-system-info.md)
    - [T-005: Git info](./epic-01-core-framework/sprint-01/ticket-005-git-info.md)
    - [T-006: Config loading](./epic-01-core-framework/sprint-01/ticket-006-config-loading.md)
  - [Sprint 2: First Collector](./epic-01-core-framework/sprint-02/00-sprint-overview.md)
    - [T-007: Collector base](./epic-01-core-framework/sprint-02/ticket-007-implement-collector-base-class.md)
    - [T-008: Timing collector](./epic-01-core-framework/sprint-02/ticket-008-implement-timing-collector.md)
    - [T-009: JSON reporter](./epic-01-core-framework/sprint-02/ticket-009-implement-json-reporter.md)
    - [T-010: Run storage/history](./epic-01-core-framework/sprint-02/ticket-010-implement-run-storage-and-history.md)
    - [T-011: Summary command](./epic-01-core-framework/sprint-02/ticket-011-implement-summary-command.md)
    - [T-012: Core module tests](./epic-01-core-framework/sprint-02/ticket-012-add-unit-tests-for-core-modules.md)
    - [T-013: Quick start README](./epic-01-core-framework/sprint-02/ticket-013-create-quick-start-readme.md)
- **Epic 2: CPU Profiling**
  - [Sprint 1: CPU Profiling](./epic-02-cpu-profiling/sprint-01/00-sprint-overview.md)
    - [T-014: Perf record wrapper](./epic-02-cpu-profiling/sprint-01/ticket-014-implement-perf-record-wrapper.md)
    - [T-015: FlameGraph integration](./epic-02-cpu-profiling/sprint-01/ticket-015-implement-flamegraph-integration.md)
    - [T-016: CPU collector](./epic-02-cpu-profiling/sprint-01/ticket-016-implement-cpu-collector.md)
    - [T-017: Perf hotspot parsing](./epic-02-cpu-profiling/sprint-01/ticket-017-parse-perf-report-for-hotspots.md)
    - [T-018: Differential flamegraph](./epic-02-cpu-profiling/sprint-01/ticket-018-implement-differential-flamegraph.md)
    - [T-019: Perf setup docs](./epic-02-cpu-profiling/sprint-01/ticket-019-document-perf-setup-and-limitations.md)
- **Epic 3: Memory Profiling**
  - [Sprint 1: Memory Tools](./epic-03-memory-profiling/sprint-01/00-sprint-overview.md)
    - [T-020: DHAT collector](./epic-03-memory-profiling/sprint-01/ticket-020-implement-dhat-collector.md)
    - [T-021: Massif collector](./epic-03-memory-profiling/sprint-01/ticket-021-implement-massif-collector.md)
    - [T-022: Cachegrind collector](./epic-03-memory-profiling/sprint-01/ticket-022-implement-cachegrind-collector.md)
    - [T-023: RSS monitor](./epic-03-memory-profiling/sprint-01/ticket-023-implement-rss-monitor.md)
    - [T-024: Unified memory collector](./epic-03-memory-profiling/sprint-01/ticket-024-create-unified-memory-collector.md)
    - [T-025: Memory comparison](./epic-03-memory-profiling/sprint-01/ticket-025-add-memory-comparison-analysis.md)
- **Epic 4: Parallelism Analysis**
  - [Sprint 1: Scalability Analysis](./epic-04-parallelism-analysis/sprint-01/00-sprint-overview.md)
    - [T-026: Scaling test runner](./epic-04-parallelism-analysis/sprint-01/ticket-026-implement-scaling-test-runner.md)
    - [T-027: Speedup/efficiency](./epic-04-parallelism-analysis/sprint-01/ticket-027-implement-speedup-efficiency-calculation.md)
    - [T-028: Amdahl estimation](./epic-04-parallelism-analysis/sprint-01/ticket-028-implement-amdahl-estimation.md)
    - [T-029: Contention detection](./epic-04-parallelism-analysis/sprint-01/ticket-029-implement-contention-detection.md)
    - [T-030: Parallel collector](./epic-04-parallelism-analysis/sprint-01/ticket-030-create-parallel-collector.md)
    - [T-031: Scaling CLI summary](./epic-04-parallelism-analysis/sprint-01/ticket-031-add-scaling-cli-summary.md)
    - [T-032: Multi-socket prep docs](./epic-04-parallelism-analysis/sprint-01/ticket-032-document-multi-socket-preparation.md)
- **Epic 5: Visualization Dashboard**
  - [Sprint 1: Visualization](./epic-05-visualization-dashboard/sprint-01/00-sprint-overview.md)
    - [T-033: Dashboard template](./epic-05-visualization-dashboard/sprint-01/ticket-033-implement-dashboard-base-template.md)
    - [T-034: Timing charts](./epic-05-visualization-dashboard/sprint-01/ticket-034-add-timing-visualization-charts.md)
    - [T-035: Memory charts](./epic-05-visualization-dashboard/sprint-01/ticket-035-add-memory-visualization-charts.md)
    - [T-036: Scaling charts](./epic-05-visualization-dashboard/sprint-01/ticket-036-add-scaling-visualization-charts.md)
    - [T-037: Embed FlameGraph](./epic-05-visualization-dashboard/sprint-01/ticket-037-embed-flamegraph-in-dashboard.md)
    - [T-038: Markdown reports](./epic-05-visualization-dashboard/sprint-01/ticket-038-implement-markdown-report-generator.md)
    - [T-039: Rich CLI summary](./epic-05-visualization-dashboard/sprint-01/ticket-039-implement-rich-cli-summary.md)
    - [T-040: Comparison dashboard](./epic-05-visualization-dashboard/sprint-01/ticket-040-add-comparison-dashboard-view.md)
- **Epic 6: Integration & Docs**
  - [Sprint 1: Integration](./epic-06-integration-docs/sprint-01/00-sprint-overview.md)
    - [T-041: Quick Start guide](./epic-06-integration-docs/sprint-01/ticket-041-write-quick-start-guide.md)
    - [T-042: Tools Reference](./epic-06-integration-docs/sprint-01/ticket-042-write-tools-reference.md)
    - [T-043: Analysis Guide](./epic-06-integration-docs/sprint-01/ticket-043-write-analysis-guide.md)
    - [T-044: v0.2.0 baseline](./epic-06-integration-docs/sprint-01/ticket-044-establish-v0-2-0-baseline.md)
    - [T-045: Remove old scripts](./epic-06-integration-docs/sprint-01/ticket-045-remove-old-scripts.md)
    - [T-046: Update clean-code plan](./epic-06-integration-docs/sprint-01/ticket-046-update-clean-code-refactoring-plan.md)
    - [T-047: End-to-end validation](./epic-06-integration-docs/sprint-01/ticket-047-end-to-end-validation.md)

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
*Last Updated: 2026-01-03*
