# Master Plan: Enterprise-Grade Performance Evaluation Infrastructure

## Progress Tracking

| Epic | Name | Status | Duration |
|------|------|--------|----------|
| 1 | Core Profiling Framework | 🟢 Completed | 3 weeks |
| 2 | CPU & Execution Profiling | 🟢 Completed | 2 weeks |
| 3 | Memory Profiling Suite | 🟡 In Progress | 2 weeks |
| 4 | Parallelism & Scalability Analysis | ⬜ Not Started | 2 weeks |
| 5 | Visualization & Reporting Dashboard | ⬜ Not Started | 2 weeks |
| 6 | Integration & Documentation | ⬜ Not Started | 1 week |

**Total Duration**: ~12 weeks

## Progress Update (2026-01-03)

- Kicked off **Epic 3 Sprint 1 (Memory Tools)** with priority on T-020 (DHAT) and T-023 (RSS) to unblock unified memory aggregation.
- Preparing valgrind 3.18 fixtures (DHAT/Massif/Cachegrind) and aligning the `memory_data` JSON contract ahead of T-024 orchestration.
- Next focus: lock config defaults (output paths, polling cadence) and finalize owner assignments before implementation starts.

---

## Executive Summary

This master plan establishes an enterprise-grade performance evaluation infrastructure for the POWE.RS HPC application. The suite will provide comprehensive profiling capabilities across CPU, memory, parallelism, and I/O domains, with machine-readable output formats for programmatic comparison across versions, interactive web dashboards for analysis, and CLI tools for quick insights. The infrastructure is designed to scale from WSL2 development to 192-core AWS c7a.48xlarge production testing.

---

## Goals & Non-Goals

### Goals

1. **Comprehensive Profiling**: Cover CPU (FlameGraphs, perf), Memory (DHAT, Massif, RSS, Cachegrind), Parallelism (thread scaling, contention), I/O, and HiGHS solver timing
2. **Machine-Readable Output**: JSON/CSV formats for all profiling data enabling programmatic comparison
3. **Time-Series Tracking**: Historical performance data indexed by git commit/version
4. **Interactive Dashboards**: Plotly-based web visualization for deep analysis
5. **CLI Summaries**: Quick terminal-based performance overviews
6. **Scalability Testing**: Support profiling from 1 to 192 cores
7. **Reproducible Benchmarks**: Deterministic, documented profiling workflows

### Non-Goals (Explicit Scope Exclusions)

1. **Continuous Profiling in Production**: No always-on profiling daemons
2. **CI Integration**: Manual profiling only (CI integration is future work)
3. **Third-party Rust Crate Dependencies**: Use Linux tools + Python, not cargo-flamegraph etc.
4. **GPU Profiling**: Not currently planned
5. **Windows/macOS Support**: Linux-only (bare-metal and WSL2)

---

## Architecture Overview

### Current State

The existing profiling infrastructure consists of:
- `scripts/monitor_rss.py` - RSS monitoring during execution
- `scripts/plot_rss.py` - matplotlib-based RSS plotting
- `scripts/compare_allocators.sh` - Allocator comparison script
- `benches/sddp_e2e.rs` - Criterion benchmarks
- Ad-hoc valgrind/DHAT usage documented in `docs/`
- Baseline profiling results in `profiling_results/`

**Limitations**:
- No unified framework or CLI
- No machine-readable output standards
- No historical tracking
- Limited visualization capabilities
- Scripts are purpose-specific, not composable

### Target State

```
powers/
├── profiling/                      # New profiling framework
│   ├── powers-profile              # Main CLI entry point (Python)
│   ├── config/
│   │   ├── default.toml            # Default profiling configuration
│   │   └── benchmarks.toml         # Benchmark definitions
│   ├── collectors/                 # Data collection modules
│   │   ├── cpu.py                  # perf, flamegraph collection
│   │   ├── memory.py               # DHAT, Massif, RSS, Cachegrind
│   │   ├── parallel.py             # Thread scaling, contention
│   │   ├── timing.py               # Internal timing extraction
│   │   └── io.py                   # I/O profiling
│   ├── analyzers/                  # Analysis modules
│   │   ├── regression.py           # Performance regression detection
│   │   ├── comparison.py           # Version-to-version comparison
│   │   ├── hotspot.py              # Hotspot identification
│   │   └── scaling.py              # Parallel scaling analysis
│   ├── reporters/                  # Output generation
│   │   ├── markdown.py             # Human-readable reports
│   │   ├── json_export.py          # Machine-readable export
│   │   └── dashboard.py            # Plotly dashboard generation
│   ├── visualizers/                # Visualization modules
│   │   ├── flamegraph.py           # FlameGraph SVG generation
│   │   ├── timeline.py             # Time-series plots
│   │   ├── scaling_chart.py        # Parallel efficiency charts
│   │   └── memory_profile.py       # Memory visualization
│   └── utils/                      # Shared utilities
│       ├── git_info.py             # Git commit/version extraction
│       ├── system_info.py          # Hardware detection
│       └── data_formats.py         # Schema definitions
│
├── profiling_results/              # Output directory (gitignored except baselines)
│   ├── baselines/                  # Committed baseline profiles
│   │   └── v0.2.0/
│   ├── runs/                       # Individual profiling runs
│   │   └── 2026-01-02_abc1234/
│   └── history.json                # Time-series index
│
└── docs/profiling/                 # Profiling documentation
    ├── QUICK_START.md
    ├── TOOLS_REFERENCE.md
    └── ANALYSIS_GUIDE.md
```

### Key Design Decisions

1. **Python-Based Framework**: Python for orchestration, visualization (Plotly), and analysis
   - *Rationale*: Ecosystem maturity, Plotly support, easy scripting, no Rust compilation overhead

2. **Linux Tool Wrapping**: Wrap perf, valgrind, FlameGraph scripts rather than using Rust crates
   - *Rationale*: Standard tools, no dependency management, proven reliability

3. **JSON as Canonical Format**: All profiling data stored as JSON with defined schemas
   - *Rationale*: Universal compatibility, easy diffing, Python/JS consumption

4. **Git-Indexed History**: Profiling runs indexed by git commit SHA
   - *Rationale*: Enables version-to-version comparison, bisection

5. **Modular Collectors**: Each profiling domain is a separate module
   - *Rationale*: Run only what you need, easy extension

6. **Plotly Dashboards**: Interactive HTML dashboards for analysis
   - *Rationale*: No server needed, sharable, interactive exploration

---

## Technical Approach

### Core Abstractions

#### 1. `ProfilingRun` - Single Profiling Session

```python
@dataclass
class ProfilingRun:
    run_id: str                    # UUID or timestamp
    git_commit: str                # Git SHA
    git_branch: str                # Branch name
    timestamp: datetime            # When run started
    system_info: SystemInfo        # Hardware/OS details
    config: ProfilingConfig        # What was profiled
    results: Dict[str, Any]        # Collected data by domain
    
    def to_json(self) -> str: ...
    def compare_to(self, other: 'ProfilingRun') -> Comparison: ...
```

#### 2. `Collector` - Data Collection Interface

```python
class Collector(ABC):
    @abstractmethod
    def collect(self, binary: Path, args: List[str], config: dict) -> CollectorResult:
        """Run the target and collect profiling data."""
        pass
    
    @abstractmethod
    def parse_output(self, raw_output: Path) -> dict:
        """Parse tool-specific output into structured data."""
        pass
```

#### 3. `Analyzer` - Analysis Interface

```python
class Analyzer(ABC):
    @abstractmethod
    def analyze(self, runs: List[ProfilingRun]) -> AnalysisResult:
        """Analyze one or more profiling runs."""
        pass
```

#### 4. `Reporter` - Output Generation Interface

```python
class Reporter(ABC):
    @abstractmethod
    def generate(self, analysis: AnalysisResult, output_dir: Path) -> Path:
        """Generate report from analysis results."""
        pass
```

### Data Flow

```
User invokes: powers-profile run --suite full
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. COLLECTION PHASE                                          │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ CPU      │  │ Memory   │  │ Parallel │  │ Timing   │     │
│  │ Collector│  │ Collector│  │ Collector│  │ Collector│     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
│       │             │             │             │            │
│       ▼             ▼             ▼             ▼            │
│  ┌────────────────────────────────────────────────────┐     │
│  │              ProfilingRun (JSON)                   │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ANALYSIS PHASE                                            │
│                                                              │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐                 │
│  │ Hotspot  │  │ Regression │  │ Scaling  │                 │
│  │ Analyzer │  │ Analyzer   │  │ Analyzer │                 │
│  └────┬─────┘  └─────┬──────┘  └────┬─────┘                 │
│       │              │              │                        │
│       ▼              ▼              ▼                        │
│  ┌────────────────────────────────────────────────────┐     │
│  │              AnalysisResult                        │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. REPORTING PHASE                                           │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ Markdown │  │ JSON     │  │ Dashboard│                   │
│  │ Reporter │  │ Exporter │  │ Generator│                   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │             │             │                          │
│       ▼             ▼             ▼                          │
│  report.md    results.json   dashboard.html                  │
└─────────────────────────────────────────────────────────────┘
```

### CLI Interface

```bash
# Full profiling suite
powers-profile run --suite full --output ./profiling_results

# CPU-only profiling
powers-profile run --collectors cpu --output ./profiling_results

# Compare two versions
powers-profile compare v0.2.0 HEAD --output comparison.html

# View history
powers-profile history --limit 20

# Quick CLI summary of last run
powers-profile summary

# Generate dashboard from existing data
powers-profile dashboard ./profiling_results/runs/latest

# Scaling analysis
powers-profile scaling --threads 1,2,4,8,16,32 --output scaling.html
```

### Profiling Domains

#### A. CPU Profiling
- **perf record/report**: CPU cycles, instructions, cache misses
- **FlameGraph**: Call stack visualization (SVG)
- **Hotspot detection**: Top functions by CPU time

#### B. Memory Profiling
- **DHAT**: Heap allocation analysis
- **Massif**: Heap usage over time
- **Cachegrind**: Cache miss analysis
- **RSS Monitoring**: Physical memory tracking

#### C. Parallelism Profiling
- **Thread scaling**: Performance vs thread count
- **Efficiency calculation**: Speedup / thread count
- **Contention detection**: Lock wait times (via perf)

#### D. Timing Profiling
- **Internal timing**: Extract from POWE.RS timing module
- **Phase breakdown**: Forward/backward pass times
- **Solver timing**: HiGHS-specific metrics

#### E. I/O Profiling
- **File I/O**: Read/write times for input/output
- **Parquet overhead**: Serialization costs

---

## Phases & Milestones

| Phase | Name | Duration | Milestone |
|-------|------|----------|-----------|
| 1 | Core Framework | 3 weeks | CLI skeleton, data schemas, basic collectors working |
| 2 | CPU & Execution | 2 weeks | FlameGraph generation, perf integration, hotspot analysis |
| 3 | Memory Suite | 2 weeks | DHAT, Massif, Cachegrind, RSS all integrated |
| 4 | Parallelism | 2 weeks | Thread scaling analysis, efficiency metrics |
| 5 | Visualization | 2 weeks | Plotly dashboards, interactive reports |
| 6 | Integration | 1 week | Documentation, cleanup, baseline establishment |

**Total**: ~12 weeks

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Valgrind overhead makes profiling slow | High | Medium | Separate fast/thorough profiling modes |
| perf requires root/capabilities | Medium | High | Document setup, use `perf_event_paranoid` settings |
| FlameGraph scripts not installed | Medium | Low | Include in repo or document installation |
| Large profiling data files | Medium | Medium | Compression, selective data collection |
| Plotly complexity | Low | Medium | Start with simple charts, iterate |
| WSL2 perf limitations | Medium | Medium | Document bare-metal alternatives |

---

## Success Metrics

- [ ] **CLI Usability**: Single command runs full profiling suite
- [ ] **Data Coverage**: All 5 profiling domains have collectors
- [ ] **Machine Readability**: 100% of data exportable as JSON
- [ ] **Comparison Capability**: Can compare any two git commits
- [ ] **Dashboard Quality**: Interactive Plotly dashboard with all key metrics
- [ ] **Documentation**: Complete user guide with examples
- [ ] **Scalability**: Tested on 1, 8, 32, 192 cores
- [ ] **Performance**: Full profiling suite completes in < 30 minutes

---

## Hardware Target Compatibility

| Environment | Cores | Notes |
|-------------|-------|-------|
| WSL2 Development | 4-16 | Primary dev environment, some perf limitations |
| Bare-metal Linux | 8-32 | Secondary dev, full perf support |
| AWS c7a.48xlarge | 192 | Production scaling tests |

---

## Dependencies

### External Tools (Must be installed)

| Tool | Purpose | Installation |
|------|---------|--------------|
| `perf` | CPU profiling | `linux-tools-$(uname -r)` |
| `valgrind` | DHAT, Massif, Cachegrind | `apt install valgrind` |
| `FlameGraph` | SVG generation | Clone from github |
| Python 3.10+ | Framework runtime | System or pyenv |
| Plotly | Visualization | `pip install plotly` |

### Python Dependencies

```
plotly>=5.18
pandas>=2.0
numpy>=1.24
toml>=0.10
rich>=13.0  # CLI formatting
typer>=0.9  # CLI framework
```

---

## Relationship to Clean Code Refactoring Plan

This plan creates a **new Epic 6** in the clean-code-refactoring plan:

| Original Epic | New Epic | Status |
|---------------|----------|--------|
| Epic 5: Memory Optimization | (unchanged) | ✅ Complete |
| Epic 6: Test Modernization | **Epic 7: Test Modernization** | ⬜ Moved |
| Epic 7: Performance Validation | **Epic 8: Final Validation** | ⬜ Moved |
| (new) | **Epic 6: Performance Evaluation Infrastructure** | ⬜ NEW |

The performance evaluation infrastructure must be complete before Test Modernization and Final Validation, as it provides the tools needed to validate performance during those phases.

---

## Next Steps

1. Drive Epic 3 Sprint 1: implement DHAT (T-020) and RSS monitor (T-023) to enable unified memory collector.
2. Produce valgrind fixtures (DHAT/Massif/Cachegrind) and finalize the `memory_data` JSON schema contract.
3. Define dashboard/report data contract for memory and scaling outputs to de-risk Epic 5 visualization work.
4. Schedule Epic 4 kickoff once memory aggregation stabilizes and perf/affinity prerequisites are documented.
