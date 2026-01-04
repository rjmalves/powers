# Tools Reference

Complete reference for all profiling collectors, analyzers, and reporters.

## Table of Contents

- [Collectors](#collectors)
  - [Timing Collector](#timing-collector)
  - [RSS Collector](#rss-collector)
  - [CPU Collector](#cpu-collector)
  - [Memory Collector](#memory-collector)
  - [Parallel Collector](#parallel-collector)
- [Analyzers](#analyzers)
- [Reporters](#reporters)
- [Configuration](#configuration)

---

## Collectors

Collectors gather performance data during program execution.

### Timing Collector

**Name**: `timing`  
**Overhead**: ~0% (parses program output)  
**Dependencies**: None

Extracts timing metrics from program stdout/stderr.

**Usage:**
```bash
python -m powers_profile run -c timing
```

**Output:**
- `timing/stdout.log` - Program standard output
- `timing/stderr.log` - Program standard error

**Metrics:**
- `timings` - Dictionary of phase names to durations

**Configuration:**
```toml
[timing]
parse_stdout = true
log_level = "debug"
```

---

### RSS Collector

**Name**: `rss`  
**Overhead**: <1% (background polling)  
**Dependencies**: Linux `/proc` filesystem

Monitors Resident Set Size (physical memory usage) via `/proc/PID/status`.

**Usage:**
```bash
python -m powers_profile run -c rss
```

**Output:**
- `rss/rss_data.json` - Timeline samples + summary

**Metrics:**
- `peak_mb` - Maximum RSS observed
- `mean_mb` - Average RSS across samples
- `final_mb` - RSS at program termination
- `growth_mb` - Change from start to peak
- `samples` - List of `{timestamp, rss_mb}` readings

**Configuration:**
```toml
[memory]
rss_interval_ms = 500  # Sampling interval
```

**Best For:** Lightweight memory monitoring, detecting memory leaks

---

### CPU Collector

**Name**: `cpu`  
**Overhead**: 1-5% (perf sampling)  
**Dependencies**: Linux `perf`, FlameGraph (optional)

Profiles CPU usage using Linux perf sampling.

**Usage:**
```bash
python -m powers_profile run -c cpu
```

**Output:**
- `cpu/perf.data` - Raw perf data
- `cpu/flamegraph.svg` - FlameGraph visualization
- `cpu/hotspots.json` - Top functions by CPU time

**Metrics:**
- `hotspots` - List of `{function, self_percent, total_percent, samples}`
- `total_samples` - Total perf samples collected

**Configuration:**
```toml
[cpu]
perf_frequency = 99  # Sampling frequency (Hz)
perf_events = ["cycles", "instructions", "cache-misses"]
flamegraph_width = 1200
flamegraph_colors = "hot"  # "hot", "mem", "io", "java", etc.
```

**Best For:** Identifying CPU hotspots, understanding call stacks

**Troubleshooting:**
- **Permission denied**: `sudo sysctl kernel.perf_event_paranoid=1`
- **perf not found**: `sudo apt install linux-tools-generic`

---

### Memory Collector

**Name**: `memory`  
**Overhead**: 3-50x slowdown (valgrind tools)  
**Dependencies**: valgrind

Comprehensive memory profiling using DHAT, Massif, and Cachegrind.

**Usage:**
```bash
# Use with small examples only
python -m powers_profile run -c memory -- run examples/01-deterministic
```

**Output:**
- `memory/dhat/dhat.out.json` - DHAT heap allocation profile
- `memory/massif/massif.out` - Massif heap timeline
- `memory/cachegrind/cachegrind.out` - Cache efficiency (if enabled)
- `memory/rss/rss_data.json` - RSS monitoring
- `memory/memory_data.json` - Aggregated metrics

**Metrics:**

**DHAT (Heap Allocation):**
- `total_bytes` - Total bytes allocated
- `total_blocks` - Total allocation count
- `max_bytes` - Peak heap usage
- `max_blocks` - Peak allocation count

**Massif (Heap Timeline):**
- `peak_bytes` - Peak heap size
- `peak_snapshot` - Snapshot number at peak
- `snapshots` - Timeline of heap usage

**Cachegrind (Cache Efficiency - opt-in):**
- `Ir` - Instructions executed
- `I1mr` - L1 instruction cache misses
- `Dr` - Data reads
- `D1mr` - L1 data read misses
- `Dw` - Data writes
- `D1mw` - L1 data write misses

**Configuration:**
```toml
[memory]
dhat_enabled = true
massif_enabled = true
massif_time_unit = "ms"
cachegrind_enabled = false  # Very slow, opt-in only
rss_interval_ms = 500
```

**Best For:** Detecting memory leaks, analyzing heap allocation patterns, cache optimization

**Slowdown:**
- DHAT: 3-10x
- Massif: 5-20x
- Cachegrind: 10-50x

---

### Parallel Collector

**Name**: `parallel`  
**Overhead**: Depends on iterations (N x program runtime)  
**Dependencies**: None for timing, `perf` for contention

Analyzes parallel scaling across multiple thread counts.

**Usage:**
```bash
python -m powers_profile scaling --threads 1,2,4,8,16
```

**Output:**
- `parallel/scaling_data.json` - Complete scaling analysis

**Metrics:**
- `scaling_results` - Per-thread-count timing results
- `speedup_metrics` - Speedup and efficiency calculations
- `amdahl_estimate` - Serial fraction, predicted max speedup
- `bottlenecks` - Detected scaling issues
- `contention_analysis` - Lock contention (if enabled)

**Configuration:**
```toml
[parallel]
thread_counts = [1, 2, 4, 8, 16, 32]
warmup_iterations = 1
```

**Command Options:**
```bash
--threads 1,2,4,8             # Thread counts to test
--iterations 3                # Measurement iterations per count
--warmup 1                    # Warmup iterations per count
--contention                  # Enable lock contention detection
--continue-on-error           # Continue if a thread count fails
--timeout 600                 # Timeout per iteration (seconds)
```

**Best For:** Understanding parallel efficiency, detecting Amdahl's law limitations, finding lock contention

---

## Analyzers

Analyzers process collected data to derive insights.

### Memory Comparison

Compares memory metrics between two runs.

**Usage:**
```bash
python -m powers_profile compare <baseline-id> <target-id>
```

**Metrics:**
- RSS deltas (peak, mean, growth)
- DHAT deltas (total bytes, blocks, peak)
- Massif deltas (peak heap)
- Cachegrind deltas (cache misses)

**Thresholds:**
```toml
[thresholds]
regression_percent = 5.0      # Flag regressions > 5%
improvement_percent = 5.0     # Flag improvements > 5%
rss_growth_mb = 10.0          # Flag RSS growth > 10 MB
```

### Scaling Analysis

Computes speedup, efficiency, and Amdahl estimates.

**Formulas:**
- **Speedup**: `S(n) = T(1) / T(n)`
- **Efficiency**: `E(n) = S(n) / n`
- **Serial Fraction**: `f = (1/S - 1/n) / (1 - 1/n)`

**Methods:**
- `harmonic` - Harmonic mean across measurements (default)
- `max_threads` - Conservative estimate using highest thread count
- `least_squares` - Curve fitting (future)

---

## Reporters

Reporters transform profiling data into readable formats.

### Dashboard Reporter

Generates interactive HTML dashboards with Plotly charts.

**Usage:**
```bash
python -m powers_profile dashboard [run-id]
python -m powers_profile dashboard --baseline <baseline> <target>
```

**Options:**
```bash
--output PATH              # Output HTML path
--baseline RUN_ID          # Baseline for comparison
--offline / --online       # Embed plotly.js (default: offline)
--theme THEME              # plotly, plotly_white, plotly_dark
```

**Features:**
- Tabbed interface (Summary, Timing, Memory, Parallel, CPU, Comparison)
- Interactive charts (hover, zoom, pan)
- Self-contained HTML (~12KB with offline mode)
- Gradient header with metadata
- Color-coded metrics

### Markdown Reporter

Generates human-readable markdown reports.

**Usage:**
```bash
python -m powers_profile report [run-id]
python -m powers_profile report --baseline <baseline> <target>
```

**Options:**
```bash
--output PATH              # Output markdown path
--baseline RUN_ID          # Baseline for comparison
```

**Sections:**
- Overview (run ID, timestamp, binary, args)
- Git information
- System information
- Timing analysis
- Memory analysis (RSS, DHAT, Massif)
- CPU analysis (hotspots)
- Parallel scaling (speedup, Amdahl)
- Comparison (if baseline provided)

### CLI Summary

Enhanced terminal output with Rich formatting.

**Usage:**
```bash
python -m powers_profile summary [run-id]
python -m powers_profile summary --verbose
```

**Features:**
- Colored panels and tables
- Emoji icons (⏱️💾🔥⚡)
- Progress indicators
- Detailed metrics in verbose mode

---

## Configuration

### Configuration File Locations

1. `profiling/config/default.toml` (repository defaults)
2. `~/.config/powers-profile/config.toml` (user overrides)
3. `--config PATH` (command-line override)

### Complete Configuration Reference

```toml
[general]
output_dir = "profiling_results"
binary = "target/release/powers"
default_example = "examples/05-large-scale-brazilian"

[collectors]
default = ["timing", "cpu", "memory"]
available = ["timing", "cpu", "memory", "parallel", "io"]

[cpu]
perf_frequency = 99
perf_events = ["cycles", "instructions", "cache-misses"]
flamegraph_width = 1200
flamegraph_colors = "hot"

[memory]
dhat_enabled = true
massif_enabled = true
massif_time_unit = "ms"
cachegrind_enabled = false
rss_interval_ms = 500

[parallel]
thread_counts = [1, 2, 4, 8, 16, 32]
warmup_iterations = 1

[timing]
parse_stdout = true
log_level = "debug"

[io]
enabled = false

[tools]
perf = null              # Auto-detect
valgrind = null          # Auto-detect
flamegraph = null        # Auto-detect

[thresholds]
regression_percent = 5.0
improvement_percent = 5.0
rss_growth_mb = 10.0
```

### Environment Variables

```bash
# Override thread count for parallel execution
export RAYON_NUM_THREADS=8

# NUMA control (multi-socket systems)
export GOMP_CPU_AFFINITY="0-7"

# Perf permissions (avoid sudo)
sudo sysctl kernel.perf_event_paranoid=1
```

---

## Collector Selection Guide

| Use Case | Collectors | Overhead | Example |
|----------|-----------|----------|---------|
| **Quick feedback** | `timing,rss` | <1% | `run -c timing,rss` |
| **CPU optimization** | `cpu,timing` | 1-5% | `run -c cpu,timing` |
| **Memory leak detection** | `rss` | <1% | `run -c rss` |
| **Heap analysis** | `memory` | 5-20x | `run -c memory -- run examples/01-deterministic` |
| **Parallel tuning** | `parallel` | N×runtime | `scaling --threads 1,2,4,8` |
| **Full profiling** | `cpu,memory,timing` | 5-20x | `run -c cpu,memory,timing -- run examples/01-deterministic` |
| **CI/Regression** | `timing,rss,cpu` | 1-5% | `run -c timing,rss,cpu` |

## Command Reference

### Run Command

```bash
python -m powers_profile run [OPTIONS] [-- ARGS...]

Options:
  -c, --collectors LIST     Collectors to use (comma-separated)
  -b, --binary PATH         Binary to profile
  -o, --output PATH         Output directory
  --continue-on-error       Continue if collector fails
```

### Scaling Command

```bash
python -m powers_profile scaling [OPTIONS] [-- ARGS...]

Options:
  -t, --threads LIST        Thread counts (comma-separated)
  -i, --iterations N        Measurement iterations per count
  -w, --warmup N            Warmup iterations per count
  -c, --contention          Enable contention detection
  --continue-on-error       Continue if thread count fails
  --timeout N               Timeout per iteration (seconds)
  --summary / --no-summary  Display summary after collection
```

### Dashboard Command

```bash
python -m powers_profile dashboard [RUN_ID] [OPTIONS]

Options:
  -o, --output PATH         Output HTML path
  -b, --baseline RUN_ID     Baseline for comparison
  --offline / --online      Embed plotly.js
  -t, --theme THEME         Plotly theme
```

### Report Command

```bash
python -m powers_profile report [RUN_ID] [OPTIONS]

Options:
  -o, --output PATH         Output markdown path
  -b, --baseline RUN_ID     Baseline for comparison
```

### Summary Command

```bash
python -m powers_profile summary [RUN_ID] [OPTIONS]

Options:
  -o, --output PATH         Output directory
  -v, --verbose             Show detailed metrics
```

### Compare Command

```bash
python -m powers_profile compare BASELINE TARGET [OPTIONS]

Options:
  -o, --output PATH         Output directory
```

### History Command

```bash
python -m powers_profile history [OPTIONS]

Options:
  -n, --limit N             Number of runs to show
  -o, --output PATH         Output directory
  -f, --format FORMAT       Output format (table, json)
```

---

## See Also

- **[Quick Start](QUICK_START.md)** - Getting started in 5 minutes
- **[Analysis Guide](ANALYSIS_GUIDE.md)** - Interpreting profiling results
- **[Comparison Guide](COMPARISON_GUIDE.md)** - Version comparison workflows
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions
- **[Extending Guide](EXTENDING.md)** - Adding custom collectors
