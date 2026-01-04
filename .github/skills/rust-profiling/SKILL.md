---
name: rust-profiling
description: Guide agents to profile Rust code using the POWERS profiling infrastructure (perf, flamegraph, timing metrics) for CPU performance analysis in the POWE.RS SDDP solver.
license: MIT
metadata:
  author: rjmalves
  version: "2.0"
  tags:
    - rust
    - profiling
    - flamegraph
    - perf
    - performance
    - optimization
    - powers-profile
---

# Rust Profiling for Performance Analysis

## Overview

This skill guides agents in profiling the POWE.RS SDDP solver using the integrated **POWERS profiling infrastructure** to identify performance bottlenecks, hotspots, and optimization opportunities. Profiling is essential for HPC workloads where every millisecond counts in solving large-scale stochastic optimization problems.

**Primary Tool**: The `powers_profile` Python package provides unified CPU profiling with perf, flamegraphs, and timing extraction.

## Quick Start

### Installation

```bash
# Install profiling infrastructure
cd profiling/
pip install -e .
```

### Basic CPU Profiling

```bash
# Profile with timing extraction (fast)
python -m powers_profile run -c timing -- run examples/01-deterministic

# Profile with perf + flamegraph (1-5% overhead)
python -m powers_profile run -c cpu -- run examples/01-deterministic

# View interactive dashboard
python -m powers_profile dashboard

# View CLI summary
python -m powers_profile summary
```

## POWERS Profiling Infrastructure

The integrated profiling system (`profiling/powers_profile/`) provides:

- **Unified CLI**: Single command for all CPU profiling
- **Multiple collectors**: Timing, perf, flamegraph generation
- **Automatic postprocessing**: Extract hotspots, generate SVG flamegraphs
- **Interactive dashboards**: HTML reports with charts and tables
- **Run history**: Track improvements across commits

### CPU Collectors

1. **Timing (Lightweight)**
   - Extracts timing from program output
   - 0% overhead (passive observation)
   - Tracks: phase durations, iteration times

2. **CPU (perf + Flamegraph)**
   - Linux `perf` integration with automatic flamegraph generation
   - 1-5% overhead
   - Tracks: hotspots, call stacks, CPU time distribution

3. **Parallel Scaling**
   - Tests performance across thread counts
   - Calculates speedup, efficiency, Amdahl's law estimates
   - Detects contention and bottlenecks

## Profiling Tools

### 1. CPU Profiling with Perf + Flamegraph (Primary)
**Best for**: Visual identification of CPU hotspots

**Run with POWERS profiling**:
```bash
# Profile CPU with flamegraph generation
python -m powers_profile run -c cpu -- run examples/01-deterministic

# View results in dashboard
python -m powers_profile dashboard

# Open flamegraph directly
firefox profiling_results/runs/latest/cpu/flamegraph.svg

# View hotspots table
jq '.hotspots[:10]' profiling_results/runs/latest/cpu/cpu_summary.json
```

**Output**:
- `flamegraph.svg` - Interactive SVG visualization
  - **Width**: Proportional to time spent in function
  - **Stack depth**: Call hierarchy (parent calls child)
  - **Colors**: Hot (red/yellow) indicates high CPU usage
  - **Search**: Click to zoom, ctrl+F to search function names
- `cpu_summary.json` - Top hotspots with percentages
- `perf.data` - Raw perf data for advanced analysis

**Output structure**:
```
profiling_results/runs/<run-id>/cpu/
├── perf.data           # Raw perf data
├── flamegraph.svg      # Interactive flamegraph
└── cpu_summary.json    # Postprocessed hotspots
```

**Advanced perf analysis**:
```bash
# Interactive perf report
perf report -i profiling_results/runs/latest/cpu/perf.data

# Check for specific function
perf report -i profiling_results/runs/latest/cpu/perf.data | grep "function_name"
```

### 2. Timing Extraction (Lightweight)
**Best for**: Quick phase-level performance breakdown

**Run with POWERS profiling**:
```bash
# Extract timing from program output
python -m powers_profile run -c timing -- run examples/01-deterministic

# View timing breakdown in dashboard
python -m powers_profile dashboard

# View raw timing data
jq '.timing_breakdown' profiling_results/runs/latest/timing/timing_summary.json
```

**Metrics**:
- **Phase durations**: Time per algorithm phase
- **Iteration times**: Per-iteration timings
- **Total duration**: End-to-end execution time

### 3. Parallel Scaling Analysis
**Best for**: Understanding thread scaling and parallelism efficiency

**Run with POWERS profiling**:
```bash
# Default thread counts (1,2,4,8)
python -m powers_profile scaling

# Custom thread counts
python -m powers_profile scaling --threads 1,2,4,8,16

# With contention detection
python -m powers_profile scaling --contention

# View results
python -m powers_profile dashboard
# Navigate to "Parallel Scaling" tab
```

**Metrics**:
- **Speedup**: Performance gain vs single thread
- **Efficiency**: Speedup / thread_count (ideal: 100%)
- **Amdahl's law**: Serial fraction estimate and predicted max speedup
- **Bottleneck detection**: Regressions, contention, efficiency cliffs

**Example output**:
```
 Threads | Duration (s) |    Speedup | Efficiency |           Notes
--------------------------------------------------------------------------------
       1 |    10.000000 |      1.00x |      100.0% |    🟢 Excellent
       2 |     5.100000 |      1.96x |       98.0% |    🟢 Excellent
       4 |     2.600000 |      3.85x |       96.2% |    🟢 Excellent
       8 |     1.400000 |      7.14x |       89.3% |       🟡 Good
```

### 4. Alternative Tools (Manual, not recommended)

**Flamegraph (cargo install)**:
```bash
# Install
cargo install flamegraph

# Profile directly (bypasses POWERS infrastructure)
cargo flamegraph --bin powers -- --config examples/config.json
```

**Samply (Cross-platform)**:
```bash
# Install
cargo install samply

# Profile with Firefox Profiler UI
samply record cargo run --release -- run examples/config.json
```

**Note**: Use `python -m powers_profile run -c cpu` instead for integrated workflow.

## Cargo Profile Configuration

The project's `Cargo.toml` includes optimized profiling configurations:

### Release Profile with Debug Info
```toml
[profile.release]
debug = true  # Include debug symbols for profiling
```

This enables profiling with function names while maintaining release-mode optimizations.

### Distribution Profile
```toml
[profile.dist]
inherits = "release"
lto = "thin"  # Thin LTO for fast builds with good optimization
```

For production profiling, use the `dist` profile for LTO optimization analysis.

## Integration with Timing Module

POWE.RS includes a built-in timing instrumentation system in `src/timing/`:

### Timing Feature Flags
```bash
# Enable basic timing collection
cargo build --features timing

# Enable detailed per-stage timing
cargo build --features timing-detailed
```

### Timing Module Components
- **`src/timing/atomic.rs`**: Lock-free atomic time storage
- **`src/timing/collector.rs`**: Centralized timing data collection
- **`src/timing/guard.rs`**: RAII timing guards for automatic measurement
- **`src/timing/metrics.rs`**: Timing metric definitions and aggregation

### Example: Using Timing Guards
```rust
use crate::timing::TimingGuard;

fn expensive_operation() {
    let _guard = TimingGuard::new("expensive_operation");
    // Operation is automatically timed until _guard is dropped
    perform_computation();
}
```

### Viewing Timing Results
When running with `--features timing`, the solver outputs timing statistics:
```
Timing Statistics:
  forward_pass:  125.34ms (40.2%)
  backward_pass: 98.76ms  (31.7%)
  cut_selection: 45.23ms  (14.5%)
  solver_calls:  42.10ms  (13.5%)
```

## Profiling Workflow

### 1. Identify the Problem

```bash
# Quick flamegraph to find hotspots
python -m powers_profile run -c cpu -- run examples/01-deterministic

# View in dashboard
python -m powers_profile dashboard
# Navigate to "CPU Hotspots" tab
```

Look for:
- **Wide plateaus**: Functions consuming significant time
- **Unexpected depth**: Excessive call stack depth
- **Frequent calls**: Small functions called many times

### 2. Measure Baseline

```bash
# Record baseline performance
python -m powers_profile run -c cpu,timing -- run examples/01-deterministic

# Note the run ID for later comparison
python -m powers_profile history
```

### 3. Analyze Hotspots

```bash
# View top hotspots
jq '.hotspots[:20]' profiling_results/runs/latest/cpu/cpu_summary.json

# Check timing breakdown
jq '.timing_breakdown' profiling_results/runs/latest/timing/timing_summary.json

# For parallel code, test thread scaling
python -m powers_profile scaling --threads 1,2,4,8
```

### 4. Optimize and Verify

```bash
# After making changes, profile again
python -m powers_profile run -c cpu,timing -- run examples/01-deterministic

# Compare with baseline
python -m powers_profile compare <baseline-run-id> <new-run-id>

# View comparison dashboard
python -m powers_profile dashboard --baseline <baseline-run-id> <new-run-id>
```

## Integration with Benchmarks

Combine POWERS profiling with Criterion benchmarks:

### Profile Benchmarks

```bash
# Profile end-to-end SDDP benchmark
python -m powers_profile run -c cpu -- bench --bench sddp_e2e

# Profile SIMD operations
python -m powers_profile run -c cpu -- bench --bench simd_dot_product

# Compare benchmark before/after optimization
cargo bench --bench sddp_e2e --save-baseline before
# Make changes...
cargo bench --bench sddp_e2e --baseline before
```

### Analyze Performance Counters (Advanced)

For detailed cache and branch analysis, use perf directly:

```bash
# Cache analysis
perf stat -e L1-dcache-load-misses,L1-dcache-loads,LLC-loads,LLC-load-misses \
  cargo bench --bench sddp_e2e

# Branch prediction
perf stat -e branch-misses,branches \
  cargo bench --bench sddp_e2e

# SIMD vectorization (AVX2)
perf stat -e fp_arith_inst_retired.256b_packed_double \
  cargo bench --features simd-optimizations --bench simd_dot_product
```

## Common Performance Patterns in POWE.RS

### 1. Subproblem Solver Calls
**File**: `src/subproblem.rs` (230KB)

Profile to optimize:
- HiGHS solver invocation overhead
- State extraction and preparation
- Dual variable retrieval

```bash
# Profile subproblem performance
python -m powers_profile run -c cpu -- run examples/simple.json

# Check hotspots for solver-related functions
jq '.hotspots[] | select(.symbol | contains("highs") or contains("subproblem"))' \
  profiling_results/runs/latest/cpu/cpu_summary.json
```

### 2. Cut Management
**Files**: `src/cut.rs`, `src/sddp/mod.rs` (138KB)

Profile for:
- Cut evaluation loops
- Dominated cut detection
- Storage and retrieval patterns

```bash
# Profile cut management
python -m powers_profile run -c cpu -- run examples/05-large-scale-brazilian

# Filter for cut-related hotspots
jq '.hotspots[] | select(.symbol | contains("cut"))' \
  profiling_results/runs/latest/cpu/cpu_summary.json
```

### 3. Parallel Forward Pass
**Feature**: Uses `rayon = "1.10.0"`

Profile with scaling analysis:
```bash
# Test thread scaling
python -m powers_profile scaling --threads 1,2,4,8,16

# View results in dashboard
python -m powers_profile dashboard
# Navigate to "Parallel Scaling" tab
```

Look for:
- Thread synchronization overhead
- Load imbalance across threads
- Efficiency drops at higher thread counts

### 4. SIMD Operations
**Benchmark**: `benches/simd_dot_product.rs`
**Feature flag**: `--features simd-optimizations`

```bash
# Profile SIMD performance
python -m powers_profile run -c cpu -- bench --bench simd_dot_product

# Check vectorization with perf
perf stat -e fp_arith_inst_retired.256b_packed_double \
  cargo bench --features simd-optimizations --bench simd_dot_product
```

## Advanced Profiling Techniques

### CPU Cache Analysis
```bash
# L1/L2/L3 cache misses
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
  cargo run --release -- run examples/01-deterministic

# Interpret results:
# - L1 miss rate > 10%: Consider data layout (SoA patterns)
# - LLC miss rate > 1%: Memory access patterns need optimization
```

### Branch Prediction
```bash
perf stat -e branches,branch-misses \
  cargo run --release -- run examples/01-deterministic

# High branch miss rate (> 5%): Consider:
# - Branchless programming techniques
# - Profile-guided optimization
# - Reordering conditional code
```

### Thread Contention Detection
```bash
# Profile with lock contention tracking
python -m powers_profile scaling --contention --threads 1,2,4,8

# View contention metrics in dashboard
python -m powers_profile dashboard
# Check "Parallel Scaling" → "Contention Metrics"
```

### Comparing Optimizations
```bash
# Profile baseline
python -m powers_profile run -c cpu,timing
BASELINE_ID=$(python -m powers_profile history | head -n1 | cut -d' ' -f1)

# Make optimizations...
# cargo build --release

# Profile optimized version
python -m powers_profile run -c cpu,timing
TARGET_ID=$(python -m powers_profile history | head -n1 | cut -d' ' -f1)

# Generate comparison report
python -m powers_profile compare $BASELINE_ID $TARGET_ID
python -m powers_profile dashboard --baseline $BASELINE_ID $TARGET_ID
```

## Memory Profiling Integration

For memory-focused profiling, see the `rust-memory-analysis` skill. Combine CPU and memory profiling:

```bash
# Profile both CPU and memory
python -m powers_profile run -c cpu,memory -- run examples/01-deterministic

# View unified dashboard
python -m powers_profile dashboard
# Tabs: Summary, Timing, Memory, CPU, etc.
```

## Profiling Large Files

POWE.RS contains several large files that may have hotspots:

- **`src/subproblem.rs`** (235KB): Subproblem formulation and solving
- **`src/state.rs`** (137KB): State management and transitions
- **`src/sddp/mod.rs`** (138KB): Core SDDP algorithm implementation

Profile these systematically:
```bash
# Full profiling with all collectors
python -m powers_profile run -c all -- run examples/05-large-scale-brazilian

# View comprehensive dashboard
python -m powers_profile dashboard
```

## POWERS Profiling Commands Quick Reference

```bash
# CPU profiling
python -m powers_profile run -c timing     # Lightweight timing only
python -m powers_profile run -c cpu        # Perf + flamegraph (1-5% overhead)
python -m powers_profile run -c all        # All collectors

# Parallel scaling
python -m powers_profile scaling                      # Default (1,2,4,8)
python -m powers_profile scaling --threads 1,2,4,8,16 # Custom
python -m powers_profile scaling --contention         # With lock tracking

# View results
python -m powers_profile summary           # CLI summary
python -m powers_profile dashboard         # Interactive HTML
python -m powers_profile history           # List all runs

# Compare runs
python -m powers_profile compare <baseline-id> <target-id>
python -m powers_profile dashboard --baseline <baseline-id> <target-id>

# Custom workload
python -m powers_profile run -c cpu -- run examples/05-large-scale-brazilian
python -m powers_profile run -c cpu -- bench --bench sddp_e2e
```

## Best Practices

1. **Use POWERS profiling infrastructure** - Unified workflow, automatic postprocessing
2. **Always profile in release mode** - Debug mode has different performance characteristics
3. **Profile realistic workloads** - Use production-like problem sizes
4. **Focus on hotspots** - Optimize the 20% of code consuming 80% of time
5. **Verify correctness** - Always run tests after optimization
6. **Compare before/after** - Use `python -m powers_profile compare` for quantitative analysis
7. **Check parallel scaling** - Test with `python -m powers_profile scaling`

## Profiling Checklist

- [ ] **Install profiling infrastructure**: `pip install -e profiling/`
- [ ] **Baseline profile**: `python -m powers_profile run -c cpu,timing`
- [ ] **View flamegraph**: Check dashboard "CPU" tab or open SVG
- [ ] **Identify top 3 hotspots**: Check `cpu_summary.json` or dashboard table
- [ ] **Test parallel scaling**: `python -m powers_profile scaling`
- [ ] **Compare with baseline**: `python -m powers_profile compare <baseline> <target>`
- [ ] **View dashboard**: `python -m powers_profile dashboard`
- [ ] **Verify correctness**: `cargo test` after optimization

## File References

- **Profiling infrastructure**: `profiling/powers_profile/` - Python package for all profiling
- **CPU collectors**: `profiling/powers_profile/collectors/{cpu,timing,scaling_runner}.py`
- **Configuration**: `profiling/config/default.toml` - Profiling settings
- **Benchmarks**: `benches/README.md`, `benches/sddp_e2e.rs`, `benches/simd_dot_product.rs`
- **Large files**: `src/subproblem.rs` (235KB), `src/state.rs` (137KB), `src/sddp/mod.rs` (138KB)
- **Profile config**: `Cargo.toml` - `[profile.release]` with `debug = true` for symbols

## Related Skills

- **rust-benchmarking**: For statistical performance measurement with Criterion.rs
- **hpc-optimization**: For SIMD and parallelization optimization strategies
- **rust-memory-analysis**: For heap profiling and memory optimization using POWERS profiling

## Resources

- **POWERS Profiling README**: `profiling/README.md` - Complete profiling guide
- **Rust Performance Book**: https://nnethercote.github.io/perf-book/
- **Flamegraph GitHub**: https://github.com/flamegraph-rs/flamegraph
- **Firefox Profiler**: https://profiler.firefox.com/
- **Perf Wiki**: https://perf.wiki.kernel.org/
