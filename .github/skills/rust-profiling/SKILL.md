---
name: rust-profiling
description: Guide agents to profile Rust code using flamegraph, perf, and samply for CPU performance analysis in the POWE.RS SDDP solver, identifying hotspots and optimization opportunities.
license: MIT
metadata:
  author: rjmalves
  version: "1.0"
  tags:
    - rust
    - profiling
    - flamegraph
    - perf
    - performance
    - optimization
---

# Rust Profiling for Performance Analysis

## Overview

This skill guides agents in profiling the POWE.RS SDDP solver to identify performance bottlenecks, hotspots, and optimization opportunities. Profiling is essential for HPC workloads where every millisecond counts in solving large-scale stochastic optimization problems.

## Profiling Tools

### 1. Flamegraph (Primary Tool)
**Best for**: Quick visual identification of hotspots

```bash
# Install
cargo install flamegraph

# Profile the application
cargo flamegraph --bin powers -- --config examples/config.json

# Profile a benchmark
cargo flamegraph --bench sddp_e2e -- --bench

# Profile specific benchmark function
cargo flamegraph --bench sddp_e2e -- forward_pass
```

**Output**: `flamegraph.svg` - Interactive SVG visualization
- **Width**: Proportional to time spent in function
- **Stack depth**: Call hierarchy (parent calls child)
- **Colors**: Random, for visual distinction only
- **Search**: Click to zoom, ctrl+F to search function names

### 2. Perf (Linux Only)
**Best for**: Detailed CPU performance counters

```bash
# Record performance data
perf record --call-graph dwarf cargo bench --bench sddp_e2e

# Analyze with interactive report
perf report

# Generate flamegraph from perf data
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

**Key Metrics**:
- **CPU cycles**: Total CPU time
- **Cache misses**: L1, L2, L3 cache efficiency
- **Branch mispredictions**: Control flow optimization opportunities
- **Page faults**: Memory access patterns

### 3. Samply (Cross-Platform)
**Best for**: Interactive profiling with Firefox Profiler UI

```bash
# Install
cargo install samply

# Profile and open in Firefox Profiler
samply record cargo bench --bench sddp_e2e -- --bench

# Profile release build
samply record target/release/powers --config examples/config.json
```

**Features**:
- Timeline view of execution
- Call tree with self/total time
- CPU usage per thread
- Memory allocation tracking

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
cargo flamegraph --bench sddp_e2e -- --bench
```

Look for:
- **Wide plateaus**: Functions consuming significant time
- **Unexpected depth**: Excessive call stack depth
- **Frequent calls**: Small functions called many times

### 2. Measure Baseline
```bash
# Benchmark current performance
cargo bench --bench sddp_e2e --save-baseline before-opt

# Record detailed profile
perf record --call-graph dwarf cargo bench --bench sddp_e2e
perf report
```

### 3. Analyze Hotspots
```bash
# Use timing features for detailed breakdown
cargo run --release --features timing-detailed -- --config examples/config.json

# Check cache performance
perf stat -e cache-references,cache-misses,cycles,instructions cargo bench --bench sddp_e2e
```

### 4. Optimize and Verify
```bash
# After making changes, compare performance
cargo bench --bench sddp_e2e --baseline before-opt

# Verify with flamegraph
cargo flamegraph --bench sddp_e2e -- --bench
```

## Integration with Benchmarks

Combine profiling with benchmarks from `benches/README.md`:

### Profile Specific Operations
```bash
# Profile SIMD operations
cargo flamegraph --bench simd_dot_product -- --bench

# Profile end-to-end SDDP
cargo flamegraph --bench sddp_e2e -- --bench
```

### Analyze with Performance Counters
```bash
# Cache analysis for SIMD optimizations
perf stat -e L1-dcache-load-misses,L1-dcache-loads \
  cargo bench --bench simd_dot_product

# Branch prediction for conditional code
perf stat -e branch-misses,branches \
  cargo bench --bench sddp_e2e
```

## Common Performance Patterns in POWE.RS

### 1. Subproblem Solver Calls
**File**: `src/subproblem.rs` (230KB)

Profile to optimize:
- HiGHS solver invocation overhead
- State extraction and preparation
- Dual variable retrieval

```bash
cargo flamegraph --bin powers -- --config examples/simple.json
# Look for time in highs_sys FFI calls
```

### 2. Cut Management
**Files**: `src/cut.rs`, `src/sddp/mod.rs` (138KB)

Profile for:
- Cut evaluation loops
- Dominated cut detection
- Storage and retrieval patterns

### 3. Parallel Forward Pass
**Feature**: Uses `rayon = "1.10.0"`

Profile with:
```bash
# Compare 1 vs 8 threads
RAYON_NUM_THREADS=1 cargo flamegraph --bench sddp_e2e -- --bench
RAYON_NUM_THREADS=8 cargo flamegraph --bench sddp_e2e -- --bench
```

Look for:
- Thread synchronization overhead
- Load imbalance across threads
- Shared data contention

### 4. SIMD Operations
**Benchmark**: `benches/simd_dot_product.rs`
**Feature flag**: `--features simd-optimizations`

```bash
# Profile with SIMD enabled
cargo flamegraph --features simd-optimizations --bench simd_dot_product -- --bench

# Check vectorization
perf stat -e fp_arith_inst_retired.256b_packed_double \
  cargo bench --features simd-optimizations --bench simd_dot_product
```

## Advanced Profiling Techniques

### CPU Cache Analysis
```bash
# L1/L2/L3 cache misses
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
  cargo bench --bench sddp_e2e

# Interpret results:
# - L1 miss rate > 10%: Consider data layout (SoA patterns)
# - LLC miss rate > 1%: Memory access patterns need optimization
```

### Branch Prediction
```bash
perf stat -e branches,branch-misses cargo bench --bench sddp_e2e

# High branch miss rate (> 5%): Consider:
# - Branchless programming techniques
# - Profile-guided optimization
# - Reordering conditional code
```

### Thread Scaling Analysis
```bash
# Test thread scalability
for threads in 1 2 4 8; do
  echo "Testing with $threads threads:"
  RAYON_NUM_THREADS=$threads cargo bench --bench sddp_e2e
done
```

## Memory Profiling Integration

For memory-focused profiling, see the `rust-memory-analysis` skill. Combine CPU and memory profiling:

```bash
# CPU profile with memory allocation tracking
samply record cargo bench --bench sddp_e2e -- --bench
# Opens Firefox Profiler with allocation timeline
```

## Profiling Large Files

POWE.RS contains several large files that may have hotspots:

- **`src/subproblem.rs`** (235KB): Subproblem formulation and solving
- **`src/state.rs`** (137KB): State management and transitions
- **`src/sddp/mod.rs`** (138KB): Core SDDP algorithm implementation

Profile these systematically:
```bash
# Focus on specific modules using timing features
cargo run --release --features timing-detailed -- --config examples/large.json
```

## Best Practices

1. **Always profile in release mode** - Debug mode has different performance characteristics
2. **Use debug symbols** - `[profile.release] debug = true` enables function names
3. **Profile realistic workloads** - Use production-like problem sizes
4. **Focus on hotspots** - Optimize the 20% of code consuming 80% of time
5. **Verify correctness** - Always run tests after optimization
6. **Benchmark before/after** - Use `cargo bench --baseline` for quantitative comparison

## Profiling Checklist

- [ ] Enable release mode: `cargo build --release` or `cargo bench`
- [ ] Generate flamegraph: `cargo flamegraph --bench <benchmark>`
- [ ] Identify top 3 hotspots from flamegraph
- [ ] Use timing features: `--features timing-detailed`
- [ ] Compare with baseline: `cargo bench --baseline before`
- [ ] Analyze cache performance: `perf stat -e cache-misses`
- [ ] Check thread scaling: Test with 1, 2, 4, 8 threads
- [ ] Verify correctness: `cargo test` after optimization

## File References

- **Timing module**: `src/timing/atomic.rs`, `src/timing/collector.rs`, `src/timing/guard.rs`, `src/timing/metrics.rs`
- **Benchmarks**: `benches/README.md`, `benches/sddp_e2e.rs`, `benches/simd_dot_product.rs`
- **Large files**: `src/subproblem.rs` (235KB), `src/state.rs` (137KB), `src/sddp/mod.rs` (138KB)
- **Profile config**: `Cargo.toml` - `[profile.release]` and `[profile.dist]`
- **Feature flags**: `timing`, `timing-detailed` in `Cargo.toml`

## Related Skills

- **rust-benchmarking**: For statistical performance measurement with Criterion.rs
- **hpc-optimization**: For SIMD and parallelization optimization strategies
- **rust-memory-analysis**: For heap profiling and memory optimization

## Resources

- **Rust Performance Book**: https://nnethercote.github.io/perf-book/
- **Flamegraph GitHub**: https://github.com/flamegraph-rs/flamegraph
- **Firefox Profiler**: https://profiler.firefox.com/
- **Perf Wiki**: https://perf.wiki.kernel.org/
