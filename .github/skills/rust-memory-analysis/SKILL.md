---
name: rust-memory-analysis
description: Guide agents to analyze memory usage, optimize allocations, and reduce memory footprint in the POWE.RS SDDP solver using the integrated profiling infrastructure (DHAT, Massif, RSS monitoring).
license: MIT
metadata:
  author: rjmalves
  version: "2.0"
  tags:
    - rust
    - memory
    - profiling
    - allocation
    - optimization
    - powers-profile
---

# Rust Memory Analysis and Optimization

## Overview

This skill guides agents in analyzing and optimizing memory usage for the POWE.RS SDDP solver using the integrated **POWERS profiling infrastructure**. Memory efficiency is critical for HPC workloads, where large-scale optimization problems can consume gigabytes of memory and benefit significantly from allocation optimization.

**Primary Tool**: The `powers_profile` Python package provides unified memory profiling with DHAT, Massif, and RSS monitoring.

## Quick Start

### Installation

```bash
# Install profiling infrastructure
cd profiling/
pip install -e .
```

### Basic Memory Profiling

```bash
# Lightweight RSS monitoring (fast, <1% overhead)
python -m powers_profile run -c rss -- run examples/01-deterministic

# Full memory profiling (DHAT + Massif, slower)
python -m powers_profile run -c memory -- run examples/01-deterministic

# View interactive dashboard
python -m powers_profile dashboard

# View CLI summary
python -m powers_profile summary
```

## POWERS Profiling Infrastructure

The integrated profiling system (`profiling/powers_profile/`) provides:

- **Unified CLI**: Single command for all memory profiling
- **Multiple collectors**: DHAT, Massif, RSS monitoring
- **Automatic postprocessing**: Extract hotspots, peaks, summaries
- **Interactive dashboards**: HTML reports with charts and tables
- **Run history**: Track improvements across commits

### Memory Collectors

1. **RSS (Resident Set Size)**
   - Lightweight physical memory tracking via `/proc` polling
   - <1% overhead, suitable for production workloads
   - Tracks: peak RSS, average RSS, RSS timeline

2. **DHAT (Heap Profiler)**
   - Detailed heap allocation analysis
   - 3-10x slowdown, use on small/medium workloads
   - Tracks: total allocations, bytes allocated, hotspots, lifetimes

3. **Massif (Heap Timeline)**
   - Heap usage over time with snapshots
   - 5-20x slowdown, for understanding growth patterns
   - Tracks: heap growth, stack usage, allocation trees

## Memory Module Infrastructure

POWE.RS includes a dedicated memory management module in `src/memory/`:

### Module Structure
```
src/memory/
├── mod.rs       # Module exports and configuration
└── buffers.rs   # Buffer reuse and pooling patterns
```

### Buffer Reuse Patterns
The `src/memory/buffers.rs` module implements buffer pooling to reduce allocations:

```rust
// Example pattern from buffers.rs
pub struct BufferPool<T> {
    buffers: Vec<Vec<T>>,
    capacity: usize,
}

impl<T: Clone> BufferPool<T> {
    pub fn acquire(&mut self) -> Vec<T> {
        // Reuse existing buffer or allocate new
        self.buffers.pop().unwrap_or_else(|| Vec::with_capacity(self.capacity))
    }
    
    pub fn release(&mut self, mut buffer: Vec<T>) {
        buffer.clear();
        self.buffers.push(buffer);
    }
}
```

**Key Principle**: Reuse allocations instead of allocating/deallocating repeatedly.

## Detailed Memory Profiling Tools

### 1. DHAT (Heap Profiler via POWERS)
**Best for**: Detailed heap allocation analysis, identifying hot allocation sites

**Run with POWERS profiling**:
```bash
# Run DHAT profiler
python -m powers_profile run -c dhat -- run examples/01-deterministic

# View results in dashboard
python -m powers_profile dashboard

# View raw data
cat profiling_results/runs/latest/dhat/dhat_summary.json
```

**Metrics** (automatically extracted):
- **Total allocations**: Number of malloc calls
- **Total bytes allocated**: Cumulative heap usage
- **Peak bytes**: Maximum heap size
- **Hot allocation sites**: Top functions by allocation volume
- **Allocation lifetimes**: Short-lived vs long-lived allocations

**Output structure**:
```
profiling_results/runs/<run-id>/dhat/
├── dhat.out.<pid>       # Raw DHAT JSON from Valgrind
└── dhat_summary.json    # Postprocessed hotspots and metrics
```

**Alternative: Manual DHAT** (not recommended - use powers_profile instead):
```bash
# Build with DHAT instrumentation (old method)
cargo build --release --features dhat-heap
./target/release/powers --config examples/config.json
# Generates dhat-heap.json - view at https://nnethercote.github.io/dh_view/dh_view.html
```

### 2. Massif (Heap Timeline via POWERS)
**Best for**: Understanding heap growth patterns over time

**Run with POWERS profiling**:
```bash
# Run Massif profiler
python -m powers_profile run -c massif -- run examples/01-deterministic

# View results in dashboard
python -m powers_profile dashboard

# View text summary
ms_print profiling_results/runs/latest/massif/massif.out.*
```

**Metrics** (automatically extracted):
- **Heap size over time**: Growth curve
- **Stack size over time**: Stack usage patterns
- **Peak heap bytes**: Maximum memory consumption
- **Allocation tree**: Call graph showing who allocated what

**Output structure**:
```
profiling_results/runs/<run-id>/massif/
├── massif.out.<pid>       # Raw Massif output
└── massif_summary.json    # Postprocessed metrics
```

### 3. RSS Monitoring (Lightweight)
**Best for**: Fast physical memory tracking in production

**Run with POWERS profiling**:
```bash
# Run RSS collector (default 500ms polling interval)
python -m powers_profile run -c rss -- run examples/01-deterministic

# View timeline in dashboard
python -m powers_profile dashboard

# View summary
jq '.summary' profiling_results/runs/latest/rss/rss_data.json
```

**Metrics**:
- **Peak RSS**: Maximum physical memory used
- **Average RSS**: Mean memory over execution
- **RSS timeline**: Memory usage samples over time
- **Growth rate**: Memory increase per second

**Configuration** (in `profiling/config/default.toml`):
```toml
[memory]
rss_interval_ms = 500  # Polling interval
```

### 4. Unified Memory Collector
**Best for**: Comprehensive memory analysis (DHAT + Massif + RSS)

```bash
# Run all memory collectors
python -m powers_profile run -c memory -- run examples/01-deterministic

# View aggregated results
cat profiling_results/runs/latest/memory/memory_data.json

# View dashboard with all memory tabs
python -m powers_profile dashboard
```

## Mimalloc Integration

POWE.RS supports the `mimalloc` allocator for improved memory performance in HPC workloads.

### Configuration
In `Cargo.toml`:
```toml
# Optional: mimalloc allocator for better memory management in HPC workloads
mimalloc = { version = "0.1", optional = true }

[features]
mimalloc = ["dep:mimalloc"]
```

### Enabling Mimalloc
```bash
# Build with mimalloc
cargo build --release --features mimalloc

# Benchmark with mimalloc
cargo bench --features mimalloc
```

### When to Use Mimalloc
- **Large allocations**: >1MB objects (scenario trees, cut storage)
- **Parallel workloads**: rayon-based forward pass
- **Long-running processes**: Multi-hour SDDP training
- **Memory-intensive**: Problems with >10,000 cuts or >100 stages

### Mimalloc vs System Allocator
```bash
# Benchmark comparison
cargo bench --bench sddp_e2e --save-baseline system-alloc
cargo bench --bench sddp_e2e --features mimalloc --baseline system-alloc
```

Expect improvements:
- **Allocation speed**: 10-30% faster in parallel workloads
- **Memory fragmentation**: Better utilization of heap space
- **Peak memory**: 5-15% lower peak usage

## Analyzing Large Files

POWE.RS contains several large files that may have memory hotspots:

### Critical Files for Memory Analysis
- **`src/subproblem.rs`** (235KB): Stores LP matrices, vectors, and solver state
- **`src/state.rs`** (137KB): State space representation and transitions
- **`src/sddp/mod.rs`** (138KB): Cut storage, scenario trees, and training data

### Common Memory Issues

#### 1. Excessive Cut Storage
**Problem**: Thousands of cuts consuming gigabytes
**File**: `src/cut.rs`, `src/sddp/mod.rs`

**Analysis**:
```bash
# Profile cut storage with DHAT
python -m powers_profile run -c dhat -- run examples/05-large-scale-brazilian

# Check hotspots for cut-related allocations
jq '.hotspots[] | select(.function | contains("cut"))' \
  profiling_results/runs/latest/dhat/dhat_summary.json
```

**Optimization**:
- Implement cut selection to limit active cuts
- Use buffer pools for temporary cut calculations
- Consider memory-mapped storage for very large problems

#### 2. Scenario Tree Allocation
**Problem**: Large scenario trees in stochastic problems
**Files**: `src/scenario.rs`, `src/scenario_generator.rs`

**Analysis**:
```bash
# Track scenario generation with RSS timeline
python -m powers_profile run -c rss -- run examples/stochastic.json

# View growth curve in dashboard
python -m powers_profile dashboard
# Navigate to "Memory" → "RSS Timeline" chart
```

**Optimization**:
- Reuse scenario buffers across iterations
- Implement lazy scenario generation
- Use arena allocation for tree nodes

#### 3. LP Matrix Construction
**Problem**: Repeated matrix allocation in subproblem solving
**File**: `src/subproblem.rs`

**Analysis**:
```bash
# Use DHAT to identify matrix allocation hotspots
python -m powers_profile run -c dhat -- run examples/01-deterministic

# Filter for subproblem-related allocations
jq '.hotspots[] | select(.function | contains("subproblem"))' \
  profiling_results/runs/latest/dhat/dhat_summary.json
```

**Optimization**:
- Reuse matrix buffers from `src/memory/buffers.rs`
- Pre-allocate with known capacity
- Use in-place operations where possible

## Memory Optimization Workflow

### 1. Measure Baseline

```bash
# Quick RSS baseline
python -m powers_profile run -c rss -- run examples/01-deterministic

# Record peak memory
python -m powers_profile summary
# Look for "Peak RSS" metric

# Full memory profile (slower)
python -m powers_profile run -c memory -- run examples/01-deterministic
```

### 2. Identify Allocation Hotspots

```bash
# Run DHAT for allocation analysis
python -m powers_profile run -c dhat -- run examples/01-deterministic

# View hotspots in dashboard
python -m powers_profile dashboard
# Navigate to "Memory" tab → "Heap Allocations" section

# Or view raw hotspots
jq '.hotspots[:10]' profiling_results/runs/latest/dhat/dhat_summary.json
```

Look for:
- **Frequent allocations**: Short-lived Vec/String allocations
- **Large allocations**: Multi-megabyte objects
- **High total bytes**: Functions dominating total allocation volume

### 3. Apply Buffer Reuse

```rust
// Before: Allocate every iteration
for scenario in scenarios {
    let mut buffer = Vec::new();
    compute_values(&scenario, &mut buffer);
    // buffer dropped here - wasteful!
}

// After: Reuse buffer
let mut buffer = Vec::with_capacity(expected_size);
for scenario in scenarios {
    buffer.clear();
    compute_values(&scenario, &mut buffer);
    // buffer reused in next iteration
}
```

### 4. Verify Improvement

```bash
# Run profiling again
python -m powers_profile run -c memory -- run examples/01-deterministic

# Compare with baseline
python -m powers_profile compare <baseline-run-id> <new-run-id>

# View comparison dashboard
python -m powers_profile dashboard --baseline <baseline-run-id> <new-run-id>
```

## Memory Optimization Patterns

### Pattern 1: Pre-allocation with Capacity
```rust
// Bad: Repeated reallocation
let mut vec = Vec::new();
for i in 0..1000 {
    vec.push(i);  // Reallocates multiple times
}

// Good: Pre-allocate
let mut vec = Vec::with_capacity(1000);
for i in 0..1000 {
    vec.push(i);  // No reallocation
}
```

### Pattern 2: Buffer Pooling (from src/memory/buffers.rs)
```rust
// Use BufferPool for frequently allocated/freed buffers
let mut pool = BufferPool::new(1024);

for iteration in iterations {
    let mut buffer = pool.acquire();
    process(&mut buffer);
    pool.release(buffer);  // Reuse in next iteration
}
```

### Pattern 3: In-Place Operations
```rust
// Bad: Allocates new Vec
fn transform(input: &Vec<f64>) -> Vec<f64> {
    input.iter().map(|x| x * 2.0).collect()
}

// Good: Modifies in-place
fn transform_in_place(vec: &mut Vec<f64>) {
    vec.iter_mut().for_each(|x| *x *= 2.0);
}
```

### Pattern 4: Lazy Allocation
```rust
// Allocate only when needed
struct Cache {
    data: Option<Vec<f64>>,
}

impl Cache {
    fn get_or_create(&mut self) -> &mut Vec<f64> {
        self.data.get_or_insert_with(|| Vec::with_capacity(1000))
    }
}
```

## Integration with Nalgebra

POWE.RS uses `nalgebra = "0.33"` for matrix operations. Optimize nalgebra usage:

### Stack vs Heap Allocation
```rust
// Small matrices: Stack allocation (fast)
use nalgebra::{SMatrix, SVector};
let m: SMatrix<f64, 3, 3> = SMatrix::zeros();

// Large matrices: Heap allocation (required)
use nalgebra::{DMatrix, DVector};
let m: DMatrix<f64> = DMatrix::zeros(1000, 1000);
```

### Reuse Matrix Buffers
```rust
// Bad: Allocate every iteration
for stage in stages {
    let matrix = DMatrix::zeros(n, m);
    // Use matrix
}

// Good: Reuse allocation
let mut matrix = DMatrix::zeros(n, m);
for stage in stages {
    matrix.fill(0.0);
    // Use matrix
}
```

## Integration with HiGHS Solver

Memory optimization for `highs-sys = "1.6.4"` interaction:

### Minimize FFI Allocations
```rust
// Reuse workspace for multiple solves
struct SolverWorkspace {
    constraint_matrix: Vec<f64>,
    bounds: Vec<f64>,
    // ... other buffers
}

impl SolverWorkspace {
    fn solve(&mut self, problem: &Problem) -> Solution {
        // Reuse self.constraint_matrix instead of allocating
        self.constraint_matrix.clear();
        problem.fill_matrix(&mut self.constraint_matrix);
        // Call HiGHS
    }
}
```

### Efficient Dual Variable Extraction
```rust
// Pre-allocate dual variable buffer
let mut duals = vec![0.0; num_constraints];
unsafe {
    highs_sys::Highs_getDualValues(highs, duals.as_mut_ptr());
}
```

## Memory Analysis Checklist

- [ ] **Install profiling infrastructure**: `pip install -e profiling/`
- [ ] **Baseline RSS**: `python -m powers_profile run -c rss`
- [ ] **Full memory profile**: `python -m powers_profile run -c memory`
- [ ] **Identify top 5 hotspots**: Check dashboard "Memory" tab or `dhat_summary.json`
- [ ] **Check buffer reuse**: Review `src/memory/buffers.rs` patterns
- [ ] **Verify pre-allocation**: Ensure `Vec::with_capacity` used
- [ ] **Test with mimalloc**: `cargo bench --features mimalloc`
- [ ] **Compare improvements**: `python -m powers_profile compare <baseline> <target>`
- [ ] **View dashboard**: `python -m powers_profile dashboard`

## POWERS Profiling Commands Quick Reference

```bash
# Memory profiling
python -m powers_profile run -c rss      # Lightweight RSS only
python -m powers_profile run -c dhat     # Heap allocations (slow)
python -m powers_profile run -c massif   # Heap timeline (slow)
python -m powers_profile run -c memory   # All memory tools

# View results
python -m powers_profile summary          # CLI summary
python -m powers_profile dashboard        # Interactive HTML dashboard
python -m powers_profile history          # List all runs

# Compare runs
python -m powers_profile compare <baseline-id> <target-id>
python -m powers_profile dashboard --baseline <baseline-id> <target-id>

# Custom workload
python -m powers_profile run -c memory -- run examples/05-large-scale-brazilian
```

## Common Memory Issues in POWE.RS

### Issue 1: Cut Explosion
**Symptom**: Memory grows unbounded during training
**Files**: `src/cut.rs`, `src/sddp/mod.rs`
**Solution**: Implement cut selection (see `benches/README.md` - cut_selection benchmarks)

### Issue 2: Scenario Tree Growth
**Symptom**: Memory increases with scenario count
**Files**: `src/scenario.rs`, `src/scenario_generator.rs`
**Solution**: Implement scenario reduction or streaming generation

### Issue 3: LP Matrix Copies
**Symptom**: Frequent large allocations in subproblem solving
**File**: `src/subproblem.rs` (235KB)
**Solution**: Reuse matrix buffers, use in-place operations

## Best Practices

1. **Use buffer pools** - See `src/memory/buffers.rs` patterns
2. **Pre-allocate with capacity** - `Vec::with_capacity(expected_size)`
3. **Profile before optimizing** - Measure to find actual hotspots
4. **Test with mimalloc** - Can improve performance in parallel workloads
5. **Monitor peak memory** - Use `/usr/bin/time -v` or Heaptrack
6. **Verify correctness** - Run tests after memory optimization

## File References

- **Profiling infrastructure**: `profiling/powers_profile/` - Python package for all profiling
- **Memory collectors**: `profiling/powers_profile/collectors/{dhat,massif,rss,memory}.py`
- **Configuration**: `profiling/config/default.toml` - Profiling settings
- **Memory module**: `src/memory/mod.rs`, `src/memory/buffers.rs`
- **Large files**: `src/subproblem.rs` (235KB), `src/state.rs` (137KB), `src/sddp/mod.rs` (138KB)
- **Cargo config**: `Cargo.toml` - mimalloc dependency and feature flag
- **Scenario generation**: `src/scenario.rs`, `src/scenario_generator.rs`
- **Cut management**: `src/cut.rs`
- **Solver integration**: `src/solver.rs` (45KB)

## Related Skills

- **rust-profiling**: For CPU profiling and flamegraph analysis using POWERS profiling
- **rust-benchmarking**: For measuring performance impact of memory optimizations
- **hpc-optimization**: For parallel memory access patterns and cache optimization

## Resources

- **POWERS Profiling README**: `profiling/README.md` - Complete profiling guide
- **DHAT Documentation**: https://docs.rs/dhat/
- **DHAT Viewer**: https://nnethercote.github.io/dh_view/dh_view.html
- **Valgrind Massif**: https://valgrind.org/docs/manual/ms-manual.html
- **Rust Performance Book (Memory Chapter)**: https://nnethercote.github.io/perf-book/heap-allocations.html
