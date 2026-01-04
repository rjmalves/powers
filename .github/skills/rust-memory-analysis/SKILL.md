---
name: rust-memory-analysis
description: Guide agents to analyze memory usage, optimize allocations, and reduce memory footprint in the POWE.RS SDDP solver using DHAT, Heaptrack, Valgrind, and mimalloc.
license: MIT
metadata:
  author: rjmalves
  version: "1.0"
  tags:
    - rust
    - memory
    - profiling
    - allocation
    - optimization
    - mimalloc
---

# Rust Memory Analysis and Optimization

## Overview

This skill guides agents in analyzing and optimizing memory usage for the POWE.RS SDDP solver. Memory efficiency is critical for HPC workloads, where large-scale optimization problems can consume gigabytes of memory and benefit significantly from allocation optimization.

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

## Memory Profiling Tools

### 1. DHAT (Heap Profiler)
**Best for**: Detailed heap allocation analysis, identifying hot allocation sites

**Setup**:
```toml
# Add to Cargo.toml [dev-dependencies]
dhat = "0.3"
```

**Instrumentation**:
```rust
// In main.rs or test file
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    
    // Your code here
}
```

**Run**:
```bash
# Build with instrumentation
cargo build --release --features dhat-heap

# Run and generate dhat-heap.json
./target/release/powers --config examples/config.json

# View with DHAT viewer
# Upload dhat-heap.json to https://nnethercote.github.io/dh_view/dh_view.html
```

**Metrics**:
- **Total allocations**: Number of malloc/free calls
- **Total bytes allocated**: Peak heap usage
- **Hot allocation sites**: Functions allocating most memory
- **Allocation lifetimes**: Short-lived vs long-lived allocations

### 2. Heaptrack (Linux)
**Best for**: Real-time heap tracking and visualization

```bash
# Install
sudo apt-get install heaptrack heaptrack-gui

# Record heap usage
heaptrack target/release/powers --config examples/config.json

# Analyze with GUI
heaptrack_gui heaptrack.powers.*.gz
```

**View**:
- Allocation timeline
- Peak memory usage
- Top allocating functions
- Call graphs for allocations

### 3. Valgrind Massif
**Best for**: Detailed heap and stack profiling

```bash
# Record memory usage
valgrind --tool=massif \
  target/release/powers --config examples/config.json

# Visualize with ms_print
ms_print massif.out.*

# Or use massif-visualizer GUI
massif-visualizer massif.out.*
```

**Metrics**:
- Heap size over time
- Stack size over time
- Allocation tree (who allocated what)

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
# Profile cut storage
cargo flamegraph --bench sddp_e2e -- --bench
# Look for Vec::push in cut management code
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
# Track scenario generation allocations
heaptrack target/release/powers --config examples/stochastic.json
```

**Optimization**:
- Reuse scenario buffers across iterations
- Implement lazy scenario generation
- Use arena allocation for tree nodes

#### 3. LP Matrix Construction
**Problem**: Repeated matrix allocation in subproblem solving
**File**: `src/subproblem.rs`

**Analysis**:
Use DHAT to identify allocation sites in `Subproblem::solve()`

**Optimization**:
- Reuse matrix buffers from `src/memory/buffers.rs`
- Pre-allocate with known capacity
- Use in-place operations where possible

## Memory Optimization Workflow

### 1. Measure Baseline
```bash
# Heap profile
heaptrack target/release/powers --config examples/config.json

# Record peak memory
/usr/bin/time -v target/release/powers --config examples/config.json
# Look for "Maximum resident set size"
```

### 2. Identify Allocation Hotspots
```bash
# DHAT analysis
cargo build --release --features dhat-heap
./target/release/powers --config examples/config.json
# Upload dhat-heap.json to viewer
```

Look for:
- **Frequent allocations**: Short-lived Vec/String allocations
- **Large allocations**: Multi-megabyte objects
- **Leaked allocations**: Memory not freed (likely a bug)

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
# Compare memory usage
heaptrack target/release/powers --config examples/config.json

# Benchmark performance impact
cargo bench --bench sddp_e2e
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

- [ ] Profile with Heaptrack: `heaptrack target/release/powers`
- [ ] Identify top 5 allocation sites
- [ ] Check for buffer reuse opportunities in `src/memory/buffers.rs`
- [ ] Verify pre-allocation with `Vec::with_capacity`
- [ ] Test with mimalloc: `cargo bench --features mimalloc`
- [ ] Measure peak memory: `/usr/bin/time -v`
- [ ] Profile with DHAT for detailed allocation analysis
- [ ] Check for memory leaks (allocations without corresponding frees)

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

- **Memory module**: `src/memory/mod.rs`, `src/memory/buffers.rs`
- **Large files**: `src/subproblem.rs` (235KB), `src/state.rs` (137KB), `src/sddp/mod.rs` (138KB)
- **Configuration**: `Cargo.toml` - mimalloc dependency and feature flag
- **Scenario generation**: `src/scenario.rs`, `src/scenario_generator.rs`
- **Cut management**: `src/cut.rs`
- **Solver integration**: `src/solver.rs` (45KB)

## Related Skills

- **rust-profiling**: For CPU profiling to identify allocation-heavy code paths
- **rust-benchmarking**: For measuring performance impact of memory optimizations
- **hpc-optimization**: For parallel memory access patterns and cache optimization

## Resources

- **DHAT Documentation**: https://docs.rs/dhat/
- **DHAT Viewer**: https://nnethercote.github.io/dh_view/dh_view.html
- **Heaptrack**: https://github.com/KDE/heaptrack
- **Mimalloc**: https://github.com/microsoft/mimalloc
- **Rust Performance Book (Memory Chapter)**: https://nnethercote.github.io/perf-book/heap-allocations.html
