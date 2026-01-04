---
name: hpc-optimization
description: Optimize POWE.RS SDDP solver for high-performance computing workloads using SIMD vectorization, parallel execution with rayon, cache optimization, and efficient data layouts.
license: MIT
metadata:
  author: rjmalves
  version: "1.0"
  tags:
    - rust
    - hpc
    - simd
    - parallelization
    - rayon
    - performance
    - optimization
---

# HPC Optimization for Rust

## Overview

This skill guides agents in optimizing the POWE.RS SDDP solver for high-performance computing (HPC) workloads. HPC optimization is critical for solving large-scale stochastic optimization problems with thousands of stages and scenarios.

## Parallelization with Rayon

### Dependency
```toml
# From Cargo.toml
rayon = "1.10.0"
```

Rayon provides work-stealing parallelism with near-linear scaling for data-parallel workloads.

### Thread Configuration
```rust
use rayon::prelude::*;
use num_cpus;

// Configure thread pool
pub fn configure_rayon(num_threads: Option<usize>) {
    let threads = num_threads.unwrap_or_else(num_cpus::get);
    
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .expect("Failed to build rayon thread pool");
}
```

### Environment Variable
```bash
# Set number of threads at runtime
RAYON_NUM_THREADS=8 cargo run --release

# Auto-detect CPU count (from num_cpus = "1.16")
cargo run --release
```

### Parallel Forward Pass Pattern

**Most critical parallelization point in SDDP**

```rust
use rayon::prelude::*;

pub fn parallel_forward_pass(
    &self,
    num_scenarios: usize,
) -> Result<Vec<ScenarioResult>, Error> {
    // Parallel scenario simulation
    (0..num_scenarios)
        .into_par_iter()
        .map(|scenario_idx| {
            self.simulate_scenario(scenario_idx)
        })
        .collect()
}

fn simulate_scenario(&self, scenario_idx: usize) -> Result<ScenarioResult, Error> {
    let mut cost = 0.0;
    let mut state = self.initial_state.clone();
    
    // Sequential stage simulation within scenario
    for stage in 0..self.num_stages {
        let scenario = self.generate_scenario(scenario_idx, stage);
        let solution = self.solve_subproblem(&state, &scenario)?;
        cost += solution.objective_value;
        state = solution.next_state;
    }
    
    Ok(ScenarioResult { cost, final_state: state })
}
```

### Parallel Performance Targets

From `benches/README.md`:
- **Forward Pass (10 scenarios)**: < 100ms
- **Thread Scaling**: 70-80% efficiency is good
- **Parallel Efficiency**: Time(1 thread) / (N × Time(N threads))

### Benchmark Thread Scaling
```bash
# Test scalability
for threads in 1 2 4 8; do
  echo "Testing with $threads threads:"
  RAYON_NUM_THREADS=$threads cargo bench --bench sddp_e2e
done
```

Expected results:
```
1 thread:  1000ms
2 threads: 550ms  (91% efficiency)
4 threads: 300ms  (83% efficiency)
8 threads: 175ms  (71% efficiency)
```

### When to Use Rayon

**✅ Use Rayon for:**
- Forward pass scenario simulation (independent scenarios)
- Batch cut evaluation (independent cuts)
- Parallel scenario generation
- Independent subproblem solves

**❌ Don't Use Rayon for:**
- Backward pass (sequential dependency between stages)
- Small loops (< 1000 iterations)
- Operations faster than 1ms (overhead dominates)

## SIMD Optimizations

### Feature Flag
```toml
# From Cargo.toml
[features]
simd-optimizations = []
```

Enable with:
```bash
cargo build --release --features simd-optimizations
```

### SIMD Benchmark
**File**: `benches/simd_dot_product.rs`

Performance targets:
- **SIMD should be 2-4x faster than scalar**
- **Auto-vectorization verification**

```bash
# Benchmark SIMD operations
cargo bench --features simd-optimizations --bench simd_dot_product
```

### SIMD Dot Product Example
```rust
#[cfg(feature = "simd-optimizations")]
pub fn dot_product_simd(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut sum = 0.0;
    
    unsafe {
        // Process 4 f64 values at a time (AVX)
        let chunks = len / 4;
        let mut sum_vec = _mm256_setzero_pd();
        
        for i in 0..chunks {
            let idx = i * 4;
            let a_vec = _mm256_loadu_pd(a.as_ptr().add(idx));
            let b_vec = _mm256_loadu_pd(b.as_ptr().add(idx));
            let prod = _mm256_mul_pd(a_vec, b_vec);
            sum_vec = _mm256_add_pd(sum_vec, prod);
        }
        
        // Horizontal sum of vector
        let mut tmp = [0.0; 4];
        _mm256_storeu_pd(tmp.as_mut_ptr(), sum_vec);
        sum = tmp.iter().sum();
        
        // Handle remainder
        for i in (chunks * 4)..len {
            sum += a[i] * b[i];
        }
    }
    
    sum
}

// Fallback for non-SIMD builds
#[cfg(not(feature = "simd-optimizations"))]
pub fn dot_product_simd(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
```

### Auto-Vectorization Tips
```rust
// ✅ CORRECT - Compiler can auto-vectorize
#[inline]
pub fn scale_vector(vec: &mut [f64], scale: f64) {
    for x in vec.iter_mut() {
        *x *= scale;
    }
}

// ❌ WRONG - Complex logic prevents vectorization
pub fn scale_vector_bad(vec: &mut [f64], scale: f64) {
    for i in 0..vec.len() {
        if vec[i] > 0.0 {
            vec[i] *= scale;
        } else {
            vec[i] *= scale * 0.5;  // Branch prevents SIMD
        }
    }
}
```

### Verify Auto-Vectorization
```bash
# Check assembly for SIMD instructions (vmulpd, vaddpd, etc.)
cargo rustc --release -- --emit asm

# Look for instructions like:
# vmulpd    %ymm0, %ymm1, %ymm2  # AVX vector multiply
# vaddpd    %ymm0, %ymm1, %ymm2  # AVX vector add
```

## Matrix Operations with nalgebra

### Dependency
```toml
# From Cargo.toml
nalgebra = "0.33"
```

### Efficient Matrix Usage
```rust
use nalgebra::{DMatrix, DVector};

// ✅ CORRECT - Pre-allocate and reuse
pub struct MatrixWorkspace {
    a_matrix: DMatrix<f64>,
    b_vector: DVector<f64>,
}

impl MatrixWorkspace {
    pub fn solve(&mut self, problem: &Problem) -> DVector<f64> {
        // Reuse allocated matrices
        self.a_matrix.fill(0.0);
        problem.fill_matrix(&mut self.a_matrix);
        
        // Solve Ax = b
        self.a_matrix.lu().solve(&self.b_vector)
    }
}

// ❌ WRONG - Allocate every time
pub fn solve_bad(problem: &Problem) -> DVector<f64> {
    let a_matrix = DMatrix::zeros(problem.n, problem.m);  // Allocate
    let b_vector = DVector::zeros(problem.n);             // Allocate
    // ...
}
```

### BLAS Integration
Nalgebra uses optimized BLAS when available:

```toml
# Optional: Use Intel MKL for maximum performance
[dependencies]
nalgebra = { version = "0.33", features = ["matrixmultiply"] }
```

## Cache Optimization

### Cache-Friendly Data Layouts

#### Structure of Arrays (SoA) Pattern
```rust
// ❌ WRONG - Array of Structures (AoS) - poor cache locality
struct Scenario {
    inflow: f64,
    demand: f64,
    price: f64,
}
let scenarios: Vec<Scenario> = vec![...];

// Processing causes cache misses
for scenario in &scenarios {
    process_inflow(scenario.inflow);  // Access scattered in memory
}

// ✅ CORRECT - Structure of Arrays (SoA) - excellent cache locality
struct ScenarioData {
    inflows: Vec<f64>,   // Contiguous in memory
    demands: Vec<f64>,   // Contiguous in memory
    prices: Vec<f64>,    // Contiguous in memory
}

impl ScenarioData {
    fn process_inflows(&self) {
        // Sequential access - great for cache and SIMD
        for inflow in &self.inflows {
            process_inflow(*inflow);
        }
    }
}
```

#### Memory Alignment
```rust
// Ensure alignment for SIMD
#[repr(align(32))]  // Align to 32 bytes for AVX
pub struct AlignedBuffer {
    data: Vec<f64>,
}
```

### Cache-Oblivious Algorithms
```rust
// Matrix transpose with cache-friendly blocking
pub fn transpose_blocked(
    src: &DMatrix<f64>,
    dst: &mut DMatrix<f64>,
    block_size: usize,
) {
    let n = src.nrows();
    let m = src.ncols();
    
    for i_block in (0..n).step_by(block_size) {
        for j_block in (0..m).step_by(block_size) {
            // Process small blocks that fit in cache
            let i_end = (i_block + block_size).min(n);
            let j_end = (j_block + block_size).min(m);
            
            for i in i_block..i_end {
                for j in j_block..j_end {
                    dst[(j, i)] = src[(i, i)];
                }
            }
        }
    }
}
```

### Prefetching
```rust
// Manual prefetch for predictable access patterns
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn process_with_prefetch(data: &[f64]) {
    const PREFETCH_DISTANCE: usize = 8;
    
    for i in 0..data.len() {
        // Prefetch future elements
        if i + PREFETCH_DISTANCE < data.len() {
            unsafe {
                _mm_prefetch(
                    data.as_ptr().add(i + PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }
        
        // Process current element
        process_value(data[i]);
    }
}
```

## Memory Bandwidth Optimization

### Minimize Data Movement
```rust
// ✅ CORRECT - In-place operations
pub fn normalize_in_place(vec: &mut [f64]) {
    let sum: f64 = vec.iter().sum();
    vec.iter_mut().for_each(|x| *x /= sum);
}

// ❌ WRONG - Creates new allocation
pub fn normalize_copy(vec: &[f64]) -> Vec<f64> {
    let sum: f64 = vec.iter().sum();
    vec.iter().map(|x| x / sum).collect()  // Extra allocation!
}
```

### Stream Through Large Data
```rust
// Process large datasets in chunks
pub fn process_large_dataset(data: &[f64], chunk_size: usize) -> Vec<f64> {
    data.chunks(chunk_size)
        .map(|chunk| {
            // Process chunk (fits in cache)
            chunk.iter().sum::<f64>() / chunk.len() as f64
        })
        .collect()
}
```

## Parallel SDDP Patterns

### Parallel Forward Pass (Primary Pattern)
```rust
use rayon::prelude::*;

pub fn forward_pass(&self, num_scenarios: usize) -> Result<Vec<f64>, Error> {
    // Each scenario is independent - perfect for parallelization
    (0..num_scenarios)
        .into_par_iter()
        .map(|scenario_idx| {
            self.simulate_single_scenario(scenario_idx)
        })
        .collect()
}
```

### Sequential Backward Pass
```rust
// Backward pass has stage-to-stage dependency - MUST be sequential
pub fn backward_pass(&mut self) -> Result<Vec<Cut>, Error> {
    let mut all_cuts = Vec::new();
    
    // Sequential: stage t depends on cuts from stage t+1
    for stage in (1..self.num_stages).rev() {
        // But scenarios within a stage CAN be parallel
        let stage_cuts: Vec<Cut> = self.scenarios[stage]
            .par_iter()
            .map(|scenario| self.generate_cut(stage, scenario))
            .collect::<Result<Vec<_>, _>>()?;
        
        all_cuts.extend(stage_cuts);
    }
    
    Ok(all_cuts)
}
```

### Parallel Cut Evaluation
```rust
// Evaluate many cuts in parallel
pub fn evaluate_cuts_parallel(
    cuts: &[Cut],
    state: &State,
) -> Vec<f64> {
    cuts.par_iter()
        .map(|cut| cut.evaluate(state))
        .collect()
}
```

## Performance Profiling Integration

### Cache Performance Analysis
```bash
# Profile cache misses with perf
perf stat -e cache-references,cache-misses,LLC-loads,LLC-load-misses \
  cargo bench --bench sddp_e2e

# Targets:
# - L1 miss rate < 10%
# - LLC miss rate < 1%
```

### SIMD Verification
```bash
# Check SIMD instruction usage
perf stat -e fp_arith_inst_retired.256b_packed_double \
  cargo bench --features simd-optimizations --bench simd_dot_product

# Should show significant AVX instruction usage
```

## Optimization Checklist for Large Files

POWE.RS has several large performance-critical files:

### `src/subproblem.rs` (235KB)
- [ ] Reuse LP matrices across solves (buffer pooling)
- [ ] Use sparse matrix format for large problems
- [ ] Pre-allocate solution vectors
- [ ] Profile solver invocation overhead

### `src/state.rs` (137KB)
- [ ] Use SoA layout for state arrays
- [ ] Minimize state copying (use references)
- [ ] Cache-friendly state transition updates
- [ ] Vectorize state arithmetic operations

### `src/sddp/mod.rs` (138KB)
- [ ] Parallel forward pass with rayon
- [ ] Efficient cut storage and retrieval
- [ ] Reuse scenario buffers
- [ ] Profile iteration overhead

## Benchmarking HPC Optimizations

### Before/After Comparison
```bash
# Baseline
cargo bench --bench sddp_e2e --save-baseline before-hpc

# Apply optimization (e.g., enable SIMD)
cargo bench --bench sddp_e2e --features simd-optimizations --baseline before-hpc

# Check improvement
# Expected: 10-30% improvement for SIMD
# Expected: 50-70% improvement for parallelization (on 8 cores)
```

### Thread Scaling Benchmark
```bash
# Measure parallel efficiency
for t in 1 2 4 8 16; do
  RAYON_NUM_THREADS=$t cargo bench --bench sddp_e2e -- --quiet | grep "time:"
done

# Calculate efficiency: T(1) / (N * T(N))
```

## Best Practices

1. **Profile first** - Use `cargo flamegraph` to find hotspots before optimizing
2. **Benchmark** - Measure impact with `cargo bench` before/after
3. **Cache locality** - Prefer SoA over AoS for numerical arrays
4. **SIMD alignment** - Ensure data is properly aligned for vector instructions
5. **Rayon for data parallelism** - Use for independent scenarios/cuts
6. **Avoid over-parallelization** - Don't parallelize operations < 1ms
7. **Reuse allocations** - Use buffer pools for hot paths

## Common Pitfalls

### 1. False Sharing
```rust
// ❌ WRONG - Threads write to adjacent memory
let mut results = vec![0.0; num_threads];
(0..num_threads).into_par_iter().for_each(|i| {
    results[i] = expensive_computation(i);  // False sharing!
});

// ✅ CORRECT - Thread-local accumulation
let results: Vec<f64> = (0..num_threads)
    .into_par_iter()
    .map(|i| expensive_computation(i))
    .collect();
```

### 2. Over-Parallelization
```rust
// ❌ WRONG - Overhead dominates for small operations
let sum: f64 = vec.par_iter().sum();  // vec is small

// ✅ CORRECT - Sequential for small data
let sum: f64 = vec.iter().sum();
```

### 3. Unaligned SIMD Access
```rust
// ⚠️ CAREFUL - May cause performance penalty
let data = vec![1.0, 2.0, 3.0];  // May not be aligned
unsafe {
    _mm256_loadu_pd(data.as_ptr())  // Use loadu (unaligned load)
}
```

## File References

- **Parallelization**: Uses `rayon = "1.10.0"`
- **Matrix operations**: Uses `nalgebra = "0.33"`
- **Thread config**: Uses `num_cpus = "1.16"`
- **SIMD benchmarks**: `benches/simd_dot_product.rs`
- **E2E benchmarks**: `benches/sddp_e2e.rs`
- **Feature flag**: `simd-optimizations` in `Cargo.toml`
- **Large files**: `src/subproblem.rs` (235KB), `src/state.rs` (137KB), `src/sddp/mod.rs` (138KB)

## Related Skills

- **rust-benchmarking**: For measuring optimization impact
- **rust-profiling**: For identifying bottlenecks and verifying improvements
- **rust-memory-analysis**: For cache and allocation optimization
- **highs-integration**: For solver parallelization strategies

## Resources

- **Rayon Documentation**: https://docs.rs/rayon/
- **Rust SIMD**: https://rust-lang.github.io/portable-simd/
- **Intel Intrinsics Guide**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- **Rust Performance Book**: https://nnethercote.github.io/perf-book/
- **nalgebra Performance**: https://nalgebra.org/performance_tricks/
