---
name: rust-benchmarking
description: Guide agents to create and run performance benchmarks using Criterion.rs for the POWE.RS SDDP solver, measuring critical code paths and detecting performance regressions.
license: MIT
metadata:
  author: rjmalves
  version: "1.0"
  tags:
    - rust
    - benchmarking
    - criterion
    - performance
    - testing
---

# Rust Benchmarking with Criterion.rs

## Overview

This skill guides agents in creating, running, and analyzing performance benchmarks for the POWE.RS project using Criterion.rs. POWE.RS is a high-performance SDDP (Stochastic Dual Dynamic Programming) solver written in Rust, where performance is critical for solving large-scale optimization problems.

## Existing Infrastructure

### Benchmark Directory Structure
```
benches/
├── README.md              # Comprehensive benchmarking guide
├── sddp_e2e.rs           # End-to-end SDDP training benchmarks
└── simd_dot_product.rs   # SIMD-optimized operations benchmarks
```

### Cargo Configuration
The project's `Cargo.toml` includes:
- `criterion = { version = "0.5", features = ["html_reports"] }` in `[dev-dependencies]`
- Benchmark harness configuration:
  ```toml
  [[bench]]
  name = "simd_dot_product"
  harness = false
  
  [[bench]]
  name = "sddp_e2e"
  harness = false
  ```

### Performance Documentation
Refer to `benches/README.md` for:
- Complete benchmarking guide and best practices
- Performance targets for critical operations
- Interpretation of Criterion.rs output
- Profiling integration (flamegraph, perf, samply)

## Running Benchmarks

### Basic Commands
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark file
cargo bench --bench sddp_e2e
cargo bench --bench simd_dot_product

# Run specific test within benchmark
cargo bench --bench sddp_e2e -- forward_pass
```

### Baseline Comparison
```bash
# Save current performance as baseline
cargo bench --save-baseline main

# Make code changes, then compare
cargo bench --baseline main

# Look for:
# - change: [-2.1234% -0.8123% +0.5234%] (p = 0.42 > 0.05)
# - Regressions > 10% require investigation
```

### HTML Reports
```bash
# Criterion automatically generates HTML reports
cargo bench

# View reports (auto-generated in target/criterion/)
firefox target/criterion/report/index.html
```

## Performance Targets

Reference the critical path targets in `benches/README.md`:

| Component | Operation | Target | Notes |
|-----------|-----------|--------|-------|
| Cut Evaluation | 1000 cuts | < 100µs | Frequently called in backward pass |
| Subproblem Solve | Single stage | < 10ms | HiGHS LP solver call |
| Forward Pass | 10 scenarios | < 100ms | Parallel execution with rayon |
| Backward Pass | 10 stages | < 500ms | Sequential stage processing |
| Cut Selection | 1000 cuts | < 1ms | Dominated cut detection |

## Writing New Benchmarks

### Basic Structure
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_operation(c: &mut Criterion) {
    // Setup (not timed)
    let data = setup_test_data();
    
    c.bench_function("operation_name", |b| {
        b.iter(|| {
            // This code is timed
            black_box(expensive_operation(&data))
        });
    });
}

criterion_group!(benches, benchmark_operation);
criterion_main!(benches);
```

### Key Principles
1. **Use `black_box()`** - Prevents compiler from optimizing away the operation
2. **Setup outside `iter()`** - Only benchmark the actual operation, not setup
3. **Return values** - Return results from `iter()` to prevent dead code elimination
4. **Realistic inputs** - Use production-like data sizes and patterns

### Example: Benchmarking Cut Evaluation
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use powers_rs::cut::Cut;
use powers_rs::state::State;

fn bench_cut_evaluation(c: &mut Criterion) {
    // Setup: Create 1000 cuts and a state
    let cuts: Vec<Cut> = (0..1000)
        .map(|i| Cut::new(/* parameters */))
        .collect();
    let state = State::new(/* parameters */);
    
    c.bench_function("evaluate_1000_cuts", |b| {
        b.iter(|| {
            cuts.iter()
                .map(|cut| black_box(cut.evaluate(&state)))
                .sum::<f64>()
        });
    });
}

criterion_group!(benches, bench_cut_evaluation);
criterion_main!(benches);
```

## Integration with Profiling

For deeper analysis beyond benchmarks, combine with profiling tools:

```bash
# CPU profiling with flamegraph
cargo flamegraph --bench sddp_e2e -- --bench

# Performance counters with perf
perf record cargo bench --bench sddp_e2e
perf report

# Interactive profiling with samply
samply record cargo bench --bench sddp_e2e -- --bench
```

See the `rust-profiling` skill for detailed profiling guidance.

## Interpreting Results

### Criterion Output Format
```
cut_evaluation         time:   [85.234 µs 85.891 µs 86.612 µs]
                       change: [-2.1234% -0.8123% +0.5234%] (p = 0.42 > 0.05)
                       No change in performance detected.
```

**Interpreting:**
- **time**: [min, estimate, max] measured execution time
- **change**: Performance change from previous run
- **p-value**: < 0.05 indicates statistically significant change

### Regression Thresholds
- **< 5% change**: Normal noise, no action needed
- **5-10% regression**: Investigate if consistent across runs
- **> 10% regression**: Requires immediate investigation
- **> 10% improvement**: Verify correctness (surprisingly large gains may indicate bugs)

## Best Practices

1. **Minimize system load** - Close other applications, plug in laptop
2. **Run multiple times** - Criterion handles statistical analysis
3. **Use realistic data** - Match production problem sizes
4. **Document baselines** - Save baselines when making intentional performance changes
5. **Test after optimizing** - Always run `cargo test` to ensure correctness

## Benchmark Categories for POWE.RS

### Core Algorithm Benchmarks
- **SDDP training** (`sddp_e2e.rs`) - End-to-end iterations
- **Subproblem solving** - LP solver invocation
- **Cut evaluation** - Piecewise linear approximation

### Numerical Operations
- **SIMD operations** (`simd_dot_product.rs`) - Vectorized computations
- **Matrix operations** - nalgebra performance
- **Correlation application** - Scenario generation

### Parallelism
- **Forward pass** - rayon parallel scenarios
- **Thread scaling** - 1, 2, 4, 8 threads
- **Overhead measurement** - Parallel vs sequential

### Memory Performance
- **Buffer reuse** - src/memory/buffers.rs patterns
- **Allocation patterns** - Using mimalloc feature flag

## File References

- **Documentation**: `benches/README.md`
- **Example benchmarks**: `benches/sddp_e2e.rs`, `benches/simd_dot_product.rs`
- **Memory module**: `src/memory/buffers.rs`
- **Timing module**: `src/timing/` (atomic.rs, collector.rs, guard.rs, metrics.rs)
- **Configuration**: `Cargo.toml` (benchmark entries, criterion dependency)
- **Profile settings**: `[profile.release]` with `debug = true` in `Cargo.toml`

## Related Skills

- **rust-profiling**: For CPU profiling with flamegraph, perf, samply
- **hpc-optimization**: For SIMD and parallelization optimization
- **rust-memory-analysis**: For memory profiling and allocation analysis

## Common Issues

### Benchmark Won't Compile
Check that the benchmark uses the current public API. Update imports and function signatures if the codebase has evolved.

### Benchmark Takes Too Long
Reduce sample size or problem size:
```rust
c.bench_function("slow_op", |b| {
    b.iter(|| slow_operation());
}).sample_size(10);  // Default is 100
```

### Results Don't Make Sense
- Ensure `black_box()` is used to prevent dead code elimination
- Check that the operation isn't being optimized away
- Verify measurement scope includes only the target operation

## Example Workflow

1. **Identify optimization target** - Profile to find hotspots
2. **Create baseline benchmark** - Measure current performance
3. **Implement optimization** - Make code changes
4. **Run benchmark** - Compare against baseline
5. **Verify correctness** - Run tests with `cargo test`
6. **Document** - Record optimization in commit message

## Resources

- **Criterion.rs User Guide**: https://bheisler.github.io/criterion.rs/book/
- **Rust Performance Book**: https://nnethercote.github.io/perf-book/
- **Project Documentation**: `benches/README.md`
