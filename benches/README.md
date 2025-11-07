# POWE.RS Performance Benchmarks

**Version**: 0.2.0  
**Framework**: Criterion.rs  
**Purpose**: Measure and track performance of critical code paths

## Overview

This directory contains performance benchmarks for POWE.RS. Benchmarks use [Criterion.rs](https://github.com/bheisler/criterion.rs) for statistical analysis and regression detection.

## Quick Start

### Run All Benchmarks
```bash
cargo bench
```

### Run Specific Benchmark
```bash
cargo bench --bench cut_selection
cargo bench --bench sddp_benchmarks
```

### Run Specific Test Within Benchmark
```bash
cargo bench --bench cut_selection -- dominated_cut
```

### Generate HTML Reports
Criterion automatically generates HTML reports in `target/criterion/`:
```bash
# After running benchmarks
firefox target/criterion/report/index.html
```

## Benchmark Categories

### 1. Core Algorithm Benchmarks

#### `sddp_benchmarks.rs` - SDDP Training Performance
Measures end-to-end SDDP training on various problem sizes.

**What it measures**:
- Full training iterations (forward + backward passes)
- Convergence time for different problem configurations
- Memory allocation during training

**Key benchmarks**:
- `deterministic_single_reservoir` - Simple 1-reservoir problem
- `stochastic_two_reservoir` - 2-reservoir with uncertainty
- `cascade_system` - Multi-reservoir cascade

**How to run**:
```bash
cargo bench --bench sddp_benchmarks
```

**Interpreting results**:
- Target: < 100ms per iteration for single reservoir
- Look for: Regression > 10% indicates performance issue
- Memory: Should stay constant across iterations

#### `comprehensive_benchmarks.rs` - Core Operations
Benchmarks fundamental SDDP operations.

**What it measures**:
- Subproblem solve times
- Cut evaluation speed
- State update performance

**How to run**:
```bash
cargo bench --bench comprehensive_benchmarks
```

### 2. Cut Management Benchmarks

#### `cut_selection.rs` - Cut Selection Algorithms
Measures performance of cut selection strategies.

**What it measures**:
- Dominated cut detection
- Level-based selection
- Batch selection performance

**Key benchmarks**:
- `select_dominated_cuts` - Domination detection speed
- `level_based_selection` - Level method performance
- `batch_cut_removal` - Bulk removal efficiency

**How to run**:
```bash
cargo bench --bench cut_selection
```

**Performance targets**:
- Dominated cut detection: < 1ms for 1000 cuts
- Selection: < 5ms for 10,000 cuts

#### `cut_id_lookup.rs` - Cut Indexing
Benchmarks cut lookup and retrieval operations.

**What it measures**:
- HashMap vs Vec lookup speeds
- Cut retrieval by ID
- Batch lookup performance

**How to run**:
```bash
cargo bench --bench cut_id_lookup
```

### 3. Parallelism Benchmarks

#### `parallel_efficiency.rs` - Parallel Forward Pass
Measures parallel execution efficiency.

**What it measures**:
- Parallel vs sequential forward pass
- Thread scaling (1, 2, 4, 8 threads)
- Overhead of parallelization

**How to run**:
```bash
cargo bench --bench parallel_efficiency
```

**Analyzing results**:
- Ideal: Linear scaling with thread count
- Reality: 70-80% efficiency is good
- Watch for: Negative scaling (parallel slower than sequential)

#### `par_generator.rs` - PAR Model Generation
Benchmarks PAR scenario generation performance.

**What it measures**:
- Noise generation speed
- Correlation application
- Memory allocation patterns

**How to run**:
```bash
cargo bench --bench par_generator
```

### 4. Memory Benchmarks

#### `memory_profiling.rs` - Memory Usage Analysis
Profiles memory allocation patterns (not a standard benchmark).

**What it measures**:
- Allocation counts
- Memory usage over time
- Reuse vs new allocation

**How to run**:
```bash
cargo bench --bench memory_profiling
```

**Note**: Uses custom instrumentation, not Criterion.

### 5. Numerical Benchmarks

#### `correlation_application.rs` - Correlation Matrix Operations
Benchmarks correlation application in scenario generation.

**What it measures**:
- Cholesky decomposition speed
- Matrix multiplication performance
- Cache efficiency

**How to run**:
```bash
cargo bench --bench correlation_application
```

#### `simd_dot_product.rs` - SIMD Optimizations
Benchmarks SIMD-accelerated operations.

**What it measures**:
- Dot product (SIMD vs scalar)
- Vector operations
- Auto-vectorization effectiveness

**How to run**:
```bash
cargo bench --bench simd_dot_product
```

**Performance targets**:
- SIMD should be 2-4x faster than scalar
- Verify auto-vectorization is working

### 6. Disabled Benchmarks

#### `lookup_structures.rs` - ⚠️ DISABLED
This benchmark is currently disabled due to API changes (NoiseLookupTable removed).

**Status**: Needs update for new uncertainty model API.

## Performance Baselines

### Reference Hardware
Baselines measured on:
- CPU: [Document your CPU]
- RAM: [Document your RAM]
- OS: Linux
- Rust: 1.70+

### Critical Path Targets

| Component | Operation | Target | Baseline |
|-----------|-----------|--------|----------|
| Cut Evaluation | 1000 cuts | < 100µs | TBD |
| Subproblem Solve | Single stage | < 10ms | TBD |
| Forward Pass | 10 scenarios | < 100ms | TBD |
| Backward Pass | 10 stages | < 500ms | TBD |
| Cut Selection | 1000 cuts | < 1ms | TBD |

**TODO**: Run `cargo bench` and record baseline values.

## Interpreting Results

### Criterion Output

```
cut_evaluation         time:   [85.234 µs 85.891 µs 86.612 µs]
                       change: [-2.1234% -0.8123% +0.5234%] (p = 0.42 > 0.05)
                       No change in performance detected.
```

**What this means**:
- **time**: [min, estimate, max] - Measured execution time
- **change**: Performance change from previous run
- **p-value**: Statistical significance (< 0.05 = significant change)

### Performance Regression Threshold

- **< 5% change**: Normal noise, no action needed
- **5-10% regression**: Investigate if consistent
- **> 10% regression**: Requires immediate investigation
- **> 10% improvement**: Verify correctness (too good to be true?)

### Common Issues

#### High Variance
```
time:   [10 ms 50 ms 100 ms]  # Large range
```
**Cause**: System load, thermal throttling, background processes  
**Solution**: Run benchmarks on idle system, multiple times

#### Unstable Results
```
Performance changed by ±20% between runs
```
**Cause**: Non-deterministic code, timing issues  
**Solution**: Ensure deterministic inputs, increase sample size

## Best Practices

### Running Benchmarks

1. **Close Other Applications**: Minimize system load
2. **Plug in Laptop**: Disable power saving
3. **Disable Turbo Boost**: For consistent results (optional)
4. **Run Multiple Times**: Criterion handles statistics
5. **Save Baselines**: `cargo bench --save-baseline main`

### Comparing Changes

```bash
# Save current performance as baseline
cargo bench --save-baseline main

# Make code changes
# ...

# Compare against baseline
cargo bench --baseline main
```

### Writing New Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn my_benchmark(c: &mut Criterion) {
    // Setup (not timed)
    let data = vec![1, 2, 3, 4, 5];
    
    c.bench_function("my_operation", |b| {
        b.iter(|| {
            // This is timed
            black_box(expensive_operation(&data))
        });
    });
}

criterion_group!(benches, my_benchmark);
criterion_main!(benches);
```

**Key points**:
- Use `black_box()` to prevent optimization
- Setup outside `iter()`
- Return results from `iter()` to prevent dead code elimination

## Performance Optimization Workflow

1. **Measure Current**: Run `cargo bench` to establish baseline
2. **Identify Hotspots**: Use `cargo flamegraph` or profiler
3. **Optimize**: Implement improvements
4. **Verify**: Run `cargo bench` again
5. **Test**: Ensure correctness with `cargo test`
6. **Document**: Record optimization in commit message

## Continuous Integration

Benchmarks can be run in CI to detect regressions:

```yaml
# .github/workflows/benchmarks.yml (example)
- name: Run benchmarks
  run: cargo bench --no-fail-fast
```

**Note**: CI results may vary due to hardware differences. Focus on detecting large regressions (> 20%).

## Profiling

For deeper analysis beyond benchmarks:

### CPU Profiling
```bash
# Install flamegraph
cargo install flamegraph

# Profile a benchmark
cargo flamegraph --bench sddp_benchmarks -- --bench
```

### Memory Profiling
```bash
# Use valgrind
valgrind --tool=massif cargo bench --bench memory_profiling

# Analyze with ms_print
ms_print massif.out.*
```

### Perf Analysis
```bash
# Record performance counters
perf record cargo bench --bench sddp_benchmarks

# Analyze
perf report
```

## Troubleshooting

### Benchmark Won't Compile
**Problem**: Code changes broke benchmark  
**Solution**: Update benchmark to match new API

### Benchmark Takes Too Long
**Problem**: Benchmark runs for minutes  
**Solution**: Reduce sample size or problem size

```rust
c.bench_function("slow_op", |b| {
    b.iter(|| slow_operation());
}).sample_size(10);  // Default is 100
```

### Results Don't Make Sense
**Problem**: Benchmark shows impossible performance  
**Solution**: Check for:
- Dead code elimination (use `black_box`)
- Compiler optimizations removing work
- Incorrect measurement scope

## Resources

- **Criterion.rs User Guide**: https://bheisler.github.io/criterion.rs/book/
- **Rust Performance Book**: https://nnethercote.github.io/perf-book/
- **POWE.RS Performance Issues**: See GitHub issues labeled `performance`

## Maintenance

- **Update baselines**: When making intentional performance changes
- **Review regularly**: Check for unexpected regressions monthly
- **Add benchmarks**: For new hot paths or optimizations
- **Archive old benchmarks**: Remove obsolete benchmarks

---

**Last Updated**: 2025-11-07  
**Maintained By**: POWE.RS Contributors

## Quick Reference

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench sddp_benchmarks

# Save baseline
cargo bench --save-baseline my_baseline

# Compare to baseline
cargo bench --baseline my_baseline

# Generate flamegraph
cargo flamegraph --bench sddp_benchmarks

# List all benchmarks
cargo bench --list
```
