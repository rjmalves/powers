# Benchmarking Guide

This guide explains how to run, interpret, and create benchmarks for POWE.RS.

## Overview

POWE.RS uses [Criterion.rs](https://github.com/bheisler/criterion.rs) for statistical benchmarking. Benchmarks are located in the `benches/` directory and measure performance-critical operations.

## Available Benchmarks

### Core Algorithm Benchmarks

#### `sddp_benchmarks`
Performance of SDDP algorithm components:
- Full training runs (2-stage, 12-stage, 60-stage)
- Forward pass performance
- Backward pass performance
- Convergence characteristics

**Run**:
```bash
cargo bench --bench sddp_benchmarks
```

#### `simulation_memory`
Memory usage and simulation performance validation:
- Peak memory usage for varying scenario counts
- Simulation throughput (scenarios/second)
- Trajectory extraction overhead
- CSV export performance

**Run**:
```bash
cargo bench --bench simulation_memory

# With detailed memory profiling (Linux)
heaptrack cargo bench --bench simulation_memory -- --sample-size 10
```

#### `comprehensive_benchmarks`
End-to-end performance testing:
- Training convergence time
- Full workflow benchmarks
- Policy evaluation performance

**Run**:
```bash
cargo bench --bench comprehensive_benchmarks
```

### Component Benchmarks

#### `cut_selection`
Cut selection and management:
- Cut filtering algorithms
- Dominance checking
- Cut pool operations

#### `subproblem_solve`
Subproblem solver interface:
- LP solve time
- Basis warm-starting effectiveness
- Solver initialization overhead

#### `state_operations`
State manipulation performance:
- State hashing
- State comparison
- State serialization

#### `parallel_efficiency`
Parallelization overhead:
- Thread pool efficiency
- Rayon performance
- Scalability with thread count

#### `scenario_benchmarks`
Scenario generation and sampling:
- AR model scenario generation
- Lognormal transformation
- Correlation application
- SAA construction

#### `ar_dynamics`
AR model dynamics:
- AR(p) coefficient updates
- State evolution
- Numerical stability

#### `correlation_application`
Correlation matrix operations:
- Cholesky decomposition
- Correlation application to scenarios
- Large-scale correlation matrices

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench --bench simulation_memory

# Run specific benchmark within a suite
cargo bench --bench simulation_memory -- simulation_throughput

# Run with custom sample size (faster, less precise)
cargo bench --bench simulation_memory -- --sample-size 10

# Run with detailed output
cargo bench --bench simulation_memory -- --verbose
```

### Viewing Results

Criterion generates HTML reports in `target/criterion/`:

```bash
# Open the report index
open target/criterion/report/index.html

# Or on Linux
xdg-open target/criterion/report/index.html
```

Reports include:
- Statistical analysis (mean, median, std dev)
- Confidence intervals
- Performance history over multiple runs
- Comparison with previous runs

### Comparing with Baseline

```bash
# Save current results as baseline
cargo bench --bench simulation_memory -- --save-baseline my_baseline

# Compare new code with baseline
cargo bench --bench simulation_memory -- --baseline my_baseline

# Script for memory comparison
./scripts/compare_simulation_memory.sh
```

## Memory Profiling

### Linux: heaptrack

```bash
# Install heaptrack
sudo apt install heaptrack

# Profile benchmark
heaptrack cargo bench --bench simulation_memory -- --sample-size 10

# Analyze results
heaptrack_gui heaptrack.cargo.*.zst
```

### Linux: Valgrind Massif

```bash
# Profile memory usage
valgrind --tool=massif --massif-out-file=massif.out \
  cargo bench --bench simulation_memory -- --sample-size 10

# View report
ms_print massif.out

# Or use GUI
massif-visualizer massif.out
```

### macOS: Instruments

```bash
# Build in release mode
cargo build --release --bench simulation_memory

# Profile with Instruments
instruments -t "Allocations" target/release/deps/simulation_memory-* -- --bench
```

## Performance Guidelines

### Interpreting Results

Criterion provides several metrics:

- **Mean**: Average execution time (most common metric)
- **Median**: Middle value (more robust to outliers)
- **Std Dev**: Variability in measurements
- **R²**: Goodness of fit (closer to 1.0 is better)

**Example output**:
```
simulation_memory/memory_peak/100
                        time:   [78.234 ms 78.891 ms 79.612 ms]
                        change: [-2.1345% -0.8234% +0.5432%]
```

Interpretation:
- Mean time: 78.891 ms
- 95% confidence interval: [78.234 ms, 79.612 ms]
- Change from previous run: -0.8% (improvement)

### Performance Baselines

Expected performance on modern hardware (8 cores, 3.0 GHz):

| Benchmark                | Expected Time | Notes                           |
|--------------------------|---------------|---------------------------------|
| 2-stage training         | ~5 ms         | Minimal problem                 |
| 12-stage training        | ~50 ms        | Small realistic problem         |
| 60-stage training        | ~800 ms       | Medium-scale problem            |
| 100 scenario simulation  | ~80 ms        | With 24 stages                  |
| 1000 scenario simulation | ~800 ms       | With 24 stages                  |
| Subproblem solve         | ~500 µs       | Single LP solve with warm start |
| Cut selection            | ~50 µs        | 1000 cuts, dominance check      |

### Regression Thresholds

CI fails if benchmarks regress by more than:
- **Time**: >10% slower
- **Memory**: >10% more memory
- **Throughput**: >5% fewer operations/sec

## Creating New Benchmarks

### Template

Create a new file in `benches/my_benchmark.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_my_operation(c: &mut Criterion) {
    c.bench_function("my_operation", |b| {
        // Setup code (not timed)
        let data = setup_data();
        
        b.iter(|| {
            // Code to benchmark (timed)
            black_box(my_operation(&data))
        });
    });
}

criterion_group!(benches, bench_my_operation);
criterion_main!(benches);
```

Add to `Cargo.toml`:
```toml
[[bench]]
name = "my_benchmark"
harness = false
```

### Best Practices

1. **Use `black_box`**: Prevents compiler from optimizing away unused results
   ```rust
   black_box(result)
   ```

2. **Setup outside benchmark**: Don't time initialization
   ```rust
   let data = setup_data();  // Not timed
   b.iter(|| {
       process(black_box(&data))  // Timed
   });
   ```

3. **Benchmark groups**: Test different input sizes
   ```rust
   let mut group = c.benchmark_group("my_group");
   for size in [100, 1000, 10000] {
       group.bench_with_input(BenchmarkId::new("op", size), &size, |b, &size| {
           b.iter(|| operation(size));
       });
   }
   group.finish();
   ```

4. **Realistic data**: Use production-like inputs
   ```rust
   // Good: Realistic system
   let system = create_cascade_system(20, 10);
   
   // Bad: Trivial system
   let system = System::new(vec![], vec![], vec![], vec![]);
   ```

5. **Memory benchmarks**: Sample periodically
   ```rust
   let mut stats = MemoryStats::new();
   stats.sample();  // Before operation
   
   let result = expensive_operation();
   
   stats.sample();  // After operation
   stats.report();
   ```

### Avoiding Common Pitfalls

❌ **Don't**: Benchmark trivial operations
```rust
// Too fast, measurement noise dominates
b.iter(|| x + y);
```

❌ **Don't**: Forget to use results
```rust
// Compiler might optimize this away
b.iter(|| compute_something());
```

✅ **Do**: Use `black_box` for results
```rust
b.iter(|| black_box(compute_something()));
```

❌ **Don't**: Include setup in benchmark
```rust
b.iter(|| {
    let data = vec![0; 1000];  // This gets timed!
    process(&data)
});
```

✅ **Do**: Setup outside iterator
```rust
let data = vec![0; 1000];  // Not timed
b.iter(|| process(black_box(&data)));
```

## Continuous Integration

### GitHub Actions

Benchmarks run on every PR (non-blocking):

```yaml
- name: Run benchmarks
  run: |
    cargo bench --bench simulation_memory -- --sample-size 10
    
- name: Check for regressions
  run: |
    ./scripts/compare_simulation_memory.sh
```

### Performance Tracking

Benchmark results are stored in `target/criterion/`:
- `base/` - Baseline measurements
- `new/` - Current measurements
- `change/` - Comparison data

Historical data enables:
- Tracking performance over time
- Detecting gradual regressions
- Validating optimization efforts

## Troubleshooting

### High Variability

If benchmarks show high variability (R² < 0.95):

1. **Close other applications**: Reduce system load
2. **Disable CPU frequency scaling**: Lock CPU frequency
   ```bash
   # Linux
   sudo cpupower frequency-set --governor performance
   ```
3. **Increase sample size**: More iterations reduce noise
   ```bash
   cargo bench -- --sample-size 100
   ```
4. **Check system load**: Run `htop` to monitor

### Benchmark Takes Too Long

For long-running benchmarks:

1. **Reduce sample size**: Use `--sample-size 10`
2. **Filter benchmarks**: Run specific tests only
   ```bash
   cargo bench --bench simulation_memory -- 100  # Only 100-scenario tests
   ```
3. **Use faster problem sizes**: Start with smaller problems

### Memory Profiling Not Working

Common issues:

- **Linux**: Need root for `perf` or kernel settings
  ```bash
  sudo sysctl -w kernel.perf_event_paranoid=-1
  ```
- **macOS**: Need to disable System Integrity Protection for some tools
- **Windows**: Use Visual Studio Profiler or Windows Performance Recorder

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [POWE.RS Performance Documentation](../performance/simulation-memory.md)

---

**Last Updated**: 2025-10-13  
**Related**: [Testing Guide](TESTING.md), [Performance Documentation](../performance/)
