# POWE.RS Performance Benchmarks

This directory contains Criterion benchmarks for measuring and tracking the performance of critical SDDP operations.

## Available Benchmarks

### Core SDDP Operations
- **`sddp_benchmarks.rs`** - Full SDDP training iterations, convergence, simulation, scaling
- **`parallel_efficiency.rs`** - Parallel scaling with 1, 2, 4, 8 threads (speedup and efficiency analysis)
- **`subproblem_solve.rs`** - Isolated solver performance (cold start, warm start, scaling)

### Optimized Hot Paths (Sprint 3)
- **`cut_selection.rs`** - Cut selection strategies, dominance computation, batch vs per-thread
- **`cut_id_lookup.rs`** - Data structure micro-benchmarks (Vec vs HashSet lookups)

### Foundation Components
- **`state_operations.rs`** - State construction, coefficient access, cloning, memory patterns

## Running Benchmarks

### Run All Benchmarks
```bash
cargo bench
```

### Run Specific Benchmark File
```bash
cargo bench --bench sddp_benchmarks
cargo bench --bench cut_selection
cargo bench --bench subproblem_solve
cargo bench --bench state_operations
```

### Run Specific Benchmark by Name
```bash
# Run all benchmarks matching "single_reservoir"
cargo bench -- single_reservoir

# Run all "scaling" benchmarks
cargo bench -- scaling
```

### Quick Development Mode (Faster Feedback)
```bash
# Reduce sample size for quick iteration
CRITERION_QUICK=1 cargo bench
```

## Criterion Configuration

### Default Configuration
- **Sample size**: 10-100 (varies by benchmark group)
- **Measurement time**: 10 seconds per benchmark group
- **Warm-up time**: 3 seconds
- **Significance level**: 5% (detects >5% performance changes)
- **Noise threshold**: 2% (ignores measurement noise)
- **Outputs**: HTML reports + JSON data + terminal summary

### Sample Size Guidelines
Benchmarks use different sample sizes based on operation cost:
- **100 samples**: Fast operations (<1ms) - state operations, cut evaluation
- **50 samples**: Medium operations (1-10ms) - single subproblem solves
- **20 samples**: Expensive operations (10-100ms) - full iterations, convergence
- **10 samples**: Very expensive operations (>100ms) - multi-iteration training

### Measurement Time
- **Default**: 10 seconds per benchmark group (good for local development)
- **CI**: 15 seconds (higher confidence, lower variance)
- **Quick**: 2 seconds (fast feedback during development)

## Baseline Management

### Save a Baseline
```bash
# Save current performance as "main" baseline
cargo bench -- --save-baseline main
```

### Compare Against Baseline
```bash
# Compare current performance to "main" baseline
cargo bench -- --baseline main
```

### Update Baseline (After Intentional Performance Changes)
```bash
# After merging a performance optimization:
git checkout main
cargo bench -- --save-baseline main
git add target/criterion/*/main
git commit -m "Update performance baselines after optimization"
```

## Viewing Results

### HTML Reports
Open in browser:
```bash
open target/criterion/report/index.html
```

### JSON Data (for scripting)
```bash
# Estimates (median, mean, std dev)
cat target/criterion/<benchmark_name>/base/estimates.json

# Raw measurements
cat target/criterion/<benchmark_name>/base/raw.csv
```

### Terminal Output
Criterion prints summary statistics after each benchmark:
- **Time**: Median time with confidence interval
- **Throughput**: Operations per second
- **Change**: % change vs baseline (if comparing)
- **Regression**: Warning if >5% slower than baseline

## Performance Targets

### Critical Thresholds (From Reference Hardware)
Based on AMD Ryzen 9 5950X, 64GB RAM, Ubuntu 22.04:

| Operation | Target Time | Throughput | Regression Threshold |
|-----------|-------------|------------|----------------------|
| Forward Pass (single) | <2 ms | >500 ops/s | >5% slower |
| Backward Pass (single) | <5 ms | >200 ops/s | >5% slower |
| Cut Selection (100 cuts) | <50 μs | >20K ops/s | >5% slower |
| Subproblem Solve (cold) | <3 ms | >333 ops/s | >5% slower |
| State Construction (5D) | <10 μs | >100K ops/s | >5% slower |

### Hot Path Budget (From Phase 2 Timing)
Expected time distribution in SDDP training:
- **Solver**: 60-80% (subproblem construction + LP solve)
- **Forward Pass**: 10-20% (scenario sampling, model prep, aggregation)
- **Backward Pass**: 10-20% (cut generation, FCF update, model update)
- **Cut Selection**: <1% (Sprint 3 optimization - batch processing)
- **State Operations**: <5% (coefficient access, updates)

## CI Integration

### Automated Regression Detection
GitHub Actions runs benchmarks on:
- Push to `main` branch
- Pull requests to `main`

CI configuration:
- **Stricter thresholds**: 1% noise threshold (vs 2% local)
- **More samples**: 100 samples (vs 10-50 local)
- **Longer measurement**: 15 seconds (vs 10 seconds local)
- **Failure condition**: >5% regression fails CI

### Benchmark Workflow
See `.github/workflows/benchmarks.yml` for full configuration.

## Interpreting Results

### Understanding Variance
- **<5% variance**: Excellent (benchmark is stable)
- **5-10% variance**: Acceptable (some noise in measurement)
- **>10% variance**: High (may need more samples or longer measurement time)

High variance can be caused by:
- Background processes (close other applications)
- CPU frequency scaling (disable TurboBoost for consistent results)
- Thermal throttling (ensure adequate cooling)
- Memory pressure (close memory-intensive applications)

### Regression Analysis
Criterion detects regressions using statistical tests:
- **Change**: Measured difference in performance
- **Confidence Interval**: Range of plausible true values (95% CI)
- **Significance**: Whether change is statistically significant (p < 0.05)

**Example**: "Change: +7.2% (5.1% to 9.3%)" means:
- Performance degraded by ~7.2%
- True degradation is likely between 5.1% and 9.3%
- This is statistically significant (exceeds noise threshold)

## Adding New Benchmarks

### 1. Create Benchmark File
```bash
touch benches/my_benchmark.rs
```

### 2. Register in Cargo.toml
```toml
[[bench]]
name = "my_benchmark"
harness = false
```

### 3. Write Benchmark Code
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_my_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_operation");
    group.sample_size(50);  // Adjust based on operation cost
    
    group.bench_function("my_test_case", |b| {
        b.iter(|| {
            // Use black_box to prevent compiler optimization
            black_box(my_operation())
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_my_operation);
criterion_main!(benches);
```

### 4. Test Benchmark
```bash
# Run in test mode (fast, doesn't save results)
cargo bench --bench my_benchmark -- --test

# Run normally (full measurement)
cargo bench --bench my_benchmark
```

## Best Practices

### DO:
- ✅ Use `black_box()` to prevent compiler optimization
- ✅ Use realistic problem sizes (match production workloads)
- ✅ Warm up before measuring (Criterion does this automatically)
- ✅ Group related benchmarks (easier comparison)
- ✅ Document expected performance characteristics
- ✅ Commit baseline updates after intentional optimizations

### DON'T:
- ❌ Benchmark trivial operations (<1μs) - measurement overhead dominates
- ❌ Use unrealistically small problems - results won't generalize
- ❌ Forget `black_box()` - compiler may optimize away your code
- ❌ Run benchmarks with background load - results will be noisy
- ❌ Commit baseline updates for unintended regressions

## Troubleshooting

### "No benchmarks found"
- Check `Cargo.toml` has `[[bench]]` entry with `harness = false`
- Verify file is in `benches/` directory
- Ensure `criterion_group!` and `criterion_main!` are present

### "High variance"
- Close background applications
- Disable CPU frequency scaling: `sudo cpupower frequency-set --governor performance`
- Increase `measurement_time()` or `sample_size()`
- Check for thermal throttling: `sensors` (install lm-sensors)

### "Benchmark takes too long"
- Reduce `sample_size()` for development (increase for CI)
- Reduce `measurement_time()` temporarily
- Use `cargo bench --bench <specific_file>` instead of `cargo bench`

### "CI benchmarks fail locally"
- CI uses stricter thresholds (1% noise vs 2%)
- CI uses more samples (100 vs 10-50)
- CI may have different hardware characteristics
- Check CI logs for specific regression details

## References

- [Criterion.rs User Guide](https://bheisler.github.io/criterion.rs/book/)
- [Performance Baselines](../docs/performance/PERFORMANCE-BASELINES.md)
- [Sprint 3: Cut Selection Optimization](../.copilot/sprints/sprint-03/)
- [Sprint 4: Performance Automation](../.copilot/sprints/sprint-04/)

## Hardware Specifications (Reference)

Baseline metrics were established on:
- **CPU**: AMD Ryzen 9 5950X (16 cores, 32 threads @ 3.4-4.9 GHz)
- **RAM**: 64 GB DDR4-3600 CL16
- **OS**: Ubuntu 22.04 LTS (kernel 5.15)
- **Rust**: 1.75.0
- **Date**: October 2025

Your hardware will have different absolute performance numbers, but relative changes (regressions/improvements) should be similar.
