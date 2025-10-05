# Testing Guide

This document describes the testing strategy and how to run tests for the POWE.RS project.

## Table of Contents

- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Performance Benchmarks](#performance-benchmarks)
- [Test Fixtures](#test-fixtures)
- [Writing Tests](#writing-tests)
- [CI/CD Integration](#cicd-integration)

## Test Organization

The project uses Rust's built-in testing framework with the following structure:

### Unit Tests

Located inline with source code in `src/`:

- `src/cut.rs` - Cut generation and selection tests
- `src/solver.rs` - HiGHS solver interface tests
- `src/state.rs` - State management tests
- `src/subproblem.rs` - Subproblem construction tests
- `src/system.rs` - Power system model tests
- etc.

### Integration Tests

Located in `tests/`:

- `tests/test_benchmarks.rs` - Numerical validation benchmarks
- `tests/test_cut.rs` - Cut algorithm integration tests
- `tests/test_numerical_validation.rs` - Algorithm correctness tests
- `tests/test_policy_validation.rs` - Policy quality tests
- `tests/test_scenario.rs` - Scenario generation tests
- etc.

### Benchmark Tests

Located in `benches/`:

- `benches/sddp_benchmarks.rs` - Performance benchmarks (see [Performance Benchmarks](#performance-benchmarks))

### Test Fixtures

Located in `tests/fixtures/`:

- `tests/fixtures/benchmarks.rs` - Standard benchmark problems
- `tests/fixtures/systems.rs` - System construction helpers
- `tests/fixtures/subproblems.rs` - Subproblem helpers
- etc.

## Running Tests

### All Tests

```bash
# Run all tests (unit + integration)
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_deterministic_single_reservoir
```

### Unit Tests Only

```bash
# Run only unit tests (inline with source)
cargo test --lib
```

### Integration Tests Only

```bash
# Run only integration tests
cargo test --test '*'

# Run specific integration test file
cargo test --test test_benchmarks
```

### With All Features

```bash
# Run tests with all features enabled
cargo test --all-features
```

### Quick Test (Fast Feedback)

```bash
# Run tests without expensive benchmarks
cargo test --lib --tests
```

## Performance Benchmarks

POWE.RS uses [Criterion.rs](https://bheisler.github.io/criterion.rs/) for performance regression testing. Benchmarks measure critical operations to detect performance degradations.

### Running Benchmarks Locally

```bash
# Run all benchmarks
cargo bench --bench sddp_benchmarks

# Run specific benchmark group
cargo bench --bench sddp_benchmarks -- full_iteration

# Run with verbose output
cargo bench --bench sddp_benchmarks -- --verbose
```

### Viewing Results

Criterion generates detailed HTML reports with statistical analysis:

```bash
# Run benchmarks
cargo bench --bench sddp_benchmarks

# Open HTML report in browser
open target/criterion/report/index.html    # macOS
xdg-open target/criterion/report/index.html # Linux
start target/criterion/report/index.html    # Windows
```

The report includes:

- Mean/median execution times with confidence intervals
- Comparison to previous runs (if available)
- Statistical analysis (outlier detection, variance)
- Performance plots and histograms

### Benchmark Groups

#### 1. Full Iteration Benchmarks

Measure a single SDDP training iteration (forward pass + backward pass + cut generation):

```bash
cargo bench --bench sddp_benchmarks -- full_iteration
```

- `full_iteration/2_stage_deterministic` - Simple 2-stage problem (~1.5 ms)
- `full_iteration/2_stage_stochastic` - 2-stage with 3 scenarios (~1.6 ms)
- `full_iteration/12_stage_deterministic` - Real-world sized problem (~7.7 ms)

**Expected Complexity**: O(num_stages × (num_scenarios + solver_time))

#### 2. Convergence Benchmarks

Measure multi-iteration training to convergence:

```bash
cargo bench --bench sddp_benchmarks -- convergence
```

- `convergence/2_stage_10iter` - Fast convergence check (~13 ms)
- `convergence/2_stage_stochastic_20iter` - Stochastic convergence (~26 ms)
- `convergence/12_stage_20iter` - Large problem convergence (~149 ms)

**Expected Complexity**: O(num_iterations × iteration_time)

#### 3. Simulation Benchmarks

Measure out-of-sample policy simulation:

```bash
cargo bench --bench sddp_benchmarks -- simulation
```

- `simulation/2_stage/100_scenarios` - Small problem simulation (~19 ms)
- `simulation/12_stage/100_scenarios` - Large problem simulation (~75 ms)

**Expected Complexity**: O(num_scenarios × num_stages × solve_time)

**Performance Note**: Simulation is parallelized with Rayon across scenarios.

#### 4. Scaling Benchmarks

Measure how performance scales with problem size:

```bash
cargo bench --bench sddp_benchmarks -- scaling
```

- `scaling/stages/2` - 2-stage baseline (~4.5 ms)
- `scaling/stages/6` - 6-stage problem (~12.4 ms)
- `scaling/stages/12` - 12-stage problem (~24.6 ms)

**Expected Scaling**: Near-linear with number of stages (O(n))

### Comparing to Baseline

Criterion can compare current performance to a saved baseline:

```bash
# Save current results as baseline
cargo bench --bench sddp_benchmarks -- --save-baseline main

# Run and compare to baseline
cargo bench --bench sddp_benchmarks -- --baseline main
```

Output shows relative change:

```
full_iteration/2_stage_deterministic
                        time:   [1.4581 ms 1.5311 ms 1.6971 ms]
                        change: [-5.2341% -2.1234% +1.4567%] (p = 0.31 > 0.05)
                        No change in performance detected.
```

### Performance Baselines

See [PERFORMANCE-BASELINES.md](PERFORMANCE-BASELINES.md) for:

- Reference hardware specifications
- Baseline measurements for all benchmarks
- Performance regression thresholds
- Historical performance trends

### Interpreting Results

**Criterion Statistics**:

- **Mean**: Average execution time (affected by outliers)
- **Median**: Middle value (robust to outliers) - **Use this for comparison**
- **Std Dev**: Measure of variability
- **Lower/Upper Bounds**: 95% confidence interval

**Performance Changes**:

- ✅ **< 5% slower**: Within normal variance
- ⚠️ **5-10% slower**: Investigate if intentional
- ❌ **> 10% slower**: Performance regression - requires investigation

**What to Do on Regression**:

1. Check if change is intentional (new feature, correctness fix)
2. Profile with `cargo flamegraph` to identify bottleneck
3. Consider optimization if regression is unintended
4. Update baselines if regression is acceptable

### Benchmark Implementation Details

Benchmarks use realistic problems to measure actual performance:

**Problem Characteristics** (all benchmarks):

- 1 hydro reservoir (100 MWh storage, 60 MW capacity)
- 1 thermal generator (40 MW, $15/MWh)
- Load: 50 MW per stage
- Inflows: 20-40 MWh (creates scarcity)
- Deficit cost: $200/MWh

These values ensure:

- Non-trivial optimization (thermal vs hydro trade-off)
- Water has value (scarcity requires storage decisions)
- Realistic solver complexity (non-degenerate LPs)

### Common Benchmark Patterns

**Testing Performance Impact of Changes**:

```bash
# Before change
cargo bench --bench sddp_benchmarks -- --save-baseline before

# Make your changes...

# After change - compare
cargo bench --bench sddp_benchmarks -- --baseline before
```

**Quick Performance Check** (faster, less statistical rigor):

```bash
cargo bench --bench sddp_benchmarks -- --quick
```

**Profiling with Flamegraph** (Linux only):

```bash
# Install flamegraph
cargo install flamegraph

# Profile a benchmark
cargo flamegraph --bench sddp_benchmarks -- --bench --profile-time 10

# View flamegraph.svg in browser
```

### Cut Selection Performance Benchmarks

Specialized benchmarks for cut selection analysis (`benches/cut_selection.rs`):

```bash
# Run all cut selection benchmarks
cargo bench --bench cut_selection

# Run specific groups
cargo bench --bench cut_selection -- scaling
cargo bench --bench cut_selection -- thread_contention
cargo bench --bench cut_selection -- batch_vs_perthread
```

**Benchmark Groups**:

1. **Scaling** - Performance with cut pool sizes (10, 100, 1000, 10000 cuts)
2. **Dimensionality** - Impact of state dimensions (1D, 5D, 20D state spaces)
3. **Thread Contention** - Lock contention with multiple threads (1, 2, 4, 8)
4. **Batch vs Per-Thread** - Comparison of cut selection strategies
5. **Dominance Components** - Breakdown of cut selection operations

**Expected Results**:

| Cut Pool Size | Selection Time | Overhead % |
| ------------- | -------------- | ---------- |
| 10            | 5-10 μs        | 2%         |
| 100           | 30-60 μs       | 5%         |
| 1,000         | 200-500 μs     | 8%         |
| 10,000        | 2-10 ms        | 12%        |

**Profiling Script** (comprehensive analysis):

```bash
# Run all profiling tools
./scripts/profile_cut_selection.sh all

# Individual tools
./scripts/profile_cut_selection.sh flamegraph  # Visual profiling
./scripts/profile_cut_selection.sh perf        # CPU analysis
./scripts/profile_cut_selection.sh bench       # Criterion benchmarks
```

**See Also**: [docs/PERFORMANCE-CUT-SELECTION.md](docs/PERFORMANCE-CUT-SELECTION.md) for detailed analysis.

## Test Fixtures

Test fixtures provide reusable benchmark problems and helper functions.

### Using Fixtures in Tests

```rust
use crate::fixtures::benchmarks::*;

#[test]
fn my_test() {
    let (mut sddp, saa) = create_deterministic_single_reservoir()
        .expect("Failed to create benchmark");

    let result = sddp.train(20, 10, &saa).expect("Training failed");

    // Assertions...
}
```

### Available Fixtures

See `tests/fixtures/benchmarks.rs` for:

- `create_deterministic_single_reservoir()` - Simple 2-stage deterministic
- `create_stochastic_single_reservoir()` - 2-stage with uncertainty
- `create_two_reservoir_cascade()` - Cascade system
- `create_single_reservoir_system()` - System builder helper

## Writing Tests

### Unit Test Example

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut_generation() {
        let cut = Cut::new(100.0, vec![0.5, 0.3]);
        assert_eq!(cut.intercept, 100.0);
        assert_eq!(cut.coefficients.len(), 2);
    }
}
```

### Integration Test Example

```rust
use powers_rs::sddp::SddpAlgorithm;

#[test]
fn test_training_convergence() {
    let (mut sddp, saa) = create_test_problem();

    let result = sddp.train(100, 20, &saa)
        .expect("Training failed");

    // Check convergence
    assert!(result.final_gap() < 10.0, "Gap too large");

    // Check monotonicity
    let lower_bounds = result.lower_bounds();
    for i in 1..lower_bounds.len() {
        assert!(
            lower_bounds[i] >= lower_bounds[i-1] - 1e-6,
            "Lower bound not monotonic"
        );
    }
}
```

### Benchmark Example

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_forward_pass(c: &mut Criterion) {
    let (mut sddp, saa) = create_test_problem();

    c.bench_function("forward_pass_10_stages", |b| {
        b.iter(|| {
            // Use black_box to prevent compiler optimization
            black_box(sddp.train(1, 1, &saa))
        });
    });
}

criterion_group!(benches, benchmark_forward_pass);
criterion_main!(benches);
```

## CI/CD Integration

### Automated Testing

GitHub Actions runs tests automatically on:

- Every push to `main` branch
- Every pull request
- Manual workflow dispatch

See `.github/workflows/ci.yml` for test configuration.

### Automated Benchmarks

GitHub Actions runs performance benchmarks:

- On push to `main` (establishes baseline)
- On pull requests (compares to main)
- Fails PR if regression > 5% without justification

See `.github/workflows/benchmark.yml` for benchmark configuration.

### CI Test Commands

```bash
# What CI runs for tests
cargo test --all-features --verbose

# What CI runs for benchmarks
cargo bench --bench sddp_benchmarks -- --output-format bencher
```

### Test Coverage

While not enforced, aim for:

- Unit tests: Cover all public APIs and edge cases
- Integration tests: Cover critical workflows and algorithms
- Benchmarks: Cover hot paths and performance-critical operations

## Troubleshooting

### Tests Failing Locally

```bash
# Clean and rebuild
cargo clean
cargo test

# Check for warnings
cargo clippy --all-targets --all-features -- -D warnings

# Format code
cargo fmt --all --check
```

### Benchmarks Too Slow

```bash
# Run quick benchmarks (less statistical rigor)
cargo bench --bench sddp_benchmarks -- --quick

# Run specific benchmark only
cargo bench --bench sddp_benchmarks -- full_iteration/2_stage
```

### Benchmark Results Noisy

- Close other applications
- Disable CPU frequency scaling: `sudo cpupower frequency-set -g performance` (Linux)
- Run multiple times and check consistency
- Increase Criterion sample size (in benchmark code)

### Out of Memory

Large problems may exhaust memory. Reduce problem size in tests:

- Fewer scenarios
- Fewer stages
- Smaller state space

## Best Practices

### Test Design

- ✅ Test one thing per test
- ✅ Use descriptive test names
- ✅ Keep tests fast (< 1s each if possible)
- ✅ Use fixtures to avoid code duplication
- ✅ Test edge cases and error conditions

### Benchmark Design

- ✅ Benchmark hot paths only (not setup code)
- ✅ Use realistic problem sizes
- ✅ Use `black_box()` to prevent optimization
- ✅ Document expected performance
- ✅ Run benchmarks before and after optimizations

### Performance Testing

- ✅ Profile before optimizing
- ✅ Measure impact of changes
- ✅ Document performance characteristics
- ✅ Keep baselines up-to-date
- ✅ Investigate regressions promptly

## References

- [Rust Testing Documentation](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [PERFORMANCE-BASELINES.md](PERFORMANCE-BASELINES.md) - Baseline metrics
- [CHANGELOG.md](CHANGELOG.md) - Performance-related changes
