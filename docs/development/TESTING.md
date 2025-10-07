# Testing Guide

This document describes the testing strategy and how to run tests for the POWE.RS project.

## Table of Contents

- [Test Organization](#test-organization)
- [Coverage Philosophy](#coverage-philosophy)
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

## Coverage Philosophy

**Current Coverage**: 89.42% (as of October 2025, with all targets)  
**Target**: 90-95% (realistic goal for production code)  
**Test Count**: 189 library tests + 473+ total tests (including integration tests)

### Testing Strategy

POWE.RS follows the **"test business logic, not infrastructure"** principle:

- **High test value/complexity ratio** - Only test what provides real value
- **Focus on algorithm correctness** - Core SDDP logic, solver integration, validation
- **In-module testing** - Use `#[cfg(test)]` modules to test private functions
- **Avoid brittle tests** - No environment variable manipulation, minimal mocking
- **Document trade-offs** - Be explicit about what we don't test and why

### What We Test ✅

#### 1. Core SDDP Algorithm Logic
- Forward and backward pass computations
- Cut generation and selection
- Convergence detection
- Policy simulation

#### 2. Solver Interface
- HiGHS solver integration via FFI
- Status code handling
- Basis management and warm-starting
- Solution extraction

#### 3. Input Validation
- JSON schema conformance
- Entity reference validation (IDs, relationships)
- Constraint validation (capacities, ranges)
- Missing file error messages

#### 4. Data Structures
- State management and coefficients
- Cut pool operations (add, remove, domination)
- Scenario generation and sampling
- Graph topology operations

#### 5. Private Helper Functions
- Tested via `#[cfg(test)]` modules in source files
- Examples: `validate_id_range()`, `set_uncertainties()`, `eval_height_at_state()`
- Rationale: Integration tests can't reach private functions, but they contain critical logic

#### 6. Edge Cases
- Empty collections
- Out-of-bounds access
- Single-element cases
- Duplicate handling

### What We Don't Test ❌

#### 1. Production Logging (`POWERS_TIMING_DETAIL`)
**Lines uncovered**: ~20  
**Rationale**:
- Optional detailed timing output for production monitoring
- Tested manually during development (T4.1)
- Would require environment variable manipulation + output capture
- Low value/high complexity ratio
- Not core algorithm logic

```rust
// Example of intentionally uncovered logging code
if std::env::var("POWERS_TIMING_DETAIL").is_ok() {
    log::training_iteration_timing(...);  // Uncovered, by design
}
```

#### 2. Rare Error Paths

**a) Invalid UTF-8 in File Paths**  
**Lines uncovered**: ~8  
**Rationale**:
- Extremely rare on modern systems (Windows/Linux/macOS use UTF-8)
- Would require `unsafe` or platform-specific code to test
- Defensive programming for edge case

```rust
// Example of defensive error handling (uncovered)
let path_str = path.to_str().ok_or_else(|| {
    IoError::GenericIoError {
        error: "Invalid UTF-8 in path".to_string(), // Rare, uncovered
        // ...
    }
})?;
```

**b) Generic I/O Errors (permissions, disk full, etc.)**  
**Lines uncovered**: ~8  
**Rationale**:
- Hard to test: requires mocking filesystem
- Users get clear `std::io::Error` messages anyway
- Testing complexity >> value

**c) JSON Parse Errors (some paths)**  
**Lines partially covered**: 4 main paths tested, some variants uncovered  
**Rationale**:
- Main error messages tested (T4.2 Phase 5d)
- Schema validation tests cover most malformed JSON cases
- serde_json provides good error messages by default

#### 3. Infrastructure Code

**a) main.rs Entry Point**  
**Lines uncovered**: 11 (100% uncovered by design)  
**Rationale**:
- Entry point for binary, not library logic
- Tested via integration tests that call library API
- Standard practice: don't unit test main()

**b) lib.rs Module Exports**  
**Lines uncovered**: 28 (intentionally minimal testing)  
**Rationale**:
- Module re-exports only
- Tested implicitly via module usage

**c) log.rs Logging Functions**  
**Lines uncovered**: 76 (intentionally minimal testing)  
**Rationale**:
- Console output formatting
- Tested manually/visually
- Not algorithm logic

### Coverage Breakdown by Module

| Module | Coverage | Missed Lines | Status | Priority |
|--------|----------|--------------|--------|----------|
| **100% Coverage (Core Complete)** | | | | |
| cut.rs | 100.00% | 0 | ✓ | Complete |
| state.rs | 100.00% | 0 | ✓ | Complete |
| system.rs | 100.00% | 0 | ✓ | Complete |
| risk_measure.rs | 100.00% | 0 | ✓ | Complete |
| stochastic_process.rs | 100.00% | 0 | ✓ | Complete |
| utils.rs | 100.00% | 0 | ✓ | Complete |
| initial_condition.rs | 100.00% | 0 | ✓ | Complete |
| **Excellent Coverage (>90%)** | | | | |
| solver.rs | 93.67% | 35 | ✓ | Low |
| input_validation.rs | 92.94% | 97 | ✓ | Low |
| fcf.rs | 85.84% | 48 | ✓ | Medium |
| output.rs | 94.42% | 11 | ✓ | Low |
| error.rs | 93.75% | 6 | ✓ | Low |
| subproblem.rs | 94.30% | 59 | ✓ | Low |
| scenario.rs | 98.94% | 3 | ✓ | Low |
| graph.rs | 90.56% | 17 | ✓ | Low |
| **Good Coverage (>80%)** | | | | |
| sddp/mod.rs | 89.46% | 287 | ✓ | Medium (mostly logging) |
| input.rs | 81.19% | 120 | ✓ | Medium (error paths) |
| sddp/builder.rs | 80.82% | 122 | ✓ | Medium (builder validation) |
| **Intentionally Excluded (Infrastructure)** | | | | |
| main.rs | 0.00% | 11 | N/A | Entry point only |
| log.rs | 39.02% | 75 | N/A | Console formatting |
| lib.rs | 39.13% | 28 | N/A | Module exports |
| sddp/instance.rs | 78.79% | 7 | N/A | Production API |

**Summary**:
- **8 modules** with 100% coverage (core algorithms)
- **10 modules** with >90% coverage (excellent)
- **3 modules** with >80% coverage (good)
- **4 modules** intentionally excluded (infrastructure)
- **Total**: 89.42% overall coverage

### Realistic Coverage Goals

- **Current**: 89.42% (with --all-targets including integration tests)
- **Library Only**: 79.99% (with --lib only, excludes integration tests)
- **Achievable**: 90-92% (with targeted validation path testing)
- **Maximum Realistic**: 92-94% (excluding intentionally uncovered code)
- **Unrealistic**: 95%+ (would require testing logging, rare errors, infrastructure)

**Gap Analysis**:
- ~115 lines intentionally uncovered (infrastructure: main.rs, log.rs, lib.rs)
- ~926 total missed lines out of 8,752 total lines
- Most remaining gaps are low-value test targets (logging, rare I/O errors)

**Note**: Always use `--all-targets` for accurate coverage measurement, as it includes integration tests that provide significant validation coverage.

### Testing Patterns

#### Pattern 1: In-Module Testing for Private Functions

```rust
// src/some_module.rs

fn public_api() {
    private_helper(42);
}

fn private_helper(value: usize) {
    // Complex logic that should be tested
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_private_helper() {
        // Can access private function within same module
        assert_eq!(private_helper(10), expected_result);
    }
}
```

**Benefit**: Test private functions without exposing them in public API

#### Pattern 2: Error Path Testing

```rust
#[test]
fn test_invalid_json_error_message() {
    let result = parse_json("{ invalid }");
    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error.to_string().contains("JSON parse error"));
}
```

**Benefit**: Validate user-facing error messages

#### Pattern 3: Edge Case Coverage

```rust
#[test]
fn test_empty_collection() {
    let result = process(vec![]);
    assert!(result.is_empty());  // Should handle gracefully
}

#[test]
fn test_out_of_bounds() {
    let result = get_item(999);
    assert!(result.is_none());  // Should return None, not panic
}
```

**Benefit**: Ensure robust handling of edge cases

### Quality Metrics

- **189 library tests** (grew from 103 in T4.2 Phase 5)
- **473+ total tests** (including all integration test binaries)
- **89.42% overall coverage** (with --all-targets)
- **Zero clippy warnings** (with `-D warnings`)
- **All tests passing** consistently
- **No flaky tests** (deterministic results)
- **No test debt** (no skipped tests, no TODOs)

### Test Count Breakdown

| Test Type | Count | Description |
|-----------|-------|-------------|
| Library unit tests | 189 | Tests in `src/` with `#[cfg(test)]` |
| Integration tests | 284+ | Tests in `tests/*.rs` files |
| Benchmark tests | N/A | Performance tests in `benches/` |
| **Total** | **473+** | All test binaries combined |

### Testing Philosophy Evolution

**T4.2 Coverage Improvement Campaign** (Final Results):

| Phase | Strategy | Tests Added | Coverage Gain | Result |
|-------|----------|-------------|---------------|--------|
| Phase 5a | In-module unit tests | 30 | +1.18% | Success |
| Phase 5b | Additional in-module tests | 20 | +0.39% | Success |
| Phase 5c | SDDP timing tests | 8 | +0.20% | Success |
| User cleanup | Remove dead code | 0 | +2.59% | Manual |
| Phase 5d | Strategic high-value tests | 16 | +0.57% | Success |
| Validation tests | Restored error path tests | 16 | +0.33% | Success |
| **Total** | **From 84.13% → 89.42%** | **+86 tests** | **+5.29%** | **Complete** |

**Final Metrics**:
- Started: 84.13% coverage, 103 library tests
- Finished: 89.42% coverage, 189 library tests, 473+ total tests
- Improvement: +5.29% coverage, +86 library tests (+83.5% growth)

**Key Learnings**:
1. **In-module testing** is highly effective for private functions
2. **Dead code removal** has significant impact on coverage metrics
3. **Strategic testing** (high value/complexity ratio) works better than exhaustive coverage
4. **Integration tests matter**: --lib shows 79.99%, --all-targets shows 89.42% (+9.43%)
5. **Documentation is critical**: Explaining intentionally uncovered code prevents false targets

### References

- [PHASE-5D-ANALYSIS.md](/PHASE-5D-ANALYSIS.md) - Detailed coverage gap analysis
- [PHASE-5D-COMPLETION.md](/PHASE-5D-COMPLETION.md) - Phase 5d completion report
- [Rust Testing Best Practices](https://doc.rust-lang.org/book/ch11-00-testing.html)

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
