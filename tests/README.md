# POWE.RS Test Suite

**Version**: 0.2.0  
**Test Count**: 2,000+  
**Execution Time**: < 15 seconds  
**Coverage**: Comprehensive (unit, integration, E2E, performance, regression)

## Overview

This directory contains the comprehensive test suite for POWE.RS, a Rust implementation of Stochastic Dual Dynamic Programming (SDDP) for hydrothermal dispatch optimization.

## Test Organization

### Unit Tests (`src/**/*_test` modules)
- Located alongside source code
- Test individual functions and modules
- Fast execution (< 1ms per test)
- Examples: cut evaluation, state operations, risk measures

### Integration Tests (`tests/`)
- Test module interactions
- Verify SDDP algorithm components work together
- Examples: forward pass, backward pass, training loop

### End-to-End Tests (`tests/test_sddp_*`)
- Full SDDP training runs
- Validate against known solutions
- Examples: deterministic problems, stochastic problems

### Performance Tests (`tests/test_performance_benchmarks.rs`)
- Smoke tests for performance infrastructure
- Verify no obvious regressions
- Full benchmarks in `benches/`

### Regression Tests (`tests/test_regression.rs`)
- Ensure consistent behavior across versions
- Test reproducibility and numerical stability

## Test Categories

### Core Algorithm Tests
- `test_algorithm_correctness.rs` - Algorithm logic (gaps, bounds, convergence)
- `test_sddp_algorithm.rs` - Full SDDP training integration
- `test_convergence_properties.rs` - Convergence behavior validation
- `test_mathematical_properties.rs` - Core SDDP mathematical properties

### Cut Tests
- `test_cut.rs` - Cut data structure and operations
- `test_cut_generation_correctness.rs` - Cut validity at training points
- `test_cut_numerical_stability.rs` - Numerical precision in cut evaluation
- `test_cut_edge_cases.rs` - Boundary conditions and edge cases
- `test_advanced_cut_properties.rs` - Domination, dual feasibility
- `test_cut_pool.rs` - Cut storage and management
- `test_cut_selection_integration.rs` - Cut selection strategies
- `test_batch_cut_selection.rs` - Batch cut selection

### State Management Tests
- `test_state.rs` - State representation (REMOVED - obsolete API)
- Integration tests cover state management via algorithm tests

### Risk Measure Tests
- `test_risk_measure.rs` - Risk measure implementations (expectation, CVaR)
- Property-based tests with ~1,500 random cases

### Scenario Generation Tests
- `test_scenario.rs` - Scenario structure and generation
- `test_scenario_validation.rs` - Statistical properties validation
- `test_oos.rs` - Out-of-sample scenario generation

### AR Model Tests
- `test_ar_model_integration.rs` - AR model integration with SDDP
- `test_ar_cut_validation.rs` - AR cut generation correctness
- `test_ar_lag_cut_coefficients.rs` - Lag coefficient validation
- `test_ar_psi_validation.rs` - AR constraint (ψ) validation

### System Validation Tests
- `test_system.rs` (in src/) - System structure validation
- `test_explicit_constraints_validation.rs` - AR constraint structure
- `test_explicit_lag_separation.rs` - Lag separation correctness

### Numerical Stability Tests
- `test_numerical_stability_deep_dive.rs` - Kahan summation, precision
- `test_numerical_validation.rs` - Numerical properties

### Error Handling Tests
- `test_error_messages.rs` - Error message clarity
- `test_subproblem_error_paths.rs` - Error path coverage

### Output Tests
- `test_output.rs` - Output file generation and performance
- `test_factory_api.rs` - Factory method patterns

### Integration Test Suites
- `tests/integration/algorithm/` - Algorithm component integration
- `tests/integration/features/` - Feature integration

### Migration & Baseline Tests
- `test_uncertainty_migration_baseline.rs` - API migration validation
- `test_simulation_extract_and_release.rs` - Extraction simulation

## Running Tests

### All Tests
```bash
cargo test
```

### Unit Tests Only
```bash
cargo test --lib
```

### Integration Tests Only
```bash
cargo test --tests
```

### Specific Test File
```bash
cargo test --test test_cut_generation_correctness
```

### Specific Test Function
```bash
cargo test test_cut_validity_at_training_point
```

### Show Test Output
```bash
cargo test -- --nocapture
```

### Run Ignored Tests
```bash
cargo test -- --ignored
```

### Parallel Execution Control
```bash
# Use 4 threads
cargo test -- --test-threads=4

# Single-threaded (for debugging)
cargo test -- --test-threads=1
```

## Test Utilities

Common test utilities are in `tests/utils/`:

### Numeric Assertions (`tests/utils/assertions.rs`)
```rust
use tests::utils::*;

assert_float_approx_eq(actual, expected, epsilon);
assert_monotonic_non_decreasing(&values, tolerance);
assert_all_positive(&values);
```

### Cut Validation (`tests/utils/cut_validation.rs`)
```rust
assert_cut_validity(cut, state_at_training, objective_at_training, epsilon);
assert_water_values_negative(cut); // For minimization
```

### Physical Validation (`tests/utils/physical_validation.rs`)
```rust
assert_water_balance(&trajectory, tolerance);
assert_power_balance(&trajectory, tolerance);
```

### Monotonic Checks (`tests/utils/monotonic.rs`)
```rust
assert_monotonic_non_decreasing(&bounds, tolerance);
```

## Test Fixtures

Reusable test fixtures are in `tests/fixtures/`:

- `systems.rs` - Test system configurations
- `simple_2stage_reservoir.rs` - Simple 2-stage system
- `scenarios.rs` - Test scenario configurations
- `subproblems.rs` - Subproblem test fixtures
- `benchmarks.rs` - Benchmark system configurations
- `oos.rs` - Out-of-sample test fixtures

### Using Fixtures
```rust
use tests::fixtures::*;

let system = create_simple_deterministic_system();
let system = create_two_hydro_cascade();
let scenarios = create_test_scenarios(num_stages, num_scenarios);
```

## Test Patterns

### Pattern 1: Simple Unit Test
```rust
#[test]
fn test_feature_works() {
    // Setup
    let input = create_test_input();
    
    // Execute
    let result = function_under_test(input);
    
    // Assert
    assert_eq!(result, expected);
}
```

### Pattern 2: Integration Test with Fixtures
```rust
#[test]
fn test_integration_scenario() {
    use tests::fixtures::*;
    
    let system = create_simple_deterministic_system();
    let config = create_test_config();
    
    let result = run_sddp_training(&system, config);
    
    assert!(result.is_ok());
    assert_float_approx_eq(result.final_lower_bound, expected, 1e-6);
}
```

### Pattern 3: Property-Based Test
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn property_always_holds(
        value in 0.0f64..1000.0,
        alpha in 0.01f64..0.99
    ) {
        let result = compute_cvar(value, alpha);
        prop_assert!(result >= value);
    }
}
```

### Pattern 4: Parameterized Test
```rust
#[test]
fn test_multiple_scenarios() {
    for &num_scenarios in &[1, 5, 10, 50] {
        let result = run_with_scenarios(num_scenarios);
        assert!(result.is_ok(), "Failed with {} scenarios", num_scenarios);
    }
}
```

## Test Naming Conventions

Follow this pattern for test names:
```
test_<what>_<condition>_<expected>
```

**Examples**:
- `test_cut_evaluation_at_zero_state_equals_rhs()`
- `test_training_with_convergence_stops_early()`
- `test_water_balance_with_cascade_conserves_volume()`

## Ignored Tests

Tests marked with `#[ignore]` require updates:

### Reason: Schema/API Changes (25 tests)
- Need updates for new temporal_model API
- Need updates for removed stochastic_process API
- Tracked in TESTING_PROGRESS_SUMMARY.md

### Running Ignored Tests
```bash
cargo test -- --ignored
```

## Performance Considerations

### Fast Tests (<1s for unit tests)
- Unit tests should complete in < 1ms each
- Integration tests should complete in < 100ms each
- Full suite target: < 3 minutes (actual: < 15 seconds)

### Optimization Tips
- Use `#[cfg(test)]` to avoid compiling test utilities in release
- Reuse fixtures instead of creating new ones
- Use smaller systems for unit tests
- Reserve large systems for E2E tests

## Coverage Goals

- **Line coverage**: 75%+ (measured with cargo-tarpaulin)
- **Branch coverage**: Focus on critical paths
- **Mutation score**: 70%+ baseline (cargo-mutants)

### Running Coverage
```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Run coverage
cargo tarpaulin --out Html --output-dir coverage/
```

## Continuous Integration

Tests run automatically on:
- Every commit
- Pull requests
- Scheduled runs (nightly)

CI configuration: `.github/workflows/` (if present)

## Benchmarks

Performance benchmarks are separate from tests:
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench cut_evaluation
```

See `benches/README.md` for benchmark documentation.

## Debugging Failed Tests

### 1. Run with output
```bash
cargo test test_name -- --nocapture
```

### 2. Run single-threaded
```bash
cargo test test_name -- --test-threads=1
```

### 3. Set RUST_BACKTRACE
```bash
RUST_BACKTRACE=1 cargo test test_name
```

### 4. Use println! debugging
```rust
#[test]
fn test_debug() {
    let value = compute();
    println!("Value: {:?}", value);
    assert_eq!(value, expected);
}
```

### 5. Use cargo test --no-fail-fast
```bash
# Run all tests even if some fail
cargo test --no-fail-fast
```

## Contributing New Tests

### Checklist
1. [ ] Test has clear, descriptive name
2. [ ] Test has doc comment explaining purpose
3. [ ] Test uses existing fixtures/utilities where possible
4. [ ] Test runs quickly (< 100ms for integration tests)
5. [ ] Test is deterministic (not flaky)
6. [ ] Test follows project patterns
7. [ ] Run `cargo test` before committing
8. [ ] Run `cargo fmt` and `cargo clippy`

### Example PR Description
```
Add test for cut domination detection

- Tests that dominated cuts are correctly identified
- Uses simple 2-cut scenario for clarity
- Validates both geometric and algebraic domination
- Execution time: < 1ms

Relates to: TEST-013 (Advanced Cut Properties)
```

## Test Quality Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Total Tests | 2,050+ | 450+ | ✅ 456% |
| Execution Time | < 15s | < 3min | ✅ 12x better |
| Flaky Tests | 0 | 0 | ✅ Perfect |
| Code Coverage | High | 75%+ | ✅ |
| Mutation Score | TBD | 70%+ | 🔄 Baseline needed |

## Resources

- **Testing Strategy**: `TESTING_STRATEGY.md`
- **Implementation Tickets**: `TESTING_TICKETS_REVISED.md`
- **Progress Summary**: `TESTING_PROGRESS_SUMMARY.md`
- **Phase 4 Completion**: `TEST-PHASE4-FINAL-COMPLETION.md`
- **Phase 5 Plan**: `PHASE5-IMPLEMENTATION-PLAN.md`

## Common Issues

### Issue: Test fails intermittently
**Solution**: Likely a floating-point comparison issue. Use `assert_float_approx_eq` with appropriate epsilon.

### Issue: Test is slow
**Solution**: Profile with `cargo test -- --nocapture` and consider:
- Using smaller test fixtures
- Reducing iteration counts
- Moving to `#[ignore]` for expensive tests

### Issue: Test depends on other tests
**Solution**: Tests must be independent. Extract shared setup to fixtures.

### Issue: Test fails in CI but passes locally
**Solution**: Likely environment-specific (timing, parallelism). Add deterministic seeds and reduce parallelism sensitivity.

## Contact

For questions about tests:
1. Check this README
2. Look at similar existing tests
3. Consult testing strategy documents
4. Ask in project discussions

---

**Last Updated**: 2025-11-07  
**Maintained By**: POWE.RS Contributors
