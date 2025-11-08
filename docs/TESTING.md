# Testing Guide

**Powers-RS Test Suite Documentation**

This guide explains how to run, write, and maintain tests for the Powers-RS SDDP optimization library.

## 📊 Test Suite Overview

### Test Statistics
- **Total tests**: 387+ passing
- **Pass rate**: 99.2%
- **Test files**: 29 integration test files
- **Unit tests**: Embedded in source files (277 in lib.rs)
- **Benchmarks**: 7/14 working (core benchmarks functional)

### Test Categories

#### 1. Unit Tests (`src/**/*.rs`)
- Located in source files using `#[cfg(test)] mod tests`
- Test individual functions and methods
- Fast, focused, isolated
- **Example**: `src/lib.rs` - 277 unit tests

#### 2. Integration Tests (`tests/*.rs`)
- Test multiple components working together
- Use real system configurations
- Validate end-to-end behavior
- **Examples**:
  - `integration_simple_2stage.rs` - 59 tests
  - `test_sddp_algorithm.rs` - 29 tests
  - `test_solver_interface.rs` - 38 tests

#### 3. Benchmarks (`benches/*.rs`)
- Performance regression detection
- Memory profiling
- Algorithm timing
- **Working benchmarks**: sddp_benchmarks, memory_profiling, simulation_memory

## 🚀 Running Tests

### Basic Commands

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test file
cargo test --test test_sddp_algorithm

# Run specific test
cargo test test_convergence

# Run tests matching pattern
cargo test scenario_generation

# Run with single thread (for debugging)
cargo test -- --test-threads=1

# Run ignored tests (slow/expensive tests)
cargo test -- --ignored

# Run library unit tests only
cargo test --lib
```

### Running Benchmarks

```bash
# Run all working benchmarks
cargo bench --bench sddp_benchmarks
cargo bench --bench memory_profiling

# Run specific benchmark
cargo bench --bench sddp_benchmarks -- single_reservoir

# See benchmark results
open target/criterion/report/index.html
```

### Test Organization

```
tests/
├── fixtures/              # Shared test utilities
│   ├── mod.rs            # Fixture module exports
│   ├── systems.rs        # System creation helpers
│   ├── scenarios.rs      # Scenario generation helpers
│   └── subproblems.rs    # Subproblem fixtures
├── common.rs             # Common test utilities
├── utils/                # Test utilities
│   └── assertions.rs     # Custom assertions
├── integration_*.rs      # Integration tests
└── test_*.rs            # Feature-specific tests
```

## ✍️ Writing Tests

### Unit Test Pattern

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_behavior() {
        // Arrange
        let input = create_test_input();
        
        // Act
        let result = function_under_test(input);
        
        // Assert
        assert_eq!(result, expected_value);
    }

    #[test]
    #[should_panic(expected = "error message")]
    fn test_error_case() {
        let invalid_input = create_invalid_input();
        function_that_should_fail(invalid_input);
    }
}
```

### Integration Test Pattern

```rust
// tests/test_my_feature.rs
mod fixtures;
use fixtures::{create_test_system, create_test_saa};

#[test]
fn test_feature_integration() {
    // Use fixtures for setup
    let system = create_test_system();
    let saa = create_test_saa();
    
    // Test integrated behavior
    let result = perform_operation(&system, &saa);
    
    // Assert expected outcomes
    assert!(result.is_ok());
    assert_eq!(result.unwrap().value, expected);
}
```

### Using Fixtures

Fixtures provide reusable test data and configuration:

```rust
use fixtures::systems::{
    create_simple_system,           // Minimal 1-hydro system
    create_cascade_system,          // 2-hydro cascade
    create_trivial_system,          // Zero-capacity for edge cases
};

use fixtures::scenarios::generate_test_saa;

#[test]
fn test_with_fixtures() {
    let system = create_simple_system();
    let saa = generate_test_saa(10, 42); // 10 scenarios, seed 42
    
    // Use in test...
}
```

### Builder Pattern for Tests

Use SddpInstanceBuilder for complex test setups:

```rust
use powers_rs::sddp::SddpInstanceBuilder;

#[test]
fn test_sddp_training() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("Failed to load")
    .with_num_iterations(5)
    .with_seed(42)
    .build()
    .expect("Failed to build");

    let result = sddp.train();
    assert!(result.is_ok());
}
```

### Programmatic Builder for Tests

For tests that don't use JSON files:

```rust
use powers_rs::sddp::SddpAlgorithm;
use powers_rs::system::{Bus, Hydro, System, Thermal};

#[test]
fn test_programmatic_setup() {
    let (mut sddp, saa) = SddpAlgorithm::builder()
        .system_factory(|| {
            // Create system programmatically
            let bus = Bus::new(0, 500.0);
            let hydro = Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.01);
            let thermal = Thermal::new(0, 0, 50.0, 0.0, 50.0);
            System::new(vec![bus], vec![], vec![thermal], vec![hydro])
        })
        .initial_storage(vec![50.0])
        .num_stages(3)
        .deterministic_inflows(vec![vec![20.0]; 3])
        .deterministic_loads(vec![vec![30.0]; 3])
        .seed(42)
        .build_with_saa()
        .expect("Build failed");

    let result = sddp.train(10, 5, &saa);
    assert!(result.is_ok());
}
```

## 🔍 Test Best Practices

### 1. Test Naming
- Use descriptive names: `test_convergence_with_multiple_scenarios`
- Follow pattern: `test_<what>_<condition>_<expected>`
- Be specific: `test_invalid_storage_bounds_returns_error`

### 2. Arrange-Act-Assert
```rust
#[test]
fn test_example() {
    // Arrange - set up test data
    let system = create_test_system();
    let config = TestConfig::default();
    
    // Act - perform the operation
    let result = perform_operation(&system, &config);
    
    // Assert - verify expectations
    assert!(result.is_ok());
    assert_eq!(result.unwrap().value, expected);
}
```

### 3. Use Custom Assertions

Utilities in `tests/utils/assertions.rs`:

```rust
use utils::assertions::{
    assert_bounds_in_range,
    assert_convergence_quality,
    assert_all_finite,
};

#[test]
fn test_with_custom_assertions() {
    let bounds = vec![100.0, 95.0, 92.0];
    assert_bounds_in_range(&bounds, 90.0, 105.0);
    assert_all_finite(&bounds);
}
```

### 4. Ignored Tests

Use `#[ignore]` for slow or expensive tests:

```rust
#[test]
#[ignore] // Slow test - run manually
fn test_large_scale_scenario() {
    // Expensive test that takes minutes
}
```

Run with: `cargo test -- --ignored`

### 5. Test Documentation

```rust
/// Tests that SDDP converges for a simple deterministic problem.
///
/// # Setup
/// - Single reservoir system
/// - Deterministic inflows
/// - 10 iterations, 5 forward passes
///
/// # Expected
/// - Training succeeds
/// - Lower bound converges
/// - Final bound within tolerance
#[test]
fn test_sddp_convergence_deterministic() {
    // Test implementation
}
```

## 📈 Code Coverage

### Measuring Coverage

```bash
# Install cargo-llvm-cov (if needed)
cargo install cargo-llvm-cov

# Run coverage (generates HTML report)
cargo llvm-cov --all-features --html

# View report
open target/llvm-cov/html/index.html

# Run with specific tests
cargo llvm-cov --lib  # Only library code
cargo llvm-cov --test test_sddp_algorithm  # Specific test file

# Generate lcov format for CI
cargo llvm-cov --all-features --lcov --output-path lcov.info
```

### Coverage Guidelines

**Targets**:
- **Core algorithm**: >90% (SDDP, subproblem, solver interface)
- **Overall**: >80%
- **Utilities**: >70%

**Focus areas**:
1. Core SDDP algorithm logic
2. Uncertainty model handling
3. State management
4. Subproblem construction
5. Error paths and validation

**Less critical**:
- Visualization/output code
- CLI argument parsing
- Debug/diagnostic code

## 🐛 Debugging Tests

### Failed Test Output

```bash
# Show full output for failed tests
cargo test -- --nocapture

# Run single test with output
cargo test test_name -- --nocapture --test-threads=1

# Show backtrace on panic
RUST_BACKTRACE=1 cargo test
RUST_BACKTRACE=full cargo test  # Full backtrace
```

### Common Issues

#### 1. Flaky Tests (Non-Deterministic)
```rust
// Bad - uses random without seed
let value = rand::random();

// Good - uses deterministic seed
use rand::SeedableRng;
let mut rng = rand::rngs::StdRng::seed_from_u64(42);
```

#### 2. Floating Point Comparisons
```rust
// Bad - exact equality
assert_eq!(result, 1.0);

// Good - tolerance-based
assert!((result - 1.0).abs() < 1e-6);

// Better - use approx crate or custom assertion
assert_approx_eq!(result, 1.0, 1e-6);
```

#### 3. Resource Cleanup
```rust
#[test]
fn test_with_temp_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    // Use temp_dir...
    // Automatically cleaned up on drop
}
```

## 📝 Known Issues

### Failing Tests (3)
- `test_deterministic_single_reservoir_convergence`
- `test_stochastic_single_reservoir_convergence`
- `test_two_reservoir_cascade_convergence`

**Issue**: Expected convergence values don't match actual  
**Status**: Non-blocking, under investigation  
**Location**: `tests/test_benchmarks.rs`

### Ignored Tests (4)
1. `test_large_scenario_count` - Intentionally slow
2. `test_performance_no_regression_large` - Expensive benchmark
3. Example 06 PAR - Known infeasibility (TICKET-012)
4. Example 07 PAR - Known infeasibility (TICKET-012)

### Commented Tests (5)
- Located in `tests/test_scenario.rs`
- Use deleted `stochastic_process` module
- Marked with TODO for migration

## 🔗 Related Documentation

- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines including test requirements
- **[BENCHMARK_BASELINE.md](../BENCHMARK_BASELINE.md)** - Benchmark status and usage
- **[TEST_MODERNIZATION_PLAN.md](../TEST_MODERNIZATION_PLAN.md)** - Test suite modernization plan

## 💡 Tips

1. **Run tests frequently** while developing
2. **Write tests first** (TDD) for new features
3. **Use fixtures** to avoid duplication
4. **Keep tests fast** - mock expensive operations
5. **Test edge cases** - empty inputs, boundaries, errors
6. **Document complex tests** - explain what and why
7. **Use descriptive assertions** - custom messages help debugging

## 🚀 Quick Reference

```bash
# Most common commands
cargo test                    # Run all tests
cargo test -- --nocapture    # See println! output
cargo test test_name         # Run specific test
cargo test --lib             # Unit tests only
cargo bench                  # Run benchmarks
cargo llvm-cov --html        # Measure coverage
cargo fmt --all              # Format before commit
cargo clippy                 # Lint before commit
```

---

**Questions or issues with testing?** Check [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue.
