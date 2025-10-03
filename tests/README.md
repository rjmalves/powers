# POWE.RS Test Infrastructure

This directory contains the test infrastructure for POWE.RS, including fixtures, utilities, and tests.

## Organization

```
tests/
├── fixtures/           # Reusable test data
│   ├── mod.rs         # Module exports
│   ├── mock_solver.rs # Mock solver for isolated testing
│   ├── systems.rs     # Power system fixtures
│   └── scenarios.rs   # Scenario tree fixtures
├── utils/             # Test utilities and assertions
│   ├── mod.rs         # Module exports
│   └── assertions.rs  # Custom assertions for numerical testing
└── README.md          # This file
```

## Fixtures

### Mock Solver (`fixtures/mock_solver.rs`)

A configurable mock solver for testing SDDP algorithm logic without depending on HiGHS.

**Features**:

- Configurable return status (Optimal, Infeasible, etc.)
- Configurable objective value and solution
- Call tracking for verification
- Thread-safe counters

**Example Usage**:

```rust
use tests::fixtures::MockSolver;

let solver = MockSolver::new()
    .with_status(HighsModelStatus::Optimal)
    .with_objective_value(42.0)
    .with_solution(vec![1.0, 2.0, 3.0]);

let status = solver.solve();
assert_eq!(status, HighsModelStatus::Optimal);
assert_eq!(solver.optimize_call_count(), 1);
```

### System Fixtures (`fixtures/systems.rs`)

Pre-configured power systems for testing:

1. **`trivial_system()`** - Minimal single-bus, single-hydro system

   - Use for: Basic algorithm mechanics, cut operations
   - Size: 1 bus, 1 hydro, 0 lines
   - Performance: ~1ms to solve

2. **`simple_system()`** - Two-bus cascaded hydro system

   - Use for: Stochastic testing, transmission constraints
   - Size: 2 buses, 2 hydros, 1 line
   - Performance: ~5ms to solve

3. **`medium_system()`** - Realistic 5-bus network
   - Use for: Integration tests, benchmarking
   - Size: 5 buses, 3 hydros, 2 thermals, 4 lines
   - Performance: ~15ms to solve

**Example Usage**:

```rust
use tests::fixtures::trivial_system;

let system = trivial_system();
assert_eq!(system.buses.len(), 1);
assert_eq!(system.hydros.len(), 1);
```

### Scenario Fixtures (`fixtures/scenarios.rs`)

Pre-configured scenario trees for testing stochastic optimization:

1. **`deterministic_scenario()`** - Single scenario per stage
   - Use for: Testing without randomness
   - Memory: O(stages)
2. **`simple_stochastic_scenario()`** - Uniform branching

   - Use for: Stochastic optimization tests
   - Memory: O(scenarios^stages)

3. **`fan_scenario()`** - Many scenarios in first stage only
   - Use for: First-stage focus, scenario reduction
   - Memory: O(first_stage_scenarios + stages)

**Example Usage**:

```rust
use tests::fixtures::deterministic_scenario;

let generator = deterministic_scenario(
    3,  // num_stages
    2,  // num_load_entities
    2,  // num_inflow_entities
);

let saa = generator.generate(42); // Fixed seed for reproducibility
```

## Test Utilities

### Assertions (`utils/assertions.rs`)

Custom assertions for numerical and physical property testing:

- **`assert_float_approx_eq(a, b, tolerance)`** - Floating-point comparison
- **`assert_vec_approx_eq(vec_a, vec_b, tolerance)`** - Vector comparison
- **`assert_cut_valid(cut)`** - Validates Benders cut properties
- **`assert_state_within_bounds(state, lower, upper, tolerance)`** - Bounds checking
- **`assert_in_range(value, min, max)`** - Range validation
- **`assert_all_finite(values, name)`** - Checks for NaN/Inf
- **`assert_not_empty(vec, name)`** - Non-empty validation

**Example Usage**:

```rust
use tests::utils::{assert_float_approx_eq, assert_cut_valid};

// Compare floating-point values
assert_float_approx_eq!(computed_value, expected_value, 1e-10);

// Validate cut properties
let cut = BendersCut::new(vec![1.0], vec![2.0], 3.0).unwrap();
assert_cut_valid(&cut);
```

### Default Tolerance

The default tolerance for floating-point comparisons is `1e-10`, which is appropriate for typical LP solver precision.

## Writing Tests

### Test File Organization

- **Integration tests**: Place in `tests/` directory (e.g., `tests/test_cut.rs`)
- **Unit tests**: Can live in module files with `#[cfg(test)]`
- Each test file should focus on one module or concept

### Using Fixtures

Always prefer fixtures over creating test data inline:

```rust
// Good: Use fixture
use tests::fixtures::trivial_system;
let system = trivial_system();

// Avoid: Creating inline (less reusable, more verbose)
let mut system = System::new();
let bus = Bus::new(0, 1000.0);
// ... many lines of setup ...
```

### Numerical Testing Best Practices

1. **Always use tolerances for floating-point comparisons**:

   ```rust
   // Good
   assert_float_approx_eq!(a, b, 1e-10);

   // Bad - will fail due to floating-point errors
   assert_eq!(a, b);
   ```

2. **Choose appropriate tolerances**:

   - LP solver results: `1e-8` to `1e-10`
   - Probability sums: `1e-10`
   - Physical quantities: `1e-6` (allows for real-world measurement precision)

3. **Test for NaN and Inf**:
   ```rust
   assert_all_finite(&result, "solution vector");
   ```

### Testing Randomness

For tests involving randomness:

1. **Always use fixed seeds**:

   ```rust
   let saa = generator.generate(42); // Reproducible
   ```

2. **Test statistical properties with large samples**:

   ```rust
   let n_samples = 10_000;
   // ... collect samples ...
   // Verify frequencies match probabilities within tolerance
   ```

3. **Document expected behavior**:
   ```rust
   // This test uses seed 42 which produces scenarios with properties X, Y, Z
   ```

## Running Tests

### Run all tests

```bash
cargo test
```

### Run specific test file

```bash
cargo test --test test_cut
```

### Run specific test

```bash
cargo test test_cut_creation
```

### Run with output (see println! statements)

```bash
cargo test -- --nocapture
```

### Run with multiple threads

```bash
cargo test -- --test-threads=4
```

### Run tests matching pattern

```bash
cargo test cut  # Runs all tests with "cut" in the name
```

## Performance Considerations

From the HPC developer perspective:

### Fixture Performance

- **trivial_system**: O(1) allocation, ~100 bytes
- **simple_system**: O(1) allocation, ~500 bytes
- **medium_system**: O(1) allocation, ~2KB

All fixtures use stack allocation where possible and pre-sized vectors to minimize allocations.

### Mock Solver Performance

The mock solver uses:

- Atomic counters (lock-free, thread-safe)
- RefCell for interior mutability (zero overhead when not borrowed)
- No allocations in hot path (solve, get_objective_value)

### Test Execution Speed

Target execution times:

- Unit tests: <1ms each
- Integration tests with trivial_system: <10ms
- Integration tests with medium_system: <100ms
- Full test suite: <30 seconds

## Adding New Tests

When adding new tests:

1. **Choose the right fixture**: Start with the smallest fixture that tests your case
2. **Use utilities**: Leverage existing assertions rather than writing custom checks
3. **Document purpose**: Add a doc comment explaining what the test validates
4. **Test edge cases**: Include boundary conditions, zero, negative, extreme values
5. **Keep tests fast**: Avoid unnecessary work; use smaller systems when possible
6. **Make tests deterministic**: Use fixed seeds for any randomness

## Common Patterns

### Testing Algorithm Correctness

```rust
#[test]
fn test_algorithm_converges() {
    let system = simple_system();
    let scenarios = deterministic_scenario(2, 1, 1);

    // Run algorithm
    let result = run_sddp(system, scenarios);

    // Validate convergence
    assert!(result.converged);
    assert_float_approx_eq!(result.upper_bound, expected_value, 1e-6);
}
```

### Testing Numerical Stability

```rust
#[test]
fn test_numerical_stability_extreme_values() {
    let cut = BendersCut::new(
        vec![1e10],  // Large state
        vec![1e-10], // Small coefficient
        1.0
    ).unwrap();

    let intercept = cut.intercept_at(&vec![1e10 + 1.0]);

    // Should not produce NaN or Inf
    assert!(intercept.is_finite());
}
```

### Testing Physical Constraints

```rust
#[test]
fn test_reservoir_bounds_respected() {
    let system = trivial_system();
    let hydro = &system.hydros[0];

    // ... run optimization ...

    // Verify volumes stay within bounds
    for volume in &result.volumes {
        assert_state_within_bounds(
            volume,
            &vec![hydro.min_volume],
            &vec![hydro.max_volume],
            1e-6
        );
    }
}
```

## Future Enhancements

Planned improvements to test infrastructure:

- Property-based testing with `proptest` (Sprint 6)
- Benchmark fixtures for performance testing (Sprint 3)
- Fuzzing infrastructure (Sprint 6)
- Coverage measurement tooling (Sprint 1, T1.9)
- Test result visualization

## Questions?

For questions about testing:

- See TESTING.md in repository root (created in Sprint 1, T1.8)
- Consult the reviewer agent for test quality guidelines
- Refer to Rust testing documentation: https://doc.rust-lang.org/book/ch11-00-testing.html
