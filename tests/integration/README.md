# Integration Tests for POWE.RS SDDP

This directory contains integration tests that validate interactions between multiple components of the SDDP implementation.

## Directory Structure

```
tests/integration/
├── mod.rs                  # Main integration module
├── fixtures.rs             # Shared test fixtures
├── algorithm/              # Algorithm integration tests
│   ├── mod.rs
│   ├── forward_pass.rs     # Forward pass tests (TEST-020)
│   ├── backward_pass.rs    # Backward pass tests (TEST-021)
│   ├── training_loop.rs    # Training loop tests (TEST-022)
│   └── convergence.rs      # Convergence tests (TEST-023)
└── features/               # Feature integration tests
    ├── mod.rs
    ├── ar_models.rs        # AR model tests (TEST-025)
    ├── risk_measures.rs    # Risk measure tests (TEST-026)
    ├── scenario_generation.rs # Scenario tests (TEST-027)
    └── multi_reservoir.rs  # Cascade tests (TEST-029)
```

## Available Fixtures

### `setup_simple_algorithm()`
- **Purpose**: Minimal SDDP problem for basic tests
- **Characteristics**: 2 stages, 1 hydro, 1 thermal, deterministic
- **Use for**: Quick sanity checks, basic algorithm flow

### `setup_stochastic_algorithm()`
- **Purpose**: Test with uncertainty and scenario trees
- **Characteristics**: 3 stages, 3 scenarios/stage, stochastic inflows
- **Use for**: Forward/backward pass, convergence tests

### `setup_par_algorithm()`
- **Purpose**: Test AR inflow modeling
- **Characteristics**: 3 stages, AR(1) model, lagged states
- **Use for**: AR integration, state management tests

### `setup_cascade_algorithm()`
- **Purpose**: Test multi-reservoir coordination
- **Characteristics**: 3 stages, 2 hydros in cascade, time lag
- **Use for**: Cascade logic, downstream flow tests

## Running Tests

```bash
# Run all integration tests
cargo test --test integration

# Run specific category
cargo test --test integration -- algorithm
cargo test --test integration -- features

# Run with output
cargo test --test integration -- --nocapture

# Run in release mode (faster for long tests)
cargo test --test integration --release
```

## Writing New Integration Tests

### Example Test Structure

```rust
use crate::integration::fixtures::setup_simple_algorithm;
use powers::sddp::SAA;

#[test]
fn test_my_integration_scenario() {
    // 1. Setup: Get fixture
    let (system, graph, config) = setup_simple_algorithm();
    
    // 2. Execute: Run SDDP algorithm
    let mut saa = SAA::new(system, graph, config).unwrap();
    let result = saa.train().unwrap();
    
    // 3. Verify: Check properties
    assert!(result.converged());
    assert!(result.lower_bound() <= result.upper_bound());
}
```

### Test Guidelines

1. **Use fixtures**: Reuse provided fixtures instead of creating new systems
2. **Test interactions**: Focus on component boundaries and interfaces
3. **Be deterministic**: Always use fixed seeds for reproducibility
4. **Keep tests fast**: Target <500ms per test, <3min for full suite
5. **Clear assertions**: Use descriptive messages and assertion helpers
6. **Document intent**: Add comments explaining what you're testing

### Adding New Fixtures

If you need a new fixture:

1. Add function to `fixtures.rs`
2. Document characteristics clearly
3. Add smoke test in `fixtures.rs` tests module
4. Keep fixture creation fast (<50ms)
5. Use fixed seeds for reproducibility

## Test Philosophy

Integration tests should:

- ✅ Test realistic end-to-end workflows
- ✅ Use real LP solvers (not mocks)
- ✅ Verify mathematical properties across components
- ✅ Catch integration bugs at component boundaries
- ❌ Not duplicate unit test coverage
- ❌ Not test implementation details
- ❌ Not be slow (>1s per test is too slow)

## Implementation Status

- ✅ TEST-019: Integration test structure (COMPLETE)
- ⏳ TEST-020: Forward pass tests (PENDING)
- ⏳ TEST-021: Backward pass tests (PENDING)
- ⏳ TEST-022: Training loop tests (PENDING)
- ⏳ TEST-023: Convergence tests (PENDING)
- ⏳ TEST-025: AR model tests (PENDING)
- ⏳ TEST-026: Risk measure tests (PENDING)
- ⏳ TEST-027: Scenario generation tests (PENDING)
- ⏳ TEST-029: Multi-reservoir tests (PENDING)

## Related Documentation

- `TESTING_TICKETS_REVISED.md`: Full testing strategy
- `TESTING_PROGRESS_SUMMARY.md`: Current progress
- `tests/utils/`: Assertion utilities for tests
- `tests/fixtures/`: Unit test fixtures

---

**Last Updated**: 2025-11-06
