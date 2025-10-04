# Testing Guide

This guide explains how to write, run, and maintain tests for POWE.RS.

## Philosophy

### Why We Test POWE.RS

POWE.RS is a high-performance implementation of Stochastic Dual Dynamic Programming (SDDP) for hydrothermal dispatch optimization. Testing is critical because:

1. **Correctness**: SDDP algorithms are complex with many interacting components (forward passes, backward passes, cut selection, state management). Tests ensure these components work correctly in isolation and together.

2. **Numerical Stability**: We work with floating-point arithmetic, large-scale linear programs, and iterative algorithms. Tests verify numerical behavior and catch precision issues early.

3. **Performance**: While we prioritize speed, correctness comes first. Tests ensure optimizations don't break functionality.

4. **Confidence**: A comprehensive test suite enables refactoring and optimization with confidence.

5. **Documentation**: Tests demonstrate how to use the API and what behavior to expect.

### Testing Goals and Priorities

Our testing strategy follows the testing pyramid:

```
        /\
       /  \      E2E: Full algorithm runs (few, slow, comprehensive)
      /____\
     /      \    Integration: Multi-module workflows (moderate coverage)
    /        \
   /__________\  Unit: Individual functions and types (extensive, fast)
```

**Coverage Targets**:

- Unit tests: >80% coverage of core logic
- Integration tests: All major workflows
- Critical paths: 100% coverage (cut generation, state updates, solver interactions)

**Test Quality Standards**:

- ✅ **Fast**: Unit tests run in milliseconds, full suite in seconds
- ✅ **Deterministic**: Fixed random seeds, reproducible results
- ✅ **Independent**: Tests don't depend on execution order
- ✅ **Clear**: Test name and body explain what's being tested
- ✅ **Focused**: Each test verifies one behavior

## Test Organization

### Directory Structure

```
powers-rs/
├── src/
│   ├── lib.rs              # Unit tests alongside code
│   ├── cut.rs              # #[cfg(test)] mod tests { ... }
│   ├── state.rs
│   └── ...
├── tests/
│   ├── fixtures/           # Shared test data and utilities
│   │   ├── mod.rs          # Re-exports for convenience
│   │   ├── mock_solver.rs  # Mock solver for testing without HiGHS
│   │   ├── scenarios.rs    # Scenario generators
│   │   ├── simple_2stage_reservoir.rs  # 2-stage test system
│   │   └── systems.rs      # System configurations
│   ├── utils/              # Testing utilities
│   │   ├── mod.rs
│   │   └── assertions.rs   # Custom assertions (float comparisons, etc.)
│   ├── test_cut.rs         # Cut functionality tests
│   ├── test_state.rs       # State management tests
│   ├── test_scenario.rs    # Scenario generation tests
│   └── integration_simple_2stage.rs  # End-to-end SDDP test
└── benches/                # Performance benchmarks (future)
```

### Unit Tests vs Integration Tests

**Unit Tests** (`src/**/*.rs` with `#[cfg(test)]`):

- Test individual functions, methods, and types
- Fast execution (microseconds to milliseconds)
- Mock external dependencies when possible
- Focus on edge cases and correctness

**Integration Tests** (`tests/**/*.rs`):

- Test interactions between modules
- Use real dependencies (actual solver, not mocked)
- Verify end-to-end workflows
- Focus on realistic scenarios

**When to Choose**:

- Use **unit tests** for: Pure functions, data structures, algorithms, edge cases
- Use **integration tests** for: SDDP algorithm runs, solver interactions, multi-stage workflows

### Fixtures and Utilities

**Fixtures** (`tests/fixtures/`):

- Reusable test data and configurations
- System definitions (buses, hydros, thermals)
- Scenario generators (deterministic, stochastic)
- Mock implementations

**Utilities** (`tests/utils/`):

- Custom assertions for floating-point comparisons
- Test helpers and convenience functions
- Shared test infrastructure

Import fixtures in tests:

```rust
mod fixtures;
use fixtures::*;  // Imports all re-exported items
```

## Writing Tests

### General Guidelines

#### Test Structure: Arrange-Act-Assert

Follow the AAA pattern for clarity:

```rust
#[test]
fn test_benders_cut_evaluation() {
    // Arrange: Set up test data
    let coefficients = vec![1.5, -0.5];
    let rhs = 42.0;
    let cut = BendersCut::new(0, coefficients, rhs);
    let state = vec![100.0, 200.0];

    // Act: Execute the behavior being tested
    let value = cut.evaluate(&state);

    // Assert: Verify the result
    let expected = 1.5 * 100.0 + (-0.5) * 200.0 + 42.0;
    assert_float_approx_eq!(value, expected, 1e-10);
}
```

#### Test Independence

Each test should be independent and not rely on execution order:

```rust
// ❌ BAD: Tests share mutable state
static mut COUNTER: i32 = 0;

#[test]
fn test_first() {
    unsafe { COUNTER += 1; }
}

#[test]
fn test_second() {
    unsafe { assert_eq!(COUNTER, 1); }  // Fails if test_first runs first!
}

// ✅ GOOD: Each test is self-contained
#[test]
fn test_first() {
    let mut counter = 0;
    counter += 1;
    assert_eq!(counter, 1);
}

#[test]
fn test_second() {
    let counter = 1;
    assert_eq!(counter, 1);
}
```

#### Test Naming

Use descriptive names that explain what's being tested:

```rust
// ❌ BAD: Unclear what's being tested
#[test]
fn test_cut() { ... }

#[test]
fn test_1() { ... }

// ✅ GOOD: Clear and descriptive
#[test]
fn test_benders_cut_evaluates_correctly_at_reference_state() { ... }

#[test]
fn test_storage_state_handles_empty_initial_storage() { ... }

#[test]
fn test_forward_pass_respects_storage_constraints() { ... }
```

Pattern: `test_<component>_<behavior>_<condition>`

#### Test Documentation

Add doc comments for complex tests:

```rust
/// Tests that the backward pass correctly propagates future costs through Benders cuts.
///
/// This test verifies that:
/// 1. Cuts are generated for each child node
/// 2. Cut coefficients reflect water values
/// 3. Cut intercepts account for expected future costs
/// 4. Cuts are properly activated in the parent subproblem
#[test]
fn test_backward_pass_cut_propagation() {
    // ... test implementation ...
}
```

### Testing Numerical Code

#### Floating-Point Comparisons

Never use `assert_eq!` for floating-point values:

```rust
// ❌ BAD: Exact equality fails due to floating-point precision
#[test]
fn test_bad_float_comparison() {
    let result = 0.1 + 0.2;
    assert_eq!(result, 0.3);  // FAILS! (0.30000000000000004 != 0.3)
}

// ✅ GOOD: Use approximate equality with tolerance
#[test]
fn test_good_float_comparison() {
    let result = 0.1 + 0.2;
    assert!((result - 0.3).abs() < 1e-10);
}

// ✅ BETTER: Use custom assertion from utils
#[test]
fn test_best_float_comparison() {
    use crate::utils::assertions::*;
    let result = 0.1 + 0.2;
    assert_float_approx_eq!(result, 0.3, 1e-10);
}
```

#### Tolerance Selection

Choose tolerance based on the numerical context:

- **1e-10**: High-precision tests, values near 1.0
- **1e-7**: Solver default tolerance (HiGHS primal/dual feasibility)
- **1e-6**: After multiple solver iterations
- **1e-4**: After many numerical operations or when comparing objective values
- **1e-2**: When testing statistical properties over many samples

```rust
#[test]
fn test_solver_primal_feasibility() {
    let solution = solve_problem(&problem);
    // Use solver tolerance for solver results
    assert_float_approx_eq!(solution.objective, expected, 1e-7);
}

#[test]
fn test_statistical_convergence() {
    let samples = run_monte_carlo(10000);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    // Looser tolerance for statistical tests
    assert_float_approx_eq!(mean, expected_mean, 1e-2);
}
```

#### Handling NaN and Inf

Test edge cases explicitly:

```rust
#[test]
fn test_handles_infinite_cost() {
    let result = compute_cost(f64::INFINITY);
    assert!(result.is_infinite());
}

#[test]
fn test_rejects_nan_input() {
    let result = process_value(f64::NAN);
    assert!(result.is_err());
}
```

### Testing SDDP Convergence

#### Understanding SDDP Bounds

SDDP generates two types of bounds during training:

**Lower Bound (LB)**: Computed from Bellman cuts in the backward pass. Should be:

- Non-decreasing across iterations (monotonic)
- Eventually converge to the optimal policy value
- Always ≤ true optimal value (optimistic bound)

**Upper Bound**: The **statistical upper bound** is the average of ALL forward pass costs across ALL iterations:

```rust
statistical_UB = (1/N) Σ_{i=1}^N cost_i  where N = total forward passes
```

**Critical Distinction**:

- **Per-iteration simulation cost**: Individual forward pass cost for one iteration (can be < LB due to sampling variance)
- **Statistical upper bound**: Average of all forward pass costs (true SDDP upper bound per theory)

**SDDP Invariant**: `Lower Bound ≤ E[optimal value] ≤ Statistical Upper Bound`

**References**:

- Shapiro, A., Tekaya, W., da Costa, J. P., & Soares, M. P. (2011). "Risk neutral and risk averse Stochastic Dual Dynamic Programming method"
- Philpott, A. B., & de Matos, V. L. (2012). "Dynamic sampling algorithms for multi-stage stochastic programs with risk aversion"

#### Using TrainingResult for Validation

All SDDP training returns `TrainingResult` which captures convergence history:

```rust
#[test]
fn test_sddp_convergence_basic() -> Result<(), String> {
    let system = create_test_system();
    let mut sddp = SDDP::new(system, config)?;
    let saa = create_test_saa();

    // Train and capture result
    let result = sddp.train(30, 10, &saa)?;

    // Use helper function for standard checks
    assert_convergence_quality(&result)?;

    // Access convergence data
    println!("Final LB: {:.2}", result.final_lower_bound);
    println!("Statistical UB: {:.2}", result.statistical_upper_bound);
    println!("Statistical Gap: {:.2}",
             result.statistical_upper_bound - result.final_lower_bound);

    Ok(())
}
```

#### Helper Functions for Convergence Validation

Import from `tests/utils/assertions`:

```rust
use tests::utils::assertions::{
    assert_convergence_quality,
    assert_bounds_in_range,
    print_convergence_summary,
};
```

**`assert_convergence_quality(result)`**: Comprehensive validation

- Verifies lower bound monotonicity (non-decreasing with 1e-6 tolerance)
- Validates SDDP invariant: `LB ≤ statistical_UB`
- Checks that statistical gap is non-negative
- Ensures all bounds are finite (not NaN, not Inf)
- For zero-cost problems, verifies small gaps

```rust
#[test]
fn test_convergence_validation() -> Result<(), String> {
    let result = train_simple_problem()?;

    // Single call checks all convergence properties
    assert_convergence_quality(&result)?;

    Ok(())
}
```

**`assert_bounds_in_range(result, min, max)`**: Range validation for known solutions

```rust
#[test]
fn test_newsvendor_benchmark() -> Result<(), String> {
    let result = train_newsvendor_problem()?;

    // For known analytical solution ≈ 1500
    assert_bounds_in_range(&result, 1400.0, 1600.0)?;

    Ok(())
}
```

**`print_convergence_summary(result)`**: Debugging output

```rust
// Use for debugging or detailed validation
print_convergence_summary(&result);
// Outputs:
// === Convergence Summary ===
// Final lower bound: 183.66
// Statistical upper bound: 185.23
// Statistical gap: 1.57
// Relative gap: 0.85%
// Best iteration: 28
// ===========================
```

#### Convergence Properties to Test

**1. Monotonicity**: Lower bounds must not decrease

```rust
#[test]
fn test_convergence_monotonicity() -> Result<(), String> {
    let result = sddp.train(50, 10, &saa)?;

    // Check each iteration
    for i in 1..result.iterations.len() {
        let prev_lb = result.iterations[i-1].lower_bound;
        let curr_lb = result.iterations[i].lower_bound;

        assert!(
            curr_lb >= prev_lb - 1e-6,  // 1e-6 tolerance for numerical precision
            "Lower bound decreased at iteration {}: {:.6} -> {:.6}",
            i, prev_lb, curr_lb
        );
    }

    Ok(())
}
```

**2. Statistical Gap Convergence**: Gap should be reasonable after training

```rust
#[test]
fn test_convergence_gap_decrease() -> Result<(), String> {
    let result = sddp.train(50, 10, &saa)?;

    let statistical_gap = result.statistical_upper_bound - result.final_lower_bound;

    // Statistical gap must be non-negative (SDDP invariant)
    assert!(
        statistical_gap >= -1e-6,
        "Statistical gap is negative: {:.6}",
        statistical_gap
    );

    // For well-converged problems, expect reasonable relative gap
    let relative_gap = statistical_gap / result.final_lower_bound.abs().max(1.0);
    assert!(
        relative_gap < 0.20,  // < 20% after 50 iterations
        "Relative gap too large: {:.2}%",
        relative_gap * 100.0
    );

    Ok(())
}
```

**3. Bounds Validity**: All bounds must be finite and properly ordered

```rust
#[test]
fn test_convergence_bounds_validity() -> Result<(), String> {
    let result = sddp.train(30, 10, &saa)?;

    // Check final bounds
    assert!(result.final_lower_bound.is_finite(), "LB not finite");
    assert!(result.statistical_upper_bound.is_finite(), "Statistical UB not finite");

    // SDDP invariant: LB ≤ Statistical UB
    assert!(
        result.final_lower_bound <= result.statistical_upper_bound + 1e-6,
        "Lower bound exceeds statistical upper bound: LB={:.6} > UB={:.6}",
        result.final_lower_bound,
        result.statistical_upper_bound
    );

    // Check each iteration's LB is monotonic
    for window in result.iterations.windows(2) {
        assert!(
            window[1].lower_bound >= window[0].lower_bound - 1e-6,
            "Lower bound decreased between iterations"
        );
    }

    Ok(())
}
```

**4. Stability**: Bounds should converge smoothly without wild oscillations

```rust
#[test]
fn test_convergence_stability() -> Result<(), String> {
    let result = sddp.train(50, 10, &saa)?;

    // Extract lower bounds
    let lower_bounds: Vec<f64> = result.iterations
        .iter()
        .map(|it| it.lower_bound)
        .collect();

    // Check for large jumps (no decrease, small increases)
    for window in lower_bounds.windows(2) {
        let prev = window[0];
        let curr = window[1];
        let jump = curr - prev;

        // Jump should be non-negative and reasonable
        assert!(
            jump >= -1e-6,
            "Lower bound decreased: {} -> {}",
            prev, curr
        );
    }

    Ok(())
}
```

#### Tolerance Guidelines

**Why 1e-6 for monotonicity?**

- MILP solvers have numerical precision limits (~1e-9 to 1e-12)
- Small floating-point errors are acceptable
- 1e-6 catches real decreases while tolerating numerical noise
- Stricter tolerances would cause false test failures

**Why allow negative statistical gaps with tolerance?**

- Per-iteration simulation costs can be below LB (sampling variance)
- Statistical average should be ≥ LB, but small violations (< 1e-6) are numerical noise
- Strict enforcement would cause flaky tests with borderline convergence

**Why 20% relative gap threshold?**

- Conservative target for general problems
- Some problems converge slower (more stages, more uncertainty)
- Allows tests to pass while still catching poor convergence
- Avoids brittle tests that fail on minor algorithmic changes

#### Edge Cases

**Zero-Cost Problems**: Handle trivial problems gracefully

```rust
// Helper automatically detects and handles zero-cost case
assert_convergence_quality(&result)?;

// Manual check if needed:
if result.final_lower_bound.abs() < 1e-6 &&
   result.statistical_upper_bound.abs() < 1e-6 {
    // Problem has zero optimal cost (load < generation capacity)
    // Gap checks not meaningful
    println!("Zero-cost problem detected");
}
```

**Small Sample Size**: Ensure enough samples for statistical upper bound

```rust
// Bad: Only 50 samples (10 iterations × 5 forward passes)
let result = sddp.train(10, 5, &saa)?;  // May have high variance

// Good: 300 samples (30 iterations × 10 forward passes)
let result = sddp.train(30, 10, &saa)?;  // More reliable statistical UB
```

**Water Scarcity Problems**: Expect meaningful convergence

For hydrothermal dispatch, create systems with tight water budgets to force trade-offs:

```rust
// Example: Load 75 MW, Hydro 60 MW, Storage 40 MWh
// With inflows that make water scarce in some scenarios
// This forces SDDP to learn water value (when to save vs. use thermal)
```

#### Common Pitfalls

❌ **Don't compare per-iteration UB to LB**:

```rust
// WRONG: Per-iteration cost can be < LB
assert!(iter_result.upper_bound >= iter_result.lower_bound);  // Can fail!
```

✅ **Do use statistical upper bound**:

```rust
// CORRECT: Statistical average must be ≥ LB
assert!(result.statistical_upper_bound >= result.final_lower_bound - 1e-6);
```

❌ **Don't expect LB to always improve each iteration**:

```rust
// WRONG: Simple problems converge quickly
assert!(result.iterations[10].lower_bound > result.iterations[5].lower_bound);
```

✅ **Do check monotonicity with tolerance**:

```rust
// CORRECT: LB is non-decreasing
assert!(curr_lb >= prev_lb - 1e-6);
```

❌ **Don't test convergence on trivial problems**:

```rust
// WRONG: Load < Hydro capacity → zero cost, no learning
let system = create_system_with_load(25.0, hydro_capacity: 50.0);
```

✅ **Do create meaningful test problems**:

```rust
// CORRECT: Load > Hydro capacity → requires thermal, has cost
let system = create_system_with_load(75.0, hydro_capacity: 60.0);
```

#### Example: Complete Convergence Test

```rust
#[test]
fn test_hydrothermal_convergence() -> Result<(), String> {
    // Create problem with meaningful water value trade-offs
    let system = create_water_scarce_system();  // Load 75 MW, Hydro 60 MW
    let config = SDDPConfig {
        cut_selection_threshold: Some(1e-6),
        ..Default::default()
    };
    let mut sddp = SDDP::new(system, config)?;
    let saa = create_stochastic_inflow_saa(num_scenarios: 5, seed: 42);

    // Train with sufficient samples for statistical reliability
    let result = sddp.train(
        num_iterations: 30,
        num_forward_passes: 10,  // 300 total forward passes
        &saa
    )?;

    // Standard convergence validation
    assert_convergence_quality(&result)?;

    // Problem-specific validation (if solution known approximately)
    assert_bounds_in_range(&result, 150.0, 220.0)?;

    // Optional: Print detailed summary for debugging
    print_convergence_summary(&result);

    // Verify reasonable statistical gap
    let statistical_gap = result.statistical_upper_bound - result.final_lower_bound;
    assert!(
        statistical_gap < 20.0,  // Absolute gap < 20 for this problem
        "Statistical gap too large: {:.2}",
        statistical_gap
    );

    Ok(())
}
```

### Testing Randomness

#### Using Fixed Seeds

Always use fixed seeds for reproducibility:

```rust
// ❌ BAD: Non-deterministic test
#[test]
fn test_random_scenario_generation() {
    let scenario = generate_random_scenario();  // Different each run!
    assert!(scenario.is_valid());
}

// ✅ GOOD: Deterministic with fixed seed
#[test]
fn test_random_scenario_generation() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let scenario = generate_random_scenario(&mut rng);
    assert!(scenario.is_valid());
    // Can verify exact values since seed is fixed
    assert_eq!(scenario.stages[0].load, expected_load);
}
```

#### Testing Statistical Properties

When testing randomness, verify statistical properties over many samples:

```rust
#[test]
fn test_scenario_generator_mean() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let generator = NormalGenerator::new(100.0, 10.0);

    let samples: Vec<f64> = (0..10000)
        .map(|_| generator.sample(&mut rng))
        .collect();

    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let std_dev = calculate_std_dev(&samples);

    // Statistical test with appropriate tolerance
    assert_float_approx_eq!(mean, 100.0, 0.5);  // ~5 sigma for 10k samples
    assert_float_approx_eq!(std_dev, 10.0, 0.5);
}
```

### Using Fixtures

#### Available Fixtures

Import from `tests/fixtures/`:

```rust
mod fixtures;
use fixtures::*;

#[test]
fn test_with_simple_system() {
    // Simple 2-stage reservoir system
    let system = create_simple_2stage_system();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa();

    // Use in test
    assert_eq!(system.hydros.len(), 1);
}
```

**Available Systems**:

- `create_simple_2stage_system()`: Minimal 2-stage reservoir (1 hydro, 1 bus, 2 thermals)
- `System::default()`: Single-stage test system

**Scenario Generators**:

- `deterministic_scenario()`: No uncertainty, fixed values
- `fan_scenario()`: Simple fan-out with 3 scenarios per stage
- `simple_stochastic_scenario()`: Configurable stochastic scenarios

**Mock Components**:

- `MockSolver`: For testing algorithm logic without HiGHS

#### Creating New Fixtures

Add fixtures to `tests/fixtures/` when they'll be reused across multiple tests:

```rust
// In tests/fixtures/systems.rs
pub fn create_three_stage_cascade_system() -> System {
    let hydros = vec![
        Hydro::new(0, Some(1), 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.01),
        Hydro::new(1, Some(2), 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.01),
        Hydro::new(2, None, 0, 1.0, 0.0, 100.0, 0.0, 50.0, 0.01),
    ];
    // ... rest of system configuration ...
}

// In tests/fixtures/mod.rs
pub use systems::create_three_stage_cascade_system;
```

**Guidelines**:

- Make fixtures generic and configurable
- Document what each fixture represents
- Use realistic parameter values
- Keep fixtures simple and focused

## Numerical Validation

### Purpose

Numerical validation tests verify that SDDP produces **numerically correct results**, not just "doesn't crash." These tests validate fundamental algorithmic properties from SDDP convergence theory.

**Standard tests**: Validate code execution (no panics, coverage, API contracts)  
**Numerical tests**: Validate algorithmic correctness (bounds, convergence, statistical properties)

### Validation Principles

Numerical validation tests in `tests/test_numerical_validation.rs` verify five fundamental properties:

#### 1. Lower Bound Monotonicity (SDDP Theory)

**Property**: Lower bound must be non-decreasing across iterations.

**Mathematical Foundation**:

$$
LB_k \leq LB_{k+1} \quad \forall k
$$

where $LB_k$ is the lower bound at iteration $k$.

**Why It Matters**: This is a fundamental property proven in Pereira & Pinto (1991). Each backward pass adds cuts that improve or maintain the lower bound approximation. Violations indicate algorithm bugs in cut generation.

**Test**: `test_lower_bound_monotonicity()`

**Tolerance**: 1e-6 (allows tiny numerical noise from LP solver, catches real violations)

**References**:

- Pereira & Pinto (1991): "Multi-stage stochastic optimization applied to energy planning"
- Shapiro (2011): "Analysis of stochastic dual dynamic programming method"

#### 2. Gap Reduction (Convergence Property)

**Property**: Optimality gap should decrease over iterations.

**Mathematical Foundation**:

$$
\text{Gap}_k = UB_k - LB_k
$$

where late iterations show: $\mathbb{E}[\text{Gap}_{\text{late}}] < \mathbb{E}[\text{Gap}_{\text{early}}]$

**Why It Matters**: More iterations improve the piecewise linear approximation of the future cost function, leading to tighter bounds and smaller gaps.

**Test**: `test_gap_reduction_trend()`

**Expected**: 50% improvement (late 20% of iterations vs early 20%)

**Note**: Gap may fluctuate due to stochastic sampling, but the trend should be downward.

#### 3. Bounds Bracket Optimal (Correctness Validation)

**Property**: Lower bound ≤ optimal ≤ Upper bound.

**Mathematical Foundation**:

$$
LB \leq V^* \leq UB
$$

where $V^*$ is the true optimal value.

**Why It Matters**: This validates SDDP converges to the correct solution. If bounds don't bracket the known optimal value, the algorithm has a correctness issue.

**Test**: `test_bounds_bracket_optimal()`

**Requires**: Benchmarks with known optimal values

**Tolerance**: Problem-dependent (deterministic: ±1.0, stochastic: ±5.0)

#### 4. Statistical Properties (Monte Carlo Theory)

**Property**: Forward pass variance reduces with more samples.

**Mathematical Foundation**:

$$
\text{Var}(\bar{X}) = \frac{\sigma^2}{n}
$$

where $n$ is the number of forward passes, $\sigma^2$ is the variance of individual samples.

**Why It Matters**: Forward passes are unbiased estimators of the optimal objective. More samples → better estimates → tighter upper bounds.

**Test**: `test_forward_pass_variance_convergence()`

**Expected**: Standard deviation scales as $1/\sqrt{n}$

**Reference**: Law of Large Numbers

#### 5. Numerical Stability (Robustness)

**Property**: No NaN or Inf in any iteration.

**Why It Matters**: NaN/Inf indicates serious problems:

- Unbounded subproblems
- Numerical overflow/underflow
- Solver issues
- Poor problem scaling

**Test**: `test_no_nan_or_inf()`

**Validates**: All bounds and costs are finite across all iterations and benchmarks.

### Benchmark-Based Testing

**Why Benchmarks?**: To validate correctness, we need problems with known solutions.

**Available Benchmarks** (`tests/fixtures/benchmarks.rs`):

1. **Deterministic Single Reservoir**
   - 2 stages, ample water
   - Expected optimal: $0 (all hydro, no thermal)
   - Convergence: Gap < $0.50
2. **Stochastic Single Reservoir**
   - 2 stages, 3 inflow scenarios (dry/average/wet)
   - Expected optimal: $20-$40 (hedging cost)
   - Convergence: Gap < $5.00
3. **Two Reservoir Cascade**
   - 3 stages, cascade topology
   - Expected optimal: $0-$50 (depends on inflows)
   - Convergence: Gap < $10.00

**Example Usage**:

```rust
use fixtures::benchmarks::create_deterministic_single_reservoir;

#[test]
fn test_bounds_bracket_optimal() {
    let (mut sddp, saa) = create_deterministic_single_reservoir()
        .expect("Failed to create benchmark");

    let result = sddp.train(30, 10, &saa)
        .expect("Training failed");

    // Validate bounds bracket known optimal ($0)
    assert!(result.final_lower_bound <= 1.0);
    assert!(result.final_upper_bound >= -1.0);
}
```

### Helper Functions

The test suite provides reusable assertion helpers:

- **`assert_monotonicity(result, tolerance)`**: Check lower bounds are non-decreasing
- **`assert_gap_reduction(result, early_frac, late_frac, improvement)`**: Validate convergence trend
- **`assert_bounds_valid(result, expected_optimal, tolerance)`**: Verify bounds bracket optimal
- **`assert_no_numerical_issues(result)`**: Detect NaN/Inf values
- **`compute_variance(values)`**: Statistical variance calculation

**Example**:

```rust
// Helper does the heavy lifting
assert_monotonicity(&result, 1e-6);

// Instead of manual loop:
// for i in 1..result.iterations().len() {
//     assert!(result.iterations()[i].lower_bound >=
//             result.iterations()[i-1].lower_bound - 1e-6);
// }
```

### Interpreting Test Failures

| Failure Type                   | Likely Cause                   | Next Steps                                              |
| ------------------------------ | ------------------------------ | ------------------------------------------------------- |
| **Monotonicity violation**     | Algorithm bug (cut generation) | Check backward pass logic, cut validity, dual recovery  |
| **Gap not reducing**           | Convergence issue              | Increase iterations, check forward pass sampling        |
| **Bounds don't bracket**       | Correctness issue              | Validate subproblem formulation, check constraints      |
| **NaN/Inf detected**           | Numerical stability issue      | Check solver settings, problem scaling, variable bounds |
| **High forward pass variance** | Sampling issue                 | Increase forward passes, verify SAA quality             |
| **Policy unreasonable**        | Model/data issue               | Check system parameters, inflows, demands               |

### Debugging Numerical Issues

If numerical validation tests fail:

1. **Check Solver Logs** (HiGHS output):

   ```bash
   RUST_LOG=debug cargo test test_lower_bound_monotonicity -- --nocapture
   ```

   Look for: infeasibility, unboundedness, large coefficients

2. **Inspect Iteration History**:

   ```rust
   for (i, iter) in result.iterations().iter().enumerate() {
       println!("Iter {}: LB={:.4}, UB={:.4}, Gap={:.4}",
                i+1, iter.lower_bound, iter.upper_bound, iter.gap);
   }
   ```

3. **Verify Problem Data**:

   - Check inflows are positive
   - Check demands are reasonable
   - Check storage bounds: $0 \leq s_t \leq \text{max\_storage}$
   - Check turbining bounds: $0 \leq q_t \leq \text{max\_turbined\_flow}$

4. **Check Cut Pool**:

   ```rust
   println!("Number of cuts: {}", sddp.num_cuts());
   // Look for cut explosion or unexpectedly few cuts
   ```

5. **Reduce Problem Size**: Test with fewer stages or scenarios to isolate issues

6. **Compare to Reference**: Run same problem in another SDDP implementation if available

### Running Numerical Validation Tests

```bash
# Run all numerical validation tests
cargo test --test test_numerical_validation

# Run specific validation test
cargo test test_lower_bound_monotonicity

# Run with detailed output
cargo test --test test_numerical_validation -- --nocapture

# Run benchmarks with validation
cargo test --test test_benchmarks
```

### Adding New Validation Tests

When adding new validation tests:

1. **Identify the property**: What mathematical property are you validating?
2. **Choose appropriate benchmark**: Use benchmark with known solution
3. **Set realistic tolerances**: Too tight → flaky tests, too loose → miss bugs
4. **Document expected behavior**: Explain why the property should hold
5. **Consider edge cases**: Deterministic vs stochastic, small vs large problems

**Template**:

```rust
#[test]
fn test_new_property() {
    // Create benchmark with known behavior
    let (mut sddp, saa) = create_benchmark()
        .expect("Failed to create benchmark");

    // Train
    let result = sddp.train(num_iterations, num_forward_passes, &saa)
        .expect("Training failed");

    // Validate property with clear assertion message
    assert!(
        property_holds(&result),
        "Property violated: details={:?}",
        result
    );
}
```

### References

Key SDDP convergence theory papers:

1. **Pereira & Pinto (1991)**: "Multi-stage stochastic optimization applied to energy planning"
   - Original SDDP convergence proof (lower bound monotonicity)
2. **Shapiro (2011)**: "Analysis of stochastic dual dynamic programming method"
   - Statistical properties, forward pass variance
3. **Philpott & de Matos (2012)**: "Dynamic sampling algorithms for multi-stage stochastic programs"
   - Sampling strategies, convergence rates
4. **Girardeau et al. (2015)**: "On the convergence of decomposition methods for multistage stochastic convex programs"
   - General convergence theory for SDDP variants

## Solver Interface Testing

### Purpose

The solver interface (`src/solver.rs`) is a **critical integration point** with the HiGHS optimization solver. Comprehensive testing ensures:

1. **Correctness**: Solver correctly formulates and solves LP/MIP problems
2. **Error Handling**: Infeasible/unbounded problems are detected gracefully
3. **Performance**: Solver handles scale without degradation
4. **Stability**: No memory leaks or numerical issues

**Test Strategy**:

- **Real Solver Tests**: Use actual HiGHS solver for integration validation
- **Mock Solver Tests**: Use mock for algorithm logic (fast, isolated)
- **Edge Cases**: Test boundary conditions, numerical limits, error states
- **Performance**: Validate scalability and memory stability

### Test Categories

#### 1. Real Solver Integration Tests

Tests that use the actual HiGHS solver to validate correct behavior:

**Basic LP Solve** (`test_simple_lp_optimal`):

```rust
// Problem: Minimize x + 2y
//          Subject to: x + y >= 1, x,y >= 0
// Optimal: x=1, y=0, obj=1

let mut problem = Problem::new();
problem.add_column(1.0, 0.0..); // x >= 0, cost 1
problem.add_column(2.0, 0.0..); // y >= 0, cost 2
problem.add_row(1.0.., [(0, 1.0), (1, 1.0)]); // x + y >= 1

let mut model = problem.optimise(Sense::Minimise);
model.solve();

assert_eq!(model.status(), HighsModelStatus::Optimal);
assert!((model.get_objective_value() - 1.0).abs() < 1e-6);
```

**Purpose**: Verify basic LP solving works correctly with expected solution.

**Infeasible Problem** (`test_infeasible_problem`):

```rust
// Problem: x >= 10 AND x <= 5 (contradictory)
let mut problem = Problem::new();
problem.add_column(1.0, 0.0..);
problem.add_row(10.0.., [(0, 1.0)]); // x >= 10
problem.add_row(..=5.0, [(0, 1.0)]); // x <= 5

let mut model = problem.optimise(Sense::Minimise);
model.solve();

assert!(model.status() == HighsModelStatus::Infeasible
        || model.status() == HighsModelStatus::UnboundedOrInfeasible);
```

**Purpose**: Verify solver detects infeasibility correctly.

**Unbounded Problem** (`test_unbounded_problem`):

```rust
// Problem: Minimize -x with no upper bound on x
let mut problem = Problem::new();
problem.add_column(-1.0, 0.0..); // Minimize -x = maximize x

let mut model = problem.optimise(Sense::Minimise);
model.solve();

assert!(model.status() == HighsModelStatus::Unbounded
        || model.status() == HighsModelStatus::UnboundedOrInfeasible);
```

**Purpose**: Verify solver detects unboundedness.

#### 2. Edge Case Tests

**Single Variable Problem** (`test_single_variable_problem`):

- Simplest possible problem (1 variable, 0 constraints)
- Tests solver handles trivial cases correctly

**Empty Problem** (`test_empty_problem_construction`):

- Zero variables and constraints
- Should fail gracefully or return appropriate status

**Equality Constraints** (`test_equality_constraint`):

- Tests that x + y = 10 (not just ≤ or ≥) works correctly
- Validates constraint handling

**Bounded Variables at Limits** (`test_bounded_variable_at_limit`):

- Variable x ∈ [3, 7] with minimize x → x = 3
- Tests variable bounds are respected

**Degenerate Problem** (`test_degenerate_problem`):

- Multiple optimal solutions (any x+y=10)
- Tests solver returns valid solution from degenerate set

**Numerical Edge Case** (`test_numerical_edge_case_large_coefficients`):

- Coefficients: 1e10 and 1e-10
- Tests numerical stability with extreme values

#### 3. Performance Tests

**Large Problem** (`test_large_problem_performance`):

```rust
// 1000 variables, 500 constraints
let mut problem = Problem::new();
for _ in 0..1000 {
    problem.add_column(1.0, 0.0..=1.0);
}
for i in 0..500 {
    // Each constraint: sum of 10 variables <= 100
    let row: Vec<_> = (0..10).map(|j| ((i*2+j)%1000, 1.0)).collect();
    problem.add_row(..=100.0, row);
}

let start = Instant::now();
model.solve();
let elapsed = start.elapsed();

assert!(elapsed < Duration::from_secs(1));
```

**Purpose**: Verify solver handles scale efficiently (<1s for 1000x500 problem).

**Repeated Solves** (`test_repeated_solves_no_memory_leak`):

```rust
// Solve same problem 100 times
for i in 0..100 {
    let problem = create_problem();
    let mut model = problem.optimise(Sense::Minimise);
    model.solve();
    // Track solve time
}

// Verify last 10 solves not significantly slower than first 10
assert!(last_10_avg < first_10_avg * 2);
```

**Purpose**: Catch memory leaks (solve time shouldn't increase).

**Model Reuse** (`test_model_reuse_with_modifications`):

```rust
// Solve, modify constraint, re-solve
model.solve(); // x + y >= 5, obj = 5
model.change_rows_bounds(0, 10.0, f64::INFINITY); // x + y >= 10
model.solve(); // obj = 10 (tighter constraint)

assert!(obj2 > obj1);
```

**Purpose**: Verify model can be modified and re-solved (basis warm-start).

#### 4. Solver Interface Contract Tests

**Solution Vector Size** (`test_solution_vector_size_matches_variables`):

- Verify `solution.colvalue.len() == num_variables`
- Validates interface contract

**Objective Value Consistency** (`test_objective_value_consistent_with_solution`):

- Verify `get_objective_value()` matches manual computation from solution
- Catches inconsistencies

**Row/Column Counts** (`test_num_cols_and_rows_correct`):

- Verify `num_cols()` and `num_rows()` return correct counts
- Validates problem structure

### Mock Solver vs Real Solver

**When to Use Mock Solver** (`tests/fixtures/mock_solver.rs`):

- ✅ Testing algorithm logic (SDDP iterations, convergence checks)
- ✅ Testing error propagation (what happens when solver fails)
- ✅ Fast unit tests (no HiGHS dependency)
- ✅ Controlled behavior (simulate specific scenarios)

Example:

```rust
let mock = MockSolver::new()
    .with_status(MockSolverStatus::Infeasible);

// Test that SDDP handles infeasibility gracefully
let result = sddp.train_with_solver(&mock);
assert!(result.is_err());
```

**When to Use Real Solver** (`tests/test_solver_interface.rs`):

- ✅ Testing solver interface correctness
- ✅ Validating LP formulation
- ✅ Testing numerical properties
- ✅ Performance validation
- ✅ Integration testing (end-to-end)

**Trade-offs**:
| Aspect | Mock Solver | Real Solver |
| --------------- | ------------------- | -------------------- |
| Speed | Very fast (<1ms) | Moderate (1-100ms) |
| Isolation | Complete | External dependency |
| Realism | Simulated | Actual behavior |
| Numerical Tests | Cannot validate | Full validation |
| Coverage | Algorithm logic | Solver integration |

### Running Solver Tests

```bash
# Run all solver interface tests
cargo test --test test_solver_interface

# Run specific solver test
cargo test test_large_problem_performance

# Run with output (see timing)
cargo test --test test_solver_interface -- --nocapture

# Run mock solver tests
cargo test mock_solver

# Run only performance tests
cargo test performance
```

### Interpreting Solver Test Failures

| Failure                           | Likely Cause                            | Next Steps                                          |
| --------------------------------- | --------------------------------------- | --------------------------------------------------- |
| **Optimal solution incorrect**    | LP formulation bug                      | Check constraint/objective formulation              |
| **Infeasible not detected**       | Missing constraints or incorrect bounds | Verify problem is actually infeasible               |
| **Unbounded not detected**        | Missing variable bounds                 | Check that all variables have appropriate bounds    |
| **Performance regression**        | Algorithm change or HiGHS update        | Profile to identify bottleneck, check HiGHS version |
| **Memory leak (time increasing)** | Resource not freed                      | Check for model/solver cleanup, use valgrind        |
| **Numerical instability**         | Poor problem scaling                    | Scale coefficients, check condition number          |
| **Solution mismatch**             | Solver interface bug                    | Verify get_solution() and get_objective_value()     |

### Adding New Solver Tests

When adding solver tests:

1. **Identify the property**: What are you validating?

   - Correctness (right answer)
   - Error handling (graceful failure)
   - Performance (fast enough)
   - Contract (interface guarantees)

2. **Choose test type**:

   - Real solver: For correctness and numerical properties
   - Mock solver: For algorithm logic and error handling

3. **Set realistic tolerances**:

   - LP solver: 1e-6 for objective/solution (HiGHS default)
   - Timing: Allow 2x variance for system noise
   - Memory: <2x slowdown acceptable (GC, system load)

4. **Document expected behavior**:

   ```rust
   #[test]
   fn test_new_property() {
       // Problem: <describe LP formulation>
       // Expected: <optimal solution>
       // Purpose: <what this validates>
   }
   ```

5. **Keep tests fast**:
   - Small problems for basic tests (<10 variables)
   - Larger problems (1000+ vars) only for performance tests
   - Target: <100ms per test, <2s total suite

**Template**:

```rust
#[test]
fn test_new_solver_property() {
    // Setup: Create problem
    let mut problem = Problem::new();
    // ... add variables and constraints ...

    // Solve
    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    // Validate property
    assert_eq!(model.status(), HighsModelStatus::Optimal);
    let solution = model.get_solution();
    assert!((solution.colvalue[0] - expected).abs() < 1e-6,
            "Property not satisfied: details");
}
```

### Performance Considerations

**Why Performance Tests Matter**:

- SDDP solves **thousands of LPs** per run (30 iterations × 20 forward passes × 10 stages = 6000 solves)
- Even small regressions (1ms → 2ms) compound (6s → 12s total)
- Memory leaks can crash long-running training

**Performance Baselines**:

- Simple LP (5 vars, 3 constraints): <1ms
- Medium LP (100 vars, 50 constraints): <10ms
- Large LP (1000 vars, 500 constraints): <100ms

**Profiling Solver Performance**:

```bash
# Profile solver tests with flamegraph
cargo test --test test_solver_interface --release -- --nocapture
cargo flamegraph --test test_solver_interface -- test_large_problem_performance

# Check for memory leaks
valgrind --leak-check=full --show-leak-kinds=all \
    ./target/debug/deps/test_solver_interface-*
```

**Optimization Opportunities**:

- **Basis warm-starting**: Reuse basis from previous solve (~2-5x speedup)
- **Cut selection**: Limit constraint growth (solver time ∝ O(rows²))
- **Model reuse**: Modify existing model vs creating new (avoids setup overhead)
- **Problem scaling**: Keep coefficients in [1e-6, 1e6] range

## Subproblem Construction Testing

### Purpose

The subproblem module (`src/subproblem.rs`, 919 lines) is one of the most complex components in POWE.RS. It constructs LP/MILP problems from system descriptions, manages state transitions, realizes uncertainties, and integrates cuts. Comprehensive testing ensures:

1. **Construction Correctness**: Variables and constraints correctly represent system physics
2. **State Transitions**: Hydro balance RHS updates preserve continuity
3. **Uncertainty Realization**: Solving with sampled noises produces valid solutions
4. **Solver Integration**: Problems are feasible, optimal, and numerically stable
5. **Edge Cases**: Single hydro, no thermals, tight bounds all work correctly

**Test Strategy**:

- **Basic Construction**: Validate variable/constraint counts for different system types
- **Constraint Structure**: Verify load balance, hydro balance, inflow process constraints
- **State Transitions**: Test hydro balance RHS updates via `update_with_current_trajectory()`
- **Uncertainty Realization**: Solve with sampled noises, extract solutions
- **Edge Cases**: Test boundary conditions (single hydro, no thermals, tight bounds)
- **Solver Integration**: Verify optimality, feasibility range, repeated solves
- **Validation**: Check structure consistency, dimension matching

### Test Categories

#### 1. Basic Construction Tests

Tests that validate subproblem variable and constraint counts for different system architectures.

**Minimal System** (`test_minimal_subproblem_construction`):

```rust
// System: 1 bus, 1 hydro, 2 thermals, 0 transmission lines
let subproblem = create_minimal_subproblem();
let vars = &subproblem.variables;

// Variables: deficit (1), thermal_gen (2), turbined_flow (1),
//            spillage (1), stored_volume (1), inflow (1),
//            inflow_process (2), alpha (1)
assert_eq!(vars.deficit.len(), 1);
assert_eq!(vars.thermal_generation.len(), 2);
assert_eq!(vars.turbined_flow.len(), 1);

// Constraints: load_balance (1), hydro_balance (1), inflow_process (2)
let cons = &subproblem.constraints;
assert_eq!(cons.load_balance.len(), 1);
assert_eq!(cons.hydro_balance.len(), 1);
assert_eq!(cons.inflow_process.len(), 2); // storage state
```

**Purpose**: Verify basic construction creates expected number of variables/constraints.

**Cascade System** (`test_cascade_subproblem_construction`):

```rust
// System: 2 buses, 2 hydros (cascade), 0 thermals, 1 line
let subproblem = create_cascade_subproblem();
let vars = &subproblem.variables;

// Variables: deficit (2 buses), turbined_flow (2 hydros),
//            direct/reverse exchange (1 line), inflow (2)
assert_eq!(vars.deficit.len(), 2);
assert_eq!(vars.turbined_flow.len(), 2);
assert_eq!(vars.direct_exchange.len(), 1);
assert_eq!(vars.reverse_exchange.len(), 1);

// Constraints: 2 load balance (2 buses), 2 hydro balance,
//              4 inflow process (2 hydros × 2 for storage state)
assert_eq!(cons.load_balance.len(), 2);
assert_eq!(cons.hydro_balance.len(), 2);
assert_eq!(cons.inflow_process.len(), 4);
```

**Purpose**: Validate cascade hydro system with upstream/downstream dependencies.

**Mixed System** (`test_mixed_subproblem_construction`):

```rust
// System: 1 bus, 1 hydro, 2 thermals, 0 lines
let subproblem = create_mixed_subproblem();

// Same as minimal (used to test different JSON construction path)
assert_eq!(vars.thermal_generation.len(), 2);
assert_eq!(vars.turbined_flow.len(), 1);
```

**Purpose**: Test system with both hydro and thermal generation.

#### 2. Constraint Generation Tests

Tests that validate constraint structure matches expected system physics.

**Load Balance Constraints** (`test_load_balance_constraint_structure`):

```rust
// Load balance: One constraint per bus
// thermal_gen + turbined_flow + exchange - deficit = load
let subproblem = create_minimal_subproblem();
assert_eq!(subproblem.constraints.load_balance.len(), 1);

// Cascade: 2 buses → 2 load balance constraints
let cascade = create_cascade_subproblem();
assert_eq!(cascade.constraints.load_balance.len(), 2);
```

**Purpose**: Verify load balance constraint count matches bus count.

**Hydro Balance Constraints** (`test_hydro_balance_constraint_structure`):

```rust
// Hydro balance: One constraint per hydro
// stored_volume[t] = stored_volume[t-1] + inflow - turbined - spillage
assert_eq!(subproblem.constraints.hydro_balance.len(), 1); // 1 hydro
assert_eq!(cascade.constraints.hydro_balance.len(), 2);    // 2 hydros
```

**Purpose**: Verify hydro balance constraint count matches hydro count.

**Inflow Process Constraints** (`test_inflow_process_constraints_structure`):

```rust
// Inflow process: 2 constraints per hydro (for storage state representation)
// These model the stochastic process for inflows
assert_eq!(subproblem.constraints.inflow_process.len(), 2); // 1 hydro × 2
assert_eq!(cascade.constraints.inflow_process.len(), 4);    // 2 hydros × 2
```

**Purpose**: Verify inflow process constraints for uncertainty representation.

#### 3. State Transition Tests

Tests that validate state updates between stages.

**Hydro Balance RHS Update** (`test_hydro_balance_rhs_update`):

```rust
let mut subproblem = create_minimal_subproblem();

// Update trajectory with new storage value
let trajectory = Trajectory {
    stage: 1,
    storage: vec![75.0], // New storage state
    inflow: vec![],
};

subproblem.update_with_current_trajectory(&trajectory);

// RHS of hydro balance constraint should be updated
// This sets the initial storage for the next stage
```

**Purpose**: Verify state transitions update hydro balance RHS correctly.

#### 4. Uncertainty Realization Tests

Tests that solve subproblems with sampled uncertainties.

**Simple Realization** (`test_realize_uncertainties_simple`):

```rust
let mut subproblem = create_minimal_subproblem();

// Set up sampled noises (1 bus load, 1 hydro inflow)
let mut noises = SampledBranchingNoises::new(1, 1);
noises.set_load_noises(&[0.0]); // Deterministic for testing
noises.set_inflow_noises(&[0.0]);

let mut realization = create_minimal_realization(Some(50.0));
let (load_sp, inflow_sp) = create_naive_stochastic_processes();

let result = subproblem.realize_uncertainties(
    &noises, load_sp.as_ref(), inflow_sp.as_ref(), &mut realization
);

assert!(result.is_ok());
assert!(realization.total_stage_objective.is_finite());
assert_eq!(realization.deficit.len(), 1);
```

**Purpose**: Verify subproblem solves with deterministic noise and extracts solution.

**Cascade Realization** (`test_realize_uncertainties_cascade`):

```rust
// Test with 2-hydro cascade system
let mut noises = SampledBranchingNoises::new(2, 2); // 2 buses, 2 hydros
noises.set_load_noises(&[0.0, 0.0]);
noises.set_inflow_noises(&[0.0, 0.0]);

// Solve and validate upstream/downstream interaction
assert_eq!(realization.turbined_flow.len(), 2);
assert_eq!(realization.spillage.len(), 2);
```

**Purpose**: Validate cascade hydro system with upstream/downstream flow.

**Deficit Realization** (`test_realize_uncertainties_with_deficit`):

```rust
// Test that deficit variables exist and can be positive
assert!(realization.deficit.len() > 0);
// Deficit should be non-negative (it's an unmet demand variable)
```

**Purpose**: Verify deficit variables handled correctly in solution extraction.

#### 5. Edge Case Tests

Tests for boundary conditions and unusual system configurations.

**Single Hydro** (`test_single_hydro_subproblem`):

```rust
// Minimal case: 1 hydro plant
let subproblem = create_minimal_subproblem();
assert_eq!(subproblem.variables.turbined_flow.len(), 1);
assert_eq!(subproblem.constraints.hydro_balance.len(), 1);
```

**Purpose**: Verify simplest hydro system works.

**No Thermal Plants** (`test_no_thermal_subproblem`):

```rust
// Cascade system has no thermals
let subproblem = create_cascade_subproblem();
assert_eq!(subproblem.variables.thermal_generation.len(), 0);
```

**Purpose**: Test system with only hydro generation.

**No Transmission Lines** (`test_no_transmission_subproblem`):

```rust
// Minimal system has no lines
assert_eq!(subproblem.variables.direct_exchange.len(), 0);
assert_eq!(subproblem.variables.reverse_exchange.len(), 0);
```

**Purpose**: Test system without transmission (single bus).

**Tight Storage Bounds** (`test_tight_storage_bounds`):

```rust
// Storage near upper bound (95.0 MW out of 100.0 MW max)
let realization = create_minimal_realization(Some(95.0));
// Should still solve feasibly
assert!(result.is_ok());
```

**Purpose**: Test near-binding constraints don't cause numerical issues.

#### 6. Solver Integration Tests

Tests that validate solver interaction.

**Solves to Optimality** (`test_subproblem_solves_to_optimality`):

```rust
let result = subproblem.realize_uncertainties(...);
assert!(result.is_ok());

// Objective should be finite and reasonable
let obj = realization.total_stage_objective;
assert!(obj.is_finite());
assert!(obj >= realization.current_stage_objective);
```

**Purpose**: Verify subproblem reaches optimal solution.

**Feasibility Range** (`test_subproblem_feasibility_range`):

```rust
// Test storage values: [0, 25, 50, 75, 100] MW
for &storage in &[0.0, 25.0, 50.0, 75.0, 100.0] {
    let realization = create_minimal_realization(Some(storage));
    let result = subproblem.realize_uncertainties(...);
    assert!(result.is_ok(), "Failed with storage = {}", storage);
}
```

**Purpose**: Verify feasibility across full storage range.

**Objective Consistency** (`test_objective_consistency`):

```rust
// Total objective >= current stage objective
// (total includes future cost via alpha)
assert!(realization.total_stage_objective >= realization.current_stage_objective);
```

**Purpose**: Validate objective decomposition (current + future).

**Repeated Solves** (`test_repeated_solves`):

```rust
// Solve same subproblem 10 times
for i in 0..10 {
    let result = subproblem.realize_uncertainties(...);
    assert!(result.is_ok());
}
// No memory leaks or performance degradation
```

**Purpose**: Check for memory leaks in repeated solving.

#### 7. Validation Tests

Tests that verify internal consistency.

**Structure Consistency** (`test_subproblem_structure_consistency`):

```rust
// Verify dimensions match system
let system = create_minimal_system();
let subproblem = create_minimal_subproblem();

assert_eq!(vars.deficit.len(), system.buses.len());
assert_eq!(vars.turbined_flow.len(), system.hydros.len());
assert_eq!(cons.load_balance.len(), system.buses.len());
assert_eq!(cons.hydro_balance.len(), system.hydros.len());
```

**Purpose**: Validate subproblem dimensions match system description.

**Realization Dimensions** (`test_realization_container_dimensions`):

```rust
// Verify realization vectors match system
assert_eq!(realization.loads.len(), system.buses.len());
assert_eq!(realization.deficit.len(), system.buses.len());
assert_eq!(realization.inflow.len(), system.hydros.len());
assert_eq!(realization.turbined_flow.len(), system.hydros.len());
```

**Purpose**: Validate solution extraction dimensions.

### Test Fixtures

The subproblem tests use comprehensive fixtures in `tests/fixtures/subproblems.rs`:

**System Creators**:

```rust
pub fn create_minimal_system() -> System;  // 1 bus, 1 hydro, 2 thermals
pub fn create_cascade_system() -> System;  // 2 buses, 2 hydros cascade, 1 line
pub fn create_mixed_system() -> System;    // 1 bus, 1 hydro, 2 thermals
```

**Subproblem Creators**:

```rust
pub fn create_minimal_subproblem() -> Subproblem;
pub fn create_cascade_subproblem() -> Subproblem;
pub fn create_mixed_subproblem() -> Subproblem;
```

**Realization Helpers**:

```rust
pub fn create_test_realization(system: &System, storage: Option<f64>) -> Realization;
pub fn create_minimal_realization(storage: Option<f64>) -> Realization;
pub fn create_cascade_realization(upstream: f64, downstream: f64) -> Realization;
```

**Stochastic Process Helper**:

```rust
pub fn create_naive_stochastic_processes() -> (Box<dyn StochasticProcess>, Box<dyn StochasticProcess>);
```

### Running Subproblem Tests

```bash
# Run all subproblem construction tests
cargo test --test test_subproblem_construction

# Run specific category
cargo test test_minimal_subproblem_construction
cargo test test_realize_uncertainties

# Run fixture tests
cargo test --test test_subproblem_construction fixtures

# Run with output (see construction details)
cargo test --test test_subproblem_construction -- --nocapture
```

### Interpreting Subproblem Test Failures

| Failure                           | Likely Cause                           | Next Steps                                            |
| --------------------------------- | -------------------------------------- | ----------------------------------------------------- |
| **Variable count mismatch**       | Construction logic error               | Check add_variables_to_subproblem()                   |
| **Constraint count mismatch**     | Constraint generation bug              | Check add_constraints_to_subproblem()                 |
| **Infeasible subproblem**         | Incorrect bounds or missing variables  | Check variable bounds, system JSON schema             |
| **Incorrect solution extraction** | Realization indexing error             | Verify solution extraction in realize_uncertainties() |
| **State transition failure**      | RHS update bug                         | Check update_with_current_trajectory()                |
| **Objective inconsistency**       | Alpha variable or cut integration bug  | Check FCF cut addition, alpha variable                |
| **Repeated solve failure**        | Memory leak or solver state corruption | Check solver cleanup, use valgrind                    |
| **Cascade flow mismatch**         | Upstream/downstream dependency bug     | Check hydro balance constraint for cascade            |

### API Discoveries

During subproblem test implementation, several API patterns were discovered:

**SampledBranchingNoises Construction**:

```rust
// CORRECT: Create with entity counts, then set values
let mut noises = SampledBranchingNoises::new(num_buses, num_hydros);
noises.set_load_noises(&[load_noise_1, load_noise_2]);
noises.set_inflow_noises(&[inflow_noise_1, inflow_noise_2]);

// INCORRECT: Trying to pass values to constructor
// let noises = SampledBranchingNoises::new(vec![...], vec![...]); // Doesn't compile
```

**StochasticProcess Trait Bounds**:

```rust
// CORRECT: Use .as_ref() to convert Box<dyn T> to &dyn T
let (load_sp, inflow_sp): (Box<dyn StochasticProcess>, Box<dyn StochasticProcess>) = ...;
subproblem.realize_uncertainties(&noises, load_sp.as_ref(), inflow_sp.as_ref(), ...);

// INCORRECT: Passing Box<dyn T> directly
// subproblem.realize_uncertainties(&noises, &load_sp, &inflow_sp, ...); // Trait bound error
```

**SystemInput JSON Schema**:

```rust
// CORRECT field names for HydroInput
{
    "id": 0,
    "downstream_hydro_id": null,
    "bus_id": 0,
    "productivity": 1.0,
    "min_storage": 0.0,
    "max_storage": 100.0,
    "min_turbined_flow": 0.0,
    "max_turbined_flow": 50.0,
    "spillage_penalty": 1e-3
}

// INCORRECT: Old field names (don't use these)
// "min_volume", "max_volume", "min_generation", "max_generation"
```

**JSON to System Conversion**:

```rust
// CORRECT: Two-step process
let system_input: SystemInput = serde_json::from_str(json_string)?;
let system: System = system_input.build_sddp_system();

// INCORRECT: No direct load_system() function
// let system = powers_rs::input::load_system(json_string)?; // Doesn't exist
```

### Adding New Subproblem Tests

When adding subproblem tests:

1. **Identify system configuration**: What buses, hydros, thermals, lines?
2. **Choose or create fixture**: Use existing or add new system creator
3. **Determine test category**: Construction, constraint, state, uncertainty, edge case, solver, validation
4. **Set deterministic inputs**: Use fixed storage, zero noises for reproducibility
5. **Validate specific property**: One assertion per test focus

**Template**:

```rust
#[test]
fn test_new_subproblem_property() {
    // Setup: Create system and subproblem
    let system = create_minimal_system(); // or create_cascade_system(), etc.
    let mut subproblem = create_minimal_subproblem();

    // Prepare inputs (deterministic for reproducibility)
    let mut noises = SampledBranchingNoises::new(1, 1);
    noises.set_load_noises(&[0.0]);
    noises.set_inflow_noises(&[0.0]);

    let mut realization = create_minimal_realization(Some(50.0));
    let (load_sp, inflow_sp) = create_naive_stochastic_processes();

    // Perform operation
    let result = subproblem.realize_uncertainties(
        &noises, load_sp.as_ref(), inflow_sp.as_ref(), &mut realization
    );

    // Validate property
    assert!(result.is_ok());
    assert!(realization.total_stage_objective.is_finite());
    // ... specific property checks ...
}
```

### Troubleshooting Subproblem Construction

**Problem**: Subproblem infeasible

- Check system JSON: Are bounds reasonable? (storage, flow, thermal capacity)
- Check load vs capacity: Is demand < total generation capacity?
- Check initial storage: Is it within [min_storage, max_storage]?
- Enable logging: `RUST_LOG=debug cargo test` to see solver output

**Problem**: Variable count doesn't match expected

- Check system description: Count buses, hydros, thermals, lines manually
- Verify JSON parsing: Print system after loading to confirm structure
- Check construction logic: Review add_variables_to_subproblem()

**Problem**: Solution extraction fails

- Check realization dimensions: Do they match system entity counts?
- Verify solver status: Is solution actually optimal?
- Check variable indexing: Are variables accessed in correct order?

**Problem**: State transition doesn't update RHS

- Verify trajectory: Is storage vector correct length?
- Check hydro balance RHS: Print before/after update_with_current_trajectory()
- Confirm stage index: Is trajectory.stage correct?

## Running Tests

### Basic Commands

```bash
# Run all tests
cargo test

# Run all tests with verbose output
cargo test --verbose

# Run specific test by name
cargo test test_benders_cut

# Run tests in a specific file
cargo test --test test_cut

# Run tests in a specific module
cargo test cut::tests

# Run tests matching a pattern
cargo test backward

# Show output from passing tests (println! statements)
cargo test -- --nocapture

# Show test execution time
cargo test -- --show-output
```

### Advanced Usage

```bash
# Run with all features enabled
cargo test --all-features

# Run with specific features
cargo test --features "feature_name"

# Run integration tests only
cargo test --test '*'

# Run unit tests only (in library)
cargo test --lib

# Run tests in parallel (default)
cargo test

# Run tests sequentially
cargo test -- --test-threads=1

# Run tests with release optimizations (faster, but longer compile)
cargo test --release
```

### Running Specific Test Types

```bash
# Run only unit tests in src/cut.rs
cargo test --lib cut::tests

# Run only integration test for 2-stage problem
cargo test --test integration_simple_2stage

# Run all integration tests
cargo test --tests

# Run documentation tests
cargo test --doc
```

### Development Workflow

Recommended workflow while developing:

```bash
# 1. Run tests frequently during development
cargo test

# 2. Before committing, run full test suite
cargo test --all-features

# 3. Check formatting and linting
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# 4. Run CI checks locally (exactly what CI runs)
cargo test --verbose --all-features
cargo clippy --all-targets --all-features -- -D warnings
```

## Debugging Test Failures

### Local Failures

#### Reading Test Output

Test failures show:

```
---- test_name stdout ----
thread 'test_name' panicked at 'assertion failed: `(left == right)`
  left: `42.0`,
 right: `42.000001`', src/module.rs:123:5
```

Key information:

- **Test name**: `test_name`
- **Assertion type**: `assertion failed: (left == right)`
- **Values**: Shows actual vs expected
- **Location**: `src/module.rs:123:5`

#### Using println! Debugging

Add debug output to understand test behavior:

```rust
#[test]
fn test_something() {
    let value = compute_value();
    println!("Computed value: {}", value);  // Won't show unless test fails
    assert_eq!(value, 42);
}

// Run with --nocapture to see output even on success
// cargo test test_something -- --nocapture
```

#### Running a Single Test

Focus on one failing test:

```bash
# Run just the failing test
cargo test test_benders_cut_evaluation -- --nocapture

# Run with backtrace for panic location
RUST_BACKTRACE=1 cargo test test_benders_cut_evaluation

# Run with full backtrace
RUST_BACKTRACE=full cargo test test_benders_cut_evaluation
```

#### Using the Debugger

For complex failures, use a debugger:

```bash
# Install rust-gdb or rust-lldb
rustup component add rust-src

# Debug a specific test
rust-gdb --args target/debug/deps/powers_rs-<hash> test_name --nocapture
```

### CI Failures

#### Accessing CI Logs

1. Go to GitHub Actions tab in repository
2. Click on failing workflow run
3. Click on failing job (e.g., "Test")
4. Expand the failing step to see full output

#### Reproducing CI Environment Locally

CI runs these commands:

```bash
# Exact CI commands
cargo test --verbose --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

Run locally to reproduce:

```bash
# Run in clean environment
cargo clean
cargo test --verbose --all-features

# Check formatting (--check doesn't modify files)
cargo fmt --all -- --check

# Check for clippy warnings (CI fails on ANY warning)
cargo clippy --all-targets --all-features -- -D warnings
```

#### Common CI Issues

**Issue**: Test passes locally but fails in CI

- **Cause**: Non-deterministic behavior (random seeds, timing)
- **Fix**: Use fixed seeds, avoid time-dependent tests

**Issue**: Clippy warnings in CI but not locally

- **Cause**: Different Rust versions or flags
- **Fix**: Run with same flags as CI: `-- -D warnings`

**Issue**: Formatting failures in CI

- **Cause**: Forgot to run `cargo fmt --all`
- **Fix**: Always run `cargo fmt --all` before committing

**Issue**: Test timeout in CI

- **Cause**: Test takes too long (>10 minutes default)
- **Fix**: Optimize test or increase timeout in workflow file

## Contributing Tests

### When to Add Tests

Add tests for:

1. **All new features**: Every new function, method, or module needs tests
2. **All bug fixes**: Reproduce the bug in a test, then fix it
3. **When coverage drops**: If a change reduces coverage, add tests
4. **Edge cases**: When you find an edge case, add a test for it
5. **Performance optimizations**: Benchmark to verify improvements

### Test Review Checklist

Before submitting a PR with tests:

- [ ] **Tests pass locally**: `cargo test --all-features`
- [ ] **Tests are documented**: Complex tests have doc comments
- [ ] **Tests are deterministic**: Fixed seeds for randomness
- [ ] **Coverage maintained**: New code is tested
- [ ] **Edge cases covered**: Boundary conditions tested
- [ ] **Formatting applied**: `cargo fmt --all` run
- [ ] **Clippy clean**: `cargo clippy --all-targets --all-features -- -D warnings` passes
- [ ] **Fast execution**: Unit tests run in milliseconds
- [ ] **Independent**: Tests don't depend on execution order
- [ ] **Clear assertions**: Failure messages are informative

### Writing Test-Worthy Code

Design code to be testable:

```rust
// ❌ HARD TO TEST: Depends on global state, hard-coded values
fn process_data() -> f64 {
    let data = read_from_disk("/path/to/file");
    let result = compute(data);
    write_to_disk("/path/to/output", result);
    result
}

// ✅ EASY TO TEST: Dependencies injected, pure logic
fn process_data(input: &[f64]) -> f64 {
    compute(input)
}

fn compute(data: &[f64]) -> f64 {
    data.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(compute(&data), 6.0);
    }
}
```

**Testability Guidelines**:

- Avoid global state
- Inject dependencies (don't hard-code)
- Separate I/O from logic
- Use pure functions when possible
- Keep functions focused (single responsibility)

## Examples

### Example 1: Good Unit Test

```rust
/// Tests that a Benders cut correctly evaluates its value at a given state.
///
/// Given a cut with coefficients [1.5, -0.5] and RHS 42.0, evaluating at
/// state [100.0, 200.0] should produce:
///   1.5 * 100.0 + (-0.5) * 200.0 + 42.0 = 150.0 - 100.0 + 42.0 = 92.0
#[test]
fn test_benders_cut_evaluates_correctly() {
    // Arrange: Create a cut with known coefficients
    let coefficients = vec![1.5, -0.5];
    let rhs = 42.0;
    let cut = BendersCut::new(0, coefficients, rhs);

    // Arrange: Create a state to evaluate at
    let state = vec![100.0, 200.0];

    // Act: Evaluate the cut
    let value = cut.evaluate(&state);

    // Assert: Verify the computation
    let expected = 1.5 * 100.0 + (-0.5) * 200.0 + 42.0;
    assert_float_approx_eq!(value, expected, 1e-10);
}
```

**Why this is good**:

- Clear doc comment explains what's being tested
- AAA structure (Arrange-Act-Assert)
- Descriptive variable names
- Manual calculation in assertion makes expected value clear
- Appropriate floating-point tolerance

### Example 2: Integration Test

```rust
/// Full integration test of 2-stage SDDP with simple reservoir system.
///
/// This test verifies:
/// - Forward pass solves subproblems and computes policy value
/// - Backward pass generates Benders cuts
/// - Training iterations converge within bounds
/// - Final policy is feasible and near-optimal
#[test]
fn test_sddp_two_stage_training() {
    // Arrange: Set up 2-stage system with deterministic scenarios
    let system = create_simple_2stage_system();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa();

    // Create SDDP handler
    let mut handler = SddpTrainHandler::new(
        &system,
        &initial_condition,
        &saa,
        5,  // max_iterations
        "storage",
    ).expect("Failed to create SDDP handler");

    // Act: Run training
    let result = handler.train();

    // Assert: Training succeeded
    assert!(result.is_ok(), "Training failed: {:?}", result.err());

    // Assert: Converged within reasonable bounds
    let final_bound = handler.get_lower_bound();
    assert!(final_bound > 0.0, "Lower bound should be positive");
    assert!(final_bound < 1000.0, "Lower bound should be reasonable");

    // Assert: Forward pass produces feasible solution
    let forward_result = handler.forward(&saa);
    assert!(forward_result.is_ok(), "Forward pass failed: {:?}", forward_result.err());
}
```

**Why this is good**:

- Tests complete workflow (integration test)
- Uses realistic test fixtures
- Verifies multiple aspects (convergence, feasibility, bounds)
- Clear doc comment explains what's tested
- Helpful error messages in assertions

### Example 3: Testing Numerical Stability

```rust
/// Tests that storage state handles near-zero storage without numerical issues.
///
/// When storage is very small (e.g., 1e-10), water values should remain finite
/// and the optimization should not produce NaN or Inf values.
#[test]
fn test_storage_state_handles_near_zero_storage() {
    // Arrange: Create state with near-zero initial storage
    let system = System::default();
    let load_sp = stochastic_process::factory("naive");
    let inflow_sp = stochastic_process::factory("naive");
    let mut state = StorageState::new(&system, load_sp.as_ref(), inflow_sp.as_ref());

    // Set storage to very small positive value
    state.set_coefficients(&[1e-10]);

    // Act: Update state (this could cause numerical issues)
    let realization = create_test_realization();
    state.update(&realization);

    // Assert: Values remain finite
    let coeffs = state.coefficients();
    assert!(!coeffs[0].is_nan(), "Coefficient became NaN");
    assert!(!coeffs[0].is_infinite(), "Coefficient became infinite");
    assert!(coeffs[0] >= 0.0, "Storage coefficient negative");
}
```

**Why this is good**:

- Tests edge case (near-zero values)
- Explicitly checks for NaN and Inf
- Tests a realistic numerical concern
- Clear explanation of what numerical issue is being prevented

### Example 4: What to Avoid

```rust
// ❌ BAD EXAMPLE: Multiple problems

#[test]
fn test_stuff() {  // Unclear name
    let c = BendersCut::new(vec![1.0], vec![2.0], 3.0).unwrap();
    assert_eq!(c.evaluate(&vec![1.0]), 5.0);  // Magic numbers

    let s = StorageState::new(&System::default(), &sp, &sp2);  // What are sp, sp2?
    s.update(&r);  // What is r?
    assert!(s.coefficients()[0] > 0.0);  // What does this verify?

    // Testing too many things in one test
    let x = random();  // Non-deterministic!
    assert!(x > 0.5);  // Will fail 50% of the time
}
```

**Problems**:

- Unclear test name
- No comments explaining what's tested
- Magic numbers without context
- Undocumented variables
- Tests multiple behaviors in one test
- Non-deterministic (random without fixed seed)
- Unclear what success means

```rust
// ✅ GOOD EXAMPLE: Fixed version

/// Tests that BendersCut correctly evaluates linear combination of state coefficients.
#[test]
fn test_benders_cut_evaluates_linear_combination() {
    // Arrange: Create cut: 2.0 * x + 3.0 (coefficient=2.0, rhs=3.0)
    let coefficients = vec![2.0];
    let rhs = 3.0;
    let cut = BendersCut::new(0, coefficients, rhs);

    // Act: Evaluate at x = 1.0
    let state = vec![1.0];
    let value = cut.evaluate(&state);

    // Assert: Should be 2.0 * 1.0 + 3.0 = 5.0
    assert_float_approx_eq!(value, 5.0, 1e-10);
}

/// Tests that storage state remains non-negative after updates.
#[test]
fn test_storage_state_stays_non_negative() {
    // Arrange
    let system = System::default();
    let load_sp = stochastic_process::factory("naive");
    let inflow_sp = stochastic_process::factory("naive");
    let mut state = StorageState::new(&system, load_sp.as_ref(), inflow_sp.as_ref());

    // Act: Update with a realization
    let realization = create_test_realization();
    state.update(&realization);

    // Assert: Storage coefficient should be non-negative
    assert!(
        state.coefficients()[0] >= 0.0,
        "Storage coefficient should never be negative"
    );
}
```

## Code Coverage

### What is Code Coverage?

Code coverage measures which lines of code are executed during testing. It helps identify:

- **Untested code paths** that may contain bugs
- **Dead code** that's never executed
- **Testing progress** over time
- **Areas needing more tests**

**Important**: Coverage is a **guide, not a goal**. High coverage doesn't guarantee correctness, and 100% coverage isn't always practical or valuable. Focus on testing critical paths and edge cases.

### Running Coverage Locally

#### Install cargo-tarpaulin

```bash
cargo install cargo-tarpaulin
```

#### Generate Coverage Reports

**HTML Report** (for local viewing):

```bash
cargo tarpaulin --out Html --output-dir coverage --all-features
```

Open the report in your browser:

```bash
# Linux
xdg-open coverage/index.html

# macOS
open coverage/index.html

# Windows
start coverage/index.html
```

**Terminal Output** (quick check):

```bash
cargo tarpaulin --all-features
```

**Both HTML and Lcov** (for CI):

```bash
cargo tarpaulin --out Html --out Lcov --output-dir coverage --all-features
```

### Interpreting Coverage Results

#### Understanding the Numbers

Coverage reports show:

- **Line coverage**: Percentage of lines executed
- **Per-module coverage**: Coverage breakdown by file
- **Untested lines**: Highlighted in red in HTML reports

Example output:

```
|| Tested/Total Lines:
|| src/cut.rs: 8/8       (100%)  ✅ Excellent
|| src/fcf.rs: 33/57     (58%)   ⚠️  Needs improvement
|| src/solver.rs: 213/294 (72%)  ✓ Good
||
|| 69.93% coverage, 1177/1683 lines covered
```

#### What Good Coverage Looks Like

**High-priority modules** (core algorithm):

- ✅ `cut.rs`: 100% - All cut logic tested
- ✅ `state.rs`: 95%+ - Critical data structure
- ✅ `utils.rs`: 100% - Utility functions fully tested

**Medium-priority modules**:

- ✓ `sddp.rs`: 87% - Main algorithm covered
- ✓ `solver.rs`: 72% - Most solver interaction tested
- ✓ `scenario.rs`: 88% - Scenario generation well-tested

**Lower-priority modules**:

- → `input.rs`: 38% - I/O code, harder to test
- → `output.rs`: 0% - Output formatting, low risk
- → `main.rs`: 0% - Entry point, tested via integration

### Coverage Targets

We aim for the following coverage levels:

| Module                  | Target   | Priority | Rationale                            |
| ----------------------- | -------- | -------- | ------------------------------------ |
| `cut.rs`                | >95%     | Critical | Fundamental to algorithm correctness |
| `state.rs`              | >95%     | Critical | Core data structure                  |
| `fcf.rs`                | >90%     | Critical | Future cost approximation            |
| `sddp.rs`               | >85%     | Critical | Main algorithm                       |
| `subproblem.rs`         | >80%     | High     | Solver interaction                   |
| `solver.rs`             | >75%     | High     | Interface to HiGHS                   |
| `scenario.rs`           | >80%     | High     | Stochastic sampling                  |
| `graph.rs`              | >85%     | High     | Problem structure                    |
| `system.rs`             | >70%     | Medium   | Data structures                      |
| `risk_measure.rs`       | >80%     | Medium   | Risk-neutral/averse measures         |
| `stochastic_process.rs` | >80%     | Medium   | Noise generation                     |
| `initial_condition.rs`  | >90%     | Medium   | Initial state setup                  |
| `input.rs`              | >60%     | Lower    | I/O, integration tested              |
| `output.rs`             | >60%     | Lower    | Output formatting                    |
| `log.rs`                | >60%     | Lower    | Logging utilities                    |
| `utils.rs`              | >95%     | High     | Shared utilities                     |
| `main.rs`               | >50%     | Lower    | Entry point                          |
| **Overall Project**     | **>75%** | -        | Project-wide minimum                 |

### Coverage in CI

Coverage is automatically measured in CI on every push and pull request. The coverage job runs in parallel with the test suite.

**CI Workflow**:

1. Builds the project with coverage instrumentation
2. Runs all tests with `cargo-tarpaulin`
3. Generates Lcov format report
4. Uploads coverage to Codecov (if configured)
5. Adds coverage summary to PR

**View Coverage Reports**:

- Check the "Coverage" job in GitHub Actions
- View detailed reports on Codecov (if integrated)
- Coverage badge shows current coverage percentage

### What Coverage Doesn't Tell You

Coverage measures **execution**, not **quality**:

❌ **100% coverage doesn't mean**:

- Code is correct (can still have logic bugs)
- All edge cases are tested
- Tests are meaningful
- Assertions are comprehensive

✅ **Good tests with good coverage**:

- Test critical logic paths
- Verify edge cases and error conditions
- Use meaningful assertions
- Are readable and maintainable

### Improving Coverage

#### 1. Identify Untested Code

Look at the HTML report to find untested lines:

```bash
cargo tarpaulin --out Html --output-dir coverage --all-features
xdg-open coverage/index.html
```

Red-highlighted lines are not executed by tests.

#### 2. Prioritize by Risk

Focus on:

1. **Critical algorithm code** (SDDP, cut generation, solver interaction)
2. **Complex logic** (multiple branches, loops, error handling)
3. **Recent changes** (new features, bug fixes)
4. **Bug-prone areas** (has caused issues before)

#### 3. Write Tests for Gaps

For untested code:

- Add unit tests for individual functions
- Add integration tests for workflows
- Test error cases and edge conditions
- Verify numerical stability

#### 4. Exclude Unreachable Code

Some code is intentionally untested:

- Debug-only code
- Panic handlers
- Error variants not yet used

Use `#[cfg(not(tarpaulin_include))]` to exclude:

```rust
#[cfg(not(tarpaulin_include))]
fn debug_only_function() {
    // This won't be included in coverage
}
```

Or add to `tarpaulin.toml`:

```toml
[configuration]
exclude-files = [
    "tests/*",
    "src/deprecated.rs",
]
```

### Coverage Best Practices

#### DO:

✅ Use coverage to find untested code
✅ Focus on critical paths first
✅ Write meaningful tests, not coverage tests
✅ Track coverage trends over time
✅ Set realistic targets per module
✅ Review coverage reports before merging PRs

#### DON'T:

❌ Write tests just to increase coverage
❌ Aim for 100% coverage everywhere
❌ Ignore test quality for coverage numbers
❌ Test private implementation details excessively
❌ Cover trivial getters/setters obsessively

### Troubleshooting Coverage

**Issue**: Coverage is lower than expected

- **Check**: Are all feature flags enabled? Use `--all-features`
- **Check**: Are tests actually running? Look for test output
- **Check**: Is code excluded in `tarpaulin.toml`?

**Issue**: Coverage is too slow

- **Solution**: Run coverage less frequently (not on every commit)
- **Solution**: Use `--timeout 300` to prevent hangs
- **Solution**: Run coverage only in CI, not locally

**Issue**: Coverage misses some code

- **Note**: Tarpaulin has limitations with some code patterns
- **Solution**: Verify coverage with manual testing
- **Solution**: Use multiple coverage tools if needed

**Issue**: Coverage reports show test code

- **Fix**: Ensure `tests/*` is in `exclude-files` in `tarpaulin.toml`
- **Fix**: Test files should be under `tests/` directory, not `src/`

## Resources

### Rust Testing Documentation

- [Testing in The Rust Book](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Rust by Example: Testing](https://doc.rust-lang.org/rust-by-example/testing.html)
- [API Guidelines: Documentation](https://rust-lang.github.io/api-guidelines/documentation.html)

### Tools

- **cargo test**: Built-in test runner
- **cargo-nextest**: Faster test runner (optional): `cargo install cargo-nextest`
- **cargo-watch**: Auto-run tests on file changes: `cargo install cargo-watch`
- **cargo-tarpaulin**: Code coverage: `cargo install cargo-tarpaulin`

### Project-Specific

- See `.github/workflows/test.yml` for CI test configuration
- See `tests/fixtures/` for available test utilities
- See existing tests in `tests/` for examples

## Getting Help

If you're unsure about:

- **What to test**: Look at similar existing tests
- **How to structure a test**: Follow the examples in this guide
- **Floating-point tolerance**: Start with 1e-10, adjust based on numerical context
- **Test fixtures**: Check `tests/fixtures/mod.rs` for available utilities

When in doubt, write the test and ask for review—getting feedback is part of the learning process!

---

**Remember**: Good tests are **clear**, **fast**, **deterministic**, and **focused**. They make refactoring safe and serve as living documentation of how the code should behave.
