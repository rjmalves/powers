# [TICKET-008] Add Comprehensive Integration Tests

**Sprint:** 3-4  
**Estimated Effort:** 5 story points (3 days)  
**Confidence:** High  
**Priority:** P1 - High

## Context

With the architectural refactoring complete, we need comprehensive integration tests to verify:
1. The critical bug is actually fixed
2. Explicit structures work correctly in full SDDP runs
3. Edge cases with mixed AR orders are handled properly
4. Numerical results are correct and valid

These tests serve as both verification of the fix and regression prevention for the future.

## Acceptance Criteria

- [ ] Given Example 07 system (the failing case), when running SDDP with fixed seed, then lower_bound ≤ simulation_mean in all iterations
- [ ] Given system where loads have AR(1) and inflows have AR(1), when running SDDP, then cuts use correct variables (load lags not confused with inflow lags)
- [ ] Given various AR order combinations, when running complete SDDP, then all converge correctly
- [ ] Given 100 random seeds, when running Example 07, then 100% produce valid bounds (LB ≤ UB)
- [ ] Performance: Full SDDP runs should not regress by more than 5%

## Tasks

### Implementation

- [ ] Create test module `tests/test_explicit_lag_separation.rs`

- [ ] Test: Example 07 regression test (the original failing case)
  - Load Example 07 configuration
  - Run SDDP for 50 iterations with seed 42
  - Assert lower_bound ≤ simulation_mean at each iteration
  - Compare final bound to reference value (should be closer to true optimum)
  
- [ ] Test: Mixed AR orders - loads and inflows both AR(1)
  - Create system: 2 buses with AR(1) loads, 2 hydros with AR(1) inflows
  - Run SDDP for 30 iterations
  - Verify cuts are correctly generated
  - Check that load lag variables ≠ inflow lag variables in subproblem
  - Verify valid bounds throughout
  
- [ ] Test: Asymmetric AR orders
  - System: Load AR(2), Inflow AR(1), Load AR(0), Inflow AR(3)
  - Run full SDDP
  - Verify correct number of lag variables created
  - Verify cuts have correct dimensions
  - Check convergence
  
- [ ] Test: Large system with mixed AR orders
  - 20 buses (AR orders: 0, 1, 2 distributed randomly)
  - 15 hydros (AR orders: 0, 1, 2 distributed randomly)
  - Run 100 iterations
  - Verify no panics or errors
  - Check performance metrics
  
- [ ] Test: State transition correctness
  - Multi-stage problem (5 stages)
  - Track lag values across stage transitions
  - Verify load lags and inflow lags maintain continuity
  - Verify lag values match expected from AR process
  
- [ ] Test: Cut coefficient verification
  - Create simple 2-stage system with known optimal solution
  - Run SDDP to generate cuts
  - Extract cut coefficients manually
  - Verify coefficients match theoretical values
  - Verify coefficients are applied to correct variables
  
- [ ] Test: Parallel execution stability
  - System with 10 scenarios per iteration
  - Run with Rayon parallelism enabled
  - Verify results are deterministic with fixed seed
  - Check for race conditions or data races

- [ ] Test: Statistical validation across seeds
  - Run Example 07 with 100 different seeds
  - Collect convergence statistics
  - Verify 100% produce valid bounds
  - Compare bound quality distribution

### Testing

All tests are integration tests, so the implementation IS the testing.

Additional validation:
- [ ] Run all new tests under Miri (Rust's memory safety checker)
- [ ] Run with address sanitizer if available
- [ ] Run with thread sanitizer for parallel tests
- [ ] Profile memory usage for large system test
- [ ] Benchmark execution time for performance regression check

### Documentation

- [ ] Add README in `tests/` explaining integration test structure
- [ ] Document expected behavior for each test case
- [ ] Add comments explaining what each test validates
- [ ] Document reference values and how they were obtained
- [ ] Create test fixtures for common system configurations

## Technical Notes

### Test Structure

```rust
// tests/test_explicit_lag_separation.rs

use powers::*;
use approx::assert_relative_eq;

#[test]
fn test_example_07_valid_bounds() {
    // Load Example 07 system
    let config = load_example_config("examples/07-ar-inflows");
    let system = System::from_config(&config);
    
    // Run SDDP with fixed seed
    let mut sddp = SDDP::new(system, config.sddp_params);
    sddp.set_seed(42);
    
    for iteration in 0..50 {
        sddp.run_iteration();
        
        let lower_bound = sddp.get_lower_bound();
        let simulation = sddp.get_simulation_statistics();
        
        // CRITICAL: Lower bound must be valid
        assert!(
            lower_bound <= simulation.mean,
            "Iteration {}: Invalid bound! LB={:.2} > UB={:.2}",
            iteration, lower_bound, simulation.mean
        );
    }
    
    // Verify convergence quality
    let final_gap = sddp.get_optimality_gap();
    assert!(final_gap < 0.05, "Failed to converge to 5% gap");
}

#[test]
fn test_mixed_ar_orders_no_confusion() {
    // System specifically designed to trigger the old bug:
    // Load at Bus 0 with AR(1) and Inflow at Hydro 0 with AR(1)
    // Old heuristic would confuse these!
    
    let system = SystemBuilder::new()
        .add_bus(0)
        .add_hydro(0, /* capacity */ 100.0)
        .add_load_temporal_model(0, ar_order: 1, /* mean */ 50.0)
        .add_inflow_temporal_model(0, ar_order: 1, /* mean */ 80.0)
        .build();
    
    let subproblem = create_test_subproblem(&system);
    
    // Verify separation
    let load_lags = subproblem.variables.load_lags.as_ref().unwrap();
    let inflow_lags = subproblem.variables.inflow_lags.as_ref().unwrap();
    
    let load_lag_var = load_lags.get_lag_var(0, 0);
    let inflow_lag_var = inflow_lags.get_lag_var(0, 0);
    
    // These MUST be different variables!
    assert_ne!(
        load_lag_var, inflow_lag_var,
        "Load and inflow lag variables must not be confused"
    );
    
    // Run SDDP and verify valid bounds
    let mut sddp = SDDP::new(system, default_params());
    sddp.set_seed(123);
    
    for _ in 0..30 {
        sddp.run_iteration();
        assert!(sddp.get_lower_bound() <= sddp.get_simulation_statistics().mean);
    }
}

#[test]
fn test_cut_coefficient_correctness() {
    // Create simple 2-stage, 1-hydro system with AR(1) inflow
    let system = create_simple_ar1_system();
    
    let mut sddp = SDDP::new(system, default_params());
    sddp.set_seed(42);
    sddp.run_iteration(); // Generate first cuts
    
    // Get cuts from stage 1
    let cuts = sddp.get_cuts_at_stage(1);
    assert!(!cuts.is_empty(), "Should have generated cuts");
    
    let cut = &cuts[0];
    
    // Cut should have: 1 storage coefficient + 1 inflow lag coefficient
    assert_eq!(
        cut.coefficients.len(), 2,
        "Cut should have 2 coefficients (storage + inflow lag)"
    );
    
    // Verify coefficients are reasonable (not NaN, not zero)
    for (i, &coef) in cut.coefficients.iter().enumerate() {
        assert!(
            coef.is_finite(),
            "Coefficient {} is not finite: {}", i, coef
        );
        // In SDDP, storage/inflow dual should be negative (marginal value)
        // Actually sign depends on problem, so just check finite
    }
    
    // Apply cut to stage 0 subproblem and verify correct variables used
    let subproblem = sddp.get_subproblem_at_stage(0);
    let inflow_lags = subproblem.variables.inflow_lags.as_ref().unwrap();
    let storage_vars = &subproblem.variables.stored_volume;
    
    // Storage coefficient should apply to storage variable
    // Inflow lag coefficient should apply to inflow lag variable
    // (Implementation detail: verify in add_cut_constraint_to_model)
}

#[test]
fn test_statistical_validation() {
    // Run Example 07 with many seeds to verify robustness
    let config = load_example_config("examples/07-ar-inflows");
    
    let mut invalid_count = 0;
    let mut bound_qualities = Vec::new();
    
    for seed in 0..100 {
        let system = System::from_config(&config);
        let mut sddp = SDDP::new(system, config.sddp_params);
        sddp.set_seed(seed);
        
        // Run to convergence
        for _ in 0..50 {
            sddp.run_iteration();
        }
        
        let lb = sddp.get_lower_bound();
        let ub = sddp.get_simulation_statistics().mean;
        
        if lb > ub {
            invalid_count += 1;
            eprintln!("Seed {} produced invalid bound: LB={}, UB={}", seed, lb, ub);
        }
        
        let gap = (ub - lb) / ub.abs();
        bound_qualities.push(gap);
    }
    
    // CRITICAL: 100% should have valid bounds
    assert_eq!(
        invalid_count, 0,
        "{} out of 100 seeds produced invalid bounds", invalid_count
    );
    
    // Statistical check: median gap should be reasonable
    bound_qualities.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_gap = bound_qualities[50];
    
    assert!(
        median_gap < 0.10,
        "Median gap too large: {:.2}%", median_gap * 100.0
    );
}

#[test]
fn test_large_system_performance() {
    // Large system stress test
    let system = create_large_mixed_system(
        n_buses: 20,
        n_hydros: 15,
        max_ar_order: 2,
    );
    
    let start = std::time::Instant::now();
    
    let mut sddp = SDDP::new(system, default_params());
    sddp.set_seed(42);
    
    for _ in 0..100 {
        sddp.run_iteration();
    }
    
    let elapsed = start.elapsed();
    
    // Should complete in reasonable time (< 5 minutes for 100 iterations)
    assert!(
        elapsed.as_secs() < 300,
        "Large system test too slow: {:?}", elapsed
    );
    
    // Verify valid results
    assert!(sddp.get_lower_bound() <= sddp.get_simulation_statistics().mean);
}
```

### Test Fixtures

Create reusable system builders:

```rust
// tests/fixtures/systems.rs

pub fn create_simple_ar1_system() -> System {
    SystemBuilder::new()
        .add_bus(0)
        .add_hydro(0, capacity: 100.0)
        .add_load_temporal_model(0, ar_order: 0, mean: 50.0)
        .add_inflow_temporal_model(0, ar_order: 1, mean: 80.0, phi: 0.8)
        .build()
}

pub fn create_mixed_ar_system() -> System {
    SystemBuilder::new()
        .add_bus(0).add_bus(1).add_bus(2)
        .add_hydro(0).add_hydro(1)
        .add_load_temporal_model(0, ar_order: 1, mean: 30.0, phi: 0.6)
        .add_load_temporal_model(1, ar_order: 0, mean: 20.0)
        .add_load_temporal_model(2, ar_order: 2, mean: 40.0, phi: [0.5, 0.3])
        .add_inflow_temporal_model(0, ar_order: 1, mean: 60.0, phi: 0.7)
        .add_inflow_temporal_model(1, ar_order: 2, mean: 50.0, phi: [0.6, 0.2])
        .build()
}

pub fn create_large_mixed_system(n_buses: usize, n_hydros: usize, max_ar_order: usize) -> System {
    use rand::{SeedableRng, Rng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    
    let mut builder = SystemBuilder::new();
    
    for bus_id in 0..n_buses {
        builder = builder.add_bus(bus_id);
        let ar_order = rng.gen_range(0..=max_ar_order);
        builder = builder.add_load_temporal_model(
            bus_id, 
            ar_order, 
            mean: rng.gen_range(20.0..100.0)
        );
    }
    
    for hydro_id in 0..n_hydros {
        builder = builder.add_hydro(hydro_id, capacity: rng.gen_range(50.0..200.0));
        let ar_order = rng.gen_range(0..=max_ar_order);
        builder = builder.add_inflow_temporal_model(
            hydro_id,
            ar_order,
            mean: rng.gen_range(30.0..150.0)
        );
    }
    
    builder.build()
}
```

### Benchmark Reference Values

Document expected performance:

```
System: Example 07 (from original bug report)
Hardware: 4-core Intel i7 @ 2.8GHz
Iterations: 50
Expected time: ~10-15 seconds
Expected final gap: < 5%
Expected lower bound: ~X (to be determined)
Expected upper bound: ~Y (to be determined)
```

### Edge Cases to Test

- [ ] System with no AR models (all AR(0))
- [ ] System with only loads having AR models
- [ ] System with only inflows having AR models
- [ ] System with very high AR orders (AR(10))
- [ ] Single-stage problem (no cuts generated)
- [ ] Deterministic system (zero variance)

## Dependencies

- Blocked by: TICKET-001 through TICKET-007 (all implementation complete)
- Blocks: None
- Related: All tickets (validates their correctness)

## Definition of Done

- [ ] All integration tests implemented and passing
- [ ] Example 07 produces valid bounds 100% of time
- [ ] No performance regression (< 5%)
- [ ] Test coverage includes all edge cases
- [ ] Tests are deterministic (fixed seeds)
- [ ] Documentation explains what each test validates
- [ ] CI runs all integration tests
