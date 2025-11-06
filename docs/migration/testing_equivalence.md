# Testing Equivalence Strategy

**Status**: TESTING GUIDE  
**Date**: 2025-11-06  
**Ticket**: TICKET-001, TICKET-002

## Overview

This document outlines the comprehensive testing strategy to ensure the migration produces numerically identical results to the baseline implementation.

## Testing Philosophy

### Core Principles

1. **Numerical Equivalence**: Results must be identical within floating-point tolerance (< 1e-10)
2. **Comprehensive Coverage**: Test all code paths, edge cases, and system configurations
3. **Regression Prevention**: Capture current behavior before changes
4. **Performance Validation**: Ensure no degradation in execution speed
5. **Gradual Verification**: Test at each migration step, not just at the end

### Testing Pyramid

```
           /\
          /  \
         / E2E \           End-to-End: Full SDDP simulations
        /______\
       /        \
      /  Integ.  \        Integration: Multi-module interactions
     /____________\
    /              \
   /   Unit Tests   \     Unit: Individual functions and structures
  /__________________\
```

## Test Categories

### 1. Unit Tests

**Purpose**: Test individual functions and data structures in isolation

#### Data Structure Tests

```rust
#[cfg(test)]
mod load_lag_data_tests {
    use super::*;
    
    #[test]
    fn test_construction() {
        let data = LoadLagData::new(3, 5);
        assert_eq!(data.n_buses, 3);
        assert_eq!(data.max_lag, 5);
        assert_eq!(data.buffer.len(), 3);
        assert_eq!(data.variables.lags_by_bus.len(), 3);
        assert_eq!(data.constraints.constraints_by_bus.len(), 3);
    }
    
    #[test]
    fn test_get_set_lag() {
        let mut data = LoadLagData::new(2, 3);
        data.buffer[0] = vec![0.0; 3];
        
        data.set_lag(0, 0, 42.0);
        data.set_lag(0, 1, 100.0);
        data.set_lag(0, 2, 200.0);
        
        assert_eq!(data.get_lag(0, 0), 42.0);
        assert_eq!(data.get_lag(0, 1), 100.0);
        assert_eq!(data.get_lag(0, 2), 200.0);
    }
    
    #[test]
    fn test_bounds_checking() {
        let mut data = LoadLagData::new(1, 2);
        data.buffer[0] = vec![0.0; 2];
        
        // Valid access
        assert_eq!(data.get_lag(0, 0), 0.0);
        
        // Invalid access should panic
        // (Rust will panic on out-of-bounds access)
    }
    
    #[test]
    #[should_panic]
    fn test_buffer_out_of_bounds_panics() {
        let data = LoadLagData::new(1, 1);
        let _ = data.get_lag(0, 5);  // Should panic
    }
}
```

#### Buffer Update Tests

```rust
#[test]
fn test_load_lag_buffer_update_from_trajectory() {
    // Setup: Create trajectory with known load values
    let trajectory = create_test_trajectory_with_loads(&[
        vec![10.0, 20.0],  // t-3
        vec![15.0, 25.0],  // t-2
        vec![20.0, 30.0],  // t-1
        vec![25.0, 35.0],  // t (current)
    ]);
    
    // Create LoadLagData for 2 buses with lag order 3
    let mut load_data = LoadLagData::new(2, 3);
    load_data.buffer[0] = vec![0.0; 3];
    load_data.buffer[1] = vec![0.0; 3];
    
    // Update from trajectory
    load_data.update_from_trajectory(&trajectory);
    
    // Verify lag values (most recent first)
    assert_eq!(load_data.get_lag(0, 0), 20.0);  // t-1 for bus 0
    assert_eq!(load_data.get_lag(0, 1), 15.0);  // t-2 for bus 0
    assert_eq!(load_data.get_lag(0, 2), 10.0);  // t-3 for bus 0
    
    assert_eq!(load_data.get_lag(1, 0), 30.0);  // t-1 for bus 1
    assert_eq!(load_data.get_lag(1, 1), 25.0);  // t-2 for bus 1
    assert_eq!(load_data.get_lag(1, 2), 20.0);  // t-3 for bus 1
}

#[test]
fn test_inflow_lag_buffer_update_from_trajectory() {
    // Similar test for inflow lags
    let trajectory = create_test_trajectory_with_inflows(&[
        vec![100.0, 200.0],  // t-2
        vec![150.0, 250.0],  // t-1
        vec![200.0, 300.0],  // t (current)
    ]);
    
    let mut inflow_data = InflowLagData::new(2, 2);
    inflow_data.buffer[0] = vec![0.0; 2];
    inflow_data.buffer[1] = vec![0.0; 2];
    
    inflow_data.update_from_trajectory(&trajectory);
    
    assert_eq!(inflow_data.get_lag(0, 0), 150.0);  // t-1 for hydro 0
    assert_eq!(inflow_data.get_lag(0, 1), 100.0);  // t-2 for hydro 0
    assert_eq!(inflow_data.get_lag(1, 0), 250.0);  // t-1 for hydro 1
    assert_eq!(inflow_data.get_lag(1, 1), 200.0);  // t-2 for hydro 1
}
```

### 2. Integration Tests

**Purpose**: Test interactions between modules and data flow

#### Constraint Update Integration

```rust
#[test]
fn test_lag_constraint_update_integration() {
    // Given: A subproblem with load and inflow lag constraints
    let mut subproblem = create_test_subproblem_with_lags();
    
    // And: A trajectory with known lag values
    let trajectory = create_test_trajectory();
    
    // When: prepare_from_trajectory is called
    subproblem.prepare_from_trajectory(&trajectory);
    
    // Then: Lag buffers should be populated
    assert!(subproblem.load_lag_data.is_some());
    let load_data = subproblem.load_lag_data.as_ref().unwrap();
    assert!(!load_data.buffer[0].is_empty());
    
    // And: Constraint RHS should be updated
    // (Verify by checking model bounds via solver interface)
    let model = subproblem.model.as_ref().unwrap();
    let constraint_idx = load_data.constraints.get_constraint(0, 0);
    let bounds = model.get_row_bounds(constraint_idx);
    assert!((bounds.0 - load_data.get_lag(0, 0)).abs() < 1e-10);
}
```

#### Uncertainty Observation Constraint Integration

```rust
#[test]
fn test_uncertainty_observation_constraint_update() {
    // Given: A subproblem with uncertainty observation data
    let mut subproblem = create_test_subproblem();
    
    // And: Known innovations
    let innovations = vec![0.5, -0.3, 1.0];  // Standardized innovations
    
    // When: update_uncertainty_constraints is called
    subproblem.update_uncertainty_constraints(&innovations);
    
    // Then: Constraint RHS should match expected values
    let model = subproblem.model.as_ref().unwrap();
    for (i, data) in subproblem.uncertainty_observation_data.iter().enumerate() {
        let expected_rhs = data.deterministic_base + 
                          data.seasonal_std * innovations[i];
        
        let constraint_idx = data.constraint_idx;
        let bounds = model.get_row_bounds(constraint_idx);
        
        assert!((bounds.0 - expected_rhs).abs() < 1e-10, 
                "Constraint {} RHS mismatch", i);
    }
}
```

### 3. Regression Tests (TICKET-002)

**Purpose**: Capture current behavior and ensure migration preserves it

#### Baseline Capture

```rust
/// Generate baseline data from current implementation
#[test]
#[ignore]  // Run manually: cargo test generate_baseline -- --ignored
fn generate_baseline_data() {
    // Create a representative system
    let (system, temporal_models) = create_medium_test_system();
    
    // Run forward pass with known seed
    let mut rng = StdRng::seed_from_u64(42);
    let subproblem = Subproblem::new(...);
    
    // Capture state at each step
    let mut baseline = BaselineData::new();
    
    for stage in 0..10 {
        // Sample innovations
        let innovations = sample_innovations(&mut rng, &temporal_models);
        
        // Update constraints
        subproblem.update_uncertainty_constraints(&innovations);
        subproblem.update_lag_fixing_constraints();
        
        // Solve
        let solution = subproblem.solve();
        
        // Record state
        baseline.add_stage(stage, BaselineStage {
            innovations: innovations.clone(),
            load_observations: extract_loads(&solution),
            inflow_observations: extract_inflows(&solution),
            lag_buffer_state: extract_lag_buffer_state(&subproblem),
            constraint_rhs_values: extract_constraint_rhs(&subproblem),
            objective_value: solution.objective,
        });
    }
    
    // Save to file
    let json = serde_json::to_string_pretty(&baseline).unwrap();
    std::fs::write("tests/fixtures/baseline.json", json).unwrap();
    
    println!("✅ Baseline data generated: tests/fixtures/baseline.json");
}
```

#### Equivalence Verification

```rust
/// Verify new implementation matches baseline
#[test]
fn test_numerical_equivalence_to_baseline() {
    // Load baseline data
    let baseline: BaselineData = serde_json::from_str(
        &std::fs::read_to_string("tests/fixtures/baseline.json").unwrap()
    ).unwrap();
    
    // Create same system with new implementation
    let (system, temporal_models) = create_medium_test_system();
    let mut rng = StdRng::seed_from_u64(42);
    let subproblem = Subproblem::new(...);
    
    // Run same simulation
    for stage in 0..baseline.stages.len() {
        let baseline_stage = &baseline.stages[stage];
        
        // Use same innovations
        let innovations = baseline_stage.innovations.clone();
        
        // Update constraints
        subproblem.update_uncertainty_constraints(&innovations);
        subproblem.update_lag_fixing_constraints();
        
        // Solve
        let solution = subproblem.solve();
        
        // Compare results
        let new_loads = extract_loads(&solution);
        let new_inflows = extract_inflows(&solution);
        
        for (i, (&baseline_val, &new_val)) in 
            baseline_stage.load_observations.iter()
                .zip(new_loads.iter()).enumerate() {
            assert!(
                (baseline_val - new_val).abs() < 1e-10,
                "Stage {}, Load {}: baseline={}, new={}",
                stage, i, baseline_val, new_val
            );
        }
        
        for (i, (&baseline_val, &new_val)) in 
            baseline_stage.inflow_observations.iter()
                .zip(new_inflows.iter()).enumerate() {
            assert!(
                (baseline_val - new_val).abs() < 1e-10,
                "Stage {}, Inflow {}: baseline={}, new={}",
                stage, i, baseline_val, new_val
            );
        }
        
        // Compare objective value
        assert!(
            (baseline_stage.objective_value - solution.objective).abs() < 1e-8,
            "Stage {}: objective mismatch", stage
        );
    }
    
    println!("✅ All {} stages match baseline", baseline.stages.len());
}
```

### 4. Edge Case Tests

**Purpose**: Test boundary conditions and unusual configurations

```rust
#[test]
fn test_zero_lag_order() {
    // System with independent models (no AR dynamics)
    let mut subproblem = create_subproblem_with_independent_models();
    
    // Should have no lag data
    assert!(subproblem.load_lag_data.is_none());
    assert!(subproblem.inflow_lag_data.is_none());
    
    // Should still work
    let trajectory = create_test_trajectory();
    subproblem.prepare_from_trajectory(&trajectory);
    // No panics = success
}

#[test]
fn test_single_entity_systems() {
    // System with only one bus
    let subproblem = create_subproblem_with_one_bus();
    assert!(subproblem.load_lag_data.is_some());
    let load_data = subproblem.load_lag_data.as_ref().unwrap();
    assert_eq!(load_data.n_buses, 1);
}

#[test]
fn test_mixed_lag_orders() {
    // Bus 0: PAR(2), Bus 1: Independent, Bus 2: PAR(5)
    let subproblem = create_subproblem_with_mixed_orders();
    let load_data = subproblem.load_lag_data.as_ref().unwrap();
    
    assert_eq!(load_data.buffer[0].len(), 2);
    assert_eq!(load_data.buffer[1].len(), 0);  // Independent
    assert_eq!(load_data.buffer[2].len(), 5);
}

#[test]
fn test_very_large_system() {
    // Stress test: 1000 buses, 500 hydros
    let subproblem = create_large_subproblem(1000, 500);
    
    // Should construct without panic
    assert!(subproblem.load_lag_data.is_some());
    assert!(subproblem.inflow_lag_data.is_some());
    
    // Should update efficiently
    let start = Instant::now();
    let trajectory = create_large_trajectory();
    subproblem.prepare_from_trajectory(&trajectory);
    let duration = start.elapsed();
    
    // Should complete in reasonable time (< 100ms for update)
    assert!(duration.as_millis() < 100, 
            "Buffer update took {}ms", duration.as_millis());
}

#[test]
fn test_high_lag_order() {
    // System with PAR(20) model
    let subproblem = create_subproblem_with_high_order(20);
    let load_data = subproblem.load_lag_data.as_ref().unwrap();
    
    assert_eq!(load_data.buffer[0].len(), 20);
    
    // Should handle trajectory correctly
    let trajectory = create_trajectory_with_length(25);
    subproblem.prepare_from_trajectory(&trajectory);
    
    // All 20 lags should be populated
    for lag_idx in 0..20 {
        assert!(load_data.get_lag(0, lag_idx) != 0.0);
    }
}
```

### 5. Performance Tests (TICKET-010)

**Purpose**: Ensure no performance degradation

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_lag_buffer_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("lag_buffer_update");
    
    for size in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let mut subproblem = create_subproblem_with_size(size);
                let trajectory = create_trajectory_with_size(size);
                
                b.iter(|| {
                    subproblem.prepare_from_trajectory(black_box(&trajectory));
                });
            },
        );
    }
    
    group.finish();
}

fn bench_constraint_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraint_update");
    
    for ar_order in [1, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(ar_order),
            ar_order,
            |b, &ar_order| {
                let mut subproblem = create_subproblem_with_ar_order(ar_order);
                
                b.iter(|| {
                    subproblem.update_lag_fixing_constraints();
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_lag_buffer_update, bench_constraint_update);
criterion_main!(benches);
```

## Test Execution Strategy

### During Development

After each significant change:

```bash
# Quick smoke test
cargo test --lib --tests

# Full test suite
cargo test --all

# Specific test module
cargo test uncertainty_migration_baseline

# With output
cargo test test_name -- --nocapture
```

### Before Each Commit

```bash
#!/bin/bash
# pre-commit-check.sh

set -e

echo "🔨 Building..."
cargo build --all-targets

echo "🧪 Testing..."
cargo test --all

echo "📋 Linting..."
cargo clippy --all-targets --all-features -- -D warnings

echo "💅 Formatting..."
cargo fmt --check

echo "✅ All checks passed!"
```

### Before Merging

```bash
#!/bin/bash
# pre-merge-check.sh

set -e

# Run all tests
cargo test --all --release

# Run ignored tests (baseline generation, etc.)
cargo test -- --ignored

# Run benchmarks
cargo bench --bench uncertainty_migration

# Check code coverage
cargo tarpaulin --out Lcov

# Generate documentation
cargo doc --no-deps

echo "✅ Ready to merge!"
```

## Test Data Fixtures

### Directory Structure

```
tests/
├── fixtures/
│   ├── baseline.json              # Baseline results
│   ├── small_system.json          # 3 buses, 2 hydros
│   ├── medium_system.json         # 50 buses, 20 hydros
│   ├── large_system.json          # 500 buses, 100 hydros
│   ├── mixed_orders.json          # Various AR orders
│   └── edge_cases/
│       ├── zero_lags.json
│       ├── single_entity.json
│       └── high_order.json
├── uncertainty_migration_baseline.rs
└── integration_tests.rs
```

### Fixture Generation

```rust
pub fn create_small_test_system() -> (System, Vec<TemporalModel>) {
    let system = System {
        buses: vec![
            Bus { id: 0, name: "Bus0".to_string(), demand: 100.0 },
            Bus { id: 1, name: "Bus1".to_string(), demand: 150.0 },
            Bus { id: 2, name: "Bus2".to_string(), demand: 200.0 },
        ],
        hydros: vec![
            Hydro { id: 0, name: "Hydro0".to_string(), ... },
            Hydro { id: 1, name: "Hydro1".to_string(), ... },
        ],
        // ... other fields ...
    };
    
    let temporal_models = vec![
        // Load models
        create_par_model(UncertaintyType::Load, 0, 2),  // Bus 0: PAR(2)
        create_par_model(UncertaintyType::Load, 1, 3),  // Bus 1: PAR(3)
        create_independent_model(UncertaintyType::Load, 2),  // Bus 2: Independent
        
        // Inflow models
        create_par_model(UncertaintyType::Inflow, 0, 2),  // Hydro 0: PAR(2)
        create_par_model(UncertaintyType::Inflow, 1, 1),  // Hydro 1: PAR(1)
    ];
    
    (system, temporal_models)
}
```

## Numerical Tolerance Guidelines

### Floating-Point Comparison

```rust
// Use appropriate tolerance based on operation
const TIGHT_TOLERANCE: f64 = 1e-10;  // For exact operations
const SOLVER_TOLERANCE: f64 = 1e-8;   // For LP solver results
const LOOSE_TOLERANCE: f64 = 1e-6;    // For accumulated errors

fn assert_nearly_equal(a: f64, b: f64, tolerance: f64, msg: &str) {
    assert!(
        (a - b).abs() < tolerance,
        "{}: expected {}, got {} (diff: {})",
        msg, a, b, (a - b).abs()
    );
}
```

### Vectorized Comparison

```rust
fn assert_vectors_nearly_equal(
    expected: &[f64],
    actual: &[f64],
    tolerance: f64,
    context: &str
) {
    assert_eq!(expected.len(), actual.len(), 
               "{}: length mismatch", context);
    
    for (i, (&exp, &act)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (exp - act).abs() < tolerance,
            "{} [{}]: expected {}, got {} (diff: {})",
            context, i, exp, act, (exp - act).abs()
        );
    }
}
```

## Coverage Goals

### Target Metrics

- **Line Coverage**: > 90%
- **Branch Coverage**: > 85%
- **Function Coverage**: > 95%
- **Integration Paths**: 100% (all major workflows)

### Measuring Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html --output-dir coverage

# View report
firefox coverage/index.html
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Migration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          profile: minimal
          toolchain: stable
      
      - name: Run tests
        run: cargo test --all
      
      - name: Run regression tests
        run: cargo test uncertainty_migration_baseline
      
      - name: Check clippy
        run: cargo clippy --all-targets -- -D warnings
      
      - name: Check formatting
        run: cargo fmt --check
      
      - name: Run benchmarks
        run: cargo bench --bench uncertainty_migration -- --test
```

## Conclusion

This comprehensive testing strategy ensures:

1. ✅ **Correctness**: Numerical equivalence verified
2. ✅ **Completeness**: All code paths tested
3. ✅ **Robustness**: Edge cases covered
4. ✅ **Performance**: No regressions
5. ✅ **Maintainability**: Clear test documentation

**Remember**: Tests are documentation. Write them clearly and comprehensively.

---

**Next**: Implement TICKET-002 to create the actual test suite based on this strategy.
