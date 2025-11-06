# Migration Steps: Unified to Separated Architecture

**Status**: IMPLEMENTATION GUIDE  
**Date**: 2025-11-06  
**Ticket**: TICKET-001

## Overview

This document provides a detailed, step-by-step guide for executing the migration from unified to separated uncertainty handling. Follow these steps sequentially to minimize risk and ensure correctness.

## Prerequisites

### Before Starting

- [ ] Read `current_architecture.md` completely
- [ ] Read `target_architecture.md` completely
- [ ] Understand Rust ownership and borrowing
- [ ] Have access to baseline test data
- [ ] Have benchmarking tools installed (`cargo install criterion`)
- [ ] Create a feature branch: `git checkout -b migration/separated-uncertainty`

### Safety Checks

```bash
# Ensure clean starting state
cargo test --all
cargo clippy --all-targets -- -D warnings
cargo fmt --check
git status  # Should be clean
```

## Step-by-Step Implementation

### TICKET-001: Document Current State ✅ COMPLETE

**Status**: Complete (this document and related docs created)

**Deliverables**:
- [x] `docs/migration/current_architecture.md`
- [x] `docs/migration/target_architecture.md`
- [x] `docs/migration/migration_steps.md` (this file)
- [ ] `docs/migration/testing_equivalence.md` (next)

---

### TICKET-002: Create Regression Test Suite

**Duration**: 2-3 days  
**Priority**: P0 (Blocker)

#### Step 2.1: Create Test Module

```bash
# Create test file
touch tests/uncertainty_migration_baseline.rs
```

#### Step 2.2: Implement Baseline Tests

Add to `tests/uncertainty_migration_baseline.rs`:

```rust
//! Regression tests for uncertainty handling migration
//! 
//! These tests capture the current behavior to ensure the migration
//! produces identical results.

use powers_rs::*;

/// Test fixture with sample temporal models
fn create_test_system() -> (System, Vec<TemporalModel>) {
    // Implement: Create a small test system
    // - 3 buses, 2 hydros
    // - 2 loads with PAR(2) models
    // - 2 inflows with PAR(3) models
    // - Known seasonal parameters for verification
}

#[test]
fn test_lag_buffer_population_from_trajectory() {
    // Given: A subproblem and trajectory
    // When: prepare_from_trajectory() is called
    // Then: Lag buffers contain correct historical values
}

#[test]
fn test_unified_vs_separated_lag_constraints_consistency() {
    // Given: A subproblem with both unified and separated structures
    // When: Constraints are populated
    // Then: Both structures contain identical constraint indices
    //       (This test should FAIL after TICKET-004)
}

#[test]
fn test_uncertainty_constraint_rhs_computation() {
    // Given: Known innovations and lag values
    // When: update_uncertainty_constraints() is called
    // Then: RHS values match hand-calculated expected values
}

#[test]
fn test_entity_data_routing_consistency() {
    // Given: entity_data with routing fields
    // When: Accessing data by bus_id or hydro_id
    // Then: Routing maps correctly to observations
}

#[test]
fn test_global_entity_indexing() {
    // Given: Mixed loads and inflows
    // When: Building entity_data
    // Then: global_entity_idx matches expected ordering
}
```

#### Step 2.3: Run Baseline Tests

```bash
# Run tests and save output
cargo test uncertainty_migration_baseline -- --nocapture > baseline_test_output.txt

# Verify all tests pass
cargo test uncertainty_migration_baseline
```

#### Step 2.4: Generate Baseline Data

```rust
// Add to test module
#[test]
#[ignore]  // Run manually to generate baseline
fn generate_baseline_data() {
    // Run SDDP simulation with current implementation
    // Save results to tests/fixtures/baseline.json
    // Include: constraint RHS values, lag buffer states, solution values
}
```

```bash
cargo test generate_baseline_data -- --ignored --nocapture
```

**Checkpoint**: All baseline tests pass, baseline data generated

---

### TICKET-003: Refactor LoadLagData and InflowLagData

**Duration**: 1-2 days  
**Priority**: P1

#### Step 3.1: Define LoadLagData

Add to `src/subproblem.rs` (after existing lag structures):

```rust
/// Type-safe container for load lag variables, constraints, and observations.
#[derive(Clone, Debug)]
pub struct LoadLagData {
    /// Variables: [bus_id][lag_idx] → LP variable index
    pub variables: LoadLagVariables,
    
    /// Constraints: [bus_id][lag_idx] → LP constraint index
    pub constraints: LoadLagConstraints,
    
    /// Buffer: [bus_id][lag_idx] → lag observation value
    pub buffer: Vec<Vec<f64>>,
    
    /// Number of buses with uncertain loads
    pub n_buses: usize,
    
    /// Maximum lag order across all buses
    pub max_lag: usize,
}

impl LoadLagData {
    pub fn new(n_buses: usize, max_lag: usize) -> Self {
        Self {
            variables: LoadLagVariables::new(n_buses),
            constraints: LoadLagConstraints::new(n_buses),
            buffer: vec![Vec::new(); n_buses],
            n_buses,
            max_lag,
        }
    }
    
    #[inline]
    pub fn get_lag(&self, bus_id: usize, lag_idx: usize) -> f64 {
        self.buffer[bus_id][lag_idx]
    }
    
    #[inline]
    pub fn set_lag(&mut self, bus_id: usize, lag_idx: usize, value: f64) {
        self.buffer[bus_id][lag_idx] = value;
    }
}
```

#### Step 3.2: Define InflowLagData

Add to `src/subproblem.rs`:

```rust
/// Type-safe container for inflow lag variables, constraints, and observations.
#[derive(Clone, Debug)]
pub struct InflowLagData {
    /// Variables: [hydro_id][lag_idx] → LP variable index
    pub variables: InflowLagVariables,
    
    /// Constraints: [hydro_id][lag_idx] → LP constraint index
    pub constraints: InflowLagConstraints,
    
    /// Buffer: [hydro_id][lag_idx] → lag observation value
    pub buffer: Vec<Vec<f64>>,
    
    /// Number of hydros with uncertain inflows
    pub n_hydros: usize,
    
    /// Maximum lag order across all hydros
    pub max_lag: usize,
}

impl InflowLagData {
    pub fn new(n_hydros: usize, max_lag: usize) -> Self {
        Self {
            variables: InflowLagVariables::new(n_hydros),
            constraints: InflowLagConstraints::new(n_hydros),
            buffer: vec![Vec::new(); n_hydros],
            n_hydros,
            max_lag,
        }
    }
    
    #[inline]
    pub fn get_lag(&self, hydro_id: usize, lag_idx: usize) -> f64 {
        self.buffer[hydro_id][lag_idx]
    }
    
    #[inline]
    pub fn set_lag(&mut self, hydro_id: usize, lag_idx: usize, value: f64) {
        self.buffer[hydro_id][lag_idx] = value;
    }
}
```

#### Step 3.3: Add Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_load_lag_data_construction() {
        let data = LoadLagData::new(3, 5);
        assert_eq!(data.n_buses, 3);
        assert_eq!(data.max_lag, 5);
        assert_eq!(data.buffer.len(), 3);
    }
    
    #[test]
    fn test_inflow_lag_data_construction() {
        let data = InflowLagData::new(2, 3);
        assert_eq!(data.n_hydros, 2);
        assert_eq!(data.max_lag, 3);
        assert_eq!(data.buffer.len(), 2);
    }
    
    #[test]
    fn test_load_lag_data_get_set() {
        let mut data = LoadLagData::new(1, 2);
        data.buffer[0] = vec![0.0, 0.0];
        
        data.set_lag(0, 0, 42.0);
        assert_eq!(data.get_lag(0, 0), 42.0);
    }
}
```

#### Step 3.4: Verify Compilation

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt
```

**Checkpoint**: New structures compile, all tests pass

---

### TICKET-004: Remove Unified lag_fixing_constraints

**Duration**: 2-3 days  
**Priority**: P1

#### Step 4.1: Add Fields to Subproblem (Transition)

In `src/subproblem.rs`, add new fields to `Subproblem` struct:

```rust
pub struct Subproblem {
    // ... existing fields ...
    
    /// Type-safe load lag data (variables, constraints, buffer)
    pub load_lag_data: Option<LoadLagData>,
    
    /// Type-safe inflow lag data (variables, constraints, buffer)
    pub inflow_lag_data: Option<InflowLagData>,
}
```

#### Step 4.2: Update Constraint Population

Modify the constraint building in `build_constraints()` (around line 1806):

```rust
// BEFORE: Parallel population
let (lag_fixing_constraints, load_lag_constraints, inflow_lag_constraints) = ...;

// AFTER: Populate separated structures only (remove old unified structure)
let (load_lag_constraints, inflow_lag_constraints) = 
    if let Some(ref lag_vars) = variables.lagged_state {
        let mut new_load_constraints = LoadLagConstraints::new(system.buses.len());
        let mut new_inflow_constraints = InflowLagConstraints::new(system.hydros.len());
        
        for (entity_idx, entity_lags) in lag_vars.iter().enumerate() {
            let mut entity_constraints = Vec::new();
            
            for &var in entity_lags {
                let constraint = pb.add_row(0.0..=0.0, vec![(var, 1.0)]);
                entity_constraints.push(constraint);
            }
            
            // Route to appropriate structure
            let model = &temporal_models[entity_idx];
            match model.entity_type {
                UncertaintyType::Load => {
                    new_load_constraints.constraints_by_bus[model.entity_id] = 
                        entity_constraints;
                }
                UncertaintyType::Inflow => {
                    new_inflow_constraints.constraints_by_hydro[model.entity_id] = 
                        entity_constraints;
                }
            }
        }
        
        (
            if new_load_constraints.total_constraint_count() > 0 {
                Some(new_load_constraints)
            } else {
                None
            },
            if new_inflow_constraints.total_constraint_count() > 0 {
                Some(new_inflow_constraints)
            } else {
                None
            }
        )
    } else {
        (None, None)
    };
```

#### Step 4.3: Remove lag_fixing_constraints Field

In `src/subproblem.rs`, update `Constraints` struct:

```rust
pub struct Constraints {
    pub load_balance: Vec<usize>,
    pub hydro_balance: Vec<usize>,
    pub uncertainty_observation: Vec<usize>,
    // pub lag_fixing_constraints: Option<Vec<Vec<usize>>>,  // ❌ REMOVE THIS
    pub load_lag_constraints: Option<LoadLagConstraints>,
    pub inflow_lag_constraints: Option<InflowLagConstraints>,
}
```

#### Step 4.4: Update All References

```bash
# Find all usages
grep -r "lag_fixing_constraints" src/ tests/

# Update each usage to use separated structures
# Most should already be using load_lag_constraints/inflow_lag_constraints
```

#### Step 4.5: Fix Baseline Test

Update `test_unified_vs_separated_lag_constraints_consistency()`:

```rust
#[test]
#[should_panic]  // This test SHOULD fail now - old structure removed
fn test_unified_structure_removed() {
    // This test verifies the old unified structure is gone
    let subproblem = create_test_subproblem();
    
    // This should not compile or should be None:
    // assert!(subproblem.constraints.lag_fixing_constraints.is_none());
}
```

#### Step 4.6: Verify

```bash
cargo build
cargo test
cargo clippy -- -D warnings
cargo fmt

# Run regression tests
cargo test uncertainty_migration_baseline
```

**Checkpoint**: lag_fixing_constraints removed, tests pass

---

### TICKET-005: Replace UncertaintyConstraintManager

**Duration**: 3-5 days  
**Priority**: P1

#### Step 5.1: Add Buffer Update Methods

Add to `LoadLagData`:

```rust
impl LoadLagData {
    /// Update lag buffer from trajectory
    pub fn update_from_trajectory(&mut self, trajectory: &[&Realization]) {
        for bus_id in 0..self.n_buses {
            let max_lag = self.buffer[bus_id].len();
            for lag_idx in 0..max_lag {
                let lookback = lag_idx + 1;
                if lookback < trajectory.len() {
                    let past_idx = trajectory.len() - 1 - lookback;
                    let lag_value = trajectory[past_idx].loads[bus_id];
                    self.buffer[bus_id][lag_idx] = lag_value;
                }
            }
        }
    }
}
```

Add similar method to `InflowLagData`.

#### Step 5.2: Remove uncertainty_manager Field

In `Subproblem`:

```rust
// ❌ REMOVE:
// pub uncertainty_manager: UncertaintyConstraintManager,

// Fields already added in TICKET-004:
pub load_lag_data: Option<LoadLagData>,
pub inflow_lag_data: Option<InflowLagData>,
```

#### Step 5.3: Update prepare_from_trajectory()

```rust
fn prepare_from_trajectory(&mut self, trajectory: &[&Realization]) {
    // Storage (unchanged)
    let storage = self.extract_storage_from_trajectory(trajectory);
    self.update_storage_constraints(&storage);
    
    // NEW: Update separated lag buffers
    if let Some(ref mut load_data) = self.load_lag_data {
        load_data.update_from_trajectory(trajectory);
    }
    
    if let Some(ref mut inflow_data) = self.inflow_lag_data {
        inflow_data.update_from_trajectory(trajectory);
    }
    
    // Update constraints
    self.update_lag_fixing_constraints();
}
```

#### Step 5.4: Update update_lag_fixing_constraints()

```rust
fn update_lag_fixing_constraints(&mut self) {
    if let Some(model) = self.model.as_mut() {
        // Update load lag constraints
        if let Some(ref load_data) = self.load_lag_data {
            for (bus_id, constraints) in load_data.constraints
                .constraints_by_bus.iter().enumerate() {
                for (lag_idx, &constraint_idx) in constraints.iter().enumerate() {
                    let lag_value = load_data.buffer[bus_id][lag_idx];
                    model.change_rows_bounds(constraint_idx, lag_value, lag_value);
                }
            }
        }
        
        // Update inflow lag constraints
        if let Some(ref inflow_data) = self.inflow_lag_data {
            for (hydro_id, constraints) in inflow_data.constraints
                .constraints_by_hydro.iter().enumerate() {
                for (lag_idx, &constraint_idx) in constraints.iter().enumerate() {
                    let lag_value = inflow_data.buffer[hydro_id][lag_idx];
                    model.change_rows_bounds(constraint_idx, lag_value, lag_value);
                }
            }
        }
    }
}
```

#### Step 5.5: Remove Entity Mapping Logic

Delete the HashMap construction code (lines 2050-2070).

#### Step 5.6: Update Construction

In `Subproblem::new()` or equivalent constructor, initialize new fields:

```rust
// Initialize load_lag_data if loads have AR dynamics
let load_lag_data = if has_load_lags {
    let mut data = LoadLagData::new(n_buses, max_load_lag);
    // Initialize buffers
    for bus_id in 0..n_buses {
        data.buffer[bus_id] = vec![0.0; ar_orders[bus_id]];
    }
    Some(data)
} else {
    None
};

// Similar for inflow_lag_data
```

#### Step 5.7: Fix Imports

Remove imports from `uncertainty_constraints` module where no longer needed.

#### Step 5.8: Verify

```bash
cargo build
cargo test
cargo test uncertainty_migration_baseline
cargo clippy -- -D warnings
```

**Checkpoint**: UncertaintyConstraintManager removed, direct buffer access works

---

### TICKET-006: Simplify UncertaintyConstraintData

**Duration**: 2-3 days  
**Priority**: P2

#### Step 6.1: Define UncertaintyObservationData

Add to `src/subproblem.rs`:

```rust
/// Precomputed coefficients for fast uncertainty observation constraint RHS updates.
#[derive(Debug, Clone)]
pub struct UncertaintyObservationData {
    /// LP constraint index to update
    pub constraint_idx: usize,
    
    /// Index in innovations array for this entity
    pub innovation_idx: usize,
    
    /// Standard deviation for current season (σ_s)
    pub seasonal_std: f64,
    
    /// Precomputed deterministic part: μ_s - Σ(φ_k · μ_{s-k})
    pub deterministic_base: f64,
}
```

#### Step 6.2: Update Subproblem Field

```rust
pub struct Subproblem {
    // ... other fields ...
    
    // pub entity_data: Vec<UncertaintyConstraintData>,  // ❌ REMOVE
    pub uncertainty_observation_data: Vec<UncertaintyObservationData>,  // ✅ ADD
}
```

#### Step 6.3: Update build_entity_constraint_data()

Rename and simplify:

```rust
fn build_uncertainty_observation_data(
    temporal_models: &[TemporalModel],
    constraints: &Constraints,
    season_id: usize,
) -> Vec<UncertaintyObservationData> {
    temporal_models.iter().enumerate().map(|(idx, model)| {
        UncertaintyObservationData {
            constraint_idx: constraints.uncertainty_observation[idx],
            innovation_idx: idx,
            seasonal_std: model.seasonal_stds[season_id],
            deterministic_base: model.deterministic_bases[season_id],
        }
    }).collect()
}
```

#### Step 6.4: Update update_uncertainty_constraints()

```rust
fn update_uncertainty_constraints(&mut self, innovations: &[f64]) {
    if let Some(model) = self.model.as_mut() {
        for data in &self.uncertainty_observation_data {
            let innovation = innovations[data.innovation_idx];
            let rhs = data.deterministic_base + data.seasonal_std * innovation;
            model.change_rows_bounds(data.constraint_idx, rhs, rhs);
        }
    }
}
```

#### Step 6.5: Remove Old Struct

Delete `UncertaintyConstraintData` struct definition.

#### Step 6.6: Verify

```bash
cargo build
cargo test
cargo clippy -- -D warnings
```

**Checkpoint**: Simplified data structure, tests pass

---

### TICKET-007: Remove uncertainty_constraints Module

**Duration**: 1 day  
**Priority**: P2

#### Step 7.1: Verify No References

```bash
grep -r "UncertaintyConstraintManager" src/
grep -r "UnifiedLagBuffer" src/
grep -r "use.*uncertainty_constraints" src/
```

All should return no results (or only in tests to be updated).

#### Step 7.2: Remove Module

```bash
# Remove module file
git rm src/uncertainty_constraints.rs

# Remove from lib.rs
# Delete line: mod uncertainty_constraints;
```

#### Step 7.3: Update lib.rs

Remove the module declaration.

#### Step 7.4: Verify

```bash
cargo build
cargo test
cargo clippy -- -D warnings
```

**Checkpoint**: Module removed, everything still compiles

---

### TICKET-008-012: Remaining Tickets

Follow similar detailed steps for:
- TICKET-008: Direct extraction optimization (optional)
- TICKET-009: Command-Query Separation
- TICKET-010: Performance benchmarking
- TICKET-011: Documentation updates
- TICKET-012: Final integration testing

See IMPLEMENTATION_TICKETS.md for full details.

## Rollback Procedures

### If Issues Discovered

#### Before TICKET-004
```bash
git stash  # Save work
git checkout main
# Easy rollback - no breaking changes yet
```

#### After TICKET-004
```bash
# Create rollback branch
git checkout -b rollback/unified-structure
git revert <commit-range>
```

#### After TICKET-007
```bash
# Must complete migration - no clean rollback
# Fix issues forward rather than reverting
```

## Quality Gates

### After Each Ticket

- [ ] Code compiles without warnings
- [ ] All tests pass (including new regression tests)
- [ ] Clippy clean: `cargo clippy -- -D warnings`
- [ ] Formatted: `cargo fmt --check`
- [ ] Documentation updated
- [ ] Git commit with clear message

### Before Merging

- [ ] All tickets complete
- [ ] Performance benchmarks show no regression
- [ ] Code coverage maintained (>90%)
- [ ] All documentation updated
- [ ] CHANGELOG.md updated
- [ ] Peer review complete

## Continuous Verification

### Run After Each Change

```bash
#!/bin/bash
# verify.sh - Run after each significant change

set -e

echo "Building..."
cargo build --all-targets

echo "Testing..."
cargo test --all

echo "Linting..."
cargo clippy --all-targets --all-features -- -D warnings

echo "Formatting..."
cargo fmt --check

echo "Benchmarking..."
cargo bench --bench uncertainty_migration

echo "✅ All checks passed!"
```

## Success Criteria

### Must Achieve
- ✅ All existing tests pass
- ✅ Numerical equivalence verified (< 1e-10 tolerance)
- ✅ No performance regression > 5%
- ✅ Code coverage maintained
- ✅ Clippy clean
- ✅ Documentation complete

### Nice to Have
- ✅ Performance improvement > 10%
- ✅ Memory reduction > 15%
- ✅ Code size reduction > 200 lines

## Conclusion

Follow these steps sequentially, verifying at each checkpoint. The migration is designed to be gradual and safe, with comprehensive testing at each stage.

**Remember**: Correctness first, performance second, elegance third.

---

**Next**: See `testing_equivalence.md` for detailed testing strategy.
