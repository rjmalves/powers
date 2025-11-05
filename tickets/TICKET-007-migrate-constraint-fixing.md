# [TICKET-007] Migrate Lag Constraint Fixing Logic

**Sprint:** 3  
**Estimated Effort:** 2 story points (1-2 days)  
**Confidence:** High  
**Priority:** P2 - Medium

## Context

When solving a subproblem at a given stage, lag variables must be "fixed" to values from the previous stage's state. This is done by setting constraint RHS bounds to fix each lag variable to its historical value: `Y_{t-k} = state_value`.

Currently, this logic iterates through unified constraint structures. With explicit lag constraints, we can simplify to direct access patterns that are clearer and more efficient.

## Acceptance Criteria

- [ ] Given state with load lag values for each bus, when fixing lag constraints, then `load_lag_constraints` RHS values are set correctly by bus_id
- [ ] Given state with inflow lag values for each hydro, when fixing lag constraints, then `inflow_lag_constraints` RHS values are set correctly by hydro_id
- [ ] Given subproblem solve with fixed lags, when extracting solution, then lag variable values equal the fixed values (within solver tolerance)
- [ ] Given system with mixed AR orders, when fixing constraints, then entities with AR(0) are skipped without error
- [ ] Performance: Constraint fixing should be negligibly fast (< 0.1ms for typical system)

## Tasks

### Implementation

- [ ] Locate lag constraint fixing methods in `src/subproblem.rs` or `src/state.rs`
  - Methods that call `model.set_rhs()` or similar for lag constraints
  - Initialization logic for new subproblems
  
- [ ] Update constraint fixing for load lags
  ```rust
  if let Some(load_constraints) = &self.constraints.load_lag_constraints {
      for bus_id in 0..n_buses {
          let lag_values = &state.load_lag_values[bus_id];
          let constraints = load_constraints.get_constraints(bus_id);
          
          for (lag_idx, &value) in lag_values.iter().enumerate() {
              let constraint = constraints[lag_idx];
              model.set_rhs(constraint, value..=value); // Fix to exact value
          }
      }
  }
  ```
  
- [ ] Update constraint fixing for inflow lags
  ```rust
  if let Some(inflow_constraints) = &self.constraints.inflow_lag_constraints {
      for hydro_id in 0..n_hydros {
          let lag_values = &state.inflow_lag_values[hydro_id];
          let constraints = inflow_constraints.get_constraints(hydro_id);
          
          for (lag_idx, &value) in lag_values.iter().enumerate() {
              let constraint = constraints[lag_idx];
              model.set_rhs(constraint, value..=value);
          }
      }
  }
  ```
  
- [ ] Remove unified constraint iteration
  - Delete old loop over `lag_fixing_constraints`
  - Remove entity type matching logic
  
- [ ] Add validation for state-constraint consistency
  - Verify lag value count matches constraint count
  - Clear error messages if mismatch occurs

### Testing

- [ ] Unit test: Fix lag constraints for simple system
  - System: 2 buses (AR(1), AR(2)), 2 hydros (AR(1), AR(0))
  - State: load_lags = [[5.0], [10.0, 15.0]], inflow_lags = [[20.0], []]
  - Fix constraints and verify model RHS values
  
- [ ] Unit test: Solve subproblem with fixed lags
  - Fix lag values
  - Solve subproblem
  - Extract solution
  - Verify lag variables equal fixed values (within tolerance 1e-6)
  
- [ ] Unit test: Empty lag vectors (AR(0) entities)
  - Verify no constraints created or fixed
  - No errors or panics
  
- [ ] Integration test: Multi-stage problem with state transitions
  - Stage 1: solve, extract state
  - Stage 2: fix lags from stage 1 state, solve
  - Stage 3: fix lags from stage 2 state, solve
  - Verify lag continuity across stages
  
- [ ] Regression test: Compare with old implementation
  - Same system, same state values
  - Old and new produce identical constraint RHS settings
  
- [ ] Performance test: Constraint fixing overhead
  - System with 50 entities (various AR orders)
  - Measure time to fix all lag constraints
  - Should be < 0.1ms
  
- [ ] Error handling test: Mismatched lag count
  - State has 3 lag values but constraint expects 2
  - Should panic with clear error message in debug mode

### Documentation

- [ ] Update doc comments for constraint fixing methods
- [ ] Explain the separation of load and inflow constraint fixing
- [ ] Document the state-constraint relationship
- [ ] Add example showing typical constraint fixing flow
- [ ] Update any state transition documentation

## Technical Notes

### Current Implementation (Conceptual)

```rust
fn fix_lag_constraints(
    &mut self,
    model: &mut solver::Model,
    state: &State,
) {
    if let Some(lag_constraints) = &self.constraints.lag_fixing_constraints {
        for (entity_idx, entity_constraints) in lag_constraints.iter().enumerate() {
            // Need to know entity type to get correct state values
            let entity = &self.entity_metadata[entity_idx];
            let lag_values = match entity.entity_type {
                UncertaintyType::Load => {
                    &state.load_lag_values[entity.entity_id]
                }
                UncertaintyType::Inflow => {
                    &state.inflow_lag_values[entity.entity_id]
                }
            };
            
            for (lag_idx, &constraint) in entity_constraints.iter().enumerate() {
                let value = lag_values[lag_idx];
                model.set_rhs(constraint, value..=value);
            }
        }
    }
}
```

### Proposed Implementation

```rust
fn fix_lag_constraints(
    &mut self,
    model: &mut solver::Model,
    state: &State,
) {
    // Fix load lag constraints
    if let Some(load_constraints) = &self.constraints.load_lag_constraints {
        for bus_id in 0..self.system.buses.len() {
            let lag_values = &state.load_lag_values[bus_id];
            if lag_values.is_empty() {
                continue; // No lags for this bus
            }
            
            let constraints = load_constraints.get_constraints(bus_id);
            
            debug_assert_eq!(
                lag_values.len(), constraints.len(),
                "Lag value count mismatch for bus {}: {} values but {} constraints",
                bus_id, lag_values.len(), constraints.len()
            );
            
            for (lag_idx, &value) in lag_values.iter().enumerate() {
                let constraint = constraints[lag_idx];
                model.set_rhs(constraint, value..=value);
            }
        }
    }
    
    // Fix inflow lag constraints
    if let Some(inflow_constraints) = &self.constraints.inflow_lag_constraints {
        for hydro_id in 0..self.system.hydros.len() {
            let lag_values = &state.inflow_lag_values[hydro_id];
            if lag_values.is_empty() {
                continue; // No lags for this hydro
            }
            
            let constraints = inflow_constraints.get_constraints(hydro_id);
            
            debug_assert_eq!(
                lag_values.len(), constraints.len(),
                "Lag value count mismatch for hydro {}: {} values but {} constraints",
                hydro_id, lag_values.len(), constraints.len()
            );
            
            for (lag_idx, &value) in lag_values.iter().enumerate() {
                let constraint = constraints[lag_idx];
                model.set_rhs(constraint, value..=value);
            }
        }
    }
}
```

### Benefits

1. **Clarity:** Explicit "fix load constraints" and "fix inflow constraints" sections
2. **Efficiency:** Direct indexed access, no entity type matching
3. **Debuggability:** Clear assertions about which entity has mismatch
4. **Maintainability:** Easy to add entity-specific logic if needed

### State Structure

Ensure `State` struct has:
```rust
pub struct State {
    // Storage state
    pub storage_values: Vec<f64>,
    
    // Lag states (explicit)
    pub load_lag_values: Vec<Vec<f64>>,   // [bus_id][lag_idx]
    pub inflow_lag_values: Vec<Vec<f64>>, // [hydro_id][lag_idx]
}
```

This should already be the case after the Realization refactoring.

### Constraint RHS Setting

The `model.set_rhs()` call sets both lower and upper bounds:
```rust
model.set_rhs(constraint, value..=value);
// Equivalent to: constraint LHS = value (fixed)
```

For lag variables: `Y_{t-k} = historical_value`

### Edge Cases

- **Empty lag vectors:** Entity with AR(0) has no lags, skip constraint fixing
- **First stage:** May not have previous state, initialize lags to zero or mean
- **Very small/large values:** Ensure solver handles full f64 range
- **NaN in state:** Should error clearly rather than silently propagate
- **Inconsistent state dimensions:** State from different system configuration

### Error Handling

```rust
#[derive(Debug)]
enum ConstraintFixingError {
    LagCountMismatch {
        entity_type: &'static str,
        entity_id: usize,
        expected: usize,
        actual: usize,
    },
    InvalidLagValue {
        entity_type: &'static str,
        entity_id: usize,
        lag_idx: usize,
        value: f64,
    },
}

// Usage
if lag_values.len() != constraints.len() {
    return Err(ConstraintFixingError::LagCountMismatch {
        entity_type: "Load",
        entity_id: bus_id,
        expected: constraints.len(),
        actual: lag_values.len(),
    });
}

if !value.is_finite() {
    return Err(ConstraintFixingError::InvalidLagValue {
        entity_type: "Inflow",
        entity_id: hydro_id,
        lag_idx,
        value,
    });
}
```

### Performance Considerations

Constraint fixing is extremely fast:
- O(total_lag_count) which is typically < 100
- Simple RHS updates, no matrix modifications
- Should complete in microseconds

Not a bottleneck, but good to verify no regression.

### Testing Strategy

Key tests:
1. **Correctness:** Fixed values actually applied
2. **Continuity:** Multi-stage problems maintain state correctly
3. **Edge cases:** Empty lags, first stage, extreme values
4. **Regression:** Identical behavior to old implementation
5. **Performance:** Negligible overhead

## Dependencies

- Blocked by: TICKET-001, TICKET-002 (need explicit structures)
- Blocks: None
- Related: TICKET-006 (extracts state values used here)

## Definition of Done

- [ ] All constraint fixing uses explicit structures
- [ ] No entity type matching in fixing logic
- [ ] All tests pass including multi-stage continuity
- [ ] Clear error messages for mismatches
- [ ] Performance verified negligible
- [ ] Code reviewed
- [ ] Documentation complete
