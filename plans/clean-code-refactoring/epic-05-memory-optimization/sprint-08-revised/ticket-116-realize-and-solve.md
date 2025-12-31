# [T-116] Update `realize_and_solve()` to Use Iteration Model

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-114
> **Blocks**: T-117
> **Priority**: 1 (Critical Path)
> **Status**: 📋 Planned

## Files to Read Before Starting

- `src/subproblem.rs` - Current `realize_and_solve()` implementation
- T-114 implementation

---

## Context

### Background

With per-iteration Model lifecycle, `realize_and_solve()` uses the existing Model (created at iteration start). RHS updates go directly to Model, not Problem.

### Key Point

RHS updates (uncertainties, lags, hydro balance) are per-solve values that don't persist. Only cuts (from backward pass) update both Problem and Model.

---

## Specification

```rust
pub fn realize_and_solve(
    &mut self,
    innovations: &[f64],
    // ... other params unchanged
) -> RealizeUncertaintiesTiming {
    let mut timing = RealizeUncertaintiesTiming::default();
    
    let model = self.model_mut();  // Panics if not created
    
    let solver_start = std::time::Instant::now();
    
    // Update RHS on Model only (not Problem)
    self.update_uncertainty_rhs_on_model(model, innovations);
    self.update_lag_constraints_on_model(model);
    self.update_hydro_balance_on_model(model, initial_storages);
    
    // Solve
    self.solve_with_retry(model);
    
    timing.solver_time = solver_start.elapsed();
    
    // Extract solution
    let extraction_start = std::time::Instant::now();
    SOLUTION_BUFFER.with(|buffer| {
        let mut solution = buffer.borrow_mut();
        model.get_solution_into(&mut solution);
        // ... process solution
    });
    timing.state_extraction_time = extraction_start.elapsed();
    
    timing
}

/// Update uncertainty observation RHS on Model.
fn update_uncertainty_rhs_on_model(&self, model: &mut solver::Model, innovations: &[f64]) {
    BATCH_ROW_INDICES.with(|indices| {
        BATCH_LOWER_BOUNDS.with(|lowers| {
            BATCH_UPPER_BOUNDS.with(|uppers| {
                let mut indices = indices.borrow_mut();
                let mut lowers = lowers.borrow_mut();
                let mut uppers = uppers.borrow_mut();
                
                indices.clear();
                lowers.clear();
                uppers.clear();
                
                for data in &self.uncertainty_observation_data {
                    let innovation = innovations[data.innovation_idx];
                    let rhs = data.deterministic_base + data.seasonal_std * innovation;
                    
                    indices.push(data.constraint_idx as HighsInt);
                    lowers.push(rhs);
                    uppers.push(rhs);
                }
                
                let _ = model.change_rows_bounds_batch(&indices, &lowers, &uppers);
            })
        })
    });
}

/// Update lag-fixing constraint RHS on Model.
fn update_lag_constraints_on_model(&self, model: &mut solver::Model) {
    if let Some(ref load_data) = self.load_lag_data {
        for (bus_id, lags) in load_data.buffer.iter().enumerate() {
            for (lag_idx, &value) in lags.iter().enumerate() {
                let row = load_data.constraints.get_constraint(bus_id, lag_idx);
                model.change_rows_bounds(row, value, value);
            }
        }
    }
    // Similar for inflow_lag_data
}

/// Update hydro balance RHS on Model.
fn update_hydro_balance_on_model(&self, model: &mut solver::Model, storages: &[f64]) {
    BATCH_ROW_INDICES.with(|indices| {
        BATCH_LOWER_BOUNDS.with(|lowers| {
            BATCH_UPPER_BOUNDS.with(|uppers| {
                let mut indices = indices.borrow_mut();
                let mut lowers = lowers.borrow_mut();
                let mut uppers = uppers.borrow_mut();
                
                indices.clear();
                lowers.clear();
                uppers.clear();
                
                for (i, &row) in self.constraints.hydro_balance.iter().enumerate() {
                    indices.push(row as HighsInt);
                    lowers.push(storages[i]);
                    uppers.push(storages[i]);
                }
                
                let _ = model.change_rows_bounds_batch(&indices, &lowers, &uppers);
            })
        })
    });
}
```

---

## Acceptance Criteria

- [ ] `realize_and_solve()` uses `model_mut()`
- [ ] RHS updates go to Model only
- [ ] Batch updates use thread-local buffers
- [ ] Solution extraction works
- [ ] All SDDP tests pass
- [ ] Golden tests pass

---

## Testing Requirements

```rust
#[test]
fn test_realize_and_solve_with_iteration_model() {
    let mut subproblem = create_initialized_subproblem();
    subproblem.preallocate_cut_constraints(100, 10).unwrap();
    subproblem.create_iteration_model(false).unwrap();
    
    let innovations = vec![0.0; num_innovations(&subproblem)];
    let timing = subproblem.realize_and_solve(&innovations, ...);
    
    assert!(timing.solver_time > Duration::ZERO);
    assert!(subproblem.has_model());
    
    subproblem.finalize_iteration(false);
}

#[test]
fn test_multiple_solves_same_iteration() {
    let mut subproblem = create_initialized_subproblem();
    subproblem.preallocate_cut_constraints(100, 10).unwrap();
    subproblem.create_iteration_model(false).unwrap();
    
    let innovations = vec![0.0; num_innovations(&subproblem)];
    
    for _ in 0..5 {
        let _ = subproblem.realize_and_solve(&innovations, ...);
    }
    
    assert!(subproblem.has_model());
    subproblem.finalize_iteration(false);
}

#[test]
#[should_panic]
fn test_solve_without_model_panics() {
    let mut subproblem = create_initialized_subproblem();
    // No create_iteration_model() call
    let innovations = vec![0.0; num_innovations(&subproblem)];
    let _ = subproblem.realize_and_solve(&innovations, ...);
}
```

---

## Effort Estimate

**Points**: 3
**Confidence**: High

---

## Definition of Done

- [ ] `realize_and_solve()` updated
- [ ] RHS methods use Model reference
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] PR merged
