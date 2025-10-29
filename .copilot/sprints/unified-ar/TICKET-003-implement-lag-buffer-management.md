# TICKET-003: Implement Lag Buffer Management

**Sprint:** 1 - Foundation  
**Phase:** 1 - Create Unified Inflow Model  
**Estimated Effort:** 3 days (8 story points)  
**Confidence:** High  
**Status:** Not Started

## Context

The lag buffer maintains historical residual values (Z'_{t-1}, Z'_{t-2}, ..., Z'\_{t-p}) needed for AR dynamics. This ticket implements methods to initialize, update, and access the lag buffer consistently across forward passes and trajectory simulation.

The lag buffer is the "state" of the AR model - it must be updated after each realization and used to set constraint RHS values before solving. Proper lag management is critical for correctness.

## Acceptance Criteria

- [ ] Given a trajectory of past realizations, when lag buffer is updated, then it extracts correct residual values from the last p realizations
- [ ] Given initial conditions from PreStudy nodes, when lag buffer is initialized, then it uses residuals from PreStudy realizations
- [ ] Given a hydro with AR(2) and trajectory [t-3, t-2, t-1], when lag buffer is queried, then it returns [Z'_{t-1}, Z'_{t-2}]
- [ ] Given a hydro with independent model (p=0), when lag buffer is accessed, then it returns empty vector
- [ ] Given lag values, when set_lag_constraints_rhs() is called, then solver model RHS is updated correctly
- [ ] Performance: Lag buffer update should be O(n\*p) where n=hydros, p=max_lag

## Tasks

### Implementation

- [ ] Implement `initialize_lag_buffer()` method:
  - Takes `&mut self`, `initial_trajectory: &[Realization]`
  - Extracts last p residuals per hydro
  - Handles PreStudy nodes with multi-node initialization (PAR case)
- [ ] Implement `update_lag_buffer()` method:
  - Takes `&mut self`, `realization: &Realization`
  - Shifts buffer: [t-1, t-2] → [t, t-1] and inserts new residual
  - Uses circular buffer or shift operation
- [ ] Implement `update_lag_buffer_from_trajectory()` method:
  - Takes `&mut self`, `trajectory: &[Realization]`
  - Extracts last p realizations for each hydro
  - More efficient than repeated update_lag_buffer calls
- [ ] Implement `get_lag_residuals(&self, hydro: usize) -> &[f64]` method:
  - Returns slice of lag values for given hydro
  - Empty slice for independent hydros
- [ ] Implement `set_lag_constraint_rhs()` method:
  - Takes `&self`, `model: &mut solver::Model`, `constraints: &ConstraintIndices`
  - Sets RHS of lag transfer constraints to current lag buffer values
  - Only for hydros with lags (skip independent hydros)
- [ ] Add `clear_lag_buffer()` method for testing/reset
- [ ] Handle edge case: insufficient trajectory length (< p realizations)

### Testing

- [ ] Unit test: Initialize lag buffer from trajectory with 3 PreStudy nodes (PAR case)
- [ ] Unit test: Initialize lag buffer from single PreStudy node (standard case)
- [ ] Unit test: Update lag buffer with single realization, verify shift behavior
- [ ] Unit test: Update lag buffer from trajectory, verify bulk update correct
- [ ] Unit test: Get lag residuals for AR(2) hydro returns correct values
- [ ] Unit test: Get lag residuals for independent hydro returns empty slice
- [ ] Unit test: Set lag constraint RHS updates solver model correctly
- [ ] Unit test: Mixed hydros (AR(1), AR(2), independent), verify all handled correctly
- [ ] Unit test: Edge case - trajectory shorter than lag order (should handle gracefully or error)
- [ ] Integration test: Full forward pass with lag buffer updates

### Documentation

- [ ] Add doc comment for `initialize_lag_buffer()` explaining PreStudy handling
- [ ] Add doc comment for `update_lag_buffer()` explaining shift semantics
- [ ] Add doc comment for `update_lag_buffer_from_trajectory()` with performance notes
- [ ] Add doc comment for `get_lag_residuals()` explaining return value
- [ ] Add doc comment for `set_lag_constraint_rhs()` explaining solver interaction
- [ ] Add module-level doc section on "Lag Buffer Management"
- [ ] Document the circular buffer vs shift tradeoff decision
- [ ] Update CHANGELOG.md with "Added: Lag buffer management for AR models"

## Technical Notes

### Lag Buffer Structure

```rust
// lag_buffer[hydro][lag_index]
// For AR(2): lag_buffer[h] = [Z'_{t-1}[h], Z'_{t-2}[h]]
// For AR(0): lag_buffer[h] = []
```

### Update Semantics

When updating with new residual z'\_t:

```
Before: [z'_{t-1}, z'_{t-2}, z'_{t-3}]
After:  [z'_t,     z'_{t-1}, z'_{t-2}]
```

Oldest value (z'\_{t-3}) is dropped.

### Implementation Pattern

```rust
impl UnifiedInflowModel {
    pub fn update_lag_buffer_from_trajectory(
        &mut self,
        trajectory: &[Realization],
    ) {
        for hydro in 0..self.dimension {
            let lag_order = self.ar_coefficients[hydro].len();

            if lag_order == 0 {
                continue; // Independent hydro, no lags
            }

            // Extract last p residuals
            for lag_idx in 0..lag_order {
                if let Some(past_real) = trajectory.get(trajectory.len() - lag_idx - 1) {
                    self.lag_buffer[hydro][lag_idx] = past_real.inflow_residual[hydro];
                } else {
                    // Trajectory too short, use 0.0 or error
                    self.lag_buffer[hydro][lag_idx] = 0.0;
                }
            }
        }
    }

    pub fn set_lag_constraint_rhs(
        &self,
        model: &mut solver::Model,
        constraints: &ConstraintIndices,
    ) {
        // This method will be in a later ticket after constraint infrastructure is ready
        // Placeholder for design documentation
    }
}
```

### Edge Cases

- **Insufficient trajectory**: If trajectory.len() < max_lag, pad with zeros or error
- **PreStudy nodes**: Trajectory may include multiple PreStudy realizations (PAR case)
- **Empty trajectory**: Should never happen (PreStudy always present), but handle defensively
- **Zero lag order**: Skip lag buffer operations entirely for independent hydros

### Performance Considerations

- **Memory**: O(n\*p) - pre-allocated during construction
- **Update**: O(n\*p) - must copy all lag values
- **Access**: O(1) - direct indexing
- **Optimization**: Consider circular buffer for large p to avoid copying

**Circular Buffer Tradeoff:**

- Pro: O(1) update per realization (just increment index)
- Con: More complex indexing, harder to debug
- **Decision**: Use simple shift for clarity (p is typically small, ≤ 3)

### Realization Structure Requirements

This method assumes `Realization` struct has:

- `inflow_residual: Vec<f64>` (residual space values Z'\_t)

This field will be added in TICKET-006 (Realization struct update).

## Dependencies

- **Blocked by**: TICKET-001 (needs UnifiedInflowModel struct)
- **Blocks**: TICKET-005 (realize_uncertainties needs lag buffer methods)
- **Related**: TICKET-006 (Realization struct must include inflow_residual field)

## References

- `src/subproblem.rs` - Realization struct definition
- UNIFIED_AR_ROADMAP.md - Section 1.3 (Lag Buffer Ownership)
- Example 07 inputs - Multi-node PreStudy case

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All unit tests pass
- [ ] Integration test with example trajectories passes
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Documentation builds without warnings
- [ ] Code reviewed by at least one team member
- [ ] Verified with both single and multi-node PreStudy cases
