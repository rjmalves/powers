# TICKET-007: Integrate UnifiedInflowModel into Subproblem Construction

**Sprint:** 2 - Subproblem Refactor  
**Phase:** 2 - Refactor Subproblem  
**Estimated Effort:** 2 days (5 story points)  
**Confidence:** Medium  
**Status:** Not Started

## Context

This ticket integrates the UnifiedInflowModel into the Subproblem struct, replacing the state-based inflow handling. The Subproblem will now own a UnifiedInflowModel instance and use it for adding variables and constraints during construction. This is a critical integration point that removes the State trait dependency for inflow modeling.

## Acceptance Criteria

- [ ] Given a Subproblem, when constructed, then it contains a UnifiedInflowModel instance
- [ ] Given a Subproblem with UnifiedInflowModel, when variables are added, then both observation and residual space inflow variables are created
- [ ] Given a Subproblem with UnifiedInflowModel, when constraints are added, then AR dynamics and transformation constraints are created
- [ ] Given a Subproblem, when solved, then it uses unified model for all inflow operations
- [ ] Given existing examples, when run with new code, then they still compile (may not solve correctly yet)
- [ ] Performance: Subproblem construction time should not increase

## Tasks

### Implementation

- [ ] Update `Subproblem` struct in `src/subproblem.rs`:
  - Add `inflow_model: UnifiedInflowModel` field
  - Keep `state: Box<dyn State>` temporarily (for storage only, remove inflow logic later)
- [ ] Update `Subproblem::new()` or construction method:
  - Create UnifiedInflowModel from unified_specs
  - Pass seasonal_params as Arc
  - Initialize inflow_model field
- [ ] Add method `add_inflow_variables()`:
  - Creates observation space variables (Y_t): `inflow: Vec<usize>`
  - Creates residual space variables (Z'\_t): `inflow_residual: Vec<usize>`
  - Creates lag variables (Z'\_{t-k}): `lag_residual: Vec<Vec<usize>>`
  - Creates innovation variables (ε_t): `innovation: Vec<usize>`
  - Sets appropriate bounds (typically -∞ to +∞ for residuals)
  - Returns indices to populate Variables struct
- [ ] Update `add_constraints()` method:
  - Call `inflow_model.add_constraints_to_lp()` after physical constraints
  - Populate Constraints struct with returned indices
- [ ] Update variable creation to match new Variables struct (from TICKET-004)
- [ ] Ensure subproblem cloning works with UnifiedInflowModel (implement Clone)
- [ ] Add validation: verify variable and constraint counts are consistent

### Testing

- [ ] Unit test: Construct Subproblem with independent inflow model
- [ ] Unit test: Construct Subproblem with AR(1) inflow model
- [ ] Unit test: Construct Subproblem with mixed AR orders
- [ ] Unit test: Verify inflow variables created for all hydros
- [ ] Unit test: Verify residual variables created for all hydros
- [ ] Unit test: Verify lag variables created only for AR hydros
- [ ] Unit test: Verify AR dynamics constraints added for all hydros
- [ ] Unit test: Verify transformation constraints added for all hydros
- [ ] Integration test: Build complete subproblem and verify solvability (simple problem)
- [ ] Regression test: Run example 06 (should compile, may not solve correctly)

### Documentation

- [ ] Add doc comment to inflow_model field explaining ownership
- [ ] Add doc comment to add_inflow_variables() explaining variable creation
- [ ] Update Subproblem struct doc comment to mention unified inflow model
- [ ] Add inline comments explaining constraint ordering
- [ ] Update CHANGELOG.md with "Changed: Subproblem uses UnifiedInflowModel for inflow dynamics"

## Technical Notes

### Integration Pattern

```rust
pub struct Subproblem {
    pub model: Option<solver::Model>,
    pub variables: Variables,
    pub constraints: Constraints,

    // NEW: Unified inflow model
    pub inflow_model: UnifiedInflowModel,

    // TEMPORARY: Keep state for storage only
    pub state: Box<dyn State>,

    // ... other fields ...
}

impl Subproblem {
    fn add_inflow_variables(
        &self,
        pb: &mut solver::Problem,
    ) -> (Vec<usize>, Vec<usize>, Vec<Vec<usize>>, Vec<usize>) {
        let n_hydros = self.inflow_model.dimension();

        // Observation space: Y_t (for hydro balance)
        let mut inflow_obs = Vec::with_capacity(n_hydros);
        for h in 0..n_hydros {
            let idx = pb.add_col(0.0, 0.0..=f64::INFINITY);
            inflow_obs.push(idx);
        }

        // Residual space: Z'_t (for AR dynamics)
        let mut inflow_res = Vec::with_capacity(n_hydros);
        for h in 0..n_hydros {
            let idx = pb.add_col(0.0, f64::NEG_INFINITY..=f64::INFINITY);
            inflow_res.push(idx);
        }

        // Lag residuals: Z'_{t-k} (for AR dynamics)
        let mut lag_res = Vec::with_capacity(n_hydros);
        for h in 0..n_hydros {
            let lag_order = self.inflow_model.lag_order(h);
            let mut lags = Vec::with_capacity(lag_order);
            for _ in 0..lag_order {
                let idx = pb.add_col(0.0, f64::NEG_INFINITY..=f64::INFINITY);
                lags.push(idx);
            }
            lag_res.push(lags);
        }

        // Innovations: ε_t (for AR dynamics RHS)
        let mut innovations = Vec::with_capacity(n_hydros);
        for h in 0..n_hydros {
            let idx = pb.add_col(0.0, f64::NEG_INFINITY..=f64::INFINITY);
            innovations.push(idx);
        }

        (inflow_obs, inflow_res, lag_res, innovations)
    }
}
```

### Variable Bounds

| Variable          | Bounds   | Rationale                            |
| ----------------- | -------- | ------------------------------------ |
| Y_t (observation) | [0, +∞)  | Physical inflow must be non-negative |
| Z'\_t (residual)  | (-∞, +∞) | Normalized, can be negative          |
| Z'\_{t-k} (lag)   | (-∞, +∞) | Normalized, can be negative          |
| ε_t (innovation)  | (-∞, +∞) | White noise, can be negative         |

Note: Lag variables will have their values fixed via constraint RHS, but bounds should still be unbounded for formulation clarity.

### Constraint Ordering

Recommended order for debugging and solver efficiency:

1. Load balance constraints
2. Hydro balance constraints
3. Inflow transformation constraints (Y = μ + σZ')
4. AR dynamics constraints (Z' = ΣφZ'\_lag + ε)
5. Cuts (added dynamically)

### Edge Cases

- **All independent**: lag_residual is empty Vec for all hydros
- **Single hydro**: dimension = 1, should create 1 variable per type
- **Zero lag order**: lag_residual[hydro] is empty Vec
- **Construction failure**: If UnifiedInflowModel can't be created, fail early with clear error

### Migration Strategy

**Phase 1 (This Ticket):**

- Add UnifiedInflowModel alongside State
- State still handles storage state
- Inflow logic moves to UnifiedInflowModel

**Phase 2 (Later Tickets):**

- Remove inflow methods from State trait
- Simplify State to just storage
- Eventually remove State trait entirely

### Performance Considerations

- **Variable creation**: O(n\*p) unavoidable
- **Memory**: Additional ~4n indices (inflow_obs, inflow_res, innovations, alpha_per_lag)
- **Construction time**: Should add < 5% overhead
- **Solving time**: Unchanged (same constraints, different organization)

## Dependencies

- **Blocked by**:
  - TICKET-001 (needs UnifiedInflowModel)
  - TICKET-002 (needs constraint generation method)
  - TICKET-004 (needs updated Variables struct)
  - TICKET-005 (needs updated Constraints struct)
- **Blocks**: TICKET-008 (realize_uncertainties refactor needs this integration)
- **Related**: TICKET-006 (Realization struct must support residuals)

## References

- `src/subproblem.rs` - Current Subproblem implementation
- `src/solver.rs` - Problem and Model interfaces
- UNIFIED_AR_ROADMAP.md - Section 1.2 (Simplified Subproblem Interface)

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] Unit tests pass
- [ ] Example 06 compiles (correctness comes later)
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Variable count validation passes
- [ ] Constraint count validation passes
- [ ] Documentation builds without warnings
- [ ] Code reviewed by at least one team member
