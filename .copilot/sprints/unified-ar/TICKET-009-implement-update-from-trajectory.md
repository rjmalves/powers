# TICKET-009: Implement update_from_trajectory() for Unified Model

**Sprint:** 2 - Subproblem Refactor  
**Phase:** 2 - Refactor Subproblem  
**Estimated Effort:** 2 days (5 story points)  
**Confidence:** High  
**Status:** Not Started

## Context

The `update_from_trajectory()` method transfers state information from past realizations to the current subproblem during forward passes. With the unified model, this method needs to:

1. Update lag buffer from trajectory (via UnifiedInflowModel)
2. Update storage state from previous realization (unchanged)
3. Update constraint RHS values in solver model

This ticket removes state-type conditionals and uses the clean UnifiedInflowModel interface.

## Acceptance Criteria

- [ ] Given a trajectory of past realizations, when update_from_trajectory() is called, then lag buffer is correctly updated
- [ ] Given a trajectory with PreStudy nodes, when update_from_trajectory() is called, then initial lags are extracted correctly
- [ ] Given AR(p) model with trajectory of length ≥ p, when update_from_trajectory() is called, then last p residuals are used
- [ ] Given independent model (p=0), when update_from_trajectory() is called, then no lag updates occur (no-op)
- [ ] Given updated lag buffer, when solver model RHS is updated, then lag constraint RHS reflects current lags
- [ ] Performance: update_from_trajectory() should be O(n\*p) with no unnecessary allocations

## Tasks

### Implementation

- [ ] Update `update_from_trajectory()` method in Subproblem or State:
  - Remove state-type conditional logic
  - Use UnifiedInflowModel methods consistently
- [ ] Implement lag buffer update:
  - Call `self.inflow_model.update_lag_buffer_from_trajectory(trajectory)`
  - Extracts last p residuals per hydro from trajectory
- [ ] Implement lag constraint RHS update:
  - Call `self.inflow_model.set_lag_constraint_rhs(model, &self.constraints)`
  - Updates solver model RHS with current lag values
- [ ] Implement storage state update (unchanged from current):
  - Extract last realization's stored_volume
  - Update hydro_balance constraint RHS
- [ ] Handle PreStudy trajectory correctly:
  - Trajectory format: [PreStudy(-p), ..., PreStudy(0), Stage(1), ..., Stage(t-1)]
  - Extract residuals from appropriate PreStudy nodes for initialization
- [ ] Add validation:
  - Check trajectory length ≥ 1 (at least PreStudy present)
  - For AR(p), warn if trajectory length < p+1 (insufficient history)

### Testing

- [ ] Unit test: update_from_trajectory with single PreStudy node (standard case)
- [ ] Unit test: update_from_trajectory with multi-node PreStudy (PAR case)
- [ ] Unit test: update_from_trajectory for AR(1) with sufficient trajectory
- [ ] Unit test: update_from_trajectory for AR(2) with sufficient trajectory
- [ ] Unit test: update_from_trajectory for independent model (verify no-op)
- [ ] Unit test: update_from_trajectory with insufficient trajectory (< p realizations)
- [ ] Unit test: Verify lag constraint RHS updated in solver model
- [ ] Unit test: Verify hydro balance RHS updated for storage
- [ ] Integration test: Full forward pass with trajectory updates at each stage
- [ ] Regression test: Compare lag buffer values with manual calculation

### Documentation

- [ ] Add doc comment to update_from_trajectory() explaining:
  - Trajectory structure and ordering
  - PreStudy node handling
  - Lag buffer update semantics
  - Storage state update
- [ ] Add inline comments for trajectory indexing logic
- [ ] Add example showing trajectory structure for multi-stage problem
- [ ] Update module-level docs with trajectory update flow
- [ ] Update CHANGELOG.md with "Changed: update_from_trajectory uses UnifiedInflowModel"

## Technical Notes

### Implementation Pattern

```rust
impl Subproblem {
    pub fn update_from_trajectory(
        &mut self,
        past_realizations: &[&Realization],
    ) -> Result<(), String> {
        let model = self.model.as_mut()
            .ok_or("Model not initialized")?;

        // Validation: trajectory must have at least PreStudy
        if past_realizations.is_empty() {
            return Err("Empty trajectory".to_string());
        }

        // STEP 1: Update lag buffer from trajectory
        // Convert &[&Realization] to &[Realization] if needed
        self.inflow_model.update_lag_buffer_from_trajectory(past_realizations);

        // STEP 2: Update lag constraint RHS in solver model
        self.inflow_model.set_lag_constraint_rhs(model, &self.constraints)?;

        // STEP 3: Update storage state (unchanged logic)
        let last_realization = past_realizations.last()
            .ok_or("Empty trajectory")?;

        for hydro in 0..self.system.n_hydros() {
            let constraint_idx = self.constraints.hydro_balance[hydro];
            let storage = last_realization.stored_volume[hydro];
            // Update RHS: inflow + storage_{t-1} = turbined + spillage + storage_t
            model.change_rhs(constraint_idx, storage, storage);
        }

        Ok(())
    }
}
```

### Trajectory Structure

**Standard Case (single PreStudy node):**

```
Stage 1: [PreStudy(0)]
Stage 2: [PreStudy(0), Stage(1)]
Stage 3: [PreStudy(0), Stage(1), Stage(2)]
```

**PAR Case (multi-node PreStudy for AR initialization):**

```
Stage 1: [PreStudy(-3), PreStudy(-2), PreStudy(-1), PreStudy(0)]
Stage 2: [PreStudy(-3), PreStudy(-2), PreStudy(-1), PreStudy(0), Stage(1)]
Stage 3: [PreStudy(-3), PreStudy(-2), PreStudy(-1), PreStudy(0), Stage(1), Stage(2)]
```

For AR(p), need last p realizations:

- AR(1): uses [trajectory[len-1]]
- AR(2): uses [trajectory[len-1], trajectory[len-2]]
- AR(3): uses [trajectory[len-1], trajectory[len-2], trajectory[len-3]]

### Lag Buffer Indexing

UnifiedInflowModel.update_lag_buffer_from_trajectory handles indexing:

```rust
// For AR(p), extract last p residuals
for lag_idx in 0..p {
    let realization_idx = trajectory.len() - lag_idx - 1;
    lag_buffer[hydro][lag_idx] = trajectory[realization_idx].inflow_residual[hydro];
}
```

Result: `lag_buffer[hydro] = [Z'_{t-1}, Z'_{t-2}, ..., Z'_{t-p}]`

### Setting Lag Constraint RHS

For AR constraint: `Z'_t - Σ(φ_k * Z'_{t-k}) = ε_t`

The lag variables Z'\_{t-k} are fixed by setting their constraint RHS. Wait, this needs clarification...

**Clarification Needed:** How are lag values transferred?

**Option A: Lag variables as parameters (preferred):**

- Lag variables are regular variables in LP
- Their values come from previous solve (trajectory)
- AR constraint includes lag variables with coefficients
- No need to "set" lag values, they're in the solution

**Option B: Lag constraints (alternative):**

- Add separate equality constraints: Z'\_{t-k} = lag_value_k
- Update these constraint RHS values with lag buffer
- More constraints but clearer semantics

**Decision for this ticket:** Use Option A (lag variables, no separate constraints). Lag values are used in AR constraint coefficients but don't need separate fixing.

**Revised understanding:** Lag buffer is used to:

1. Initialize LP variable values (warm start) - optional
2. Provide state for cut generation (used in backward pass)

The AR constraint in LP doesn't actually reference lag _variables_ - it references the current residual Z'\_t and innovation ε_t. The lag relationship is enforced by the algorithm (forward pass updates lag buffer from solutions).

**Correction:** Need to re-examine constraint formulation. This might need adjustment in TICKET-002.

### Edge Cases

- **Empty trajectory**: Should never happen (PreStudy always present), but handle defensively
- **Insufficient trajectory**: Trajectory length < p+1 (not enough history for AR(p))
  - Option 1: Pad with zeros
  - Option 2: Use available history only
  - Option 3: Error
  - **Decision:** Pad with zeros and log warning
- **No PreStudy**: Invalid, should error
- **Mixed lag orders**: Some hydros need more history than others

### Performance Considerations

- **Trajectory slicing**: O(p) per hydro, unavoidable
- **RHS updates**: O(n) for storage, O(n\*p) for lags
- **Total complexity**: O(n\*p) dominated by lag buffer update
- **Optimization**: Batch RHS updates if solver supports it

## Dependencies

- **Blocked by**:
  - TICKET-003 (needs lag buffer methods)
  - TICKET-007 (needs Subproblem integration)
  - TICKET-008 (realize_uncertainties must work first)
- **Blocks**: TICKET-012 (cut generation needs updated trajectories)
- **Related**: TICKET-006 (Realization must have inflow_residual field)

## References

- Current `src/state.rs` lines 41-76 - update_from_trajectory documentation
- `src/subproblem.rs` - Realization trajectory structure
- UNIFIED_AR_ROADMAP.md - Section 1.1 (Unified Inflow Model lag buffer management)

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All unit tests pass
- [ ] Integration test with full forward pass passes
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Lag buffer values match manual calculation
- [ ] Storage state updates correctly
- [ ] Documentation builds without warnings
- [ ] Code reviewed by at least one team member
- [ ] Verified with both single and multi-node PreStudy
