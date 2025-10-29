# TICKET-010: Refactor State Trait Interface for Unified AR Model

**Sprint:** 3 - Cleanup  
**Phase:** 3 - Clean Up Dependencies  
**Estimated Effort:** 2 days (5 story points)  
**Confidence:** Medium  
**Status:** Not Started

## Context

With the unified inflow model handling AR dynamics through explicit constraints, the State trait interface needs simplification. Both `StorageState` and `StorageAndInflowState` are **valid and kept**, but they should no longer contain AR-specific logic.

The goal is to decouple state choice (which variables are state) from AR model representation (how dynamics are modeled). This ticket cleans up the State trait to focus on what states actually do: define state variables and handle state transitions.

## Acceptance Criteria

- [ ] Given State trait, when inflow-specific methods are removed, then AR logic is handled by UnifiedInflowModel
- [ ] Given StorageState, when implemented, then it only handles storage state variables
- [ ] Given StorageAndInflowState, when implemented, then it handles storage AND lagged inflow state variables
- [ ] Given both state types, when used with AR models, then they both use the same UnifiedInflowModel
- [ ] Given existing examples, when run, then both state types work correctly with unified AR constraints
- [ ] Performance: No impact (interface cleanup only)

## Tasks

### Implementation

- [ ] Remove `set_inflows_in_subproblem()` method from State trait
  - This method mixes concerns (state definition + inflow handling)
  - AR inflows now handled by UnifiedInflowModel
- [ ] Keep `add_constraints_to_subproblem()` but simplify interface:
  - State types only add state-specific constraints
  - AR constraints added by UnifiedInflowModel separately
- [ ] Keep `update_from_trajectory()` but clarify purpose:
  - Each state extracts its specific state variables from trajectory
  - StorageState: extracts previous storage
  - StorageAndInflowState: extracts previous storage AND lagged inflows
- [ ] Update StorageState implementation:
  - Remove any AR-specific logic
  - Focus on storage state variables only
- [ ] Update StorageAndInflowState implementation:
  - Remove AR constraint generation (now in UnifiedInflowModel)
  - Keep lag state variable tracking
  - Focus on connecting state variables to LP
- [ ] Add clear documentation distinguishing:
  - **State choice**: What variables are treated as state
  - **AR representation**: How dynamics are modeled (always via UnifiedInflowModel)

### Testing

- [ ] Unit test: StorageState with independent inflows (AR(0))
- [ ] Unit test: StorageState with AR(1) inflows
- [ ] Unit test: StorageAndInflowState with AR(2) inflows
- [ ] Unit test: Verify State trait has no AR-specific methods
- [ ] Integration test: Run example 06 with StorageState
- [ ] Integration test: Run example 07 with StorageAndInflowState
- [ ] Regression test: Compare outputs with previous implementation

### Documentation

- [ ] Update State trait doc comment explaining separation of concerns
- [ ] Document when to use StorageState vs StorageAndInflowState:
  - StorageState: Simpler, smaller state space, faster
  - StorageAndInflowState: Richer cuts, potentially better convergence
- [ ] Add examples showing both state types work with unified AR model
- [ ] Update architecture docs to clarify state vs AR representation
- [ ] Update CHANGELOG.md with "Changed: State trait simplified, decoupled from AR representation"

## Technical Notes

### State Trait Simplification

**Before (Mixed Concerns):**

```rust
pub trait State {
    fn add_constraints_to_subproblem(...);  // Includes AR logic
    fn set_inflows_in_subproblem(...);      // AR-specific, shouldn't be here
    fn update_from_trajectory(...);          // Mixed storage + AR
}
```

**After (Clean Separation):**

```rust
pub trait State {
    fn add_state_constraints(...);           // Only state-specific constraints
    fn update_from_trajectory(...);          // Only state variable extraction
    fn extract_state_variables(...);         // Get state from realization
}

// AR logic moved to:
impl UnifiedInflowModel {
    fn add_ar_constraints(...);              // AR dynamics for ALL states
}
```

### What Each State Type Does

**StorageState:**

- **State variables**: Previous storage volumes
- **Constraints**: Links previous storage to current storage (hydro balance)
- **Cut generation**: Gradient w.r.t. storage state only
- **Use case**: Simpler problems, faster solves, standard SDDP

**StorageAndInflowState:**

- **State variables**: Previous storage + lagged inflows (p lags per hydro)
- **Constraints**: Links storage AND lagged inflows to current state
- **Cut generation**: Gradient w.r.t. storage AND inflow lags
- **Use case**: Better cuts for AR models, potentially better convergence

**Key Insight**: Both use the SAME UnifiedInflowModel for AR dynamics!

### How AR Constraints Work with Both States

**With StorageState:**

```rust
// State defines: storage[t-1] → storage[t]
// AR model defines: Z'[t] = φ*Z'[t-1] + ε[t]
// LP has BOTH sets of constraints, independent
```

**With StorageAndInflowState:**

```rust
// State defines: storage[t-1], Z'[t-1] → storage[t], Z'[t]
// AR model defines: Z'[t] = φ*Z'[t-1] + ε[t]  (same!)
// State variables can be used in cuts for better gradient info
```

### Why This Is Better Than Before

**Old Approach (Conditional):**

- StorageState: Generate AR constraints
- StorageAndInflowState: Handle AR via trajectory preprocessing
- Result: Two different code paths, complex conditionals

**New Approach (Unified):**

- Both states: AR constraints always via UnifiedInflowModel
- Difference: What goes into state space (affects cuts, not dynamics)
- Result: Single AR handling, state choice independent

### Interface Changes

**Removed Methods:**

```rust
// FROM State trait - mixed concerns
fn set_inflows_in_subproblem(&self, model: &mut Model, inflows: &[f64]);
```

**Kept Methods:**

```rust
// State trait - clean, focused on state definition
fn add_state_constraints(&self, pb: &mut Problem, vars: &Variables);
fn update_from_trajectory(&self, trajectory: &[&Realization]);
fn extract_state_variables(&self, realization: &Realization) -> StateVector;
```

**New Pattern:**

```rust
impl Subproblem {
    fn build_model(&mut self) {
        // Add physical constraints
        self.add_physical_constraints();

        // Add AR constraints (always, via UnifiedInflowModel)
        self.inflow_model.add_ar_constraints(&mut self.model);

        // Add state-specific constraints
        self.state.add_state_constraints(&mut self.model);
    }
}
```

### Edge Cases

- **State choice independence**: AR model should work identically with both states
- **Cut generation**: StorageAndInflowState cuts include lag gradients, StorageState doesn't
- **Performance**: StorageAndInflowState may converge better but has larger state space
- **Backward compatibility**: Both state types remain available to users

### Files to Update

Based on the corrected understanding:

- `src/state.rs` - Simplify State trait interface, update both implementations
- `src/subproblem.rs` - Ensure AR constraints added consistently for both states
- `tests/test_sddp_algorithm.rs` - Test both state types with AR
- `examples/06-par-model/` - Uses StorageState (verify)
- `examples/07-par-model-with-inflow-state/` - Uses StorageAndInflowState (verify)

### Performance Impact

**Before:**

- Different AR handling for each state type
- Conditional branches based on state

**After:**

- Unified AR handling
- No conditionals on state type
- **Expected impact:** No change or slight improvement from better code locality

## Dependencies

- **Blocked by**:
  - TICKET-008 (realize_uncertainties must work with both states)
  - TICKET-009 (update_from_trajectory must work with both states)
- **Blocks**: TICKET-012 (tests need simplified interface)
- **Related**: TICKET-007 (Subproblem integration with UnifiedInflowModel)

## References

- `src/state.rs` - State trait and implementations
- `src/subproblem.rs` - Subproblem integration with states
- UNIFIED_AR_ROADMAP_REVISED.md - Corrected architecture understanding
- Example 06 - StorageState usage
- Example 07 - StorageAndInflowState usage

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Example 06 runs successfully (StorageState)
- [ ] Example 07 runs successfully (StorageAndInflowState)
- [ ] Both state types produce correct results with AR models
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] State trait interface is clean and focused
- [ ] No AR-specific logic in State implementations
- [ ] Documentation updated explaining state choice vs AR representation
- [ ] CHANGELOG.md updated
- [ ] Code reviewed by at least one team member
