# TICKET-004: Refactor Variables Struct for Dual Space Representation

**Sprint:** 2 - Subproblem Refactor  
**Phase:** 2 - Refactor Subproblem  
**Estimated Effort:** 2 days (5 story points)  
**Confidence:** High  
**Status:** Not Started

## Context

The current `Variables` struct needs to support dual space representation: both observation space (Y_t for physical constraints) and residual space (Z'\_t for AR dynamics). Additionally, it needs to handle optional lagged inflow state variables (only present when using `StorageAndInflowState`).

This ticket adds the new variable indices needed for the unified AR model while maintaining backward compatibility with existing physical variables and supporting both state type choices.

This is a pure data structure change that enables subsequent refactoring without breaking existing code.

## Acceptance Criteria

- [ ] Given a Variables struct, when accessed, then it has both `inflow` (observation) and `inflow_residual` (residual) fields
- [ ] Given a Variables struct, when accessed, then it has `lag_residual: Vec<Vec<usize>>` for AR lag variables
- [ ] Given a Variables struct, when accessed, then it has `innovation: Vec<usize>` for white noise variables
- [ ] Given existing code using Variables, when refactored struct is used, then physical variables (deficit, thermal_gen, etc.) are unchanged
- [ ] Given variable creation in subproblem, when new fields are populated, then indices are sequential and correct
- [ ] Performance: Variable struct size should not increase unnecessarily (use Vec not HashMap)

## Tasks

### Implementation

- [ ] Update `Variables` struct in `src/subproblem.rs`:
  - Keep existing fields: deficit, thermal_gen, turbined_flow, spillage, stored_volume, alpha
  - Rename current `inflow` to document observation space: add comment `// Observation space (for hydro balance)`
  - Add `inflow_residual: Vec<usize>` field with comment `// Residual space (for AR dynamics)`
  - Add `innovation: Vec<usize>` field with comment `// ε_t white noise`
  - Add `lagged_inflow_state: Option<Vec<Vec<usize>>>` field with comment `// [hydro][lag] - Only if StorageAndInflowState`
  - Remove obsolete `inflow_process: Vec<Vec<usize>>` field (replaced by new structure)
- [ ] Update `Variables` construction to initialize new fields:
  - Set `lagged_inflow_state` to None initially (Some(...) only if StorageAndInflowState)
  - Set `innovation` to empty Vec initially (populated during constraint generation)
  - Keep `inflow_residual` same size as `inflow`
- [ ] Add helper methods to Variables:
  - `has_lagged_inflow_state(&self) -> bool` - returns self.lagged_inflow_state.is_some()
  - `num_inflow_lags(&self, hydro: usize) -> usize` - returns lag count if state lags exist
- [ ] Update all references to `inflow_process` field (should be minimal at this stage)
- [ ] Ensure Variables remains `Clone` derivable

### Testing

- [ ] Unit test: Create Variables struct with new fields, verify all indices accessible
- [ ] Unit test: Verify has_lagged_inflow_state() returns false when None
- [ ] Unit test: Verify has_lagged_inflow_state() returns true when Some
- [ ] Unit test: Verify num_inflow_lags() returns correct count
- [ ] Unit test: Clone Variables struct, verify all fields copied correctly
- [ ] Unit test: Variables with lagged_inflow_state = None (StorageState case)
- [ ] Unit test: Variables with lagged_inflow_state = Some(...) (StorageAndInflowState case)
- [ ] Compilation test: Verify existing code referencing old fields still compiles
- [ ] Integration test: Create subproblem with new Variables struct

### Documentation

- [ ] Add doc comment to Variables struct explaining dual space representation
- [ ] Document each new field with clear space annotation (observation vs residual)
- [ ] Add inline comment explaining why both inflow and inflow_residual are needed
- [ ] Add example to module docs showing variable structure for AR(2) case
- [ ] Update CHANGELOG.md with "Changed: Variables struct for dual space AR representation"

## Technical Notes

### Variable Structure Design

```rust
#[derive(Clone)]
pub struct Variables {
    // Physical variables (observation space)
    pub deficit: Vec<usize>,
    pub thermal_gen: Vec<usize>,
    pub turbined_flow: Vec<usize>,
    pub spillage: Vec<usize>,
    pub stored_volume: Vec<usize>,

    // Inflow variables (dual representation)
    pub inflow: Vec<usize>,           // Observation space Y_t (for hydro balance)
    pub inflow_residual: Vec<usize>,  // Residual space Z'_t (for AR dynamics)

    // AR model variables
    pub innovation: Vec<usize>,        // ε_t white noise

    // State-dependent variables (only if StorageAndInflowState)
    pub lagged_inflow_state: Option<Vec<Vec<usize>>>,  // [hydro][lag] state variables

    // Future cost
    pub alpha: usize,
}
```

### Lagged Inflow State Variables

**Key Distinction:**

- `lagged_inflow_state`: **State variables** (only if StorageAndInflowState)
  - These are decision variables that appear in cuts
  - Only present when user chooses StorageAndInflowState
  - Set to None when using StorageState

The AR dynamics (via UnifiedInflowModel) always exist regardless of state choice, but lagged inflows may or may not be state variables.

### Why Dual Representation?

**Observation Space (Y_t):**

- Used in hydro balance: inflow + turbined = stored_volume + spillage
- Physical units (m³/s or MWh)
- Must match system bounds and capacities

**Residual Space (Z'\_t):**

- Used in AR dynamics: Z'_t = Σφ_k Z'_{t-k} + ε_t (in UnifiedInflowModel)
- Normalized, zero-mean
- Makes AR coefficients stationary and stable

**Transformation:**

```
Y_t = μ_s + σ_s * Z'_t  (via LP constraint)
```

**State Variables (Optional):**

- If StorageState: Only storage is state, lags tracked internally by UnifiedInflowModel
- If StorageAndInflowState: Storage AND lagged inflows are state variables (better cuts)

### Migration Notes

**Removed Field:**

- `inflow_process: Vec<Vec<usize>>` - was used for conditional logic based on state type

**Why Removal is Safe:**

- Only used in `realize_uncertainties()` which will be completely rewritten (TICKET-005)
- No other code should reference this field directly
- Grep search confirms limited usage (see Context analysis)

### Edge Cases

- **Zero lags**: lag_residual[hydro] is empty Vec for independent hydros
- **Variable lag orders**: lag_residual vectors can have different lengths
- **Indexing**: Ensure lag_residual[hydro][lag] doesn't panic on empty vectors

### Memory Considerations

**Old Structure:**

```
inflow_process: Vec<Vec<usize>>  // ~(n * 2..p+2) usizes
```

**New Structure:**

```
inflow_residual: Vec<usize>                      // n usizes
innovation: Vec<usize>                            // n usizes
lagged_inflow_state: Option<Vec<Vec<usize>>>     // None or ~(n * p) usizes
```

**Net change:** Similar or slightly more memory, but clearer semantics and supports both state types.

## Dependencies

- **Blocked by**: None (pure data structure change)
- **Blocks**: TICKET-002 (constraint generation needs new variables)
- **Related**: TICKET-005 (realize_uncertainties uses new structure)

## References

- Current `src/subproblem.rs` lines 98-111 - Variables struct definition
- UNIFIED_AR_ROADMAP.md - Section 1.3 (Clean Variable Structure)

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All existing tests still pass
- [ ] New unit tests for Variables pass
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] No regressions in example runs
- [ ] Documentation builds without warnings
- [ ] Code reviewed by at least one team member
