# TICKET-005: Refactor Constraints Struct for Unified AR Model

**Sprint:** 2 - Subproblem Refactor  
**Phase:** 2 - Refactor Subproblem  
**Estimated Effort:** 1 day (3 story points)  
**Confidence:** High  
**Status:** Not Started

## Context

Similar to the Variables refactor, the `Constraints` struct needs to reflect the new unified AR model. This ticket reorganizes constraint indices to separate physical constraints from inflow model constraints, making the architecture clearer and eliminating the need for conditional constraint identification.

## Acceptance Criteria

- [ ] Given a Constraints struct, when accessed, then it clearly separates physical vs inflow model constraints
- [ ] Given a Constraints struct, when accessed, then it has `ar_dynamics: Vec<usize>` for AR constraints
- [ ] Given a Constraints struct, when accessed, then it has `inflow_transform: Vec<usize>` for Y = μ + σZ' constraints
- [ ] Given existing code using Constraints, when refactored struct is used, then physical constraints (load_balance, hydro_balance) are unchanged
- [ ] Given constraint creation in subproblem, when new fields are populated, then indices reference correct constraint rows
- [ ] Performance: Constraint struct size should not increase unnecessarily

## Tasks

### Implementation

- [ ] Update `Constraints` struct in `src/subproblem.rs`:
  - Keep existing fields: load_balance, hydro_balance, cuts
  - Add `inflow_transform: Vec<usize>` field with comment `// Y_t = μ_s + σ_s * Z'_t`
  - Add `ar_dynamics: Vec<usize>` field with comment `// Z'_t = Σφ_k Z'_{t-k} + ε_t`
  - Remove obsolete `inflow_process: Vec<Vec<usize>>` field
- [ ] Update `Constraints` construction to initialize new fields:
  - Set `inflow_transform` and `ar_dynamics` to empty Vec initially
  - Populated during UnifiedInflowModel.add_constraints_to_lp()
- [ ] Add helper methods to Constraints:
  - `num_inflow_constraints(&self) -> usize` - returns inflow_transform.len()
  - `has_ar_dynamics(&self) -> bool` - returns !ar_dynamics.is_empty()
- [ ] Update any code referencing `inflow_process` field
- [ ] Ensure Constraints remains `Clone` derivable

### Testing

- [ ] Unit test: Create Constraints struct with new fields, verify all indices accessible
- [ ] Unit test: Verify num_inflow_constraints() returns correct count
- [ ] Unit test: Verify has_ar_dynamics() correctly identifies presence of AR constraints
- [ ] Unit test: Clone Constraints struct, verify all fields copied correctly
- [ ] Compilation test: Verify existing code still compiles after changes
- [ ] Integration test: Create subproblem with new Constraints struct

### Documentation

- [ ] Add doc comment to Constraints struct explaining organization
- [ ] Document each new field with clear purpose
- [ ] Add inline comment explaining separation of physical vs inflow constraints
- [ ] Add example to module docs showing constraint structure for AR case
- [ ] Update CHANGELOG.md with "Changed: Constraints struct for unified AR representation"

## Technical Notes

### Constraint Structure Design

```rust
#[derive(Clone)]
pub struct Constraints {
    // Physical constraints
    pub load_balance: Vec<usize>,
    pub hydro_balance: Vec<usize>,

    // Inflow model constraints
    pub inflow_transform: Vec<usize>,  // Y_t = μ_s + σ_s * Z'_t (one per hydro)
    pub ar_dynamics: Vec<usize>,       // Z'_t = Σφ_k Z'_{t-k} + ε_t (one per hydro)

    // Cut constraints (added dynamically)
    pub cuts: Vec<usize>,
}
```

### Constraint Semantics

**inflow_transform[h]:**

- Maps residual space to observation space
- RHS = μ_s[h] (seasonal mean)
- Coefficient on Z'\_t = -σ_s[h] (seasonal std dev)
- Always present for every hydro

**ar_dynamics[h]:**

- Enforces autoregressive relationship
- RHS = ε_t[h] (innovation, set at solve time)
- Coefficients on Z'\_{t-k} = -φ_k[h]
- Always present (even for independent case with empty φ)

### Migration Notes

**Removed Field:**

- `inflow_process: Vec<Vec<usize>>` - was multi-dimensional, unclear semantics

**Why Removal is Safe:**

- Used only in state-specific code paths that will be rewritten
- New structure is clearer: one constraint per hydro per type
- Grep search shows limited usage pattern

### Design Rationale

**Why Separate Fields Instead of Nested Vec?**

Old (ambiguous):

```rust
inflow_process: Vec<Vec<usize>>  // What does inner Vec represent?
```

New (explicit):

```rust
inflow_transform: Vec<usize>  // Clear: one per hydro
ar_dynamics: Vec<usize>       // Clear: one per hydro
```

**Benefits:**

- Self-documenting structure
- No magic number checks (e.g., `if constraints.inflow_process[0].len() <= 2`)
- Type system enforces correct usage
- Easier to extend (e.g., add lag transfer constraints later if needed)

### Edge Cases

- **Empty constraints**: Should never happen after proper initialization, but handle defensively
- **Mismatched sizes**: ar_dynamics.len() should equal inflow_transform.len() should equal n_hydros
- **Constraint indexing**: Ensure indices are valid row numbers in solver model

## Dependencies

- **Blocked by**: None (pure data structure change)
- **Blocks**: TICKET-002 (constraint generation populates these fields)
- **Related**: TICKET-005 (realize_uncertainties uses new structure), TICKET-004 (Variables refactor)

## References

- Current `src/subproblem.rs` lines 114-119 - Constraints struct definition
- UNIFIED_AR_ROADMAP.md - Section 1.4 (Constraint Organization)

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All existing tests still pass
- [ ] New unit tests for Constraints pass
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] No regressions in example runs
- [ ] Documentation builds without warnings
- [ ] Code reviewed by at least one team member
