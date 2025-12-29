# [T-041] Consolidate StorageState Methods

> **Epic**: [Epic 4: State Simplification](../00-epic-overview.md)
> **Sprint**: [Sprint 1: State Consolidation](./00-sprint-overview.md)
> **Dependencies**: [T-040](./ticket-040-extract-state-utilities.md)
> **Blocks**: [T-043](./ticket-043-pool-compatible-extensions.md)

## Files to Read Before Starting

- `src/state.rs` - StorageState implementation (lines 643-993)
- `src/state/common.rs` (or equivalent) - StateCore from T-040
- T-038 analysis results

---

## Context

### Background

With `StateCore` extracted (T-040), we now refactor `StorageState` to use composition, delegating common methods to the embedded core struct. This eliminates code duplication while preserving exact behavior.

### Current State

```rust
pub struct StorageState {
    dimension: usize,
    state_coefficients: Vec<f64>,
    dominating_objective: f64,
    dominating_cut_id: usize,
    iteration: usize,
    forward_pass_idx: usize,
}
```

### Target State

```rust
pub struct StorageState {
    core: StateCore,  // Contains all common fields
}
```

---

## Specification

### Changes Required

1. **Replace fields with `StateCore` embed**
2. **Update constructor** to use `StateCore::new()`
3. **Delegate common trait methods** to `self.core`
4. **Keep type-specific methods** (e.g., `extract_storage_from_trajectory`)

### Behavioral Invariant

**All numerical outputs must remain bit-for-bit identical.** This is a structural refactoring only.

---

## Acceptance Criteria

- [ ] `StorageState` uses `StateCore` composition
- [ ] All common methods delegate to `core`
- [ ] Type-specific methods preserved
- [ ] All existing `StorageState` tests pass
- [ ] Golden tests pass
- [ ] No performance regression

---

## Implementation Guide

### Suggested Approach

1. **Update struct definition**:
   ```rust
   pub struct StorageState {
       core: StateCore,
   }
   ```

2. **Update constructor**:
   ```rust
   impl StorageState {
       pub fn new(system: &system::System) -> Self {
           Self {
               core: StateCore::new(system.meta.hydros_count),
           }
       }
   }
   ```

3. **Delegate common trait methods**:
   ```rust
   impl State for StorageState {
       fn coefficients(&self) -> &[f64] {
           self.core.coefficients()
       }
       
       fn update_coefficients(&mut self, coefficients: &[f64]) {
           self.core.update_coefficients(coefficients)
       }
       
       fn get_dominating_objective(&self) -> f64 {
           self.core.dominating_objective
       }
       
       fn set_dominating_objective(&mut self, val: f64) {
           self.core.dominating_objective = val;
       }
       
       // ... other delegations
   }
   ```

4. **Keep type-specific implementations**:
   - `extract_storage_from_trajectory` - accesses `core.state_coefficients` directly
   - `add_cut_constraint_to_model` - StorageState-specific logic
   - `evaluate_cut` - StorageState-specific coefficient structure
   - etc.

### Key Files to Modify

- `src/state.rs` - StorageState struct and impl

### Patterns to Follow

For fields that need direct access:
```rust
// Access internal coefficient buffer
self.core.state_coefficients.copy_from_slice(data);

// Or via method if available
self.core.update_coefficients(data);
```

### Pitfalls to Avoid

- ⚠️ **Don't change behavior** - All logic must remain identical
- ⚠️ **Check dimension access** - May need to access `self.core.dimension`
- ⚠️ **Watch for `self.state_coefficients`** - Now `self.core.state_coefficients`

---

## Testing Requirements

### Unit Tests

- [ ] `StorageState::new()` creates correctly with core
- [ ] All trait methods work correctly via delegation
- [ ] `extract_storage_from_trajectory()` still works

### Regression Tests

- [ ] All existing `StorageState` tests pass unchanged
- [ ] Tests that create states with specific values still work

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Update struct doc comments
- [ ] Note composition pattern in module docs

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Mechanical refactoring with clear before/after

---

## Definition of Done

- [ ] `StorageState` uses `StateCore` composition
- [ ] All common methods delegate correctly
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] No performance regression
- [ ] Ready for T-043
