# [T-077] Update FCF for Shared Layout State Access

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 4: Pool Architecture Refinement](./00-sprint-overview.md)
> **Dependencies**: [T-076](./ticket-076-refactor-state-pool.md)
> **Blocks**: [T-078](./ticket-078-remove-concrete-state.md)

## Files to Read Before Starting

- `src/fcf.rs:52-70` - FCF struct with state_pool field
- `src/fcf.rs:156-175` - `preallocate_pools()` method
- `src/fcf.rs:180-230` - `eval_new_cut_domination()` method
- `src/fcf.rs:300-380` - `add_cuts_batch()` method
- `src/fcf.rs:520-620` - `eval_existing_cuts_at_state()` and domination methods
- `src/cut.rs:460-520` - `update_cut_and_state_slots()` method

---

## Context

### Background

After T-076, `VisitedStatePool` stores `Vec<StateData>` instead of `Vec<ConcreteState>`. The FCF code that accesses states needs to be updated to work with `StateData` and potentially access the shared layout when needed.

### Current State (After T-076)

```rust
pub struct VisitedStatePool {
    pub pool: Vec<StateData>,          // Changed from Vec<ConcreteState>
    pub layout: Option<StateLayout>,   // Shared layout
    pub state_type: StateTypeId,
    pub num_hydros: usize,
}
```

### Key FCF Operations Using States

1. **Domination evaluation**: `state.coefficients()`, `state.update_dominating_cut()`
2. **State update in add_cuts_batch**: `state_pool.update_state()`
3. **Cut evaluation**: Uses `state.coefficients()` for dot product

---

## Specification

### Required Changes

Most operations should work unchanged because `StateData` has the same method names as `ConcreteState`:

| Method | ConcreteState | StateData | Compatible |
|--------|---------------|-----------|------------|
| `coefficients()` | ✓ | ✓ | ✅ Yes |
| `update_coefficients()` | ✓ | ✓ | ✅ Yes |
| `get_iteration()` | ✓ | ✓ | ✅ Yes |
| `set_iteration()` | ✓ | ✓ | ✅ Yes |
| `get_dominating_cut_id()` | ✓ | ✓ | ✅ Yes |
| `update_dominating_cut()` | ✓ | ✓ | ✅ Yes |

### Potential Breaking Points

1. **Type annotations**: Any explicit `ConcreteState` type annotations need updating
2. **Pattern matching**: Any match on `ConcreteState` variants
3. **state_type() method**: Was on ConcreteState, now on pool

---

## Implementation Guide

### Step 1: Audit FCF state access patterns

Search for all state pool access:

```bash
grep -n "state_pool\.pool\[" src/fcf.rs
grep -n "state_pool\.update_state" src/fcf.rs
grep -n "ConcreteState" src/fcf.rs
```

### Step 2: Update type annotations if any

If there are explicit type annotations like:
```rust
let state: &ConcreteState = &self.state_pool.pool[slot];
```

Change to:
```rust
let state: &StateData = &self.state_pool.pool[slot];
```

Or just remove the type annotation and let inference work.

### Step 3: Update any state_type checks

If code checks `state.state_type()`:
```rust
// Before
if state.state_type() == StateTypeId::StorageAndInflow { ... }

// After
if self.state_pool.state_type == StateTypeId::StorageAndInflow { ... }
```

### Step 4: Update BendersCutPool::update_cut_and_state_slots

In `src/cut.rs`, this method takes `&mut VisitedStatePool`:

```rust
pub fn update_cut_and_state_slots(
    &mut self,
    iteration: usize,
    forward_pass_idx: usize,
    cut_coefficients: &[f64],
    cut_rhs: f64,
    state_coefficients: &[f64],
    state_pool: &mut crate::state::VisitedStatePool,
) -> usize
```

This should still work since it calls `state_pool.update_state()` which returns `&mut StateData`.

### Step 5: Verify domination evaluation

In `src/fcf.rs`, `eval_new_cut_domination()`:

```rust
for state in self.state_pool.pool.iter_mut() {
    let state_coefs = state.coefficients();
    let height = new_cut.eval_height_at_state(state_coefs);
    // ...
    state.update_dominating_cut(new_cut, height);
}
```

This pattern works unchanged with `StateData`.

### Step 6: Run tests and fix any remaining issues

```bash
cargo build -j1 2>&1 | head -50
cargo test -j1 --lib 2>&1 | tail -30
```

---

## Acceptance Criteria

- [x] All FCF state access works with StateData
- [x] No `ConcreteState` references in FCF code
- [x] All 573+ tests pass
- [x] Golden tests pass
- [x] No performance regression

---

## Testing Requirements

### Existing Tests

All existing FCF tests should pass without modification if the migration is done correctly.

### Specific Tests to Verify

```bash
cargo test -j1 --lib fcf
cargo test -j1 --lib add_cuts_batch
cargo test -j1 --lib domination
```

### Integration Tests

```bash
cargo test -j1 --test test_sddp_algorithm
```

---

## Pitfalls to Avoid

- ⚠️ **Don't change behavior** - Only update types, not logic
- ⚠️ **Check all pool.pool[slot] patterns** - They now return StateData
- ⚠️ **state_type is now on pool** - Not on individual states

---

## Documentation Requirements

- [ ] Update any FCF documentation mentioning ConcreteState
- [ ] No new public API changes needed

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Mostly mechanical updates. StateData has compatible interface. Need to audit all usages.

---

## Definition of Done

- [x] No ConcreteState references in FCF code
- [x] All FCF tests pass
- [x] Integration tests pass
- [x] Code is clippy-clean
