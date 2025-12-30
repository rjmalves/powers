# [T-078] Remove ConcreteState Enum

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 4: Pool Architecture Refinement](./00-sprint-overview.md)
> **Dependencies**: [T-077](./ticket-077-update-fcf-state-access.md)
> **Blocks**: [T-079](./ticket-079-benchmark-memory.md)

## Files to Read Before Starting

- `src/state.rs:480-787` - Current `ConcreteState` enum
- `src/state.rs` - Search for all `ConcreteState` usages
- `src/fcf.rs` - Verify no remaining usages
- `src/cut.rs` - Verify no remaining usages

---

## Context

### Background

After T-076 and T-077, `VisitedStatePool` uses `StateData` directly. The `ConcreteState` enum is no longer used in production code and can be removed.

### Current State

The `ConcreteState` enum still exists but is not used by `VisitedStatePool`:

```rust
pub enum ConcreteState {
    Storage { core: StateCore, num_hydros: usize },
    StorageAndInflow { core: StateCore, num_hydros: usize, layout: StateLayout },
}
```

### Target State

Remove the enum entirely since it's now dead code.

---

## Specification

### Items to Remove

| Item | Location | Status |
|------|----------|--------|
| `ConcreteState` enum | `src/state.rs:480-516` | Remove |
| `ConcreteState` impl block | `src/state.rs:518-787` | Remove |
| `ConcreteState` tests | `src/state.rs` (in tests module) | Remove |

### Items to Keep

| Item | Reason |
|------|--------|
| `StateConfig` | Used by pool initialization |
| `StateCore` | Used by `StorageState` and `StorageAndInflowState` |
| `StateData` | Used by `VisitedStatePool` |
| `StateLayout` | Used by pool and state implementations |
| `StateTypeId` | Used by pool and config |

---

## Implementation Guide

### Step 1: Find all ConcreteState usages

```bash
grep -rn "ConcreteState" src/
```

### Step 2: Verify no production usages remain

After T-076 and T-077, there should be no usages in:
- `src/fcf.rs`
- `src/cut.rs`
- `src/sddp/mod.rs`
- `src/algorithm/`

### Step 3: Remove ConcreteState enum definition

Delete lines ~480-516 (adjust based on current line numbers):

```rust
// DELETE THIS ENTIRE BLOCK
pub enum ConcreteState {
    Storage {
        core: StateCore,
        num_hydros: usize,
    },
    StorageAndInflow {
        core: StateCore,
        num_hydros: usize,
        layout: StateLayout,
    },
}
```

### Step 4: Remove ConcreteState impl block

Delete the entire `impl ConcreteState { ... }` block (~lines 518-787).

### Step 5: Remove ConcreteState tests

Find and remove tests for ConcreteState:

```bash
grep -n "test_concrete_state" src/state.rs
```

Remove all tests starting with `test_concrete_state_*`.

### Step 6: Update StateConfig::create_state

This method currently returns `ConcreteState`. It should now be removed or changed to return `StateData`:

```rust
// Option A: Remove the method (if no longer needed)
// StateConfig is now only used by VisitedStatePool::preallocate_concrete

// Option B: Return StateData instead
impl StateConfig {
    pub fn create_state_data(&self) -> StateData {
        StateData::new(self.dimension())
    }
}
```

Check if `create_state()` is used anywhere and update accordingly.

### Step 7: Clean up imports

Remove any unused imports related to ConcreteState.

### Step 8: Run tests

```bash
cargo build -j1
cargo test -j1 --lib
```

---

## Acceptance Criteria

- [x] `ConcreteState` enum removed from codebase
- [x] No compilation errors
- [x] All remaining tests pass
- [x] `StateConfig` updated if needed
- [x] No unused code warnings

---

## Testing Requirements

### Compile Check

```bash
cargo build -j1
```

### Run All Tests

```bash
cargo test -j1 --lib
cargo test -j1 --test test_sddp_algorithm
cargo test -j1 --test test_integration_suite
```

---

## Pitfalls to Avoid

- ⚠️ **Check StateConfig::create_state()** - This returns ConcreteState, needs updating
- ⚠️ **Don't remove StateConfig** - It's still used for pool initialization
- ⚠️ **Keep tests that test StateData** - Only remove ConcreteState-specific tests

---

## Documentation Requirements

- [ ] Remove any documentation referencing ConcreteState
- [ ] Update module documentation if it mentions ConcreteState

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Straightforward deletion once dependencies are removed.

---

## Definition of Done

- [x] ConcreteState enum removed
- [x] ConcreteState impl block removed
- [x] ConcreteState tests removed
- [x] StateConfig updated (create_state_data instead of create_state)
- [x] All tests pass (567 lib tests)
- [x] No dead code warnings
