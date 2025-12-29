# [T-043] Add Pool-Compatible Trait Extensions

> **Epic**: [Epic 4: State Simplification](../00-epic-overview.md)
> **Sprint**: [Sprint 1: State Consolidation](./00-sprint-overview.md)
> **Dependencies**: [T-041](./ticket-041-consolidate-storage-state.md), [T-042](./ticket-042-consolidate-inflow-state.md)
> **Blocks**: [T-044](./ticket-044-state-extraction-module.md)

## Files to Read Before Starting

- `src/state.rs` - State trait and implementations (after T-041/T-042)
- `src/fcf.rs` - VisitedStatePool, add_cuts_batch_from_data
- `src/cut.rs` - BendersCutPool, slot-based allocation
- T-039 - State-Cut relationship documentation

---

## Context

### Background

Epic 5 will implement pool-based memory allocation for cuts and states. This ticket prepares the `State` trait with methods that enable zero-allocation state updates:

1. **`copy_coefficients_into()`** - Copy coefficients to external buffer
2. **`coefficient_count()`** - Return size for preallocation
3. **`state_type_id()`** - Type identifier for pool slot selection

These methods complement the existing `update_coefficients()` which copies *into* the state.

### Current State

`VisitedStatePool` already has slot-based update:
```rust
pub fn update_state(
    &mut self,
    slot: usize,
    coefficients: &[f64],
    iteration: usize,
    forward_pass_idx: usize,
) -> &mut Box<dyn State>
```

But there's no standardized way to:
- Copy coefficients *out* to an external buffer
- Know the coefficient count before allocation
- Identify state type for heterogeneous pools

---

## Specification

### New Trait Methods

Add to the `State` trait:

```rust
pub trait State: Send + Sync {
    // Existing methods...

    /// Copy state coefficients into a preallocated buffer.
    ///
    /// Used by Epic 5 pools to store state data without Box allocation.
    /// The target buffer must have at least `coefficient_count()` capacity.
    ///
    /// # Arguments
    ///
    /// * `target` - Preallocated buffer to copy into
    ///
    /// # Panics
    ///
    /// Debug panics if `target.len() < self.coefficient_count()`
    fn copy_coefficients_into(&self, target: &mut [f64]) {
        debug_assert!(
            target.len() >= self.coefficients().len(),
            "target buffer too small"
        );
        target[..self.coefficients().len()]
            .copy_from_slice(self.coefficients());
    }

    /// Returns the number of coefficients in this state.
    ///
    /// Used for preallocating pool slots with correct capacity.
    /// This equals `coefficients().len()` but can be called without
    /// needing access to the coefficient data.
    fn coefficient_count(&self) -> usize {
        self.dimension()
    }

    /// Returns a type identifier for this state implementation.
    ///
    /// Used by heterogeneous pools to select the correct slot structure.
    /// Each State implementation returns a unique identifier.
    fn state_type_id(&self) -> StateTypeId;
}

/// Identifier for State implementation types.
///
/// Used by pools to manage heterogeneous state types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateTypeId {
    Storage,
    StorageAndInflow,
}
```

### Implementations

```rust
impl State for StorageState {
    fn state_type_id(&self) -> StateTypeId {
        StateTypeId::Storage
    }
    
    // copy_coefficients_into and coefficient_count use defaults
}

impl State for StorageAndInflowState {
    fn state_type_id(&self) -> StateTypeId {
        StateTypeId::StorageAndInflow
    }
    
    fn coefficient_count(&self) -> usize {
        self.layout.total_dim  // Override: total_dim, not dimension
    }
}
```

---

## Acceptance Criteria

- [ ] `copy_coefficients_into()` method added to State trait with default impl
- [ ] `coefficient_count()` method added with default impl
- [ ] `StateTypeId` enum created
- [ ] `state_type_id()` method added (required, no default)
- [ ] Both implementations provide `state_type_id()`
- [ ] `StorageAndInflowState` overrides `coefficient_count()`
- [ ] All tests pass
- [ ] Golden tests pass

---

## Implementation Guide

### Suggested Approach

1. **Add `StateTypeId` enum** at top of state.rs:
   ```rust
   #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
   pub enum StateTypeId {
       Storage,
       StorageAndInflow,
   }
   ```

2. **Extend State trait** with new methods:
   - Default impl for `copy_coefficients_into`
   - Default impl for `coefficient_count`
   - Required method `state_type_id` (no default)

3. **Implement for StorageState**:
   - Just add `state_type_id()` returning `StateTypeId::Storage`

4. **Implement for StorageAndInflowState**:
   - Add `state_type_id()` returning `StateTypeId::StorageAndInflow`
   - Override `coefficient_count()` to return `self.layout.total_dim`

### Pitfalls to Avoid

- ⚠️ **coefficient_count for StorageAndInflowState** must return `total_dim`, not `dimension` (num_hydros)
- ⚠️ **Ensure Send + Sync** bounds still satisfied (StateTypeId is Copy)

---

## Testing Requirements

### Unit Tests

- [ ] `copy_coefficients_into()` copies correctly for StorageState
- [ ] `copy_coefficients_into()` copies correctly for StorageAndInflowState
- [ ] `coefficient_count()` returns correct value for each type
- [ ] `state_type_id()` returns correct enum variant

### Integration Tests

- [ ] Existing state pool tests still work
- [ ] State creation and coefficient access unchanged

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Doc comments on all new trait methods
- [ ] Doc comments on `StateTypeId` enum
- [ ] Example usage in module docs

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Small additions with clear spec, extends existing trait

---

## Definition of Done

- [ ] New trait methods added
- [ ] `StateTypeId` enum created
- [ ] Both implementations updated
- [ ] Unit tests for new methods
- [ ] All existing tests pass
- [ ] Golden tests pass
- [ ] Ready for Epic 5 pool implementation
