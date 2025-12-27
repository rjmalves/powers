# [TICKET-008] Implement update methods for StorageState

> **Epic**: [Epic 2: Preallocated State Pool](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [TICKET-007](./ticket-007-add-state-trait-methods.md)
> **Blocks**: [TICKET-010](./ticket-010-implement-statepool-preallocate.md)

## Context

### Background

`StorageState` is the simpler state implementation with only storage coefficients. Implementing in-place update methods for this type establishes the pattern.

### Relation to Epic

First concrete implementation of the in-place update pattern.

### Current State

`StorageState` has a `state_coefficients: Vec<f64>` field but no methods to update it in place (only through `extract_storage_from_trajectory()`).

## Files to Read Before Starting

- `src/state.rs` - `StorageState` implementation (lines 488-719)
- [TICKET-007](./ticket-007-add-state-trait-methods.md) - Trait method signatures

## Specification

### Implementation

```rust
impl State for StorageState {
    fn update_coefficients(&mut self, coefficients: &[f64]) {
        debug_assert_eq!(
            self.state_coefficients.len(),
            coefficients.len(),
            "Coefficient dimension mismatch"
        );
        self.state_coefficients.copy_from_slice(coefficients);
    }
    
    fn reset_to_zero(&mut self) {
        self.state_coefficients.fill(0.0);
    }
    
    fn dimension(&self) -> usize {
        self.state_coefficients.len()
    }
}
```

### Behavior

- `update_coefficients()`: Direct copy, O(n) where n = num_hydros
- `reset_to_zero()`: Fill with zeros, O(n)
- `dimension()`: Returns `self.dimension` (which equals `state_coefficients.len()`)

## Acceptance Criteria

- [ ] All three methods implemented correctly
- [ ] No memory allocation during update
- [ ] Dimension validated in debug mode
- [ ] Unit tests for all methods
- [ ] Existing tests still pass

## Implementation Guide

### Suggested Approach

1. Add implementations to `impl State for StorageState`
2. Use `copy_from_slice()` for efficiency
3. Add comprehensive unit tests

### Key Files to Modify

- `src/state.rs`: Implement methods in `impl State for StorageState`

### Pitfalls to Avoid

- ⚠️ Don't use `= coefficients.to_vec()` (allocates!)
- ⚠️ Use `fill(0.0)` not `iter_mut().for_each()` (more idiomatic)

## Testing Requirements

### Unit Tests

- [ ] Test `update_coefficients()` correctly updates values
- [ ] Test `reset_to_zero()` zeros all values
- [ ] Test `dimension()` returns correct value
- [ ] Test debug_assert fires on dimension mismatch
- [ ] Test that capacity is preserved after update

```rust
#[test]
fn test_storage_state_update_coefficients() {
    let system = create_test_system_with_hydros(3);
    let mut state = StorageState::new(&system);
    
    state.update_coefficients(&[10.0, 20.0, 30.0]);
    
    assert_eq!(state.coefficients(), &[10.0, 20.0, 30.0]);
}

#[test]
fn test_storage_state_reset_to_zero() {
    let system = create_test_system_with_hydros(3);
    let mut state = StorageState::new(&system);
    
    state.update_coefficients(&[10.0, 20.0, 30.0]);
    state.reset_to_zero();
    
    assert_eq!(state.coefficients(), &[0.0, 0.0, 0.0]);
}
```

## Documentation Requirements

- [ ] Doc comments on implementation

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Simple implementation following established patterns
