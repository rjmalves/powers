# [TICKET-007] Add State trait methods for in-place updates

> **Epic**: [Epic 2: Preallocated State Pool](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: Epic 1 complete
> **Blocks**: [TICKET-008](./ticket-008-implement-storagestate-update.md), [TICKET-009](./ticket-009-implement-storageinflowstate-update.md)

## Context

### Background

To enable state preallocation, we need methods to update state coefficients in place without allocating new memory. The `State` trait needs to be extended with these methods.

### Relation to Epic

Foundation for state preallocation - defines the interface that implementations must provide.

### Current State

The `State` trait has `coefficients() -> &[f64]` for reading but no method for in-place writing.

## Files to Read Before Starting

- `src/state.rs` - State trait and implementations
- `MEMORY_STABILITY_ANALYSIS.md` - Required trait extensions section

## Specification

### New Trait Methods

```rust
pub trait State: Send + Sync {
    // Existing methods...
    
    /// Update coefficient values in place (no allocation).
    ///
    /// # Arguments
    /// * `coefficients` - New values to copy into internal storage
    ///
    /// # Panics
    /// Panics if `coefficients.len() != self.dimension()`
    fn update_coefficients(&mut self, coefficients: &[f64]);
    
    /// Reset coefficients to zero while preserving capacity.
    /// Used for initializing preallocated states.
    fn reset_to_zero(&mut self);
    
    /// Get the state dimension (number of coefficients).
    fn dimension(&self) -> usize;
}
```

### Behavior

- `update_coefficients()`: Copy values into internal storage using `copy_from_slice()`
- `reset_to_zero()`: Fill coefficients with 0.0 using `fill(0.0)`
- `dimension()`: Return the length of the coefficient vector

### Error Handling

- `debug_assert_eq!()` for dimension mismatch in `update_coefficients()`

## Acceptance Criteria

- [ ] All three methods added to State trait
- [ ] Methods documented with doc comments
- [ ] Compile-time enforcement of implementation
- [ ] No breaking changes to existing code

## Implementation Guide

### Suggested Approach

1. Add method signatures to `State` trait in `src/state.rs`
2. Add placeholder implementations to both state types (will be replaced in next tickets)
3. Update trait documentation

### Key Files to Modify

- `src/state.rs`: Add methods to `State` trait

### Pitfalls to Avoid

- ⚠️ Don't add default implementations that allocate
- ⚠️ Ensure method signatures allow efficient implementation

## Testing Requirements

### Unit Tests

- [ ] Compile-time test: both implementations must compile with new methods
- [ ] Placeholder implementations can be `unimplemented!()` initially

## Documentation Requirements

- [ ] Doc comments on all three methods
- [ ] Note about dimension requirements for `update_coefficients()`

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward trait extension
