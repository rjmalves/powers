# [TICKET-002] Add BendersCut::update() for in-place modification

> **Epic**: [Epic 1: Preallocated Cut Pool](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [TICKET-003](./ticket-003-implement-cutpool-preallocate.md)

## Context

### Background

With preallocation, cuts are created once at initialization with zeroed coefficients. During training, we need to update these preallocated cuts in place rather than creating new ones.

### Relation to Epic

Enables the core preallocation strategy by allowing cuts to be modified without allocation.

### Current State

`BendersCut::new()` allocates a new cut with its coefficient vector. There is no method to update an existing cut's data in place.

## Specification

### Inputs

- `&mut self` - Mutable reference to preallocated cut
- `coefficients: &[f64]` - New coefficient values (must match preallocated length)
- `rhs: f64` - New RHS value
- `iteration: usize` - Iteration that created this cut
- `forward_pass_idx: usize` - Forward pass that created this cut

### Outputs

- None (modifies `self` in place)

### Behavior

1. Copy coefficients into existing Vec using `copy_from_slice()`
2. Set `rhs` to new value
3. Set `iteration` and `forward_pass_idx` metadata
4. Set `active = true`
5. Reset `non_dominated_state_count = 1` (new cut dominates its source state)

### Error Handling

- `debug_assert_eq!(self.coefficients.len(), coefficients.len())` - Dimension mismatch

## Acceptance Criteria

- [ ] Method updates all fields correctly without allocation
- [ ] Coefficient vector length validated in debug mode
- [ ] Preallocated capacity preserved (no reallocation)
- [ ] Unit tests verify all fields are updated

## Implementation Guide

### Suggested Approach

1. Add `update()` method to `impl BendersCut`
2. Use `copy_from_slice()` for efficient memory copy
3. Add unit test verifying no allocation occurs

### Key Files to Modify

- `src/cut.rs`: Add `update()` method to `impl BendersCut`

### Patterns to Follow

- See `BendersCut::new()` for field initialization pattern

### Pitfalls to Avoid

- ⚠️ Don't use `self.coefficients = coefficients.to_vec()` (allocates!)
- ⚠️ Don't forget to reset `non_dominated_state_count`

## Testing Requirements

### Unit Tests

- [ ] Test update correctly modifies all fields
- [ ] Test coefficients are copied correctly (not just length)
- [ ] Test with various coefficient dimensions (1, 10, 156)
- [ ] Test debug_assert fires on dimension mismatch

### Performance Tests

- [ ] Verify no allocation during update (use allocator profiling or capacity check)

## Documentation Requirements

- [ ] Doc comment explaining in-place update for preallocation
- [ ] Note about dimension requirement

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Simple method, but needs careful testing for no-allocation guarantee
