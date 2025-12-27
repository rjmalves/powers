# [TICKET-001] Add slot computation utility function

> **Epic**: [Epic 1: Preallocated Cut Pool](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [TICKET-003](./ticket-003-implement-cutpool-preallocate.md)

## Context

### Background

The memory stability approach uses `(iteration, forward_pass_idx)` as a deterministic key to access preallocated cuts and states. This requires a utility function to compute the slot index from these coordinates.

### Relation to Epic

This is the foundational utility that enables slot-based access throughout the preallocation system.

### Current State

Currently, cuts are pushed to a Vec with auto-incrementing IDs. There is no slot computation logic.

## Specification

### Inputs

- `iteration: usize` - Current iteration number (1-based, range: 1..=num_iterations)
- `forward_pass_idx: usize` - Forward pass index (0-based, range: 0..num_forward_passes)
- `num_forward_passes: usize` - Total forward passes per iteration

### Outputs

- `usize` - Slot index in the preallocated pool (0-based)

### Behavior

- Formula: `slot = (iteration - 1) * num_forward_passes + forward_pass_idx`
- Example with `num_forward_passes = 16`:
  - `(1, 0)` → slot 0
  - `(1, 15)` → slot 15
  - `(2, 0)` → slot 16
  - `(8, 15)` → slot 127

### Error Handling

- `debug_assert!(iteration >= 1)` - Catches 0-based iteration errors in debug builds
- `debug_assert!(forward_pass_idx < num_forward_passes)` - Catches out-of-bounds fp index

## Acceptance Criteria

- [ ] Function computes correct slot for all valid inputs
- [ ] debug_assert catches invalid inputs in debug mode
- [ ] Function is marked `#[inline]` for zero-cost abstraction
- [ ] Unit tests cover edge cases: first slot, last slot, boundary conditions

## Implementation Guide

### Suggested Approach

1. Add function to `src/fcf.rs` (or create `src/utils/slot.rs` if preferred)
2. Implement with formula and debug assertions
3. Add comprehensive unit tests

### Key Files to Modify

- `src/fcf.rs`: Add `compute_slot()` function

### Patterns to Follow

- See `src/utils.rs` for similar utility function patterns

### Pitfalls to Avoid

- ⚠️ Don't use 0-based iteration (SDDP iterations are 1-based)
- ⚠️ Don't forget the `#[inline]` attribute

## Testing Requirements

### Unit Tests

- [ ] `compute_slot(1, 0, 16) == 0` (first slot)
- [ ] `compute_slot(1, 15, 16) == 15` (end of first iteration)
- [ ] `compute_slot(2, 0, 16) == 16` (start of second iteration)
- [ ] `compute_slot(8, 15, 16) == 127` (last slot for 8 iterations)
- [ ] `compute_slot(1, 0, 1) == 0` (single forward pass)
- [ ] Debug assertion fires for `iteration = 0`
- [ ] Debug assertion fires for `forward_pass_idx >= num_forward_passes`

## Documentation Requirements

- [ ] Doc comment explaining the formula and example usage
- [ ] Note that iteration is 1-based (matching SDDP convention)

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple function with clear specification
