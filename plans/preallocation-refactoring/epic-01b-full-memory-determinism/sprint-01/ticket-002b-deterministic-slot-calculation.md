# [TICKET-002b] Implement deterministic slot calculation

> **Epic**: [Epic 1b: Full Memory Determinism](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-001b](./ticket-001b-add-fields.md)  
> **Blocks**: [TICKET-003b](./ticket-003b-update-cut-addition.md)

## Context

### Background

With `num_forward_passes` stored in Subproblem, we can compute the slot index for any cut deterministically. This replaces the current `allocate_cut_slot()` method that uses a free list and sequential counter.

### Formula

```rust
slot_index = (iteration - 1) * num_forward_passes + forward_pass_idx
```

Where:
- `iteration` is 1-based (1..=num_iterations)
- `forward_pass_idx` is 0-based (0..num_forward_passes)

### Examples (4 forward passes)

| Iteration | Forward Pass | Slot Index |
|-----------|--------------|------------|
| 1 | 0 | 0 |
| 1 | 1 | 1 |
| 1 | 2 | 2 |
| 1 | 3 | 3 |
| 2 | 0 | 4 |
| 2 | 1 | 5 |
| 32 | 3 | 127 |

## Files to Read Before Starting

- `src/subproblem.rs` - Current `allocate_cut_slot()` method (lines 1084-1099)

## Specification

### New Method on Subproblem

Add a deterministic slot calculation method:

```rust
/// Compute deterministic slot index for a cut.
///
/// Each (iteration, forward_pass_idx) pair maps to exactly one slot.
/// This eliminates slot tracking overhead and ensures memory determinism.
///
/// # Formula
///
/// ```text
/// slot = (iteration - 1) * num_forward_passes + forward_pass_idx
/// ```
///
/// # Arguments
///
/// * `iteration` - 1-based iteration number (1..=num_iterations)
/// * `forward_pass_idx` - 0-based forward pass index (0..num_forward_passes)
///
/// # Panics
///
/// Panics if computed slot exceeds `num_preallocated_cuts`.
///
/// # Example
///
/// ```ignore
/// // With 4 forward passes
/// let slot = subproblem.compute_cut_slot(1, 0); // slot = 0
/// let slot = subproblem.compute_cut_slot(2, 0); // slot = 4
/// let slot = subproblem.compute_cut_slot(32, 3); // slot = 127
/// ```
#[inline]
pub fn compute_cut_slot(&self, iteration: usize, forward_pass_idx: usize) -> usize {
    debug_assert!(iteration >= 1, "iteration must be 1-based");
    debug_assert!(
        forward_pass_idx < self.num_forward_passes,
        "forward_pass_idx {} >= num_forward_passes {}",
        forward_pass_idx, self.num_forward_passes
    );
    
    let slot = (iteration - 1) * self.num_forward_passes + forward_pass_idx;
    
    assert!(
        slot < self.num_preallocated_cuts,
        "Cut slot {} exceeds preallocated count {}. \
         This indicates a bug: iteration={}, forward_pass_idx={}, num_forward_passes={}",
        slot, self.num_preallocated_cuts, iteration, forward_pass_idx, self.num_forward_passes
    );
    
    slot
}
```

### Validation Method (Optional)

Add a method to verify preallocation is sufficient:

```rust
/// Verify that preallocated slots cover all possible cuts.
///
/// Called once after preallocation to catch configuration errors early.
#[inline]
pub fn validate_preallocation(&self, num_iterations: usize) -> Result<(), String> {
    let required_slots = num_iterations * self.num_forward_passes;
    if required_slots > self.num_preallocated_cuts {
        return Err(format!(
            "Insufficient preallocated slots: need {} ({} iters × {} fp), have {}",
            required_slots, num_iterations, self.num_forward_passes, self.num_preallocated_cuts
        ));
    }
    Ok(())
}
```

## Acceptance Criteria

- [x] `compute_cut_slot()` method implemented
- [x] Method uses correct formula: `(iteration - 1) * num_fp + fp_idx`
- [x] Debug assertions validate inputs
- [x] Panic assertion for slot overflow
- [x] Optional: `validate_preallocation()` method
- [x] Code compiles
- [x] Examples pass (method not yet called)

## Implementation Guide

### Step 1: Add compute_cut_slot()

1. Locate the existing `allocate_cut_slot()` method (lines 1084-1099)
2. Add `compute_cut_slot()` above or below it
3. Do NOT remove `allocate_cut_slot()` yet (done in TICKET-005b)

### Step 2: Add validate_preallocation() (Optional)

1. Add validation method
2. Consider calling it from `preallocate_cut_constraints()`

### Pitfalls to Avoid

- ⚠️ Iteration is 1-based (1, 2, 3...), not 0-based
- ⚠️ Forward pass index is 0-based (0, 1, 2...)
- ⚠️ Don't remove `allocate_cut_slot()` yet - other code depends on it

## Testing Requirements

### Unit Tests

Add unit test for slot calculation:

```rust
#[test]
fn test_compute_cut_slot_deterministic() {
    // Setup: 4 forward passes, 32 iterations = 128 slots
    let mut subproblem = create_test_subproblem();
    subproblem.num_forward_passes = 4;
    subproblem.num_preallocated_cuts = 128;
    
    // Test iteration 1
    assert_eq!(subproblem.compute_cut_slot(1, 0), 0);
    assert_eq!(subproblem.compute_cut_slot(1, 1), 1);
    assert_eq!(subproblem.compute_cut_slot(1, 2), 2);
    assert_eq!(subproblem.compute_cut_slot(1, 3), 3);
    
    // Test iteration 2
    assert_eq!(subproblem.compute_cut_slot(2, 0), 4);
    assert_eq!(subproblem.compute_cut_slot(2, 3), 7);
    
    // Test last slot
    assert_eq!(subproblem.compute_cut_slot(32, 3), 127);
}

#[test]
#[should_panic(expected = "exceeds preallocated count")]
fn test_compute_cut_slot_overflow_panics() {
    let mut subproblem = create_test_subproblem();
    subproblem.num_forward_passes = 4;
    subproblem.num_preallocated_cuts = 128;
    
    // This should panic: slot 128 exceeds 128 preallocated
    subproblem.compute_cut_slot(33, 0);
}
```

### Integration Tests

- [ ] Examples 01 and 07 still pass

## Documentation Requirements

- [ ] Doc comment with formula
- [ ] Example in doc comment
- [ ] Note about 1-based iteration

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Simple arithmetic, main work is testing edge cases

## Definition of Done

- [x] Method implemented
- [x] Assertions in place
- [x] Unit tests pass
- [x] Examples pass
