# [TICKET-001b] Add slot_index to BendersCut and num_forward_passes to Subproblem

> **Epic**: [Epic 1b: Full Memory Determinism](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [TICKET-002b](./ticket-002b-deterministic-slot-calculation.md)

## Context

### Background

To enable deterministic slot calculation, we need two pieces of information:
1. **In Subproblem**: `num_forward_passes` to compute `slot = (iter-1) * num_fp + fp_idx`
2. **In BendersCut**: `slot_index` to store the assigned slot for O(1) deactivation lookup

### Current State

- `BendersCut` has `iteration` and `forward_pass_idx` but no `slot_index`
- `Subproblem` has `num_preallocated_cuts` but no `num_forward_passes`

## Files to Read Before Starting

- `src/cut.rs` - BendersCut struct definition
- `src/subproblem.rs` - Subproblem struct (lines 760-790 for cut fields)

## Specification

### Changes to BendersCut (src/cut.rs)

Add a new field to store the assigned slot index:

```rust
pub struct BendersCut {
    pub id: usize,
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    pub active: bool,
    pub non_dominated_state_count: usize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
    /// Preallocated slot index in HiGHS model (0-based).
    /// Set when cut is added via preallocation. Used for O(1) deactivation.
    pub slot_index: Option<usize>,  // NEW
}
```

Update `BendersCut::new()` to initialize `slot_index: None`.

### Changes to Subproblem (src/subproblem.rs)

Add field to Subproblem struct (near the other cut preallocation fields):

```rust
/// Number of forward passes per iteration.
/// Used for deterministic slot calculation: slot = (iter-1) * num_fp + fp_idx
num_forward_passes: usize,
```

Update the constructor to initialize `num_forward_passes: 0`.

Update `preallocate_cut_constraints()` signature to accept `num_forward_passes`:

```rust
pub fn preallocate_cut_constraints(
    &mut self,
    max_cuts: usize,
    num_forward_passes: usize,  // NEW parameter
) -> Result<(), String>
```

Store `num_forward_passes` in the method.

## Acceptance Criteria

- [x] `BendersCut` has `slot_index: Option<usize>` field
- [x] `BendersCut::new()` initializes `slot_index: None`
- [x] `Subproblem` has `num_forward_passes: usize` field
- [x] Constructor initializes `num_forward_passes: 0`
- [x] `preallocate_cut_constraints()` accepts and stores `num_forward_passes`
- [x] All call sites of `preallocate_cut_constraints()` updated
- [x] Code compiles without warnings
- [x] Examples 01 and 07 still pass

## Implementation Guide

### Step 1: Update BendersCut (src/cut.rs)

1. Add `pub slot_index: Option<usize>` to struct
2. Update `BendersCut::new()` to include `slot_index: None`
3. Check if any tests create BendersCut directly and update them

### Step 2: Update Subproblem (src/subproblem.rs)

1. Add `num_forward_passes: usize` field (near line 775)
2. Initialize to `0` in constructor (near line 936)
3. Update `preallocate_cut_constraints()` signature and body
4. Store `self.num_forward_passes = num_forward_passes`

### Step 3: Update Call Sites

Find all calls to `preallocate_cut_constraints()`:

```bash
grep -n "preallocate_cut_constraints" src/
```

Update each to pass `num_forward_passes`. In `src/sddp/mod.rs`, this value is already available.

### Pitfalls to Avoid

- ⚠️ Remember to update ALL constructors (if Subproblem has multiple)
- ⚠️ Check for test files that may construct BendersCut directly
- ⚠️ The `state_dimension` parameter was removed from `preallocate_cut_constraints()` earlier - verify current signature

## Testing Requirements

### Unit Tests

- [ ] Existing cut tests still pass
- [ ] Existing subproblem initialization tests still pass

### Integration Tests

- [ ] Examples 01 and 07 run without errors
- [ ] Results identical to before

## Documentation Requirements

- [ ] Doc comment on `slot_index` field
- [ ] Doc comment on `num_forward_passes` field
- [ ] Update `preallocate_cut_constraints()` doc to mention new parameter

## Effort Estimate

**Points**: 1  
**Confidence**: High  
**Rationale**: Simple field additions with clear call site updates

## Definition of Done

- [x] Fields added to both structs
- [x] Constructors updated
- [x] Method signature updated
- [x] All call sites updated
- [x] Code compiles
- [x] Examples pass
