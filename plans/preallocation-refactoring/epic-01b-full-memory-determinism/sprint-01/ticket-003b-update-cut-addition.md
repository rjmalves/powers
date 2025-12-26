# [TICKET-003b] Update cut addition flow to use deterministic slots

> **Epic**: [Epic 1b: Full Memory Determinism](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-002b](./ticket-002b-deterministic-slot-calculation.md)  
> **Blocks**: [TICKET-004b](./ticket-004b-update-deactivation.md), [TICKET-006b](./ticket-006b-update-sddp-callsites.md)

## Context

### Background

Currently, `add_cut_with_preallocation()` calls `allocate_cut_slot()` which uses a free list and sequential counter. We need to replace this with the deterministic `compute_cut_slot()` method.

### Current Flow

```rust
pub fn add_cut_with_preallocation(&mut self, cut: &mut BendersCut) -> Option<usize> {
    let slot = self.allocate_cut_slot()?;  // Dynamic allocation
    // ... set coefficients and bounds ...
    self.cut_slot_to_id[slot] = Some(cut.id);  // Track mapping
    Some(slot)
}
```

### Target Flow

```rust
pub fn add_cut_with_preallocation(
    &mut self,
    cut: &mut BendersCut,
    iteration: usize,
    forward_pass_idx: usize,
) -> usize {
    let slot = self.compute_cut_slot(iteration, forward_pass_idx);  // Deterministic
    // ... set coefficients and bounds ...
    cut.slot_index = Some(slot);  // Store in cut for later deactivation
    slot
}
```

## Files to Read Before Starting

- `src/subproblem.rs` - Current `add_cut_with_preallocation()` (lines 1119-1149)
- `src/subproblem.rs` - Current `add_cut_to_model()` (lines 1155-1177)

## Specification

### Update add_cut_with_preallocation()

Change signature to accept iteration and forward_pass_idx:

```rust
/// Add cut constraint using preallocated slot with deterministic placement.
///
/// Uses `compute_cut_slot(iteration, forward_pass_idx)` to determine slot.
/// Stores slot index in the cut for O(1) deactivation lookup.
///
/// # Arguments
///
/// * `cut` - The Benders cut to add (slot_index will be set)
/// * `iteration` - 1-based iteration number
/// * `forward_pass_idx` - 0-based forward pass index
///
/// # Returns
///
/// The slot index where the cut was placed.
///
/// # Panics
///
/// Panics if slot exceeds preallocated count (via `compute_cut_slot`).
pub fn add_cut_with_preallocation(
    &mut self,
    cut: &mut cut::BendersCut,
    iteration: usize,
    forward_pass_idx: usize,
) -> usize {
    let slot = self.compute_cut_slot(iteration, forward_pass_idx);
    let row = self.slot_to_row(slot);

    if let Some(model) = self.model.as_mut() {
        // Update coefficients using the cached variable indices from State
        for (i, &var_idx) in self.cut_var_indices.iter().enumerate() {
            let coef = if i == 0 {
                1.0 // Alpha coefficient
            } else {
                -cut.coefficients[i - 1] // State variable coefficients (negated)
            };
            model
                .change_coefficient(row, var_idx, coef)
                .expect("Failed to set coefficient");
        }

        // Activate constraint by setting bounds: [rhs, ∞]
        model.change_rows_bounds(row, cut.rhs, f64::INFINITY);
    }

    // Store slot index in cut for O(1) deactivation lookup
    cut.slot_index = Some(slot);

    slot
}
```

### Update add_cut_to_model()

**CRITICAL**: Remove the fallback to dynamic allocation. This method should:
1. Require iteration and forward_pass_idx parameters
2. Panic if preallocation is not enabled (instead of falling back)

```rust
/// Add cut constraint to model using preallocated slot.
///
/// # Arguments
///
/// * `cut` - The Benders cut to add
/// * `iteration` - 1-based iteration number  
/// * `forward_pass_idx` - 0-based forward pass index
///
/// # Panics
///
/// Panics if:
/// - Preallocation is not enabled (call `preallocate_cut_constraints()` first)
/// - Slot exceeds preallocated count
pub fn add_cut_to_model(
    &mut self,
    cut: &mut cut::BendersCut,
    iteration: usize,
    forward_pass_idx: usize,
) {
    assert!(
        self.has_preallocated_cuts(),
        "Preallocation not enabled. Call preallocate_cut_constraints() before training."
    );
    
    self.add_cut_with_preallocation(cut, iteration, forward_pass_idx);
}
```

### Remove Slot Tracking Update

The line `self.cut_slot_to_id[slot] = Some(cut.id)` should be **removed** since we're storing slot in the cut instead. This will be fully cleaned up in TICKET-005b.

## Acceptance Criteria

- [x] `add_cut_with_preallocation()` accepts `iteration` and `forward_pass_idx`
- [x] Uses `compute_cut_slot()` instead of `allocate_cut_slot()`
- [x] Stores `slot_index` in the cut
- [x] `add_cut_to_model()` requires `iteration` and `forward_pass_idx`
- [x] `add_cut_to_model()` panics if preallocation not enabled (no fallback)
- [x] Removed update to `cut_slot_to_id` (or marked as TODO for TICKET-005b)
- [x] Code compiles (call sites will be updated in TICKET-006b)

## Implementation Guide

### Step 1: Update add_cut_with_preallocation()

1. Change signature to add `iteration: usize, forward_pass_idx: usize`
2. Replace `allocate_cut_slot()` call with `compute_cut_slot()`
3. Remove `Option<usize>` return - now returns `usize` (panics on error)
4. Add `cut.slot_index = Some(slot)`
5. Comment out or remove `self.cut_slot_to_id[slot] = Some(cut.id)`

### Step 2: Update add_cut_to_model()

1. Change signature to add `iteration: usize, forward_pass_idx: usize`
2. Remove dynamic allocation fallback code
3. Add panic assertion for preallocation
4. Call `add_cut_with_preallocation()` with new parameters

### Step 3: Temporarily Allow Compilation

The call sites in SDDP will be updated in TICKET-006b. For now:
- Either update the signatures only (code won't compile until TICKET-006b)
- Or add temporary dummy parameters and #[allow(unused)] (not recommended)

**Recommended approach**: Do TICKET-003b and TICKET-006b together as they're tightly coupled.

### Pitfalls to Avoid

- ⚠️ Don't forget to update the return type from `Option<usize>` to `usize`
- ⚠️ The cut has `iteration` and `forward_pass_idx` fields - these should match the parameters
- ⚠️ Call sites need updating (TICKET-006b) - coordinate if doing separately

## Testing Requirements

### Unit Tests

- [ ] Method accepts iteration and forward_pass_idx
- [ ] Slot index is stored in cut
- [ ] Panics on overflow

### Integration Tests

- [ ] Deferred to TICKET-006b when call sites are updated

## Documentation Requirements

- [ ] Updated doc comments for both methods
- [ ] Note about panic behavior

## Effort Estimate

**Points**: 2  
**Confidence**: Medium  
**Rationale**: Simple changes but coupled with TICKET-006b

## Definition of Done

- [x] Signatures updated
- [x] Deterministic slot calculation used
- [x] Slot stored in cut
- [x] Fallback removed
- [x] Ready for TICKET-006b call site updates
