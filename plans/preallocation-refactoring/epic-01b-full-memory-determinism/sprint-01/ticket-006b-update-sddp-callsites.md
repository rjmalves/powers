# [TICKET-006b] Update SDDP call sites for deterministic slots

> **Epic**: [Epic 1b: Full Memory Determinism](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-003b](./ticket-003b-update-cut-addition.md)  
> **Blocks**: [TICKET-007b](./ticket-007b-validation.md)

## Context

### Background

After updating `add_cut_to_model()` to require `iteration` and `forward_pass_idx` parameters, all call sites in the SDDP module need to be updated to pass these values.

The good news: **iteration and forward_pass_idx are already available** at all call sites because `BendersCut` stores them.

### Key Call Sites

1. **apply_aggregated_cut_selection_result()** in `src/subproblem.rs`
2. **add_cuts_batch()** in `src/fcf.rs` (if cuts are added there)
3. **SDDP backward pass** in `src/sddp/mod.rs`

## Files to Read Before Starting

- `src/subproblem.rs` - `apply_aggregated_cut_selection_result()` (lines 1733-1780)
- `src/sddp/mod.rs` - Backward pass cut handling (lines 2070-2180)
- `src/cut.rs` - BendersCut struct (has iteration, forward_pass_idx fields)

## Specification

### Update apply_aggregated_cut_selection_result()

Currently adds cuts by cloning and calling `add_cut_to_model()`:

```rust
// Current:
for (_cut_id, cut) in cuts_to_process {
    let mut cut_copy = cut.clone();
    self.add_cut_to_model(&mut cut_copy);
}
```

Update to pass iteration and forward_pass_idx from the cut:

```rust
// Updated:
for (_cut_id, cut) in cuts_to_process {
    let mut cut_copy = cut.clone();
    self.add_cut_to_model(&mut cut_copy, cut.iteration, cut.forward_pass_idx);
}
```

### Update preallocate_cut_constraints() call sites

In `src/sddp/mod.rs`, the preallocation call needs `num_forward_passes`:

```rust
// Current:
handler.preallocate_cut_constraints(max_cuts_per_node)?;

// Updated:
handler.preallocate_cut_constraints(max_cuts_per_node, num_forward_passes)?;
```

Find all call sites:
```bash
grep -n "preallocate_cut_constraints" src/
```

### Consider: Returning Cuts

Cuts that "return" (were deactivated but become active again) need to be re-added at the **same slot**. Since the slot is computed from (iteration, forward_pass_idx), returning a cut will correctly place it at its original slot.

**Important**: When a returning cut is re-added, its slot_index should already be `Some(slot)` from when it was first added. The `add_cut_with_preallocation()` will:
1. Compute the slot (same as before since iteration/fp unchanged)
2. Overwrite slot_index with the same value
3. Re-set coefficients and bounds

This should work correctly without changes.

### Consider: Cut Removal (Deactivation)

Cuts to remove come via `aggregated_result.removing_cut_ids`. We need access to the cuts to get their `slot_index`.

**Option 1**: Look up cuts by ID in the cuts_to_add parameter or FCF pool.

**Option 2**: Compute slot from cut's stored iteration/forward_pass_idx:
```rust
for &cut_id in &aggregated_result.removing_cut_ids {
    if let Some((_, cut)) = cuts_to_add.iter().find(|(id, _)| *id == cut_id) {
        self.deactivate_cut_constraint(cut);
    }
}
```

**Option 3**: Change `AggregatedCutSelectionResult` to include slot indices:
```rust
pub struct AggregatedCutSelectionResult {
    pub new_cut_ids: HashSet<usize>,
    pub returning_cut_ids: HashSet<usize>,
    pub removing_cuts: Vec<(usize, usize)>,  // (cut_id, slot_index)
}
```

**Recommended**: Option 2 is simplest - iterate through the cuts we have and deactivate by reference.

## Acceptance Criteria

- [x] `apply_aggregated_cut_selection_result()` passes iteration/fp to `add_cut_to_model()`
- [x] `preallocate_cut_constraints()` calls pass `num_forward_passes`
- [x] Cut deactivation works with the new approach
- [x] Returning cuts are correctly re-added at their original slots
- [x] Code compiles
- [x] Examples pass

## Implementation Guide

### Step 1: Update preallocate_cut_constraints() call sites

Find all calls:
```bash
grep -rn "preallocate_cut_constraints" src/
```

Update each to pass `num_forward_passes`.

### Step 2: Update apply_aggregated_cut_selection_result()

Add iteration/forward_pass_idx to the `add_cut_to_model()` call:

```rust
self.add_cut_to_model(&mut cut_copy, cut.iteration, cut.forward_pass_idx);
```

### Step 3: Update cut deactivation logic

Restructure the removal loop to use cuts instead of just cut_ids:

```rust
// Instead of iterating over removing_cut_ids,
// iterate over cuts and check if they should be removed
for (cut_id, cut) in cuts_to_add {
    if aggregated_result.removing_cut_ids.contains(cut_id) {
        self.deactivate_cut_constraint(cut);
    }
}
```

Note: `cuts_to_add` contains new and returning cuts but may not contain cuts to remove (which were added in previous iterations). This needs careful handling.

**Alternative approach**: Store all cuts with their slots in the FCF, then look them up:
```rust
// In SDDP phase 3a, collect cuts to remove with their slots
let cuts_to_remove: Vec<(usize, usize)> = aggregated_result.removing_cut_ids
    .iter()
    .filter_map(|&cut_id| {
        fcf_locked.cut_pool.pool.get(cut_id)
            .and_then(|cut| cut.slot_index.map(|slot| (cut_id, slot)))
    })
    .collect();

// Pass to apply_aggregated_cut_selection_result()
```

### Step 4: Handle SddpTrainHandler.apply_aggregated_cut_result()

This method in `src/sddp/mod.rs` calls `apply_aggregated_cut_selection_result()`. Ensure it passes the right data structure.

### Pitfalls to Avoid

- ⚠️ Cuts to remove may have been added in previous iterations (not in cuts_to_add)
- ⚠️ Need to get slot_index from FCF pool for removal
- ⚠️ Consider thread safety when accessing FCF pool for slot lookup

## Testing Requirements

### Unit Tests

- [ ] New cut addition with iteration/fp works
- [ ] Returning cut re-addition works
- [ ] Cut deactivation works

### Integration Tests

- [ ] Example 01 passes
- [ ] Example 07 passes with cut selection
- [ ] Results identical to before

## Documentation Requirements

- [ ] Update method doc comments to reflect new parameters

## Effort Estimate

**Points**: 2  
**Confidence**: Medium  
**Rationale**: Multiple call sites and some logic restructuring needed

## Definition of Done

- [x] All call sites updated
- [x] Cut addition/return/removal all work
- [x] Code compiles
- [x] Examples pass with identical results
