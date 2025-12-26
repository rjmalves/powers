# [TICKET-004b] Update cut deactivation to use stored slot index

> **Epic**: [Epic 1b: Full Memory Determinism](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-003b](./ticket-003b-update-cut-addition.md)  
> **Blocks**: [TICKET-005b](./ticket-005b-remove-slot-tracking.md)

## Context

### Background

Currently, `deactivate_cut_constraint(cut_id)` performs a **linear search** through `cut_slot_to_id` to find the slot for a given cut ID. With the new design, the slot is stored directly in the `BendersCut`, enabling O(1) lookup.

### Current Flow (O(n) lookup)

```rust
pub fn deactivate_cut_constraint(&mut self, cut_id: usize) -> bool {
    // Linear search through cut_slot_to_id
    let slot = self.cut_slot_to_id
        .iter()
        .position(|&id| id == Some(cut_id));
    // ...
}
```

### Target Flow (O(1) lookup)

```rust
pub fn deactivate_cut_constraint(&mut self, cut: &BendersCut) -> bool {
    let slot = cut.slot_index?;  // O(1) - stored in cut
    // ...
}
```

## Files to Read Before Starting

- `src/subproblem.rs` - Current `deactivate_cut_constraint()` (lines 1191-1227)
- `src/subproblem.rs` - Current `remove_cut_from_model()` (lines 1238-1248)

## Specification

### Update deactivate_cut_constraint()

Change to accept a reference to the cut instead of just the ID:

```rust
/// Deactivate a cut constraint by relaxing its bounds.
///
/// The constraint remains in the model but becomes trivially satisfied
/// with bounds `[-∞, ∞]`. Uses stored `slot_index` for O(1) lookup.
///
/// # Arguments
///
/// * `cut` - Reference to the cut with stored slot_index
///
/// # Returns
///
/// `true` if cut was deactivated, `false` if:
/// - Preallocation not enabled
/// - Cut has no slot_index (wasn't added via preallocation)
/// - Model not initialized
pub fn deactivate_cut_constraint(&mut self, cut: &cut::BendersCut) -> bool {
    if !self.has_preallocated_cuts() {
        return false;
    }

    // O(1) lookup via stored slot index
    let slot = match cut.slot_index {
        Some(s) => s,
        None => return false,
    };

    if self.model.is_none() {
        return false;
    }

    let row = self.slot_to_row(slot);
    let model = self.model.as_mut().unwrap();

    // Relax bounds to deactivate: [-∞, ∞] is trivially satisfied
    model.change_rows_bounds(row, f64::NEG_INFINITY, f64::INFINITY);

    true
}
```

**Key changes**:
- Takes `&cut::BendersCut` instead of `cut_id: usize`
- Uses `cut.slot_index` directly (O(1) instead of O(n))
- Removes update to `cut_slot_to_id` (will be removed in TICKET-005b)
- Removes push to `free_cut_slots` (will be removed in TICKET-005b)

### Update remove_cut_from_model()

Update to accept the cut instead of just the ID:

```rust
/// Remove cut from model via preallocation deactivation.
///
/// # Arguments
///
/// * `cut` - Reference to the cut to remove
/// * `_row_index` - Legacy parameter (unused with preallocation, kept for compatibility)
pub fn remove_cut_from_model(&mut self, cut: &cut::BendersCut, _row_index: usize) {
    self.deactivate_cut_constraint(cut);
    // Note: With preallocation, we never need the fallback delete_row path
}
```

Alternatively, simplify to just call `deactivate_cut_constraint()` directly from callers.

### Update Call Sites

The main caller is `apply_aggregated_cut_selection_result()`. Update it to pass the cut:

```rust
// Before
for (cut_id, index) in removals {
    let row_idx = self.first_cut_row_index() + index;
    self.remove_cut_from_model(cut_id, row_idx);
}

// After
for cut in cuts_to_remove {
    self.deactivate_cut_constraint(cut);
}
```

This may require restructuring how removed cuts are tracked.

## Acceptance Criteria

- [x] `deactivate_cut_constraint()` takes `&BendersCut` not `cut_id`
- [x] Uses `cut.slot_index` for O(1) lookup
- [x] Removes linear search through `cut_slot_to_id`
- [x] Removes push to `free_cut_slots` (or marks as TODO for TICKET-005b)
- [x] `remove_cut_from_model()` updated or inlined
- [x] Call sites updated
- [x] Code compiles

## Implementation Guide

### Step 1: Update deactivate_cut_constraint()

1. Change parameter from `cut_id: usize` to `cut: &cut::BendersCut`
2. Replace linear search with `cut.slot_index`
3. Comment out `self.cut_slot_to_id[slot] = None`
4. Comment out `self.free_cut_slots.push(slot)`

### Step 2: Update remove_cut_from_model()

1. Change parameter to accept cut reference
2. Remove fallback to `delete_row` (or keep as dead code until TICKET-005b)

### Step 3: Update apply_aggregated_cut_selection_result()

This is the main caller. The challenge is that we receive `removing_cut_ids: HashSet<usize>` but need the actual cuts to get their slot indices.

**Solution**: Look up cuts in `cuts_to_add` parameter which contains cloned cuts:

```rust
// Get cuts to remove from the cuts_to_add slice
for &cut_id in &aggregated_result.removing_cut_ids {
    // Find cut in cuts_to_add by ID
    if let Some((_, cut)) = cuts_to_add.iter().find(|(id, _)| *id == cut_id) {
        self.deactivate_cut_constraint(cut);
    }
}
```

**Alternative**: Store slot_index in a simpler structure or pass it explicitly.

### Pitfalls to Avoid

- ⚠️ Cuts to remove may not be in `cuts_to_add` (they're from previous iterations)
- ⚠️ May need to look up cuts in FCF pool instead
- ⚠️ Consider passing slot indices explicitly in AggregatedCutSelectionResult

## Alternative Design

If looking up cuts for their slot_index is complex, consider:

1. **Store (cut_id, slot_index) pairs in removal list**:
   ```rust
   removing_cut_slots: Vec<(usize, usize)>  // (cut_id, slot_index)
   ```

2. **Compute slot from iteration/fp stored in the cut**:
   ```rust
   fn deactivate_cut_by_id(&mut self, cut_id: usize, iteration: usize, forward_pass_idx: usize) {
       let slot = self.compute_cut_slot(iteration, forward_pass_idx);
       // deactivate at slot
   }
   ```

Evaluate which approach is cleanest for the call sites.

## Testing Requirements

### Unit Tests

- [ ] O(1) slot lookup works correctly
- [ ] Deactivation with stored slot succeeds
- [ ] Deactivation with None slot_index returns false

### Integration Tests

- [ ] Examples 01 and 07 pass
- [ ] Cut selection (removal) works correctly

## Documentation Requirements

- [ ] Updated doc comments
- [ ] Note about O(1) performance

## Effort Estimate

**Points**: 1  
**Confidence**: Medium  
**Rationale**: Simple change but call site updates need care

## Definition of Done

- [x] O(1) slot lookup implemented
- [x] Linear search removed
- [x] Call sites updated
- [x] Examples pass
