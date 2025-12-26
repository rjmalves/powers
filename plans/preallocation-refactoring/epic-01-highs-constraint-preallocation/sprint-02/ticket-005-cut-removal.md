# [TICKET-005] Implement cut removal via bound relaxation

> **Epic**: [Epic 1: HiGHS Constraint Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 2](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-003](../sprint-01/ticket-003-subproblem-cut-slots.md)  
> **Blocks**: [TICKET-006](./ticket-006-slot-reuse.md)

## Context

### Background

Cut selection removes dominated cuts from the active set. Currently this may use `model.delete_row()`. With preallocation, we instead deactivate cuts by relaxing bounds to `[-∞, ∞]`, making the constraint trivially satisfied without removing the row.

### Relation to Epic

This enables slot reuse: when a cut is "removed", its slot becomes available for a new cut.

### Current State

Cut removal (if implemented) likely uses `model.delete_row()` or similar. This needs to change to bound relaxation.

## Files to Read Before Starting

- `src/subproblem.rs` - Find cut removal logic
- `src/cut.rs` - BendersCut structure
- `HIGHS_SOLVER_PREALLOCATION_ANALYSIS.md` - Lines 255-285

## Specification

### Modified/New Method

**`deactivate_cut_constraint(cut_id: usize)`** or modify existing removal:

1. Look up slot from cut ID (via `cut_slot_to_id` mapping)
2. Relax bounds: `model.change_rows_bounds(row, -∞, ∞)`
3. Mark slot as free in `cut_slot_to_id`
4. Push slot to `free_cut_slots` stack

### Behavior

- Constraint remains in matrix but is trivially satisfied
- No row deletion (preserves sparse matrix structure)
- Slot can be reused for new cuts

### Error Handling

- Log warning if cut ID not found in mapping
- Return without action (idempotent)

## Acceptance Criteria

- [ ] Cut removal uses bound relaxation, not row deletion
- [ ] Slot is marked free after deactivation
- [ ] Slot is added to free list for reuse
- [ ] Examples 01 and 07 still produce correct results

## Implementation Guide

### Suggested Approach

1. Find existing cut removal logic
2. Replace with bound relaxation
3. Update slot tracking structures
4. Verify cut selection still works

### Key Files to Modify

- `src/subproblem.rs`: Modify or add cut deactivation method

### Code Template

```rust
/// Deactivate a cut constraint by relaxing its bounds.
///
/// The constraint remains in the model but becomes trivially satisfied
/// with bounds `[-∞, ∞]`. The slot is marked as free for reuse.
///
/// # Arguments
///
/// * `cut_id` - ID of the cut to deactivate
///
/// # Returns
///
/// `true` if cut was deactivated, `false` if not found
pub fn deactivate_cut_constraint(&mut self, cut_id: usize) -> bool {
    let model = match self.model.as_mut() {
        Some(m) => m,
        None => return false,
    };
    
    // Find slot for this cut
    let slot = self.cut_slot_to_id.iter()
        .position(|&id| id == Some(cut_id));
    
    let slot = match slot {
        Some(s) => s,
        None => {
            log::warn!("Cut {} not found in slot mapping", cut_id);
            return false;
        }
    };
    
    let row = self.slot_to_row(slot);
    
    // Relax bounds to deactivate: [-∞, ∞] is trivially satisfied
    model.change_rows_bounds(row, f64::NEG_INFINITY, f64::INFINITY);
    
    // Mark slot as free
    self.cut_slot_to_id[slot] = None;
    
    // Add to free list for reuse
    self.free_cut_slots.push(slot);
    
    true
}

/// Free a cut slot directly (when slot index is known).
///
/// Use this when the slot is known, e.g., from cut selection.
fn free_cut_slot(&mut self, slot: usize) {
    if let Some(model) = self.model.as_mut() {
        let row = self.slot_to_row(slot);
        model.change_rows_bounds(row, f64::NEG_INFINITY, f64::INFINITY);
    }
    
    self.cut_slot_to_id[slot] = None;
    self.free_cut_slots.push(slot);
}
```

### Integration with Cut Selection

Find where cut selection removes cuts and update to use deactivation:

```rust
// Instead of:
model.delete_row(row_index);

// Use:
subproblem.deactivate_cut_constraint(cut_id);
// OR
subproblem.free_cut_slot(slot);
```

### Pitfalls to Avoid

- ⚠️ Don't call `delete_row()` on preallocated cuts
- ⚠️ Ensure slot is added to free list after deactivation
- ⚠️ `cut_slot_to_id[slot] = None` must happen before pushing to free list

## Testing Requirements

### Unit Tests

Deferred to TICKET-007 integration validation.

### Integration Tests

- [ ] Example 07 with cut selection enabled still converges
- [ ] Active cut count matches expected after selection

### Validation

```bash
# Example 07 has cut selection enabled
cargo run --release -- run examples/07-par-model-with-inflow-state 2>&1 | tail -20
```

## Documentation Requirements

- [ ] Doc comment on deactivation method
- [ ] Explain why bounds `[-∞, ∞]` deactivates constraint

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Simple bound change and bookkeeping

## Definition of Done

- [ ] Cut deactivation uses bound relaxation
- [ ] Slot marked free and added to reuse list
- [ ] No `delete_row()` in cut removal path
- [ ] Examples with cut selection work correctly
