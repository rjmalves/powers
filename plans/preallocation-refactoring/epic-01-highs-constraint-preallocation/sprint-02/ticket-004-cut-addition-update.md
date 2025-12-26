# [TICKET-004] Update cut addition to use coefficient changes

> **Epic**: [Epic 1: HiGHS Constraint Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 2](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-003](../sprint-01/ticket-003-subproblem-cut-slots.md)  
> **Blocks**: [TICKET-006](./ticket-006-slot-reuse.md), [TICKET-007](./ticket-007-integration.md)

## Context

### Background

Currently, `add_cut_constraint_to_model()` uses `model.add_row()` which causes HiGHS to allocate memory for a new row. We need to modify this to use preallocated slots via coefficient updates.

### Relation to Epic

This is the core change that eliminates dynamic allocation during cut addition.

### Current State

In `src/subproblem.rs` or `src/state.rs`, there's a method like:
```rust
fn add_cut_constraint_to_model(...) {
    let mut factors = Vec::with_capacity(self.dimension + 1);
    // ... build factors ...
    model.add_row(cut.rhs.., factors);  // ❌ Dynamic allocation
}
```

## Files to Read Before Starting

- `src/subproblem.rs` - Find `add_cut_constraint_to_model` or similar
- `src/state.rs` - May contain cut addition logic
- `HIGHS_SOLVER_PREALLOCATION_ANALYSIS.md` - Lines 205-252

## Specification

### Modified Behavior

1. Allocate a slot from preallocated pool (from free list or sequential)
2. Update alpha coefficient via `model.change_coefficient(row, alpha, 1.0)`
3. Update storage coefficients via `model.change_coefficient(row, storage[i], -coef[i])`
4. Activate constraint via `model.change_rows_bounds(row, cut.rhs, INFINITY)`
5. Record slot → cut ID mapping

### New Helper Method

**`allocate_cut_slot() -> usize`**:
- Return slot from `free_cut_slots` if available
- Otherwise return `next_available_cut_slot` and increment
- Panic if all slots exhausted (temporary; TICKET-006 handles gracefully)

### Inputs

Same as current `add_cut_constraint_to_model()`

### Outputs

Same as current (adds constraint to model, updates cut state)

### Error Handling

- Panic if slots exhausted (with informative message)
- Future: graceful fallback to dynamic allocation

## Acceptance Criteria

- [ ] `add_cut_constraint_to_model()` uses coefficient changes, not `add_row()`
- [ ] `allocate_cut_slot()` helper implemented
- [ ] Cut's model row index stored for later removal
- [ ] Examples 01 and 07 produce identical results
- [ ] No `add_row()` calls in cut addition path

## Implementation Guide

### Suggested Approach

1. Find `add_cut_constraint_to_model()` in codebase
2. Add `allocate_cut_slot()` helper method
3. Replace `model.add_row()` with coefficient updates
4. Store row index in cut or mapping structure
5. Test with examples

### Key Files to Modify

- `src/subproblem.rs` or `src/state.rs`: Modify cut addition logic

### Code Template

```rust
// Add helper method

/// Allocate a preallocated cut slot.
///
/// Returns the slot index (0-based from first_preallocated_cut_row).
/// Reuses freed slots when available (LIFO order).
///
/// # Panics
///
/// Panics if all preallocated slots are exhausted.
fn allocate_cut_slot(&mut self) -> usize {
    // Prefer reusing freed slots
    if let Some(slot) = self.free_cut_slots.pop() {
        return slot;
    }
    
    // Use next sequential slot
    let slot = self.next_available_cut_slot;
    if slot >= self.num_preallocated_cuts {
        panic!(
            "Cut slot exhaustion! Preallocated {} slots but need more. \
             Consider increasing iterations × forward_passes estimate.",
            self.num_preallocated_cuts
        );
    }
    self.next_available_cut_slot += 1;
    slot
}

/// Get HiGHS row index for a cut slot.
fn slot_to_row(&self, slot: usize) -> usize {
    self.first_preallocated_cut_row + slot
}
```

```rust
// Modified add_cut_constraint_to_model

fn add_cut_constraint_to_model(
    &mut self,
    cut: &mut cut::BendersCut,
    variables: &subproblem::Variables,
    model: &mut solver::Model,
) {
    // Allocate slot
    let slot = self.allocate_cut_slot();
    let row = self.slot_to_row(slot);
    
    // Update alpha coefficient (always 1.0)
    model.change_coefficient(row, variables.alpha, 1.0)
        .expect("Failed to set alpha coefficient");
    
    // Update storage variable coefficients
    for (hydro_id, &var_idx) in variables.stored_volume.iter().enumerate() {
        let coef = -cut.coefficients[hydro_id];
        model.change_coefficient(row, var_idx, coef)
            .expect("Failed to set storage coefficient");
    }
    
    // Activate constraint by setting bounds: [rhs, ∞]
    model.change_rows_bounds(row, cut.rhs, f64::INFINITY);
    
    // Store slot mapping
    self.cut_slot_to_id[slot] = Some(cut.id);
    
    // Store row index on cut for removal (optional, depends on cut structure)
    // cut.model_row_index = Some(row);
}
```

### Pitfalls to Avoid

- ⚠️ Don't call `model.add_row()` anywhere in the new path
- ⚠️ Coefficients are negated: `−cut.coefficients[i]` for storage
- ⚠️ Bounds for active cut: `[rhs, ∞]` (lower bound is RHS)
- ⚠️ Alpha coefficient is `+1.0`, not `-1.0`

## Testing Requirements

### Unit Tests

Deferred to TICKET-007 integration validation.

### Integration Tests

- [ ] Example 01: Deterministic case still converges to same value
- [ ] Example 07: Stochastic case still produces correct lower bounds

### Validation

```bash
# Run examples and check output
cargo run --release -- run examples/01-deterministic 2>&1 | grep "lower"
cargo run --release -- run examples/07-par-model-with-inflow-state 2>&1 | grep "lower"
```

## Documentation Requirements

- [ ] Doc comment on `allocate_cut_slot()`
- [ ] Inline comments explaining coefficient update logic

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Core logic change, requires finding and modifying existing code

## Definition of Done

- [ ] Cut addition uses coefficient changes
- [ ] No `add_row()` in cut addition path
- [ ] Examples produce correct results
- [ ] Code compiles without warnings
