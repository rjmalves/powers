# [TICKET-006] Add cut slot reuse for selection

> **Epic**: [Epic 1: HiGHS Constraint Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 2](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-004](./ticket-004-cut-addition-update.md), [TICKET-005](./ticket-005-cut-removal.md)  
> **Blocks**: [TICKET-007](./ticket-007-integration.md)

## Context

### Background

With cut selection enabled, cuts are frequently added and removed. The slot reuse mechanism ensures we don't exhaust preallocated slots by recycling freed slots for new cuts.

### Relation to Epic

This ticket ensures the preallocation system works correctly with cut selection, which is the common case in production.

### Current State

TICKET-004 and TICKET-005 implement basic slot allocation and freeing. This ticket ensures they work together correctly and handles edge cases.

## Files to Read Before Starting

- `src/fcf.rs` - Cut selection logic (domination evaluation)
- `src/sddp/mod.rs` - Backward pass where cuts are added/removed

## Specification

### Requirements

1. **Slot Reuse Order**: LIFO (last-freed slot is first-reused)
   - Keeps recently-used slots hot in cache
   - Simple stack-based implementation

2. **Deterministic Allocation**: Same sequence of add/remove produces same slot assignments
   - Important for reproducibility
   - LIFO naturally provides this

3. **Graceful Exhaustion Handling**: Log warning, then fallback to dynamic allocation
   - Don't panic in production
   - Allow algorithm to complete (slower but correct)

### Modified Behavior

Update `allocate_cut_slot()` to handle exhaustion gracefully:

```rust
fn allocate_cut_slot(&mut self) -> Option<usize> {
    // Prefer freed slots
    if let Some(slot) = self.free_cut_slots.pop() {
        return Some(slot);
    }
    
    // Use sequential if available
    if self.next_available_cut_slot < self.num_preallocated_cuts {
        let slot = self.next_available_cut_slot;
        self.next_available_cut_slot += 1;
        return Some(slot);
    }
    
    // Exhausted - return None for fallback
    None
}
```

### Fallback Path

When `allocate_cut_slot()` returns `None`:

```rust
let row = match self.allocate_cut_slot() {
    Some(slot) => self.slot_to_row(slot),
    None => {
        log::warn!(
            "Cut slots exhausted ({} preallocated), falling back to dynamic allocation",
            self.num_preallocated_cuts
        );
        // Fallback to dynamic add_row
        model.add_row(cut.rhs.., factors)
    }
};
```

## Acceptance Criteria

- [ ] Freed slots are reused for new cuts
- [ ] Slot exhaustion logs warning (not panic)
- [ ] Fallback to dynamic allocation works
- [ ] Example 07 with cut selection works correctly
- [ ] Multiple add/remove cycles work correctly

## Implementation Guide

### Suggested Approach

1. Modify `allocate_cut_slot()` to return `Option<usize>`
2. Update cut addition to handle `None` with fallback
3. Test with high iteration count to stress slot reuse
4. Verify determinism with fixed seed

### Key Files to Modify

- `src/subproblem.rs`: Update allocation and addition logic

### Code Template

```rust
/// Allocate a cut slot, returning None if exhausted.
///
/// Prefers reusing freed slots (LIFO order), then uses sequential
/// allocation. Returns None when all slots are exhausted.
fn allocate_cut_slot(&mut self) -> Option<usize> {
    // Prefer freed slots (LIFO for cache locality)
    if let Some(slot) = self.free_cut_slots.pop() {
        return Some(slot);
    }
    
    // Use next sequential slot
    if self.next_available_cut_slot < self.num_preallocated_cuts {
        let slot = self.next_available_cut_slot;
        self.next_available_cut_slot += 1;
        return Some(slot);
    }
    
    // All slots exhausted
    None
}

/// Track whether we've warned about slot exhaustion
/// (avoid log spam)
static WARNED_EXHAUSTION: std::sync::atomic::AtomicBool = 
    std::sync::atomic::AtomicBool::new(false);

fn add_cut_constraint_to_model(...) {
    match self.allocate_cut_slot() {
        Some(slot) => {
            // Use preallocated slot (fast path)
            let row = self.slot_to_row(slot);
            // ... coefficient updates ...
            self.cut_slot_to_id[slot] = Some(cut.id);
        }
        None => {
            // Fallback to dynamic allocation (slow path)
            if !WARNED_EXHAUSTION.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::warn!(
                    "Cut slots exhausted (preallocated {}). \
                     Falling back to dynamic allocation. \
                     Consider increasing num_iterations × num_forward_passes estimate.",
                    self.num_preallocated_cuts
                );
            }
            
            // Dynamic allocation (original path)
            let mut factors = Vec::with_capacity(dimension + 1);
            factors.push((variables.alpha, 1.0));
            for (i, &var) in variables.stored_volume.iter().enumerate() {
                factors.push((var, -cut.coefficients[i]));
            }
            model.add_row(cut.rhs.., factors);
        }
    }
}
```

### Test Scenario

To stress-test slot reuse:

1. Use example 07 with many iterations
2. Verify cut selection removes cuts
3. Verify new cuts reuse freed slots
4. Check no exhaustion warnings

### Pitfalls to Avoid

- ⚠️ Don't panic on exhaustion in production
- ⚠️ Warn only once to avoid log spam
- ⚠️ Fallback path must match original behavior exactly
- ⚠️ Ensure thread-safety if parallel backward pass

## Testing Requirements

### Integration Tests

- [ ] Example 07: 20 iterations with cut selection
- [ ] Verify no exhaustion warnings with default settings
- [ ] Force exhaustion (low preallocation) and verify fallback works

### Stress Test

```bash
# Increase iterations to stress slot reuse
# (modify config temporarily)
cargo run --release -- run examples/07-par-model-with-inflow-state
```

## Documentation Requirements

- [ ] Doc comment explaining LIFO reuse strategy
- [ ] Comment on fallback path

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward logic building on previous tickets

## Definition of Done

- [ ] Slot reuse works correctly
- [ ] Exhaustion handled gracefully
- [ ] No log spam on exhaustion
- [ ] Examples work correctly
- [ ] Deterministic behavior verified
