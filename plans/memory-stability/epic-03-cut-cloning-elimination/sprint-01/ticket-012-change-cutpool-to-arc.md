# [TICKET-012] Change BendersCutPool to use Arc<BendersCut>

> **Epic**: [Epic 3: Cut Cloning Elimination](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: Epic 2 complete
> **Blocks**: [TICKET-013](./ticket-013-update-handler-application.md)

## Context

### Background

Currently, cuts are cloned when passed to handlers for parallel application. This creates ~80 MB of transient allocations. By storing `Arc<BendersCut>` in the pool, we can clone the Arc (~16 bytes) instead of the data (~1 KB per cut).

### Relation to Epic

Core change that enables cheap cut sharing.

### Current State

```rust
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
    // ...
}

// Cloning ~1 KB per cut
let cuts: Vec<(usize, BendersCut)> = /* full clone */;
```

## Files to Read Before Starting

- `src/cut.rs` - BendersCut and BendersCutPool
- `src/fcf.rs` - Cut pool access patterns
- `src/sddp/mod.rs:2166-2180` - Handler cut application (cloning location)

## Specification

### Structural Changes

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug)]
pub struct BendersCut {
    pub id: usize,
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    pub active: AtomicBool,  // Changed from bool
    pub non_dominated_state_count: AtomicUsize,  // Changed from usize
    pub iteration: usize,
    pub forward_pass_idx: usize,
    pub slot_index: Option<usize>,
    pub populated: bool,
}

pub struct BendersCutPool {
    pub pool: Vec<Arc<BendersCut>>,  // Changed from Vec<BendersCut>
    pub active_cut_indices: HashMap<usize, usize>,
    pub total_cut_count: usize,
    num_forward_passes: usize,
}
```

### Atomic Field Access

```rust
impl BendersCut {
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
    
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }
    
    pub fn get_non_dominated_count(&self) -> usize {
        self.non_dominated_state_count.load(Ordering::Relaxed)
    }
    
    pub fn increment_non_dominated_count(&self) {
        self.non_dominated_state_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn decrement_non_dominated_count(&self) {
        // Use saturating subtraction
        self.non_dominated_state_count.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |x| Some(x.saturating_sub(1)),
        );
    }
}
```

### Pool Access Changes

```rust
impl BendersCutPool {
    pub fn preallocate(...) -> Self {
        let pool: Vec<Arc<BendersCut>> = (0..total_cuts)
            .map(|id| Arc::new(BendersCut { ... }))
            .collect();
        // ...
    }
    
    // For mutation, need Arc::get_mut or interior mutability
    pub fn update_cut(&mut self, ...) -> usize {
        let cut = Arc::get_mut(&mut self.pool[slot])
            .expect("Cannot mutate cut with multiple references");
        cut.update(coefficients, rhs, iteration, forward_pass_idx);
        slot
    }
}
```

### Mutation Strategy

Since cuts are mutated during `add_cuts_batch()` while the FCF lock is held, we can use `Arc::get_mut()` which succeeds when there's only one reference. Handlers receive Arc clones only after batch processing completes.

## Acceptance Criteria

- [ ] BendersCutPool uses `Vec<Arc<BendersCut>>`
- [ ] Atomic fields for `active` and `non_dominated_state_count`
- [ ] Helper methods for atomic access
- [ ] All existing tests pass (with updated assertions)
- [ ] Preallocation still works with Arc

## Implementation Guide

### Suggested Approach

1. Add atomic fields to BendersCut
2. Add helper methods for atomic access
3. Change pool to use Vec<Arc<BendersCut>>
4. Update preallocate() to wrap in Arc
5. Update all pool access patterns
6. Update tests for atomic types

### Key Files to Modify

- `src/cut.rs`: Add atomics, change pool type
- `src/fcf.rs`: Update all cut pool access

### Pitfalls to Avoid

- ⚠️ `Arc::get_mut()` panics with multiple refs - ensure single owner during mutation
- ⚠️ Atomic Ordering::Relaxed is fine for counters, but document assumption
- ⚠️ BendersCut can't derive Clone anymore (atomics don't Clone)

## Testing Requirements

### Unit Tests

- [ ] Test Arc wrapping in preallocate
- [ ] Test atomic field access
- [ ] Test Arc::get_mut works during update
- [ ] Test helper methods for atomic ops

### Performance Tests

- [ ] Benchmark atomic operations vs regular ops
- [ ] Verify no significant overhead

## Documentation Requirements

- [ ] Document atomic field access patterns
- [ ] Note about mutation requirements (single owner)

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Significant structural change, atomic patterns need care
