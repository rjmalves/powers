# [TICKET-002] Update cut consumers to work with Arc

> **Epic**: [Epic 2: Cut Cloning Elimination](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-001](./ticket-001-update-cutpool-to-arc.md)  
> **Blocks**: None

## Context

### Background

After changing `BendersCutPool` to store `Arc<BendersCut>`, all code that accesses cuts needs to be updated. Most accesses are read-only, but there is one critical mutation site.

### Mutation Discovered

```rust
// src/sddp/mod.rs:2155-2157
if let Some(cut) = fcf_locked.cut_pool.pool.get_mut(cut_id) {
    cut.active = false;  // MUTATES CUT!
}
```

This requires either:
1. **Use `Arc<RwLock<BendersCut>>`** - Adds locking overhead
2. **Separate active tracking** - Keep `active` flag outside cut struct
3. **Use interior mutability** - `AtomicBool` for `active` field

### Files to Read Before Starting

- `src/cut.rs` - BendersCut fields
- `src/sddp/mod.rs` - Cut usage (~lines 2150-2200, 776-830)
- `src/fcf.rs` - add_cut method
- `src/subproblem.rs` - Cut application

## Specification

### Option A: AtomicBool for `active` (RECOMMENDED)

Change only the `active` field to `AtomicBool`, preserving Arc without locking:

```rust
pub struct BendersCut {
    pub id: usize,
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    pub active: AtomicBool,  // Changed from bool
    // ...
}
```

**Pros**: Minimal changes, no locking, Arc works  
**Cons**: Slightly different API for active flag

### Option B: Separate active tracking

Keep `active_cut_indices` HashMap as source of truth for active status:

```rust
// Instead of cut.active = false:
fcf_locked.cut_pool.active_cut_indices.remove(&cut_id);
// Check active via:
let is_active = fcf_locked.cut_pool.active_cut_indices.contains_key(&cut_id);
```

**Pros**: No struct change, HashMap already exists  
**Cons**: Need to audit all `cut.active` usages

### Changes Required

1. **src/fcf.rs**: `add_cut` wraps in Arc
2. **src/sddp/mod.rs:2155-2157**: Change mutation strategy
3. **src/sddp/mod.rs:2196**: Use `Arc::clone()` instead of `.clone()`
4. **src/sddp/mod.rs:776**: Update function signature
5. **src/subproblem.rs**: Update cut coefficient access

## Acceptance Criteria

- [ ] All cut pool accesses work with `Arc<BendersCut>`
- [ ] Cut mutation uses AtomicBool or separate tracking
- [ ] Clone sites use `Arc::clone()` (cheap)
- [ ] `cargo build` succeeds
- [ ] Examples produce identical results

## Implementation Guide

### Step 1: Update BendersCut with AtomicBool

```rust
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]  // Remove Clone derive
pub struct BendersCut {
    pub id: usize,
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    pub active: AtomicBool,
    pub non_dominated_state_count: usize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
    pub slot_index: Option<usize>,
}

impl BendersCut {
    pub fn new(...) -> Self {
        Self {
            // ...
            active: AtomicBool::new(true),
            // ...
        }
    }
    
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
    
    pub fn set_inactive(&self) {
        self.active.store(false, Ordering::Relaxed);
    }
}
```

### Step 2: Update FCF add_cut

```rust
// src/fcf.rs
pub fn add_cut(&mut self, new_cut: Arc<cut::BendersCut>) {
    self.cut_pool.pool.push(new_cut);
}
```

### Step 3: Update sddp mutation site

```rust
// src/sddp/mod.rs:2155-2157
for &cut_id in &aggregated_result.removing_cut_ids {
    if let Some(cut) = fcf_locked.cut_pool.pool.get(cut_id) {
        cut.set_inactive();  // No mut needed with AtomicBool
    }
    // ...
}
```

### Step 4: Update cloning site

```rust
// src/sddp/mod.rs:2184-2198
let cuts: Vec<(usize, Arc<crate::cut::BendersCut>)> = aggregated_result
    .new_cut_ids
    .iter()
    .chain(aggregated_result.returning_cut_ids.iter())
    .filter_map(|&cut_id| {
        fcf_locked.cut_pool.pool.get(cut_id)
            .map(|cut| (cut_id, Arc::clone(cut)))  // Cheap clone!
    })
    .collect();
```

### Step 5: Update function signatures

Audit all functions that receive `BendersCut` and update to `Arc<BendersCut>`:

```rust
// src/sddp/mod.rs:776
fn apply_cuts_to_handler(
    cuts_to_add: &[(usize, Arc<crate::cut::BendersCut>)],
    // ...
)
```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/cut.rs` | AtomicBool for active, remove Clone derive |
| `src/fcf.rs` | add_cut takes Arc |
| `src/sddp/mod.rs` | Update signatures, use Arc::clone |
| `src/subproblem.rs` | Update cut access patterns |

### Pitfalls to Avoid

- ⚠️ AtomicBool requires `std::sync::atomic::Ordering` on access
- ⚠️ BendersCut no longer implements Clone (Arc clones instead)
- ⚠️ Ordering::Relaxed is fine here (no synchronization needed)
- ⚠️ Debug derive still works with AtomicBool

## Testing Requirements

### Unit Tests

- [ ] `BendersCut::is_active()` returns true initially
- [ ] `BendersCut::set_inactive()` changes state
- [ ] Arc cloning works correctly

### Integration Tests

- [ ] Example 01-deterministic produces identical output
- [ ] Example 07-par-model produces identical output

### Memory Tests

- [ ] Profile cloning site - should show near-zero allocation

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Many files to update, AtomicBool changes API

## Definition of Done

- [ ] All cut access sites updated
- [ ] AtomicBool for active flag
- [ ] Arc::clone used for sharing
- [ ] Examples produce identical results
- [ ] Tests passing
- [ ] Code reviewed
- [ ] PR merged
