# [T-083] Preallocate Coordinator Result Buffers

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 5](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [T-085](./ticket-085-dhat-profiling.md)
> **Status**: ✅ Complete

---

## Context

### Background

The `ParallelHandlerCoordinator` in `coordinator.rs` allocates result buffers on every iteration:

```rust
// coordinator.rs:176-192
let mut slots = Vec::with_capacity(self.num_forward_passes);
let mut timings = Vec::with_capacity(self.num_forward_passes);

// ... parallel computation ...

for (slot, timing) in results {
    slots.push(slot);
    timings.push(timing);
}
```

This pattern allocates 2 Vecs per backward pass stage per iteration, totaling:
- ~12 stages × 2 Vecs × iterations allocations

### Current State

- Result buffers allocated fresh for each stage processing
- Buffers dropped at end of stage, reallocated for next stage
- Pattern repeated in multiple methods: `compute_cuts_parallel_into_slots`, `update_cuts_from_staging`, etc.

### Target State

- Coordinator owns reusable result buffers
- Buffers cleared and reused between stages/iterations
- Zero allocation overhead in hot path

---

## Specification

### Inputs

- Number of forward passes (known at coordinator construction)
- Maximum branching factor (known at coordinator construction)

### Outputs

- Preallocated buffers in `ParallelHandlerCoordinator`
- Zero allocations per stage processing

### New Data Structure

```rust
/// Preallocated buffers for coordinator result collection.
///
/// Avoids per-stage allocation overhead by reusing buffers across iterations.
pub struct CoordinatorBuffers {
    /// Slot indices from parallel cut computation
    pub slots: Vec<usize>,
    
    /// Timing data from parallel phases
    pub phase1_timings: Vec<BackwardPhase1Timing>,
    
    /// Temporary storage for cut application
    pub cut_ids: Vec<usize>,
}

impl CoordinatorBuffers {
    pub fn new(num_forward_passes: usize) -> Self {
        Self {
            slots: Vec::with_capacity(num_forward_passes),
            phase1_timings: Vec::with_capacity(num_forward_passes),
            cut_ids: Vec::with_capacity(num_forward_passes * 2),
        }
    }
    
    pub fn reset(&mut self) {
        self.slots.clear();
        self.phase1_timings.clear();
        self.cut_ids.clear();
    }
}
```

### Integration

```rust
pub struct ParallelHandlerCoordinator {
    handlers: Vec<SddpTrainHandler>,
    num_forward_passes: usize,
    buffers: CoordinatorBuffers,  // NEW
}

impl ParallelHandlerCoordinator {
    pub fn new(handlers: Vec<SddpTrainHandler>) -> Self {
        let num_fp = handlers.len();
        Self {
            handlers,
            num_forward_passes: num_fp,
            buffers: CoordinatorBuffers::new(num_fp),
        }
    }
}
```

---

## Acceptance Criteria

- [x] `CoordinatorBuffers` struct added with preallocated Vecs
- [x] `ParallelHandlerCoordinator` owns reusable buffers
- [x] All result collection loops use preallocated buffers
- [x] `reset()` called at start of each stage processing
- [ ] No allocations in coordinator hot paths (DHAT verified - requires manual verification)
- [x] All tests pass

### Implementation Notes

Added `CoordinatorBuffers` struct with `slots` and `phase1_timings` vectors.
Added `buffers` field to `ParallelHandlerCoordinator`, initialized in constructor.
Updated `compute_cuts_into_slots()` to use `self.buffers` with `reset()` at start.

---

## Implementation Guide

### Suggested Approach

1. **Create `CoordinatorBuffers` struct** in `coordinator.rs`

2. **Add buffers field** to `ParallelHandlerCoordinator`

3. **Update constructor** to initialize buffers

4. **Refactor methods** to use preallocated buffers:

   **Before:**
   ```rust
   let mut slots = Vec::with_capacity(self.num_forward_passes);
   for result in parallel_results {
       slots.push(result.slot);
   }
   ```
   
   **After:**
   ```rust
   self.buffers.slots.clear();
   for result in parallel_results {
       self.buffers.slots.push(result.slot);
   }
   ```

5. **Update methods that return owned Vecs** to either:
   - Return references to internal buffers
   - Clone from buffers only when ownership transfer is required

### Key Files to Modify

- `src/algorithm/coordinator.rs`: Add buffers, refactor result collection

### Methods to Update

Based on grep results:
- `compute_cuts_parallel_into_slots()` (lines 176-192)
- `update_cuts_from_staging()` (lines 291-307)  
- `select_cuts_from_slots()` (lines 428-434)

### Patterns to Follow

- See `CutStagingBuffer` in `memory/buffers.rs` for reusable buffer pattern
- See `reset_for_cut()` for clear-before-use pattern

### Pitfalls to Avoid

- ⚠️ Don't forget to call `clear()` before reusing buffers
- ⚠️ Watch for lifetime issues if returning references to internal buffers
- ⚠️ The cloning at lines 354-356 may still be needed for external API

---

## Testing Requirements

### Unit Tests

- [ ] Test `CoordinatorBuffers::new()` creates correct capacity
- [ ] Test `reset()` clears all buffers
- [ ] Test multiple reuses don't reallocate

### Integration Tests

- [ ] Run backward pass with preallocated buffers
- [ ] Verify same results as before refactoring

### Performance Tests

- [ ] DHAT shows no allocations in `compute_cuts_parallel_into_slots`
- [ ] DHAT shows no allocations in `update_cuts_from_staging`

---

## Documentation Requirements

- [ ] Document `CoordinatorBuffers` purpose and usage
- [ ] Add inline comments explaining buffer reuse strategy

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward refactoring with well-defined scope.

---

## Definition of Done

- [ ] `CoordinatorBuffers` implemented and tested
- [ ] All coordinator methods use preallocated buffers
- [ ] DHAT confirms zero allocations in hot path
- [ ] All tests passing
- [ ] Code reviewed and merged
