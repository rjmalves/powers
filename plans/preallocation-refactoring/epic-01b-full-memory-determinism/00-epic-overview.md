# Epic 1b: Full Memory Determinism for Cut Management

## Summary

Remove all dynamic allocation fallbacks and slot tracking complexity from cut management. Replace the current slot allocation system with deterministic index calculation based on `(iteration, forward_pass_idx)`.

## Motivation

Epic 1 implemented preallocation with a **fallback to dynamic allocation** if slots are exhausted. This was a safety measure, but it:

1. **Violates memory determinism** - Dynamic allocation can still occur during training
2. **Adds unnecessary complexity** - Slot reuse tracking (`free_cut_slots`, `cut_slot_to_id`)
3. **Obscures the algorithm** - LIFO slot reuse is an optimization that complicates debugging

Since we **know** `num_iterations` and `num_forward_passes` at training start, we can calculate **exactly** how many cuts will be created and where each cut should be stored.

## Key Insight: Deterministic Cut Slot Mapping

Each cut is uniquely identified by `(iteration, forward_pass_idx)`. The slot index is:

```rust
slot_index = (iteration - 1) * num_forward_passes + forward_pass_idx
```

Examples (for 4 forward passes):
- iteration=1, fp=0 → slot 0
- iteration=1, fp=3 → slot 3
- iteration=2, fp=0 → slot 4
- iteration=32, fp=3 → slot 127

**Benefits**:
- **Zero tracking overhead** - No `Vec<Option<usize>>`, no free list
- **O(1) slot calculation** - Simple arithmetic instead of data structure lookups
- **Deterministic memory layout** - Same (iter, fp) always maps to same slot
- **Simplified debugging** - Slot location is trivially predictable
- **No slot reuse complexity** - Deactivated slots stay in place

## Scope

### Included

- Remove dynamic allocation fallback from `add_cut_to_model()`
- Remove slot tracking fields (`free_cut_slots`, `cut_slot_to_id`, `next_available_cut_slot`)
- Add `num_forward_passes` field to `Subproblem`
- Add `slot_index` field to `BendersCut` for O(1) deactivation lookup
- Implement `compute_cut_slot(iteration, forward_pass_idx)` method
- Update `add_cut_with_preallocation()` to accept iteration/fp parameters
- Update `deactivate_cut_constraint()` to use stored slot index
- Update all call sites to pass iteration/forward_pass_idx

### Excluded

- FCF pool changes (Epic 2)
- SoA block refactoring (Epic 3)
- Changes to cut selection algorithm

## Dependencies

- **Requires**: Epic 1 complete (HiGHS preallocation infrastructure)
- **Enables**: Epic 2 (FCF), Epic 3 (SoA blocks)

## Acceptance Criteria

- [x] **No dynamic allocation**: `add_cut_constraint_to_model()` (State trait) is never called during training
- [x] **No slot tracking**: `free_cut_slots`, `cut_slot_to_id`, `next_available_cut_slot` removed
- [x] **Deterministic slots**: Each (iteration, forward_pass_idx) maps to fixed slot
- [x] **Panic on overflow**: If slot calculation exceeds preallocated count, panic (not fallback)
- [x] **O(1) deactivation**: Cut deactivation uses stored slot index, not linear search
- [x] **All examples pass**: 01 and 07 produce identical results
- [x] **No performance regression**: ≥23% improvement maintained

## Technical Approach

### Phase 1: Add Deterministic Slot Calculation

1. Add `num_forward_passes: usize` to `Subproblem`
2. Implement `compute_cut_slot(iteration: usize, forward_pass_idx: usize) -> usize`
3. Add `slot_index: Option<usize>` to `BendersCut`

### Phase 2: Update Cut Addition Flow

1. Modify `add_cut_with_preallocation()` to accept `(iteration, forward_pass_idx)`
2. Compute slot deterministically instead of using `allocate_cut_slot()`
3. Store slot index in the cut
4. Panic if slot exceeds `num_preallocated_cuts`

### Phase 3: Update Cut Deactivation

1. Modify `deactivate_cut_constraint()` to accept the cut directly (not just ID)
2. Use stored `slot_index` for O(1) lookup
3. Remove linear search through `cut_slot_to_id`

### Phase 4: Remove Slot Tracking

1. Remove `free_cut_slots: Vec<usize>`
2. Remove `cut_slot_to_id: Vec<Option<usize>>`
3. Remove `next_available_cut_slot: usize`
4. Remove `allocate_cut_slot()` method

### Phase 5: Update Call Sites

1. Update `apply_aggregated_cut_selection_result()` to pass iteration/fp
2. Update `add_cut_to_model()` to require iteration/fp or panic
3. Update SDDP backward pass to thread iteration/fp through

## Estimated Effort

- **Sprint 1**: 5 story points (1 week)
- **Total**: 5 story points

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Cut returning logic breaks | Low | Returning cuts use same slot calculation |
| Missing iteration/fp at call sites | Low | Compiler enforces - required parameters |
| Performance regression from extra params | Very Low | Parameters already available in context |

## Files to Modify

- `src/cut.rs` - Add `slot_index` field to `BendersCut`
- `src/subproblem.rs` - Main changes: slot calculation, remove tracking
- `src/sddp/mod.rs` - Thread iteration/fp through cut addition
- `src/fcf.rs` - May need minor updates for cut creation

## Success Metrics

- [x] Zero dynamic allocations during training (valgrind massif)
- [x] Simplified code: ~50 fewer lines in subproblem.rs
- [x] All examples produce identical results
- [x] Performance improvement maintained (≥23%)
