# [T-080] Audit and Eliminate Remaining add_row Calls in Training

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 5](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [T-082](./ticket-082-highs-warmup.md), [T-085](./ticket-085-dhat-profiling.md)
> **Status**: ✅ Complete

---

## Context

### Background

Despite implementing cut constraint preallocation infrastructure in Sprints 1-4, code analysis reveals that `model.add_row()` is still called in training paths. Each `add_row()` call:
1. Allocates 2 temporary `Vec`s in `try_add_row()` (Rust side)
2. May trigger HiGHS internal reallocation (C++ side)
3. Disrupts warm-starting and factorization caching

### Current State

The following paths still use `add_row()`:

1. **`StorageState::add_cut_constraint_to_model()`** (state.rs:1223-1238)
   - Creates `Vec<(usize, f64)>` with capacity
   - Calls `model.add_row(cut.rhs.., factors)`

2. **`StorageAndInflowState::add_cut_constraint_to_model()`** (state.rs)
   - Same pattern as StorageState

### Target State

All cut additions during training must use the preallocation path:
- `subproblem.add_cut_to_model()` → `add_cut_with_preallocation()`
- No `model.add_row()` calls after initialization

---

## Specification

### Inputs

- Existing codebase with mixed allocation patterns
- Cut preallocation infrastructure from Sprint 1

### Outputs

- All training paths use preallocation
- `add_cut_constraint_to_model()` deprecated or removed from training path
- Clear separation between initialization (allocation allowed) and training (no allocation)

### Behavior

- **Initialization phase**: Model construction may use `add_row()` for initial constraints
- **Preallocation phase**: Cut slots preallocated via `preallocate_cut_constraints()`
- **Training phase**: Only `add_cut_with_preallocation()` and `deactivate_cut_constraint()` used

### Error Handling

- Panic if attempting to add cut without preallocation during training
- Debug assertions for slot capacity overflow

---

## Acceptance Criteria

- [x] `add_cut_constraint_to_model()` is NOT called during training iterations
- [x] All cut additions during training use `add_cut_with_preallocation()`
- [x] Added `#[deprecated]` attribute to `add_cut_constraint_to_model()` with migration guidance
- [x] All tests pass
- [x] No numerical result changes (golden tests)

### Implementation Notes

The audit found that `add_cut_constraint_to_model()` in `state.rs` is only defined as a trait method
but is never called in the training path. Training already uses `Subproblem::add_cut_to_model()` which
delegates to `add_cut_with_preallocation()`. Added `#[deprecated]` attribute to the trait method
and implementations with `#[allow(deprecated)]` to suppress warnings.

---

## Implementation Guide

### Suggested Approach

1. **Audit all call sites** of `add_cut_constraint_to_model()`:
   ```bash
   grep -rn "add_cut_constraint_to_model" src/
   ```

2. **Trace the call chain** from SDDP training loop:
   - `sddp/mod.rs` → `fcf.rs` → `state.rs`

3. **Verify preallocation path is used**:
   - Check `FCF::add_cut_to_model()` or equivalent
   - Ensure it calls `subproblem.add_cut_to_model()` which uses preallocation

4. **Add compile-time deprecation**:
   ```rust
   #[deprecated(
       since = "0.x.x",
       note = "Use add_cut_with_preallocation() for zero-allocation cut management"
   )]
   fn add_cut_constraint_to_model(...) { ... }
   ```

5. **Add runtime assertion** for debug builds:
   ```rust
   debug_assert!(
       self.has_preallocated_cuts(),
       "Training requires cut preallocation. Call preallocate_cut_constraints() first."
   );
   ```

### Key Files to Modify

- `src/state.rs`: Deprecate `add_cut_constraint_to_model()` methods
- `src/fcf.rs`: Ensure cut addition path uses preallocation
- `src/subproblem.rs`: Add assertions for preallocation requirement
- `src/sddp/mod.rs`: Verify training loop uses correct path

### Patterns to Follow

- See `subproblem.rs:add_cut_to_model()` for the preallocation pattern
- See `subproblem.rs:add_cut_with_preallocation()` for coefficient updates

### Pitfalls to Avoid

- ⚠️ Don't remove `add_row()` from initialization paths (model construction needs it)
- ⚠️ Don't break simulation mode which may not use preallocation
- ⚠️ Ensure backward compatibility for tests that don't preallocate

---

## Testing Requirements

### Unit Tests

- [ ] Test that `add_cut_to_model()` panics without preallocation in debug builds
- [ ] Test that deprecated warning appears when using `add_cut_constraint_to_model()`

### Integration Tests

- [ ] Run example 05 and verify no `add_row` calls in training phase (via debug logging)
- [ ] Golden tests pass with same numerical results

### Validation Tests

- [ ] DHAT profile shows no allocations from `add_row` during training

---

## Documentation Requirements

- [ ] Update `docs/architecture/MEMORY.md` with preallocation requirements
- [ ] Add migration notes for any downstream code using `add_cut_constraint_to_model()`
- [ ] Document the initialization vs. training phase distinction

---

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Requires tracing through multiple modules to find all call sites and ensuring the change doesn't break non-training paths.

---

## Definition of Done

- [ ] All training paths verified to use preallocation
- [ ] Deprecated attributes added with clear migration guidance
- [ ] Debug assertions added for preallocation requirement
- [ ] All tests passing
- [ ] Golden tests verify numerical correctness
- [ ] Code reviewed and merged
