# [T-101] Eliminate noises.to_vec() and forward_costs.clone()

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 7: Comprehensive Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: None
> **Priority**: 3 (Rust Allocation Optimization)

## Files to Read Before Starting

- `docs/HOT_PATH_ALLOCATION_AUDIT.md` - Original audit identifying these allocations
- `src/sddp/mod.rs` - Call sites with clones

---

## Context

### Background

Two unnecessary clones occur in the training hot path:

#### 1. `noises.to_vec()` (Lines 1953, 2317)

```rust
.map(|(handler, noises)| self.forward(noises.to_vec(), handler))
```

The `forward()` method receives owned `Vec`, but only needs read access.

#### 2. `forward_costs.clone()` (Line 2081)

```rust
iterations.push(IterationResult {
    forward_costs: forward_costs.clone(),
    ...
});
```

The `forward_costs` Vec is cloned for `IterationResult`, but could be moved since it's not used afterward.

---

## Specification

### Task 1: Remove `noises.to_vec()`

**Current signature:**
```rust
fn forward(
    &self,
    sampled_noises: Vec<&OptimizedSampledBranchingNoises>,
    handler: &mut SddpTrainHandler,
) -> Result<(f64, ForwardPassTimingAccumulator), String>
```

**Target signature:**
```rust
fn forward(
    &self,
    sampled_noises: &[&OptimizedSampledBranchingNoises],
    handler: &mut SddpTrainHandler,
) -> Result<(f64, ForwardPassTimingAccumulator), String>
```

### Task 2: Remove `forward_costs.clone()`

**Current:**
```rust
iterations.push(IterationResult {
    forward_costs: forward_costs.clone(),
    ...
});
```

**Target:**
```rust
iterations.push(IterationResult {
    forward_costs,  // Move ownership
    ...
});
```

---

## Acceptance Criteria

- [ ] `forward()` method signature changed to accept slice reference
- [ ] All call sites updated (no more `to_vec()`)
- [ ] `forward_costs` moved instead of cloned
- [ ] All tests pass
- [ ] DHAT shows reduced allocations

---

## Implementation Guide

### Suggested Approach

#### Part 1: Remove `noises.to_vec()`

1. **Change `forward()` signature** in `src/sddp/mod.rs`:
   ```rust
   fn forward(
       &self,
       sampled_noises: &[&OptimizedSampledBranchingNoises],  // Slice reference
       handler: &mut SddpTrainHandler,
   ) -> Result<(f64, ForwardPassTimingAccumulator), String>
   ```

2. **Update call sites** (lines 1953, 2317):
   ```rust
   // Before:
   .map(|(handler, noises)| self.forward(noises.to_vec(), handler))
   
   // After:
   .map(|(handler, noises)| self.forward(noises, handler))
   ```

3. **Update any internal usage** of `sampled_noises`:
   - Should already work since slice has same iteration interface as Vec

#### Part 2: Remove `forward_costs.clone()`

1. **Check if `forward_costs` is used after the push** (line 2081):
   - If not used: change `clone()` to move
   - If used: need to restructure or keep clone

2. **Update the push**:
   ```rust
   // Before:
   iterations.push(IterationResult {
       forward_costs: forward_costs.clone(),
       ...
   });
   
   // After:
   iterations.push(IterationResult {
       forward_costs,  // Move
       ...
   });
   ```

3. **If forward_costs is used after**, restructure:
   ```rust
   let forward_costs_copy = forward_costs.clone();  // Only if needed elsewhere
   iterations.push(IterationResult {
       forward_costs,
       ...
   });
   // Use forward_costs_copy if needed
   ```

### Key Files to Modify

- `src/sddp/mod.rs`:
  - `forward()` method signature
  - Call sites at lines 1953, 2317
  - `IterationResult` push at line 2081

### Patterns to Follow

- Prefer `&[T]` over `Vec<T>` for read-only parameters
- Use `std::mem::take()` if moving from a mutable binding

### Pitfalls to Avoid

- ⚠️ Ensure `sampled_noises` slice lifetime is sufficient
- ⚠️ Check all internal usages of `sampled_noises` in `forward()`
- ⚠️ Verify `forward_costs` is truly unused after the push
- ⚠️ The parallel iterator context may have lifetime constraints

---

## Testing Requirements

### Unit Tests

- [ ] Existing forward pass tests still pass
- [ ] Existing iteration result tests still pass

### Integration Tests

- [ ] Golden tests pass (numerical correctness)
- [ ] Full training run completes successfully

### Performance Tests

- [ ] DHAT shows reduced allocations
- [ ] No performance regression

---

## Documentation Requirements

- [ ] Update doc comments if parameter types change
- [ ] Note in `HOT_PATH_ALLOCATION_AUDIT.md` that fix is complete

---

## Dependencies

- **Blocked By**: None
- **Blocks**: None
- **Related**: T-099, T-100 (other Rust allocation optimizations)

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple signature change and clone removal

---

## Definition of Done

- [ ] `forward()` accepts slice reference
- [ ] No more `to_vec()` calls at call sites
- [ ] `forward_costs` moved instead of cloned
- [ ] All tests passing
- [ ] DHAT shows improvement
- [ ] PR merged
