# [T-096] Remove noises.to_vec() Clone in Forward Pass

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 7: Rust Application Allocation Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-101

---

## Context

### Background

In the training loop, `noises.to_vec()` is called to clone the noise reference vector for each forward pass. This creates unnecessary allocations.

### Relation to Epic

Eliminates a per-forward-pass allocation in the training loop.

### Current State

```rust
// src/sddp/mod.rs:1952 (approximate)
.map(|(handler, noises)| self.forward(noises.to_vec(), handler))
```

The `forward()` method signature likely takes `Vec<...>` when it could take `&[...]`.

## Specification

### Changes Required

1. **Change `forward()` signature** to accept slice reference instead of Vec
2. **Remove `.to_vec()` call** at call site
3. **Propagate slice through forward pass** if needed

### Inputs

- `noises: &[&OptimizedSampledBranchingNoises]` instead of `Vec<...>`

### Outputs

- Same forward pass behavior
- No allocation for noise conversion

### Behavior

- Forward pass receives slice reference to noise data
- No ownership transfer needed (forward pass only reads)

## Acceptance Criteria

- [ ] `forward()` accepts slice reference
- [ ] No `.to_vec()` at call site
- [ ] All tests pass
- [ ] Golden tests pass

## Implementation Guide

### Suggested Approach

1. **Find the `forward()` method**:
   ```bash
   rg "fn forward\(" src/sddp/mod.rs
   ```

2. **Change signature**:
   ```rust
   // Before:
   fn forward(
       &mut self,
       noises: Vec<&OptimizedSampledBranchingNoises>,
       handler: &mut SddpTrainHandler,
   ) -> Result<...>
   
   // After:
   fn forward(
       &mut self,
       noises: &[&OptimizedSampledBranchingNoises],
       handler: &mut SddpTrainHandler,
   ) -> Result<...>
   ```

3. **Update internal usage**:
   - If `noises` is indexed: works with slice
   - If `noises` is stored: may need adjustment

4. **Update call site**:
   ```rust
   // Before:
   self.forward(noises.to_vec(), handler)
   
   // After:
   self.forward(noises, handler)  // Pass slice directly
   ```

### Key Files to Modify

- `src/sddp/mod.rs` - `forward()` method and call site

### Patterns to Follow

- Prefer `&[T]` over `Vec<T>` for read-only parameters
- Only use `Vec<T>` when ownership is needed

### Pitfalls to Avoid

- ⚠️ Ensure lifetime of slice is sufficient
- ⚠️ If `forward()` stores the noises, need different approach
- ⚠️ May need to update trait bounds

## Testing Requirements

### Unit Tests

- [ ] Forward pass works with slice input

### Integration Tests

- [ ] Full training completes
- [ ] Golden tests pass

## Documentation Requirements

- [ ] Update doc comments on `forward()` method

## Dependencies

- **Blocked By**: None (can be done independently of T-095)
- **Blocks**: T-101 (DHAT verification)
- **Related**: T-095 (scenario sampling)

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple signature change

## Definition of Done

- [ ] Signature changed to slice
- [ ] Call site updated
- [ ] Tests passing
- [ ] Code reviewed
- [ ] PR merged
