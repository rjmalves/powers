# [T-099] Preallocated Probability Buffers

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 7: Comprehensive Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: None
> **Priority**: 3 (Rust Allocation Optimization)

## Files to Read Before Starting

- `docs/HOT_PATH_ALLOCATION_AUDIT.md` - Original audit identifying this allocation
- `src/utils/mod.rs` - `uniform_prob_by_count()` function
- `src/state.rs` - Call sites using uniform probabilities
- `src/sddp/mod.rs` - Call sites using uniform probabilities

---

## Context

### Background

`uniform_prob_by_count()` allocates a new `Vec<f64>` every time it's called:

```rust
pub fn uniform_prob_by_count(count: usize) -> Vec<f64> {
    let p = 1.0 / count as f64;
    vec![p; count]
}
```

This is called during every cut evaluation, multiplied by stages × forward passes × branchings.

### Call Sites

| Location | Context |
|----------|---------|
| `src/sddp/mod.rs:2492` | `eval_first_stage_bound()` |
| `src/state.rs:1298` | Storage state cut evaluation |
| `src/state.rs:1375` | Storage state cut evaluation |
| `src/state.rs:1842` | StorageAndInflow state cut evaluation |
| `src/state.rs:1935` | StorageAndInflow state cut evaluation |

### Target

Replace with in-place computation using preallocated buffers, eliminating allocations.

---

## Specification

### New API

```rust
/// Fill a buffer with uniform probabilities.
/// 
/// Each element is set to `1.0 / buffer.len()`.
/// 
/// # Panics
/// 
/// Panics if buffer is empty.
pub fn fill_uniform_probabilities(buffer: &mut [f64]) {
    assert!(!buffer.is_empty(), "Cannot compute uniform probabilities for empty buffer");
    let p = 1.0 / buffer.len() as f64;
    buffer.fill(p);
}
```

### Behavior

- All call sites refactored to use preallocated buffers
- Thread-local buffers for cut evaluation paths
- Original function deprecated (but kept for compatibility if needed)

---

## Acceptance Criteria

- [ ] `fill_uniform_probabilities()` function added to `src/utils/mod.rs`
- [ ] All hot path call sites refactored to use buffer version
- [ ] Thread-local probability buffers added where needed
- [ ] All tests pass
- [ ] DHAT shows reduced Rust allocations

---

## Implementation Guide

### Suggested Approach

1. **Add new function** to `src/utils/mod.rs`:
   ```rust
   /// Fill a buffer with uniform probabilities (1/n for each of n elements).
   pub fn fill_uniform_probabilities(buffer: &mut [f64]) {
       assert!(!buffer.is_empty(), "Cannot compute uniform probabilities for empty buffer");
       let p = 1.0 / buffer.len() as f64;
       buffer.fill(p);
   }
   ```

2. **Add thread-local buffer** in `src/state.rs`:
   ```rust
   thread_local! {
       static PROBABILITY_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(64));
   }
   ```

3. **Refactor call sites** in `src/state.rs`:
   ```rust
   // Before:
   let probabilities = utils::uniform_prob_by_count(num_branchings);
   
   // After:
   PROBABILITY_BUFFER.with(|buf| {
       let mut buf = buf.borrow_mut();
       buf.resize(num_branchings, 0.0);
       utils::fill_uniform_probabilities(&mut buf);
       // Use buf as probabilities...
   });
   ```

4. **Refactor `eval_first_stage_bound()`** in `src/sddp/mod.rs`:
   - Add thread-local buffer or use existing allocation infrastructure

5. **Deprecate original function** (optional):
   ```rust
   #[deprecated(note = "Use fill_uniform_probabilities() with preallocated buffer")]
   pub fn uniform_prob_by_count(count: usize) -> Vec<f64> { ... }
   ```

### Key Files to Modify

- `src/utils/mod.rs`: Add `fill_uniform_probabilities()`
- `src/state.rs`: Add thread-local buffer, refactor 4 call sites
- `src/sddp/mod.rs`: Refactor `eval_first_stage_bound()`

### Patterns to Follow

- See existing thread-local buffers in `src/memory/buffers.rs`
- Follow pattern from `CutComputationBuffers`

### Pitfalls to Avoid

- ⚠️ Buffer must be resized before filling (use `resize()` not `with_capacity()`)
- ⚠️ Handle edge case of empty branching count
- ⚠️ The borrow must not escape the `with()` closure

---

## Testing Requirements

### Unit Tests

- [ ] `fill_uniform_probabilities` with various sizes
- [ ] Test that sum equals 1.0 (within epsilon)
- [ ] Test edge case: single element (probability = 1.0)

### Integration Tests

- [ ] Golden tests pass (numerical correctness)
- [ ] Training completes successfully

### Performance Tests

- [ ] DHAT shows reduced Rust allocations
- [ ] No solve time regression

---

## Documentation Requirements

- [ ] Doc comments for new function
- [ ] Update `HOT_PATH_ALLOCATION_AUDIT.md` to note fix

---

## Dependencies

- **Blocked By**: None
- **Blocks**: None
- **Related**: T-100, T-101 (other Rust allocation optimizations)

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Clear pattern, multiple call sites but mechanical refactoring

---

## Definition of Done

- [ ] New function implemented
- [ ] All call sites refactored
- [ ] All tests passing
- [ ] DHAT shows improvement
- [ ] Documentation updated
- [ ] PR merged
