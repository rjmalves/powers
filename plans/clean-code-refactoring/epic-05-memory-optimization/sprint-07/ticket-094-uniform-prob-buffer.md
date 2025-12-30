# [T-094] Replace uniform_prob_by_count with Preallocated Buffers

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 7: Rust Application Allocation Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-101

---

## Context

### Background

`uniform_prob_by_count()` is called **every cut evaluation** to create uniform probability distributions. With ~60 stages × 4 forward passes × 20 branchings = 4,800 calls per iteration, this creates significant allocation churn (~768 KB/iteration).

### Relation to Epic

Eliminates one of the highest-frequency Rust allocation sites.

### Current State

```rust
// src/utils/mod.rs:269
pub fn uniform_prob_by_count(count: usize) -> Vec<f64> {
    let p = 1.0 / count as f64;
    vec![p; count]
}
```

Called from:
- `state.rs:1298` - `evaluate_cut_ref`
- `state.rs:1375` - `evaluate_cut_ref`
- `state.rs:1842` - `evaluate_cut_ref`
- `state.rs:1935` - `evaluate_cut_ref`

## Specification

### Changes Required

1. **Add `fill_uniform_probabilities()` function** that writes to existing buffer
2. **Add thread-local probability buffer** for cut evaluation
3. **Update call sites** to use buffer-based approach

### Inputs

- `buffer: &mut [f64]` - Preallocated buffer to fill
- Buffer length determines the count

### Outputs

- Buffer filled with `1.0 / len` values

### Behavior

- `fill_uniform_probabilities(&mut buffer)` fills buffer with uniform probabilities
- Thread-local buffer is reused across calls
- Buffer is resized if needed (rare, once per thread per max size)

### Error Handling

- Empty buffer is a no-op (or assert in debug builds)

## Acceptance Criteria

- [ ] `fill_uniform_probabilities()` function implemented
- [ ] Thread-local buffer for probability values
- [ ] All call sites updated to use new approach
- [ ] `uniform_prob_by_count()` marked `#[deprecated]` or removed
- [ ] No allocations in cut evaluation path
- [ ] All tests pass

## Implementation Guide

### Suggested Approach

1. **Add new function**:
   ```rust
   // src/utils/mod.rs
   
   /// Fill buffer with uniform probabilities.
   /// Each element is set to 1.0 / buffer.len().
   #[inline]
   pub fn fill_uniform_probabilities(buffer: &mut [f64]) {
       if buffer.is_empty() {
           return;
       }
       let p = 1.0 / buffer.len() as f64;
       buffer.fill(p);
   }
   ```

2. **Add thread-local buffer**:
   ```rust
   // src/memory/buffers.rs
   
   thread_local! {
       static PROBABILITY_BUFFER: RefCell<Vec<f64>> = 
           RefCell::new(Vec::with_capacity(64));
   }
   
   pub fn with_probability_buffer<F, R>(size: usize, f: F) -> R
   where
       F: FnOnce(&[f64]) -> R,
   {
       PROBABILITY_BUFFER.with(|buf| {
           let mut buf = buf.borrow_mut();
           buf.resize(size, 0.0);
           fill_uniform_probabilities(&mut buf);
           f(&buf)
       })
   }
   ```

3. **Update call sites in state.rs**:
   ```rust
   // Before:
   let probs = uniform_prob_by_count(branching_count);
   let expected_value = weighted_sum(&values, &probs);
   
   // After:
   with_probability_buffer(branching_count, |probs| {
       weighted_sum(&values, probs)
   })
   ```

4. **Consider alternative: inline computation**:
   ```rust
   // If weighted_sum is simple, just compute inline
   let p = 1.0 / values.len() as f64;
   let expected_value: f64 = values.iter().sum::<f64>() * p;
   ```

### Key Files to Modify

- `src/utils/mod.rs` - Add `fill_uniform_probabilities()`
- `src/memory/buffers.rs` - Add thread-local probability buffer
- `src/state.rs` - Update all call sites

### Patterns to Follow

- See existing thread-local buffers in `src/memory/buffers.rs`
- Use `RefCell::borrow_mut()` for thread-local access

### Pitfalls to Avoid

- ⚠️ Ensure buffer is large enough for max branching count
- ⚠️ Clear buffer reuse semantics (resize vs. truncate)
- ⚠️ Consider SIMD optimization for fill (Rust's `fill` is already optimized)

## Testing Requirements

### Unit Tests

- [ ] Test `fill_uniform_probabilities` with various sizes
- [ ] Test with empty buffer
- [ ] Test with size 1
- [ ] Verify probabilities sum to 1.0 (within epsilon)

### Integration Tests

- [ ] Cut evaluation produces same results
- [ ] Golden tests pass

### Performance Tests

- [ ] DHAT shows no allocation in `uniform_prob_by_count`

## Documentation Requirements

- [ ] Doc comments on new functions
- [ ] Update `docs/MEMORY_BEHAVIOR.md`

## Dependencies

- **Blocked By**: None
- **Blocks**: T-101 (DHAT verification)
- **Related**: T-095 (scenario buffers), T-100 (trajectory buffers)

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward buffer replacement

## Definition of Done

- [ ] Implementation complete
- [ ] All call sites updated
- [ ] Tests passing
- [ ] No allocations in DHAT for this function
- [ ] Code reviewed
- [ ] PR merged
