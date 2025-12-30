# [T-100] Thread-Local Scenario Sampling Buffers

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 7: Comprehensive Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: None
> **Priority**: 3 (Rust Allocation Optimization)

## Files to Read Before Starting

- `docs/HOT_PATH_ALLOCATION_AUDIT.md` - Original audit identifying this allocation
- `src/scenario.rs` - `sample_scenario()` function
- `src/sddp/mod.rs` - Call sites sampling scenarios

---

## Context

### Background

`sample_scenario()` allocates vectors every time it's called:

```rust
pub fn sample_scenario(
    &self,
    rng: &mut impl Rng,
) -> Vec<&OptimizedSampledBranchingNoises> {
    let branching_indices: Vec<usize> =
        self.index_samplers.iter().map(|d| d.sample(rng)).collect();
    
    branching_indices.iter()
        .enumerate()
        .map(|(period_idx, &branch_idx)| {
            &self.branching_noises[period_idx][branch_idx]
        })
        .collect()
}
```

Two allocations occur:
1. `branching_indices: Vec<usize>` - Indices sampled from each period
2. Return `Vec<&OptimizedSampledBranchingNoises>` - References to noise data

### Call Sites

| Location | Context |
|----------|---------|
| `src/sddp/mod.rs:1942-1944` | Per-iteration scenario sampling |

This is called `num_forward_passes` times per iteration.

---

## Specification

### Target Architecture

Use thread-local buffers for both allocations:

```rust
thread_local! {
    static BRANCHING_INDICES: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
    static SAMPLED_NOISES: RefCell<Vec<*const OptimizedSampledBranchingNoises>> = 
        RefCell::new(Vec::with_capacity(128));
}
```

### New API

```rust
/// Sample a scenario into preallocated buffers.
/// 
/// Returns a slice of references to the sampled noise data.
pub fn sample_scenario_into<'a>(
    &'a self,
    rng: &mut impl Rng,
    indices_buffer: &mut Vec<usize>,
    result_buffer: &'a mut Vec<&'a OptimizedSampledBranchingNoises>,
) -> &'a [&'a OptimizedSampledBranchingNoises] {
    // Clear and fill indices buffer
    indices_buffer.clear();
    indices_buffer.extend(self.index_samplers.iter().map(|d| d.sample(rng)));
    
    // Clear and fill result buffer
    result_buffer.clear();
    result_buffer.extend(
        indices_buffer.iter()
            .enumerate()
            .map(|(period_idx, &branch_idx)| {
                &self.branching_noises[period_idx][branch_idx]
            })
    );
    
    result_buffer.as_slice()
}
```

---

## Acceptance Criteria

- [ ] `sample_scenario_into()` function added with buffer parameters
- [ ] Thread-local buffers added for scenario sampling
- [ ] Call sites refactored to use buffer version
- [ ] Original function kept for non-hot-path usage (tests, etc.)
- [ ] All tests pass
- [ ] DHAT shows reduced Rust allocations

---

## Implementation Guide

### Suggested Approach

1. **Add new method** to `ScenarioApproximation` in `src/scenario.rs`:
   ```rust
   /// Sample a scenario into provided buffers (zero-allocation hot path).
   pub fn sample_scenario_into<'a>(
       &'a self,
       rng: &mut impl Rng,
       indices_buffer: &mut Vec<usize>,
       result_buffer: &'a mut Vec<&'a OptimizedSampledBranchingNoises>,
   ) {
       indices_buffer.clear();
       indices_buffer.extend(self.index_samplers.iter().map(|d| d.sample(rng)));
       
       result_buffer.clear();
       result_buffer.extend(
           indices_buffer.iter()
               .enumerate()
               .map(|(period_idx, &branch_idx)| {
                   &self.branching_noises[period_idx][branch_idx]
               })
       );
   }
   ```

2. **Add thread-local buffers** in `src/sddp/mod.rs` near training loop:
   ```rust
   thread_local! {
       static SCENARIO_INDICES: RefCell<Vec<usize>> = RefCell::new(Vec::with_capacity(128));
       static SCENARIO_NOISES: RefCell<Vec<&'static OptimizedSampledBranchingNoises>> = 
           RefCell::new(Vec::with_capacity(128));
   }
   ```
   
   Note: Lifetime management for thread-locals with references is tricky. Alternative approach:

3. **Alternative: Use indices-only buffer**:
   ```rust
   /// Sample scenario indices into buffer, return slice.
   pub fn sample_scenario_indices_into(
       &self,
       rng: &mut impl Rng,
       buffer: &mut Vec<usize>,
   ) {
       buffer.clear();
       buffer.extend(self.index_samplers.iter().map(|d| d.sample(rng)));
   }
   
   /// Get noise reference by period and branch index.
   pub fn get_noise(&self, period_idx: usize, branch_idx: usize) -> &OptimizedSampledBranchingNoises {
       &self.branching_noises[period_idx][branch_idx]
   }
   ```

4. **Refactor call site** in `src/sddp/mod.rs`:
   ```rust
   // Current:
   let all_sampled_noises: Vec<_> = (0..num_forward_passes)
       .map(|_| saa.sample_scenario(&mut rng))
       .collect();
   
   // New (using indices approach):
   SCENARIO_INDICES.with(|buf| {
       let mut buf = buf.borrow_mut();
       for fp_idx in 0..num_forward_passes {
           saa.sample_scenario_indices_into(&mut rng, &mut buf);
           // Store indices or process immediately
       }
   });
   ```

### Key Files to Modify

- `src/scenario.rs`: Add `sample_scenario_into()` or `sample_scenario_indices_into()`
- `src/sddp/mod.rs`: Add thread-local buffers, refactor sampling loop

### Patterns to Follow

- See existing thread-local patterns in `src/memory/buffers.rs`
- The indices-only approach may be simpler due to lifetime constraints

### Pitfalls to Avoid

- ⚠️ Lifetime issues with thread-local references are complex
- ⚠️ The indices-only approach is safer and may be sufficient
- ⚠️ Don't break the parallel iteration over forward passes
- ⚠️ Ensure determinism is preserved (same sampling order)

---

## Testing Requirements

### Unit Tests

- [ ] `sample_scenario_into` produces same results as `sample_scenario`
- [ ] Buffer is properly cleared between calls
- [ ] Works with various scenario sizes

### Integration Tests

- [ ] Golden tests pass (same scenario sampling)
- [ ] Training produces identical results

### Performance Tests

- [ ] DHAT shows reduced allocations
- [ ] No performance regression

---

## Documentation Requirements

- [ ] Doc comments for new function(s)
- [ ] Note in `HOT_PATH_ALLOCATION_AUDIT.md` that fix is complete

---

## Dependencies

- **Blocked By**: None
- **Blocks**: None
- **Related**: T-099, T-101 (other Rust allocation optimizations)

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Lifetime management with thread-locals can be tricky

---

## Definition of Done

- [ ] Buffer-based sampling implemented
- [ ] Thread-local buffers added
- [ ] Call sites refactored
- [ ] All tests passing
- [ ] DHAT shows improvement
- [ ] PR merged
