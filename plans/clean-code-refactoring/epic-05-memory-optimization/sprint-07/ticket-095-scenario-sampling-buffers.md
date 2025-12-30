# [T-095] Add Thread-Local Scenario Sampling Buffers

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 7: Rust Application Allocation Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-101

---

## Context

### Background

`sample_scenario()` in `scenario.rs` allocates two `Vec`s per call:
1. `branching_indices: Vec<usize>` - Sampled indices per stage
2. Return `Vec<&OptimizedSampledBranchingNoises>` - References to noise data

With 4 forward passes × 8 iterations × 2 Vecs = 64 allocations per training run minimum.

### Relation to Epic

Eliminates per-iteration allocations in scenario sampling.

### Current State

```rust
// src/scenario.rs:452-466
pub fn sample_scenario<R: Rng>(&self, rng: &mut R) -> Vec<&OptimizedSampledBranchingNoises> {
    let branching_indices: Vec<usize> =
        self.index_samplers.iter().map(|d| d.sample(rng)).collect();
    
    branching_indices.iter()
        .enumerate()
        .map(|(stage, &idx)| &self.noises[stage][idx])
        .collect()
}
```

## Specification

### Changes Required

1. **Add thread-local buffers** for sampling indices
2. **Change return type** to use preallocated output buffer
3. **Or: Return iterator** instead of collected Vec

### Option A: Preallocated Output Buffer

```rust
pub fn sample_scenario_into<R: Rng>(
    &self,
    rng: &mut R,
    output: &mut Vec<&OptimizedSampledBranchingNoises>,
) {
    output.clear();
    for (stage, sampler) in self.index_samplers.iter().enumerate() {
        let idx = sampler.sample(rng);
        output.push(&self.noises[stage][idx]);
    }
}
```

### Option B: Return Slice from Thread-Local

```rust
thread_local! {
    static SCENARIO_BUFFER: RefCell<Vec<*const OptimizedSampledBranchingNoises>> = 
        RefCell::new(Vec::with_capacity(128));
}
```

### Behavior

- Same noise selection as current implementation
- No allocation after initial buffer creation
- Thread-safe (each thread has own buffer)

## Acceptance Criteria

- [ ] Thread-local buffers for scenario sampling
- [ ] `sample_scenario()` path uses preallocated buffers
- [ ] No per-iteration allocations in sampling
- [ ] All tests pass
- [ ] Golden tests pass (same random sequences)

## Implementation Guide

### Suggested Approach

1. **Add buffer infrastructure**:
   ```rust
   // src/scenario.rs
   
   thread_local! {
       static SAMPLED_INDICES: RefCell<Vec<usize>> = 
           RefCell::new(Vec::with_capacity(128));
   }
   ```

2. **Add sampling method with output buffer**:
   ```rust
   /// Sample a scenario into a preallocated output buffer.
   pub fn sample_scenario_into<'a, R: Rng>(
       &'a self,
       rng: &mut R,
       output: &mut Vec<&'a OptimizedSampledBranchingNoises>,
   ) {
       output.clear();
       output.reserve(self.index_samplers.len());
       
       for (stage, sampler) in self.index_samplers.iter().enumerate() {
           let idx = sampler.sample(rng);
           output.push(&self.noises[stage][idx]);
       }
   }
   ```

3. **Update call site in sddp/mod.rs**:
   ```rust
   // Before (Line 1942-1944):
   let all_sampled_noises: Vec<_> = (0..num_forward_passes)
       .map(|_| saa.sample_scenario(&mut rng))
       .collect();
   
   // After:
   // Use preallocated buffers per handler
   let mut noise_buffers: Vec<Vec<&OptimizedSampledBranchingNoises>> = 
       (0..num_forward_passes)
           .map(|_| Vec::with_capacity(num_stages))
           .collect();
   
   for buffer in &mut noise_buffers {
       saa.sample_scenario_into(&mut rng, buffer);
   }
   ```

4. **Alternative: Store in handler**:
   ```rust
   // Add to SddpTrainHandler
   struct SddpTrainHandler {
       // ...existing fields...
       noise_buffer: Vec<&'scenario OptimizedSampledBranchingNoises>,
   }
   ```

### Key Files to Modify

- `src/scenario.rs` - Add `sample_scenario_into()`
- `src/sddp/mod.rs` - Update sampling call sites
- `src/memory/buffers.rs` - Thread-local buffers (optional)

### Patterns to Follow

- See existing handler staging buffers from Sprint 1
- Lifetime annotations for borrowed noise data

### Pitfalls to Avoid

- ⚠️ Lifetime of borrowed noise must outlive forward pass
- ⚠️ Buffer must be cleared before each sampling
- ⚠️ Don't break parallel forward pass semantics

## Testing Requirements

### Unit Tests

- [ ] `sample_scenario_into` produces same results as `sample_scenario`
- [ ] Buffer is properly cleared between calls
- [ ] Works with various stage counts

### Integration Tests

- [ ] Full training with new sampling
- [ ] Golden tests pass

## Documentation Requirements

- [ ] Doc comments on new method
- [ ] Update sampling usage in code comments

## Dependencies

- **Blocked By**: None
- **Blocks**: T-101 (DHAT verification)
- **Related**: T-094 (probability buffers), T-096 (noises.to_vec)

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Lifetime management may need care

## Definition of Done

- [ ] Preallocated sampling implemented
- [ ] No per-iteration allocations
- [ ] Tests passing
- [ ] Golden tests pass
- [ ] Code reviewed
- [ ] PR merged
