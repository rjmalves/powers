# [T-100-r] Scenario Sampling Indices Buffer (Revised)

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 8: Model Rebuild Strategy](./00-sprint-overview.md)
> **Original**: [T-100](../sprint-07/ticket-100-scenario-sampling-buffers.md)
> **Dependencies**: None
> **Blocks**: T-106
> **Priority**: 2 (Deferred Optimization)
> **Status**: 🔵 Ready

## Files to Read Before Starting

- `src/scenario.rs` - `sample_scenario()` function at line 452
- `src/sddp/mod.rs` - Call sites in `train()` at line 1929
- `docs/HOT_PATH_ALLOCATION_AUDIT.md` - Original allocation analysis

---

## Context

### Why Original T-100 Was Deferred

The original approach tried to use thread-local buffers for `Vec<&OptimizedSampledBranchingNoises>`. This failed because:

1. Thread-local statics require `'static` lifetime
2. The noise references are tied to `ScenarioTree`'s lifetime
3. Rust's borrow checker correctly rejects the pattern

### Revised Approach

Use **indices-only** sampling:
1. Sample branching indices into a reusable buffer
2. Access noises directly at call sites via `get_noises_by_stage_and_branching()`

This eliminates the lifetime issue entirely while achieving zero allocation.

### Current Allocation (from DHAT)

```rust
// Current: allocates 2 Vecs per sample_scenario() call
let branching_indices: Vec<usize> = ...  // First allocation
    .collect()                            // Second allocation (return Vec)
```

With 500 forward passes × 300 iterations = 150,000 calls, this is significant.

---

## Specification

### New API

```rust
impl ScenarioTree {
    /// Sample scenario indices into a preallocated buffer.
    ///
    /// This is the zero-allocation version of `sample_scenario()`.
    /// The buffer is cleared and filled with sampled branching indices.
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `buffer` - Buffer to store sampled indices (cleared and filled)
    ///
    /// # Usage
    ///
    /// After calling, use `get_noises_by_stage_and_branching(stage, buffer[stage])`
    /// to access noise data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut indices = Vec::with_capacity(num_stages);
    /// saa.sample_scenario_indices_into(&mut rng, &mut indices);
    /// for stage in 0..num_stages {
    ///     let noise = saa.get_noises_by_stage_and_branching(stage, indices[stage])
    ///         .expect("valid indices");
    /// }
    /// ```
    pub fn sample_scenario_indices_into(
        &self,
        rng: &mut rand_xoshiro::Xoshiro256Plus,
        buffer: &mut Vec<usize>,
    ) {
        buffer.clear();
        buffer.extend(self.index_samplers.iter().map(|d| d.sample(rng)));
    }
}
```

### Call Site Refactoring

**Before** (in `train()`):

```rust
let all_sampled_noises: Vec<Vec<&OptimizedSampledBranchingNoises>> = 
    (0..num_forward_passes)
    .map(|_| saa.sample_scenario(&mut rng))
    .collect();

// Used as:
handler.forward(&all_sampled_noises[fp_idx], ...)
```

**After**:

```rust
// Pre-allocate indices storage (once per iteration)
let mut all_branching_indices: Vec<Vec<usize>> = 
    (0..num_forward_passes)
    .map(|_| Vec::with_capacity(num_stages))
    .collect();

// Sample into buffers (reuses allocation across iterations)
for indices in &mut all_branching_indices {
    saa.sample_scenario_indices_into(&mut rng, indices);
}

// Convert to noise references just before use (single allocation)
let all_sampled_noises: Vec<Vec<&OptimizedSampledBranchingNoises>> =
    all_branching_indices.iter()
    .map(|indices| {
        indices.iter().enumerate()
            .map(|(stage, &branch_idx)| {
                saa.get_noises_by_stage_and_branching(stage, branch_idx).unwrap()
            })
            .collect()
    })
    .collect();
```

### Alternative: Direct Indices in Forward Pass

For maximum efficiency, modify `forward()` to accept indices directly:

```rust
// In handler.forward():
pub fn forward_with_indices(
    &mut self,
    branching_indices: &[usize],
    saa: &scenario::ScenarioTree,
    // ... other args
) -> Result<(f64, ForwardPassTimingAccumulator), String> {
    // Access noises on-demand:
    let noise = saa.get_noises_by_stage_and_branching(stage, branching_indices[stage]);
    // ...
}
```

This avoids the intermediate `Vec<&Noise>` entirely but requires more invasive changes.

---

## Acceptance Criteria

- [ ] `sample_scenario_indices_into()` added to `ScenarioTree`
- [ ] Training loop uses indices approach
- [ ] Same RNG sequence preserved (determinism)
- [ ] DHAT shows reduced allocations in scenario sampling
- [ ] All tests pass
- [ ] Golden tests pass

---

## Implementation Guide

### Suggested Approach (Minimal Change)

1. **Add `sample_scenario_indices_into()`** to `src/scenario.rs`:
   ```rust
   pub fn sample_scenario_indices_into(
       &self,
       rng: &mut rand_xoshiro::Xoshiro256Plus,
       buffer: &mut Vec<usize>,
   ) {
       buffer.clear();
       buffer.extend(self.index_samplers.iter().map(|d| d.sample(rng)));
   }
   ```

2. **Add persistent buffers in training loop**:
   ```rust
   // Before iteration loop:
   let mut all_branching_indices: Vec<Vec<usize>> = 
       (0..num_forward_passes)
       .map(|_| Vec::with_capacity(num_stages))
       .collect();
   
   for index in 0..num_iterations {
       // Reuse buffers each iteration
       for indices in &mut all_branching_indices {
           saa.sample_scenario_indices_into(&mut rng, indices);
       }
       
       // Convert to noise refs (still allocates, but once per iteration)
       let all_sampled_noises: Vec<Vec<&_>> = all_branching_indices.iter()
           .map(|indices| {
               indices.iter().enumerate()
                   .map(|(s, &b)| saa.get_noises_by_stage_and_branching(s, b).unwrap())
                   .collect()
           })
           .collect();
       
       // ... rest of iteration ...
   }
   ```

3. **Test determinism**: Verify golden tests pass

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/scenario.rs` | Add `sample_scenario_indices_into()` |
| `src/sddp/mod.rs` | Refactor sampling in `train()` |

### Pitfalls to Avoid

- ⚠️ **Preserve RNG order**: The indices must be sampled in the same order
- ⚠️ **Keep original `sample_scenario()`**: Used in tests and simulation
- ⚠️ **Don't change forward pass signature yet**: Keep compatibility

---

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_sample_scenario_indices_into_length() {
    let saa = create_test_saa(5);  // 5 stages
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    let mut buffer = Vec::new();
    
    saa.sample_scenario_indices_into(&mut rng, &mut buffer);
    
    assert_eq!(buffer.len(), 5);
}

#[test]
fn test_sample_scenario_indices_deterministic() {
    let saa = create_test_saa(5);
    let mut rng1 = Xoshiro256Plus::seed_from_u64(42);
    let mut rng2 = Xoshiro256Plus::seed_from_u64(42);
    
    let mut buf1 = Vec::new();
    let mut buf2 = Vec::new();
    
    saa.sample_scenario_indices_into(&mut rng1, &mut buf1);
    saa.sample_scenario_indices_into(&mut rng2, &mut buf2);
    
    assert_eq!(buf1, buf2);
}

#[test]
fn test_sample_scenario_indices_matches_original() {
    let saa = create_test_saa(5);
    let mut rng1 = Xoshiro256Plus::seed_from_u64(42);
    let mut rng2 = Xoshiro256Plus::seed_from_u64(42);
    
    let mut indices = Vec::new();
    saa.sample_scenario_indices_into(&mut rng1, &mut indices);
    
    let noises_original = saa.sample_scenario(&mut rng2);
    
    // Verify same noises accessed
    for (stage, &idx) in indices.iter().enumerate() {
        let noise_from_indices = saa.get_noises_by_stage_and_branching(stage, idx).unwrap();
        assert!(std::ptr::eq(noise_from_indices, noises_original[stage]));
    }
}
```

### Integration Tests

- [ ] Golden tests pass with indices approach
- [ ] Training produces identical results

---

## Documentation Requirements

- [ ] Doc comments for `sample_scenario_indices_into()`
- [ ] Example in doc comment
- [ ] Update HOT_PATH_ALLOCATION_AUDIT.md

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Clear implementation path, low risk

---

## Definition of Done

- [ ] `sample_scenario_indices_into()` implemented
- [ ] Training loop refactored
- [ ] Unit tests passing
- [ ] Golden tests passing
- [ ] Documentation complete
- [ ] PR merged
