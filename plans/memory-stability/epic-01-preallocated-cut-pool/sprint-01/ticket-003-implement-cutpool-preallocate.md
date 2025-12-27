# [TICKET-003] Implement BendersCutPool::preallocate()

> **Epic**: [Epic 1: Preallocated Cut Pool](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [TICKET-001](./ticket-001-add-slot-computation.md), [TICKET-002](./ticket-002-add-benderscut-update.md)
> **Blocks**: [TICKET-004](./ticket-004-modify-fcf-initialization.md)

## Context

### Background

The cut pool currently uses `with_capacity()` which preallocates the Vec's internal pointer array but not the actual `BendersCut` objects or their coefficient vectors. We need full preallocation of all cuts.

### Relation to Epic

This is the core implementation that enables zero allocation during training.

### Current State

```rust
pub fn with_capacity(num_cuts: usize, _state_dim: usize) -> Self {
    Self {
        pool: Vec::with_capacity(num_cuts),  // Only reserves pointers!
        active_cut_indices: HashMap::with_capacity(num_cuts),
        total_cut_count: 0,
    }
}
```

## Specification

### Inputs

- `num_iterations: usize` - Number of training iterations
- `num_forward_passes: usize` - Forward passes per iteration
- `state_dimension: usize` - State dimension for coefficient vectors

### Outputs

- `BendersCutPool` - Pool with all cuts preallocated

### Behavior

1. Compute `total_cuts = num_iterations * num_forward_passes`
2. Create `total_cuts` `BendersCut` instances, each with:
   - `id` = slot index
   - `coefficients` = `vec![0.0; state_dimension]`
   - `rhs` = 0.0
   - `active` = false (inactive until populated)
   - `non_dominated_state_count` = 0
   - `iteration` = 0
   - `forward_pass_idx` = 0
   - `slot_index` = None
3. Preallocate `HashMap` with `total_cuts` capacity

### Error Handling

- None required (valid inputs assumed from caller)

## Acceptance Criteria

- [ ] All cuts fully preallocated with coefficient vectors
- [ ] Total memory allocated = `total_cuts * (sizeof(BendersCut) + state_dimension * 8)`
- [ ] Pool accessible by slot index
- [ ] Unit tests verify preallocation correctness

## Implementation Guide

### Suggested Approach

1. Add `preallocate()` method to `impl BendersCutPool`
2. Store `num_forward_passes` in struct for slot computation
3. Add getter method `update_cut()` that computes slot and calls `BendersCut::update()`

### Key Files to Modify

- `src/cut.rs`: Add `preallocate()` method and `num_forward_passes` field

### Proposed Structure Changes

```rust
#[derive(Debug)]
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
    pub active_cut_indices: HashMap<usize, usize>,
    pub total_cut_count: usize,
    num_forward_passes: usize,  // NEW: for slot computation
}

impl BendersCutPool {
    pub fn preallocate(
        num_iterations: usize,
        num_forward_passes: usize,
        state_dimension: usize,
    ) -> Self {
        let total_cuts = num_iterations * num_forward_passes;
        
        let pool: Vec<BendersCut> = (0..total_cuts)
            .map(|id| BendersCut {
                id,
                coefficients: vec![0.0; state_dimension],
                rhs: 0.0,
                active: false,
                non_dominated_state_count: 0,
                iteration: 0,
                forward_pass_idx: 0,
                slot_index: None,
            })
            .collect();
        
        Self {
            pool,
            active_cut_indices: HashMap::with_capacity(total_cuts),
            total_cut_count: 0,
            num_forward_passes,
        }
    }
    
    /// Update cut at slot computed from (iteration, forward_pass_idx).
    /// Returns the slot index (which is also the cut_id).
    pub fn update_cut(
        &mut self,
        iteration: usize,
        forward_pass_idx: usize,
        coefficients: &[f64],
        rhs: f64,
    ) -> usize {
        let slot = compute_slot(iteration, forward_pass_idx, self.num_forward_passes);
        self.pool[slot].update(coefficients, rhs, iteration, forward_pass_idx);
        
        if slot >= self.total_cut_count {
            self.total_cut_count = slot + 1;
        }
        
        slot
    }
}
```

### Pitfalls to Avoid

- ⚠️ Don't forget to update `new()` and `with_capacity()` to set `num_forward_passes = 0`
- ⚠️ Ensure backward compatibility with existing tests

## Testing Requirements

### Unit Tests

- [ ] Test preallocate creates correct number of cuts
- [ ] Test each cut has correct coefficient dimension
- [ ] Test cuts start as inactive
- [ ] Test `update_cut()` correctly computes slot and updates cut
- [ ] Test memory is fully allocated (not lazy)

### Performance Tests

- [ ] Benchmark preallocation time for large systems (e.g., 156 hydros, 128 cuts)

## Documentation Requirements

- [ ] Doc comment explaining full preallocation strategy
- [ ] Document memory usage formula

## Effort Estimate

**Points**: 5
**Confidence**: High
**Rationale**: Core implementation with struct changes, needs careful testing
