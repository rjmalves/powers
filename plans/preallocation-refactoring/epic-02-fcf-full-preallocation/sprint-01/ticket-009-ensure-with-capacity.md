# [TICKET-009] Update FCF creation to use with_capacity() ✅ ALREADY IMPLEMENTED

> **Epic**: [Epic 2: FCF Full Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Status**: ✅ Already implemented via reserve() pattern  
> **Completed**: 2025-12-26

## Summary

No code changes needed. The FCF preallocation is already correctly implemented using a **deferred reserve() pattern** in `train()` at lines 1797-1805.

## Current Implementation

```rust
// In SddpAlgorithm::train() - before hot training loop
let max_cuts = num_forward_passes * num_iterations;
let max_states = num_forward_passes * num_iterations;

for fcf_node in self.future_cost_function_graph.iter_nodes() {
    let mut fcf = fcf_node.data.lock().unwrap();
    fcf.cut_pool.pool.reserve(max_cuts);           // ✅ Vec preallocation
    fcf.cut_pool.active_cut_indices.reserve(max_cuts);  // ✅ HashMap preallocation
    fcf.state_pool.pool.reserve(max_states);       // ✅ Vec preallocation
}
```

## Why This Pattern is Correct

1. `SddpAlgorithm::new()` creates the FCF graph structure
2. `SddpAlgorithm::train()` has access to `num_iterations` and `num_forward_passes`
3. `reserve()` is called before entering the training loop
4. Zero allocations during training - equivalent to `with_capacity()`

## Pools Covered

| Pool | Preallocation | Location |
|------|---------------|----------|
| `cut_pool.pool` | ✅ `reserve(max_cuts)` | Line 1802 |
| `cut_pool.active_cut_indices` | ✅ `reserve(max_cuts)` | Line 1803 |
| `state_pool.pool` | ✅ `reserve(max_states)` | Line 1804 |
