# [TICKET-008] Audit FCF instantiation sites ✅ COMPLETE

> **Epic**: [Epic 2: FCF Full Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Status**: ✅ Complete  
> **Completed**: 2025-12-26

## Summary

Audit complete. Found 2 sites using `FutureCostFunction::new()`:
1. `src/sddp/mod.rs:1667` - Production (SddpAlgorithm::new())
2. `src/sddp/mod.rs:3031` - Test code (acceptable)

**Finding**: FCF preallocation is already correctly implemented via `reserve()` in `train()` at lines 1800-1805, called before the hot training loop.

## Audit Results

### Production Site: `src/sddp/mod.rs:1667`

```rust
// In SddpAlgorithm::new()
let future_cost_function_graph =
    node_data_graph.map_topology_with(|_node_data, _id| {
        Arc::new(Mutex::new(fcf::FutureCostFunction::new()))
    });
```

**Preallocation**: Happens later in `train()` at lines 1797-1805:

```rust
let max_cuts = num_forward_passes * num_iterations;
let max_states = num_forward_passes * num_iterations;

for fcf_node in self.future_cost_function_graph.iter_nodes() {
    let mut fcf = fcf_node.data.lock().unwrap();
    fcf.cut_pool.pool.reserve(max_cuts);
    fcf.cut_pool.active_cut_indices.reserve(max_cuts);
    fcf.state_pool.pool.reserve(max_states);
}
```

### Test Site: `src/sddp/mod.rs:3031`

Located inside `#[cfg(test)] mod tests` block - acceptable for test code.

## Conclusion

No code changes needed. The deferred `reserve()` pattern is correct because:
- `SddpAlgorithm::new()` doesn't have training parameters
- `train()` has `num_iterations` and `num_forward_passes`
- Capacity is reserved before the hot loop begins
