# Epic 3: Handler-Level SoA Blocks

## Status

**Status**: ⬜ Not Started  
**Next Up**: 2025-12-26

## Summary

Convert handler hot data (loads, inflows, storage) to contiguous Struct-of-Arrays (SoA) blocks for improved cache locality, while preserving the graph structure for topology.

## Pre-Implementation Analysis (2025-12-26)

### Current Architecture

The `SddpTrainHandler` uses graph-based storage:

```rust
pub struct SddpTrainHandler {
    subproblem_graph: graph::DirectedGraph<subproblem::Subproblem>,
    realization_graph: graph::DirectedGraph<subproblem::Realization>,
    branching_graph: graph::DirectedGraph<Vec<subproblem::Realization>>,
    // ...
}
```

### Hot Path Access Patterns

1. **Forward Pass** (`iterate_forward_pass`):
   - `realization_graph.get_node(past_id)` - Multiple calls per stage
   - `realization_graph.get_node_mut(id)` - One per stage
   - Sequential stage iteration via `study_period_ids`

2. **Backward Pass** (`compute_cut_for_backward_step`):
   - `realization_graph.get_node(past_id)` - Multiple calls
   - `branching_graph` access for scenario branchings

### Complexity Assessment

| Aspect | Impact | Notes |
|--------|--------|-------|
| Realization fields | High | 20+ fields including nested Vec<Vec<f64>> |
| Graph integration | High | Graph topology determines access order |
| Memory layout | Medium | ~320 KB per handler, fits L2 cache |
| Expected gain | 6-10% | Based on GRAPH_TO_SOA_REFACTORING_ANALYSIS.md |

### Recommendation

This is a significant refactoring. Consider:
1. **Profile first**: Measure actual cache miss rates
2. **Start small**: Only hot fields (loads, inflows, storage)
3. **Keep graph**: Use graph for topology, SoA for data

## Scope

### Included

- Design RealizationBlock and SubproblemBlock structures
- Implement contiguous memory allocation for hot data
- Integrate blocks with SddpTrainHandler
- Performance validation

### Excluded

- Full graph elimination (hybrid approach only)
- Markovian graph topology changes
- Simulation handler changes (train only first)

## Dependencies

- **Requires**: Epic 2 complete (FCF preallocation) ✅
- **Enables**: Future Markovian graph extensions (data blocks per branch)

## Acceptance Criteria

- [ ] Hot data stored in contiguous blocks
- [ ] O(1) access via stage offset tables
- [ ] Graph structure preserved for topology
- [ ] 6-10% performance improvement
- [ ] Examples produce identical results

## Technical Approach

### Phase 1: RealizationBlock Design (Sprint 1)

Design contiguous storage for realization data:

```rust
pub struct RealizationBlock {
    num_stages: usize,
    num_buses: usize,
    num_hydros: usize,
    
    // Contiguous memory for all stages
    all_loads: Vec<f64>,      // size: num_stages × num_buses
    all_inflows: Vec<f64>,    // size: num_stages × num_hydros
    all_storage: Vec<f64>,    // size: num_stages × num_hydros
    
    // Offset table for O(1) access
    stage_offsets: Vec<StageOffsets>,
}

struct StageOffsets {
    loads_start: usize,
    inflows_start: usize,
    storage_start: usize,
}
```

### Phase 2: Integration (Sprint 2)

Integrate blocks with handler:

```rust
pub struct SddpTrainHandler {
    // Keep graph for topology (cold path)
    topology: DirectedGraph<NodeMetadata>,
    
    // Hot data in contiguous blocks
    realizations: RealizationBlock,
    subproblems: SubproblemBlock,
    
    // ... rest unchanged ...
}
```

## Estimated Effort

- **Sprint 1**: 5 story points (design + RealizationBlock)
- **Sprint 2**: 5 story points (SubproblemBlock + integration)
- **Total**: 10 story points (2-3 weeks)

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Access pattern changes break code | Medium | Comprehensive testing |
| Cache improvement less than expected | Medium | Profile before/after, accept partial gain |
| Increased code complexity | Medium | Clear abstractions, good documentation |
