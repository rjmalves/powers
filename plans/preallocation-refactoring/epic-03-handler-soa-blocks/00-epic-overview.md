# Epic 3: Handler-Level SoA Blocks

## Summary

Convert handler hot data (loads, inflows, storage) to contiguous Struct-of-Arrays (SoA) blocks for improved cache locality, while preserving the graph structure for topology.

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

- **Requires**: Epics 1 and 2 for complete memory determinism
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
