# [TICKET-011] Design RealizationBlock structure

> **Epic**: [Epic 3: Handler-Level SoA Blocks](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [TICKET-012](./ticket-012-implement-realization-block.md)

## Context

### Background

The current `DirectedGraph<Realization>` stores each realization's data (loads, inflows, storage) in separate heap allocations. Converting to SoA blocks improves cache locality for sequential stage access.

### Relation to Epic

Design foundation for all SoA block implementations.

## Files to Read Before Starting

- `src/sddp/mod.rs` - Current Realization structure
- `src/memory/sizing.rs` - SizingInfo for dimensions
- `GRAPH_TO_SOA_REFACTORING_ANALYSIS.md` - Hybrid approach recommendation

## Specification

### Design Goals

1. **Contiguous Memory**: All loads/inflows/storage in single allocations
2. **O(1) Access**: Stage data via offset table lookup
3. **Cache-Friendly**: Sequential access benefits from prefetching
4. **Preallocated**: Single allocation at training start

### Structure Design

```rust
/// Contiguous storage for realization data across all stages.
///
/// Memory layout (example with 3 stages, 2 buses, 4 hydros):
///
/// all_loads:   [s0_b0, s0_b1, s1_b0, s1_b1, s2_b0, s2_b1]
///              |----stage 0---|----stage 1---|----stage 2---|
///
/// all_inflows: [s0_h0, s0_h1, s0_h2, s0_h3, s1_h0, ...]
///              |-------stage 0-------|-------stage 1-------|...
///
/// Benefits:
/// - Sequential stage iteration: perfect cache line utilization
/// - Single allocation: no fragmentation
/// - Predictable prefetching: hardware prefetcher effective
pub struct RealizationBlock {
    /// Number of stages in the block
    num_stages: usize,
    
    /// Number of buses (for load data)
    num_buses: usize,
    
    /// Number of hydros (for inflow/storage data)
    num_hydros: usize,
    
    /// Contiguous load data: size = num_stages × num_buses
    all_loads: Vec<f64>,
    
    /// Contiguous inflow data: size = num_stages × num_hydros
    all_inflows: Vec<f64>,
    
    /// Contiguous end-of-stage storage: size = num_stages × num_hydros
    all_storage: Vec<f64>,
    
    /// Contiguous turbined volumes: size = num_stages × num_hydros
    all_turbined: Vec<f64>,
    
    /// Contiguous spillage: size = num_stages × num_hydros
    all_spillage: Vec<f64>,
    
    /// Pre-computed offsets for O(1) stage access
    stage_offsets: Vec<StageOffsets>,
}

/// Pre-computed byte offsets for a single stage.
#[derive(Clone, Copy)]
struct StageOffsets {
    /// Start index in all_loads for this stage
    loads_start: usize,
    
    /// Start index in all_inflows for this stage
    inflows_start: usize,
    
    /// Start index in all_storage for this stage
    storage_start: usize,
    
    /// Start index in all_turbined for this stage
    turbined_start: usize,
    
    /// Start index in all_spillage for this stage
    spillage_start: usize,
}
```

### API Design

```rust
impl RealizationBlock {
    /// Create block with preallocated capacity.
    pub fn new(sizing: &SizingInfo) -> Self;
    
    /// Get loads for a stage as mutable slice.
    pub fn get_loads_mut(&mut self, stage_id: usize) -> &mut [f64];
    
    /// Get loads for a stage as immutable slice.
    pub fn get_loads(&self, stage_id: usize) -> &[f64];
    
    /// Get inflows for a stage.
    pub fn get_inflows_mut(&mut self, stage_id: usize) -> &mut [f64];
    pub fn get_inflows(&self, stage_id: usize) -> &[f64];
    
    /// Get end-of-stage storage for a stage.
    pub fn get_storage_mut(&mut self, stage_id: usize) -> &mut [f64];
    pub fn get_storage(&self, stage_id: usize) -> &[f64];
    
    /// Get turbined volumes for a stage.
    pub fn get_turbined_mut(&mut self, stage_id: usize) -> &mut [f64];
    pub fn get_turbined(&self, stage_id: usize) -> &[f64];
    
    /// Get spillage for a stage.
    pub fn get_spillage_mut(&mut self, stage_id: usize) -> &mut [f64];
    pub fn get_spillage(&self, stage_id: usize) -> &[f64];
    
    /// Reset all values to zero (for reuse between iterations).
    pub fn reset(&mut self);
}
```

## Acceptance Criteria

- [ ] Structure design documented with memory layout
- [ ] API design documented with all accessor methods
- [ ] Design reviewed for cache locality
- [ ] Decision on which fields to include (loads, inflows, storage, etc.)

## Implementation Guide

### Design Questions to Answer

1. **Which fields?**: Start with loads, inflows, storage. Add turbined/spillage if accessed in hot path.

2. **Uniform vs heterogeneous?**: If hydro count varies by stage, need more complex indexing. Start with uniform assumption.

3. **Read/write patterns?**: Forward pass writes, backward pass reads? Design for common pattern.

4. **Thread safety?**: Each forward pass has its own block? Or shared with careful indexing?

### Memory Layout Visualization

```
RealizationBlock for 120 stages, 32 buses, 156 hydros:

all_loads:   [────────── 120 × 32 = 3,840 f64 = 30 KB ──────────]
all_inflows: [────────── 120 × 156 = 18,720 f64 = 146 KB ───────]
all_storage: [────────── 120 × 156 = 18,720 f64 = 146 KB ───────]

Total: ~320 KB contiguous memory (fits in L2/L3 cache)

Compare to current: 120 separate Realization objects with scattered heap allocations
```

### Pitfalls to Avoid

- ⚠️ Don't over-engineer: start with essential fields
- ⚠️ Keep offset computation simple (multiplication, not lookup)
- ⚠️ Consider alignment for SIMD (optional optimization)

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Design task, no implementation

## Definition of Done

- [ ] Structure design complete
- [ ] API design complete
- [ ] Memory layout documented
- [ ] Design approved for implementation
