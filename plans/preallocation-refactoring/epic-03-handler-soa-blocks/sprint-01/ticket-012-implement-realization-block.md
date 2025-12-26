# [TICKET-012] Implement RealizationBlock

> **Epic**: [Epic 3: Handler-Level SoA Blocks](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-011](./ticket-011-design-realization-block.md)  
> **Blocks**: [TICKET-013](../sprint-02/ticket-013-integrate-blocks.md)

## Context

### Background

Implement the RealizationBlock structure designed in TICKET-011.

### Relation to Epic

Core implementation of SoA block pattern.

## Files to Read Before Starting

- TICKET-011 design document
- `src/memory/mod.rs` - Memory module structure
- `src/memory/sizing.rs` - SizingInfo for capacity

## Specification

### Implementation Requirements

1. Create new file `src/memory/blocks.rs`
2. Implement RealizationBlock per TICKET-011 design
3. Add to memory module exports
4. Add unit tests

### Code Location

```
src/memory/
├── mod.rs          # Add: pub mod blocks;
├── blocks.rs       # NEW: RealizationBlock, StageOffsets
├── sizing.rs
├── buffers.rs
└── deep_sizing.rs
```

## Acceptance Criteria

- [ ] RealizationBlock implemented in `src/memory/blocks.rs`
- [ ] All accessor methods work correctly
- [ ] Unit tests for construction and access
- [ ] Exported from memory module
- [ ] Code compiles without warnings

## Implementation Guide

### Suggested Approach

1. Create `src/memory/blocks.rs`
2. Implement StageOffsets struct
3. Implement RealizationBlock struct
4. Add accessor methods
5. Add unit tests
6. Export from `src/memory/mod.rs`

### Code Template

```rust
// src/memory/blocks.rs

//! Contiguous memory blocks for hot-path data.
//!
//! Implements Struct-of-Arrays (SoA) pattern for cache-efficient
//! access to realization and subproblem data.

use crate::memory::SizingInfo;

/// Pre-computed offsets for O(1) stage data access.
#[derive(Clone, Copy, Debug)]
struct StageOffsets {
    loads_start: usize,
    inflows_start: usize,
    storage_start: usize,
}

/// Contiguous storage for realization data across all stages.
///
/// See module documentation for memory layout details.
#[derive(Debug)]
pub struct RealizationBlock {
    num_stages: usize,
    num_buses: usize,
    num_hydros: usize,
    
    all_loads: Vec<f64>,
    all_inflows: Vec<f64>,
    all_storage: Vec<f64>,
    
    stage_offsets: Vec<StageOffsets>,
}

impl RealizationBlock {
    /// Create block with preallocated capacity from sizing info.
    pub fn new(sizing: &SizingInfo) -> Self {
        let num_stages = sizing.num_stages;
        let num_buses = sizing.num_buses;
        let num_hydros = sizing.num_hydros;
        
        // Allocate contiguous memory
        let loads_size = num_stages * num_buses;
        let inflows_size = num_stages * num_hydros;
        let storage_size = num_stages * num_hydros;
        
        let mut all_loads = Vec::with_capacity(loads_size);
        all_loads.resize(loads_size, 0.0);
        
        let mut all_inflows = Vec::with_capacity(inflows_size);
        all_inflows.resize(inflows_size, 0.0);
        
        let mut all_storage = Vec::with_capacity(storage_size);
        all_storage.resize(storage_size, 0.0);
        
        // Build offset table
        let stage_offsets: Vec<StageOffsets> = (0..num_stages)
            .map(|stage| StageOffsets {
                loads_start: stage * num_buses,
                inflows_start: stage * num_hydros,
                storage_start: stage * num_hydros,
            })
            .collect();
        
        Self {
            num_stages,
            num_buses,
            num_hydros,
            all_loads,
            all_inflows,
            all_storage,
            stage_offsets,
        }
    }
    
    /// Get loads for a stage as mutable slice.
    #[inline]
    pub fn get_loads_mut(&mut self, stage_id: usize) -> &mut [f64] {
        let offset = self.stage_offsets[stage_id];
        let start = offset.loads_start;
        &mut self.all_loads[start..start + self.num_buses]
    }
    
    /// Get loads for a stage as immutable slice.
    #[inline]
    pub fn get_loads(&self, stage_id: usize) -> &[f64] {
        let offset = self.stage_offsets[stage_id];
        let start = offset.loads_start;
        &self.all_loads[start..start + self.num_buses]
    }
    
    /// Get inflows for a stage as mutable slice.
    #[inline]
    pub fn get_inflows_mut(&mut self, stage_id: usize) -> &mut [f64] {
        let offset = self.stage_offsets[stage_id];
        let start = offset.inflows_start;
        &mut self.all_inflows[start..start + self.num_hydros]
    }
    
    /// Get inflows for a stage as immutable slice.
    #[inline]
    pub fn get_inflows(&self, stage_id: usize) -> &[f64] {
        let offset = self.stage_offsets[stage_id];
        let start = offset.inflows_start;
        &self.all_inflows[start..start + self.num_hydros]
    }
    
    /// Get storage for a stage as mutable slice.
    #[inline]
    pub fn get_storage_mut(&mut self, stage_id: usize) -> &mut [f64] {
        let offset = self.stage_offsets[stage_id];
        let start = offset.storage_start;
        &mut self.all_storage[start..start + self.num_hydros]
    }
    
    /// Get storage for a stage as immutable slice.
    #[inline]
    pub fn get_storage(&self, stage_id: usize) -> &[f64] {
        let offset = self.stage_offsets[stage_id];
        let start = offset.storage_start;
        &self.all_storage[start..start + self.num_hydros]
    }
    
    /// Reset all values to zero.
    pub fn reset(&mut self) {
        self.all_loads.fill(0.0);
        self.all_inflows.fill(0.0);
        self.all_storage.fill(0.0);
    }
    
    /// Number of stages in this block.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_test_sizing() -> SizingInfo {
        // Minimal sizing for tests
        SizingInfo {
            node_sizing: vec![],
            max_state_dimension: 10,
            min_state_dimension: 10,
            avg_state_dimension: 10.0,
            max_scenarios_per_node: 4,
            max_subproblem_vars: 50,
            num_hydros: 5,
            num_thermals: 2,
            num_buses: 3,
            num_lines: 4,
            num_stages: 10,
            num_nodes: 10,
            max_iterations: 20,
            num_forward_passes: 4,
            num_simulations: 100,
            num_threads: 4,
        }
    }
    
    #[test]
    fn test_realization_block_creation() {
        let sizing = make_test_sizing();
        let block = RealizationBlock::new(&sizing);
        
        assert_eq!(block.num_stages(), 10);
        assert_eq!(block.all_loads.len(), 10 * 3);  // stages × buses
        assert_eq!(block.all_inflows.len(), 10 * 5);  // stages × hydros
        assert_eq!(block.all_storage.len(), 10 * 5);  // stages × hydros
    }
    
    #[test]
    fn test_stage_access() {
        let sizing = make_test_sizing();
        let mut block = RealizationBlock::new(&sizing);
        
        // Write to stage 5
        let loads = block.get_loads_mut(5);
        loads[0] = 100.0;
        loads[1] = 200.0;
        loads[2] = 300.0;
        
        // Read from stage 5
        let loads = block.get_loads(5);
        assert_eq!(loads[0], 100.0);
        assert_eq!(loads[1], 200.0);
        assert_eq!(loads[2], 300.0);
        
        // Other stages unaffected
        let loads_0 = block.get_loads(0);
        assert_eq!(loads_0[0], 0.0);
    }
    
    #[test]
    fn test_reset() {
        let sizing = make_test_sizing();
        let mut block = RealizationBlock::new(&sizing);
        
        // Write data
        block.get_loads_mut(0)[0] = 999.0;
        block.get_inflows_mut(3)[2] = 888.0;
        
        // Reset
        block.reset();
        
        // All zeros
        assert_eq!(block.get_loads(0)[0], 0.0);
        assert_eq!(block.get_inflows(3)[2], 0.0);
    }
}
```

### Module Export

```rust
// src/memory/mod.rs - Add:

pub mod blocks;
pub use blocks::RealizationBlock;
```

## Testing Requirements

### Unit Tests

- [ ] Construction with SizingInfo
- [ ] Stage access (read/write)
- [ ] Multiple stages independent
- [ ] Reset clears all data
- [ ] Bounds checking (debug mode)

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Straightforward implementation from design

## Definition of Done

- [ ] Implementation complete
- [ ] Unit tests pass
- [ ] Exported from memory module
- [ ] Code compiles without warnings
