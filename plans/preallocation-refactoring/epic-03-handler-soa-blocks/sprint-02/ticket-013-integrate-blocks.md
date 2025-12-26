# [TICKET-013] Integrate RealizationBlock with handler

> **Epic**: [Epic 3: Handler-Level SoA Blocks](../00-epic-overview.md)  
> **Sprint**: [Sprint 2](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-012](../sprint-01/ticket-012-implement-realization-block.md)  
> **Blocks**: [TICKET-016](./ticket-016-performance-validation.md)

## Context

### Background

RealizationBlock is implemented but not integrated with SddpTrainHandler. This ticket wires the block into the forward pass data flow.

### Relation to Epic

Core integration that enables cache locality benefits.

## Files to Read Before Starting

- `src/sddp/mod.rs` - SddpTrainHandler structure
- `src/memory/blocks.rs` - RealizationBlock implementation
- Forward pass code that accesses realization data

## Specification

### Integration Strategy

**Option A: Replace realization_graph entirely**
- High impact, high risk
- Requires changing all access patterns

**Option B: Add RealizationBlock alongside, migrate gradually** (Recommended)
- Add block to handler
- Update forward pass to write to block
- Keep graph for backward pass (initially)
- Migrate backward pass later

### Changes Required

1. Add `realizations: RealizationBlock` to SddpTrainHandler
2. Initialize block in handler construction
3. Update forward pass to write realization data to block
4. Update backward pass to read from block (or keep graph initially)

## Acceptance Criteria

- [ ] RealizationBlock added to SddpTrainHandler
- [ ] Forward pass writes realization data to block
- [ ] Examples produce identical results
- [ ] No performance regression

## Implementation Guide

### Suggested Approach

1. Add RealizationBlock field to SddpTrainHandler
2. Initialize in handler constructor
3. Find forward pass realization write code
4. Update to write to block (AND graph, for compatibility)
5. Gradually migrate reads to block
6. Remove graph writes when all reads migrated

### Key Files to Modify

- `src/sddp/mod.rs`: SddpTrainHandler struct and methods

### Code Pattern

```rust
// SddpTrainHandler struct
pub struct SddpTrainHandler {
    // Existing fields
    subproblem_graph: DirectedGraph<Subproblem>,
    realization_graph: DirectedGraph<Realization>,  // Keep for now
    
    // NEW: Contiguous block for cache-efficient access
    realization_block: RealizationBlock,
    
    // ...
}

// Forward pass - write to both (temporary)
fn forward_pass_solve(...) {
    // ... solve subproblem ...
    
    // Write to graph (existing)
    self.realization_graph.get_node_mut(stage_id).data.loads = loads.clone();
    
    // Write to block (new)
    let block_loads = self.realization_block.get_loads_mut(stage_id);
    block_loads.copy_from_slice(&loads);
}

// Later: migrate reads to use block
fn backward_pass_access(...) {
    // Before: self.realization_graph.get_node(stage_id).data.loads
    // After: self.realization_block.get_loads(stage_id)
}
```

### Pitfalls to Avoid

- ⚠️ Don't remove graph access until all code migrated
- ⚠️ Ensure stage_id matches between graph and block
- ⚠️ Handle pre-study stage (stage 0) correctly

## Testing Requirements

### Integration Tests

- [ ] Example 01: Identical results
- [ ] Example 07: Identical results (convergence within 0.001%)

### Validation

```bash
cargo run --release -- run examples/01-deterministic 2>&1 | grep "lower"
cargo run --release -- run examples/07-par-model-with-inflow-state 2>&1 | tail -5
```

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Requires understanding data flow through handler

## Definition of Done

- [ ] RealizationBlock integrated with handler
- [ ] Forward pass writes to block
- [ ] Examples produce correct results
- [ ] No performance regression
