# [T-023] Revise BackwardPassContext for Coordinator Architecture

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Handler Coordination Infrastructure](./00-sprint-overview.md)
> **Dependencies**: [T-021](../sprint-01/ticket-021-forward-timing-integration.md) rework complete
> **Blocks**: [T-024](./ticket-024-design-processor-trait.md)
> **Status**: ⬜ Not Started

---

## ⚠️ ARCHITECTURAL DECISION: Timing Separation

Per the lesson learned from T-021, **timing must NOT be inside context structs**.

The existing `BackwardPassContext` has a timing field that must be removed:
```rust
// CURRENT (problematic):
pub struct BackwardPassContext<'a> {
    pub timing: &'a BackwardTiming,  // ❌ Will cause borrow conflicts
    // ...
}

// REQUIRED (correct):
pub struct BackwardPassContext<'a> {
    // NO timing field
    // ...
}
```

---

## Files to Read Before Starting

- `src/algorithm/context.rs` - Current `BackwardPassContext` implementation
- `src/sddp/mod.rs:1760-2060` - Current backward pass loop
- [T-021](../sprint-01/ticket-021-forward-timing-integration.md) - Timing separation rationale
- `BACKWARD_PASS_EXTRACTION_ANALYSIS.md` - Architecture analysis

---

## Context

### Current State

`BackwardPassContext` was created with a timing field (and helper methods) but this will cause the same borrow conflicts as `ForwardPassContext`:

```rust
pub struct BackwardPassContext<'a> {
    pub node_data_graph: &'a DirectedGraph<NodeData>,
    pub fcf_graph: &'a DirectedGraph<Mutex<FutureCostFunction>>,
    pub saa: &'a ScenarioTree,
    pub graph_bfs_table: &'a [Vec<usize>],
    pub study_period_ids: &'a [usize],
    pub iteration: usize,
    pub enable_cut_selection: bool,
    pub timing: &'a BackwardTiming,  // ❌ REMOVE THIS
}
```

### Target State

1. **Remove timing from `BackwardPassContext`**
2. **Add `BackwardStageContext`** for per-stage data passed to processor
3. **Neither context contains timing** - timing passed separately

---

## Specification

### Step 1: Update `BackwardPassContext` - Remove Timing

```rust
/// Context for backward pass execution.
///
/// # Design Note: Timing Separation
///
/// Timing is NOT included in this context to avoid borrow checker conflicts.
/// When using `TimingGuard`, the guard borrows the timing struct. If timing
/// were inside this context, we couldn't mutably access other fields while
/// timing is active.
///
/// Pass timing as a separate parameter to `backward_pass::execute()`.
pub struct BackwardPassContext<'a> {
    /// Node data graph for backward traversal information.
    pub node_data_graph: &'a DirectedGraph<NodeData>,

    /// FCF graph for cut updates.
    pub fcf_graph: &'a DirectedGraph<Mutex<FutureCostFunction>>,

    /// Scenario tree for branching scenario generation.
    pub saa: &'a ScenarioTree,

    /// BFS traversal table for past node lookup.
    pub graph_bfs_table: &'a [Vec<usize>],

    /// IDs of stages (study periods) in forward order.
    pub study_period_ids: &'a [usize],

    /// Current iteration number (1-indexed).
    pub iteration: usize,

    /// Whether cut selection is enabled.
    pub enable_cut_selection: bool,
    
    // NO timing field - passed separately to backward_pass::execute()
}
```

### Step 2: Add `BackwardStageContext` - No Timing

```rust
/// Per-stage context for backward pass processing.
///
/// Passed to `BackwardStageProcessor` methods. Contains the specific
/// data needed to process a single stage.
///
/// # Design Note: Timing Separation
///
/// Like `BackwardPassContext`, timing is NOT included here.
/// Timing is accumulated in `BackwardPassTimingAccumulator` which
/// is passed separately to the backward pass functions.
pub struct BackwardStageContext<'a> {
    /// Current stage ID.
    pub stage_id: usize,
    
    /// Stage index in the study period sequence.
    pub stage_idx: usize,
    
    /// Past node IDs for this stage (BFS path).
    pub past_node_ids: &'a [usize],
    
    /// Parent stage ID (for FCF updates). None for first stage.
    pub parent_id: Option<usize>,
    
    /// Node data graph reference.
    pub node_data_graph: &'a DirectedGraph<NodeData>,
    
    /// Scenario tree for branching counts.
    pub saa: &'a ScenarioTree,
    
    /// Current iteration number.
    pub iteration: usize,
    
    /// Whether cut selection is enabled.
    pub enable_cut_selection: bool,
    
    // NO timing field
}
```

### Step 3: Add Builder Method

```rust
impl<'a> BackwardPassContext<'a> {
    /// Create a stage context for the given stage index.
    pub fn stage_context(&'a self, stage_idx: usize) -> Option<BackwardStageContext<'a>> {
        let stage_id = self.study_period_ids.get(stage_idx).copied()?;
        let past_node_ids = self.graph_bfs_table.get(stage_idx)?;
        let parent_id = if stage_idx > 0 {
            past_node_ids.last().copied()
        } else {
            None
        };
        
        Some(BackwardStageContext {
            stage_id,
            stage_idx,
            past_node_ids,
            parent_id,
            node_data_graph: self.node_data_graph,
            saa: self.saa,
            iteration: self.iteration,
            enable_cut_selection: self.enable_cut_selection,
        })
    }
}
```

### Step 4: Update Exports

```rust
// src/algorithm/mod.rs
pub use context::{
    BackwardPassContext, BackwardPassResult, BackwardStageContext,
    BackwardStageTiming, ForwardPassContext, ForwardPassResult,
    TrajectoryTiming,
};
```

---

## Acceptance Criteria

- [ ] `BackwardPassContext` does NOT contain timing field
- [ ] `BackwardStageContext` added without timing field
- [ ] `BackwardPassContext::stage_context()` builder method works
- [ ] Documentation explains timing separation pattern
- [ ] Unit tests for stage context creation
- [ ] `cargo build -j1` succeeds
- [ ] `cargo test -j1` passes
- [ ] Golden tests pass

---

## Key Files to Modify

| File | Action |
|------|--------|
| `src/algorithm/context.rs` | REMOVE timing from `BackwardPassContext`, ADD `BackwardStageContext` |
| `src/algorithm/mod.rs` | UPDATE exports |

---

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Clear specification, builds on T-021 pattern

---

## Definition of Done

- [ ] Timing removed from `BackwardPassContext`
- [ ] `BackwardStageContext` added
- [ ] Builder method works
- [ ] Documentation complete
- [ ] Tests pass
- [ ] Code reviewed
