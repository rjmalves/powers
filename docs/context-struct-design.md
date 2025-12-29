# Context Struct Design

> **Epic**: [Epic 3: Algorithm Separation](../plans/clean-code-refactoring/epic-03-algorithm-separation/00-epic-overview.md)
> **Status**: ✅ Complete

---

## Overview

This document defines the context struct interfaces for SDDP algorithm phases and documents preallocation opportunities for future memory optimization.

---

## Table of Contents

1. [ForwardPassContext](#forwardpasscontext)
2. [BackwardPassContext](#backwardpasscontext)
3. [Preallocation Analysis](#preallocation-analysis)
4. [Lifetime Analysis](#lifetime-analysis)
5. [Integration Strategy](#integration-strategy)

---

## ForwardPassContext

```rust
/// Context for forward pass execution.
///
/// Bundles all data needed for a single forward pass trajectory, reducing
/// parameter counts and making data flow explicit.
///
/// # Preallocation Opportunities
///
/// All data sizes are known at initialization from input:
///
/// | Data | Source | Size |
/// |------|--------|------|
/// | `realizations` | `config.stages` | One per stage |
/// | Stage solution buffers | `subproblem.num_variables()` | Per stage |
/// | State coefficients | `system.hydros.len()` | Per hydro |
///
/// # Thread Safety
///
/// This context is NOT thread-safe. For parallel forward passes, create
/// one context per trajectory with disjoint graph node references.
///
/// # Example
///
/// ```ignore
/// use powers_rs::algorithm::ForwardPassContext;
///
/// let mut ctx = ForwardPassContext::new(
///     &mut subproblem_graph,
///     &mut realization_graph,
///     &sampled_noises,
///     &graph_bfs_table,
///     &study_period_ids,
///     &iteration_timing.forward,
/// );
///
/// let result = forward_pass::execute(&mut ctx)?;
/// ```
pub struct ForwardPassContext<'a> {
    /// Mutable access to subproblem graph for LP operations.
    pub subproblem_graph: &'a mut DirectedGraph<Subproblem>,

    /// Mutable access to realization graph for storing results.
    pub realization_graph: &'a mut DirectedGraph<Realization>,

    /// Sampled noises for this trajectory's forward pass.
    /// Indexed by node ID.
    pub sampled_noises: &'a [&'a OptimizedSampledBranchingNoises],

    /// BFS traversal table for past node lookup.
    /// `graph_bfs_table[idx]` gives past node IDs for `study_period_ids[idx]`.
    pub graph_bfs_table: &'a [Vec<usize>],

    /// IDs of stages to visit in this trajectory, in execution order.
    pub study_period_ids: &'a [usize],

    /// Timing storage for this forward pass.
    /// Uses `Cell<Duration>` for interior mutability.
    pub timing: &'a ForwardTiming,
}
```

### ForwardPassContext Fields

| Field | Type | Description |
|-------|------|-------------|
| `subproblem_graph` | `&'a mut DirectedGraph<Subproblem>` | Graph of LP subproblems |
| `realization_graph` | `&'a mut DirectedGraph<Realization>` | Graph of solution realizations |
| `sampled_noises` | `&'a [&'a OptimizedSampledBranchingNoises]` | Sampled scenario noises |
| `graph_bfs_table` | `&'a [Vec<usize>]` | Past node IDs for each stage |
| `study_period_ids` | `&'a [usize]` | Stage IDs to visit |
| `timing` | `&'a ForwardTiming` | Timing accumulator |

### ForwardPassResult

```rust
/// Result of a forward pass execution.
#[derive(Debug, Clone)]
pub struct ForwardPassResult {
    /// Total trajectory cost (sum of stage objectives).
    pub trajectory_cost: f64,

    /// Number of solver calls made during this forward pass.
    pub solver_calls: usize,
}
```

---

## BackwardPassContext

```rust
/// Context for backward pass execution.
///
/// Bundles all data needed for backward pass, including cut computation
/// and FCF updates.
///
/// # Preallocation Opportunities
///
/// | Data | Source | Size |
/// |------|--------|------|
/// | Branching realizations | `config.backward_scenarios` × `config.stages` | Per backward |
/// | Cut coefficients | `state.num_coefficients()` | Per cut |
/// | Dual values buffer | `subproblem.num_constraints()` | Per solve |
///
/// # Thread Safety
///
/// The backward pass has parallel branching solves. Context is NOT thread-safe;
/// each thread should have its own context for its assigned work.
pub struct BackwardPassContext<'a> {
    /// Mutable access to subproblem graph.
    pub subproblem_graph: &'a mut DirectedGraph<Subproblem>,

    /// Mutable access to realization graph (read from forward, write branchings).
    pub realization_graph: &'a mut DirectedGraph<Realization>,

    /// Mutable access to branching graph for backward solves.
    pub branching_graph: &'a mut DirectedGraph<Vec<Realization>>,

    /// Node data graph for backward traversal.
    pub node_data_graph: &'a DirectedGraph<NodeData>,

    /// Scenario tree for branching.
    pub saa: &'a ScenarioTree,

    /// FCF for cut updates.
    pub fcf: &'a mut Fcf,

    /// Risk measure for cut computation.
    pub risk_measure: &'a dyn RiskMeasure,

    /// Current iteration number.
    pub iteration: usize,

    /// Forward pass index (for training state identification).
    pub forward_pass_idx: usize,

    /// Timing storage for this backward pass.
    pub timing: &'a BackwardTiming,
}
```

### BackwardPassContext Fields

| Field | Type | Description |
|-------|------|-------------|
| `subproblem_graph` | `&'a mut DirectedGraph<Subproblem>` | Graph of LP subproblems |
| `realization_graph` | `&'a mut DirectedGraph<Realization>` | Forward pass realizations |
| `branching_graph` | `&'a mut DirectedGraph<Vec<Realization>>` | Backward branching results |
| `node_data_graph` | `&'a DirectedGraph<NodeData>` | Node metadata |
| `saa` | `&'a ScenarioTree` | Scenario tree for branching |
| `fcf` | `&'a mut Fcf` | Future cost function |
| `risk_measure` | `&'a dyn RiskMeasure` | Risk measure for cuts |
| `iteration` | `usize` | Current iteration |
| `forward_pass_idx` | `usize` | Forward pass index |
| `timing` | `&'a BackwardTiming` | Timing accumulator |

---

## Preallocation Analysis

### Known at Initialization

From input data, we know:

| Parameter | Source | Used By |
|-----------|--------|---------|
| `num_stages` | `config.stages` | Realization graph, study_period_ids |
| `num_hydros` | `system.hydros.len()` | State coefficients, water values |
| `num_thermals` | `system.thermals.len()` | Thermal generation |
| `num_buses` | `system.buses.len()` | Deficit, load balance |
| `num_forward_passes` | `config.forward_passes` | Parallel trajectories |
| `num_backward_scenarios` | SAA per stage | Branching realizations |

### Preallocatable Structures

| Structure | Count | Size Each | Total |
|-----------|-------|-----------|-------|
| `Realization` per trajectory | `num_forward_passes × num_stages` | ~500 bytes | ~500KB |
| State coefficients | `num_stages` | `num_hydros × 8` bytes | ~100KB |
| Solution buffer | `num_stages` | `num_variables × 8` bytes | ~1MB |
| Cut coefficients | Per cut | `num_hydros × 8` bytes | Grows |

### Current Allocation Pattern

1. **Per-iteration**: SAA sampling creates new noise references
2. **Per-stage**: `Realization` fields populated via `clone_from_slice`
3. **Per-cut**: `CutData` allocated, stored in FCF

### Migration Path

1. **Phase 1** (Epic 3): Context structs with references
2. **Phase 2** (Epic 5): Add preallocated buffer fields
3. **Phase 3** (Epic 5): Replace allocations with pool access

---

## Lifetime Analysis

### ForwardPassContext Lifetimes

```
'a: Lifetime of all borrowed data
    ├── subproblem_graph: Lives for entire SDDP run
    ├── realization_graph: Lives for entire SDDP run
    ├── sampled_noises: Lives for current iteration
    ├── graph_bfs_table: Lives for entire SDDP run
    ├── study_period_ids: Lives for entire SDDP run
    └── timing: Lives for current iteration
```

Key observations:
- Most data lives for the entire SDDP run
- `sampled_noises` is regenerated each iteration
- `timing` is reset each iteration

### BackwardPassContext Lifetimes

```
'a: Lifetime of all borrowed data
    ├── subproblem_graph: Lives for entire SDDP run
    ├── realization_graph: Lives for entire SDDP run
    ├── branching_graph: Lives for entire SDDP run
    ├── node_data_graph: Lives for entire SDDP run
    ├── saa: Lives for entire SDDP run
    ├── fcf: Lives for entire SDDP run
    ├── risk_measure: Lives for entire SDDP run
    └── timing: Lives for current iteration
```

---

## Integration Strategy

### Step 1: Create Context Types (T-019, T-023)

Define context structs in `src/algorithm/context.rs`:

```rust
// src/algorithm/context.rs

pub mod forward;
pub mod backward;

pub use forward::{ForwardPassContext, ForwardPassResult, TrajectoryTiming};
pub use backward::{BackwardPassContext, BackwardPassResult};
```

### Step 2: Create Forward Pass Module (T-020)

Move the forward pass logic:

```rust
// src/algorithm/forward_pass.rs

pub fn execute(ctx: &mut ForwardPassContext) -> Result<ForwardPassResult, String> {
    // Same logic, cleaner interface
}
```

### Step 3: Update Call Sites (T-022)

Replace scattered parameters with context construction:

**Before:**
```rust
let (cost, timing) = trajectory.forward(
    sampled_noises,
    graph_bfs_table,
    study_period_ids,
)?;
```

**After:**
```rust
let mut ctx = ForwardPassContext::new(
    &mut trajectory.subproblem_graph,
    &mut trajectory.realization_graph,
    &sampled_noises,
    &graph_bfs_table,
    &study_period_ids,
    &iteration_timing.forward,
);
let result = forward_pass::execute(&mut ctx)?;
```

### Parameter Count Reduction

| Function | Before | After |
|----------|--------|-------|
| `forward()` | 3 explicit + `&mut self` | 1 context |
| `backward()` | 5+ explicit + `&mut self` | 1 context |
| `step()` | 3 explicit | 3 explicit (unchanged) |

---

## Thread Safety Considerations

### Forward Pass

The training loop executes multiple forward passes in parallel (via rayon). Each trajectory operates on:
- Different graph nodes (disjoint)
- Same noise data (read-only, shared via reference)
- Separate timing accumulators (per-thread)

**Design**: Create one `ForwardPassContext` per trajectory with disjoint mutable references.

### Backward Pass

The backward pass has two levels of parallelism:
1. Per-stage: Sequential (dependencies between stages)
2. Per-branching: Parallel (independent scenarios)

**Design**: Create contexts at the appropriate granularity. For branching parallelism, the context holds shared immutable data while each thread has its own mutable buffers.

---

## Future Enhancements (Epic 5)

### Preallocated Buffer Fields

```rust
pub struct ForwardPassContext<'a> {
    // ... existing fields ...

    // Future: Preallocated buffers
    // pub solution_buffers: &'a mut [SolutionBuffer],
    // pub basis_buffers: &'a mut [BasisBuffer],
}
```

### Memory Pool Integration

```rust
pub struct ForwardPassContext<'a> {
    // ... existing fields ...

    // Future: Memory pool access
    // pub memory_pool: &'a MemoryPool,
}
```

These additions will enable:
- Zero-allocation forward passes
- Reusable solution storage
- Pool-based cut allocation
