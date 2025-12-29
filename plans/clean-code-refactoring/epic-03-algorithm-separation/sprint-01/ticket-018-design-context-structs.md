# [T-018] Design Context Structs with Preallocation Awareness

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Forward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: Epic 2 complete
> **Blocks**: [T-019](./ticket-019-forward-pass-context.md), [T-023](../sprint-02/ticket-023-backward-pass-context.md)

---

## ⚠️ CRITICAL: Design Only

This ticket is **design and documentation only**—minimal code changes. The goal is to define the context struct interfaces and document preallocation opportunities before implementing them.

---

## Files to Read Before Starting

- `src/sddp/mod.rs:596-680` - Current `forward()` function signature and parameters
- `src/sddp/mod.rs:1310-1500` - Training loop forward pass call patterns
- `src/sddp/mod.rs:2519-2540` - `step()` function signature
- `src/timing/metrics.rs` - `IterationTiming`, `ForwardTiming`, `BackwardTiming`
- `src/subproblem.rs` - `Realization` struct
- `plans/clean-code-refactoring/00-master-plan.md` - Context struct requirements

---

## Context

### Background

The current SDDP implementation passes 6-10 parameters through each function call, making the code hard to read and modify. Context structs bundle related parameters into coherent groups, reducing parameter counts and making data flow explicit.

**Current pattern** (in `sddp/mod.rs`):
```rust
fn forward(
    &mut self,
    sampled_noises: Vec<&scenario::OptimizedSampledBranchingNoises>,
    graph_bfs_table: &[Vec<usize>],
    study_period_ids: &[usize],
) -> Result<(f64, ForwardPassTimingAccumulator), String>
```

**Target pattern**:
```rust
fn execute(ctx: &mut ForwardPassContext) -> Result<ForwardPassResult, Error>
```

### Design Constraints

1. **Preallocation Awareness**: Structs must document what can be preallocated
2. **Lifetime Correctness**: Use references to avoid ownership issues
3. **Minimal Overhead**: Context should be cheap to construct
4. **Clear Ownership**: Each field has one clear owner
5. **Future-Ready**: Enable Epic 5 memory optimization

---

## Specification

### Deliverable

Create a design document at `docs/context-struct-design.md` containing:

1. **ForwardPassContext** - Full struct definition with doc comments
2. **BackwardPassContext** - Full struct definition with doc comments
3. **Preallocation Analysis** - What can be preallocated from input data
4. **Lifetime Analysis** - How borrows work through the algorithm
5. **Integration Strategy** - How to migrate existing code

### ForwardPassContext Design

```rust
/// Context for forward pass execution.
///
/// Bundles all data needed for a single forward pass, reducing parameter
/// counts and making data flow explicit.
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
/// # Lifetime Design
///
/// - `'graph`: Lifetime of the subproblem and realization graphs
/// - `'saa`: Lifetime of the sampled noises (per iteration)
/// - `'timing`: Lifetime of timing storage (per iteration)
///
/// # Example
///
/// ```ignore
/// let mut ctx = ForwardPassContext {
///     subproblem_graph: &mut engine.subproblem_graph,
///     realization_graph: &mut engine.realization_graph,
///     sampled_noises: &sampled_noises,
///     graph_bfs_table: &graph_bfs_table,
///     study_period_ids: &study_period_ids,
///     timing: &iteration_timing.forward,
/// };
/// let (cost, _) = forward_pass::execute(&mut ctx)?;
/// ```
pub struct ForwardPassContext<'graph, 'saa, 'timing> {
    /// Mutable access to subproblem graph for LP operations.
    pub subproblem_graph: &'graph mut graph::DirectedGraph<subproblem::Subproblem>,

    /// Mutable access to realization graph for storing results.
    pub realization_graph: &'graph mut graph::DirectedGraph<subproblem::Realization>,

    /// Sampled noises for this iteration's forward passes.
    pub sampled_noises: &'saa [&'saa scenario::OptimizedSampledBranchingNoises],

    /// BFS traversal table for past node lookup.
    pub graph_bfs_table: &'graph [Vec<usize>],

    /// IDs of stages to visit in this trajectory.
    pub study_period_ids: &'graph [usize],

    /// Timing storage for this forward pass.
    pub timing: &'timing ForwardTiming,
}
```

### BackwardPassContext Design

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
pub struct BackwardPassContext<'a, 'timing> {
    /// Mutable access to subproblem graph.
    pub subproblem_graph: &'a mut graph::DirectedGraph<subproblem::Subproblem>,

    /// Mutable access to realization graph (read from forward, write branchings).
    pub realization_graph: &'a mut graph::DirectedGraph<subproblem::Realization>,

    /// Node data graph for backward traversal.
    pub node_data_graph: &'a graph::DirectedGraph<NodeData>,

    /// Scenario tree for branching.
    pub saa: &'a scenario::ScenarioTree,

    /// FCF for cut updates (may need Arc<Mutex<>> for thread safety).
    pub fcf: &'a mut fcf::Fcf,

    /// Risk measure for cut computation.
    pub risk_measure: &'a dyn risk_measure::RiskMeasure,

    /// Current iteration number.
    pub iteration: usize,

    /// Timing storage for this backward pass.
    pub timing: &'timing BackwardTiming,
}
```

### Preallocation Analysis

Document what can be preallocated before the training loop starts:

```markdown
## Preallocation Analysis

### Known at Initialization

From input data, we know:
- `num_stages`: Number of stages in the study period
- `num_hydros`: Number of hydroelectric plants
- `num_thermals`: Number of thermal plants
- `num_buses`: Number of buses in the network
- `num_forward_passes`: Number of parallel forward trajectories
- `num_backward_scenarios`: Number of branching scenarios per backward step

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
```

### Integration Strategy

Document how to migrate existing code:

```markdown
## Integration Strategy

### Step 1: Create Context Types (T-019, T-023)

Define context structs in `src/algorithm/context.rs`.

### Step 2: Create Parallel Wrapper (T-020)

The forward pass runs parallel trajectories. Create a per-trajectory context:

```rust
struct TrajectoryContext<'a> {
    subproblem_node: &'a mut graph::Node<subproblem::Subproblem>,
    realization_node: &'a mut graph::Node<subproblem::Realization>,
    noises: &'a scenario::OptimizedSampledBranchingNoises,
    past_realizations: Vec<&'a subproblem::Realization>,
}
```

### Step 3: Extract to Module (T-020)

Move the logic, keeping the same structure:

```rust
// src/algorithm/forward_pass.rs

pub fn execute(ctx: &mut ForwardPassContext) -> Result<ForwardPassResult, Error> {
    // Same logic, cleaner interface
}
```

### Step 4: Update Call Sites (T-022)

Replace scattered parameters with context construction:

```rust
// Before:
let (cost, timing) = trajectory.forward(
    sampled_noises,
    graph_bfs_table,
    study_period_ids,
)?;

// After:
let mut ctx = ForwardPassContext {
    subproblem_graph: &mut trajectory.subproblem_graph,
    realization_graph: &mut trajectory.realization_graph,
    sampled_noises: &sampled_noises,
    graph_bfs_table: &graph_bfs_table,
    study_period_ids: &study_period_ids,
    timing: &iteration_timing.forward,
};
let result = forward_pass::execute(&mut ctx)?;
```
```

---

## Acceptance Criteria

- [ ] Design document created at `docs/context-struct-design.md`
- [ ] `ForwardPassContext` fully documented with all fields
- [ ] `BackwardPassContext` fully documented with all fields
- [ ] Preallocation opportunities identified and documented
- [ ] Lifetime analysis complete
- [ ] Integration strategy documented
- [ ] No breaking code changes
- [ ] `cargo build` still succeeds
- [ ] `cargo test` still passes
- [ ] Golden tests still pass

---

## Implementation Guide

### Suggested Approach

1. **Analyze current function signatures**:
   ```bash
   grep -n "fn forward\|fn backward\|fn step" src/sddp/mod.rs | head -20
   ```

2. **Identify all parameters** passed through the call chain

3. **Group parameters** by:
   - Graph data (subproblems, realizations)
   - SAA data (noises, scenarios)
   - Configuration (stage IDs, BFS tables)
   - Timing (accumulators)

4. **Design lifetime annotations**:
   - What lives for the whole algorithm run?
   - What lives for one iteration?
   - What lives for one stage/trajectory?

5. **Document preallocation opportunities**:
   - Read input data structures
   - Note all size-determining fields
   - Calculate memory requirements

6. **Write integration examples** showing before/after

### Key Files to Analyze

| File | Lines | Purpose |
|------|-------|---------|
| `src/sddp/mod.rs` | 596-680 | Forward pass implementation |
| `src/sddp/mod.rs` | 750-900 | Backward pass implementation |
| `src/sddp/mod.rs` | 2519-2540 | Step function |
| `src/timing/metrics.rs` | All | Timing types |

### Pitfalls to Avoid

- ⚠️ Don't try to make context structs thread-safe at this stage
- ⚠️ Don't add preallocated buffers yet—that's Epic 5
- ⚠️ Don't change existing code—this is design only
- ⚠️ Be careful with lifetime annotations—get them right in design
- ⚠️ Document thread safety requirements for backward pass

---

## Testing Requirements

### Verification

- [ ] Design document is complete and readable
- [ ] Struct definitions compile (can test in scratch file)
- [ ] Lifetime annotations are consistent
- [ ] Preallocation analysis is accurate

### No Functional Tests

This ticket produces documentation only. No functional tests needed.

---

## Documentation Requirements

- [ ] Create `docs/context-struct-design.md`
- [ ] Include table of contents
- [ ] Include code examples for each context struct
- [ ] Include preallocation analysis table
- [ ] Include integration strategy with before/after examples

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Design work with clear scope; requires careful analysis of existing code

---

## Definition of Done

- [ ] Design document complete
- [ ] `ForwardPassContext` designed and documented
- [ ] `BackwardPassContext` designed and documented
- [ ] Preallocation analysis complete
- [ ] Integration strategy documented
- [ ] No code changes
- [ ] Reviewed by team
