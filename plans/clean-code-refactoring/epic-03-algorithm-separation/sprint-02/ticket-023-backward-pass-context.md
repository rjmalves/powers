# [T-023] Create BackwardPassContext Struct

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Backward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [Sprint 1](../sprint-01/00-sprint-overview.md) complete
> **Blocks**: [T-024](./ticket-024-extract-backward-pass.md)

---

## ⚠️ CRITICAL: Structural Change Only

This ticket creates the `BackwardPassContext` struct and related types. **No algorithm logic changes.** The context will be used by subsequent tickets to extract backward pass logic.

---

## Files to Read Before Starting

- `docs/context-struct-design.md` - Design from T-018
- `src/sddp/mod.rs:680-1000` - Current backward pass area
- `src/sddp/mod.rs:688-820` - `compute_cut_data_for_backward_step()`
- `src/algorithm/context.rs` - `ForwardPassContext` pattern from T-019
- `src/fcf.rs` - FCF (Future Cost Function) types
- `src/risk_measure.rs` - Risk measure trait

---

## Context

### Background

The `BackwardPassContext` bundles all data needed for backward pass execution. The backward pass is more complex than the forward pass due to:

1. **Reverse iteration** - Stages processed in reverse order
2. **Branching** - Multiple scenarios solved at each stage
3. **Cut computation** - Benders cuts generated from dual values
4. **FCF updates** - Cuts added to the future cost function
5. **Parallel execution** - Branching solves can be parallelized

### Current State

The backward pass has many parameters and complex state management:

```rust
// Current pattern (scattered across methods)
fn compute_cut_data_for_backward_step(
    &mut self,
    id: usize,
    past_node_ids: &[usize],
    node_data_graph: &graph::DirectedGraph<NodeData>,
    saa: &scenario::ScenarioTree,
    iteration: usize,
    forward_pass_idx: usize,
) -> Result<(fcf::CutData, BackwardPhase1Timing), String>
```

### Target State

A coherent context struct that bundles related data:

```rust
pub struct BackwardPassContext<'a> {
    // ... bundled parameters ...
}
```

---

## Specification

### Add to `src/algorithm/context.rs`

```rust
use crate::fcf::Fcf;
use crate::risk_measure::RiskMeasure;
use crate::scenario::ScenarioTree;
use crate::sddp::NodeData;
use crate::timing::BackwardTiming;

/// Context for backward pass execution.
///
/// Bundles all data needed for backward pass, including cut computation
/// and FCF updates. The backward pass iterates through stages in reverse
/// order, computing Benders cuts at each stage.
///
/// # Preallocation Opportunities
///
/// | Data | Source | Size |
/// |------|--------|------|
/// | Branching realizations | `config.backward_scenarios × num_stages` | ~1MB |
/// | Cut coefficients | `num_state_variables` | Per cut |
/// | Dual values buffer | `subproblem.num_constraints()` | Per solve |
///
/// # Thread Safety
///
/// The backward pass has parallel branching solves within each stage.
/// This context is NOT thread-safe—each thread working on branchings
/// should have access to disjoint portions of the data.
///
/// # Algorithm Phases
///
/// The backward pass has three phases per stage:
/// 1. **Phase 1**: Parallel branching solves (generate cut data)
/// 2. **Phase 2**: Sequential FCF updates (critical section)
/// 3. **Phase 3**: Handler application
///
/// # Example
///
/// ```ignore
/// use powers_rs::algorithm::BackwardPassContext;
///
/// let mut ctx = BackwardPassContext::new(
///     &mut subproblem_graph,
///     &mut realization_graph,
///     &node_data_graph,
///     &saa,
///     &mut fcf,
///     risk_measure.as_ref(),
///     iteration,
///     &iteration_timing.backward,
/// );
///
/// let result = backward_pass::execute(&mut ctx)?;
/// ```
pub struct BackwardPassContext<'a> {
    /// Mutable access to subproblem graph for LP operations.
    pub subproblem_graph: &'a mut DirectedGraph<Subproblem>,

    /// Mutable access to realization graph (read from forward, write branchings).
    pub realization_graph: &'a mut DirectedGraph<Realization>,

    /// Node data graph for backward traversal information.
    pub node_data_graph: &'a DirectedGraph<NodeData>,

    /// Scenario tree for branching scenario generation.
    pub saa: &'a ScenarioTree,

    /// Mutable access to FCF for cut updates.
    /// Note: In parallel execution, this may need Arc<Mutex<>> wrapper.
    pub fcf: &'a mut Fcf,

    /// Risk measure for cut computation (CVaR, expectation, etc.).
    pub risk_measure: &'a dyn RiskMeasure,

    /// Current iteration number.
    pub iteration: usize,

    /// Current forward pass index (for multi-trajectory training).
    pub forward_pass_idx: usize,

    /// Timing storage for this backward pass.
    pub timing: &'a BackwardTiming,

    /// Stage IDs in backward order (reverse of forward order).
    pub backward_stage_ids: &'a [usize],
}

impl<'a> BackwardPassContext<'a> {
    /// Create a new backward pass context.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subproblem_graph: &'a mut DirectedGraph<Subproblem>,
        realization_graph: &'a mut DirectedGraph<Realization>,
        node_data_graph: &'a DirectedGraph<NodeData>,
        saa: &'a ScenarioTree,
        fcf: &'a mut Fcf,
        risk_measure: &'a dyn RiskMeasure,
        iteration: usize,
        forward_pass_idx: usize,
        timing: &'a BackwardTiming,
        backward_stage_ids: &'a [usize],
    ) -> Self {
        Self {
            subproblem_graph,
            realization_graph,
            node_data_graph,
            saa,
            fcf,
            risk_measure,
            iteration,
            forward_pass_idx,
            timing,
            backward_stage_ids,
        }
    }

    /// Get the number of stages to process in backward pass.
    #[inline]
    pub fn num_stages(&self) -> usize {
        self.backward_stage_ids.len()
    }

    /// Get the subproblem node for a given ID.
    #[inline]
    pub fn get_subproblem_mut(
        &mut self,
        id: usize,
    ) -> Option<&mut crate::graph::Node<Subproblem>> {
        self.subproblem_graph.get_node_mut(id)
    }

    /// Get the realization node for a given ID.
    #[inline]
    pub fn get_realization(&self, id: usize) -> Option<&crate::graph::Node<Realization>> {
        self.realization_graph.get_node(id)
    }

    /// Get the node data for a given ID.
    #[inline]
    pub fn get_node_data(&self, id: usize) -> Option<&crate::graph::Node<NodeData>> {
        self.node_data_graph.get_node(id)
    }
}

/// Result of a backward pass execution.
#[derive(Debug, Clone)]
pub struct BackwardPassResult {
    /// Lower bound from first stage evaluation.
    pub lower_bound: f64,

    /// Number of cuts added during this backward pass.
    pub cuts_added: usize,

    /// Number of cuts removed (if cut management active).
    pub cuts_removed: usize,

    /// Number of solver calls made.
    pub solver_calls: usize,
}

impl BackwardPassResult {
    /// Create a new backward pass result.
    pub fn new(
        lower_bound: f64,
        cuts_added: usize,
        cuts_removed: usize,
        solver_calls: usize,
    ) -> Self {
        Self {
            lower_bound,
            cuts_added,
            cuts_removed,
            solver_calls,
        }
    }
}

/// Timing data collected during backward pass Phase 1 (per-stage).
///
/// This captures timing for the parallel branching solves at a single stage.
#[derive(Debug, Clone, Copy, Default)]
pub struct BackwardStageTiming {
    /// Time spent in model preprocessing for this stage.
    pub model_preprocessing: Duration,

    /// Time spent in solver for this stage.
    pub solver: Duration,

    /// Time spent in model postprocessing for this stage.
    pub model_postprocessing: Duration,

    /// Time spent computing the cut.
    pub cut_computation: Duration,

    /// Number of solver calls (branching scenarios solved).
    pub solver_calls: usize,
}

impl BackwardStageTiming {
    /// Add timing from another stage.
    pub fn add(&mut self, other: &BackwardStageTiming) {
        self.model_preprocessing += other.model_preprocessing;
        self.solver += other.solver;
        self.model_postprocessing += other.model_postprocessing;
        self.cut_computation += other.cut_computation;
        self.solver_calls += other.solver_calls;
    }
}
```

### Update `src/algorithm/mod.rs`

```rust
pub use context::{
    BackwardPassContext, BackwardPassResult, BackwardStageTiming,
    ForwardPassContext, ForwardPassResult, TrajectoryTiming,
};
```

---

## Acceptance Criteria

- [ ] `BackwardPassContext` struct added to `src/algorithm/context.rs`
- [ ] `BackwardPassResult` struct defined
- [ ] `BackwardStageTiming` struct defined
- [ ] All doc comments complete with preallocation and thread safety notes
- [ ] Unit tests for result and timing types
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] Golden tests pass (no behavioral changes)

### Correctness Verification

- [ ] No changes to existing algorithm code
- [ ] Types align with existing FCF, RiskMeasure, ScenarioTree types
- [ ] Lifetime annotations are correct

---

## Implementation Guide

### Suggested Approach

1. **Find required types**:
   ```bash
   grep -rn "pub struct NodeData\|pub struct ScenarioTree\|pub trait RiskMeasure" src/
   ```

2. **Open `src/algorithm/context.rs`**

3. **Add imports** for backward pass types:
   ```rust
   use crate::fcf::Fcf;
   use crate::risk_measure::RiskMeasure;
   // etc.
   ```

4. **Add `BackwardPassContext`** struct

5. **Add `BackwardPassResult`** and `BackwardStageTiming`**

6. **Update exports** in `mod.rs`

7. **Verify compilation**:
   ```bash
   cargo build 2>&1 | head -30
   ```

8. **Run tests**:
   ```bash
   cargo test context
   ```

### Key Files to Modify

| File | Action |
|------|--------|
| `src/algorithm/context.rs` | ADD backward pass types |
| `src/algorithm/mod.rs` | UPDATE exports |

### Pitfalls to Avoid

- ⚠️ The `RiskMeasure` is a trait object (`&dyn RiskMeasure`)
- ⚠️ `NodeData` may be in `sddp/mod.rs` or `sddp/instance.rs`
- ⚠️ Don't add thread safety (Arc/Mutex) yet—that's existing pattern
- ⚠️ Keep `#[allow(clippy::too_many_arguments)]` on constructor

---

## Testing Requirements

### Unit Tests

- [ ] Test `BackwardPassResult::new()`
- [ ] Test `BackwardStageTiming::add()`
- [ ] Test `BackwardStageTiming::default()`

### Compile Tests

- [ ] Context struct compiles with correct lifetimes
- [ ] Trait object reference works for `RiskMeasure`

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Struct-level doc comments on `BackwardPassContext`
- [ ] Document algorithm phases (Phase 1, 2, 3)
- [ ] Document thread safety characteristics
- [ ] Document preallocation opportunities
- [ ] Include usage example

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Similar to ForwardPassContext; main work is identifying correct types

---

## Definition of Done

- [ ] `BackwardPassContext` struct created
- [ ] `BackwardPassResult` struct created
- [ ] `BackwardStageTiming` struct created
- [ ] Module exports updated
- [ ] Unit tests passing
- [ ] Documentation complete
- [ ] `cargo build` succeeds
- [ ] Golden tests pass
- [ ] Code reviewed
