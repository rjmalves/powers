# [T-019] Create ForwardPassContext Struct

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Forward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-018](./ticket-018-design-context-structs.md)
> **Blocks**: [T-020](./ticket-020-extract-forward-step.md)

---

## ⚠️ CRITICAL: Structural Change Only

This ticket creates the `ForwardPassContext` struct and related types. **No algorithm logic changes.** The context will be used by subsequent tickets to reduce parameter counts.

---

## Files to Read Before Starting

- `docs/context-struct-design.md` - Design from T-018
- `src/sddp/mod.rs:596-680` - Current forward function
- `src/algorithm/mod.rs` - Algorithm module (currently placeholder)
- `src/timing/metrics.rs` - `ForwardTiming` struct
- `src/graph.rs` - `DirectedGraph` type
- `src/subproblem.rs` - `Subproblem`, `Realization` types
- `src/scenario.rs` - `OptimizedSampledBranchingNoises`

---

## Context

### Background

The `ForwardPassContext` bundles all data needed for forward pass execution into a single struct. This reduces the parameter count from 8+ to 1-2, making the code more readable and maintainable.

### Current State

The algorithm module is a placeholder:
```rust
// src/algorithm/mod.rs - currently empty
```

The forward pass takes scattered parameters:
```rust
fn forward(
    &mut self,
    sampled_noises: Vec<&scenario::OptimizedSampledBranchingNoises>,
    graph_bfs_table: &[Vec<usize>],
    study_period_ids: &[usize],
) -> Result<(f64, ForwardPassTimingAccumulator), String>
```

### Target State

A dedicated context file with properly documented structs:
```rust
// src/algorithm/context.rs
pub struct ForwardPassContext<'a> { ... }
pub struct ForwardPassResult { ... }
```

---

## Specification

### Files to Create

#### 1. `src/algorithm/context.rs`

```rust
//! Context structs for SDDP algorithm phases.
//!
//! These structs bundle related parameters to reduce function parameter counts
//! and make data flow explicit.

use crate::graph::DirectedGraph;
use crate::scenario::OptimizedSampledBranchingNoises;
use crate::subproblem::{Realization, Subproblem};
use crate::timing::ForwardTiming;
use std::time::Duration;

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

impl<'a> ForwardPassContext<'a> {
    /// Create a new forward pass context.
    pub fn new(
        subproblem_graph: &'a mut DirectedGraph<Subproblem>,
        realization_graph: &'a mut DirectedGraph<Realization>,
        sampled_noises: &'a [&'a OptimizedSampledBranchingNoises],
        graph_bfs_table: &'a [Vec<usize>],
        study_period_ids: &'a [usize],
        timing: &'a ForwardTiming,
    ) -> Self {
        Self {
            subproblem_graph,
            realization_graph,
            sampled_noises,
            graph_bfs_table,
            study_period_ids,
            timing,
        }
    }

    /// Get the number of stages in this trajectory.
    #[inline]
    pub fn num_stages(&self) -> usize {
        self.study_period_ids.len()
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
    pub fn get_realization_mut(
        &mut self,
        id: usize,
    ) -> Option<&mut crate::graph::Node<Realization>> {
        self.realization_graph.get_node_mut(id)
    }

    /// Get past realizations for a stage index.
    pub fn get_past_realizations(
        &self,
        stage_idx: usize,
    ) -> Result<Vec<&Realization>, String> {
        let past_node_ids = self.graph_bfs_table.get(stage_idx).ok_or_else(|| {
            format!("Could not find past node ids for stage index {}", stage_idx)
        })?;

        past_node_ids
            .iter()
            .map(|&past_id| {
                self.realization_graph
                    .get_node(past_id)
                    .map(|node| &node.data)
                    .ok_or_else(|| {
                        format!("Could not find realization for past_node {}", past_id)
                    })
            })
            .collect()
    }

    /// Get noises for a given node ID.
    #[inline]
    pub fn get_noises(&self, id: usize) -> Option<&OptimizedSampledBranchingNoises> {
        self.sampled_noises.get(id).copied()
    }
}

/// Result of a forward pass execution.
#[derive(Debug, Clone)]
pub struct ForwardPassResult {
    /// Total trajectory cost (sum of stage objectives).
    pub trajectory_cost: f64,

    /// Number of solver calls made during this forward pass.
    pub solver_calls: usize,
}

impl ForwardPassResult {
    /// Create a new forward pass result.
    pub fn new(trajectory_cost: f64, solver_calls: usize) -> Self {
        Self {
            trajectory_cost,
            solver_calls,
        }
    }
}

/// Timing data collected during a single trajectory's forward pass.
///
/// This is the internal timing that gets aggregated across parallel trajectories.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrajectoryTiming {
    /// Time spent in model preprocessing for this trajectory.
    pub model_preprocessing: Duration,

    /// Time spent in solver for this trajectory.
    pub solver: Duration,

    /// Time spent in model postprocessing for this trajectory.
    pub model_postprocessing: Duration,

    /// Number of solver calls in this trajectory.
    pub solver_calls: usize,
}

impl TrajectoryTiming {
    /// Add timing from another trajectory.
    pub fn add(&mut self, other: &TrajectoryTiming) {
        self.model_preprocessing += other.model_preprocessing;
        self.solver += other.solver;
        self.model_postprocessing += other.model_postprocessing;
        self.solver_calls += other.solver_calls;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_pass_result_new() {
        let result = ForwardPassResult::new(1000.0, 10);
        assert_eq!(result.trajectory_cost, 1000.0);
        assert_eq!(result.solver_calls, 10);
    }

    #[test]
    fn test_trajectory_timing_add() {
        let mut timing1 = TrajectoryTiming {
            model_preprocessing: Duration::from_millis(100),
            solver: Duration::from_millis(200),
            model_postprocessing: Duration::from_millis(50),
            solver_calls: 5,
        };

        let timing2 = TrajectoryTiming {
            model_preprocessing: Duration::from_millis(50),
            solver: Duration::from_millis(100),
            model_postprocessing: Duration::from_millis(25),
            solver_calls: 3,
        };

        timing1.add(&timing2);

        assert_eq!(timing1.model_preprocessing, Duration::from_millis(150));
        assert_eq!(timing1.solver, Duration::from_millis(300));
        assert_eq!(timing1.model_postprocessing, Duration::from_millis(75));
        assert_eq!(timing1.solver_calls, 8);
    }
}
```

#### 2. Update `src/algorithm/mod.rs`

```rust
//! SDDP Algorithm Phases
//!
//! This module contains the core SDDP algorithm logic, separated by phase:
//!
//! - `context`: Context structs for algorithm phases
//! - `forward_pass`: Forward simulation through the scenario tree
//! - `backward_pass`: Backward cut generation and FCF updates
//! - `cut_computation`: Benders cut calculation
//!
//! # Status
//!
//! 🚧 **In Progress**: Logic being migrated from `src/sddp/mod.rs`
//! in Epic 3: Algorithm Separation.
//!
//! # Structure
//!
//! ```text
//! algorithm/
//! ├── mod.rs
//! ├── context.rs         ✅ Complete
//! ├── forward_pass.rs    ⬜ In Progress
//! ├── backward_pass.rs   ⬜ Not Started
//! └── cut_computation.rs ⬜ Not Started
//! ```

pub mod context;

pub use context::{ForwardPassContext, ForwardPassResult, TrajectoryTiming};

// Future submodules (uncomment as implemented):
// pub mod forward_pass;
// pub mod backward_pass;
// pub mod cut_computation;
```

---

## Acceptance Criteria

- [ ] `src/algorithm/context.rs` created with `ForwardPassContext`
- [ ] `ForwardPassResult` struct defined
- [ ] `TrajectoryTiming` struct defined
- [ ] `src/algorithm/mod.rs` updated to export new types
- [ ] All doc comments complete
- [ ] Unit tests for result and timing types
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] Golden tests pass (no behavioral changes)

### Correctness Verification

- [ ] No changes to existing algorithm code
- [ ] Context struct lifetime annotations are correct
- [ ] Types align with existing `DirectedGraph`, `Subproblem`, `Realization`

---

## Implementation Guide

### Suggested Approach

1. **Create `src/algorithm/context.rs`**:
   - Copy the struct definitions from Specification above
   - Adjust import paths as needed

2. **Find correct import paths**:
   ```bash
   grep -rn "pub struct DirectedGraph\|pub struct Subproblem\|pub struct Realization" src/
   ```

3. **Update `src/algorithm/mod.rs`**:
   - Uncomment/add `pub mod context;`
   - Add `pub use` statements

4. **Verify imports work**:
   ```bash
   cargo build 2>&1 | head -30
   ```

5. **Run tests**:
   ```bash
   cargo test context
   cargo test algorithm
   ```

6. **Run golden tests**:
   ```bash
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Modify/Create

| File | Action |
|------|--------|
| `src/algorithm/context.rs` | CREATE |
| `src/algorithm/mod.rs` | MODIFY |

### Patterns to Follow

- Use references (`&'a`) not owned data in context
- Use `Cell<Duration>` for timing (interior mutability)
- Document preallocation opportunities in struct-level docs
- Keep helper methods simple—complex logic goes in `forward_pass.rs`

### Pitfalls to Avoid

- ⚠️ Don't add owned data to context—use references
- ⚠️ Don't try to make it thread-safe yet—one context per trajectory
- ⚠️ Don't implement `execute()` yet—that's T-020
- ⚠️ Check that `graph::Node` is accessible (may need `pub use`)
- ⚠️ Don't modify any existing files except `algorithm/mod.rs`

---

## Testing Requirements

### Unit Tests

- [ ] Test `ForwardPassResult::new()`
- [ ] Test `TrajectoryTiming::add()`
- [ ] Test `TrajectoryTiming::default()`

### Compile Tests

- [ ] Context struct compiles with correct lifetimes
- [ ] All imports resolve correctly
- [ ] Methods don't cause borrow checker errors

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Struct-level doc comments on `ForwardPassContext`
- [ ] Field-level doc comments on all fields
- [ ] Document preallocation opportunities
- [ ] Document thread safety characteristics
- [ ] Include usage example in struct docs

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward struct creation; main work is getting lifetimes correct

---

## Definition of Done

- [ ] `ForwardPassContext` struct created
- [ ] `ForwardPassResult` struct created
- [ ] `TrajectoryTiming` struct created
- [ ] Module exports set up
- [ ] Unit tests passing
- [ ] Documentation complete
- [ ] `cargo build` succeeds
- [ ] Golden tests pass
- [ ] Code reviewed
