# [T-020] Extract Forward Pass Step Logic

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Forward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-019](./ticket-019-forward-pass-context.md)
> **Blocks**: [T-021](./ticket-021-forward-timing-integration.md)

---

## ⚠️ CRITICAL: Exact Behavioral Preservation

This ticket extracts the forward pass logic from `sddp/mod.rs` to `algorithm/forward_pass.rs`. **The algorithm logic must remain EXACTLY unchanged.** 

Run golden tests after EVERY significant change. If ANY test fails or output differs, **STOP IMMEDIATELY** and investigate.

Do NOT:
- ❌ "Improve" the algorithm
- ❌ "Fix" anything that looks like a bug
- ❌ Change the order of operations
- ❌ Modify numerical computations

---

## Files to Read Before Starting

- `src/sddp/mod.rs:596-680` - Current `forward()` function to extract
- `src/sddp/mod.rs:2519-2540` - Current `step()` function
- `src/algorithm/context.rs` - `ForwardPassContext` from T-019
- `src/subproblem.rs` - `realize_and_solve()` method
- `docs/context-struct-design.md` - Design document from T-018

---

## Context

### Background

The forward pass is the first phase of each SDDP iteration. It simulates the system forward through time, solving LP subproblems at each stage and recording the trajectory. The current implementation is in `TrajectoryEngine::forward()` in `sddp/mod.rs`.

### Current Implementation

```rust
// src/sddp/mod.rs:596-680 (simplified)
pub fn forward(
    &mut self,
    sampled_noises: Vec<&scenario::OptimizedSampledBranchingNoises>,
    graph_bfs_table: &[Vec<usize>],
    study_period_ids: &[usize],
) -> Result<(f64, ForwardPassTimingAccumulator), String> {
    let mut timing = ForwardPassTimingAccumulator::default();

    for (idx, id) in study_period_ids.iter().enumerate() {
        // Model preparation timing
        let prep_start = std::time::Instant::now();

        let subproblem_node = self.subproblem_graph.get_node_mut(*id)...;
        let past_realizations = ...;  // Collect past realizations
        subproblem_node.data.prepare_from_trajectory(&past_realizations)?;

        let realization_node = self.realization_graph.get_node_mut(*id)...;
        let current_stage_noises = sampled_noises.get(*id)...;
        timing.model_preprocessing_time += prep_start.elapsed();

        let step_timing = step(
            &mut subproblem_node.data,
            &mut realization_node.data,
            current_stage_noises,
        )?;
        timing.solver_time += step_timing.solver_time;
        timing.model_postprocessing_time += step_timing.state_update_time;
        timing.solver_calls += 1;
    }

    let trajectory_cost = ...; // Sum stage objectives
    Ok((trajectory_cost, timing))
}
```

### Target Implementation

```rust
// src/algorithm/forward_pass.rs
pub fn execute(ctx: &mut ForwardPassContext) -> Result<ForwardPassResult, String> {
    // Same logic, cleaner interface
}
```

---

## Specification

### Create `src/algorithm/forward_pass.rs`

```rust
//! Forward pass execution for SDDP algorithm.
//!
//! The forward pass simulates the system forward through time, solving
//! LP subproblems at each stage and recording the trajectory.
//!
//! # Algorithm
//!
//! For each stage in the trajectory:
//! 1. Prepare subproblem from past realizations
//! 2. Realize uncertainties (apply noises)
//! 3. Solve the LP
//! 4. Extract solution into realization
//!
//! # Thread Safety
//!
//! A single `ForwardPassContext` is NOT thread-safe. For parallel forward
//! passes, the training loop creates separate contexts for each trajectory.

use crate::algorithm::context::{ForwardPassContext, ForwardPassResult, TrajectoryTiming};
use crate::scenario::OptimizedSampledBranchingNoises;
use crate::subproblem::{Realization, Subproblem};
use std::time::Instant;

/// Execute a forward pass using the provided context.
///
/// This is the main entry point for forward pass execution. It iterates
/// through all stages in `ctx.study_period_ids`, solving each subproblem
/// and recording results.
///
/// # Arguments
///
/// * `ctx` - Mutable reference to forward pass context containing all data
///
/// # Returns
///
/// * `Ok(ForwardPassResult)` - Trajectory cost and timing data
/// * `Err(String)` - Error if any stage fails
///
/// # Errors
///
/// Returns an error if:
/// - A subproblem node is not found
/// - A realization node is not found
/// - Past realizations cannot be retrieved
/// - Noises for a stage are not found
/// - The LP solve fails
///
/// # Example
///
/// ```ignore
/// use powers_rs::algorithm::{forward_pass, ForwardPassContext};
///
/// let mut ctx = ForwardPassContext::new(...);
/// let result = forward_pass::execute(&mut ctx)?;
/// println!("Trajectory cost: {}", result.trajectory_cost);
/// ```
pub fn execute(ctx: &mut ForwardPassContext) -> Result<ForwardPassResult, String> {
    let mut timing = TrajectoryTiming::default();

    for (idx, id) in ctx.study_period_ids.iter().enumerate() {
        execute_stage(ctx, idx, *id, &mut timing)?;
    }

    let trajectory_cost = compute_trajectory_cost(ctx)?;

    Ok(ForwardPassResult::new(trajectory_cost, timing.solver_calls))
}

/// Execute a single stage of the forward pass.
///
/// This function:
/// 1. Retrieves the subproblem and realization nodes
/// 2. Prepares the subproblem from past realizations
/// 3. Applies noises and solves
/// 4. Records timing
fn execute_stage(
    ctx: &mut ForwardPassContext,
    stage_idx: usize,
    node_id: usize,
    timing: &mut TrajectoryTiming,
) -> Result<(), String> {
    // Model preparation timing
    let prep_start = Instant::now();

    // Get subproblem node
    let subproblem_node = ctx
        .subproblem_graph
        .get_node_mut(node_id)
        .ok_or_else(|| format!("Could not find subproblem for node {}", node_id))?;

    // Get past realizations for this stage
    let past_node_ids = ctx
        .graph_bfs_table
        .get(stage_idx)
        .ok_or_else(|| format!("Could not find past node ids for node {}", node_id))?;

    let past_realizations: Vec<&Realization> = past_node_ids
        .iter()
        .map(|&past_id| {
            ctx.realization_graph
                .get_node(past_id)
                .map(|node| &node.data)
                .ok_or_else(|| {
                    format!(
                        "Could not find realization for past_node {} (current_id {})",
                        past_id, node_id
                    )
                })
        })
        .collect::<Result<_, _>>()?;

    // Prepare subproblem from trajectory
    subproblem_node.data.prepare_from_trajectory(&past_realizations)?;

    // Get realization node
    let realization_node = ctx
        .realization_graph
        .get_node_mut(node_id)
        .ok_or_else(|| format!("Could not find realization for node {}", node_id))?;

    // Get noises for this stage
    let current_stage_noises = ctx
        .sampled_noises
        .get(node_id)
        .ok_or_else(|| format!("Could not find noises for node {}", node_id))?;

    timing.model_preprocessing += prep_start.elapsed();

    // Execute step (realize uncertainties and solve)
    let step_timing = step(
        &mut subproblem_node.data,
        &mut realization_node.data,
        current_stage_noises,
    )?;

    timing.solver += step_timing.solver_time;
    timing.model_postprocessing += step_timing.state_update_time;
    timing.solver_calls += 1;

    Ok(())
}

/// Compute the total trajectory cost by summing stage objectives.
fn compute_trajectory_cost(ctx: &ForwardPassContext) -> Result<f64, String> {
    ctx.study_period_ids
        .iter()
        .map(|&id| {
            ctx.realization_graph
                .get_node(id)
                .map(|node| node.data.current_stage_objective)
                .ok_or_else(|| {
                    format!("Could not find realization node {} in iterate", id)
                })
        })
        .sum::<Result<f64, String>>()
}

/// Simple timing structure for step function operations.
#[derive(Debug, Clone, Copy, Default)]
struct StepTiming {
    solver_time: std::time::Duration,
    state_update_time: std::time::Duration,
}

/// Execute a single step: realize uncertainties and solve.
///
/// This function is extracted from `sddp/mod.rs::step()` and performs:
/// 1. Get all innovations from noises
/// 2. Call `realize_and_solve` on the subproblem
/// 3. Return timing information
fn step(
    subproblem: &mut Subproblem,
    realization_container: &mut Realization,
    noises: &OptimizedSampledBranchingNoises,
) -> Result<StepTiming, String> {
    let all_innovations = noises.get_all_innovations();
    let realize_timing = subproblem.realize_and_solve(&all_innovations, realization_container)?;

    Ok(StepTiming {
        solver_time: realize_timing.solver_time,
        state_update_time: realize_timing.state_extraction_time,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require the SDDP infrastructure.
    // Unit tests focus on individual helper functions.

    #[test]
    fn test_step_timing_default() {
        let timing = StepTiming::default();
        assert_eq!(timing.solver_time, std::time::Duration::ZERO);
        assert_eq!(timing.state_update_time, std::time::Duration::ZERO);
    }
}
```

### Update `src/algorithm/mod.rs`

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
//! ├── forward_pass.rs    ✅ Complete
//! ├── backward_pass.rs   ⬜ Not Started
//! └── cut_computation.rs ⬜ Not Started
//! ```

pub mod context;
pub mod forward_pass;

pub use context::{ForwardPassContext, ForwardPassResult, TrajectoryTiming};

// Future submodules (uncomment as implemented):
// pub mod backward_pass;
// pub mod cut_computation;
```

---

## Acceptance Criteria

- [ ] `src/algorithm/forward_pass.rs` created
- [ ] `execute()` function implemented with same logic as original
- [ ] `execute_stage()` helper function extracted
- [ ] `compute_trajectory_cost()` helper function extracted
- [ ] `step()` function copied from `sddp/mod.rs`
- [ ] Module exported from `algorithm/mod.rs`
- [ ] All doc comments complete
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass** (CRITICAL - must be bit-for-bit identical)

### Correctness Verification

- [ ] Logic is EXACTLY the same as original `forward()` function
- [ ] Order of operations unchanged
- [ ] Error messages unchanged
- [ ] Timing accumulation unchanged
- [ ] No numerical changes

---

## Implementation Guide

### Suggested Approach

1. **Create `src/algorithm/forward_pass.rs`**:
   - Start with module-level documentation
   - Add imports

2. **Copy `step()` function** from `sddp/mod.rs:2519-2540`:
   ```bash
   sed -n '2519,2540p' src/sddp/mod.rs
   ```
   - Copy exactly, adjusting imports only

3. **Implement `execute()`**:
   - Follow the structure of original `forward()` exactly
   - Use context fields instead of method parameters
   - Keep timing accumulation the same

4. **Extract helper functions**:
   - `execute_stage()` - one iteration of the loop
   - `compute_trajectory_cost()` - final sum

5. **Update module exports**:
   ```rust
   pub mod forward_pass;
   ```

6. **Verify compilation**:
   ```bash
   cargo build 2>&1 | head -50
   ```

7. **Run tests**:
   ```bash
   cargo test forward_pass
   ```

8. **Run golden tests** (CRITICAL):
   ```bash
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Modify/Create

| File | Action |
|------|--------|
| `src/algorithm/forward_pass.rs` | CREATE |
| `src/algorithm/mod.rs` | MODIFY |

### Critical Patterns to Preserve

1. **Loop structure**: `for (idx, id) in study_period_ids.iter().enumerate()`
2. **Error handling**: Same error messages with same format strings
3. **Timing accumulation**: Same fields, same accumulation points
4. **Past realizations**: Same collection pattern

### Pitfalls to Avoid

- ⚠️ Do NOT change the order of operations
- ⚠️ Do NOT "optimize" the past realizations collection
- ⚠️ Do NOT change error message text
- ⚠️ Do NOT change timing accumulation points
- ⚠️ Keep `step()` as a separate function (for now)
- ⚠️ Do NOT modify `sddp/mod.rs` yet—that's T-022

---

## Testing Requirements

### Unit Tests

- [ ] Test `StepTiming::default()`
- [ ] Test helper functions with mock data (if feasible)

### Integration Tests

The real test is the golden tests—they verify end-to-end correctness.

### Golden Tests (CRITICAL)

- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] Run golden tests after EVERY significant change

---

## Documentation Requirements

- [ ] Module-level documentation explaining forward pass
- [ ] Function-level docs on `execute()`, `execute_stage()`, `step()`
- [ ] Document algorithm steps in order
- [ ] Document error conditions
- [ ] Include usage example

---

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Extraction requires careful attention to preserve exact behavior; borrow checker may require adjustments

---

## Definition of Done

- [ ] `forward_pass.rs` created with complete implementation
- [ ] Logic is EXACTLY the same as original
- [ ] All helper functions extracted
- [ ] Module exports updated
- [ ] Documentation complete
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass**
- [ ] Code reviewed for behavioral equivalence
