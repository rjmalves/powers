# [T-025A] Implement ParallelHandlerCoordinator (No Unsafe)

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2 Revised](./00-sprint-overview.md)
> **Dependencies**: [T-024A](./ticket-024a-revise-processor-trait.md)
> **Blocks**: [T-026](../sprint-02/ticket-026-migrate-handlers.md)

---

## ⚠️ CRITICAL: Structural Change Only

This ticket implements `ParallelHandlerCoordinator` and its `BackwardStageProcessor` implementation. **No changes to algorithm logic** — we are wrapping existing functionality.

---

## Context

### Background

This is the revised implementation of T-025 that avoids unsafe code by passing the FCF graph to methods that need it, rather than storing a raw pointer.

### Current State

`SddpTrainHandler` is managed directly in `train()`:

```rust
let mut train_handlers: Vec<SddpTrainHandler> = (0..num_forward_passes)
    .map(|_| SddpTrainHandler::new(...))
    .collect::<Result<_, _>>()?;
```

Parallel operations use `par_iter_mut()` on this vector directly.

### Target State

`ParallelHandlerCoordinator` encapsulates the handlers and implements `BackwardStageProcessor`:

```rust
pub struct ParallelHandlerCoordinator {
    handlers: Vec<SddpTrainHandler>,
    num_forward_passes: usize,
    // NO fcf_graph field - passed to methods instead
}
```

---

## Files to Read Before Starting

- `src/algorithm/processor.rs` - `BackwardStageProcessor` trait (after T-024A)
- `src/algorithm/context.rs` - `BackwardStageContext` definition
- `src/sddp/mod.rs:319-345` - `SddpTrainHandler` struct
- `src/sddp/mod.rs:636-731` - `compute_cut_data_for_backward_step()`
- `src/sddp/mod.rs:733-754` - `apply_aggregated_cut_result()`
- `src/sddp/mod.rs:868-924` - `eval_first_stage_bound()`
- `src/sddp/mod.rs:1793-2027` - Current backward pass phases (reference)

---

## Specification

### Create `src/algorithm/coordinator.rs`

```rust
//! Parallel handler coordination for SDDP training.
//!
//! This module provides `ParallelHandlerCoordinator` which encapsulates
//! the management of multiple `SddpTrainHandler` instances and implements
//! the `BackwardStageProcessor` trait for backward pass execution.
//!
//! # Design Note: No Unsafe Code
//!
//! The FCF graph is passed to methods that need it rather than stored as
//! a raw pointer. This avoids `unsafe` blocks while maintaining clean APIs.

use crate::algorithm::context::BackwardStageContext;
use crate::algorithm::processor::{
    BackwardStageProcessor, CutComputationTiming, FirstStageTiming,
    Phase1Result, Phase2Result,
};
use crate::cut::BendersCut;
use crate::fcf::{
    AggregatedCutSelectionResult, BatchCutSelectionResult, CutData,
    FutureCostFunction,
};
use crate::graph::DirectedGraph;
use crate::sddp::{BackwardPhase1Timing, SddpTrainHandler};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Coordinates parallel execution across train handlers.
///
/// Encapsulates `Vec<SddpTrainHandler>` and implements `BackwardStageProcessor`.
/// This enables the backward pass loop to be extracted while keeping
/// parallel coordination details hidden.
///
/// # Thread Safety
///
/// The coordinator owns the handlers and uses `par_iter_mut()` for parallel
/// phases. The FCF is passed to `select_cuts_batch` which performs sequential
/// operations under lock.
///
/// # Design: FCF Access
///
/// Instead of storing an FCF reference (which would require unsafe code),
/// the FCF graph is passed to `select_cuts_batch()`. This keeps the coordinator
/// safe and the API explicit about FCF dependencies.
///
/// # Future Extensions
///
/// This is where buffer pools will be added in Epic 5:
/// - `TrajectoryPool` for forward pass buffers
/// - `CutStatePool` for state management
pub struct ParallelHandlerCoordinator {
    /// The train handlers, one per forward pass.
    handlers: Vec<SddpTrainHandler>,

    /// Number of forward passes.
    num_forward_passes: usize,
}

impl ParallelHandlerCoordinator {
    /// Create a new coordinator with the given handlers.
    pub fn new(handlers: Vec<SddpTrainHandler>) -> Self {
        let num_forward_passes = handlers.len();
        Self {
            handlers,
            num_forward_passes,
        }
    }

    /// Get mutable access to handlers (for forward pass execution).
    ///
    /// This is used by `train()` to run forward passes on handlers directly.
    pub fn handlers_mut(&mut self) -> &mut [SddpTrainHandler] {
        &mut self.handlers
    }

    /// Get immutable access to handlers.
    pub fn handlers(&self) -> &[SddpTrainHandler] {
        &self.handlers
    }

    /// Convert timing from handler format to processor format.
    ///
    /// Scales internal timing measurements to wall-clock time.
    fn scale_timing(
        &self,
        phase1_timings: &[BackwardPhase1Timing],
        phase1_wall_time: Duration,
        solver_calls: usize,
    ) -> CutComputationTiming {
        if phase1_timings.is_empty() {
            return CutComputationTiming::default();
        }

        let n = phase1_timings.len() as u32;

        // Compute raw averages from internal measurements
        let raw_model_prep: Duration = phase1_timings
            .iter()
            .map(|t| t.model_preprocessing_time)
            .sum::<Duration>()
            / n;
        let raw_solver: Duration = phase1_timings
            .iter()
            .map(|t| t.solver_time)
            .sum::<Duration>()
            / n;
        let raw_model_post: Duration = phase1_timings
            .iter()
            .map(|t| t.model_postprocessing_time)
            .sum::<Duration>()
            / n;

        // Sum of internal timing estimates
        let internal_total = raw_model_prep + raw_solver + raw_model_post;

        // Scale to wall clock time
        if internal_total > Duration::ZERO {
            let scale = phase1_wall_time.as_secs_f64() / internal_total.as_secs_f64();
            CutComputationTiming {
                model_preprocessing: raw_model_prep.mul_f64(scale),
                solver: raw_solver.mul_f64(scale),
                model_postprocessing: raw_model_post.mul_f64(scale),
                solver_calls,
            }
        } else {
            CutComputationTiming::default()
        }
    }
}

impl BackwardStageProcessor for ParallelHandlerCoordinator {
    fn compute_cuts_parallel(
        &mut self,
        stage_ctx: &BackwardStageContext,
    ) -> Result<Phase1Result, String> {
        let phase1_begin = Instant::now();

        // Phase 1: Parallel cut computation
        let results: Vec<(CutData, BackwardPhase1Timing)> = self
            .handlers
            .par_iter_mut()
            .enumerate()
            .map(|(fp_idx, handler)| {
                handler.compute_cut_data_for_backward_step(
                    stage_ctx.stage_id,
                    stage_ctx.past_node_ids,
                    stage_ctx.node_data_graph,
                    stage_ctx.saa,
                    stage_ctx.iteration,
                    fp_idx,
                )
            })
            .collect::<Result<Vec<_>, String>>()?;

        let phase1_wall_time = phase1_begin.elapsed();

        // Unzip results with pre-allocated capacity
        let mut cut_data = Vec::with_capacity(results.len());
        let mut timings = Vec::with_capacity(results.len());

        for (data, timing) in results {
            cut_data.push(data);
            timings.push(timing);
        }

        // Calculate solver calls: num_forward_passes × num_branching_scenarios
        let num_branchings = stage_ctx.get_branching_count().unwrap_or(1);
        let solver_calls = self.num_forward_passes * num_branchings;

        let timing = self.scale_timing(&timings, phase1_wall_time, solver_calls);

        Ok(Phase1Result { cut_data, timing })
    }

    fn select_cuts_batch(
        &mut self,
        mut cut_data: Vec<CutData>,
        stage_ctx: &BackwardStageContext,
        fcf_graph: &DirectedGraph<Mutex<FutureCostFunction>>,
    ) -> Result<Phase2Result, String> {
        let parent_id = stage_ctx.parent_id.ok_or_else(|| {
            format!(
                "No parent ID for stage {} (stage_idx {})",
                stage_ctx.stage_id, stage_ctx.stage_idx
            )
        })?;

        // Sort for deterministic ordering (CRITICAL for reproducibility)
        cut_data.sort_unstable_by_key(|data| data.forward_pass_idx);

        let phase2_begin = Instant::now();

        // Access FCF and perform batch selection
        let batch_result: BatchCutSelectionResult = {
            let parent_fcf_node = fcf_graph.get_node(parent_id).ok_or_else(|| {
                format!("Could not find FCF for parent node {}", parent_id)
            })?;
            let mut fcf_locked = parent_fcf_node.data.lock().unwrap();
            fcf_locked.add_cuts_batch_from_data(cut_data, stage_ctx.enable_cut_selection)
        };

        let cut_selection_time = phase2_begin.elapsed();

        // Phase 3a: Update FCF state and clone cuts
        let fcf_update_begin = Instant::now();

        let aggregated = AggregatedCutSelectionResult {
            new_cut_ids: batch_result.new_cut_ids.clone(),
            returning_cut_ids: batch_result.returning_cut_ids.clone(),
            removing_cut_ids: batch_result.removing_cut_ids.clone(),
        };

        let (fcf_state_update_time, cut_cloning_time, cuts) = {
            let parent_fcf_node = fcf_graph.get_node(parent_id).ok_or_else(|| {
                format!("Could not find FCF for parent node {}", parent_id)
            })?;
            let mut fcf_locked = parent_fcf_node.data.lock().unwrap();

            // PART 1: Update FCF state (mark cuts inactive)
            let state_begin = Instant::now();
            let mut removed_indices: Vec<usize> = Vec::new();
            for &cut_id in &aggregated.removing_cut_ids {
                if let Some(cut) = fcf_locked.cut_pool.pool.get_mut(cut_id) {
                    cut.set_active(false);
                }
                if let Some(index) = fcf_locked.cut_pool.active_cut_indices.remove(&cut_id) {
                    removed_indices.push(index);
                }
            }

            // Sort removed indices for efficient adjustment
            removed_indices.sort_unstable();

            // Adjust indices for all remaining cuts
            for (_cut_id, index) in fcf_locked.cut_pool.active_cut_indices.iter_mut() {
                let count_below = removed_indices.partition_point(|&removed| removed < *index);
                *index -= count_below;
            }
            let state_time = state_begin.elapsed();

            // PART 2: Clone cuts for parallel application
            // With Arc, this clones the Arc pointer (~16 bytes) not the data (~1KB)
            let clone_begin = Instant::now();
            let cuts: Vec<(usize, Arc<BendersCut>)> = aggregated
                .new_cut_ids
                .iter()
                .chain(aggregated.returning_cut_ids.iter())
                .filter_map(|&cut_id| {
                    fcf_locked
                        .cut_pool
                        .pool
                        .get(cut_id)
                        .map(|cut| (cut_id, Arc::clone(cut)))
                })
                .collect();
            let clone_time = clone_begin.elapsed();

            (state_time, clone_time, cuts)
        }; // FCF lock released

        let _fcf_update_total = fcf_update_begin.elapsed();

        Ok(Phase2Result {
            batch_result,
            aggregated,
            cuts,
            cut_selection_time,
            fcf_update_time: fcf_state_update_time,
            cut_cloning_time,
        })
    }

    fn apply_cuts_parallel(
        &mut self,
        phase2_result: &Phase2Result,
        stage_ctx: &BackwardStageContext,
    ) -> Result<Duration, String> {
        let parent_id = stage_ctx
            .parent_id
            .ok_or_else(|| "No parent ID for cut application".to_string())?;

        let phase3b_begin = Instant::now();

        self.handlers
            .par_iter_mut()
            .map(|handler| {
                handler.apply_aggregated_cut_result(
                    parent_id,
                    &phase2_result.aggregated,
                    &phase2_result.cuts,
                )
            })
            .collect::<Result<(), String>>()?;

        Ok(phase3b_begin.elapsed())
    }

    fn eval_first_stage_bound(
        &mut self,
        stage_ctx: &BackwardStageContext,
    ) -> Result<(f64, FirstStageTiming), String> {
        // Use first handler to evaluate first stage
        let handler = self
            .handlers
            .get_mut(0)
            .ok_or_else(|| "No handlers available".to_string())?;

        let (lb, timing) = handler.eval_first_stage_bound(
            stage_ctx.stage_id,
            stage_ctx.past_node_ids,
            stage_ctx.node_data_graph,
            stage_ctx.saa,
        )?;

        Ok((
            lb,
            FirstStageTiming {
                solver: timing.solver_time,
                state_extraction: timing.state_extraction_time,
            },
        ))
    }

    fn num_forward_passes(&self) -> usize {
        self.num_forward_passes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_new_empty() {
        let coordinator = ParallelHandlerCoordinator::new(Vec::new());
        assert_eq!(coordinator.num_forward_passes(), 0);
        assert!(coordinator.handlers().is_empty());
    }

    #[test]
    fn test_scale_timing_empty() {
        let coordinator = ParallelHandlerCoordinator::new(Vec::new());
        let timing = coordinator.scale_timing(&[], Duration::from_secs(1), 0);
        assert_eq!(timing.solver_calls, 0);
        assert_eq!(timing.solver, Duration::ZERO);
    }

    #[test]
    fn test_scale_timing_scales_correctly() {
        let coordinator = ParallelHandlerCoordinator::new(Vec::new());

        // Create timing with 1:2:1 ratio (prep:solver:post)
        let timings = vec![
            BackwardPhase1Timing {
                model_preprocessing_time: Duration::from_millis(100),
                solver_time: Duration::from_millis(200),
                model_postprocessing_time: Duration::from_millis(100),
            },
            BackwardPhase1Timing {
                model_preprocessing_time: Duration::from_millis(100),
                solver_time: Duration::from_millis(200),
                model_postprocessing_time: Duration::from_millis(100),
            },
        ];

        // Wall time is 2x internal (simulating parallel execution)
        let wall_time = Duration::from_millis(800);
        let result = coordinator.scale_timing(&timings, wall_time, 10);

        // Ratios should be preserved, scaled to wall time
        // Internal avg: 100+200+100 = 400ms, wall = 800ms, scale = 2.0
        assert_eq!(result.model_preprocessing, Duration::from_millis(200));
        assert_eq!(result.solver, Duration::from_millis(400));
        assert_eq!(result.model_postprocessing, Duration::from_millis(200));
        assert_eq!(result.solver_calls, 10);
    }
}
```

---

## Update `src/algorithm/mod.rs`

Add the coordinator module and export:

```rust
pub mod coordinator;

pub use coordinator::ParallelHandlerCoordinator;
```

---

## Acceptance Criteria

- [ ] `src/algorithm/coordinator.rs` created
- [ ] `ParallelHandlerCoordinator` struct implemented
- [ ] `BackwardStageProcessor` trait implemented for coordinator
- [ ] **No unsafe code** in the implementation
- [ ] All phases match existing behavior in `sddp/mod.rs`
- [ ] Module exported in `src/algorithm/mod.rs`
- [ ] `cargo build -j1` succeeds
- [ ] `cargo clippy -j1 -- -D warnings` passes
- [ ] Unit tests pass

---

## Implementation Guide

### Step 1: Create coordinator.rs

Create `src/algorithm/coordinator.rs` with the implementation above.

### Step 2: Update mod.rs

Add to `src/algorithm/mod.rs`:

```rust
pub mod coordinator;

pub use coordinator::ParallelHandlerCoordinator;
```

### Step 3: Verify compilation

```bash
cargo build -j1 2>&1 | head -50
cargo clippy -j1 -- -D warnings 2>&1 | head -50
cargo test coordinator 2>&1 | head -30
```

---

## Key Files to Create/Modify

| File | Action |
|------|--------|
| `src/algorithm/coordinator.rs` | CREATE |
| `src/algorithm/mod.rs` | UPDATE exports |

---

## Pitfalls to Avoid

- ⚠️ Keep timing scaling logic EXACTLY as in original code (lines 1833-1881)
- ⚠️ Maintain deterministic cut ordering (sort by `forward_pass_idx`)
- ⚠️ Don't change `SddpTrainHandler` internals—just call its methods
- ⚠️ The FCF pool access pattern must match exactly (Phase 3a)

---

## Testing Requirements

### Unit Tests

- [ ] `test_coordinator_new_empty` - Empty coordinator works
- [ ] `test_scale_timing_empty` - Empty timing returns default
- [ ] `test_scale_timing_scales_correctly` - Timing ratios preserved

### Integration Tests

Integration with actual handlers will be tested in T-026.

---

## Effort Estimate

**Points**: 5  
**Confidence**: High  
**Rationale**: Clear implementation from reference code; no unsafe complexities

---

## Definition of Done

- [ ] Coordinator implemented with all trait methods
- [ ] No unsafe code
- [ ] Timing calculations match original
- [ ] Parallel phases use `par_iter_mut()`
- [ ] Module exported
- [ ] Unit tests pass
- [ ] `cargo build -j1` succeeds
- [ ] `cargo clippy -j1 -- -D warnings` passes
- [ ] Code reviewed
