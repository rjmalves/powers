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
//!
//! # Example (conceptual)
//!
//! ```ignore
//! use powers_rs::algorithm::{ParallelHandlerCoordinator, BackwardStageProcessor};
//!
//! let coordinator = ParallelHandlerCoordinator::new(handlers);
//!
//! // Forward pass uses handlers directly
//! for handler in coordinator.handlers_mut() {
//!     // ... forward pass operations ...
//! }
//!
//! // Backward pass uses trait methods (zero-allocation path)
//! let phase1 = coordinator.compute_cuts_parallel_into_slots(&stage_ctx, &mut fcf_graph)?;
//! let phase2 = coordinator.select_cuts_from_slots(phase1.slots, &stage_ctx, &mut fcf_graph)?;
//! coordinator.apply_cuts_parallel(&phase2, &stage_ctx, &cut_pool)?;
//! ```

use crate::algorithm::context::BackwardStageContext;
use crate::algorithm::processor::{
    BackwardStageProcessor, CutComputationTiming, FirstStageTiming,
    Phase1SlotResult, Phase2Result,
};
use crate::cut::BendersCut;
use crate::fcf::{AggregatedCutSelectionResult, FutureCostFunction};
use crate::graph::DirectedGraph;
use crate::sddp::{BackwardPhase1Timing, SddpTrainHandler};
use rayon::prelude::*;
use std::time::{Duration, Instant};

/// Preallocated buffers for coordinator result collection.
///
/// Avoids per-stage allocation overhead by reusing buffers across iterations.
struct CoordinatorBuffers {
    /// Slot indices from parallel cut computation.
    slots: Vec<usize>,
    /// Timing data from parallel phases.
    phase1_timings: Vec<BackwardPhase1Timing>,
}

impl CoordinatorBuffers {
    fn new(num_forward_passes: usize) -> Self {
        Self {
            slots: Vec::with_capacity(num_forward_passes),
            phase1_timings: Vec::with_capacity(num_forward_passes),
        }
    }

    fn reset(&mut self) {
        self.slots.clear();
        self.phase1_timings.clear();
    }
}

/// Coordinates parallel execution across train handlers.
///
/// Encapsulates `Vec<SddpTrainHandler>` and implements `BackwardStageProcessor`.
/// This enables the backward pass loop to be extracted while keeping
/// parallel coordination details hidden.
///
/// # Thread Safety
///
/// The coordinator owns the handlers and uses `par_iter_mut()` for parallel
/// phases. The FCF graph is passed to methods that need it.
///
/// # Design: FCF Access
///
/// Instead of storing an FCF reference (which would require unsafe code),
/// the FCF graph is passed to methods that need pool access. This keeps the
/// coordinator safe and the API explicit about FCF dependencies.
pub struct ParallelHandlerCoordinator {
    /// The train handlers, one per forward pass.
    handlers: Vec<SddpTrainHandler>,

    /// Number of forward passes.
    num_forward_passes: usize,

    /// Preallocated buffers for result collection.
    buffers: CoordinatorBuffers,
}

impl ParallelHandlerCoordinator {
    /// Create a new coordinator with the given handlers.
    pub fn new(handlers: Vec<SddpTrainHandler>) -> Self {
        let num_forward_passes = handlers.len();
        Self {
            handlers,
            buffers: CoordinatorBuffers::new(num_forward_passes),
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
            return CutComputationTiming {
                solver_calls,
                ..Default::default()
            };
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
            let scale =
                phase1_wall_time.as_secs_f64() / internal_total.as_secs_f64();
            CutComputationTiming {
                model_preprocessing: raw_model_prep.mul_f64(scale),
                solver: raw_solver.mul_f64(scale),
                model_postprocessing: raw_model_post.mul_f64(scale),
                solver_calls,
            }
        } else {
            CutComputationTiming {
                solver_calls,
                ..Default::default()
            }
        }
    }

    /// Compute cuts with zero allocation into preallocated FCF slots.
    ///
    /// # Zero Allocation
    ///
    /// Writes cut and state coefficients directly to preallocated FCF pool slots.
    ///
    /// # Architecture
    ///
    /// This combines Phase 1 and partial Phase 2:
    /// 1. Parallel cut computation → writes to preallocated slots
    /// 2. Returns slot IDs for domination evaluation
    ///
    /// Caller must then call `fcf.finalize_cuts_batch(&slots, enable_cut_selection)`
    /// to complete Phase 2.
    ///
    /// # Thread Safety
    ///
    /// Each handler writes to a different slot (based on forward_pass_idx),
    /// so parallel execution is safe.
    pub fn compute_cuts_into_slots(
        &mut self,
        stage_ctx: &BackwardStageContext,
        cut_pool: &mut crate::cut::BendersCutPool,
        state_pool: &mut crate::state::VisitedStatePool,
    ) -> Result<(Vec<usize>, CutComputationTiming), String> {
        let phase1_begin = Instant::now();

        // Reset preallocated buffers for this stage
        self.buffers.reset();

        // Since we need mutable access to cut_pool and state_pool from multiple threads,
        // we must use sequential execution for now. A future optimization could use
        // per-handler pools and merge at the end.
        for (fp_idx, handler) in self.handlers.iter_mut().enumerate() {
            let (slot, timing) = handler
                .compute_cut_into_slot_for_backward_step(
                    stage_ctx.stage_id,
                    stage_ctx.past_node_ids,
                    stage_ctx.node_data_graph,
                    stage_ctx.saa,
                    cut_pool,
                    state_pool,
                    stage_ctx.iteration,
                    fp_idx,
                )?;
            self.buffers.slots.push(slot);
            self.buffers.phase1_timings.push(timing);
        }

        let phase1_wall_time = phase1_begin.elapsed();

        // Calculate solver calls
        let num_branchings = stage_ctx.get_branching_count().unwrap_or(1);
        let solver_calls = self.num_forward_passes * num_branchings;

        let timing = self.scale_timing(
            &self.buffers.phase1_timings,
            phase1_wall_time,
            solver_calls,
        );

        // Sort slots for deterministic ordering (clone to return owned vec)
        let mut slots = self.buffers.slots.clone();
        slots.sort_unstable();

        Ok((slots, timing))
    }
}

impl BackwardStageProcessor for ParallelHandlerCoordinator {
    fn apply_cuts_parallel(
        &mut self,
        phase2_result: &Phase2Result,
        stage_ctx: &BackwardStageContext,
        cut_pool: &[BendersCut],
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
                    &phase2_result.cut_ids,
                    cut_pool,
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

    fn compute_cuts_into_slots(
        &mut self,
        stage_ctx: &BackwardStageContext,
        fcf_graph: &mut DirectedGraph<FutureCostFunction>,
    ) -> Result<Phase1SlotResult, String> {
        let parent_id = stage_ctx.parent_id.ok_or_else(|| {
            format!(
                "No parent ID for stage {} (stage_idx {})",
                stage_ctx.stage_id, stage_ctx.stage_idx
            )
        })?;

        let parent_fcf_node =
            fcf_graph.get_node_mut(parent_id).ok_or_else(|| {
                format!("Could not find FCF for parent node {}", parent_id)
            })?;
        let fcf = &mut parent_fcf_node.data;

        let phase1_begin = Instant::now();

        // Sequential execution required due to mutable pool access.
        // Each handler writes to a different slot.
        let mut slots = Vec::with_capacity(self.num_forward_passes);
        let mut timings = Vec::with_capacity(self.num_forward_passes);

        for (fp_idx, handler) in self.handlers.iter_mut().enumerate() {
            let (slot, timing) = handler
                .compute_cut_into_slot_for_backward_step(
                    stage_ctx.stage_id,
                    stage_ctx.past_node_ids,
                    stage_ctx.node_data_graph,
                    stage_ctx.saa,
                    &mut fcf.cut_pool,
                    &mut fcf.state_pool,
                    stage_ctx.iteration,
                    fp_idx,
                )?;
            slots.push(slot);
            timings.push(timing);
        }

        let phase1_wall_time = phase1_begin.elapsed();

        let num_branchings = stage_ctx.get_branching_count().unwrap_or(1);
        let solver_calls = self.num_forward_passes * num_branchings;

        let timing =
            self.scale_timing(&timings, phase1_wall_time, solver_calls);

        // Sort for deterministic ordering
        slots.sort_unstable();

        Ok(Phase1SlotResult { slots, timing })
    }

    fn select_cuts_from_slots(
        &mut self,
        slots: Vec<usize>,
        stage_ctx: &BackwardStageContext,
        fcf_graph: &mut DirectedGraph<FutureCostFunction>,
    ) -> Result<Phase2Result, String> {
        let parent_id = stage_ctx.parent_id.ok_or_else(|| {
            format!(
                "No parent ID for stage {} (stage_idx {})",
                stage_ctx.stage_id, stage_ctx.stage_idx
            )
        })?;

        let parent_fcf_node =
            fcf_graph.get_node_mut(parent_id).ok_or_else(|| {
                format!("Could not find FCF for parent node {}", parent_id)
            })?;
        let fcf = &mut parent_fcf_node.data;

        let phase2_begin = Instant::now();

        // Finalize cuts at slots - this runs domination evaluation
        let batch_result =
            fcf.finalize_cuts_batch(&slots, stage_ctx.enable_cut_selection);

        let cut_selection_time = phase2_begin.elapsed();

        let fcf_update_begin = Instant::now();

        let aggregated = AggregatedCutSelectionResult {
            new_cut_ids: batch_result.new_cut_ids.clone(),
            returning_cut_ids: batch_result.returning_cut_ids.clone(),
            removing_cut_ids: batch_result.removing_cut_ids.clone(),
        };

        // Update FCF state (mark cuts inactive)
        // With preallocation, cuts are never removed from model - just marked inactive
        let state_begin = Instant::now();
        for &cut_id in &aggregated.removing_cut_ids {
            if let Some(cut) = fcf.cut_pool.pool.get_mut(cut_id) {
                cut.set_active(false);
            }
        }
        let fcf_state_update_time = state_begin.elapsed();

        // Collect cut IDs for Phase 3 (zero allocation - just indices)
        let cut_ids: Vec<usize> = aggregated
            .new_cut_ids
            .iter()
            .chain(aggregated.returning_cut_ids.iter())
            .copied()
            .collect();

        let _fcf_update_total = fcf_update_begin.elapsed();

        Ok(Phase2Result {
            batch_result,
            aggregated,
            cut_ids,
            cut_selection_time,
            fcf_update_time: fcf_state_update_time,
        })
    }

    fn compute_cuts_parallel_into_slots(
        &mut self,
        stage_ctx: &BackwardStageContext,
        fcf_graph: &mut DirectedGraph<FutureCostFunction>,
    ) -> Result<Phase1SlotResult, String> {
        let parent_id = stage_ctx.parent_id.ok_or_else(|| {
            format!(
                "No parent ID for stage {} (stage_idx {})",
                stage_ctx.stage_id, stage_ctx.stage_idx
            )
        })?;

        let phase1_begin = Instant::now();

        // Phase 1a: Parallel compute into staging buffers
        // Each handler writes to its own staging buffer - no contention
        let timings: Vec<crate::sddp::BackwardPhase1Timing> = self
            .handlers
            .par_iter_mut()
            .enumerate()
            .map(|(fp_idx, handler)| {
                handler.compute_cut_into_staging(
                    stage_ctx.stage_id,
                    stage_ctx.past_node_ids,
                    stage_ctx.node_data_graph,
                    stage_ctx.saa,
                    stage_ctx.iteration,
                    fp_idx,
                )
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Phase 1b: Sequential copy to pools
        // Deterministic order (0, 1, 2, ...) for reproducibility
        let parent_fcf_node =
            fcf_graph.get_node_mut(parent_id).ok_or_else(|| {
                format!("Could not find FCF for parent node {}", parent_id)
            })?;
        let fcf = &mut parent_fcf_node.data;

        let mut slots = Vec::with_capacity(self.num_forward_passes);
        for handler in self.handlers.iter() {
            let slot = fcf.cut_pool.update_from_staging(
                handler.staging_buffer(),
                &mut fcf.state_pool,
            );
            slots.push(slot);
        }

        let phase1_wall_time = phase1_begin.elapsed();

        // Calculate solver calls
        let num_branchings = stage_ctx.get_branching_count().unwrap_or(1);
        let solver_calls = self.num_forward_passes * num_branchings;

        let timing =
            self.scale_timing(&timings, phase1_wall_time, solver_calls);

        // Sort for deterministic ordering (already should be in order by construction)
        slots.sort_unstable();

        Ok(Phase1SlotResult { slots, timing })
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
    fn test_coordinator_handlers_mut() {
        let mut coordinator = ParallelHandlerCoordinator::new(Vec::new());
        // Verify handlers_mut returns an empty mutable slice for empty coordinator
        let handlers = coordinator.handlers_mut();
        assert!(handlers.is_empty());
        // Verify we can iterate (even though empty)
        assert_eq!(handlers.iter_mut().count(), 0);
    }

    #[test]
    fn test_scale_timing_empty() {
        let coordinator = ParallelHandlerCoordinator::new(Vec::new());
        let timing = coordinator.scale_timing(&[], Duration::from_secs(1), 5);
        assert_eq!(timing.solver_calls, 5);
        assert_eq!(timing.solver, Duration::ZERO);
    }

    #[test]
    fn test_scale_timing_zero_internal() {
        // When internal timing is zero, should return default with solver_calls preserved
        let coordinator = ParallelHandlerCoordinator::new(Vec::new());
        let timings = vec![BackwardPhase1Timing::default()];
        let result =
            coordinator.scale_timing(&timings, Duration::from_secs(1), 5);

        assert_eq!(result.model_preprocessing, Duration::ZERO);
        assert_eq!(result.solver, Duration::ZERO);
        assert_eq!(result.model_postprocessing, Duration::ZERO);
        assert_eq!(result.solver_calls, 5);
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

    #[test]
    fn test_scale_timing_preserves_ratios() {
        let coordinator = ParallelHandlerCoordinator::new(Vec::new());

        // 10:60:30 ratio (prep:solver:post)
        let timings = vec![BackwardPhase1Timing {
            model_preprocessing_time: Duration::from_millis(10),
            solver_time: Duration::from_millis(60),
            model_postprocessing_time: Duration::from_millis(30),
        }];

        // Scale to 1 second wall time
        let result =
            coordinator.scale_timing(&timings, Duration::from_secs(1), 100);

        // Verify ratios preserved (10:60:30 = 100:600:300 ms)
        assert_eq!(result.model_preprocessing, Duration::from_millis(100));
        assert_eq!(result.solver, Duration::from_millis(600));
        assert_eq!(result.model_postprocessing, Duration::from_millis(300));
        assert_eq!(result.solver_calls, 100);
    }

    #[test]
    fn test_scale_timing_multiple_handlers_averages() {
        let coordinator = ParallelHandlerCoordinator::new(Vec::new());

        // Two handlers with different timings
        let timings = vec![
            BackwardPhase1Timing {
                model_preprocessing_time: Duration::from_millis(100),
                solver_time: Duration::from_millis(200),
                model_postprocessing_time: Duration::from_millis(100),
            },
            BackwardPhase1Timing {
                model_preprocessing_time: Duration::from_millis(200),
                solver_time: Duration::from_millis(400),
                model_postprocessing_time: Duration::from_millis(200),
            },
        ];

        // Averages: prep=150, solver=300, post=150 (total=600)
        // Wall time 1200ms = 2x scaling
        let result =
            coordinator.scale_timing(&timings, Duration::from_millis(1200), 50);

        assert_eq!(result.model_preprocessing, Duration::from_millis(300));
        assert_eq!(result.solver, Duration::from_millis(600));
        assert_eq!(result.model_postprocessing, Duration::from_millis(300));
        assert_eq!(result.solver_calls, 50);
    }
}
