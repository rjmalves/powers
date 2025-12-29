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
//! // Backward pass uses trait methods
//! let phase1 = coordinator.compute_cuts_parallel(&stage_ctx)?;
//! let phase2 = coordinator.select_cuts_batch(phase1.cut_data, &stage_ctx, &fcf_graph)?;
//! coordinator.apply_cuts_parallel(&phase2, &stage_ctx)?;
//! ```

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
            let scale = phase1_wall_time.as_secs_f64() / internal_total.as_secs_f64();
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
        let result = coordinator.scale_timing(&timings, Duration::from_secs(1), 5);

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
        let result = coordinator.scale_timing(&timings, Duration::from_secs(1), 100);

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
        let result = coordinator.scale_timing(&timings, Duration::from_millis(1200), 50);

        assert_eq!(result.model_preprocessing, Duration::from_millis(300));
        assert_eq!(result.solver, Duration::from_millis(600));
        assert_eq!(result.model_postprocessing, Duration::from_millis(300));
        assert_eq!(result.solver_calls, 50);
    }
}
