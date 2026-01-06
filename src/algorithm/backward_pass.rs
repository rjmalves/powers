//! Backward pass execution for SDDP algorithm.
//!
//! The backward pass iterates through stages in reverse order, computing
//! Benders cuts at each stage and updating the future cost function.
//!
//! # Architecture
//!
//! The backward pass has a 3-phase structure per stage:
//!
//! 1. **Phase 1**: Parallel cut computation (via `BackwardStageProcessor`)
//! 2. **Phase 2**: Sequential batch cut selection (deterministic ordering)
//! 3. **Phase 3**: Parallel cut application to handler models
//!
//! # FCF Graph Access (Epic 4 - T-046)
//!
//! The FCF graph is passed as `&mut DirectedGraph<FutureCostFunction>` instead
//! of using `Arc<Mutex<>>` wrapping. This is safe because:
//! - Phase 1 doesn't access FCF (parallel cut computation)
//! - Phase 2 accesses FCF sequentially (single-threaded cut selection)
//! - Phase 3 uses cloned cuts (no FCF access)
//!
//! # Timing Separation
//!
//! Timing is passed as a separate parameter (not inside context) to enable
//! `TimingGuard` usage without borrow conflicts. See Epic 3 Sprint 1 (T-021)
//! for the architectural rationale.

use crate::algorithm::context::{BackwardPassContext, BackwardPassResult};
use crate::algorithm::processor::BackwardStageProcessor;
use crate::fcf::FutureCostFunction;
use crate::graph::DirectedGraph;
use crate::timing::NewBackwardTiming;

/// Execute the backward pass using the provided processor.
///
/// Iterates through stages in reverse order, computing Benders cuts at each
/// stage and updating the future cost function.
///
/// # Arguments
///
/// * `processor` - Implementation of `BackwardStageProcessor` (typically `ParallelHandlerCoordinator`)
/// * `ctx` - Backward pass context (no timing inside - passed separately)
/// * `timing` - Timing accumulator (uses `Cell<Duration>` for `TimingGuard` compatibility)
/// * `fcf_graph` - FCF graph for cut operations (passed explicitly to avoid unsafe storage)
///
/// # Returns
///
/// * `Ok(BackwardPassResult)` - Lower bound and cut statistics
/// * `Err(String)` - If any stage fails
///
/// # Example
///
/// ```ignore
/// use powers_rs::algorithm::backward_pass;
/// use powers_rs::algorithm::{BackwardPassContext, ParallelHandlerCoordinator};
/// use powers_rs::timing::NewBackwardTiming;
///
/// let timing = NewBackwardTiming::new();
/// let result = backward_pass::execute(
///     &mut coordinator,
///     &backward_ctx,
///     &timing,
///     &mut fcf_graph,
/// )?;
///
/// println!("Lower bound: {}", result.lower_bound);
/// println!("Cuts added: {}", result.cuts_added);
/// ```
pub fn execute<P: BackwardStageProcessor>(
    processor: &mut P,
    ctx: &BackwardPassContext,
    timing: &NewBackwardTiming,
    fcf_graph: &mut DirectedGraph<FutureCostFunction>,
) -> Result<BackwardPassResult, String> {
    let mut result = BackwardPassResult::new(0.0, 0, 0, 0, 0);

    for stage_idx in ctx.backward_stage_indices() {
        let stage_ctx = ctx
            .stage_context(stage_idx)
            .ok_or_else(|| format!("Invalid stage index {}", stage_idx))?;

        if stage_ctx.is_first_stage() {
            // First stage: evaluate bound only (no cut generation)
            execute_first_stage(processor, &stage_ctx, &mut result, timing)?;
        } else {
            // Other stages: compute and apply cuts
            execute_stage(
                processor,
                &stage_ctx,
                &mut result,
                timing,
                fcf_graph,
            )?;
        }
    }

    // Set final solver calls count in result
    result.solver_calls = timing.solver_calls.get();

    Ok(result)
}

/// Execute first stage evaluation (no cut generation).
fn execute_first_stage<P: BackwardStageProcessor>(
    processor: &mut P,
    stage_ctx: &crate::algorithm::context::BackwardStageContext,
    result: &mut BackwardPassResult,
    timing: &NewBackwardTiming,
) -> Result<(), String> {
    let (first_stage_result, first_timing) =
        processor.eval_first_stage_bound(stage_ctx)?;
    result.lower_bound = first_stage_result.bound;
    result.first_stage_branching_costs = first_stage_result.branching_costs;

    // Accumulate first stage timing
    timing
        .phase1
        .solver
        .set(timing.phase1.solver.get() + first_timing.solver);
    timing.phase1.model_postprocessing.set(
        timing.phase1.model_postprocessing.get()
            + first_timing.state_extraction,
    );

    // Count solver calls for first stage
    let num_branchings = stage_ctx.get_branching_count().unwrap_or(1);
    timing.solver_calls.set(
        timing.solver_calls.get()
            + num_branchings * processor.num_forward_passes(),
    );

    Ok(())
}

/// Execute a single stage of the backward pass (non-first stage).
///
/// Performs the 3-phase architecture using zero-allocation path:
/// 1. Phase 1: Parallel cut computation into staging buffers, then sequential pool update
/// 2. Phase 2: Sequential batch cut finalization and selection
/// 3. Phase 3: FCF state update + parallel handler application
fn execute_stage<P: BackwardStageProcessor>(
    processor: &mut P,
    stage_ctx: &crate::algorithm::context::BackwardStageContext,
    result: &mut BackwardPassResult,
    timing: &NewBackwardTiming,
    fcf_graph: &mut DirectedGraph<FutureCostFunction>,
) -> Result<(), String> {
    // Phase 1: Parallel cut computation into staging buffers, then sequential pool update
    let phase1 =
        processor.compute_cuts_parallel_into_slots(stage_ctx, fcf_graph)?;

    // Accumulate Phase 1 timing
    timing.phase1.model_preprocessing.set(
        timing.phase1.model_preprocessing.get()
            + phase1.timing.model_preprocessing,
    );
    timing
        .phase1
        .solver
        .set(timing.phase1.solver.get() + phase1.timing.solver);
    timing.phase1.model_postprocessing.set(
        timing.phase1.model_postprocessing.get()
            + phase1.timing.model_postprocessing,
    );
    timing
        .solver_calls
        .set(timing.solver_calls.get() + phase1.timing.solver_calls);

    // Phase 2: Sequential batch cut finalization and selection
    let phase2 =
        processor.select_cuts_from_slots(phase1.slots, stage_ctx, fcf_graph)?;

    // Accumulate Phase 2 timing
    timing
        .phase2
        .cut_selection
        .set(timing.phase2.cut_selection.get() + phase2.cut_selection_time);
    // Phase 2 fcf_update_time goes to phase3.problem_update (combined with handler_application)
    let fcf_update = phase2.fcf_update_time;

    // Update result counts
    result.cuts_added += phase2.aggregated.new_cut_ids.len();
    result.cuts_removed += phase2.aggregated.removing_cut_ids.len();
    result.cuts_returned += phase2.aggregated.returning_cut_ids.len();

    // Phase 3: Parallel cut application
    // Get cut pool slice for zero-allocation read access
    let parent_id = stage_ctx
        .parent_id
        .ok_or_else(|| "No parent ID for cut application".to_string())?;
    let cut_pool = &fcf_graph
        .get_node(parent_id)
        .ok_or_else(|| format!("FCF node {} not found", parent_id))?
        .data
        .cut_pool
        .pool;
    let handler_time =
        processor.apply_cuts_parallel(&phase2, stage_ctx, cut_pool)?;
    // Combine fcf_update and handler_application into phase3.problem_update
    timing
        .phase3
        .problem_update
        .set(timing.phase3.problem_update.get() + fcf_update + handler_time);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_pass_result() {
        let result = BackwardPassResult::new(1000.0, 10, 2, 1, 50);

        assert!((result.lower_bound - 1000.0).abs() < f64::EPSILON);
        assert_eq!(result.cuts_added, 10);
        assert_eq!(result.cuts_removed, 2);
        assert_eq!(result.cuts_returned, 1);
        assert_eq!(result.solver_calls, 50);
    }
}
