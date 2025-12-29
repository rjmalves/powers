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
//!
//! # Timing Architecture
//!
//! Timing is passed separately from the context to avoid borrow checker
//! conflicts with `TimingGuard`. The `TrajectoryTiming` struct uses
//! `Cell<Duration>` for interior mutability, allowing `TimingGuard` to
//! accumulate time while the context is mutably borrowed for graph access.
//!
//! Key timing principles:
//! - Precise values are preserved (never redistributed)
//! - Parallel overhead is computed separately by the training loop
//! - Timing is feature-gated via the `timing` feature flag

use crate::algorithm::context::{
    ForwardPassContext, ForwardPassResult, TrajectoryTiming,
};
use crate::scenario::OptimizedSampledBranchingNoises;
use crate::subproblem::{Realization, Subproblem};
use crate::timing::TimingGuard;
use std::time::Duration;

/// Execute a forward pass using the provided context.
///
/// This is the main entry point for forward pass execution. It iterates
/// through all stages in `ctx.study_period_ids`, solving each subproblem
/// and recording results.
///
/// # Arguments
///
/// * `ctx` - Mutable reference to forward pass context containing all data
/// * `timing` - Timing struct for accumulating performance metrics
///
/// # Returns
///
/// * `Ok(ForwardPassResult)` - Trajectory cost and solver call count
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
/// use powers_rs::algorithm::{forward_pass, ForwardPassContext, TrajectoryTiming};
///
/// let timing = TrajectoryTiming::default();
/// let mut ctx = ForwardPassContext::new(...);
/// let result = forward_pass::execute(&mut ctx, &timing)?;
/// println!("Trajectory cost: {}", result.trajectory_cost);
/// ```
pub fn execute(
    ctx: &mut ForwardPassContext,
    timing: &TrajectoryTiming,
) -> Result<ForwardPassResult, String> {
    for (idx, &id) in ctx.study_period_ids.iter().enumerate() {
        execute_stage(ctx, idx, id, timing)?;
    }

    // Cost calculation timing
    let trajectory_cost = {
        let _guard = TimingGuard::new(&timing.model_postprocessing);
        compute_trajectory_cost(ctx)?
    };

    Ok(ForwardPassResult::new(
        trajectory_cost,
        timing.get_solver_calls(),
    ))
}

/// Execute a single stage of the forward pass.
///
/// This function:
/// 1. Retrieves the subproblem and realization nodes
/// 2. Prepares the subproblem from past realizations
/// 3. Applies noises and solves
/// 4. Records timing via TimingGuard
fn execute_stage(
    ctx: &mut ForwardPassContext,
    stage_idx: usize,
    node_id: usize,
    timing: &TrajectoryTiming,
) -> Result<(), String> {
    // Phase 1: Model preparation with timing guard
    {
        let _guard = TimingGuard::new(&timing.model_preprocessing);

        // Get subproblem node
        let subproblem_node =
            ctx.subproblem_graph.get_node_mut(node_id).ok_or_else(|| {
                format!("Could not find subproblem for node {}", node_id)
            })?;

        // Get past realizations for this stage
        let past_node_ids =
            ctx.graph_bfs_table.get(stage_idx).ok_or_else(|| {
                format!("Could not find past node ids for node {}", node_id)
            })?;

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
        subproblem_node
            .data
            .prepare_from_trajectory(&past_realizations)?;
    }

    // Phase 2: Get nodes for step execution (outside timing guard)
    let realization_node =
        ctx.realization_graph.get_node_mut(node_id).ok_or_else(|| {
            format!("Could not find realization for node {}", node_id)
        })?;

    let current_stage_noises = ctx
        .sampled_noises
        .get(node_id)
        .ok_or_else(|| format!("Could not find noises for node {}", node_id))?;

    let subproblem_node =
        ctx.subproblem_graph.get_node_mut(node_id).ok_or_else(|| {
            format!("Could not find subproblem for node {}", node_id)
        })?;

    // Phase 3: Execute step (internal timing returned from realize_and_solve)
    let step_timing = step(
        &mut subproblem_node.data,
        &mut realization_node.data,
        current_stage_noises,
    )?;

    // Phase 4: Record timing from step internals
    // Note: solver_time comes from realize_and_solve internally, not measured externally
    timing.add_solver_time(step_timing.solver_time);
    timing.add_model_postprocessing(step_timing.state_update_time);
    timing.increment_solver_calls();

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
    solver_time: Duration,
    state_update_time: Duration,
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
    let realize_timing = subproblem
        .realize_and_solve(&all_innovations, realization_container)?;

    Ok(StepTiming {
        solver_time: realize_timing.solver_time,
        state_update_time: realize_timing.state_extraction_time,
    })
}

/// Aggregate timing from multiple trajectories into ForwardTiming.
///
/// CRITICAL: This function preserves precise values and does NOT redistribute.
/// Parallel overhead is computed separately by the training loop.
///
/// # Arguments
///
/// * `trajectory_timings` - Slice of timing data from parallel trajectories
/// * `target` - Target `ForwardTiming` struct to populate
///
/// # Behavior
///
/// Computes averages from the trajectory timings:
/// - `model_preprocessing`: Average across all trajectories
/// - `solver`: Average across all trajectories
/// - `model_postprocessing`: Average across all trajectories
pub fn aggregate_trajectory_timings(
    trajectory_timings: &[TrajectoryTiming],
    target: &crate::timing::ForwardTiming,
) {
    if trajectory_timings.is_empty() {
        return;
    }

    let n = trajectory_timings.len() as u32;

    // Sum all timings (precise values preserved via Cell::get())
    let total_prep: Duration = trajectory_timings
        .iter()
        .map(|t| t.model_preprocessing.get())
        .sum();
    let total_solver: Duration =
        trajectory_timings.iter().map(|t| t.solver.get()).sum();
    let total_post: Duration = trajectory_timings
        .iter()
        .map(|t| t.model_postprocessing.get())
        .sum();

    // Store averages (for representative per-trajectory metrics)
    target.model_preprocessing.set(total_prep / n);
    target.solver.set(total_solver / n);
    target.model_postprocessing.set(total_post / n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::ForwardTiming;

    // Note: Full integration tests require the SDDP infrastructure.
    // Unit tests focus on individual helper functions.

    #[test]
    fn test_step_timing_default() {
        let timing = StepTiming::default();
        assert_eq!(timing.solver_time, Duration::ZERO);
        assert_eq!(timing.state_update_time, Duration::ZERO);
    }

    #[test]
    fn test_aggregate_trajectory_timings_empty() {
        let target = ForwardTiming::default();
        aggregate_trajectory_timings(&[], &target);
        // Should not panic and leave target unchanged
        assert_eq!(target.model_preprocessing.get(), Duration::ZERO);
    }

    #[test]
    fn test_aggregate_trajectory_timings_single() {
        let timing = TrajectoryTiming::default();
        timing.model_preprocessing.set(Duration::from_millis(100));
        timing.solver.set(Duration::from_millis(200));
        timing.model_postprocessing.set(Duration::from_millis(50));
        timing.solver_calls.set(5);

        let target = ForwardTiming::default();
        aggregate_trajectory_timings(&[timing], &target);

        assert_eq!(
            target.model_preprocessing.get(),
            Duration::from_millis(100)
        );
        assert_eq!(target.solver.get(), Duration::from_millis(200));
        assert_eq!(
            target.model_postprocessing.get(),
            Duration::from_millis(50)
        );
    }

    #[test]
    fn test_aggregate_trajectory_timings_average() {
        let timing1 = TrajectoryTiming::default();
        timing1.model_preprocessing.set(Duration::from_millis(100));
        timing1.solver.set(Duration::from_millis(200));
        timing1.model_postprocessing.set(Duration::from_millis(50));
        timing1.solver_calls.set(5);

        let timing2 = TrajectoryTiming::default();
        timing2.model_preprocessing.set(Duration::from_millis(200));
        timing2.solver.set(Duration::from_millis(400));
        timing2.model_postprocessing.set(Duration::from_millis(100));
        timing2.solver_calls.set(5);

        let target = ForwardTiming::default();
        aggregate_trajectory_timings(&[timing1, timing2], &target);

        // Average: (100+200)/2=150, (200+400)/2=300, (50+100)/2=75
        assert_eq!(
            target.model_preprocessing.get(),
            Duration::from_millis(150)
        );
        assert_eq!(target.solver.get(), Duration::from_millis(300));
        assert_eq!(
            target.model_postprocessing.get(),
            Duration::from_millis(75)
        );
    }
}
