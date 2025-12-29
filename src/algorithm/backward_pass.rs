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
//! # Timing Separation
//!
//! Timing is passed as a separate parameter (not inside context) to enable
//! `TimingGuard` usage without borrow conflicts. See Epic 3 Sprint 1 (T-021)
//! for the architectural rationale.
//!
//! ```text
//! // CORRECT: timing separate from context
//! pub fn execute<P: BackwardStageProcessor>(
//!     processor: &mut P,
//!     ctx: &BackwardPassContext,
//!     timing: &BackwardPassTimingAccumulator,
//!     fcf_graph: &DirectedGraph<Arc<Mutex<FutureCostFunction>>>,
//! ) -> Result<BackwardPassResult, String>
//! ```

use crate::algorithm::context::{BackwardPassContext, BackwardPassResult};
use crate::algorithm::processor::BackwardStageProcessor;
use crate::fcf::FutureCostFunction;
use crate::graph::DirectedGraph;
use std::cell::Cell;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Timing accumulator for backward pass.
///
/// Uses `Cell<Duration>` for interior mutability, enabling `TimingGuard`
/// to accumulate time without requiring `&mut self`. This is critical
/// for avoiding borrow checker conflicts.
///
/// # Design Note
///
/// All timing fields use `Cell<Duration>` rather than `Duration` because:
/// 1. `TimingGuard` needs to add time on drop via shared reference
/// 2. The backward pass context may be borrowed while timing is active
/// 3. `Cell` provides interior mutability without runtime cost for `Copy` types
#[derive(Debug, Default)]
pub struct BackwardPassTimingAccumulator {
    /// Time spent in backward preprocessing (per-stage setup).
    pub preprocessing: Cell<Duration>,

    /// Time spent in model preprocessing (Phase 1 parallel).
    pub model_preprocessing: Cell<Duration>,

    /// Time spent in solver (Phase 1 parallel).
    pub solver: Cell<Duration>,

    /// Time spent in model postprocessing (Phase 1 parallel).
    pub model_postprocessing: Cell<Duration>,

    /// Time spent in cut selection (Phase 2 sequential).
    pub cut_selection: Cell<Duration>,

    /// Time spent updating FCF state (Phase 3a sequential).
    pub fcf_state_update: Cell<Duration>,

    /// Time spent cloning cuts for parallel application.
    pub cut_cloning: Cell<Duration>,

    /// Time spent in handler application (Phase 3b parallel).
    pub handler_application: Cell<Duration>,

    /// Number of solver calls made.
    pub solver_calls: Cell<usize>,
}

impl BackwardPassTimingAccumulator {
    /// Create a new timing accumulator with all fields zeroed.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment the solver call count.
    #[inline]
    pub fn increment_solver_calls(&self, count: usize) {
        self.solver_calls.set(self.solver_calls.get() + count);
    }

    /// Get the current solver call count.
    #[inline]
    pub fn get_solver_calls(&self) -> usize {
        self.solver_calls.get()
    }

    /// Add duration to a timing field.
    ///
    /// This is a utility method for accumulating timing from sub-operations.
    #[inline]
    pub fn add_duration(field: &Cell<Duration>, duration: Duration) {
        field.set(field.get() + duration);
    }

    /// Convert accumulated timing to a snapshot struct for reporting.
    #[inline]
    pub fn snapshot(&self) -> BackwardPassTimingSnapshot {
        BackwardPassTimingSnapshot {
            preprocessing: self.preprocessing.get(),
            model_preprocessing: self.model_preprocessing.get(),
            solver: self.solver.get(),
            model_postprocessing: self.model_postprocessing.get(),
            cut_selection: self.cut_selection.get(),
            fcf_state_update: self.fcf_state_update.get(),
            cut_cloning: self.cut_cloning.get(),
            handler_application: self.handler_application.get(),
            solver_calls: self.solver_calls.get(),
        }
    }
}

/// Immutable snapshot of backward pass timing.
///
/// Unlike `BackwardPassTimingAccumulator`, this struct has no interior
/// mutability and can be freely cloned and passed around.
#[derive(Debug, Clone, Default)]
pub struct BackwardPassTimingSnapshot {
    /// Time spent in backward preprocessing.
    pub preprocessing: Duration,

    /// Time spent in model preprocessing (Phase 1).
    pub model_preprocessing: Duration,

    /// Time spent in solver (Phase 1).
    pub solver: Duration,

    /// Time spent in model postprocessing (Phase 1).
    pub model_postprocessing: Duration,

    /// Time spent in cut selection (Phase 2).
    pub cut_selection: Duration,

    /// Time spent updating FCF state (Phase 3a).
    pub fcf_state_update: Duration,

    /// Time spent cloning cuts.
    pub cut_cloning: Duration,

    /// Time spent in handler application (Phase 3b).
    pub handler_application: Duration,

    /// Number of solver calls made.
    pub solver_calls: usize,
}

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
/// use powers_rs::algorithm::backward_pass::{execute, BackwardPassTimingAccumulator};
/// use powers_rs::algorithm::{BackwardPassContext, ParallelHandlerCoordinator};
///
/// let timing = BackwardPassTimingAccumulator::new();
/// let result = execute(
///     &mut coordinator,
///     &backward_ctx,
///     &timing,
///     &fcf_graph,
/// )?;
///
/// println!("Lower bound: {}", result.lower_bound);
/// println!("Cuts added: {}", result.cuts_added);
/// ```
pub fn execute<P: BackwardStageProcessor>(
    processor: &mut P,
    ctx: &BackwardPassContext,
    timing: &BackwardPassTimingAccumulator,
    fcf_graph: &DirectedGraph<Arc<Mutex<FutureCostFunction>>>,
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
    result.solver_calls = timing.get_solver_calls();

    Ok(result)
}

/// Execute first stage evaluation (no cut generation).
fn execute_first_stage<P: BackwardStageProcessor>(
    processor: &mut P,
    stage_ctx: &crate::algorithm::context::BackwardStageContext,
    result: &mut BackwardPassResult,
    timing: &BackwardPassTimingAccumulator,
) -> Result<(), String> {
    let (lb, first_timing) = processor.eval_first_stage_bound(stage_ctx)?;
    result.lower_bound = lb;

    // Accumulate first stage timing
    BackwardPassTimingAccumulator::add_duration(
        &timing.solver,
        first_timing.solver,
    );
    BackwardPassTimingAccumulator::add_duration(
        &timing.model_postprocessing,
        first_timing.state_extraction,
    );

    // Count solver calls for first stage
    let num_branchings = stage_ctx.get_branching_count().unwrap_or(1);
    timing.increment_solver_calls(
        num_branchings * processor.num_forward_passes(),
    );

    Ok(())
}

/// Execute a single stage of the backward pass (non-first stage).
///
/// Performs the 3-phase architecture:
/// 1. Phase 1: Parallel cut computation
/// 2. Phase 2: Sequential batch cut selection
/// 3. Phase 3: FCF state update + parallel handler application
fn execute_stage<P: BackwardStageProcessor>(
    processor: &mut P,
    stage_ctx: &crate::algorithm::context::BackwardStageContext,
    result: &mut BackwardPassResult,
    timing: &BackwardPassTimingAccumulator,
    fcf_graph: &DirectedGraph<Arc<Mutex<FutureCostFunction>>>,
) -> Result<(), String> {
    // Phase 1: Parallel cut computation
    let phase1 = processor.compute_cuts_parallel(stage_ctx)?;

    // Accumulate Phase 1 timing (returned from parallel computation)
    BackwardPassTimingAccumulator::add_duration(
        &timing.model_preprocessing,
        phase1.timing.model_preprocessing,
    );
    BackwardPassTimingAccumulator::add_duration(
        &timing.solver,
        phase1.timing.solver,
    );
    BackwardPassTimingAccumulator::add_duration(
        &timing.model_postprocessing,
        phase1.timing.model_postprocessing,
    );
    timing.increment_solver_calls(phase1.timing.solver_calls);

    // Phase 2: Sequential batch cut selection
    let phase2 =
        processor.select_cuts_batch(phase1.cut_data, stage_ctx, fcf_graph)?;

    // Accumulate Phase 2 timing
    BackwardPassTimingAccumulator::add_duration(
        &timing.cut_selection,
        phase2.cut_selection_time,
    );
    BackwardPassTimingAccumulator::add_duration(
        &timing.fcf_state_update,
        phase2.fcf_update_time,
    );
    BackwardPassTimingAccumulator::add_duration(
        &timing.cut_cloning,
        phase2.cut_cloning_time,
    );

    // Update result counts
    result.cuts_added += phase2.batch_result.new_cut_ids.len();
    result.cuts_removed += phase2.batch_result.removing_cut_ids.len();
    result.cuts_returned += phase2.batch_result.returning_cut_ids.len();

    // Phase 3: Parallel cut application
    let handler_time = processor.apply_cuts_parallel(&phase2, stage_ctx)?;
    BackwardPassTimingAccumulator::add_duration(
        &timing.handler_application,
        handler_time,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_accumulator_default() {
        let timing = BackwardPassTimingAccumulator::new();
        assert_eq!(timing.preprocessing.get(), Duration::ZERO);
        assert_eq!(timing.solver.get(), Duration::ZERO);
        assert_eq!(timing.cut_selection.get(), Duration::ZERO);
        assert_eq!(timing.get_solver_calls(), 0);
    }

    #[test]
    fn test_timing_accumulator_increment_solver_calls() {
        let timing = BackwardPassTimingAccumulator::new();

        timing.increment_solver_calls(5);
        assert_eq!(timing.get_solver_calls(), 5);

        timing.increment_solver_calls(10);
        assert_eq!(timing.get_solver_calls(), 15);
    }

    #[test]
    fn test_timing_accumulator_add_duration() {
        let timing = BackwardPassTimingAccumulator::new();

        BackwardPassTimingAccumulator::add_duration(
            &timing.solver,
            Duration::from_millis(100),
        );
        assert_eq!(timing.solver.get(), Duration::from_millis(100));

        BackwardPassTimingAccumulator::add_duration(
            &timing.solver,
            Duration::from_millis(50),
        );
        assert_eq!(timing.solver.get(), Duration::from_millis(150));
    }

    #[test]
    fn test_timing_snapshot() {
        let timing = BackwardPassTimingAccumulator::new();

        timing.preprocessing.set(Duration::from_millis(10));
        timing.solver.set(Duration::from_millis(200));
        timing.cut_selection.set(Duration::from_millis(50));
        timing.increment_solver_calls(25);

        let snapshot = timing.snapshot();

        assert_eq!(snapshot.preprocessing, Duration::from_millis(10));
        assert_eq!(snapshot.solver, Duration::from_millis(200));
        assert_eq!(snapshot.cut_selection, Duration::from_millis(50));
        assert_eq!(snapshot.solver_calls, 25);
    }

    #[test]
    fn test_timing_snapshot_is_independent() {
        let timing = BackwardPassTimingAccumulator::new();
        timing.solver.set(Duration::from_millis(100));

        let snapshot = timing.snapshot();

        // Modify original
        timing.solver.set(Duration::from_millis(200));

        // Snapshot should be unchanged
        assert_eq!(snapshot.solver, Duration::from_millis(100));
    }

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
