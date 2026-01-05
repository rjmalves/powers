//! Backward pass processor traits.
//!
//! These traits abstract the parallel coordination of backward pass stages,
//! enabling extraction of the backward pass loop while hiding handler details.
//!
//! # Architecture
//!
//! The backward pass has a strict 3-phase structure per stage:
//!
//! ```text
//! Phase 1: PARALLEL - Compute cuts across all forward passes
//!          → par_iter_mut on handlers into staging buffers
//!          → Sequential copy to preallocated pool slots
//!          → Returns slot indices
//!          
//! Phase 2: SEQUENTIAL - Batch cut finalization and selection
//!          → Finalize cuts at slots (deterministic ordering)
//!          → Apply cut selection to FCF
//!          → Returns BatchCutSelectionResult
//!          
//! Phase 3a: SEQUENTIAL - Update FCF state (mark inactive)
//! Phase 3b: PARALLEL - Apply results to all models
//!          → par_iter_mut on handlers
//! ```
//!
//! # Thread Safety
//!
//! The `BackwardStageProcessor` trait methods are called sequentially from
//! the backward loop, but implementations may use internal parallelism in
//! Phase 1 and Phase 3b.

use crate::algorithm::context::BackwardStageContext;
use crate::cut::BendersCut;
use crate::fcf::{AggregatedCutSelectionResult, FutureCostFunction};
use crate::graph::DirectedGraph;
use std::time::Duration;

/// Timing from Phase 1 cut computation.
///
/// This struct uses plain `Duration` (not `Cell<Duration>`) because it's
/// a return value from processor methods. The caller (backward_pass::execute)
/// accumulates these values into `timing::NewBackwardTiming` which uses
/// `Cell<Duration>` for interior mutability.
///
/// # Design Pattern
///
/// **Return types use `Duration`**, **accumulators use `Cell<Duration>`**:
/// - Methods return `CutComputationTiming` with plain `Duration`
/// - `backward_pass::execute()` accumulates into `NewBackwardTiming` with `Cell<Duration>`
/// - This avoids borrow checker conflicts when using `TimingGuard`
///
/// # Conversion Example
///
/// ```ignore
/// // Phase 1 returns CutComputationTiming (plain Duration)
/// let phase1 = processor.compute_cuts_parallel_into_slots(stage_ctx, fcf_graph)?;
///
/// // Accumulate into NewBackwardTiming (Cell<Duration>)
/// timing.phase1.model_preprocessing.set(
///     timing.phase1.model_preprocessing.get() + phase1.timing.model_preprocessing
/// );
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CutComputationTiming {
    /// Time spent in model preprocessing.
    pub model_preprocessing: Duration,
    /// Time spent in solver.
    pub solver: Duration,
    /// Time spent in model postprocessing.
    pub model_postprocessing: Duration,
    /// Number of solver calls made.
    pub solver_calls: usize,
}

/// Timing from first stage evaluation.
///
/// Like `CutComputationTiming`, this uses plain `Duration` as it's a return value.
/// The caller accumulates it into the appropriate timing accumulator.
#[derive(Debug, Clone, Copy, Default)]
pub struct FirstStageTiming {
    /// Time spent in solver.
    pub solver: Duration,
    /// Time spent extracting state.
    pub state_extraction: Duration,
}

/// Result of Phase 1 cut computation using zero-allocation path.
///
/// Contains slot indices where cuts were written directly to preallocated FCF pools.
pub struct Phase1Result {
    /// Slot indices in the FCF pools where cuts were written.
    pub slots: Vec<usize>,
    /// Aggregated timing from cut computation.
    pub timing: CutComputationTiming,
}

/// Result of Phase 2 cut selection.
pub struct Phase2Result {
    /// Aggregated result from cut selection and handler coordination.
    pub aggregated: AggregatedCutSelectionResult,
    /// Cut IDs to apply (references pool by index). Zero allocation.
    pub cut_ids: Vec<usize>,
    /// Time spent in cut selection.
    pub cut_selection_time: Duration,
    /// Time spent updating FCF state.
    pub fcf_update_time: Duration,
}

/// Trait for processing backward pass stages.
///
/// This trait abstracts the 3-phase backward pass processing, enabling
/// the backward loop to be extracted while encapsulating parallel
/// coordination details.
///
/// # Thread Safety
///
/// Implementations must handle parallel execution in Phase 1 and Phase 3b.
/// The trait methods are called sequentially from the backward loop.
///
/// # Design Note: Timing
///
/// Timing is returned as part of result types rather than accumulated into
/// a timing context. This avoids the borrow checker conflicts that occur
/// when timing is embedded in context structs.
///
/// # Example
///
/// ```ignore
/// fn execute_backward_stage<P: BackwardStageProcessor>(
///     processor: &mut P,
///     stage_ctx: &BackwardStageContext,
///     fcf_graph: &mut DirectedGraph<FutureCostFunction>,
/// ) -> Result<(), String> {
///     // Phase 1: Parallel cut computation into staging buffers, then sequential pool update
///     let phase1 = processor.compute_cuts_parallel_into_slots(stage_ctx, fcf_graph)?;
///     
///     // Phase 2: Sequential cut finalization and selection
///     let phase2 = processor.select_cuts_from_slots(phase1.slots, stage_ctx, fcf_graph)?;
///     
///     // Phase 3: Parallel cut application
///     processor.apply_cuts_parallel(&phase2, stage_ctx, &fcf_cut_pool)?;
///     
///     Ok(())
/// }
/// ```
pub trait BackwardStageProcessor {
    /// Phase 3: Apply cut results in parallel to all handler models.
    ///
    /// Updates all handler subproblem models with the new cuts.
    /// This phase can be parallel since each handler has independent models.
    ///
    /// # Arguments
    ///
    /// * `phase2_result` - Results from Phase 2 with cut IDs to apply
    /// * `stage_ctx` - Per-stage context
    /// * `cut_pool` - Shared read access to the FCF cut pool (zero allocation)
    ///
    /// # Returns
    ///
    /// * `Ok(Duration)` - Time spent in handler application
    /// * `Err(String)` - If any handler fails
    fn apply_cuts_parallel(
        &mut self,
        phase2_result: &Phase2Result,
        stage_ctx: &BackwardStageContext,
        cut_pool: &[BendersCut],
    ) -> Result<Duration, String>;

    /// Evaluate first stage bound (no cut generation).
    ///
    /// Called for the first stage where no cuts are generated,
    /// only the lower bound is computed.
    ///
    /// # Arguments
    ///
    /// * `stage_ctx` - Per-stage context for first stage
    ///
    /// # Returns
    ///
    /// * `Ok((lower_bound, timing))` - Lower bound and evaluation timing
    /// * `Err(String)` - If evaluation fails
    fn eval_first_stage_bound(
        &mut self,
        stage_ctx: &BackwardStageContext,
    ) -> Result<(f64, FirstStageTiming), String>;

    /// Get the number of forward passes (handlers).
    fn num_forward_passes(&self) -> usize;

    /// Phase 1 (Zero-Allocation): Compute cuts directly into FCF pool slots.
    ///
    /// Writes cut and state coefficients directly to preallocated FCF pool slots.
    ///
    /// # Arguments
    ///
    /// * `stage_ctx` - Per-stage context with node and scenario info
    /// * `fcf_graph` - FCF graph for pool access
    ///
    /// # Returns
    ///
    /// * `Ok(Phase1Result)` - Slot indices and timing
    /// * `Err(String)` - If any handler fails
    fn compute_cuts_into_slots(
        &mut self,
        stage_ctx: &BackwardStageContext,
        fcf_graph: &mut DirectedGraph<FutureCostFunction>,
    ) -> Result<Phase1Result, String>;

    /// Phase 2 (Zero-Allocation): Finalize cuts at slots and select.
    ///
    /// Runs domination evaluation for cuts already written to pool slots,
    /// prepares results for Phase 3. Must be called after `compute_cuts_into_slots`.
    ///
    /// # Arguments
    ///
    /// * `slots` - Slot indices from Phase 1
    /// * `timing` - Timing from Phase 1 (for result aggregation)
    /// * `stage_ctx` - Per-stage context
    /// * `fcf_graph` - FCF graph for cut operations
    ///
    /// # Returns
    ///
    /// * `Ok(Phase2Result)` - Selection result with cuts to apply
    /// * `Err(String)` - If FCF access fails
    fn select_cuts_from_slots(
        &mut self,
        slots: Vec<usize>,
        stage_ctx: &BackwardStageContext,
        fcf_graph: &mut DirectedGraph<FutureCostFunction>,
    ) -> Result<Phase2Result, String>;

    /// Phase 1 (Parallel Zero-Allocation): Compute cuts into staging buffers, then copy to slots.
    ///
    /// This method enables parallel cut computation while maintaining zero allocation.
    /// Each handler computes into its own staging buffer (parallel), then results
    /// are copied to global pools in forward_pass_idx order (sequential).
    ///
    /// # Algorithm
    ///
    /// 1. **Phase 1a (Parallel)**: `par_iter_mut` on handlers, each calls `compute_cut_into_staging()`
    /// 2. **Phase 1b (Sequential)**: Iterate handlers in order, copy staging to pools
    ///
    /// # Arguments
    ///
    /// * `stage_ctx` - Per-stage context with node and scenario info
    /// * `fcf_graph` - FCF graph for pool access
    ///
    /// # Returns
    ///
    /// * `Ok(Phase1Result)` - Slot indices and timing
    /// * `Err(String)` - If any handler fails
    fn compute_cuts_parallel_into_slots(
        &mut self,
        stage_ctx: &BackwardStageContext,
        fcf_graph: &mut DirectedGraph<FutureCostFunction>,
    ) -> Result<Phase1Result, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut_computation_timing_default() {
        let timing = CutComputationTiming::default();
        assert_eq!(timing.model_preprocessing, Duration::ZERO);
        assert_eq!(timing.solver, Duration::ZERO);
        assert_eq!(timing.model_postprocessing, Duration::ZERO);
        assert_eq!(timing.solver_calls, 0);
    }

    #[test]
    fn test_first_stage_timing_default() {
        let timing = FirstStageTiming::default();
        assert_eq!(timing.solver, Duration::ZERO);
        assert_eq!(timing.state_extraction, Duration::ZERO);
    }
}
