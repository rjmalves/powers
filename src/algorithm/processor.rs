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
//!          → par_iter_mut on handlers
//!          → Returns Vec<CutData>
//!          
//! Phase 2: SEQUENTIAL - Batch cut selection (deterministic ordering)
//!          → Sort by forward_pass_idx for reproducibility
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
use crate::fcf::{
    AggregatedCutSelectionResult, BatchCutSelectionResult, CutData,
    FutureCostFunction,
};
use crate::graph::DirectedGraph;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Timing from Phase 1 cut computation.
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
#[derive(Debug, Clone, Copy, Default)]
pub struct FirstStageTiming {
    /// Time spent in solver.
    pub solver: Duration,
    /// Time spent extracting state.
    pub state_extraction: Duration,
}

/// Result of Phase 1 cut computation.
pub struct Phase1Result {
    /// Cut data from all forward passes.
    pub cut_data: Vec<CutData>,
    /// Aggregated timing from parallel computation.
    pub timing: CutComputationTiming,
}

/// Result of Phase 2 cut selection.
pub struct Phase2Result {
    /// Batch selection result from FCF.
    pub batch_result: BatchCutSelectionResult,
    /// Aggregated result for handler application.
    pub aggregated: AggregatedCutSelectionResult,
    /// Cloned cuts for parallel application (forward_pass_idx, cut).
    pub cuts: Vec<(usize, Arc<BendersCut>)>,
    /// Time spent in cut selection.
    pub cut_selection_time: Duration,
    /// Time spent updating FCF state.
    pub fcf_update_time: Duration,
    /// Time spent cloning cuts.
    pub cut_cloning_time: Duration,
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
///     fcf_graph: &DirectedGraph<Arc<Mutex<FutureCostFunction>>>,
/// ) -> Result<(), String> {
///     // Phase 1: Parallel cut computation
///     let phase1 = processor.compute_cuts_parallel(stage_ctx)?;
///     
///     // Phase 2: Sequential cut selection (FCF passed here)
///     let phase2 = processor.select_cuts_batch(phase1.cut_data, stage_ctx, fcf_graph)?;
///     
///     // Phase 3: Parallel cut application
///     processor.apply_cuts_parallel(&phase2, stage_ctx)?;
///     
///     Ok(())
/// }
/// ```
pub trait BackwardStageProcessor {
    /// Phase 1: Compute cuts in parallel across all handlers.
    ///
    /// Each handler computes cut data for its branching scenarios.
    /// Results are collected into a vector for Phase 2 processing.
    ///
    /// # Arguments
    ///
    /// * `stage_ctx` - Per-stage context with node and scenario info
    ///
    /// # Returns
    ///
    /// * `Ok(Phase1Result)` - Cut data and timing from all handlers
    /// * `Err(String)` - If any handler fails
    fn compute_cuts_parallel(
        &mut self,
        stage_ctx: &BackwardStageContext,
    ) -> Result<Phase1Result, String>;

    /// Phase 2: Sequential batch cut selection.
    ///
    /// Sorts cuts for deterministic ordering, applies cut selection to FCF,
    /// and prepares results for Phase 3. This phase is ALWAYS sequential
    /// to ensure reproducibility.
    ///
    /// # Arguments
    ///
    /// * `cut_data` - Cut data from Phase 1 (will be sorted by forward_pass_idx)
    /// * `stage_ctx` - Per-stage context
    /// * `fcf_graph` - FCF graph for cut operations (passed to avoid unsafe storage)
    ///
    /// # Returns
    ///
    /// * `Ok(Phase2Result)` - Selection result with cuts to apply
    /// * `Err(String)` - If FCF access fails
    fn select_cuts_batch(
        &mut self,
        cut_data: Vec<CutData>,
        stage_ctx: &BackwardStageContext,
        fcf_graph: &DirectedGraph<Arc<Mutex<FutureCostFunction>>>,
    ) -> Result<Phase2Result, String>;

    /// Phase 3: Apply cut results in parallel to all handler models.
    ///
    /// Updates all handler subproblem models with the new cuts.
    /// This phase can be parallel since each handler has independent models.
    ///
    /// # Arguments
    ///
    /// * `phase2_result` - Results from Phase 2 with cuts to apply
    /// * `stage_ctx` - Per-stage context
    ///
    /// # Returns
    ///
    /// * `Ok(Duration)` - Time spent in handler application
    /// * `Err(String)` - If any handler fails
    fn apply_cuts_parallel(
        &mut self,
        phase2_result: &Phase2Result,
        stage_ctx: &BackwardStageContext,
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
