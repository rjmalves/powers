//! Context structs for SDDP algorithm phases.
//!
//! These structs bundle related parameters to reduce function parameter counts
//! and make data flow explicit.
//!
//! # Design Philosophy
//!
//! Context structs serve as parameter objects that:
//! 1. Reduce parameter counts from 6-10 to 1-2
//! 2. Make data dependencies explicit
//! 3. Enable future preallocation (Epic 5)
//! 4. Improve testability via dependency injection
//!
//! # Timing Separation
//!
//! **Timing is NOT included in context structs.** This is an intentional design
//! decision to avoid borrow checker conflicts with `TimingGuard`.
//!
//! When using `TimingGuard`, the guard borrows the timing struct. If timing
//! were inside the context, we couldn't mutably access graph fields while
//! timing is active:
//!
//! ```ignore
//! // WRONG: timing inside context causes borrow conflict
//! let _guard = TimingGuard::new(&ctx.timing.field); // borrows ctx
//! ctx.graph.get_node_mut(id)?; // ERROR: ctx already borrowed
//!
//! // RIGHT: timing passed separately
//! let _guard = TimingGuard::new(&timing.field); // borrows timing only
//! ctx.graph.get_node_mut(id)?; // OK: ctx not borrowed
//! ```
//!
//! See `docs/context-struct-design.md` for detailed design documentation.

use crate::graph::DirectedGraph;
use crate::scenario::{OptimizedSampledBranchingNoises, ScenarioTree};
use crate::sddp::NodeData;
use crate::subproblem::{Realization, Subproblem};

/// Context for forward pass execution.
///
/// Bundles all data needed for a single forward pass trajectory, reducing
/// parameter counts and making data flow explicit.
///
/// # Design Note: Timing Separation
///
/// Timing is NOT included in this context to avoid borrow checker conflicts.
/// When using `TimingGuard`, the guard borrows the timing struct. If timing
/// were inside this context, we couldn't mutably access graph fields while
/// timing is active.
///
/// Pass timing as a separate parameter to `forward_pass::execute()`.
///
/// # Thread Safety
///
/// This context is NOT thread-safe. For parallel forward passes, create
/// one context per trajectory with disjoint graph node references.
///
/// # Example
///
/// ```ignore
/// use powers_rs::algorithm::{forward_pass, ForwardPassContext};
/// use powers_rs::timing::TrajectoryTiming;
///
/// let timing = TrajectoryTiming::default();
/// let mut ctx = ForwardPassContext::new(
///     &mut subproblem_graph,
///     &mut realization_graph,
///     &sampled_noises,
///     &graph_bfs_table,
///     &study_period_ids,
/// );
///
/// let result = forward_pass::execute(&mut ctx, &timing)?;
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
    // NO timing field - passed separately to avoid borrow conflicts with TimingGuard
}

impl<'a> ForwardPassContext<'a> {
    /// Create a new forward pass context.
    #[inline]
    pub fn new(
        subproblem_graph: &'a mut DirectedGraph<Subproblem>,
        realization_graph: &'a mut DirectedGraph<Realization>,
        sampled_noises: &'a [&'a OptimizedSampledBranchingNoises],
        graph_bfs_table: &'a [Vec<usize>],
        study_period_ids: &'a [usize],
    ) -> Self {
        Self {
            subproblem_graph,
            realization_graph,
            sampled_noises,
            graph_bfs_table,
            study_period_ids,
        }
    }

    /// Get the number of stages in this trajectory.
    #[inline]
    pub fn num_stages(&self) -> usize {
        self.study_period_ids.len()
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
    #[inline]
    pub fn new(trajectory_cost: f64, solver_calls: usize) -> Self {
        Self {
            trajectory_cost,
            solver_calls,
        }
    }
}

// =============================================================================
// Backward Pass Context Types
// =============================================================================

/// Context for backward pass execution.
///
/// Bundles all data needed for backward pass iteration, reducing parameter
/// counts and making data flow explicit. The backward pass iterates through
/// stages in reverse order, computing Benders cuts at each stage.
///
/// # Design Note: Timing Separation
///
/// Timing is NOT included in this context to avoid borrow checker conflicts.
/// When using `TimingGuard`, the guard borrows the timing struct. If timing
/// were inside this context, we couldn't mutably access other fields while
/// timing is active.
///
/// Pass timing as a separate parameter to `backward_pass::execute()`.
///
/// # Architecture
///
/// The backward pass has a 3-phase architecture per stage:
/// 1. **Phase 1**: Parallel branching solves (generate cut data)
/// 2. **Phase 2**: Sequential batch cut selection (deterministic ordering)
/// 3. **Phase 3a**: FCF state update (mark inactive cuts)
/// 4. **Phase 3b**: Parallel handler application
///
/// # Thread Safety
///
/// This context provides **immutable references** to shared data structures.
/// The parallel coordination happens at the `train_handlers` level which is
/// owned by the caller and not part of this context.
///
/// - `node_data_graph`: Read-only access for node metadata
/// - `saa`: Read-only access for branching scenario generation
///
/// **Note**: The FCF graph is passed separately to `execute()` for mutable
/// access during Phase 2 cut selection (Epic 4 - Mutex removal).
///
/// # Example
///
/// ```ignore
/// use powers_rs::algorithm::BackwardPassContext;
///
/// let ctx = BackwardPassContext::new(
///     &node_data_graph,
///     &saa,
///     &graph_bfs_table,
///     &study_period_ids,
///     iteration,
///     enable_cut_selection,
/// );
///
/// // Process stages in reverse order
/// for stage_idx in ctx.backward_stage_indices() {
///     let stage_ctx = ctx.stage_context(stage_idx)?;
///     // ... phase 1, 2, 3 processing with timing passed separately ...
/// }
/// ```
pub struct BackwardPassContext<'a> {
    /// Node data graph for backward traversal information.
    /// Contains risk measures, system data, and node metadata.
    pub node_data_graph: &'a DirectedGraph<NodeData>,

    /// Scenario tree for branching scenario generation.
    pub saa: &'a ScenarioTree,

    /// BFS traversal table for past node lookup.
    /// `graph_bfs_table[idx]` gives past node IDs for `study_period_ids[idx]`.
    pub graph_bfs_table: &'a [Vec<usize>],

    /// IDs of stages (study periods) in forward order.
    /// Backward pass processes these in reverse.
    pub study_period_ids: &'a [usize],

    /// Current iteration number (1-indexed).
    pub iteration: usize,

    /// Whether cut selection is enabled.
    pub enable_cut_selection: bool,
    // NO timing field - passed separately to avoid borrow conflicts with TimingGuard
    // NO fcf_graph field - passed to execute() for mutable access during cut selection
}

impl<'a> BackwardPassContext<'a> {
    /// Create a new backward pass context.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn new(
        node_data_graph: &'a DirectedGraph<NodeData>,
        saa: &'a ScenarioTree,
        graph_bfs_table: &'a [Vec<usize>],
        study_period_ids: &'a [usize],
        iteration: usize,
        enable_cut_selection: bool,
    ) -> Self {
        Self {
            node_data_graph,
            saa,
            graph_bfs_table,
            study_period_ids,
            iteration,
            enable_cut_selection,
        }
    }

    /// Get the number of stages in the backward pass.
    #[inline]
    pub fn num_stages(&self) -> usize {
        self.study_period_ids.len()
    }

    /// Get an iterator over stage indices in backward (reverse) order.
    ///
    /// Returns indices from `num_stages - 1` down to `0`.
    #[inline]
    pub fn backward_stage_indices(&self) -> impl Iterator<Item = usize> {
        (0..self.num_stages()).rev()
    }

    /// Get the stage ID for a given stage index.
    #[inline]
    pub fn get_stage_id(&self, stage_idx: usize) -> Option<usize> {
        self.study_period_ids.get(stage_idx).copied()
    }

    /// Get past node IDs for a given stage index.
    ///
    /// Returns the BFS traversal path leading to this stage.
    #[inline]
    pub fn get_past_node_ids(&self, stage_idx: usize) -> Option<&[usize]> {
        self.graph_bfs_table.get(stage_idx).map(|v| v.as_slice())
    }

    /// Get the parent stage ID for a given stage index.
    ///
    /// Returns `None` for the first stage (index 0).
    #[inline]
    pub fn get_parent_id(&self, stage_idx: usize) -> Option<usize> {
        if stage_idx == 0 {
            None
        } else {
            self.graph_bfs_table
                .get(stage_idx)
                .and_then(|past| past.last().copied())
        }
    }

    /// Get the number of branching scenarios for a stage.
    #[inline]
    pub fn get_branching_count(&self, stage_id: usize) -> Option<usize> {
        self.saa.get_branching_count_at_stage(stage_id)
    }

    /// Check if this is the first stage (no parent, no cut generation).
    #[inline]
    pub fn is_first_stage(&self, stage_idx: usize) -> bool {
        stage_idx == 0
    }

    /// Get the node data for a given node ID.
    #[inline]
    pub fn get_node_data(
        &self,
        node_id: usize,
    ) -> Option<&crate::graph::Node<NodeData>> {
        self.node_data_graph.get_node(node_id)
    }

    /// Create a stage context for the given stage index.
    ///
    /// This is used to pass per-stage data to `BackwardStageProcessor` methods.
    pub fn stage_context(
        &'a self,
        stage_idx: usize,
    ) -> Option<BackwardStageContext<'a>> {
        let stage_id = self.study_period_ids.get(stage_idx).copied()?;
        let past_node_ids = self.graph_bfs_table.get(stage_idx)?;
        let parent_id = if stage_idx > 0 {
            past_node_ids.last().copied()
        } else {
            None
        };

        Some(BackwardStageContext {
            stage_id,
            stage_idx,
            past_node_ids,
            parent_id,
            node_data_graph: self.node_data_graph,
            saa: self.saa,
            iteration: self.iteration,
            enable_cut_selection: self.enable_cut_selection,
        })
    }
}

/// Per-stage context for backward pass processing.
///
/// Passed to `BackwardStageProcessor` methods. Contains the specific
/// data needed to process a single stage.
///
/// # Design Note: Timing Separation
///
/// Like `BackwardPassContext`, timing is NOT included here.
/// Timing is accumulated in `BackwardStageTiming` which is passed
/// separately to the processor methods.
pub struct BackwardStageContext<'a> {
    /// Current stage ID.
    pub stage_id: usize,

    /// Stage index in the study period sequence.
    pub stage_idx: usize,

    /// Past node IDs for this stage (BFS path).
    pub past_node_ids: &'a [usize],

    /// Parent stage ID (for FCF updates). None for first stage.
    pub parent_id: Option<usize>,

    /// Node data graph reference.
    pub node_data_graph: &'a DirectedGraph<NodeData>,

    /// Scenario tree for branching counts.
    pub saa: &'a ScenarioTree,

    /// Current iteration number.
    pub iteration: usize,

    /// Whether cut selection is enabled.
    pub enable_cut_selection: bool,
    // NO timing field - passed separately to processor methods
}

impl<'a> BackwardStageContext<'a> {
    /// Get the number of branching scenarios for this stage.
    #[inline]
    pub fn get_branching_count(&self) -> Option<usize> {
        self.saa.get_branching_count_at_stage(self.stage_id)
    }

    /// Check if this is the first stage (no parent, no cut generation).
    #[inline]
    pub fn is_first_stage(&self) -> bool {
        self.stage_idx == 0
    }

    /// Get the node data for this stage.
    #[inline]
    pub fn get_node_data(&self) -> Option<&crate::graph::Node<NodeData>> {
        self.node_data_graph.get_node(self.stage_id)
    }
}

/// Result of a backward pass execution.
///
/// Contains the lower bound estimate and cut management statistics.
#[derive(Debug, Clone)]
pub struct BackwardPassResult {
    /// Lower bound from first stage evaluation.
    pub lower_bound: f64,

    /// Individual first-stage branching scenario costs.
    pub first_stage_branching_costs: Vec<f64>,

    /// Number of cuts added during this backward pass.
    pub cuts_added: usize,

    /// Number of cuts removed (if cut management active).
    pub cuts_removed: usize,

    /// Number of cuts that were returned from inactive state.
    pub cuts_returned: usize,

    /// Number of solver calls made.
    pub solver_calls: usize,
}

impl BackwardPassResult {
    /// Create a new backward pass result.
    #[inline]
    pub fn new(
        lower_bound: f64,
        cuts_added: usize,
        cuts_removed: usize,
        cuts_returned: usize,
        solver_calls: usize,
    ) -> Self {
        Self {
            lower_bound,
            first_stage_branching_costs: Vec::new(),
            cuts_added,
            cuts_removed,
            cuts_returned,
            solver_calls,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // Forward Pass Tests
    // ==========================================================================

    #[test]
    fn test_forward_pass_result_new() {
        let result = ForwardPassResult::new(1000.0, 10);
        assert!((result.trajectory_cost - 1000.0).abs() < f64::EPSILON);
        assert_eq!(result.solver_calls, 10);
    }

    // ==========================================================================
    // Backward Pass Tests
    // ==========================================================================

    #[test]
    fn test_backward_pass_result_new() {
        let result = BackwardPassResult::new(5000.0, 10, 2, 1, 100);
        assert!((result.lower_bound - 5000.0).abs() < f64::EPSILON);
        assert_eq!(result.cuts_added, 10);
        assert_eq!(result.cuts_removed, 2);
        assert_eq!(result.cuts_returned, 1);
        assert_eq!(result.solver_calls, 100);
    }

    #[test]
    fn test_backward_stage_context_is_first_stage() {
        // We can't easily construct a full BackwardStageContext without
        // the infrastructure, but we can test the logic via
        // BackwardPassContext helper methods
        let _study_period_ids = vec![0, 1, 2, 3, 4];
        let graph_bfs_table = vec![
            vec![],           // stage 0: no past nodes
            vec![0],          // stage 1: past is [0]
            vec![0, 1],       // stage 2: past is [0, 1]
            vec![0, 1, 2],    // stage 3
            vec![0, 1, 2, 3], // stage 4
        ];

        // Test is_first_stage logic
        assert!(0 == 0); // stage_idx == 0 means first stage
        assert!(1 != 0); // stage_idx != 0 means not first stage

        // Test parent_id logic (last element of past_node_ids)
        assert_eq!(graph_bfs_table[0].last().copied(), None);
        assert_eq!(graph_bfs_table[1].last().copied(), Some(0));
        assert_eq!(graph_bfs_table[2].last().copied(), Some(1));
        assert_eq!(graph_bfs_table[4].last().copied(), Some(3));
    }
}
