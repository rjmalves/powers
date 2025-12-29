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
//! # Preallocation Awareness
//!
//! All context structs are designed with future preallocation in mind:
//! - Data sizes are known from input at initialization
//! - Fields use references to avoid ownership issues
//! - Future fields for preallocated buffers are documented
//!
//! See `docs/context-struct-design.md` for detailed design documentation.

use crate::fcf::FutureCostFunction;
use crate::graph::DirectedGraph;
use crate::scenario::{OptimizedSampledBranchingNoises, ScenarioTree};
use crate::sddp::NodeData;
use crate::subproblem::{Realization, Subproblem};
use crate::timing::{BackwardTiming, ForwardTiming};
use std::sync::Mutex;
use std::time::Duration;

/// Context for forward pass execution.
///
/// Bundles all data needed for a single forward pass trajectory, reducing
/// parameter counts and making data flow explicit.
///
/// # Preallocation Opportunities
///
/// All data sizes are known at initialization from input:
///
/// | Data | Source | Size |
/// |------|--------|------|
/// | `realizations` | `config.stages` | One per stage |
/// | Stage solution buffers | `subproblem.num_variables()` | Per stage |
/// | State coefficients | `system.hydros.len()` | Per hydro |
///
/// # Thread Safety
///
/// This context is NOT thread-safe. For parallel forward passes, create
/// one context per trajectory with disjoint graph node references.
///
/// # Example
///
/// ```ignore
/// use powers_rs::algorithm::ForwardPassContext;
///
/// let mut ctx = ForwardPassContext::new(
///     &mut subproblem_graph,
///     &mut realization_graph,
///     &sampled_noises,
///     &graph_bfs_table,
///     &study_period_ids,
///     &iteration_timing.forward,
/// );
///
/// let result = forward_pass::execute(&mut ctx)?;
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

    /// Timing storage for this forward pass.
    /// Uses `Cell<Duration>` for interior mutability.
    pub timing: &'a ForwardTiming,
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
        timing: &'a ForwardTiming,
    ) -> Self {
        Self {
            subproblem_graph,
            realization_graph,
            sampled_noises,
            graph_bfs_table,
            study_period_ids,
            timing,
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

/// Timing data collected during a single trajectory's forward pass.
///
/// This is the internal timing that gets aggregated across parallel trajectories.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrajectoryTiming {
    /// Time spent in model preprocessing for this trajectory.
    pub model_preprocessing: Duration,

    /// Time spent in solver for this trajectory.
    pub solver: Duration,

    /// Time spent in model postprocessing for this trajectory.
    pub model_postprocessing: Duration,

    /// Number of solver calls in this trajectory.
    pub solver_calls: usize,
}

impl TrajectoryTiming {
    /// Add timing from another trajectory.
    #[inline]
    pub fn add(&mut self, other: &TrajectoryTiming) {
        self.model_preprocessing += other.model_preprocessing;
        self.solver += other.solver;
        self.model_postprocessing += other.model_postprocessing;
        self.solver_calls += other.solver_calls;
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
/// - `fcf_graph`: Shared via `Mutex<FutureCostFunction>` - locked during Phase 2/3a
/// - `saa`: Read-only access for branching scenario generation
///
/// # Preallocation Opportunities
///
/// | Data | Source | Size |
/// |------|--------|------|
/// | Branching realizations | `config.backward_scenarios × num_stages` | ~1MB |
/// | Cut coefficients | `num_state_variables` | Per cut |
/// | Dual values buffer | `subproblem.num_constraints()` | Per solve |
/// | Phase 1 results | `num_forward_passes` | Per stage |
///
/// # Example
///
/// ```ignore
/// use powers_rs::algorithm::BackwardPassContext;
///
/// let ctx = BackwardPassContext::new(
///     &node_data_graph,
///     &fcf_graph,
///     &saa,
///     &graph_bfs_table,
///     &study_period_ids,
///     iteration,
///     enable_cut_selection,
///     &iteration_timing.backward,
/// );
///
/// // Process stages in reverse order
/// for stage_idx in ctx.backward_stage_indices() {
///     let stage_id = ctx.study_period_ids[stage_idx];
///     let past_node_ids = ctx.get_past_node_ids(stage_idx)?;
///     // ... phase 1, 2, 3 processing ...
/// }
/// ```
pub struct BackwardPassContext<'a> {
    /// Node data graph for backward traversal information.
    /// Contains risk measures, system data, and node metadata.
    pub node_data_graph: &'a DirectedGraph<NodeData>,

    /// FCF graph for cut updates.
    /// Each node's FCF is protected by Mutex for thread-safe access.
    pub fcf_graph: &'a DirectedGraph<Mutex<FutureCostFunction>>,

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

    /// Timing storage for this backward pass.
    pub timing: &'a BackwardTiming,
}

impl<'a> BackwardPassContext<'a> {
    /// Create a new backward pass context.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn new(
        node_data_graph: &'a DirectedGraph<NodeData>,
        fcf_graph: &'a DirectedGraph<Mutex<FutureCostFunction>>,
        saa: &'a ScenarioTree,
        graph_bfs_table: &'a [Vec<usize>],
        study_period_ids: &'a [usize],
        iteration: usize,
        enable_cut_selection: bool,
        timing: &'a BackwardTiming,
    ) -> Self {
        Self {
            node_data_graph,
            fcf_graph,
            saa,
            graph_bfs_table,
            study_period_ids,
            iteration,
            enable_cut_selection,
            timing,
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

    /// Get the FCF node for a given node ID.
    #[inline]
    pub fn get_fcf_node(
        &self,
        node_id: usize,
    ) -> Option<&crate::graph::Node<Mutex<FutureCostFunction>>> {
        self.fcf_graph.get_node(node_id)
    }
}

/// Result of a backward pass execution.
///
/// Contains the lower bound estimate and cut management statistics.
#[derive(Debug, Clone)]
pub struct BackwardPassResult {
    /// Lower bound from first stage evaluation.
    pub lower_bound: f64,

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
            cuts_added,
            cuts_removed,
            cuts_returned,
            solver_calls,
        }
    }
}

/// Timing data collected during backward pass Phase 1 (per-stage).
///
/// This captures timing for the parallel branching solves at a single stage.
/// It corresponds to the existing `BackwardPhase1Timing` in sddp/mod.rs.
#[derive(Debug, Clone, Copy, Default)]
pub struct BackwardStageTiming {
    /// Time spent in model preprocessing for this stage.
    pub model_preprocessing: Duration,

    /// Time spent in solver for this stage.
    pub solver: Duration,

    /// Time spent in model postprocessing for this stage.
    pub model_postprocessing: Duration,

    /// Number of solver calls (branching scenarios solved).
    pub solver_calls: usize,
}

impl BackwardStageTiming {
    /// Add timing from another stage.
    #[inline]
    pub fn add(&mut self, other: &BackwardStageTiming) {
        self.model_preprocessing += other.model_preprocessing;
        self.solver += other.solver;
        self.model_postprocessing += other.model_postprocessing;
        self.solver_calls += other.solver_calls;
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

    #[test]
    fn test_trajectory_timing_default() {
        let timing = TrajectoryTiming::default();
        assert_eq!(timing.model_preprocessing, Duration::ZERO);
        assert_eq!(timing.solver, Duration::ZERO);
        assert_eq!(timing.model_postprocessing, Duration::ZERO);
        assert_eq!(timing.solver_calls, 0);
    }

    #[test]
    fn test_trajectory_timing_add() {
        let mut timing1 = TrajectoryTiming {
            model_preprocessing: Duration::from_millis(100),
            solver: Duration::from_millis(200),
            model_postprocessing: Duration::from_millis(50),
            solver_calls: 5,
        };

        let timing2 = TrajectoryTiming {
            model_preprocessing: Duration::from_millis(50),
            solver: Duration::from_millis(100),
            model_postprocessing: Duration::from_millis(25),
            solver_calls: 3,
        };

        timing1.add(&timing2);

        assert_eq!(timing1.model_preprocessing, Duration::from_millis(150));
        assert_eq!(timing1.solver, Duration::from_millis(300));
        assert_eq!(timing1.model_postprocessing, Duration::from_millis(75));
        assert_eq!(timing1.solver_calls, 8);
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
    fn test_backward_stage_timing_default() {
        let timing = BackwardStageTiming::default();
        assert_eq!(timing.model_preprocessing, Duration::ZERO);
        assert_eq!(timing.solver, Duration::ZERO);
        assert_eq!(timing.model_postprocessing, Duration::ZERO);
        assert_eq!(timing.solver_calls, 0);
    }

    #[test]
    fn test_backward_stage_timing_add() {
        let mut timing1 = BackwardStageTiming {
            model_preprocessing: Duration::from_millis(100),
            solver: Duration::from_millis(200),
            model_postprocessing: Duration::from_millis(50),
            solver_calls: 10,
        };

        let timing2 = BackwardStageTiming {
            model_preprocessing: Duration::from_millis(50),
            solver: Duration::from_millis(100),
            model_postprocessing: Duration::from_millis(25),
            solver_calls: 5,
        };

        timing1.add(&timing2);

        assert_eq!(timing1.model_preprocessing, Duration::from_millis(150));
        assert_eq!(timing1.solver, Duration::from_millis(300));
        assert_eq!(timing1.model_postprocessing, Duration::from_millis(75));
        assert_eq!(timing1.solver_calls, 15);
    }
}
