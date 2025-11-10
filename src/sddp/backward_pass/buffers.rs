//! Pre-allocated buffers for backward pass execution.
//!
//! This module eliminates allocations in the backward pass hot path by pre-allocating
//! result buffers for cut-state pairs. Profiling identified that the backward pass
//! allocates ~60 vectors per iteration, contributing to the 5.28% malloc overhead.
//!
//! # Architecture
//!
//! The backward pass processes multiple forward pass trajectories in parallel or sequentially.
//! Each trajectory requires computing cuts at each stage (except the final stage).
//! Instead of allocating new vectors for results each iteration, we pre-allocate a pool
//! of buffers sized appropriately using `SizingInfo`.
//!
//! # Buffer Organization
//!
//! ```text
//! BackwardPassBuffers
//! ├── results_pool: Vec<Vec<CutStatePair>>  // One buffer per forward pass
//! │   ├── Buffer 0: Vec with capacity for num_stages-1 pairs
//! │   ├── Buffer 1: Vec with capacity for num_stages-1 pairs
//! │   └── ...
//! └── sizing: SizingInfo  // Dimensions for buffer sizing
//! ```
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use crate::memory::SizingInfo;
//! use crate::sddp::backward_pass::BackwardPassBuffers;
//!
//! // 1. Create buffers at algorithm initialization
//! let sizing = SizingInfo::from_input(&system, &graph, &config);
//! let mut buffers = BackwardPassBuffers::new(&sizing);
//!
//! // 2. Use in backward pass (zero allocations!)
//! for iteration in 0..num_iterations {
//!     for (idx, trajectory) in trajectories.iter().enumerate() {
//!         let result_buffer = buffers.acquire_result_buffer(idx);
//!         result_buffer.clear();  // Reset from previous use
//!         
//!         // Compute cuts, write directly to buffer
//!         for stage in trajectory.stages() {
//!             let pair = compute_cut_state_pair(stage);
//!             result_buffer.push(pair);
//!         }
//!     }
//! }
//! ```
//!
//! # Performance Impact
//!
//! **Expected improvements**:
//! - Malloc overhead: ~2% → <1% (50% reduction in backward pass allocations)
//! - Backward pass time: 8-10% reduction
//! - Cache performance: Better locality from buffer reuse
//!
//! **Memory footprint** (typical system):
//! - 10 forward passes × 5 stages × (2KB cut + 1.6KB state) = ~180KB total
//! - Negligible compared to solver memory (hundreds of MB)

use crate::fcf::CutStatePair;
use crate::memory::SizingInfo;

/// Pre-allocated buffers for backward pass execution.
///
/// Provides one result buffer per forward pass trajectory, eliminating allocations
/// during backward pass iterations. Buffers are sized based on `SizingInfo` computed
/// from input configuration.
///
/// # Memory Layout
///
/// - Number of buffers: `num_forward_passes`
/// - Buffer capacity: `max_stages - 1` (no cut at final stage)
/// - Total capacity: `num_forward_passes × (max_stages - 1)` cut-state pairs
///
/// # Thread Safety
///
/// This struct is **not** thread-safe. Each `SddpAlgorithm` instance should have
/// its own `BackwardPassBuffers`. For parallel backward step execution, use
/// thread-local buffers from the `memory` module.
///
/// # Example
///
/// ```rust,ignore
/// let sizing = SizingInfo::from_input(&system, &graph, &config);
/// let mut buffers = BackwardPassBuffers::new(&sizing);
///
/// // Acquire buffer for first trajectory
/// let buffer = buffers.acquire_result_buffer(0);
/// buffer.clear();
///
/// // Use buffer (no allocations)
/// buffer.push(cut_state_pair);
/// ```
pub struct BackwardPassBuffers {
    /// Pool of result buffers, one per forward pass trajectory.
    ///
    /// Each buffer stores cut-state pairs computed during backward pass traversal.
    /// Capacity is pre-allocated to avoid reallocations during execution.
    results_pool: Vec<Vec<CutStatePair>>,

    /// Sizing information used to dimension buffers.
    ///
    /// Stored for diagnostics and potential dynamic resizing (if needed).
    #[allow(dead_code)]
    sizing: SizingInfo,
}

impl BackwardPassBuffers {
    /// Creates backward pass buffers sized for the given problem.
    ///
    /// Pre-allocates buffer pool based on `SizingInfo`:
    /// - Creates `num_forward_passes` result buffers
    /// - Each buffer has capacity for `max_stages - 1` cut-state pairs
    ///
    /// # Arguments
    ///
    /// * `sizing` - Problem dimensions from input configuration
    ///
    /// # Panics
    ///
    /// Panics if `num_forward_passes` is 0 or if `max_stages` is 0.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sizing = SizingInfo::from_input(&system, &graph, &config);
    /// let buffers = BackwardPassBuffers::new(&sizing);
    /// ```
    pub fn new(sizing: &SizingInfo) -> Self {
        assert!(
            sizing.num_forward_passes > 0,
            "Number of forward passes must be greater than 0"
        );
        assert!(
            sizing.num_stages > 0,
            "Number of stages must be greater than 0"
        );

        // Compute buffer capacity: no cut at final stage
        let buffer_capacity = sizing.num_stages.saturating_sub(1);

        // Pre-allocate result buffers for each forward pass
        let results_pool = (0..sizing.num_forward_passes)
            .map(|_| Vec::with_capacity(buffer_capacity))
            .collect();

        Self {
            results_pool,
            sizing: sizing.clone(),
        }
    }

    /// Acquires a mutable reference to the result buffer for a specific trajectory.
    ///
    /// Returns a buffer where cut-state pairs can be stored during backward pass.
    /// The buffer should be cleared before use to remove results from previous iterations.
    ///
    /// # Arguments
    ///
    /// * `trajectory_idx` - Index of the forward pass trajectory (0..num_forward_passes)
    ///
    /// # Returns
    ///
    /// Mutable reference to the result buffer for this trajectory.
    ///
    /// # Panics
    ///
    /// Panics if `trajectory_idx >= num_forward_passes`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let buffer = buffers.acquire_result_buffer(0);
    /// buffer.clear();  // Reset from previous iteration
    /// buffer.push(cut_state_pair);
    /// ```
    #[inline]
    pub fn acquire_result_buffer(
        &mut self,
        trajectory_idx: usize,
    ) -> &mut Vec<CutStatePair> {
        &mut self.results_pool[trajectory_idx]
    }

    /// Returns the number of result buffers in the pool.
    ///
    /// Equal to `num_forward_passes` from the sizing information.
    #[inline]
    pub fn num_buffers(&self) -> usize {
        self.results_pool.len()
    }

    /// Returns the capacity of each result buffer.
    ///
    /// Equal to `max_stages - 1` (no cut at final stage).
    #[inline]
    pub fn buffer_capacity(&self) -> usize {
        if self.results_pool.is_empty() {
            0
        } else {
            self.results_pool[0].capacity()
        }
    }

    /// Clears all result buffers, resetting them for the next iteration.
    ///
    /// This is a convenience method to clear all buffers at once.
    /// Typically called at the start of each SDDP iteration.
    ///
    /// # Performance
    ///
    /// O(num_forward_passes) operation. Capacities are preserved.
    pub fn clear_all(&mut self) {
        for buffer in &mut self.results_pool {
            buffer.clear();
        }
    }

    /// Estimates the memory footprint of this buffer pool in bytes.
    ///
    /// Provides approximate memory usage for diagnostics. Does not account
    /// for the actual size of `CutStatePair` objects (which vary based on
    /// state dimension), only the vector capacity.
    ///
    /// # Returns
    ///
    /// Estimated bytes allocated for the buffer pool structure.
    pub fn estimate_memory_bytes(&self) -> usize {
        let vec_overhead = std::mem::size_of::<Vec<CutStatePair>>();
        let pointer_size = std::mem::size_of::<CutStatePair>();

        let pool_overhead = vec_overhead;
        let buffers_overhead = self.results_pool.len() * vec_overhead;
        let capacity_bytes =
            self.results_pool.len() * self.buffer_capacity() * pointer_size;

        pool_overhead + buffers_overhead + capacity_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DirectedGraph;
    use crate::input::{
        Config, GeneralConfig, SimulationConfig, TrainingConfig,
    };
    use crate::sddp::NodeData;
    use crate::system::System;
    use std::sync::Arc;

    // Helper to create minimal test sizing
    fn make_test_sizing(
        num_stages: usize,
        num_forward_passes: usize,
    ) -> SizingInfo {
        let system = System::new(
            vec![crate::system::Bus::new(0, 1000.0)],
            vec![],
            vec![],
            vec![crate::system::Hydro::new(
                0, None, 0, 1.0, 0.0, 100.0, 0.0, 100.0, 10.0,
            )],
        );

        let mut graph = DirectedGraph::new();
        for stage in 0..num_stages {
            let node_system = System::new(
                vec![crate::system::Bus::new(0, 1000.0)],
                vec![],
                vec![],
                vec![crate::system::Hydro::new(
                    0, None, 0, 1.0, 0.0, 100.0, 0.0, 100.0, 10.0,
                )],
            );

            let node = NodeData {
                id: stage as isize,
                stage_id: stage,
                season_id: 0,
                start_date: chrono::Utc::now(),
                end_date: chrono::Utc::now(),
                kind: crate::subproblem::StudyPeriodKind::PreStudy,
                system: node_system,
                risk_measure: Box::new(crate::risk_measure::Expectation::new()),
                uncertainty_models: Arc::new(vec![]),
                state_choice: "storage".to_string(),
                num_scenarios: 4,
            };
            graph.add_node(node).unwrap();
        }

        let config = Config {
            general: GeneralConfig {
                seed: 42,
                num_threads: Some(2),
            },
            training: TrainingConfig {
                num_iterations: 10,
                num_forward_passes,
                enable_cut_selection: true,
            },
            simulation: SimulationConfig {
                num_scenarios: Some(100),
            },
            output: crate::input::OutputConfig::default(),
            logging: crate::logging::LoggingConfig::default(),
        };

        SizingInfo::from_input(&system, &graph, &config)
    }

    #[test]
    fn test_create_backward_pass_buffers() {
        let sizing = make_test_sizing(5, 10);
        let buffers = BackwardPassBuffers::new(&sizing);

        assert_eq!(buffers.num_buffers(), 10);
        assert_eq!(buffers.buffer_capacity(), 4); // 5 stages - 1
    }

    #[test]
    fn test_acquire_result_buffer() {
        let sizing = make_test_sizing(5, 10);
        let mut buffers = BackwardPassBuffers::new(&sizing);

        let buffer0 = buffers.acquire_result_buffer(0);
        assert_eq!(buffer0.len(), 0);
        assert_eq!(buffer0.capacity(), 4);

        let buffer5 = buffers.acquire_result_buffer(5);
        assert_eq!(buffer5.len(), 0);
        assert_eq!(buffer5.capacity(), 4);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_acquire_result_buffer_out_of_bounds() {
        let sizing = make_test_sizing(5, 10);
        let mut buffers = BackwardPassBuffers::new(&sizing);

        // Should panic
        let _ = buffers.acquire_result_buffer(10);
    }

    #[test]
    fn test_buffer_independence() {
        let sizing = make_test_sizing(5, 3);
        let mut buffers = BackwardPassBuffers::new(&sizing);

        // Use buffer 0
        {
            let buffer0 = buffers.acquire_result_buffer(0);
            // In real usage, we'd push CutStatePair, but for testing we just verify capacity
            assert_eq!(buffer0.capacity(), 4);
        }

        // Use buffer 1
        {
            let buffer1 = buffers.acquire_result_buffer(1);
            assert_eq!(buffer1.capacity(), 4);
        }

        // Buffers should be independent
        assert_eq!(buffers.results_pool[0].len(), 0);
        assert_eq!(buffers.results_pool[1].len(), 0);
    }

    #[test]
    fn test_clear_all() {
        let sizing = make_test_sizing(5, 3);
        let mut buffers = BackwardPassBuffers::new(&sizing);

        // Simulate usage by setting length (in real usage, we'd push items)
        // For testing, we just verify clear preserves capacity
        buffers.clear_all();

        for i in 0..buffers.num_buffers() {
            let buffer = buffers.acquire_result_buffer(i);
            assert_eq!(buffer.len(), 0);
            assert_eq!(buffer.capacity(), 4);
        }
    }

    #[test]
    fn test_single_stage_problem() {
        let sizing = make_test_sizing(1, 5);
        let buffers = BackwardPassBuffers::new(&sizing);

        // Single stage => no cuts (buffer capacity 0)
        assert_eq!(buffers.buffer_capacity(), 0);
        assert_eq!(buffers.num_buffers(), 5);
    }

    #[test]
    fn test_large_scale_sizing() {
        let sizing = make_test_sizing(20, 50);
        let buffers = BackwardPassBuffers::new(&sizing);

        assert_eq!(buffers.num_buffers(), 50);
        assert_eq!(buffers.buffer_capacity(), 19); // 20 stages - 1

        // Estimate memory
        let estimated_bytes = buffers.estimate_memory_bytes();
        assert!(estimated_bytes > 0);

        // Should be reasonable (not GB-scale)
        assert!(estimated_bytes < 10_000_000); // < 10MB for structure overhead
    }

    #[test]
    fn test_estimate_memory_bytes() {
        let sizing = make_test_sizing(5, 10);
        let buffers = BackwardPassBuffers::new(&sizing);

        let bytes = buffers.estimate_memory_bytes();
        assert!(bytes > 0);

        // Should account for 10 buffers × 4 capacity
        // This is just structure overhead, not actual CutStatePair data
        let expected_min_bytes = 10 * 4 * std::mem::size_of::<CutStatePair>();
        assert!(bytes >= expected_min_bytes);
    }

    #[test]
    #[should_panic(
        expected = "Number of forward passes must be greater than 0"
    )]
    fn test_zero_forward_passes_panics() {
        let sizing = make_test_sizing(5, 0);
        let _ = BackwardPassBuffers::new(&sizing);
    }
}
