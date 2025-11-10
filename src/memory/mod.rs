//! Memory management and buffer pre-allocation infrastructure.
//!
//! This module provides efficient memory management for the SDDP algorithm by
//! pre-allocating buffers and reusing them across iterations. Profiling showed
//! that malloc/memset operations consume 5.28% of CPU time. By computing buffer
//! sizes upfront and reusing buffers, we can reduce this overhead to <2%.
//!
//! # Components
//!
//! - [`SizingInfo`]: Computes all buffer dimensions from input configuration
//! - [`NodeSizing`]: Per-node sizing information
//! - [`MemoryBreakdown`]: Detailed memory component breakdown
//! - [`Buffer`]: Generic pre-allocated buffer with reuse semantics
//! - [`BufferPool`]: Buffer pool for cycling and reuse
//! - [`ThreadLocalBuffers`]: Thread-local storage for parallel execution
//!
//! # Architecture
//!
//! The memory module is based on a key insight: **all data structures in POWE.RS
//! have deterministic sizes computable from input files**. Once we know:
//! - Number of hydros, thermals, buses, lines (from system.json)
//! - Number of stages and scenarios (from graph.json and config.json)
//! - AR orders for uncertainty models (from recourse.json)
//!
//! We can compute exact buffer sizes for:
//! - Forward pass trajectories
//! - Backward pass cut computation
//! - Subproblem realizations
//! - Thread-local working buffers
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use powers_rs::memory::SizingInfo;
//!
//! // 1. Compute sizing from input at startup
//! let sizing = SizingInfo::from_input(
//!     &system,
//!     &graph,
//!     &config,
//! );
//!
//! // 2. Log sizing summary for diagnostics
//! sizing.log_summary();
//!
//! // 3. Get detailed breakdown
//! let breakdown = sizing.estimate_memory_detailed();
//! println!("Cuts: {} MB", breakdown.cuts / 1_000_000);
//!
//! // 4. Use sizing to pre-allocate buffers
//! let backward_buffers = BackwardPassBuffers::new(&sizing);
//! let forward_buffers = ForwardPassBuffers::new(&sizing);
//!
//! // 5. Reuse buffers across iterations (zero allocations in hot path)
//! for iteration in 0..num_iterations {
//!     backward_pass(&mut backward_buffers, &sizing);
//!     forward_pass(&mut forward_buffers, &sizing);
//! }
//! ```
//!
//! # Performance Impact
//!
//! Based on profiling data from PROFILING_ANALYSIS.md:
//!
//! | Metric | Before | After (Target) | Improvement |
//! |--------|--------|----------------|-------------|
//! | Runtime | 34.0s | ~28.9s | -15% to -20% |
//! | Malloc overhead | 5.28% | <2% | -60% |
//! | Allocations/iter | ~60 | ~1 | -98% |
//! | Peak memory | 2.4GB | ~2.6GB | +8% (acceptable) |
//!
//! # Related Documentation
//!
//! - `PERFORMANCE_IMPLEMENTATION_PLAN.md`: Detailed implementation strategy
//! - `PROFILING_ANALYSIS.md`: Profiling data and bottleneck identification
//! - `PERFORMANCE_REFACTORING_PLAN.md`: Overall performance optimization roadmap

pub mod buffers;
pub mod deep_sizing;
pub mod sizing;

pub use buffers::{
    initialize_thread_local_buffers, with_thread_buffers, Buffer, BufferPool,
    ThreadLocalBuffers,
};
pub use deep_sizing::DeepSizeEstimate;
pub use sizing::{MemoryBreakdown, NodeSizing, SizingInfo};

#[cfg(test)]
mod integration_tests {
    use super::*;

    // Helper to create minimal test system
    fn make_test_system() -> (
        crate::system::System,
        crate::graph::DirectedGraph<crate::sddp::NodeData>,
        crate::input::Config,
    ) {
        use crate::graph::DirectedGraph;
        use crate::input::{
            Config, GeneralConfig, SimulationConfig, TrainingConfig,
        };
        use crate::sddp::NodeData;
        use crate::system::System;
        use std::sync::Arc;

        let system = System::new(
            vec![crate::system::Bus::new(0, 1000.0)],
            vec![],
            vec![],
            vec![crate::system::Hydro::new(
                0, None, 0, 1.0, 0.0, 100.0, 0.0, 100.0, 10.0,
            )],
        );

        let mut graph = DirectedGraph::new();
        let node_system = System::new(
            vec![crate::system::Bus::new(0, 1000.0)],
            vec![],
            vec![],
            vec![crate::system::Hydro::new(
                0, None, 0, 1.0, 0.0, 100.0, 0.0, 100.0, 10.0,
            )],
        );

        let node = NodeData {
            id: 0,
            stage_id: 0,
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

        let config = Config {
            general: GeneralConfig {
                seed: 42,
                num_threads: Some(2),
            },
            training: TrainingConfig {
                num_iterations: 10,
                num_forward_passes: 4,
                enable_cut_selection: true,
            },
            simulation: SimulationConfig {
                num_scenarios: Some(100),
            },
            output: crate::input::OutputConfig::default(),
            logging: crate::logging::LoggingConfig::default(),
        };

        (system, graph, config)
    }

    #[test]
    fn test_complete_workflow() {
        // This test demonstrates the complete memory management workflow

        let (system, graph, config) = make_test_system();

        // 1. Compute sizing from input
        let sizing = SizingInfo::from_input(&system, &graph, &config);

        // 2. Verify sizing information is available
        assert!(sizing.max_state_dimension > 0);
        assert!(sizing.max_subproblem_vars > 0);

        // 3. Create buffer pool
        let mut pool = BufferPool::<f64>::new(4, sizing.max_subproblem_vars);

        // 4. Use buffers across iterations
        for i in 0..10 {
            let buffer = pool.acquire(i);
            buffer.resize(sizing.max_subproblem_vars / 2);
            buffer.reset();
            assert_eq!(buffer.len(), sizing.max_subproblem_vars / 2);
        }

        // 5. Get memory breakdown
        let breakdown = sizing.estimate_memory_detailed();
        assert!(breakdown.total > 0);
        assert_eq!(
            breakdown.total,
            breakdown.cuts
                + breakdown.states
                + breakdown.trajectories
                + breakdown.thread_buffers
        );
    }

    #[test]
    fn test_sizing_info_public_api() {
        let (system, graph, config) = make_test_system();
        let sizing = SizingInfo::from_input(&system, &graph, &config);

        // Test public accessor methods
        assert!(sizing.node(0).is_some());
        assert!(sizing.state_dimension_for_node(0).is_some());

        // We have 1 node with "storage" state choice
        assert!(!sizing.nodes_with_state_choice("storage").is_empty());
        assert!(sizing.nodes_with_state_choice("nonexistent").is_empty());

        // Test aggregate statistics
        assert!(sizing.max_state_dimension >= sizing.min_state_dimension);
        assert!(
            sizing.avg_state_dimension >= sizing.min_state_dimension as f64
        );
        assert!(
            sizing.avg_state_dimension <= sizing.max_state_dimension as f64
        );
    }

    #[test]
    fn test_buffer_public_api() {
        let mut buffer = Buffer::<f64>::with_capacity(100);

        // Test all public methods
        assert_eq!(buffer.capacity(), 100);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());

        buffer.resize(50);
        assert_eq!(buffer.len(), 50);
        assert!(!buffer.is_empty());

        buffer.as_mut_slice()[0] = 42.0;
        assert_eq!(buffer.as_slice()[0], 42.0);

        buffer.reset();
        assert_eq!(buffer.as_slice()[0], 0.0);

        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 100);
    }

    #[test]
    fn test_buffer_pool_public_api() {
        let mut pool = BufferPool::<f64>::new(5, 100);

        // Test public methods
        assert_eq!(pool.len(), 5);
        assert!(!pool.is_empty());

        // Test acquire and cycling
        let buf0 = pool.acquire(0);
        buf0.resize(10);

        let buf1 = pool.acquire(1);
        buf1.resize(20);

        // Cycling works
        let buf5 = pool.acquire(5);
        assert_eq!(buf5.len(), 10); // Same as buf0
    }

    #[test]
    fn test_thread_local_public_api() {
        let (system, graph, config) = make_test_system();
        let sizing = SizingInfo::from_input(&system, &graph, &config);

        // Initialize thread locals
        initialize_thread_local_buffers(&sizing);

        // Use thread locals
        with_thread_buffers(|buffers| {
            assert!(buffers.realization_buffer.capacity() > 0);
            assert!(buffers.gradient_buffer.capacity() > 0);
            assert!(buffers.state_buffer.capacity() > 0);
            assert!(buffers.lag_buffer.capacity() > 0);
            assert!(buffers.cut_eval_buffer.capacity() > 0);

            buffers.realization_buffer.resize(10);
            buffers.reset_all();
            assert_eq!(buffers.realization_buffer.as_slice()[0], 0.0);
        });
    }

    #[test]
    fn test_memory_breakdown_public_api() {
        let (system, graph, config) = make_test_system();
        let sizing = SizingInfo::from_input(&system, &graph, &config);

        let breakdown = sizing.estimate_memory_detailed();

        // Test all public fields
        assert!(breakdown.cuts > 0);
        assert!(breakdown.states > 0);
        assert!(breakdown.trajectories > 0);
        assert!(breakdown.thread_buffers > 0);
        assert!(breakdown.total > 0);

        // Verify invariant
        assert_eq!(
            breakdown.total,
            breakdown.cuts
                + breakdown.states
                + breakdown.trajectories
                + breakdown.thread_buffers
        );
    }

    #[test]
    fn test_parallel_thread_local_integration() {
        use rayon::prelude::*;

        let (system, graph, config) = make_test_system();
        let sizing = SizingInfo::from_input(&system, &graph, &config);

        initialize_thread_local_buffers(&sizing);

        // Parallel execution
        let results: Vec<_> = (0..10)
            .into_par_iter()
            .map(|i| {
                with_thread_buffers(|buffers| {
                    buffers.realization_buffer.resize(5);
                    buffers.realization_buffer.as_mut_slice()[0] = i as f64;
                    buffers.realization_buffer.as_slice()[0]
                })
            })
            .collect();

        // Verify independence
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(result, i as f64);
        }
    }
}
