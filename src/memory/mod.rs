//! Cut computation buffer management for zero-allocation hot paths.
//!
//! This module provides efficient buffer management for the SDDP backward pass
//! cut computation. By pre-allocating buffers and reusing them across iterations,
//! we eliminate allocation overhead in the hot path.
//!
//! # Components
//!
//! - [`CutComputationBuffers`]: Thread-local buffers for cut coefficient computation
//! - [`initialize_cut_buffers`]: Initialize buffers with problem dimensions
//! - [`with_cut_buffers`]: Access thread-local buffers in hot path
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use powers_rs::memory::{initialize_cut_buffers, with_cut_buffers};
//!
//! // 1. Initialize once before training (main thread + Rayon workers)
//! initialize_cut_buffers(max_state_dim, max_scenarios);
//! rayon::broadcast(|_| {
//!     initialize_cut_buffers(max_state_dim, max_scenarios);
//! });
//!
//! // 2. Use in hot path (zero allocations)
//! with_cut_buffers(|buffers| {
//!     buffers.reset_for_cut(state_dim, num_scenarios);
//!     // ... compute cut coefficients ...
//! });
//! ```
//!
//! # Thread Safety
//!
//! Each thread has independent buffer instances via thread-local storage.
//! No synchronization or locking required during parallel execution.

pub mod buffers;

pub use buffers::{
    initialize_cut_buffers, with_cut_buffers, CutComputationBuffers,
    CutStagingBuffer,
};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_cut_buffers_workflow() {
        // Initialize buffers with problem dimensions
        initialize_cut_buffers(100, 10);

        // Use in simulated hot path
        with_cut_buffers(|buffers| {
            buffers.reset_for_cut(50, 5);
            assert_eq!(buffers.coefficients.len(), 50);
            buffers.coefficients[0] = 42.0;
            assert_eq!(buffers.coefficients[0], 42.0);
        });

        // Second use - reset should clear
        with_cut_buffers(|buffers| {
            buffers.reset_for_cut(50, 5);
            assert_eq!(buffers.coefficients[0], 0.0);
        });
    }

    #[test]
    fn test_parallel_cut_buffers() {
        use rayon::prelude::*;

        // Initialize main thread
        initialize_cut_buffers(100, 10);

        // Initialize all Rayon worker threads
        rayon::broadcast(|_| {
            initialize_cut_buffers(100, 10);
        });

        // Parallel execution - each thread gets independent buffers
        let results: Vec<_> = (0..10)
            .into_par_iter()
            .map(|i| {
                with_cut_buffers(|buffers| {
                    buffers.reset_for_cut(50, 5);
                    buffers.coefficients[0] = i as f64;
                    buffers.coefficients[0]
                })
            })
            .collect();

        // Verify each iteration got its own buffer value
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(result, i as f64);
        }
    }
}
