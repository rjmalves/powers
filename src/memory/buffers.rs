//! Cut computation buffer management for zero-allocation hot paths.
//!
//! This module provides specialized buffers for cut coefficient computation
//! in the SDDP backward pass. By pre-allocating buffers and reusing them
//! across cut computations, we eliminate allocation overhead in the hot path.
//!
//! # Components
//!
//! - [`CutComputationBuffers`]: Thread-local buffers for cut computation
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
//!
//! # Performance
//!
//! - **Zero allocations** in hot paths after initial setup
//! - **Cache-friendly**: Reusing same memory improves cache hit rates
//! - **Thread-safe**: Thread-local storage eliminates contention

use std::cell::RefCell;

/// Specialized buffers for cut coefficient computation in evaluate_cut hot path.
///
/// These buffers eliminate allocations per training run by reusing
/// pre-allocated vectors across cut computations. Thread-local storage ensures
/// thread-safety in parallel backward pass execution.
///
/// # Performance Impact
///
/// **Before**: Each cut computation allocated:
/// - 1× cut_coefficients Vec
/// - N× contribution Vecs (N = number of scenarios)
///
/// **After**: Buffers allocated once per thread, reused across all cuts
///
/// # Capacity Enforcement
///
/// After initialization, the buffers enforce capacity limits. If a cut
/// computation requests dimensions exceeding the preallocated capacity,
/// the buffers will panic with a clear error message. This ensures
/// 100% memory determinism in the hot path.
///
/// # Usage
///
/// ```rust,ignore
/// // Initialize before parallel execution
/// initialize_cut_buffers(max_state_dim, max_scenarios);
///
/// // Use in hot path
/// with_cut_buffers(|buffers| {
///     buffers.reset_for_cut(state_dim, num_scenarios);
///     // ... compute cut coefficients ...
///     // buffers.coefficients contains result
/// });
/// ```
pub struct CutComputationBuffers {
    /// Reusable buffer for final cut coefficients
    pub coefficients: Vec<f64>,

    /// Reusable buffers for contribution vectors (one per scenario)
    /// Pre-allocated to avoid inner vector allocations
    pub contributions_outer: Vec<Vec<f64>>,

    /// Reusable buffer for scenario costs (for risk measure adjustment)
    pub costs: Vec<f64>,

    /// Reusable buffer for objective contributions (one per scenario)
    pub objective_contributions: Vec<f64>,

    /// Maximum state dimension for capacity enforcement
    max_state_dim: usize,

    /// Maximum scenarios for capacity enforcement
    max_scenarios: usize,
}

impl CutComputationBuffers {
    /// Create new buffers with specified maximum capacities.
    ///
    /// Pre-allocates all inner vectors to eliminate allocations during
    /// cut computation. Capacity is based on worst-case problem dimensions.
    ///
    /// # Arguments
    ///
    /// * `max_state_dim` - Maximum state dimension across all nodes
    /// * `max_scenarios` - Maximum branching scenarios per node
    pub fn new(max_state_dim: usize, max_scenarios: usize) -> Self {
        // Pre-allocate outer vector and all inner vectors
        let mut contributions_outer = Vec::with_capacity(max_scenarios);
        for _ in 0..max_scenarios {
            contributions_outer.push(Vec::with_capacity(max_state_dim));
        }

        Self {
            coefficients: Vec::with_capacity(max_state_dim),
            contributions_outer,
            costs: Vec::with_capacity(max_scenarios),
            objective_contributions: Vec::with_capacity(max_scenarios),
            max_state_dim,
            max_scenarios,
        }
    }

    /// Reset buffers for a new cut computation.
    ///
    /// Clears existing data while preserving capacity. Panics if requested
    /// dimensions exceed preallocated capacity.
    ///
    /// # Arguments
    ///
    /// * `state_dim` - Actual state dimension for this cut
    /// * `num_scenarios` - Actual number of scenarios for this cut
    ///
    /// # Panics
    ///
    /// Panics if `state_dim > max_state_dim` or `num_scenarios > max_scenarios`.
    /// This indicates incorrect initialization - check that max_state_dim accounts
    /// for inflow lags in StorageAndInflowState.
    pub fn reset_for_cut(&mut self, state_dim: usize, num_scenarios: usize) {
        assert!(
            state_dim <= self.max_state_dim,
            "Cut buffer capacity overflow: state_dim {} > max_state_dim {}. \
             This indicates incorrect initialization. Check that max_state_dim \
             accounts for inflow lags in StorageAndInflowState.",
            state_dim,
            self.max_state_dim
        );
        assert!(
            num_scenarios <= self.max_scenarios,
            "Cut buffer capacity overflow: num_scenarios {} > max_scenarios {}. \
             This indicates incorrect initialization.",
            num_scenarios,
            self.max_scenarios
        );

        // Reset coefficient buffer
        self.coefficients.clear();
        self.coefficients.resize(state_dim, 0.0);

        // Clear existing inner vectors (preserve capacity)
        for contrib in self.contributions_outer.iter_mut().take(num_scenarios) {
            contrib.clear();
        }

        // Clear costs and objective_contributions buffers (preserve capacity)
        self.costs.clear();
        self.objective_contributions.clear();
    }

    /// Returns the preallocated capacity (max_state_dim, max_scenarios).
    #[inline]
    pub fn capacity(&self) -> (usize, usize) {
        (self.max_state_dim, self.max_scenarios)
    }
}

thread_local! {
    /// Thread-local storage for cut computation buffers.
    ///
    /// Each thread in Rayon's thread pool gets independent buffer instances,
    /// ensuring thread-safety without locks.
    static CUT_BUFFERS: RefCell<Option<CutComputationBuffers>> = const { RefCell::new(None) };
}

/// Initialize cut computation buffers for the current thread.
///
/// Must be called before using `with_cut_buffers`. For parallel execution,
/// call this in the main thread and use `rayon::broadcast()` to initialize
/// all worker threads.
///
/// # Arguments
///
/// * `max_state_dim` - Maximum state dimension (including inflow lags)
/// * `max_scenarios` - Maximum scenarios per node
///
/// # Example
///
/// ```rust,ignore
/// // Initialize main thread
/// initialize_cut_buffers(max_state_dim, max_scenarios);
///
/// // Initialize all Rayon worker threads
/// rayon::broadcast(|_| {
///     initialize_cut_buffers(max_state_dim, max_scenarios);
/// });
/// ```
pub fn initialize_cut_buffers(max_state_dim: usize, max_scenarios: usize) {
    CUT_BUFFERS.with(|buffers| {
        *buffers.borrow_mut() =
            Some(CutComputationBuffers::new(max_state_dim, max_scenarios));
    });
}

/// Execute a closure with access to thread-local cut computation buffers.
///
/// Provides mutable access to pre-allocated buffers for cut coefficient
/// computation. Panics if buffers have not been initialized.
///
/// # Thread Safety
///
/// Each thread has independent buffers via `thread_local!` storage.
/// No locks or synchronization needed.
///
/// # Panics
///
/// Panics if `initialize_cut_buffers()` has not been called for this thread.
///
/// # Example
///
/// ```rust,ignore
/// with_cut_buffers(|buffers| {
///     buffers.reset_for_cut(state_dim, num_scenarios);
///     
///     // Use pre-allocated buffers (zero allocations)
///     for (i, realization) in realizations.iter().enumerate() {
///         let contrib = &mut buffers.contributions_outer[i];
///         contrib.extend(realization.water_value.iter().map(|&v| prob * v));
///     }
///     
///     // Return computed cut
///     BendersCut::new(0, buffers.coefficients.clone(), rhs, iter, fp_idx)
/// })
/// ```
pub fn with_cut_buffers<F, R>(f: F) -> R
where
    F: FnOnce(&mut CutComputationBuffers) -> R,
{
    CUT_BUFFERS.with(|buffers| {
        let mut buffers = buffers.borrow_mut();

        match buffers.as_mut() {
            Some(b) => f(b),
            None => panic!(
                "Cut computation buffers not initialized! \
                 Call initialize_cut_buffers() before training."
            ),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut_buffers_initialization() {
        initialize_cut_buffers(10, 4);

        with_cut_buffers(|buffers| {
            assert_eq!(buffers.coefficients.capacity(), 10);
            assert_eq!(buffers.contributions_outer.capacity(), 4);
            assert_eq!(buffers.contributions_outer.len(), 4);
            assert_eq!(buffers.capacity(), (10, 4));
        });
    }

    #[test]
    fn test_cut_buffers_reset() {
        initialize_cut_buffers(10, 4);

        // First use
        with_cut_buffers(|buffers| {
            buffers.reset_for_cut(5, 3);
            buffers.coefficients[0] = 42.0;
            buffers.contributions_outer[0].push(1.0);
        });

        // Second use - buffers should be reset
        with_cut_buffers(|buffers| {
            buffers.reset_for_cut(5, 3);
            assert_eq!(buffers.coefficients[0], 0.0); // Reset to zero
            assert_eq!(buffers.coefficients.len(), 5);
            assert_eq!(buffers.coefficients.capacity(), 10); // Capacity preserved
            assert_eq!(buffers.contributions_outer[0].len(), 0); // Cleared
        });
    }

    #[test]
    #[should_panic(expected = "state_dim")]
    fn test_cut_buffers_overflow_state_dim() {
        initialize_cut_buffers(10, 4);
        with_cut_buffers(|buffers| {
            buffers.reset_for_cut(100, 4); // 100 > 10, should panic
        });
    }

    #[test]
    #[should_panic(expected = "num_scenarios")]
    fn test_cut_buffers_overflow_scenarios() {
        initialize_cut_buffers(10, 4);
        with_cut_buffers(|buffers| {
            buffers.reset_for_cut(10, 100); // 100 > 4, should panic
        });
    }

    #[test]
    #[should_panic(expected = "not initialized")]
    fn test_cut_buffers_uninitialized_panics() {
        CUT_BUFFERS.with(|b| *b.borrow_mut() = None);
        with_cut_buffers(|_| {}); // Should panic
    }

    #[test]
    fn test_cut_buffers_thread_local() {
        use rayon::prelude::*;

        // Initialize in main thread
        initialize_cut_buffers(10, 4);

        // Initialize all Rayon worker threads
        rayon::broadcast(|_| {
            initialize_cut_buffers(10, 4);
        });

        // Parallel execution - each thread gets independent buffers
        let results: Vec<_> = (0..8)
            .into_par_iter()
            .map(|i| {
                with_cut_buffers(|buffers| {
                    buffers.reset_for_cut(5, 4);
                    buffers.coefficients[0] = i as f64;
                    buffers.coefficients[0]
                })
            })
            .collect();

        // Each thread should have computed independently
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(result, i as f64);
        }
    }
}
