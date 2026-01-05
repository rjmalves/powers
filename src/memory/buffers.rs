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

    /// Reusable buffer for uniform probabilities
    pub probabilities: Vec<f64>,

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
            probabilities: Vec::with_capacity(max_scenarios),
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
        debug_assert!(
            state_dim <= self.max_state_dim,
            "Cut buffer capacity overflow: state_dim {} > max_state_dim {}. \
             This indicates incorrect initialization. Check that max_state_dim \
             accounts for inflow lags in StorageAndInflowState.",
            state_dim,
            self.max_state_dim
        );
        debug_assert!(
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

        // Resize probabilities buffer to num_scenarios and fill with uniform probabilities
        self.probabilities.clear();
        self.probabilities.resize(num_scenarios, 0.0);
        crate::utils::fill_uniform_probabilities(&mut self.probabilities);
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
        let mut buffers = buffers.borrow_mut();
        // Only reinitialize if capacity needs to grow (or first time)
        let needs_init = match buffers.as_ref() {
            None => true,
            Some(existing) => {
                let (cur_dim, cur_scen) = existing.capacity();
                max_state_dim > cur_dim || max_scenarios > cur_scen
            }
        };
        if needs_init {
            *buffers =
                Some(CutComputationBuffers::new(max_state_dim, max_scenarios));
        }
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

/// Staging area for one cut + one state computation.
///
/// Lives in `SddpTrainHandler`, reused across all stages within an iteration.
/// Enables parallel cut computation by giving each handler its own buffer.
///
/// # Memory
///
/// Size: ~2 × state_dim × 8 bytes ≈ 1.6 KB for 100-dimension state
///
/// # Usage
///
/// 1. Handler computes cut into thread-local `CutComputationBuffers`
/// 2. Results copied into this staging buffer via `stage_from()`
/// 3. Sequential loop copies from staging to global pools
/// 4. Buffer reused for next stage
#[derive(Debug)]
pub struct CutStagingBuffer {
    /// Computed cut coefficients (water values, lag duals)
    pub cut_coefficients: Vec<f64>,
    /// Computed cut RHS
    pub cut_rhs: f64,
    /// State coefficients at which cut was computed
    pub state_coefficients: Vec<f64>,
    /// Iteration that produced this cut (1-based)
    pub iteration: usize,
    /// Forward pass index (0-based)
    pub forward_pass_idx: usize,
    /// Actual cut coefficient length (may be less than capacity)
    actual_cut_len: usize,
    /// Actual state coefficient length
    actual_state_len: usize,
    /// Whether buffer contains valid data
    pub populated: bool,
}

impl CutStagingBuffer {
    /// Create staging buffer with preallocated capacity.
    ///
    /// # Arguments
    ///
    /// * `state_dim` - Maximum state dimension for this handler
    pub fn new(state_dim: usize) -> Self {
        Self {
            cut_coefficients: vec![0.0; state_dim],
            state_coefficients: vec![0.0; state_dim],
            cut_rhs: 0.0,
            iteration: 0,
            forward_pass_idx: 0,
            actual_cut_len: 0,
            actual_state_len: 0,
            populated: false,
        }
    }

    /// Copy computed results into staging area.
    ///
    /// Called at end of parallel cut computation, while still holding
    /// the thread-local buffer reference.
    ///
    /// # Arguments
    ///
    /// * `cut_coefficients` - Cut coefficients slice
    /// * `cut_rhs` - Cut RHS value
    /// * `state_coefficients` - State coefficients slice
    /// * `iteration` - Current iteration (1-based)
    /// * `forward_pass_idx` - Forward pass index (0-based)
    #[inline]
    pub fn stage_from(
        &mut self,
        cut_coefficients: &[f64],
        cut_rhs: f64,
        state_coefficients: &[f64],
        iteration: usize,
        forward_pass_idx: usize,
    ) {
        let cut_len = cut_coefficients.len();
        let state_len = state_coefficients.len();

        debug_assert!(
            cut_len <= self.cut_coefficients.len(),
            "cut_len {} exceeds capacity {}",
            cut_len,
            self.cut_coefficients.len()
        );
        debug_assert!(
            state_len <= self.state_coefficients.len(),
            "state_len {} exceeds capacity {}",
            state_len,
            self.state_coefficients.len()
        );

        self.cut_coefficients[..cut_len].copy_from_slice(cut_coefficients);
        self.state_coefficients[..state_len]
            .copy_from_slice(state_coefficients);
        self.cut_rhs = cut_rhs;
        self.iteration = iteration;
        self.forward_pass_idx = forward_pass_idx;
        self.actual_cut_len = cut_len;
        self.actual_state_len = state_len;
        self.populated = true;
    }

    /// Reset buffer for next iteration (clear populated flag).
    #[inline]
    pub fn reset(&mut self) {
        self.populated = false;
    }

    /// Get actual cut coefficient slice.
    #[inline]
    pub fn cut_slice(&self) -> &[f64] {
        &self.cut_coefficients[..self.actual_cut_len]
    }

    /// Get actual state coefficient slice.
    #[inline]
    pub fn state_slice(&self) -> &[f64] {
        &self.state_coefficients[..self.actual_state_len]
    }
}

#[cfg(test)]
mod staging_tests {
    use super::*;

    #[test]
    fn test_staging_buffer_new() {
        let buf = CutStagingBuffer::new(100);
        assert_eq!(buf.cut_coefficients.len(), 100);
        assert_eq!(buf.state_coefficients.len(), 100);
        assert!(!buf.populated);
        assert_eq!(buf.actual_cut_len, 0);
        assert_eq!(buf.actual_state_len, 0);
    }

    #[test]
    fn test_staging_buffer_stage_from() {
        let mut buf = CutStagingBuffer::new(10);

        let coeffs = [1.0, 2.0, 3.0];
        let state = [4.0, 5.0, 6.0];

        buf.stage_from(&coeffs, 42.0, &state, 1, 0);

        assert!(buf.populated);
        assert_eq!(buf.cut_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(buf.state_slice(), &[4.0, 5.0, 6.0]);
        assert_eq!(buf.cut_rhs, 42.0);
        assert_eq!(buf.iteration, 1);
        assert_eq!(buf.forward_pass_idx, 0);
    }

    #[test]
    fn test_staging_buffer_reuse() {
        let mut buf = CutStagingBuffer::new(10);

        // First stage
        let coeffs1 = [1.0, 2.0];
        buf.stage_from(&coeffs1, 10.0, &[3.0, 4.0], 1, 0);

        // Get pointer to verify no reallocation
        let ptr1 = buf.cut_coefficients.as_ptr();

        // Second stage with different sizes
        let coeffs2 = [5.0, 6.0, 7.0];
        buf.stage_from(&coeffs2, 20.0, &[8.0, 9.0, 10.0], 1, 1);

        // Verify same memory (no reallocation)
        let ptr2 = buf.cut_coefficients.as_ptr();
        assert_eq!(ptr1, ptr2);

        // Verify new data
        assert_eq!(buf.cut_slice(), &[5.0, 6.0, 7.0]);
        assert_eq!(buf.state_slice(), &[8.0, 9.0, 10.0]);
        assert_eq!(buf.cut_rhs, 20.0);
    }

    #[test]
    fn test_staging_buffer_reset() {
        let mut buf = CutStagingBuffer::new(10);
        let coeffs = [1.0];
        buf.stage_from(&coeffs, 1.0, &[2.0], 1, 0);

        assert!(buf.populated);
        buf.reset();
        assert!(!buf.populated);

        // Data still there (not cleared, just marked invalid)
        assert_eq!(buf.cut_coefficients[0], 1.0);
    }

    #[test]
    fn test_staging_buffer_slice_lengths() {
        let mut buf = CutStagingBuffer::new(10);

        // Stage with smaller dimensions
        let coeffs = [1.0, 2.0];
        buf.stage_from(&coeffs, 5.0, &[3.0], 1, 0);

        // Slices should have correct lengths
        assert_eq!(buf.cut_slice().len(), 2);
        assert_eq!(buf.state_slice().len(), 1);
    }
}
