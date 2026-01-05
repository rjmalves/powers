//! Forward pass timing hierarchy.

use std::cell::Cell;
use std::time::Duration;

use super::output::ForwardTimingOutput;
use super::TrajectoryTiming;

/// Forward pass timing with hierarchical structure.
///
/// The forward pass has three phases:
/// 1. Preprocessing (sequential): SAA sampling
/// 2. Parallel execution: N trajectories solving in parallel
/// 3. Postprocessing (sequential): Detail capturing
///
/// # Example
///
/// ```ignore
/// use powers_rs::timing::{ForwardTiming, TimingGuard};
///
/// let timing = ForwardTiming::new(10); // 10 trajectories
///
/// // Preprocessing
/// {
///     let _guard = TimingGuard::new(&timing.preprocessing.saa_sampling);
///     // ... sample scenarios ...
/// }
///
/// // Parallel section (measured by wall clock)
/// {
///     let _guard = TimingGuard::new(&timing.parallel.wall);
///     // ... parallel trajectory execution ...
/// }
///
/// // Compute aggregates after parallel section
/// timing.parallel.compute_aggregates();
/// ```
pub struct ForwardTiming {
    /// Sequential preprocessing phase.
    pub preprocessing: ForwardPreprocessingTiming,

    /// Parallel trajectory execution.
    pub parallel: ForwardParallelTiming,

    /// Sequential postprocessing phase.
    pub postprocessing: ForwardPostprocessingTiming,

    /// Total forward pass time (wall clock).
    pub total: Cell<Duration>,
}

/// Forward pass preprocessing (sequential, before parallel section).
#[derive(Debug, Clone, Default)]
pub struct ForwardPreprocessingTiming {
    /// Time spent sampling scenarios from SAA tree.
    pub saa_sampling: Cell<Duration>,
}

/// Forward pass parallel section timing.
///
/// Stores BOTH wall time and individual trajectory times:
/// - `wall`: What we observe from outside the parallel section
/// - `trajectories`: Raw per-trajectory timing for statistical analysis
/// - Computed fields populated after parallel section completes
///
/// # Important
///
/// Call `compute_aggregates()` after the parallel section completes to
/// populate the computed fields (cpu_total, overhead, averages, max).
pub struct ForwardParallelTiming {
    /// Wall-clock time for entire parallel section.
    pub wall: Cell<Duration>,

    /// Individual trajectory timings (preallocated).
    pub trajectories: Vec<TrajectoryTiming>,

    // Computed fields (populated by compute_aggregates):
    /// Sum of all trajectory CPU times.
    pub cpu_total: Cell<Duration>,

    /// Parallel overhead: wall - cpu_total.
    pub overhead: Cell<Duration>,

    /// Average model preprocessing time per trajectory.
    pub model_preprocessing_avg: Cell<Duration>,

    /// Average solver time per trajectory.
    pub solver_avg: Cell<Duration>,

    /// Average model postprocessing time per trajectory.
    pub model_postprocessing_avg: Cell<Duration>,

    /// Maximum solver time across trajectories.
    pub solver_max: Cell<Duration>,
}

/// Forward pass postprocessing (sequential, after parallel section).
#[derive(Debug, Clone, Default)]
pub struct ForwardPostprocessingTiming {
    /// Time spent capturing trajectory details when requested.
    pub detail_capturing: Cell<Duration>,
}

impl ForwardTiming {
    /// Create new ForwardTiming with preallocated trajectory Vec.
    ///
    /// # Arguments
    ///
    /// * `num_trajectories` - Number of forward passes (trajectories) per iteration
    pub fn new(num_trajectories: usize) -> Self {
        Self {
            preprocessing: ForwardPreprocessingTiming::default(),
            parallel: ForwardParallelTiming::new(num_trajectories),
            postprocessing: ForwardPostprocessingTiming::default(),
            total: Cell::new(Duration::ZERO),
        }
    }

    /// Compute total forward time from phases.
    ///
    /// Should be called after all phases complete.
    pub fn compute_total(&self) {
        let total = self.preprocessing.saa_sampling.get()
            + self.parallel.wall.get()
            + self.postprocessing.detail_capturing.get();
        self.total.set(total);
    }

    /// Reset all timing values.
    pub fn reset(&self) {
        self.preprocessing.saa_sampling.set(Duration::ZERO);
        self.parallel.reset();
        self.postprocessing.detail_capturing.set(Duration::ZERO);
        self.total.set(Duration::ZERO);
    }

    /// Convert to output format.
    ///
    /// # Panics
    ///
    /// Panics if `compute_aggregates()` hasn't been called on the parallel section.
    /// The aggregates are required for the output format.
    pub fn to_output(&self) -> ForwardTimingOutput {
        ForwardTimingOutput {
            saa_sampling: self.preprocessing.saa_sampling.get(),
            model_preprocessing: self.parallel.model_preprocessing_avg.get(),
            solver: self.parallel.solver_avg.get(),
            model_postprocessing: self.parallel.model_postprocessing_avg.get(),
            postprocessing: self.postprocessing.detail_capturing.get(),
            total: self.total.get(),
            parallel_wall: self.parallel.wall.get(),
            parallel_overhead: self.parallel.overhead.get(),
            solver_max: self.parallel.solver_max.get(),
            solver_calls: self.parallel.total_solver_calls(),
        }
    }
}

impl ForwardParallelTiming {
    /// Create with preallocated trajectory Vec.
    ///
    /// # Arguments
    ///
    /// * `num_trajectories` - Number of trajectories to preallocate
    pub fn new(num_trajectories: usize) -> Self {
        Self {
            wall: Cell::new(Duration::ZERO),
            trajectories: (0..num_trajectories)
                .map(|_| TrajectoryTiming::new())
                .collect(),
            cpu_total: Cell::new(Duration::ZERO),
            overhead: Cell::new(Duration::ZERO),
            model_preprocessing_avg: Cell::new(Duration::ZERO),
            solver_avg: Cell::new(Duration::ZERO),
            model_postprocessing_avg: Cell::new(Duration::ZERO),
            solver_max: Cell::new(Duration::ZERO),
        }
    }

    /// Compute aggregate statistics from trajectory timings.
    ///
    /// Call this AFTER the parallel forward section completes.
    /// Populates: `cpu_total`, `overhead`, `*_avg`, `solver_max`.
    ///
    /// # Panics
    ///
    /// Panics if `trajectories` is empty.
    ///
    /// # Algorithm
    ///
    /// - `cpu_total`: Sum of all trajectory CPU times
    /// - `overhead`: wall - cpu_total (uses saturating_sub, cannot be negative)
    /// - Averages: Simple arithmetic mean
    /// - `solver_max`: Maximum solver time across trajectories
    ///
    /// Unlike the legacy code, this does NOT redistribute timing proportionally.
    /// We preserve precise measured values.
    pub fn compute_aggregates(&self) {
        let n = self.trajectories.len();
        assert!(n > 0, "Cannot compute aggregates with zero trajectories");

        // Sum CPU times
        let cpu_total: Duration =
            self.trajectories.iter().map(|t| t.cpu_time()).sum();
        self.cpu_total.set(cpu_total);

        // Compute overhead (can be zero if wall < cpu_total due to parallelism)
        let wall = self.wall.get();
        let overhead = wall.saturating_sub(cpu_total);
        self.overhead.set(overhead);

        // Compute sums for averaging
        let total_model_prep: Duration = self
            .trajectories
            .iter()
            .map(|t| t.model_preprocessing.get())
            .sum();
        let total_solver: Duration =
            self.trajectories.iter().map(|t| t.solver.get()).sum();
        let total_model_post: Duration = self
            .trajectories
            .iter()
            .map(|t| t.model_postprocessing.get())
            .sum();

        // Compute averages
        let n_u32 = n as u32;
        self.model_preprocessing_avg.set(total_model_prep / n_u32);
        self.solver_avg.set(total_solver / n_u32);
        self.model_postprocessing_avg.set(total_model_post / n_u32);

        // Compute max solver time
        let max_solver = self
            .trajectories
            .iter()
            .map(|t| t.solver.get())
            .max()
            .unwrap_or(Duration::ZERO);
        self.solver_max.set(max_solver);
    }

    /// Check if aggregates have been computed.
    ///
    /// Returns true if `cpu_total` is non-zero OR if all trajectories have zero CPU time.
    pub fn aggregates_computed(&self) -> bool {
        self.cpu_total.get() > Duration::ZERO
            || self
                .trajectories
                .iter()
                .all(|t| t.cpu_time() == Duration::ZERO)
    }

    /// Reset all fields including trajectories.
    pub fn reset(&self) {
        self.wall.set(Duration::ZERO);
        for t in &self.trajectories {
            t.reset();
        }
        self.cpu_total.set(Duration::ZERO);
        self.overhead.set(Duration::ZERO);
        self.model_preprocessing_avg.set(Duration::ZERO);
        self.solver_avg.set(Duration::ZERO);
        self.model_postprocessing_avg.set(Duration::ZERO);
        self.solver_max.set(Duration::ZERO);
    }

    /// Get total solver calls across all trajectories.
    pub fn total_solver_calls(&self) -> usize {
        self.trajectories.iter().map(|t| t.get_solver_calls()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_timing_new_preallocates() {
        let ft = ForwardTiming::new(10);
        assert_eq!(ft.parallel.trajectories.len(), 10);
    }

    #[test]
    fn test_compute_total() {
        let ft = ForwardTiming::new(2);
        ft.preprocessing.saa_sampling.set(Duration::from_millis(10));
        ft.parallel.wall.set(Duration::from_millis(100));
        ft.postprocessing
            .detail_capturing
            .set(Duration::from_millis(5));
        ft.compute_total();
        assert_eq!(ft.total.get(), Duration::from_millis(115));
    }

    #[test]
    fn test_reset() {
        let ft = ForwardTiming::new(2);
        ft.preprocessing.saa_sampling.set(Duration::from_millis(10));
        ft.parallel.trajectories[0]
            .solver
            .set(Duration::from_millis(50));
        ft.reset();
        assert_eq!(ft.preprocessing.saa_sampling.get(), Duration::ZERO);
        assert_eq!(ft.parallel.trajectories[0].solver.get(), Duration::ZERO);
    }

    #[test]
    fn test_total_solver_calls() {
        let ft = ForwardTiming::new(3);
        ft.parallel.trajectories[0].solver_calls.set(5);
        ft.parallel.trajectories[1].solver_calls.set(5);
        ft.parallel.trajectories[2].solver_calls.set(5);
        assert_eq!(ft.parallel.total_solver_calls(), 15);
    }

    // T-005: Aggregation tests
    #[test]
    fn test_compute_aggregates_basic() {
        let parallel = ForwardParallelTiming::new(3);

        // Set up trajectory timings
        parallel.trajectories[0]
            .model_preprocessing
            .set(Duration::from_millis(10));
        parallel.trajectories[0]
            .solver
            .set(Duration::from_millis(100));
        parallel.trajectories[0]
            .model_postprocessing
            .set(Duration::from_millis(5));

        parallel.trajectories[1]
            .model_preprocessing
            .set(Duration::from_millis(20));
        parallel.trajectories[1]
            .solver
            .set(Duration::from_millis(150));
        parallel.trajectories[1]
            .model_postprocessing
            .set(Duration::from_millis(10));

        parallel.trajectories[2]
            .model_preprocessing
            .set(Duration::from_millis(15));
        parallel.trajectories[2]
            .solver
            .set(Duration::from_millis(120));
        parallel.trajectories[2]
            .model_postprocessing
            .set(Duration::from_millis(8));

        // Wall time is less than CPU total (parallel speedup)
        parallel.wall.set(Duration::from_millis(200));

        parallel.compute_aggregates();

        // CPU total = (10+100+5) + (20+150+10) + (15+120+8) = 115 + 180 + 143 = 438
        assert_eq!(parallel.cpu_total.get(), Duration::from_millis(438));

        // Overhead = 200 - 438 = 0 (saturating)
        assert_eq!(parallel.overhead.get(), Duration::ZERO);

        // Averages: model_prep = (10+20+15)/3 = 15
        assert_eq!(
            parallel.model_preprocessing_avg.get(),
            Duration::from_millis(15)
        );

        // solver_avg = (100+150+120)/3 = 370/3 = 123.333... → 123ms (truncated)
        let expected_avg = Duration::from_millis(370) / 3;
        assert_eq!(parallel.solver_avg.get(), expected_avg);

        // solver_max = 150
        assert_eq!(parallel.solver_max.get(), Duration::from_millis(150));
    }

    #[test]
    fn test_compute_aggregates_with_overhead() {
        let parallel = ForwardParallelTiming::new(2);

        parallel.trajectories[0]
            .solver
            .set(Duration::from_millis(50));
        parallel.trajectories[1]
            .solver
            .set(Duration::from_millis(50));

        // Wall time greater than CPU total (scheduling overhead)
        parallel.wall.set(Duration::from_millis(150));

        parallel.compute_aggregates();

        assert_eq!(parallel.cpu_total.get(), Duration::from_millis(100));
        assert_eq!(parallel.overhead.get(), Duration::from_millis(50));
    }

    #[test]
    #[should_panic(
        expected = "Cannot compute aggregates with zero trajectories"
    )]
    fn test_compute_aggregates_empty_panics() {
        let parallel = ForwardParallelTiming::new(0);
        parallel.compute_aggregates();
    }

    #[test]
    fn test_aggregates_computed() {
        let parallel = ForwardParallelTiming::new(2);

        // Before computation
        parallel.trajectories[0]
            .solver
            .set(Duration::from_millis(50));
        assert!(!parallel.aggregates_computed());

        // After computation
        parallel.compute_aggregates();
        assert!(parallel.aggregates_computed());
    }

    #[test]
    fn test_single_trajectory() {
        let ft = ForwardTiming::new(1);
        ft.parallel.trajectories[0]
            .solver
            .set(Duration::from_millis(100));
        ft.parallel.wall.set(Duration::from_millis(120));
        ft.parallel.compute_aggregates();

        assert_eq!(ft.parallel.solver_avg.get(), Duration::from_millis(100));
        assert_eq!(ft.parallel.solver_max.get(), Duration::from_millis(100));
        assert_eq!(ft.parallel.overhead.get(), Duration::from_millis(20));
    }

    #[test]
    fn test_all_zero_trajectories() {
        let ft = ForwardTiming::new(3);
        ft.parallel.wall.set(Duration::from_millis(10));
        ft.parallel.compute_aggregates();

        assert_eq!(ft.parallel.cpu_total.get(), Duration::ZERO);
        assert_eq!(ft.parallel.solver_avg.get(), Duration::ZERO);
        assert_eq!(ft.parallel.overhead.get(), Duration::from_millis(10));
    }

    #[test]
    fn test_overhead_saturates_to_zero() {
        // CPU time > wall time (parallel speedup)
        let ft = ForwardTiming::new(2);
        ft.parallel.trajectories[0]
            .solver
            .set(Duration::from_millis(100));
        ft.parallel.trajectories[1]
            .solver
            .set(Duration::from_millis(100));
        ft.parallel.wall.set(Duration::from_millis(50)); // Less than CPU total
        ft.parallel.compute_aggregates();

        assert_eq!(ft.parallel.overhead.get(), Duration::ZERO);
    }

    // T-006: Output conversion tests
    #[test]
    fn test_to_output() {
        let ft = ForwardTiming::new(2);
        ft.preprocessing.saa_sampling.set(Duration::from_millis(10));
        ft.parallel.wall.set(Duration::from_millis(100));
        ft.parallel.trajectories[0]
            .solver
            .set(Duration::from_millis(40));
        ft.parallel.trajectories[1]
            .solver
            .set(Duration::from_millis(60));
        ft.postprocessing
            .detail_capturing
            .set(Duration::from_millis(5));
        ft.total.set(Duration::from_millis(115));

        ft.parallel.compute_aggregates();
        let out = ft.to_output();

        assert_eq!(out.saa_sampling, Duration::from_millis(10));
        assert_eq!(out.solver, Duration::from_millis(50)); // avg
        assert_eq!(out.solver_max, Duration::from_millis(60));
        assert_eq!(out.total, Duration::from_millis(115));
    }
}
