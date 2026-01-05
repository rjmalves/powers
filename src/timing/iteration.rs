//! Iteration and training timing structures.

use std::cell::Cell;
use std::time::Duration;

use super::output::IterationTimingOutput;
use super::{NewBackwardTiming, NewForwardTiming};

/// Complete timing for one SDDP training iteration.
///
/// This is the top-level timing struct - created at iteration start,
/// passed to sub-operations, and converted to output format at iteration end.
///
/// # Usage
///
/// ```ignore
/// use powers_rs::timing::{NewIterationTiming, TimingGuard};
///
/// let timing = NewIterationTiming::new(num_forward_passes);
///
/// // Model allocation
/// {
///     let _guard = TimingGuard::new(&timing.model_allocation);
///     create_models();
/// }
///
/// // Forward pass
/// forward_pass(&timing.forward);
///
/// // Backward pass
/// backward_pass(&timing.backward);
///
/// // Compute totals
/// timing.forward.parallel.compute_aggregates();
/// timing.forward.compute_total();
/// timing.compute_total();
///
/// // Convert to output
/// let output = timing.to_output();
/// ```
pub struct NewIterationTiming {
    /// Time to create solver Models from cached Problems.
    pub model_allocation: Cell<Duration>,

    /// Forward pass timing.
    pub forward: NewForwardTiming,

    /// Backward pass timing.
    pub backward: NewBackwardTiming,

    /// Time to cleanup solver Models at iteration end.
    pub model_cleanup: Cell<Duration>,

    /// Total iteration wall-clock time.
    pub total: Cell<Duration>,
}

impl NewIterationTiming {
    /// Create new IterationTiming with preallocated trajectory Vec.
    ///
    /// # Arguments
    ///
    /// * `num_forward_passes` - Number of forward passes (trajectories) per iteration
    pub fn new(num_forward_passes: usize) -> Self {
        Self {
            model_allocation: Cell::new(Duration::ZERO),
            forward: NewForwardTiming::new(num_forward_passes),
            backward: NewBackwardTiming::new(),
            model_cleanup: Cell::new(Duration::ZERO),
            total: Cell::new(Duration::ZERO),
        }
    }

    /// Compute total iteration time from components.
    ///
    /// Should be called at the end of an iteration before converting to output.
    pub fn compute_total(&self) {
        let total = self.model_allocation.get()
            + self.forward.total.get()
            + self.backward.total.get()
            + self.model_cleanup.get();
        self.total.set(total);
    }

    /// Reset all timing values for reuse.
    ///
    /// Call this at the start of each iteration if reusing the same struct.
    pub fn reset(&self) {
        self.model_allocation.set(Duration::ZERO);
        self.forward.reset();
        self.backward.reset();
        self.model_cleanup.set(Duration::ZERO);
        self.total.set(Duration::ZERO);
    }

    /// Get total solver calls (forward + backward).
    pub fn total_solver_calls(&self) -> usize {
        self.forward.parallel.total_solver_calls()
            + self.backward.get_solver_calls()
    }

    /// Convert to output format.
    ///
    /// Call `compute_total()` and `forward.parallel.compute_aggregates()` first.
    pub fn to_output(&self) -> IterationTimingOutput {
        IterationTimingOutput {
            model_allocation: self.model_allocation.get(),
            forward: self.forward.to_output(),
            backward: self.backward.to_output(),
            model_cleanup: self.model_cleanup.get(),
            total: self.total.get(),
            solver_calls: self.total_solver_calls(),
        }
    }
}

/// Timing for the complete training run.
///
/// Contains preprocessing/postprocessing times and per-iteration timing.
pub struct TrainingTiming {
    /// Preprocessing before iterations begin (graph construction, warmup).
    pub preprocessing: Cell<Duration>,

    /// Per-iteration timing (preallocated, length = num_iterations).
    pub iterations: Vec<NewIterationTiming>,

    /// Postprocessing after iterations complete.
    pub postprocessing: Cell<Duration>,

    /// Total training time.
    pub total: Cell<Duration>,
}

impl TrainingTiming {
    /// Create new TrainingTiming with preallocated iteration Vec.
    ///
    /// # Arguments
    ///
    /// * `num_iterations` - Number of training iterations
    /// * `num_forward_passes` - Number of forward passes per iteration
    pub fn new(num_iterations: usize, num_forward_passes: usize) -> Self {
        Self {
            preprocessing: Cell::new(Duration::ZERO),
            iterations: (0..num_iterations)
                .map(|_| NewIterationTiming::new(num_forward_passes))
                .collect(),
            postprocessing: Cell::new(Duration::ZERO),
            total: Cell::new(Duration::ZERO),
        }
    }

    /// Compute total training time.
    pub fn compute_total(&self) {
        let iter_total: Duration =
            self.iterations.iter().map(|it| it.total.get()).sum();
        let total =
            self.preprocessing.get() + iter_total + self.postprocessing.get();
        self.total.set(total);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iteration_timing_new() {
        let it = NewIterationTiming::new(5);
        assert_eq!(it.forward.parallel.trajectories.len(), 5);
        assert_eq!(it.total.get(), Duration::ZERO);
    }

    #[test]
    fn test_compute_total() {
        let it = NewIterationTiming::new(2);
        it.model_allocation.set(Duration::from_millis(10));
        it.forward.total.set(Duration::from_millis(100));
        it.backward.total.set(Duration::from_millis(50));
        it.model_cleanup.set(Duration::from_millis(5));
        it.compute_total();
        assert_eq!(it.total.get(), Duration::from_millis(165));
    }

    #[test]
    fn test_reset() {
        let it = NewIterationTiming::new(2);
        it.model_allocation.set(Duration::from_millis(10));
        it.forward
            .preprocessing
            .saa_sampling
            .set(Duration::from_millis(5));
        it.reset();
        assert_eq!(it.model_allocation.get(), Duration::ZERO);
        assert_eq!(it.forward.preprocessing.saa_sampling.get(), Duration::ZERO);
    }

    #[test]
    fn test_total_solver_calls() {
        let it = NewIterationTiming::new(2);
        it.forward.parallel.trajectories[0].solver_calls.set(5);
        it.forward.parallel.trajectories[1].solver_calls.set(5);
        it.backward.solver_calls.set(10);
        assert_eq!(it.total_solver_calls(), 20);
    }

    #[test]
    fn test_training_timing_new() {
        let tt = TrainingTiming::new(10, 5);
        assert_eq!(tt.iterations.len(), 10);
        assert_eq!(tt.iterations[0].forward.parallel.trajectories.len(), 5);
    }

    #[test]
    fn test_training_timing_compute_total() {
        let tt = TrainingTiming::new(2, 1);
        tt.preprocessing.set(Duration::from_millis(100));
        tt.iterations[0].total.set(Duration::from_millis(50));
        tt.iterations[1].total.set(Duration::from_millis(50));
        tt.postprocessing.set(Duration::from_millis(10));
        tt.compute_total();
        assert_eq!(tt.total.get(), Duration::from_millis(210));
    }

    // T-006: Output conversion test
    #[test]
    fn test_to_output() {
        let it = NewIterationTiming::new(1);
        it.model_allocation.set(Duration::from_millis(10));
        it.forward.total.set(Duration::from_millis(100));
        it.backward.total.set(Duration::from_millis(50));
        it.backward.solver_calls.set(5);
        it.model_cleanup.set(Duration::from_millis(5));
        it.total.set(Duration::from_millis(165));

        it.forward.parallel.compute_aggregates();
        let out = it.to_output();

        assert_eq!(out.model_allocation, Duration::from_millis(10));
        assert_eq!(out.total, Duration::from_millis(165));
        assert_eq!(out.backward.solver_calls, 5);
    }
}

// T-007: Additional comprehensive tests
#[test]
fn test_nested_reset() {
    let it = NewIterationTiming::new(3);

    // Set various values
    it.model_allocation.set(Duration::from_millis(10));
    it.forward
        .preprocessing
        .saa_sampling
        .set(Duration::from_millis(20));
    it.forward.parallel.wall.set(Duration::from_millis(100));
    it.forward.parallel.trajectories[0]
        .solver
        .set(Duration::from_millis(50));
    it.forward.parallel.trajectories[1]
        .solver
        .set(Duration::from_millis(60));
    it.backward.phase1.solver.set(Duration::from_millis(30));
    it.backward.solver_calls.set(10);

    // Reset
    it.reset();

    // Verify all zeroed
    assert_eq!(it.model_allocation.get(), Duration::ZERO);
    assert_eq!(it.forward.preprocessing.saa_sampling.get(), Duration::ZERO);
    assert_eq!(it.forward.parallel.wall.get(), Duration::ZERO);
    assert_eq!(
        it.forward.parallel.trajectories[0].solver.get(),
        Duration::ZERO
    );
    assert_eq!(
        it.forward.parallel.trajectories[1].solver.get(),
        Duration::ZERO
    );
    assert_eq!(it.backward.phase1.solver.get(), Duration::ZERO);
    assert_eq!(it.backward.solver_calls.get(), 0);
}

#[test]
fn test_full_iteration_workflow() {
    let timing = NewIterationTiming::new(2);

    // Simulate model allocation
    timing.model_allocation.set(Duration::from_millis(10));

    // Simulate SAA sampling
    timing
        .forward
        .preprocessing
        .saa_sampling
        .set(Duration::from_millis(5));

    // Simulate parallel forward pass
    timing.forward.parallel.wall.set(Duration::from_millis(100));
    timing.forward.parallel.trajectories[0]
        .model_preprocessing
        .set(Duration::from_millis(10));
    timing.forward.parallel.trajectories[0]
        .solver
        .set(Duration::from_millis(80));
    timing.forward.parallel.trajectories[0]
        .model_postprocessing
        .set(Duration::from_millis(5));
    timing.forward.parallel.trajectories[0].solver_calls.set(5);

    timing.forward.parallel.trajectories[1]
        .model_preprocessing
        .set(Duration::from_millis(12));
    timing.forward.parallel.trajectories[1]
        .solver
        .set(Duration::from_millis(90));
    timing.forward.parallel.trajectories[1]
        .model_postprocessing
        .set(Duration::from_millis(6));
    timing.forward.parallel.trajectories[1].solver_calls.set(5);

    // Compute aggregates
    timing.forward.parallel.compute_aggregates();

    // Forward postprocessing
    timing
        .forward
        .postprocessing
        .detail_capturing
        .set(Duration::from_millis(2));
    timing.forward.compute_total();

    // Backward pass
    timing
        .backward
        .phase1
        .solver
        .set(Duration::from_millis(200));
    timing
        .backward
        .phase2
        .cut_selection
        .set(Duration::from_millis(10));
    timing
        .backward
        .phase3
        .problem_update
        .set(Duration::from_millis(5));
    timing.backward.solver_calls.set(50);
    timing.backward.total.set(Duration::from_millis(215));

    // Model cleanup
    timing.model_cleanup.set(Duration::from_millis(3));

    // Compute total
    timing.compute_total();

    // Convert to output
    let output = timing.to_output();

    // Verify output
    assert_eq!(output.model_allocation, Duration::from_millis(10));
    assert_eq!(output.forward.saa_sampling, Duration::from_millis(5));
    assert_eq!(output.forward.solver, Duration::from_millis(85)); // avg(80, 90)
    assert_eq!(output.forward.solver_max, Duration::from_millis(90));
    assert_eq!(output.forward.solver_calls, 10);
    assert_eq!(output.backward.solver, Duration::from_millis(200));
    assert_eq!(output.backward.solver_calls, 50);
    assert_eq!(output.solver_calls, 60); // 10 + 50
}
