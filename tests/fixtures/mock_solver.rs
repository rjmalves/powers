// Mock solver implementation for testing SDDP algorithm logic independently of HiGHS
//
// This mock solver allows testing the algorithm flow without depending on the actual
// solver, enabling faster and more isolated unit tests.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Status enum for mock solver (simplified from HiGHS)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockSolverStatus {
    Optimal,
    Infeasible,
    Unbounded,
    Error,
}

/// Mock solver that implements a configurable solver interface for testing
///
/// The mock solver can be configured to return specific results (optimal, infeasible, etc.)
/// and tracks calls for verification in tests. This is essential for testing algorithm logic
/// without depending on HiGHS behavior.
///
/// # Performance Note
/// Uses RefCell for interior mutability and atomic counters for thread-safe call tracking.
/// This is acceptable for tests where performance is not critical.
#[derive(Debug)]
pub struct MockSolver {
    status: MockSolverStatus,
    objective_value: f64,
    solution: Vec<f64>,
    optimize_count: AtomicUsize,
    add_row_count: AtomicUsize,
    // Track row additions for validation
    rows: RefCell<Vec<MockRow>>,
}

#[derive(Debug, Clone)]
struct MockRow {
    lower: f64,
    upper: f64,
    num_nonzeros: usize,
}

impl MockSolver {
    /// Create a new mock solver with default optimal status
    pub fn new() -> Self {
        Self {
            status: MockSolverStatus::Optimal,
            objective_value: 0.0,
            solution: vec![],
            optimize_count: AtomicUsize::new(0),
            add_row_count: AtomicUsize::new(0),
            rows: RefCell::new(vec![]),
        }
    }

    /// Configure the solver to return a specific status
    pub fn with_status(mut self, status: MockSolverStatus) -> Self {
        self.status = status;
        self
    }

    /// Configure the solver to return a specific objective value
    pub fn with_objective_value(mut self, value: f64) -> Self {
        self.objective_value = value;
        self
    }

    /// Configure the solver to return a specific solution vector
    pub fn with_solution(mut self, solution: Vec<f64>) -> Self {
        self.solution = solution;
        self
    }

    /// Get the number of times optimize was called
    pub fn optimize_call_count(&self) -> usize {
        self.optimize_count.load(Ordering::Relaxed)
    }

    /// Get the number of times add_row was called
    pub fn add_row_call_count(&self) -> usize {
        self.add_row_count.load(Ordering::Relaxed)
    }

    /// Get the number of rows added
    pub fn num_rows(&self) -> usize {
        self.rows.borrow().len()
    }

    /// Reset all call counters (useful between test phases)
    pub fn reset_counters(&self) {
        self.optimize_count.store(0, Ordering::Relaxed);
        self.add_row_count.store(0, Ordering::Relaxed);
    }

    /// Simulate solving (just increments counter and returns configured status)
    pub fn solve(&self) -> MockSolverStatus {
        self.optimize_count.fetch_add(1, Ordering::Relaxed);
        self.status
    }

    /// Get the configured objective value
    pub fn get_objective_value(&self) -> f64 {
        self.objective_value
    }

    /// Get the configured solution
    pub fn get_solution(&self) -> &[f64] {
        &self.solution
    }

    /// Simulate adding a row to the model
    pub fn add_row(
        &self,
        lower: f64,
        upper: f64,
        num_nonzeros: usize,
    ) -> usize {
        self.add_row_count.fetch_add(1, Ordering::Relaxed);
        let mut rows = self.rows.borrow_mut();
        rows.push(MockRow {
            lower,
            upper,
            num_nonzeros,
        });
        rows.len() - 1
    }

    /// Get information about a specific row
    pub fn get_row(&self, index: usize) -> Option<(f64, f64, usize)> {
        self.rows
            .borrow()
            .get(index)
            .map(|r| (r.lower, r.upper, r.num_nonzeros))
    }
}

impl Default for MockSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_solver_creation() {
        let solver = MockSolver::new();
        assert_eq!(solver.optimize_call_count(), 0);
        assert_eq!(solver.add_row_call_count(), 0);
        assert_eq!(solver.num_rows(), 0);
    }

    #[test]
    fn test_mock_solver_with_configuration() {
        let solver = MockSolver::new()
            .with_status(MockSolverStatus::Infeasible)
            .with_objective_value(42.0)
            .with_solution(vec![1.0, 2.0, 3.0]);

        assert_eq!(solver.solve(), MockSolverStatus::Infeasible);
        assert_eq!(solver.get_objective_value(), 42.0);
        assert_eq!(solver.get_solution(), &[1.0, 2.0, 3.0]);
        assert_eq!(solver.optimize_call_count(), 1);
    }

    #[test]
    fn test_mock_solver_call_tracking() {
        let solver = MockSolver::new();

        // Multiple solves
        solver.solve();
        solver.solve();
        solver.solve();
        assert_eq!(solver.optimize_call_count(), 3);

        // Multiple row additions
        solver.add_row(0.0, 100.0, 5);
        solver.add_row(10.0, 50.0, 3);
        assert_eq!(solver.add_row_call_count(), 2);
        assert_eq!(solver.num_rows(), 2);
    }

    #[test]
    fn test_mock_solver_row_tracking() {
        let solver = MockSolver::new();

        let row_id = solver.add_row(0.0, 100.0, 5);
        assert_eq!(row_id, 0);

        let (lower, upper, nnz) = solver.get_row(0).unwrap();
        assert_eq!(lower, 0.0);
        assert_eq!(upper, 100.0);
        assert_eq!(nnz, 5);

        assert!(solver.get_row(999).is_none());
    }

    #[test]
    fn test_mock_solver_reset() {
        let solver = MockSolver::new();

        solver.solve();
        solver.add_row(0.0, 1.0, 1);
        assert_eq!(solver.optimize_call_count(), 1);
        assert_eq!(solver.add_row_call_count(), 1);

        solver.reset_counters();
        assert_eq!(solver.optimize_call_count(), 0);
        assert_eq!(solver.add_row_call_count(), 0);
        // Note: num_rows is not reset (rows persist)
        assert_eq!(solver.num_rows(), 1);
    }
}
