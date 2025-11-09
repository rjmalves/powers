//! Output writer trait for multi-format export support.
//!
//! Provides a trait-based abstraction for writing SDDP algorithm outputs
//! to various formats (CSV, Parquet, etc.). The design prioritizes:
//! - **Flexibility**: Support multiple output formats
//! - **Performance**: Minimize overhead, allow buffering
//! - **Type Safety**: Leverage Rust's type system for correctness
//!
//! # Design Rationale
//!
//! We chose a single trait with all methods (Option A) rather than
//! composition with sub-traits because:
//! - Simpler implementation for concrete writers
//! - Clear contract for all output operations
//! - Easier to add optional methods with default implementations
//! - Better error handling consistency
//!
//! # Example
//!
//! ```ignore
//! use powers_rs::output::writer::OutputWriter;
//!
//! struct CsvWriter {
//!     output_dir: PathBuf,
//! }
//!
//! impl OutputWriter for CsvWriter {
//!     fn write_training(&mut self, data: &[IterationResult]) -> Result<()> {
//!         // CSV-specific implementation
//!         Ok(())
//!     }
//!     // ... implement other methods
//! }
//! ```

use crate::fcf;
use crate::graph;
use crate::scenario;
use crate::sddp;
use std::error::Error;
use std::sync::{Arc, Mutex};

/// Result type for output operations using boxed errors for flexibility
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Trait for writing SDDP algorithm outputs to various formats.
///
/// Implementors must provide methods for writing all output types:
/// - Training convergence data
/// - Forward/backward pass details
/// - Simulation results
/// - Cuts and states
/// - Sampled noises
///
/// # Error Handling
///
/// All methods return `Result<()>` to allow implementors to handle
/// format-specific errors (I/O, serialization, validation). Errors
/// should provide context about what failed.
///
/// # Performance
///
/// Implementors should consider:
/// - **Buffering**: Use buffered writers for better I/O performance
/// - **Allocation**: Pre-allocate buffers when output size is known
/// - **Batching**: Write records in batches rather than one-by-one
///
/// # Thread Safety
///
/// Methods take `&mut self` for interior mutability flexibility.
/// Implementors can use interior mutability (Mutex, RefCell) if
/// concurrent writes are needed, but this is not required.
pub trait OutputWriter {
    /// Writes training convergence results.
    ///
    /// Contains per-iteration metrics: bounds, costs, gaps, timings.
    ///
    /// # Arguments
    ///
    /// * `results` - Slice of iteration results from training
    ///
    /// # Errors
    ///
    /// Returns error if writing fails (I/O, serialization, etc.)
    fn write_training(
        &mut self,
        results: &[sddp::IterationResult],
    ) -> Result<()>;

    /// Writes forward pass detail data.
    ///
    /// Contains state, decision, and uncertainty realizations for
    /// each stage of each forward pass across all iterations.
    ///
    /// # Arguments
    ///
    /// * `details` - Slice of forward pass details
    ///
    /// # Performance Note
    ///
    /// This can be a large dataset (iterations × forward_passes × stages × variables).
    /// Implementors should use efficient serialization and consider
    /// streaming or batching for large outputs.
    ///
    /// # Errors
    ///
    /// Returns error if writing fails
    fn write_forward_detail(
        &mut self,
        details: &[sddp::ForwardPassDetail],
    ) -> Result<()>;

    /// Writes backward pass detail data.
    ///
    /// Contains dual variables, optimality information, and cut data
    /// from backward pass computations.
    ///
    /// # Arguments
    ///
    /// * `details` - Slice of backward pass details
    ///
    /// # Errors
    ///
    /// Returns error if writing fails
    fn write_backward_detail(
        &mut self,
        details: &[sddp::BackwardPassDetail],
    ) -> Result<()>;

    /// Writes Benders cuts from the future cost function.
    ///
    /// Exports all cuts (RHS and coefficients) generated during training.
    /// This is essential for analyzing the polyhedral approximation quality.
    ///
    /// # Arguments
    ///
    /// * `graph` - Graph of future cost functions containing cut pools
    ///
    /// # Format Note
    ///
    /// Cuts are written in coefficient-value pairs. For PAR models,
    /// this includes storage and lag coefficients. Implementors should
    /// use a normalized format (one row per coefficient) for consistency.
    ///
    /// # Errors
    ///
    /// Returns error if writing fails
    fn write_cuts(
        &mut self,
        graph: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    ) -> Result<()>;

    /// Writes visited states from the state pool.
    ///
    /// Exports all unique states visited during training, with their
    /// dominating cuts and objective values.
    ///
    /// # Arguments
    ///
    /// * `graph` - Graph of future cost functions containing state pools
    ///
    /// # Errors
    ///
    /// Returns error if writing fails
    fn write_states(
        &mut self,
        graph: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    ) -> Result<()>;

    /// Writes simulation trajectory results.
    ///
    /// Contains out-of-sample policy evaluation results with full
    /// state and decision trajectories for each simulation scenario.
    ///
    /// # Arguments
    ///
    /// * `trajectories` - Slice of simulation trajectories
    /// * `system` - Power system configuration for metadata
    ///
    /// # Performance Note
    ///
    /// For large simulations (e.g., 1000+ scenarios), this can be
    /// substantial data. Consider compression or efficient encoding.
    ///
    /// # Errors
    ///
    /// Returns error if writing fails
    fn write_simulation(
        &mut self,
        trajectories: &[sddp::SimulationTrajectory],
        system: &crate::system::System,
    ) -> Result<()>;

    /// Writes sampled noises from scenario tree.
    ///
    /// Exports the noise samples used during training for reproducibility
    /// and analysis of scenario quality.
    ///
    /// # Arguments
    ///
    /// * `tree` - Scenario tree containing sampled noises
    /// * `system` - Power system configuration for metadata
    ///
    /// # Errors
    ///
    /// Returns error if writing fails
    fn write_noises(
        &mut self,
        tree: &scenario::ScenarioTree,
        system: &crate::system::System,
    ) -> Result<()>;

    /// Flushes any buffered data to storage.
    ///
    /// Called at the end of output generation to ensure all data
    /// is persisted. Implementors should flush all internal buffers.
    ///
    /// # Default Implementation
    ///
    /// The default does nothing, suitable for unbuffered writers.
    ///
    /// # Errors
    ///
    /// Returns error if flushing fails
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Mock output writer for testing.
///
/// Records which methods were called and how many times, without
/// performing actual I/O. Useful for unit testing output orchestration.
///
/// # Example
///
/// ```
/// use powers_rs::output::writer::{OutputWriter, MockWriter};
///
/// let mut writer = MockWriter::new();
/// writer.write_training(&[]).unwrap();
/// assert_eq!(writer.training_calls, 1);
/// ```
#[derive(Debug, Default)]
pub struct MockWriter {
    pub training_calls: usize,
    pub forward_detail_calls: usize,
    pub backward_detail_calls: usize,
    pub cuts_calls: usize,
    pub states_calls: usize,
    pub simulation_calls: usize,
    pub noises_calls: usize,
    pub flush_calls: usize,
}

impl MockWriter {
    /// Creates a new mock writer with zero call counts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resets all call counts to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns total number of write method calls (excluding flush).
    pub fn total_write_calls(&self) -> usize {
        self.training_calls
            + self.forward_detail_calls
            + self.backward_detail_calls
            + self.cuts_calls
            + self.states_calls
            + self.simulation_calls
            + self.noises_calls
    }
}

impl OutputWriter for MockWriter {
    fn write_training(
        &mut self,
        _results: &[sddp::IterationResult],
    ) -> Result<()> {
        self.training_calls += 1;
        Ok(())
    }

    fn write_forward_detail(
        &mut self,
        _details: &[sddp::ForwardPassDetail],
    ) -> Result<()> {
        self.forward_detail_calls += 1;
        Ok(())
    }

    fn write_backward_detail(
        &mut self,
        _details: &[sddp::BackwardPassDetail],
    ) -> Result<()> {
        self.backward_detail_calls += 1;
        Ok(())
    }

    fn write_cuts(
        &mut self,
        _graph: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    ) -> Result<()> {
        self.cuts_calls += 1;
        Ok(())
    }

    fn write_states(
        &mut self,
        _graph: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    ) -> Result<()> {
        self.states_calls += 1;
        Ok(())
    }

    fn write_simulation(
        &mut self,
        _trajectories: &[sddp::SimulationTrajectory],
        _system: &crate::system::System,
    ) -> Result<()> {
        self.simulation_calls += 1;
        Ok(())
    }

    fn write_noises(
        &mut self,
        _tree: &scenario::ScenarioTree,
        _system: &crate::system::System,
    ) -> Result<()> {
        self.noises_calls += 1;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.flush_calls += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_writer_basic() {
        let writer = MockWriter::new();
        assert_eq!(writer.training_calls, 0);
        assert_eq!(writer.total_write_calls(), 0);
    }

    #[test]
    fn test_mock_writer_counts_calls() {
        let mut writer = MockWriter::new();

        writer.write_training(&[]).unwrap();
        assert_eq!(writer.training_calls, 1);
        assert_eq!(writer.total_write_calls(), 1);

        writer.write_training(&[]).unwrap();
        assert_eq!(writer.training_calls, 2);
        assert_eq!(writer.total_write_calls(), 2);
    }

    #[test]
    fn test_mock_writer_tracks_all_methods() {
        let mut writer = MockWriter::new();

        writer.write_training(&[]).unwrap();
        writer.write_forward_detail(&[]).unwrap();
        writer.write_backward_detail(&[]).unwrap();
        writer
            .write_noises(
                &scenario::ScenarioTree::new_empty(),
                &crate::system::System::new_empty(),
            )
            .unwrap();
        writer.flush().unwrap();

        assert_eq!(writer.training_calls, 1);
        assert_eq!(writer.forward_detail_calls, 1);
        assert_eq!(writer.backward_detail_calls, 1);
        assert_eq!(writer.noises_calls, 1);
        assert_eq!(writer.flush_calls, 1);
        assert_eq!(writer.total_write_calls(), 4); // Doesn't include flush
    }

    #[test]
    fn test_mock_writer_reset() {
        let mut writer = MockWriter::new();

        writer.write_training(&[]).unwrap();
        writer.write_training(&[]).unwrap();
        assert_eq!(writer.training_calls, 2);

        writer.reset();
        assert_eq!(writer.training_calls, 0);
        assert_eq!(writer.total_write_calls(), 0);
    }

    #[test]
    fn test_mock_writer_multiple_types() {
        let mut writer = MockWriter::new();

        // Simulate a typical output sequence
        writer.write_training(&[]).unwrap();
        writer.write_forward_detail(&[]).unwrap();
        writer.write_backward_detail(&[]).unwrap();
        writer
            .write_noises(
                &scenario::ScenarioTree::new_empty(),
                &crate::system::System::new_empty(),
            )
            .unwrap();
        writer.flush().unwrap();

        // Verify all tracked correctly
        assert_eq!(writer.total_write_calls(), 4);
        assert!(writer.training_calls > 0);
        assert!(writer.forward_detail_calls > 0);
        assert!(writer.backward_detail_calls > 0);
        assert!(writer.noises_calls > 0);
        assert!(writer.flush_calls > 0);
    }

    #[test]
    fn test_mock_writer_error_propagation() {
        // MockWriter never returns errors, but trait contract allows it
        let mut writer = MockWriter::new();
        let result = writer.write_training(&[]);
        assert!(result.is_ok());
    }
}
