//! CSV writer implementation.
//!
//! Wraps existing CSV output functions with the OutputWriter trait interface.
//! This provides a unified API for multi-format output while delegating to
//! the proven, efficient CSV implementations.

use super::cuts_states_indexed::{
    write_benders_cuts_indexed, write_visited_states_indexed,
};
use super::detail_indexed::{
    write_backward_detail_indexed, write_forward_detail_indexed,
};
use super::simulation_normalized::write_simulation_normalized;
use super::training::{write_sampled_noises_indexed, write_training_results};
use crate::output::writer::{OutputWriter, Result};

use crate::fcf;
use crate::graph;
use crate::scenario;
use crate::sddp;
use crate::system;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// CSV format output writer.
///
/// Writes SDDP algorithm outputs to CSV files with indexed/normalized format.
/// This is the default output format, always available without feature flags.
///
/// # Format Details
///
/// All outputs use indexed/normalized format for consistency and efficiency:
/// - Variables referenced by integer indices (requires variable_dictionary.csv)
/// - Cut/state coefficients use integer indices (requires coefficient_dictionary.csv)
/// - Simulation uses normalized long format (one row per value)
/// - Sampled noises use indexed format
///
/// # Performance
///
/// CSV writing is I/O bound. The implementation uses buffered writers
/// automatically via the csv crate for optimal throughput.
///
/// # Example
///
/// ```ignore
/// use powers_rs::output::csv_writer::CsvWriter;
/// use powers_rs::output::writer::OutputWriter;
///
/// let mut writer = CsvWriter::new("./output")?;
/// writer.write_training(&results)?;
/// writer.flush()?;
/// ```
pub struct CsvWriter {
    output_dir: PathBuf,
}

impl CsvWriter {
    /// Creates a new CSV writer for the specified output directory.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory where CSV files will be written
    ///
    /// # Errors
    ///
    /// Returns error if directory creation fails or path is invalid
    pub fn new(output_dir: impl Into<PathBuf>) -> Result<Self> {
        let output_dir = output_dir.into();
        std::fs::create_dir_all(&output_dir)?;
        Ok(Self { output_dir })
    }

    /// Gets the output directory path as a string slice.
    ///
    /// This is used internally to pass to the existing CSV writing functions.
    fn output_path(&self) -> Option<&str> {
        self.output_dir.to_str()
    }
}

impl OutputWriter for CsvWriter {
    fn write_training(
        &mut self,
        results: &[sddp::IterationResult],
    ) -> Result<()> {
        write_training_results(results, self.output_path())
    }

    fn write_forward_detail(
        &mut self,
        details: &[sddp::ForwardPassDetail],
    ) -> Result<()> {
        write_forward_detail_indexed(details, self.output_path())
    }

    fn write_backward_detail(
        &mut self,
        details: &[sddp::BackwardPassDetail],
    ) -> Result<()> {
        write_backward_detail_indexed(details, self.output_path())
    }

    fn write_cuts(
        &mut self,
        graph: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    ) -> Result<()> {
        write_benders_cuts_indexed(graph, self.output_path())
    }

    fn write_states(
        &mut self,
        graph: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    ) -> Result<()> {
        write_visited_states_indexed(graph, self.output_path())
    }

    fn write_simulation(
        &mut self,
        trajectories: &[sddp::SimulationTrajectory],
        system: &system::System,
    ) -> Result<()> {
        write_simulation_normalized(trajectories, system, self.output_path())
    }

    fn write_noises(
        &mut self,
        tree: &scenario::ScenarioTree,
        system: &system::System,
    ) -> Result<()> {
        write_sampled_noises_indexed(tree, system, self.output_path())
    }

    fn flush(&mut self) -> Result<()> {
        // CSV writers flush automatically on drop
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_writer_creation() {
        let temp_dir = std::env::temp_dir().join("powers_test_csv_writer");
        let result = CsvWriter::new(&temp_dir);
        assert!(result.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_csv_writer_path() {
        let temp_dir = std::env::temp_dir().join("powers_test_csv_path");
        let writer = CsvWriter::new(&temp_dir).unwrap();
        let path = writer.output_path();
        assert!(path.is_some());
        assert!(path.unwrap().contains("powers_test_csv_path"));

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_csv_writer_flush() {
        let temp_dir = std::env::temp_dir().join("powers_test_csv_flush");
        let mut writer = CsvWriter::new(&temp_dir).unwrap();

        // Flush should succeed even with no data written
        assert!(writer.flush().is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_csv_writer_training_empty() {
        let temp_dir = std::env::temp_dir().join("powers_test_csv_training");
        let mut writer = CsvWriter::new(&temp_dir).unwrap();

        // Writing empty results should succeed
        let results: Vec<sddp::IterationResult> = vec![];
        assert!(writer.write_training(&results).is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
