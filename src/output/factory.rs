//! Factory for creating output writers based on format selection.
//!
//! Provides a unified interface for instantiating the appropriate writer
//! implementation based on configuration.

use super::csv::CsvWriter;
use super::parquet::ParquetWriter;
use super::writer::{OutputWriter, Result};
use crate::input::OutputFormat;

use std::path::Path;

/// Creates an output writer based on format selection.
///
/// This factory function abstracts over concrete writer implementations,
/// allowing the rest of the codebase to work with the `OutputWriter` trait.
///
/// # Arguments
///
/// * `format` - Desired output format (CSV, PARQUET, or Auto)
/// * `output_dir` - Directory where output files will be written
///
/// # Format Selection
///
/// - `CSV`: Human-readable text format (always available)
/// - `PARQUET`: Columnar binary format (always available)
/// - `Auto`: Currently selects CSV (intelligent selection not yet implemented)
///
/// # Errors
///
/// Returns error if:
/// - Output directory cannot be created
/// - Writer initialization fails for any reason
///
/// # Examples
///
/// ```ignore
/// use powers_rs::output::factory::create_writer;
/// use powers_rs::input::OutputFormat;
///
/// let writer = create_writer(OutputFormat::CSV, Path::new("./output"))?;
/// // Use writer through OutputWriter trait
/// ```
pub fn create_writer(
    format: OutputFormat,
    output_dir: &Path,
) -> Result<Box<dyn OutputWriter>> {
    match format {
        OutputFormat::CSV => {
            let writer = CsvWriter::new(output_dir)?;
            Ok(Box::new(writer))
        }

        OutputFormat::PARQUET => {
            let writer = ParquetWriter::new(output_dir)?;
            Ok(Box::new(writer))
        }

        OutputFormat::Auto => {
            // For now, Auto always selects CSV
            // In the future, this could inspect data size and choose accordingly
            let writer = CsvWriter::new(output_dir)?;
            Ok(Box::new(writer))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_create_csv_writer() {
        let temp_dir = std::env::temp_dir().join("powers_test_factory_csv");
        let result = create_writer(OutputFormat::CSV, &temp_dir);
        assert!(result.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_create_auto_writer() {
        let temp_dir = std::env::temp_dir().join("powers_test_factory_auto");
        let result = create_writer(OutputFormat::Auto, &temp_dir);
        assert!(result.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_create_parquet_writer() {
        let temp_dir = std::env::temp_dir().join("powers_test_factory_parquet");
        let result = create_writer(OutputFormat::PARQUET, &temp_dir);
        assert!(result.is_ok());

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_invalid_directory() {
        // Try to create in a path that should fail (e.g., root with no permissions)
        // Note: This test might behave differently on different systems
        let invalid_path = PathBuf::from("/proc/powers_test_invalid");
        let result = create_writer(OutputFormat::CSV, &invalid_path);
        // We don't assert error here because behavior is system-dependent
        // Just ensure it doesn't panic
        let _ = result;
    }
}
