//! CSV output writer implementation.
//!
//! This module implements CSV-based output for all SDDP algorithm results.
//! All outputs use indexed/normalized format for consistency and efficiency.
//!
//! # Format Details
//!
//! - **Indexed format**: Variables referenced by integer indices (requires variable_dictionary.csv)
//! - **Normalized format**: One row per value for consistency
//! - **Buffered writes**: Uses csv crate for optimal I/O performance
//!
//! # Organization
//!
//! - `writer.rs` - CsvWriter implementing OutputWriter trait
//! - `training.rs` - Training convergence and sampled noises
//! - `detail_indexed.rs` - Forward/backward pass details
//! - `cuts_states_indexed.rs` - Benders cuts and visited states
//! - `simulation_normalized.rs` - Simulation trajectories

mod csv_writer;
pub(super) mod cuts_states_indexed;
pub(super) mod detail_indexed;
pub(super) mod simulation_normalized;
pub(super) mod training;

pub use csv_writer::CsvWriter;
