//! Parquet output writer implementation.
//!
//! Provides columnar storage with compression for massive file size reduction
//! (75-85% smaller than CSV for typical SDDP outputs). Requires the
//! `parquet-output` feature flag.
//!
//! # Benefits
//!
//! - **Compression**: 75-85% file size reduction vs CSV
//! - **Fast reads**: Columnar format optimized for analytics
//! - **Type preservation**: Maintains data types (no string conversion)
//! - **Metadata**: Embeds schema and statistics
//!
//! # Usage
//!
//! ```ignore
//! use powers_rs::output::parquet::ParquetWriter;
//! use powers_rs::output::writer::OutputWriter;
//!
//! let writer = ParquetWriter::new("./output")?;
//! writer.write_training(&results)?;
//! ```

mod schemas;
mod writer;

pub use writer::{ParquetConfig, ParquetWriter};
