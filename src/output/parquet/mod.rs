//! Parquet output writer implementation.
//!
//! **⚠️ IMPORTANT: This module is currently non-functional after the v0.4.0 output redesign.**
//!
//! The output system was redesigned to always use CSV format with indexed data.
//! The OutputWriter trait pattern was removed for simplicity, which means this
//! Parquet implementation is no longer integrated with the output generation pipeline.
//!
//! **Status**: Code preserved for future integration. See issue #XXX for tracking.
//!
//! # Original Benefits (when integrated)
//!
//! - **Compression**: 75-85% file size reduction vs CSV
//! - **Fast reads**: Columnar format optimized for analytics
//! - **Type preservation**: Maintains data types (no string conversion)
//! - **Metadata**: Embeds schema and statistics
//!
//! # Future Integration Plan
//!
//! To restore Parquet support:
//! 1. Add format parameter to each output function (conditional compilation)
//! 2. Or restore OutputWriter trait with factory pattern
//! 3. Update generate_outputs() to route to appropriate writer
//!
//! See REDESIGN_STATUS.md for details on the v0.4.0 changes.

mod schemas;
mod writer;

pub use writer::{ParquetConfig, ParquetWriter};
