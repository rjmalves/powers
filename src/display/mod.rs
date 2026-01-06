//! Display system for POWE.RS terminal output.
//!
//! This module provides a rich, profile-based display system for training
//! and simulation progress. Supports multiple output profiles from detailed
//! advanced views to minimal progress bars and machine-readable JSON.
//!
//! # Profiles
//!
//! - `Advanced`: Full metrics with colors, statistics, and trend indicators
//! - `Standard`: Key metrics with simplified layout
//! - `Minimal`: Progress bar with final summary only
//! - `Automation`: JSON lines for machine parsing
//!
//! # Example
//!
//! ```ignore
//! use powers_rs::display::{DisplayConfig, DisplayProfile};
//!
//! let config = DisplayConfig::default();
//! assert_eq!(config.profile, DisplayProfile::Advanced);
//! ```

pub mod components;
pub mod config;
pub mod context;
pub mod renderer;
pub mod renderers;
pub mod terminal;

// Re-exports will be added as types are implemented
pub use config::{
    ColorMode, DisplayConfig, DisplayConfigInput, DisplayProfile,
};
pub use context::{CostStatistics, DisplayContext, GapTrend, IterationTracker};
pub use renderer::{create_renderer, DisplayManager, DisplayRenderer};
pub use terminal::{is_piped, terminal_width_or, TerminalCapabilities};
