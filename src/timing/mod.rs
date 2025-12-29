//! Zero-pollution timing infrastructure for performance measurement.
//!
//! This module provides RAII-based timing that can be completely eliminated
//! at compile time when the `timing` feature is disabled.
//!
//! # Design Principles
//!
//! 1. **Preserve Precise Values**: Timing values are NEVER overwritten or redistributed
//! 2. **Explicit Parallel Overhead**: Track scheduling overhead as a separate metric
//! 3. **Zero-Cost When Disabled**: Compile-time elimination via feature flags
//! 4. **RAII-Based**: TimingGuard automatically records on drop
//!
//! # Features
//!
//! - `timing`: Enable basic timing collection
//! - `timing-detailed`: Enable per-stage timing breakdown (implies `timing`)
//!
//! # Example
//!
//! ```ignore
//! use std::cell::Cell;
//! use std::time::Duration;
//! use powers_rs::timing::{TimingGuard, IterationTiming};
//!
//! let timing = IterationTiming::default();
//! {
//!     let _guard = TimingGuard::new(&timing.forward.solver);
//!     // ... solver work ...
//! }
//! // timing.forward.solver now contains elapsed time
//! ```

mod collector;
mod guard;
mod metrics;

#[cfg(feature = "timing")]
mod atomic;

pub use collector::{NullTimingCollector, TimingCollector};
pub use guard::TimingGuard;
pub use metrics::{
    BackwardTiming, ForwardTiming, IterationTiming, TimingMetric,
};

#[cfg(feature = "timing")]
pub use atomic::AtomicTimingCollector;
