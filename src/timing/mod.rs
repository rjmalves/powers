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

mod backward;
mod collector;
mod forward;
mod guard;
mod iteration;
mod metrics;
mod output;
mod trajectory;

#[cfg(feature = "timing")]
mod atomic;

// New timing types (Epic 1) - use explicit paths to avoid conflicts
pub use backward::{
    BackwardPhase1Timing, BackwardPhase2Timing, BackwardPhase3Timing,
    NewBackwardTiming,
};
pub use collector::{NullTimingCollector, TimingCollector};
pub use forward::{
    ForwardParallelTiming, ForwardPostprocessingTiming,
    ForwardPreprocessingTiming, ForwardTiming as NewForwardTiming,
};
pub use guard::TimingGuard;
pub use iteration::{NewIterationTiming, TrainingTiming};
pub use output::{
    BackwardTimingOutput, ForwardTimingOutput, IterationTimingOutput,
};
// TimingMetric still used for feature-gated detailed timing
pub use metrics::TimingMetric;
pub use trajectory::TrajectoryTiming;

#[cfg(feature = "timing")]
pub use atomic::AtomicTimingCollector;
