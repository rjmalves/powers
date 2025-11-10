//! Memory management and buffer pre-allocation infrastructure.
//!
//! This module provides efficient memory management for the SDDP algorithm by
//! pre-allocating buffers and reusing them across iterations. Profiling showed
//! that malloc/memset operations consume 5.28% of CPU time. By computing buffer
//! sizes upfront and reusing buffers, we can reduce this overhead to <2%.
//!
//! # Components
//!
//! - [`SizingInfo`]: Computes all buffer dimensions from input configuration
//! - [`NodeSizing`]: Per-node sizing information
//! - [`MemoryBreakdown`]: Detailed memory component breakdown
//! - [`Buffer`]: Generic pre-allocated buffer with reuse semantics (future)
//! - [`BufferPool`]: Thread-safe buffer pool for parallel execution (future)
//!
//! # Architecture
//!
//! The memory module is based on a key insight: **all data structures in POWE.RS
//! have deterministic sizes computable from input files**. Once we know:
//! - Number of hydros, thermals, buses, lines (from system.json)
//! - Number of stages and scenarios (from graph.json and config.json)
//! - AR orders for uncertainty models (from recourse.json)
//!
//! We can compute exact buffer sizes for:
//! - Forward pass trajectories
//! - Backward pass cut computation
//! - Subproblem realizations
//! - Thread-local working buffers
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use powers_rs::memory::SizingInfo;
//!
//! // 1. Compute sizing from input at startup
//! let sizing = SizingInfo::from_input(
//!     &system,
//!     &graph,
//!     &config,
//! );
//!
//! // 2. Log sizing summary for diagnostics
//! sizing.log_summary();
//!
//! // 3. Get detailed breakdown
//! let breakdown = sizing.estimate_memory_detailed();
//! println!("Cuts: {} MB", breakdown.cuts / 1_000_000);
//!
//! // 4. Use sizing to pre-allocate buffers
//! let backward_buffers = BackwardPassBuffers::new(&sizing);
//! let forward_buffers = ForwardPassBuffers::new(&sizing);
//!
//! // 5. Reuse buffers across iterations (zero allocations in hot path)
//! for iteration in 0..num_iterations {
//!     backward_pass(&mut backward_buffers, &sizing);
//!     forward_pass(&mut forward_buffers, &sizing);
//! }
//! ```
//!
//! # Performance Impact
//!
//! Based on profiling data from PROFILING_ANALYSIS.md:
//!
//! | Metric | Before | After (Target) | Improvement |
//! |--------|--------|----------------|-------------|
//! | Runtime | 34.0s | ~28.9s | -15% to -20% |
//! | Malloc overhead | 5.28% | <2% | -60% |
//! | Allocations/iter | ~60 | ~1 | -98% |
//! | Peak memory | 2.4GB | ~2.6GB | +8% (acceptable) |
//!
//! # Related Documentation
//!
//! - `PERFORMANCE_IMPLEMENTATION_PLAN.md`: Detailed implementation strategy
//! - `PROFILING_ANALYSIS.md`: Profiling data and bottleneck identification
//! - `PERFORMANCE_REFACTORING_PLAN.md`: Overall performance optimization roadmap

pub mod buffers;
pub mod sizing;

pub use buffers::{
    initialize_thread_local_buffers, with_thread_buffers, Buffer, BufferPool,
    ThreadLocalBuffers,
};
pub use sizing::{MemoryBreakdown, NodeSizing, SizingInfo};
