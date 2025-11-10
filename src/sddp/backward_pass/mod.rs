//! Backward pass execution with pre-allocated buffers.
//!
//! This module optimizes the SDDP backward pass by eliminating allocations in the hot path.
//! Profiling identified that backward pass allocations contribute significantly to the
//! 5.28% malloc overhead. By pre-allocating result buffers and reusing them across
//! iterations, we target an 8-10% reduction in backward pass execution time.
//!
//! # Problem
//!
//! The SDDP backward pass processes multiple forward pass trajectories to compute cuts.
//! Each trajectory traversal computes cuts at each stage (except the final stage),
//! allocating new vectors for cut-state pairs every iteration. For a typical problem
//! with 10 forward passes and 5 stages, this results in ~40-50 allocations per iteration.
//!
//! # Solution
//!
//! Pre-allocate a pool of result buffers sized appropriately using `SizingInfo`:
//! - One buffer per forward pass trajectory
//! - Buffer capacity = `max_stages - 1` (no cut at final stage)
//! - Buffers reused across iterations (clear + refill pattern)
//!
//! # Architecture
//!
//! ```text
//! SddpAlgorithm
//! └── backward_buffers: BackwardPassBuffers
//!     └── results_pool: Vec<Vec<CutStatePair>>
//!         ├── Buffer 0 (capacity: num_stages-1)
//!         ├── Buffer 1 (capacity: num_stages-1)
//!         └── ...
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::memory::SizingInfo;
//! use crate::sddp::backward_pass::BackwardPassBuffers;
//!
//! // 1. At algorithm initialization
//! let sizing = SizingInfo::from_input(&system, &graph, &config);
//! let mut buffers = BackwardPassBuffers::new(&sizing);
//!
//! // 2. In training loop (TICKET-006 will implement this)
//! for iteration in 0..num_iterations {
//!     buffers.clear_all();  // Reset for new iteration
//!     
//!     for (idx, trajectory) in trajectories.iter().enumerate() {
//!         let buffer = buffers.acquire_result_buffer(idx);
//!         
//!         // Backward pass writes directly to buffer (zero allocations!)
//!         backward_step_to_buffer(trajectory, buffer)?;
//!     }
//! }
//! ```
//!
//! # Performance Impact
//!
//! **Expected improvements** (to be validated in TICKET-013):
//! - Malloc overhead: ~2% → <1% (50% reduction in backward allocations)
//! - Backward pass time: 8-10% faster
//! - Cache hit rate: Improved from buffer reuse
//!
//! **Memory footprint** (typical system):
//! - 10 forward passes × 5 stages = 40 cut-state pair slots
//! - ~(2KB cut + 1.6KB state) × 40 = ~180KB total
//! - Negligible compared to solver memory (hundreds of MB)
//!
//! # Related Tickets
//!
//! - **TICKET-001**: Provides `SizingInfo` for buffer sizing
//! - **TICKET-002**: Provides generic `Buffer<T>` abstraction
//! - **TICKET-005** (this): Creates `BackwardPassBuffers` structure
//! - **TICKET-006** (next): Integrates buffers into backward pass algorithm

pub mod buffers;

pub use buffers::BackwardPassBuffers;
