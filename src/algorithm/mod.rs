//! SDDP Algorithm Phases
//!
//! This module contains the core SDDP algorithm logic, separated by phase:
//!
//! - `context`: Context structs for algorithm phases
//! - `forward_pass`: Forward simulation through the scenario tree
//! - `backward_pass`: Backward cut generation and FCF updates
//! - `cut_computation`: Benders cut calculation
//!
//! # Status
//!
//! 🚧 **In Progress**: Logic being migrated from `src/sddp/mod.rs`
//! in Epic 3: Algorithm Separation.
//!
//! # Structure
//!
//! ```text
//! algorithm/
//! ├── mod.rs
//! ├── context.rs         ✅ Complete
//! ├── forward_pass.rs    ✅ Complete
//! ├── backward_pass.rs   ⬜ Not Started
//! └── cut_computation.rs ⬜ Not Started
//! ```

pub mod context;
pub mod forward_pass;

pub use context::{
    BackwardPassContext, BackwardPassResult, BackwardStageTiming,
    ForwardPassContext, ForwardPassResult, TrajectoryTiming,
};

// Future submodules (uncomment as implemented):
// pub mod backward_pass;
// pub mod cut_computation;
