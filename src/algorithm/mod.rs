//! SDDP Algorithm Phases
//!
//! This module contains the core SDDP algorithm logic, separated by phase:
//!
//! - `context`: Context structs for algorithm phases
//! - `forward_pass`: Forward simulation through the scenario tree
//! - `processor`: Backward pass processor trait
//! - `coordinator`: Parallel handler coordination
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
//! ├── processor.rs       ✅ Complete (trait definition)
//! ├── coordinator.rs     ✅ Complete
//! ├── backward_pass.rs   ✅ Complete
//! └── cut_computation.rs ✅ Complete
//! ```

pub mod backward_pass;
pub mod context;
pub mod coordinator;
pub mod cut_computation;
pub mod forward_pass;
pub mod processor;

pub use context::{
    BackwardPassContext, BackwardPassResult, BackwardStageContext,
    BackwardStageTiming, ForwardPassContext, ForwardPassResult,
    TrajectoryTiming,
};

pub use coordinator::ParallelHandlerCoordinator;

pub use processor::{
    BackwardStageProcessor, CutComputationTiming, FirstStageTiming,
    Phase1Result, Phase2Result,
};

pub use backward_pass::{
    BackwardPassTimingAccumulator, BackwardPassTimingSnapshot,
};
