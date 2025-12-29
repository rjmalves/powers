//! LP Model Operations
//!
//! This module contains LP model building and solver interaction:
//!
//! - [`VariableIndices`]: Precomputed variable index ranges for extraction
//! - [`ConstraintIndices`]: Precomputed constraint index ranges for dual extraction
//! - [`SolutionExtractor`]: Solution extraction into domain types
//! - [`constraints`]: Constraint generation by type
//!
//! # Design
//!
//! The module uses precomputed index ranges for O(1) access during hot-path
//! solution extraction. The dual API pattern (`extract_X_into` + `extract_X`)
//! enables future SoA migration while preserving current functionality.
//!
//! # Structure
//!
//! ```text
//! model/
//! ├── mod.rs
//! ├── variable_indices.rs    ✅
//! ├── constraint_indices.rs  ✅
//! ├── solution_extract.rs    ✅
//! └── constraints/
//!     ├── mod.rs             ✅
//!     ├── bus_balance.rs     ✅
//!     ├── hydro_balance.rs   ✅
//!     └── ar_dynamics.rs     ✅
//! ```

pub mod constraint_indices;
pub mod constraints;
pub mod solution_extract;
pub mod variable_indices;

pub use constraint_indices::ConstraintIndices;
pub use solution_extract::SolutionExtractor;
pub use variable_indices::VariableIndices;

// Future submodules (uncomment as implemented):
// pub mod builder;
// pub mod solver_interface;
