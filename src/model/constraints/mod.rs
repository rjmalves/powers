//! Constraint building for LP models.
//!
//! This module provides constraint builders for the SDDP subproblem LP model.
//! Each constraint type has its own submodule with focused, testable functions.
//!
//! # Constraint Types
//!
//! - **Bus balance**: Power balance at network buses (load = generation + imports - exports)
//! - **Hydro balance**: Water balance at hydro plants (storage + outflow = inflow + upstream)
//! - **AR dynamics**: Auto-regressive observation constraints and lag fixing
//!
//! # Design
//!
//! The constraint builders use a context struct pattern to reduce parameter counts
//! and make dependencies explicit. Each builder returns constraint indices that
//! are stored in the [`Constraints`](crate::subproblem::Constraints) struct.
//!
//! # Future: Preallocation Support
//!
//! These builders are designed to support future preallocation patterns:
//! ```ignore
//! trait ConstraintBuilder {
//!     fn build_into(
//!         &self,
//!         context: &ConstraintContext,
//!         row_indices: &mut [usize],
//!         col_indices: &mut [usize],
//!         values: &mut [f64],
//!     ) -> usize;
//! }
//! ```

pub mod ar_dynamics;
pub mod bus_balance;
pub mod hydro_balance;

pub use ar_dynamics::{
    build_lag_fixing_constraints, build_uncertainty_observation_constraints,
};
pub use bus_balance::build_bus_balance_constraints;
pub use hydro_balance::build_hydro_balance_constraints;
