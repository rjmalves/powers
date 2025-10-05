// Test fixtures for POWE.RS
//
// This module provides reusable test data structures for testing the SDDP algorithm.
// Fixtures are designed to be simple, well-documented, and easy to reason about.

pub mod benchmarks;
pub mod mock_solver;
pub mod oos;
pub mod scenarios;
pub mod simple_2stage_reservoir;
pub mod subproblems;
pub mod systems;
pub mod validation;

// Re-export commonly used fixtures
#[allow(unused_imports)]
pub use mock_solver::{MockSolver, MockSolverStatus};
#[allow(unused_imports)]
pub use scenarios::{
    deterministic_scenario, fan_scenario, simple_stochastic_scenario,
};
#[allow(unused_imports)]
pub use simple_2stage_reservoir::{
    create_simple_2stage_initial_condition, create_simple_2stage_system,
    expected_solution_bounds, generate_2stage_saa,
};
#[allow(unused_imports)]
pub use systems::{simple_system_json, trivial_system_json};
