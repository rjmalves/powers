//! Feature Integration Tests
//!
//! Tests that verify specific features integrate correctly with SDDP:
//! - AR models: PAR inflow modeling with lagged states
//! - Risk measures: CVaR integration with backward pass
//! - Scenario generation: Stochastic process integration
//! - Multi-reservoir: Cascaded hydro systems

pub mod ar_models;
pub mod multi_reservoir;
pub mod risk_measures;
pub mod scenario_generation;
