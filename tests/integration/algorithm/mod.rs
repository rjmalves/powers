//! Algorithm Integration Tests
//!
//! Tests that verify core SDDP algorithm components work together:
//! - Forward pass: state initialization, uncertainty, simulation
//! - Backward pass: cut generation, dual extraction, cut storage
//! - Training loop: iteration management, convergence detection
//! - Full algorithm: end-to-end SDDP training

pub mod backward_pass;
pub mod convergence;
pub mod forward_pass;
pub mod training_loop;
