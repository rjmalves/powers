//! Integration Tests for POWE.RS SDDP Algorithm
//!
//! This module organizes integration tests that verify interactions between
//! multiple components of the SDDP implementation.
//!
//! # Organization
//!
//! - `algorithm/`: Tests for forward pass, backward pass, training loop
//! - `features/`: Tests for AR models, risk measures, scenario generation
//!
//! # Test Philosophy
//!
//! Integration tests verify that components work together correctly. They:
//! - Test realistic workflows (e.g., full training iterations)
//! - Use real LP solvers (HiGHS) not mocks
//! - Validate end-to-end behavior
//! - Check mathematical properties hold across components
//!
//! # Running Integration Tests
//!
//! ```bash
//! # Run all integration tests
//! cargo test --test integration
//!
//! # Run specific integration category
//! cargo test --test integration -- algorithm
//! cargo test --test integration -- features
//! ```

pub mod algorithm;
pub mod features;
pub mod fixtures;
