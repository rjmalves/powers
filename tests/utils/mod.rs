// Test utilities for POWE.RS
//
// Common helper functions and assertions for testing

pub mod assertions;

// Re-export commonly used utilities
pub use assertions::{
    assert_float_approx_eq, assert_state_within_bounds, assert_vec_approx_eq,
};
