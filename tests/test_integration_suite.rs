//! Integration Test Entry Point
//!
//! Runs integration tests from the tests/integration/ module.

#[path = "integration/mod.rs"]
mod integration;

// Re-export fixtures for integration tests
#[path = "fixtures/mod.rs"]
mod fixtures;

#[cfg(test)]
mod tests {
    /// Smoke test to verify integration test infrastructure works
    #[test]
    fn test_integration_infrastructure_exists() {
        // Integration test structure is in place
        assert!(true);
    }
}
