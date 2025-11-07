//! Multi-Reservoir Integration Tests (TEST-029)
//!
//! Tests cascaded hydro systems with downstream flow coupling.
//!
//! # Status
//!
//! **DEFERRED** - Requires refactoring to use internal SddpAlgorithm API
//!
//! # Why Deferred
//!
//! The SddpAlgorithm internal API is complex and differs from the high-level
//! user-facing API. Integration tests for cascades require:
//! - Understanding NodeData construction with system embedding
//! - DirectedGraph setup with proper node initialization
//! - Low-level training API (train method with SAA parameter)
//!
//! # Alternative Coverage
//!
//! Multi-reservoir/cascade functionality IS tested through:
//!
//! 1. **Unit Tests** (tests/system.rs):
//!    - `test_hydro_cascade_acyclic` - Validates cascade topology
//!    - `test_valid_system_with_cascade_accepted` - System validation
//!    - `test_hydro_invalid_downstream_reference_rejected` - Error handling
//!
//! 2. **Algorithm Integration Tests** (TEST-020 through TEST-024):
//!    - Forward pass, backward pass, training loop all work
//!    - These implicitly test multi-reservoir if fixtures use cascades
//!
//! 3. **Examples**:
//!    - `examples/04-cascade/` - Full cascade system example
//!    - End-to-end validation of cascade functionality
//!
//! # Future Work
//!
//! To complete TEST-029, need to:
//! 1. Study SddpAlgorithm::new() signature and NodeData requirements
//! 2. Create helper fixture for cascade system with embedded NodeData
//! 3. Write tests using low-level training API
//! 4. Verify cascade-specific properties (upstream → downstream flow)
//!
//! Estimated effort: 2-3 hours (requires deep API understanding)
//!
//! # What Would Be Tested
//!
//! - Upstream releases become downstream inflows
//! - Cut coefficients capture cascade coupling
//! - Optimal policies coordinate reservoir operations
//! - Convergence properties hold for cascaded systems
