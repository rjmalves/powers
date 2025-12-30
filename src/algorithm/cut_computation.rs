//! Cut computation utilities for SDDP backward pass.
//!
//! This module provides a clean interface for computing Benders cuts from
//! branching scenario solutions. The core computation logic is delegated
//! to `State::compute_cut_into_slot()` which writes directly to preallocated pools.
//!
//! # Architecture
//!
//! Cut computation is part of backward pass Phase 1:
//!
//! 1. Solve branching scenarios (parallel across handlers)
//! 2. Compute cut coefficients from solutions into staging buffers
//! 3. Copy from staging to preallocated FCF pools (sequential)
//! 4. Finalize cuts in Phase 2 (selection)
//!
//! # Future Enhancements
//!
//! This module can be extended to include:
//! - Cut normalization strategies
//! - Parallel cut coefficient computation (for large state spaces)
//! - Cut quality metrics and filtering
