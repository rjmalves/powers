# [T-027] Unit Tests for Coordinator

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2 Revised](./00-sprint-overview.md)
> **Dependencies**: [T-026](./ticket-026-migrate-handlers.md)
> **Blocks**: Sprint 3

---

## Context

This ticket adds comprehensive unit tests for `ParallelHandlerCoordinator` to verify the trait implementation works correctly and to enable future refactoring with confidence.

T-025A already includes basic unit tests for timing scaling. This ticket adds:
1. More thorough timing edge case tests
2. Mock-based trait implementation tests
3. Verification that the trait methods are callable

---

## Files to Read Before Starting

- `src/algorithm/coordinator.rs` - Coordinator implementation (from T-025A)
- `src/algorithm/processor.rs` - `BackwardStageProcessor` trait definition
- Existing tests in `src/algorithm/coordinator.rs` - Unit tests from T-025A

---

## Specification

### Additional Tests in `src/algorithm/coordinator.rs`

Add to the existing `#[cfg(test)]` module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // ... existing tests from T-025A ...

    #[test]
    fn test_coordinator_handlers_mut() {
        let coordinator = ParallelHandlerCoordinator::new(Vec::new());
        let handlers = coordinator.handlers();
        assert!(handlers.is_empty());
        
        // Can't easily test with real handlers without full setup,
        // but we verify the accessor works
    }

    #[test]
    fn test_scale_timing_zero_internal() {
        // When internal timing is zero, should return default
        let coordinator = ParallelHandlerCoordinator::new(Vec::new());
        let timings = vec![BackwardPhase1Timing::default()];
        let result = coordinator.scale_timing(&timings, Duration::from_secs(1), 5);
        
        // With zero internal timing, should return default
        assert_eq!(result.model_preprocessing, Duration::ZERO);
        assert_eq!(result.solver, Duration::ZERO);
        assert_eq!(result.model_postprocessing, Duration::ZERO);
        // But solver_calls should still be set
        assert_eq!(result.solver_calls, 5);
    }

    #[test]
    fn test_scale_timing_preserves_ratios() {
        let coordinator = ParallelHandlerCoordinator::new(Vec::new());
        
        // 10:60:30 ratio (prep:solver:post)
        let timings = vec![BackwardPhase1Timing {
            model_preprocessing_time: Duration::from_millis(10),
            solver_time: Duration::from_millis(60),
            model_postprocessing_time: Duration::from_millis(30),
        }];
        
        // Scale to 1 second wall time
        let result = coordinator.scale_timing(&timings, Duration::from_secs(1), 100);
        
        // Verify ratios preserved (10:60:30 = 100:600:300 ms)
        assert_eq!(result.model_preprocessing, Duration::from_millis(100));
        assert_eq!(result.solver, Duration::from_millis(600));
        assert_eq!(result.model_postprocessing, Duration::from_millis(300));
        assert_eq!(result.solver_calls, 100);
    }

    #[test]
    fn test_scale_timing_multiple_handlers_averages() {
        let coordinator = ParallelHandlerCoordinator::new(Vec::new());
        
        // Two handlers with different timings
        let timings = vec![
            BackwardPhase1Timing {
                model_preprocessing_time: Duration::from_millis(100),
                solver_time: Duration::from_millis(200),
                model_postprocessing_time: Duration::from_millis(100),
            },
            BackwardPhase1Timing {
                model_preprocessing_time: Duration::from_millis(200),
                solver_time: Duration::from_millis(400),
                model_postprocessing_time: Duration::from_millis(200),
            },
        ];
        
        // Averages: prep=150, solver=300, post=150 (total=600)
        // Wall time 1200ms = 2x scaling
        let result = coordinator.scale_timing(&timings, Duration::from_millis(1200), 50);
        
        assert_eq!(result.model_preprocessing, Duration::from_millis(300));
        assert_eq!(result.solver, Duration::from_millis(600));
        assert_eq!(result.model_postprocessing, Duration::from_millis(300));
        assert_eq!(result.solver_calls, 50);
    }
}
```

### Documentation Test

Ensure the module documentation has a working example:

```rust
//! # Example (conceptual)
//!
//! ```ignore
//! use powers_rs::algorithm::{ParallelHandlerCoordinator, BackwardStageProcessor};
//! 
//! let coordinator = ParallelHandlerCoordinator::new(handlers);
//! 
//! // Forward pass uses handlers directly
//! for handler in coordinator.handlers_mut() {
//!     // ... forward pass operations ...
//! }
//! 
//! // Backward pass uses trait methods
//! let phase1 = coordinator.compute_cuts_parallel(&stage_ctx)?;
//! let phase2 = coordinator.select_cuts_batch(phase1.cut_data, &stage_ctx, &fcf_graph)?;
//! coordinator.apply_cuts_parallel(&phase2, &stage_ctx)?;
//! ```
```

---

## Acceptance Criteria

- [ ] Additional unit tests for edge cases
- [ ] Test for zero internal timing
- [ ] Test for ratio preservation
- [ ] Test for multi-handler averaging
- [ ] All tests pass with `RUST_TEST_THREADS=1 cargo test -j1 coordinator`
- [ ] Tests document expected behavior
- [ ] Code coverage for `scale_timing` function is 100%

---

## Implementation Guide

### Step 1: Add additional tests

Add the new tests to the existing `#[cfg(test)]` module in `coordinator.rs`.

### Step 2: Run tests

```bash
RUST_TEST_THREADS=1 cargo test -j1 coordinator -- --nocapture
```

### Step 3: Verify coverage

Check that all branches in `scale_timing` are covered:
- Empty input
- Zero internal timing
- Normal timing with scaling

---

## Key Files to Modify

| File | Action |
|------|--------|
| `src/algorithm/coordinator.rs` | ADD additional unit tests |

---

## Testing Requirements

### Unit Tests

- [ ] `test_coordinator_handlers_mut` - Accessor works correctly
- [ ] `test_scale_timing_zero_internal` - Zero timing returns default
- [ ] `test_scale_timing_preserves_ratios` - Ratios maintained after scaling
- [ ] `test_scale_timing_multiple_handlers_averages` - Multiple handlers averaged

### Edge Cases

- [ ] Empty handler list
- [ ] Single handler
- [ ] All-zero timing
- [ ] Very large timing values

---

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Testing existing implementation, clear scope

---

## Definition of Done

- [ ] All unit tests added
- [ ] Tests pass with single thread
- [ ] Tests document expected behavior
- [ ] Edge cases covered
- [ ] Code reviewed
