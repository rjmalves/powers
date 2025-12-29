# [T-027] Unit Tests for Coordinator

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Handler Coordination Infrastructure](./00-sprint-overview.md)
> **Dependencies**: [T-026](./ticket-026-migrate-handlers.md)
> **Blocks**: Sprint 3

---

## Context

This ticket adds unit tests for `ParallelHandlerCoordinator` to verify the trait implementation works correctly and to enable future refactoring with confidence.

---

## Files to Read Before Starting

- `src/algorithm/coordinator.rs` - Coordinator implementation
- `src/algorithm/processor.rs` - Trait definition
- `tests/` - Existing test patterns

---

## Specification

### Test Categories

1. **Coordinator construction tests**
2. **Phase 1 timing aggregation tests**
3. **Phase 2 deterministic ordering tests**
4. **Integration tests with real handlers**

### Test File: `src/algorithm/coordinator.rs` (inline tests)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_num_forward_passes() {
        // Test that num_forward_passes returns correct count
    }

    #[test]
    fn test_cut_computation_timing_aggregation() {
        // Test that timing is correctly aggregated from multiple handlers
    }

    #[test]
    fn test_cut_data_sorted_by_forward_pass_idx() {
        // Verify deterministic ordering in select_cuts_batch
    }
}
```

### Integration Test: `tests/test_coordinator_integration.rs`

```rust
//! Integration tests for ParallelHandlerCoordinator.

use powers_rs::algorithm::coordinator::ParallelHandlerCoordinator;
use powers_rs::algorithm::processor::BackwardStageProcessor;

/// Test that coordinator correctly wraps handlers.
#[test]
fn test_coordinator_wraps_handlers() {
    // Create minimal test setup
    // Verify handlers are accessible
}

/// Test backward stage processing produces same results as direct handler calls.
#[test]  
fn test_backward_stage_equivalence() {
    // Compare coordinator output to original inline code
}
```

---

## Acceptance Criteria

- [ ] Unit tests for coordinator construction
- [ ] Unit tests for timing aggregation
- [ ] Test for deterministic cut ordering
- [ ] Integration test comparing coordinator to original behavior
- [ ] All tests pass with `RUST_TEST_THREADS=1 cargo test -j1`
- [ ] Tests document expected behavior

---

## Implementation Guide

### Step 1: Add inline unit tests

Add `#[cfg(test)]` module to `coordinator.rs`.

### Step 2: Create integration test

Create `tests/test_coordinator_integration.rs` if complex setup needed.

### Step 3: Run tests

```bash
RUST_TEST_THREADS=1 cargo test -j1 coordinator
```

---

## Key Files to Modify

| File | Action |
|------|--------|
| `src/algorithm/coordinator.rs` | ADD inline tests |
| `tests/test_coordinator_integration.rs` | CREATE if needed |

---

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Testing existing implementation, clear scope

---

## Definition of Done

- [ ] Unit tests cover key coordinator functionality
- [ ] Tests pass with single thread
- [ ] Tests document expected behavior
- [ ] Code reviewed
