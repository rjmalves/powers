# [T-044] Create State Extraction Module

> **Epic**: [Epic 4: State Simplification](../00-epic-overview.md)
> **Sprint**: [Sprint 1: State Consolidation](./00-sprint-overview.md)
> **Dependencies**: [T-043](./ticket-043-pool-compatible-extensions.md)
> **Blocks**: None (sprint completion)

## Files to Read Before Starting

- `src/state.rs` - Current extraction methods
- `src/subproblem.rs` - Realization struct
- T-039 - State-Cut relationship documentation

---

## Context

### Background

State extraction from trajectories currently lives inside the State implementations:
- `extract_storage_from_trajectory()` - Get storage from last realization
- `extract_lags_from_trajectory()` - Get lagged inflows (StorageAndInflowState)

This ticket either:
1. **Option A**: Moves extraction logic to a dedicated module for reuse
2. **Option B**: Documents why extraction should stay in implementations

Based on analysis, if extraction is tightly coupled to state layout, we document rather than extract.

---

## Specification

### Analysis First

Before implementing, analyze:

1. **Coupling level**: How tightly is extraction tied to state internals?
2. **Reuse potential**: Would other code benefit from shared extraction?
3. **Performance**: Would extraction indirection hurt hot path performance?

### Option A: Extract Module (If Beneficial)

Create `src/state/extraction.rs`:

```rust
//! State coefficient extraction from trajectories.
//!
//! Provides utilities for extracting state values from subproblem
//! realizations. Used by State implementations and pool-based allocation.

use crate::subproblem::Realization;
use crate::state::StateLayout;

/// Extract storage values from the last realization in a trajectory.
///
/// Returns the final storage levels from the most recent solve.
/// O(n) where n = number of hydros.
pub fn extract_storage(trajectory: &[&Realization]) -> &[f64] {
    &trajectory.last().unwrap().final_storage
}

/// Extract lagged inflows from trajectory window.
///
/// For each hydro, extracts the last `lag_count` inflows from history.
/// Handles trajectories shorter than required by padding with zeros.
///
/// Returns Vec<Vec<f64>> indexed by [hydro_id][lag_idx].
pub fn extract_lags(
    trajectory: &[&Realization],
    layout: &StateLayout,
) -> Vec<Vec<f64>> {
    // Implementation moved from StorageAndInflowState
}
```

### Option B: Document and Keep In Place (If Tightly Coupled)

If extraction is tightly coupled to state layout and performance-critical:

1. Add documentation explaining why extraction stays in implementations
2. Ensure consistent patterns across implementations
3. Mark as internal implementation detail

---

## Acceptance Criteria

- [ ] Analysis completed documenting coupling level
- [ ] Decision made: Extract vs Document
- [ ] If Extract: Module created with shared utilities
- [ ] If Document: Clear documentation added explaining decision
- [ ] All tests pass
- [ ] Golden tests pass

---

## Implementation Guide

### Step 1: Analyze Coupling

Check current `extract_storage_from_trajectory` implementations:

```bash
grep -A 20 "extract_storage_from_trajectory" src/state.rs
```

For StorageState:
- Accesses `self.state_coefficients` directly
- Copies into internal buffer

For StorageAndInflowState:
- Uses `self.layout` for structure
- More complex internal state update

### Step 2: Evaluate Options

**Extract if**:
- Logic is >80% identical between implementations
- Other code could reuse extraction
- No performance penalty from indirection

**Keep if**:
- Implementations differ significantly
- Extraction modifies internal state
- Hot path performance is critical

### Step 3: Implement Decision

**If Extracting**:
1. Create `src/state/extraction.rs`
2. Move common extraction logic
3. Have implementations call into shared module
4. Update tests

**If Documenting**:
1. Add module-level doc explaining pattern
2. Add inline comments explaining why not extracted
3. Ensure consistent naming/patterns

### Likely Decision

Based on current code analysis:
- `extract_storage_from_trajectory` modifies `self.state_coefficients`
- It's part of the state update flow, not just extraction
- **Recommendation: Document and keep in place**

---

## Testing Requirements

### If Extracting

- [ ] Unit tests for extraction functions
- [ ] Verify implementations still work with shared module
- [ ] Performance comparison (no regression)

### If Documenting

- [ ] Existing tests pass unchanged
- [ ] Documentation review

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Document extraction pattern and decision rationale
- [ ] Update module docs if extraction module created
- [ ] Add inline comments explaining the pattern

---

## Effort Estimate

**Points**: 2
**Confidence**: Medium
**Rationale**: Depends on analysis outcome; may be documentation-only

---

## Definition of Done

- [ ] Analysis documented with decision rationale
- [ ] Either: extraction module created, OR documentation added
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] Sprint 1 complete

---

## Analysis Results (To Be Completed)

### Coupling Analysis

| Method | Modifies Self? | Uses Layout? | Hot Path? |
|--------|----------------|--------------|-----------|
| extract_storage_from_trajectory (Storage) | | | |
| extract_storage_from_trajectory (Inflow) | | | |
| extract_lags_from_trajectory | | | |

### Decision

**Choice**: [Extract / Document]

**Rationale**:
- [ ] Point 1
- [ ] Point 2
- [ ] Point 3

### Implementation

[Describe what was done]
