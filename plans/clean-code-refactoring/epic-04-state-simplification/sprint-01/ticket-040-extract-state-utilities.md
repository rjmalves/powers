# [T-040] Extract Common State Utilities

> **Epic**: [Epic 4: State Simplification](../00-epic-overview.md)
> **Sprint**: [Sprint 1: State Consolidation](./00-sprint-overview.md)
> **Dependencies**: [T-038](./ticket-038-analyze-state-structure.md)
> **Blocks**: [T-041](./ticket-041-consolidate-storage-state.md), [T-042](./ticket-042-consolidate-inflow-state.md)

## Files to Read Before Starting

- `src/state.rs` - Current state implementations
- T-038 analysis results - Identified duplication patterns
- `src/memory/mod.rs` - Existing memory module pattern

---

## Context

### Background

T-038 identified significant duplication between `StorageState` and `StorageAndInflowState`. Both implementations share:
- Identical getter/setter implementations for domination tracking
- Identical getter/setter implementations for iteration tracking  
- Similar patterns for coefficient access

This ticket extracts shared functionality to eliminate duplication.

### Current State

Both state types have these identical fields:
```rust
struct {
    dimension: usize,
    state_coefficients: Vec<f64>,
    dominating_objective: f64,
    dominating_cut_id: usize,
    iteration: usize,
    forward_pass_idx: usize,
}
```

And identical implementations for ~12 trait methods.

---

## Specification

### Approach: Common Base Struct

Create a shared struct containing common fields and provide delegating implementations:

```rust
// src/state/common.rs (NEW)

/// Common state fields shared by all State implementations.
/// 
/// Contains domination tracking, iteration tracking, and coefficient storage.
/// State implementations embed this struct and delegate common methods.
#[derive(Debug, Clone)]
pub struct StateCore {
    pub dimension: usize,
    pub state_coefficients: Vec<f64>,
    pub dominating_objective: f64,
    pub dominating_cut_id: usize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
}

impl StateCore {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            state_coefficients: vec![0.0; dimension],
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }

    pub fn with_coefficients(state_coefficients: Vec<f64>) -> Self {
        let dimension = state_coefficients.len();
        Self {
            dimension,
            state_coefficients,
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }

    // Common implementations
    #[inline]
    pub fn coefficients(&self) -> &[f64] {
        &self.state_coefficients
    }

    #[inline]
    pub fn update_coefficients(&mut self, coefficients: &[f64]) {
        debug_assert_eq!(self.state_coefficients.len(), coefficients.len());
        self.state_coefficients.copy_from_slice(coefficients);
    }

    pub fn reset_to_zero(&mut self) {
        self.state_coefficients.fill(0.0);
        self.dominating_objective = 0.0;
        self.dominating_cut_id = 0;
        self.iteration = 0;
        self.forward_pass_idx = 0;
    }

    // ... other common getters/setters
}
```

### Changes Required

1. **Create `src/state/common.rs`** with `StateCore` struct
2. **Update `src/state.rs`** to re-export from submodule
3. **DO NOT modify `StorageState` or `StorageAndInflowState` yet** (T-041, T-042)

---

## Acceptance Criteria

- [ ] `StateCore` struct created with all common fields
- [ ] Common methods implemented on `StateCore`
- [ ] Module exports via `src/state.rs` or `src/state/mod.rs`
- [ ] All existing tests pass (no behavior changes yet)
- [ ] Golden tests pass

---

## Implementation Guide

### Suggested Approach

1. **Create new file**:
   ```bash
   touch src/state/common.rs  # Or add to state.rs initially
   ```

2. **Extract fields** that are identical between implementations:
   - `dimension: usize`
   - `state_coefficients: Vec<f64>`
   - `dominating_objective: f64`
   - `dominating_cut_id: usize`
   - `iteration: usize`
   - `forward_pass_idx: usize`

3. **Implement common methods**:
   - `coefficients()` → `&self.state_coefficients`
   - `update_coefficients()` → copy_from_slice
   - `reset_to_zero()` → fill + reset tracking fields
   - All getter/setter pairs

4. **Add module declaration** in `state.rs`:
   ```rust
   mod common;
   pub use common::StateCore;
   ```

### Key Pattern

**Composition over inheritance** - Rust doesn't have inheritance, so we use composition:

```rust
// Future usage in T-041/T-042:
pub struct StorageState {
    core: StateCore,  // Embed shared fields
}

impl StorageState {
    fn coefficients(&self) -> &[f64] {
        self.core.coefficients()  // Delegate
    }
}
```

### Pitfalls to Avoid

- ⚠️ **Don't modify existing structs yet** - That's T-041/T-042
- ⚠️ **Ensure `#[derive(Debug, Clone)]`** for compatibility
- ⚠️ **Keep same visibility** as original fields

---

## Testing Requirements

### Unit Tests

- [ ] `StateCore::new()` creates with correct dimension
- [ ] `StateCore::with_coefficients()` captures coefficients
- [ ] `coefficients()` returns slice reference
- [ ] `update_coefficients()` copies correctly
- [ ] `reset_to_zero()` clears all tracking fields
- [ ] Getter/setter pairs work correctly

### Regression Tests

- [ ] All existing state tests pass (no modifications yet)
- [ ] Golden tests pass

---

## Documentation Requirements

- [ ] Doc comments on `StateCore` struct
- [ ] Doc comments on all public methods
- [ ] Module-level documentation explaining purpose

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Extract and test, no integration complexity yet

---

## Definition of Done

- [ ] `StateCore` struct implemented with all common fields
- [ ] All common methods implemented
- [ ] Unit tests for `StateCore`
- [ ] Module properly exported
- [ ] All existing tests pass
- [ ] Golden tests pass
- [ ] Ready for T-041 and T-042
