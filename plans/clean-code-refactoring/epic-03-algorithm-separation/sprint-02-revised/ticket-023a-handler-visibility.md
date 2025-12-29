# [T-023A] Make Handler Methods and Timing Types Public

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2 Revised](./00-sprint-overview.md)
> **Dependencies**: [T-024](../sprint-02/ticket-024-design-processor-trait.md)
> **Blocks**: [T-024A](./ticket-024a-revise-processor-trait.md)

---

## Context

### Background

The `ParallelHandlerCoordinator` will live in `src/algorithm/coordinator.rs` and needs to call methods on `SddpTrainHandler`. Currently, some methods and types are `pub(crate)`, making them inaccessible from the `algorithm` module.

### Current State

```rust
// src/sddp/mod.rs

pub(crate) fn compute_cut_data_for_backward_step(...) -> Result<(CutData, BackwardPhase1Timing), String>
pub(crate) fn eval_first_stage_bound(...) -> Result<(f64, BranchingsTiming), String>

pub(crate) struct BackwardPhase1Timing { ... }
pub(crate) struct BranchingsTiming { ... }
```

### Target State

```rust
// src/sddp/mod.rs

pub fn compute_cut_data_for_backward_step(...) -> Result<(CutData, BackwardPhase1Timing), String>
pub fn eval_first_stage_bound(...) -> Result<(f64, BranchingsTiming), String>

pub struct BackwardPhase1Timing { ... }
pub struct BranchingsTiming { ... }
```

---

## Files to Read Before Starting

- `src/sddp/mod.rs:636-731` - `compute_cut_data_for_backward_step()` method
- `src/sddp/mod.rs:868-924` - `eval_first_stage_bound()` method
- `src/sddp/mod.rs:2427-2431` - `BackwardPhase1Timing` struct
- `src/sddp/mod.rs:927-931` - `BranchingsTiming` struct
- `src/algorithm/processor.rs` - Trait that will use these types

---

## Specification

### Changes Required

#### 1. Make `compute_cut_data_for_backward_step` public

```rust
// Line 636: Change from
pub(crate) fn compute_cut_data_for_backward_step(

// To
pub fn compute_cut_data_for_backward_step(
```

#### 2. Make `eval_first_stage_bound` public

```rust
// Line 868: Change from
pub(crate) fn eval_first_stage_bound(

// To
pub fn eval_first_stage_bound(
```

#### 3. Make `BackwardPhase1Timing` public with public fields

```rust
// Line 2427: Change from
pub(crate) struct BackwardPhase1Timing {
    model_preprocessing_time: Duration,
    solver_time: Duration,
    model_postprocessing_time: Duration,
}

// To
/// Timing data from backward pass Phase 1 (branching solves).
#[derive(Debug, Clone, Copy, Default)]
pub struct BackwardPhase1Timing {
    /// Time spent in model preprocessing.
    pub model_preprocessing_time: Duration,
    /// Time spent in solver.
    pub solver_time: Duration,
    /// Time spent in model postprocessing.
    pub model_postprocessing_time: Duration,
}
```

#### 4. Make `BranchingsTiming` public with public fields

```rust
// Line 927: Change from
pub(crate) struct BranchingsTiming {
    pub solver_time: Duration,
    pub state_extraction_time: Duration,
}

// To
/// Timing data from branching solves.
#[derive(Debug, Clone, Copy, Default)]
pub struct BranchingsTiming {
    /// Time spent in solver.
    pub solver_time: Duration,
    /// Time spent extracting state.
    pub state_extraction_time: Duration,
}
```

#### 5. Export types from sddp module

Add to module exports (near top of `src/sddp/mod.rs` after existing `pub use` statements):

```rust
pub use crate::sddp::{BackwardPhase1Timing, BranchingsTiming, SddpTrainHandler};
```

Or ensure they are accessible via `crate::sddp::BackwardPhase1Timing` etc.

---

## Acceptance Criteria

- [ ] `SddpTrainHandler::compute_cut_data_for_backward_step` is `pub`
- [ ] `SddpTrainHandler::eval_first_stage_bound` is `pub`
- [ ] `BackwardPhase1Timing` is `pub` with `pub` fields
- [ ] `BranchingsTiming` is `pub` with `pub` fields
- [ ] Both timing types have `#[derive(Debug, Clone, Copy, Default)]`
- [ ] Both timing types have doc comments
- [ ] `cargo build -j1` succeeds
- [ ] `cargo clippy -j1 -- -D warnings` passes
- [ ] All existing tests pass

---

## Implementation Guide

### Step 1: Update method visibility

Edit `src/sddp/mod.rs`:
- Line 636: `pub(crate) fn` → `pub fn`
- Line 868: `pub(crate) fn` → `pub fn`

### Step 2: Update BackwardPhase1Timing

Edit `src/sddp/mod.rs` around line 2427:

```rust
/// Timing data from backward pass Phase 1 (branching solves).
///
/// Captures the time spent in each phase of cut computation for a single stage.
#[derive(Debug, Clone, Copy, Default)]
pub struct BackwardPhase1Timing {
    /// Time spent in model preprocessing (state setup, basis reuse).
    pub model_preprocessing_time: Duration,
    /// Time spent in LP solver.
    pub solver_time: Duration,
    /// Time spent in model postprocessing (extracting results).
    pub model_postprocessing_time: Duration,
}
```

### Step 3: Update BranchingsTiming

Edit `src/sddp/mod.rs` around line 927:

```rust
/// Timing data from branching solves.
///
/// Used for first-stage evaluation timing.
#[derive(Debug, Clone, Copy, Default)]
pub struct BranchingsTiming {
    /// Time spent in LP solver.
    pub solver_time: Duration,
    /// Time spent extracting state after solve.
    pub state_extraction_time: Duration,
}
```

### Step 4: Verify compilation

```bash
cargo build -j1 2>&1 | head -50
cargo clippy -j1 -- -D warnings 2>&1 | head -50
```

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `src/sddp/mod.rs` | Make methods and types public |

---

## Pitfalls to Avoid

- ⚠️ Don't change field names yet - that's for T-024A
- ⚠️ Don't modify method signatures - just visibility
- ⚠️ Ensure `Default` derive works (all fields must have Default)

---

## Testing Requirements

### Compilation Tests

- [ ] `cargo build -j1` succeeds
- [ ] `cargo clippy -j1 -- -D warnings` passes

### Regression Tests

- [ ] `cargo test` passes (no behavioral changes)
- [ ] Golden tests pass (no output changes)

---

## Documentation Requirements

- [ ] Add doc comments to `BackwardPhase1Timing` struct
- [ ] Add doc comments to `BranchingsTiming` struct
- [ ] Document each field's purpose

---

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Simple visibility changes, no logic modifications

---

## Definition of Done

- [ ] All visibility changes applied
- [ ] Doc comments added
- [ ] `cargo build -j1` succeeds
- [ ] `cargo clippy -j1 -- -D warnings` passes
- [ ] `cargo test` passes
- [ ] Code reviewed
- [ ] PR merged
