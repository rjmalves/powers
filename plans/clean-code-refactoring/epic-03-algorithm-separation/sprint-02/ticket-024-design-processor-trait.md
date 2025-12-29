# [T-024] Design BackwardStageProcessor Trait

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Handler Coordination Infrastructure](./00-sprint-overview.md)
> **Dependencies**: [T-023](./ticket-023-backward-pass-context.md)
> **Blocks**: [T-025](./ticket-025-implement-coordinator.md)

---

## ⚠️ CRITICAL: Design Only

This ticket defines the `BackwardStageProcessor` trait interface. **Minimal implementation**—just the trait definition and associated types. The implementation comes in T-025.

---

## Files to Read Before Starting

- `src/sddp/mod.rs:1793-2027` - Current 3-phase backward architecture
- `src/sddp/mod.rs:636-731` - `compute_cut_data_for_backward_step()`
- `src/fcf.rs` - `CutData`, `BatchCutSelectionResult`, `AggregatedCutSelectionResult`
- `src/algorithm/context.rs` - `BackwardStageContext`
- `BACKWARD_PASS_EXTRACTION_ANALYSIS.md` - Architecture rationale

---

## Context

### The 3-Phase Backward Architecture

The current backward pass has a strict 3-phase structure per stage:

```
Phase 1: PARALLEL - Compute cuts across all forward passes
         → par_iter_mut on handlers
         → Returns Vec<CutData>
         
Phase 2: SEQUENTIAL - Batch cut selection (deterministic ordering)
         → Sort by forward_pass_idx for reproducibility
         → Apply cut selection to FCF
         → Returns BatchCutSelectionResult
         
Phase 3a: SEQUENTIAL - Update FCF state (mark inactive)
Phase 3b: PARALLEL - Apply results to all models
         → par_iter_mut on handlers
```

### Why a Trait?

A trait enables:
1. **Extraction**: The backward loop can call trait methods without knowing about handlers
2. **Encapsulation**: Parallel coordination details hidden in implementation
3. **Testability**: Can create mock implementations for unit testing
4. **Extensibility**: Future implementations (e.g., distributed) can use same interface

---

## Specification

### Create `src/algorithm/processor.rs`

```rust
//! Backward pass processor traits.
//!
//! These traits abstract the parallel coordination of backward pass stages,
//! enabling extraction of the backward pass loop while hiding handler details.

use crate::algorithm::context::BackwardStageContext;
use crate::cut::BendersCut;
use crate::fcf::{AggregatedCutSelectionResult, BatchCutSelectionResult, CutData};
use std::sync::Arc;
use std::time::Duration;

/// Timing from Phase 1 cut computation.
#[derive(Debug, Clone, Copy, Default)]
pub struct CutComputationTiming {
    pub model_preprocessing: Duration,
    pub solver: Duration,
    pub model_postprocessing: Duration,
    pub solver_calls: usize,
}

/// Timing from first stage evaluation.
#[derive(Debug, Clone, Copy, Default)]
pub struct FirstStageTiming {
    pub solver: Duration,
    pub state_extraction: Duration,
}

/// Result of Phase 1 cut computation.
#[derive(Debug)]
pub struct Phase1Result {
    /// Cut data from all forward passes.
    pub cut_data: Vec<CutData>,
    /// Aggregated timing from parallel computation.
    pub timing: CutComputationTiming,
}

/// Result of Phase 2 cut selection.
#[derive(Debug)]
pub struct Phase2Result {
    /// Batch selection result from FCF.
    pub batch_result: BatchCutSelectionResult,
    /// Aggregated result for handler application.
    pub aggregated: AggregatedCutSelectionResult,
    /// Cloned cuts for parallel application.
    pub cuts: Vec<(usize, Arc<BendersCut>)>,
    /// Time spent in cut selection.
    pub cut_selection_time: Duration,
    /// Time spent updating FCF state.
    pub fcf_update_time: Duration,
    /// Time spent cloning cuts.
    pub cut_cloning_time: Duration,
}

/// Trait for processing backward pass stages.
///
/// This trait abstracts the 3-phase backward pass processing, enabling
/// the backward loop to be extracted while encapsulating parallel
/// coordination details.
///
/// # Thread Safety
///
/// Implementations must handle parallel execution in Phase 1 and Phase 3b.
/// The trait methods are called sequentially from the backward loop.
///
/// # Example
///
/// ```ignore
/// fn execute_backward_stage<P: BackwardStageProcessor>(
///     processor: &mut P,
///     stage_ctx: &BackwardStageContext,
/// ) -> Result<(), Error> {
///     // Phase 1: Parallel cut computation
///     let phase1 = processor.compute_cuts_parallel(stage_ctx)?;
///     
///     // Phase 2: Sequential cut selection
///     let phase2 = processor.select_cuts_batch(phase1.cut_data, stage_ctx)?;
///     
///     // Phase 3: Parallel cut application
///     processor.apply_cuts_parallel(&phase2, stage_ctx)?;
///     
///     Ok(())
/// }
/// ```
pub trait BackwardStageProcessor {
    /// Phase 1: Compute cuts in parallel across all handlers.
    ///
    /// Each handler computes cut data for its branching scenarios.
    /// Results are collected into a vector for Phase 2 processing.
    ///
    /// # Arguments
    ///
    /// * `stage_ctx` - Per-stage context with node and scenario info
    ///
    /// # Returns
    ///
    /// * `Ok(Phase1Result)` - Cut data and timing from all handlers
    /// * `Err(String)` - If any handler fails
    fn compute_cuts_parallel(
        &mut self,
        stage_ctx: &BackwardStageContext,
    ) -> Result<Phase1Result, String>;

    /// Phase 2: Sequential batch cut selection.
    ///
    /// Sorts cuts for deterministic ordering, applies cut selection to FCF,
    /// and prepares results for Phase 3. This phase is ALWAYS sequential
    /// to ensure reproducibility.
    ///
    /// # Arguments
    ///
    /// * `cut_data` - Cut data from Phase 1 (will be sorted by forward_pass_idx)
    /// * `stage_ctx` - Per-stage context
    ///
    /// # Returns
    ///
    /// * `Ok(Phase2Result)` - Selection result with cuts to apply
    /// * `Err(String)` - If FCF access fails
    fn select_cuts_batch(
        &mut self,
        cut_data: Vec<CutData>,
        stage_ctx: &BackwardStageContext,
    ) -> Result<Phase2Result, String>;

    /// Phase 3: Apply cut results in parallel to all handler models.
    ///
    /// Updates all handler subproblem models with the new cuts.
    /// This phase can be parallel since each handler has independent models.
    ///
    /// # Arguments
    ///
    /// * `phase2_result` - Results from Phase 2 with cuts to apply
    /// * `stage_ctx` - Per-stage context
    ///
    /// # Returns
    ///
    /// * `Ok(Duration)` - Time spent in handler application
    /// * `Err(String)` - If any handler fails
    fn apply_cuts_parallel(
        &mut self,
        phase2_result: &Phase2Result,
        stage_ctx: &BackwardStageContext,
    ) -> Result<Duration, String>;

    /// Evaluate first stage bound (no cut generation).
    ///
    /// Called for the first stage where no cuts are generated,
    /// only the lower bound is computed.
    ///
    /// # Arguments
    ///
    /// * `stage_ctx` - Per-stage context for first stage
    ///
    /// # Returns
    ///
    /// * `Ok((lower_bound, timing))` - Lower bound and evaluation timing
    /// * `Err(String)` - If evaluation fails
    fn eval_first_stage_bound(
        &mut self,
        stage_ctx: &BackwardStageContext,
    ) -> Result<(f64, FirstStageTiming), String>;

    /// Get the number of forward passes (handlers).
    fn num_forward_passes(&self) -> usize;
}
```

### Update `src/algorithm/mod.rs`

```rust
pub mod processor;
pub use processor::{
    BackwardStageProcessor, CutComputationTiming, FirstStageTiming,
    Phase1Result, Phase2Result,
};
```

---

## Acceptance Criteria

- [ ] `src/algorithm/processor.rs` created with trait definition
- [ ] All associated types defined (`Phase1Result`, `Phase2Result`, timing types)
- [ ] Comprehensive doc comments on trait and all methods
- [ ] Module exported in `src/algorithm/mod.rs`
- [ ] `cargo build -j1` succeeds
- [ ] `cargo doc --no-deps` generates clean documentation
- [ ] No implementation yet (just trait definition)

---

## Implementation Guide

### Step 1: Create processor.rs

Create `src/algorithm/processor.rs` with the trait definition as specified above.

### Step 2: Add module to mod.rs

```rust
pub mod processor;
pub use processor::{...};
```

### Step 3: Verify documentation

```bash
cargo doc --no-deps --open
```

Check that trait documentation is clear and complete.

---

## Key Files to Create/Modify

| File | Action |
|------|--------|
| `src/algorithm/processor.rs` | CREATE with trait definition |
| `src/algorithm/mod.rs` | UPDATE exports |

---

## Pitfalls to Avoid

- ⚠️ Don't implement the trait yet—that's T-025
- ⚠️ Keep types simple—avoid over-engineering
- ⚠️ Use `String` for errors (matches existing pattern)
- ⚠️ Document thread safety requirements clearly

---

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Design-only ticket, clear specification from analysis

---

## Definition of Done

- [ ] Trait defined with all methods
- [ ] Associated types defined
- [ ] Documentation complete
- [ ] Module exported
- [ ] `cargo build -j1` succeeds
- [ ] Code reviewed
