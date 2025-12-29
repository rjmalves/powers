# [T-024A] Revise BackwardStageProcessor Trait Signatures

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2 Revised](./00-sprint-overview.md)
> **Dependencies**: [T-023A](./ticket-023a-handler-visibility.md)
> **Blocks**: [T-025A](./ticket-025a-implement-coordinator.md)

---

## Context

### Background

The original `BackwardStageProcessor` trait (T-024) assumed the coordinator would store an FCF graph reference. To avoid unsafe code, we need to pass the FCF graph to methods that require it.

Additionally, we need to handle the method name mismatch (`branching_count()` vs `get_branching_count()`).

### Current State (from T-024)

```rust
pub trait BackwardStageProcessor {
    fn compute_cuts_parallel(&mut self, stage_ctx: &BackwardStageContext) -> Result<Phase1Result, String>;
    fn select_cuts_batch(&mut self, cut_data: Vec<CutData>, stage_ctx: &BackwardStageContext) -> Result<Phase2Result, String>;
    fn apply_cuts_parallel(&mut self, phase2_result: &Phase2Result, stage_ctx: &BackwardStageContext) -> Result<Duration, String>;
    fn eval_first_stage_bound(&mut self, stage_ctx: &BackwardStageContext) -> Result<(f64, FirstStageTiming), String>;
    fn num_forward_passes(&self) -> usize;
}
```

### Target State

```rust
pub trait BackwardStageProcessor {
    fn compute_cuts_parallel(&mut self, stage_ctx: &BackwardStageContext) -> Result<Phase1Result, String>;
    
    fn select_cuts_batch(
        &mut self,
        cut_data: Vec<CutData>,
        stage_ctx: &BackwardStageContext,
        fcf_graph: &DirectedGraph<Mutex<FutureCostFunction>>,  // NEW: FCF passed here
    ) -> Result<Phase2Result, String>;
    
    fn apply_cuts_parallel(&mut self, phase2_result: &Phase2Result, stage_ctx: &BackwardStageContext) -> Result<Duration, String>;
    fn eval_first_stage_bound(&mut self, stage_ctx: &BackwardStageContext) -> Result<(f64, FirstStageTiming), String>;
    fn num_forward_passes(&self) -> usize;
}
```

---

## Files to Read Before Starting

- `src/algorithm/processor.rs` - Current trait definition (T-024 output)
- `src/algorithm/context.rs` - `BackwardStageContext` definition
- `src/sddp/mod.rs:1895-2007` - Current Phase 2/3a implementation using FCF
- `src/fcf.rs:50-53` - `FutureCostFunction` struct

---

## Specification

### Changes to `src/algorithm/processor.rs`

#### 1. Add FCF import

```rust
use crate::fcf::FutureCostFunction;
use crate::graph::DirectedGraph;
use std::sync::Mutex;
```

#### 2. Update `select_cuts_batch` signature

Change the trait method to accept FCF graph as parameter:

```rust
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
/// * `fcf_graph` - FCF graph for cut operations (passed to avoid unsafe storage)
///
/// # Returns
///
/// * `Ok(Phase2Result)` - Selection result with cuts to apply
/// * `Err(String)` - If FCF access fails
fn select_cuts_batch(
    &mut self,
    cut_data: Vec<CutData>,
    stage_ctx: &BackwardStageContext,
    fcf_graph: &DirectedGraph<Mutex<FutureCostFunction>>,
) -> Result<Phase2Result, String>;
```

#### 3. Update doc example

Update the example in the trait documentation:

```rust
/// # Example
///
/// ```ignore
/// fn execute_backward_stage<P: BackwardStageProcessor>(
///     processor: &mut P,
///     stage_ctx: &BackwardStageContext,
///     fcf_graph: &DirectedGraph<Mutex<FutureCostFunction>>,
/// ) -> Result<(), String> {
///     // Phase 1: Parallel cut computation
///     let phase1 = processor.compute_cuts_parallel(stage_ctx)?;
///     
///     // Phase 2: Sequential cut selection (FCF passed here)
///     let phase2 = processor.select_cuts_batch(phase1.cut_data, stage_ctx, fcf_graph)?;
///     
///     // Phase 3: Parallel cut application
///     processor.apply_cuts_parallel(&phase2, stage_ctx)?;
///     
///     Ok(())
/// }
/// ```
```

#### 4. Verify `CutComputationTiming` has correct fields

Ensure the struct matches what `compute_cut_data_for_backward_step` returns:

```rust
/// Timing from Phase 1 cut computation.
#[derive(Debug, Clone, Copy, Default)]
pub struct CutComputationTiming {
    /// Time spent in model preprocessing.
    pub model_preprocessing: Duration,
    /// Time spent in solver.
    pub solver: Duration,
    /// Time spent in model postprocessing.
    pub model_postprocessing: Duration,
    /// Number of solver calls made.
    pub solver_calls: usize,
}
```

---

## Acceptance Criteria

- [ ] `select_cuts_batch` takes `fcf_graph` parameter
- [ ] Import for `FutureCostFunction`, `DirectedGraph`, `Mutex` added
- [ ] Doc example updated to show FCF passing
- [ ] `cargo build -j1` succeeds
- [ ] `cargo clippy -j1 -- -D warnings` passes

---

## Implementation Guide

### Step 1: Add imports

At the top of `src/algorithm/processor.rs`, add:

```rust
use crate::fcf::FutureCostFunction;
use crate::graph::DirectedGraph;
use std::sync::Mutex;
```

### Step 2: Update trait signature

Modify the `select_cuts_batch` method signature:

```rust
fn select_cuts_batch(
    &mut self,
    cut_data: Vec<CutData>,
    stage_ctx: &BackwardStageContext,
    fcf_graph: &DirectedGraph<Mutex<FutureCostFunction>>,
) -> Result<Phase2Result, String>;
```

### Step 3: Update documentation

Update the example and method docs to reflect the new signature.

### Step 4: Verify

```bash
cargo build -j1 2>&1 | head -50
```

Note: This will cause compilation errors if any code already implements this trait. Since T-024 just defined the trait without implementations, this should compile.

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `src/algorithm/processor.rs` | Add FCF parameter to `select_cuts_batch` |

---

## Pitfalls to Avoid

- ⚠️ The trait has no implementations yet, so changing the signature is safe
- ⚠️ Don't add FCF to other methods - only Phase 2 needs it
- ⚠️ Keep `apply_cuts_parallel` unchanged - it uses `Phase2Result.cuts` (already cloned)

---

## Testing Requirements

### Compilation Tests

- [ ] `cargo build -j1` succeeds
- [ ] `cargo clippy -j1 -- -D warnings` passes

### Unit Tests

The existing tests in `processor.rs` only test timing structs, not the trait itself. No test changes needed.

---

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Signature change is straightforward; no implementations exist yet

---

## Definition of Done

- [ ] Trait signature updated
- [ ] Imports added
- [ ] Documentation updated
- [ ] `cargo build -j1` succeeds
- [ ] `cargo clippy -j1 -- -D warnings` passes
- [ ] Code reviewed
- [ ] PR merged
