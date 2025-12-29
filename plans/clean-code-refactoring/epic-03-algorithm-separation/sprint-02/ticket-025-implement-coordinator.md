# [T-025] Implement ParallelHandlerCoordinator

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Handler Coordination Infrastructure](./00-sprint-overview.md)
> **Dependencies**: [T-024](./ticket-024-design-processor-trait.md)
> **Blocks**: [T-026](./ticket-026-migrate-handlers.md)

---

## ⚠️ CRITICAL: Structural Change Only

This ticket implements the `ParallelHandlerCoordinator` struct and its `BackwardStageProcessor` implementation. **No changes to algorithm logic**—we are wrapping existing functionality.

---

## Files to Read Before Starting

- `src/algorithm/processor.rs` - `BackwardStageProcessor` trait (from T-024)
- `src/sddp/mod.rs:319-345` - `SddpTrainHandler` struct
- `src/sddp/mod.rs:636-731` - `compute_cut_data_for_backward_step()`
- `src/sddp/mod.rs:733-760` - `apply_aggregated_cut_result()`
- `src/sddp/mod.rs:1793-2027` - Current backward pass phases

---

## Context

### Current State

`SddpTrainHandler` is defined in `sddp/mod.rs` and handlers are managed directly in the `train()` function:

```rust
let mut train_handlers: Vec<SddpTrainHandler> = (0..num_forward_passes)
    .map(|_| SddpTrainHandler::new(...))
    .collect::<Result<_, _>>()?;
```

Parallel operations use `par_iter_mut()` on this vector directly.

### Target State

`ParallelHandlerCoordinator` wraps the handlers and implements `BackwardStageProcessor`:

```rust
pub struct ParallelHandlerCoordinator {
    handlers: Vec<SddpTrainHandler>,
    num_forward_passes: usize,
}

impl BackwardStageProcessor for ParallelHandlerCoordinator { ... }
```

---

## Specification

### Create `src/algorithm/coordinator.rs`

```rust
//! Parallel handler coordination for SDDP training.
//!
//! This module provides `ParallelHandlerCoordinator` which encapsulates
//! the management of multiple `SddpTrainHandler` instances and implements
//! the `BackwardStageProcessor` trait for backward pass execution.

use crate::algorithm::context::BackwardStageContext;
use crate::algorithm::processor::{
    BackwardStageProcessor, CutComputationTiming, FirstStageTiming,
    Phase1Result, Phase2Result,
};
use crate::cut::BendersCut;
use crate::fcf::{AggregatedCutSelectionResult, BatchCutSelectionResult, CutData, FutureCostFunction};
use crate::graph::DirectedGraph;
use crate::sddp::{BackwardPhase1Timing, SddpTrainHandler};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Coordinates parallel execution across train handlers.
///
/// Encapsulates `Vec<SddpTrainHandler>` and implements `BackwardStageProcessor`.
/// This enables the backward pass loop to be extracted while keeping
/// parallel coordination details hidden.
///
/// # Thread Safety
///
/// The coordinator owns the handlers and uses `par_iter_mut()` for parallel
/// phases. The FCF access uses `Mutex` as in the original implementation.
///
/// # Future Extensions
///
/// This is where buffer pools will be added in Epic 5:
/// - `TrajectoryPool` for forward pass buffers
/// - `CutStatePool` for state management
pub struct ParallelHandlerCoordinator {
    /// The train handlers, one per forward pass.
    handlers: Vec<SddpTrainHandler>,
    
    /// Number of forward passes.
    num_forward_passes: usize,
    
    /// Reference to FCF graph for Phase 2/3a operations.
    /// Note: This is a shared reference, not owned.
    fcf_graph: *const DirectedGraph<Mutex<FutureCostFunction>>,
}

// Safety: The fcf_graph pointer is only used during method calls
// where the caller guarantees the graph is valid.
unsafe impl Send for ParallelHandlerCoordinator {}
unsafe impl Sync for ParallelHandlerCoordinator {}

impl ParallelHandlerCoordinator {
    /// Create a new coordinator with the given handlers.
    ///
    /// # Safety
    ///
    /// The `fcf_graph` reference must remain valid for the lifetime
    /// of the coordinator.
    pub fn new(
        handlers: Vec<SddpTrainHandler>,
        fcf_graph: &DirectedGraph<Mutex<FutureCostFunction>>,
    ) -> Self {
        let num_forward_passes = handlers.len();
        Self {
            handlers,
            num_forward_passes,
            fcf_graph: fcf_graph as *const _,
        }
    }
    
    /// Get mutable access to handlers (for forward pass).
    pub fn handlers_mut(&mut self) -> &mut [SddpTrainHandler] {
        &mut self.handlers
    }
    
    /// Get the FCF graph reference.
    fn fcf_graph(&self) -> &DirectedGraph<Mutex<FutureCostFunction>> {
        // Safety: Caller guarantees fcf_graph is valid
        unsafe { &*self.fcf_graph }
    }
}

impl BackwardStageProcessor for ParallelHandlerCoordinator {
    fn compute_cuts_parallel(
        &mut self,
        stage_ctx: &BackwardStageContext,
    ) -> Result<Phase1Result, String> {
        let phase1_begin = Instant::now();
        
        // Phase 1: Parallel cut computation
        let results: Vec<(CutData, BackwardPhase1Timing)> = self.handlers
            .par_iter_mut()
            .enumerate()
            .map(|(fp_idx, handler)| {
                handler.compute_cut_data_for_backward_step(
                    stage_ctx.stage_id,
                    stage_ctx.past_node_ids,
                    stage_ctx.node_data_graph,
                    stage_ctx.saa,
                    stage_ctx.iteration,
                    fp_idx,
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        
        let phase1_time = phase1_begin.elapsed();
        
        // Unzip results with pre-allocated capacity
        let mut cut_data = Vec::with_capacity(results.len());
        let mut timings = Vec::with_capacity(results.len());
        
        for (data, timing) in results {
            cut_data.push(data);
            timings.push(timing);
        }
        
        // Aggregate timing (average across handlers)
        let n = timings.len() as u32;
        let raw_model_prep: Duration = timings.iter()
            .map(|t| t.model_preprocessing_time)
            .sum::<Duration>() / n;
        let raw_solver: Duration = timings.iter()
            .map(|t| t.solver_time)
            .sum::<Duration>() / n;
        let raw_model_post: Duration = timings.iter()
            .map(|t| t.model_postprocessing_time)
            .sum::<Duration>() / n;
        
        // Scale to wall clock time
        let internal_total = raw_model_prep + raw_solver + raw_model_post;
        let timing = if internal_total > Duration::ZERO {
            let scale = phase1_time.as_secs_f64() / internal_total.as_secs_f64();
            CutComputationTiming {
                model_preprocessing: raw_model_prep.mul_f64(scale),
                solver: raw_solver.mul_f64(scale),
                model_postprocessing: raw_model_post.mul_f64(scale),
                solver_calls: stage_ctx.branching_count().unwrap_or(1) * self.num_forward_passes,
            }
        } else {
            CutComputationTiming::default()
        };
        
        Ok(Phase1Result { cut_data, timing })
    }

    fn select_cuts_batch(
        &mut self,
        mut cut_data: Vec<CutData>,
        stage_ctx: &BackwardStageContext,
    ) -> Result<Phase2Result, String> {
        let parent_id = stage_ctx.parent_id.ok_or_else(|| {
            format!("No parent ID for stage {} (stage_idx {})", 
                    stage_ctx.stage_id, stage_ctx.stage_idx)
        })?;
        
        // Sort for deterministic ordering
        cut_data.sort_unstable_by_key(|data| data.forward_pass_idx);
        
        let phase2_begin = Instant::now();
        
        // Access FCF and perform batch selection
        let batch_result = {
            let fcf_graph = self.fcf_graph();
            let parent_fcf_node = fcf_graph
                .get_node(parent_id)
                .ok_or_else(|| format!("Could not find FCF for parent node {}", parent_id))?;
            let mut fcf_locked = parent_fcf_node.data.lock().unwrap();
            fcf_locked.add_cuts_batch_from_data(cut_data, stage_ctx.enable_cut_selection)
        };
        
        let cut_selection_time = phase2_begin.elapsed();
        
        // Phase 3a: Update FCF state and clone cuts
        let fcf_update_begin = Instant::now();
        
        let aggregated = AggregatedCutSelectionResult {
            new_cut_ids: batch_result.new_cut_ids.clone(),
            returning_cut_ids: batch_result.returning_cut_ids.clone(),
            removing_cut_ids: batch_result.removing_cut_ids.clone(),
        };
        
        let (fcf_state_update_time, cut_cloning_time, cuts) = {
            let fcf_graph = self.fcf_graph();
            let parent_fcf_node = fcf_graph
                .get_node(parent_id)
                .ok_or_else(|| format!("Could not find FCF for parent node {}", parent_id))?;
            let mut fcf_locked = parent_fcf_node.data.lock().unwrap();
            
            // Update FCF state (mark inactive)
            let state_begin = Instant::now();
            let mut removed_indices: Vec<usize> = Vec::new();
            for &cut_id in &aggregated.removing_cut_ids {
                if let Some(cut) = fcf_locked.cut_pool.pool.get_mut(cut_id) {
                    cut.set_active(false);
                }
                if let Some(index) = fcf_locked.cut_pool.active_cut_indices.remove(&cut_id) {
                    removed_indices.push(index);
                }
            }
            
            // Sort removed indices for efficient adjustment
            removed_indices.sort_unstable();
            
            // Adjust indices for all remaining cuts
            for (_cut_id, index) in fcf_locked.cut_pool.active_cut_indices.iter_mut() {
                let count_below = removed_indices.partition_point(|&removed| removed < *index);
                *index -= count_below;
            }
            let state_time = state_begin.elapsed();
            
            // Clone cuts for parallel application
            let clone_begin = Instant::now();
            let cuts: Vec<(usize, Arc<BendersCut>)> = aggregated.new_cut_ids.iter()
                .chain(aggregated.returning_cut_ids.iter())
                .filter_map(|&cut_id| {
                    fcf_locked.cut_pool.pool.get(cut_id)
                        .map(|cut| (cut_id, Arc::clone(cut)))
                })
                .collect();
            let clone_time = clone_begin.elapsed();
            
            (state_time, clone_time, cuts)
        };
        
        let fcf_update_time = fcf_update_begin.elapsed();
        
        Ok(Phase2Result {
            batch_result,
            aggregated,
            cuts,
            cut_selection_time,
            fcf_update_time: fcf_state_update_time,
            cut_cloning_time,
        })
    }

    fn apply_cuts_parallel(
        &mut self,
        phase2_result: &Phase2Result,
        stage_ctx: &BackwardStageContext,
    ) -> Result<Duration, String> {
        let parent_id = stage_ctx.parent_id.ok_or_else(|| {
            "No parent ID for cut application".to_string()
        })?;
        
        let phase3b_begin = Instant::now();
        
        self.handlers
            .par_iter_mut()
            .map(|handler| {
                handler.apply_aggregated_cut_result(
                    parent_id,
                    &phase2_result.aggregated,
                    &phase2_result.cuts,
                )
            })
            .collect::<Result<(), String>>()?;
        
        Ok(phase3b_begin.elapsed())
    }

    fn eval_first_stage_bound(
        &mut self,
        stage_ctx: &BackwardStageContext,
    ) -> Result<(f64, FirstStageTiming), String> {
        // Use first handler to evaluate first stage
        let handler = self.handlers.get_mut(0)
            .ok_or_else(|| "No handlers available".to_string())?;
        
        let (lb, timing) = handler.eval_first_stage_bound(
            stage_ctx.stage_id,
            stage_ctx.past_node_ids,
            stage_ctx.node_data_graph,
            stage_ctx.saa,
        )?;
        
        Ok((lb, FirstStageTiming {
            solver: timing.solver_time,
            state_extraction: timing.state_extraction_time,
        }))
    }

    fn num_forward_passes(&self) -> usize {
        self.num_forward_passes
    }
}
```

---

## Acceptance Criteria

- [ ] `src/algorithm/coordinator.rs` created
- [ ] `ParallelHandlerCoordinator` struct implemented
- [ ] `BackwardStageProcessor` trait implemented for coordinator
- [ ] All phases match existing behavior in `sddp/mod.rs`
- [ ] Module exported in `src/algorithm/mod.rs`
- [ ] `cargo build -j1` succeeds
- [ ] `cargo clippy -j1 -- -D warnings` passes

---

## Implementation Guide

### Step 1: Create coordinator.rs

Create `src/algorithm/coordinator.rs` with the implementation above.

### Step 2: Update mod.rs

```rust
pub mod coordinator;
pub use coordinator::ParallelHandlerCoordinator;
```

### Step 3: Handle SddpTrainHandler visibility

If `SddpTrainHandler` is `pub(crate)` in `sddp/mod.rs`, you may need to:
- Add `pub use sddp::SddpTrainHandler;` if appropriate, OR
- Keep coordinator in sddp module temporarily

### Step 4: Verify compilation

```bash
cargo build -j1 2>&1 | head -50
```

Fix any import or visibility issues.

---

## Key Files to Create/Modify

| File | Action |
|------|--------|
| `src/algorithm/coordinator.rs` | CREATE |
| `src/algorithm/mod.rs` | UPDATE exports |
| `src/sddp/mod.rs` | May need visibility changes |

---

## Pitfalls to Avoid

- ⚠️ The FCF pointer pattern is unsafe but necessary—document carefully
- ⚠️ Keep timing calculation EXACTLY as in original code
- ⚠️ Maintain deterministic cut ordering (sort by forward_pass_idx)
- ⚠️ Don't change `SddpTrainHandler` internals—just wrap it

---

## Effort Estimate

**Points**: 5  
**Confidence**: Medium  
**Rationale**: Complex implementation but clear specification; main risk is visibility/import issues

---

## Definition of Done

- [ ] Coordinator implemented with all trait methods
- [ ] Timing calculations match original
- [ ] Parallel phases use `par_iter_mut()`
- [ ] Module exported
- [ ] `cargo build -j1` succeeds
- [ ] `cargo clippy -j1 -- -D warnings` passes
- [ ] Code reviewed
