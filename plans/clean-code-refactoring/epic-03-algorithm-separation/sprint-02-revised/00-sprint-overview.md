# Sprint 2 (Revised): Handler Coordination Infrastructure

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: 🔄 Ready to Start
> **Supersedes**: [Original Sprint 2](../sprint-02/00-sprint-overview.md)

---

## Revision Summary

This sprint was revised after discovering architectural challenges documented in
[T-025 Implementation Challenges](../../../../docs/T-025-implementation-challenges.md).

### Key Changes from Original

1. **New T-023A**: Visibility changes for handler methods (prerequisite)
2. **Revised T-024A**: Simplified `BackwardStageProcessor` trait (FCF passed per-method)
3. **Revised T-025A**: Coordinator without unsafe code (FCF reference passed to methods)
4. **T-026/T-027**: Unchanged (migration and tests)

### Decisions Made

| Question | Decision | Rationale |
|----------|----------|-----------|
| Coordinator location | `src/algorithm/coordinator.rs` | Clean separation |
| Unsafe code | Avoid | Pass FCF to methods instead |
| Trait implementation | Full | Future flexibility |
| Timing types | Unify to `CutComputationTiming` | Clean API |

---

## Goals

- Create `ParallelHandlerCoordinator` in `src/algorithm/coordinator.rs`
- Implement `BackwardStageProcessor` trait for coordinator
- No unsafe code - FCF passed to methods that need it
- Unify timing types (`BackwardPhase1Timing` → `CutComputationTiming`)

---

## Tickets

| ID | Title | Points | Status | Dependencies |
|----|-------|--------|--------|--------------|
| T-023 | Revise BackwardPassContext (remove timing) | 2 | ✅ Done | None |
| T-024 | Design BackwardStageProcessor trait | 3 | ✅ Done | T-023 |
| T-023A | Make handler methods and timing types public | 2 | ✅ Done | T-024 |
| T-024A | Revise BackwardStageProcessor trait signatures | 3 | ✅ Done | T-023A |
| T-025A | Implement ParallelHandlerCoordinator (no unsafe) | 5 | ✅ Done | T-024A |
| T-026 | Migrate SddpTrainHandler into coordinator | 5 | ✅ Done | T-025A |
| T-027 | Unit tests for coordinator | 3 | ✅ Done | T-026 |

**Sprint 2 Points**: 23 (increased due to visibility changes and trait revision)

---

## Architecture Overview

### Before (Current)

```
train() in sddp/mod.rs
    │
    ├── train_handlers: Vec<SddpTrainHandler>  (direct management)
    │
    ├── Phase 1: train_handlers.par_iter_mut().map(compute_cut_data...)
    ├── Phase 2: fcf_locked.add_cuts_batch_from_data(...)  
    ├── Phase 3a: fcf_locked.cut_pool manipulation
    └── Phase 3b: train_handlers.par_iter_mut().map(apply_aggregated...)
```

### After (Target)

```
train() in sddp/mod.rs
    │
    ├── coordinator: ParallelHandlerCoordinator  (encapsulated)
    │
    └── backward_pass (extracted later):
        │
        ├── coordinator.compute_cuts_parallel(stage_ctx)
        ├── coordinator.select_cuts_batch(cut_data, stage_ctx, fcf_graph)  ← FCF passed here
        ├── coordinator.apply_cuts_parallel(phase2_result, stage_ctx)
        └── coordinator.eval_first_stage_bound(stage_ctx)
```

### Key Design: FCF Access Without Unsafe

Instead of storing a raw pointer to FCF graph in the coordinator:

```rust
// BEFORE (original ticket - unsafe)
pub struct ParallelHandlerCoordinator {
    fcf_graph: *const DirectedGraph<Mutex<FutureCostFunction>>,  // ❌ unsafe
}

// AFTER (revised - safe)
pub struct ParallelHandlerCoordinator {
    handlers: Vec<SddpTrainHandler>,
    num_forward_passes: usize,
    // NO fcf_graph field
}

impl BackwardStageProcessor for ParallelHandlerCoordinator {
    fn select_cuts_batch(
        &mut self,
        cut_data: Vec<CutData>,
        stage_ctx: &BackwardStageContext,
        fcf_graph: &DirectedGraph<Mutex<FutureCostFunction>>,  // ✅ passed per-call
    ) -> Result<Phase2Result, String>;
}
```

---

## Dependencies

- **From Sprint 1**: T-023 (BackwardPassContext) ✅ Complete
- **To Sprint 3**: Enables backward pass extraction

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Visibility changes break encapsulation | Low | Low | Only expose what's needed |
| Trait signature changes affect usability | Medium | Medium | Keep signatures minimal |
| Timing unification introduces bugs | Low | High | Golden tests validate |

---

## Definition of Done

- [ ] All tickets complete
- [ ] `SddpTrainHandler::compute_cut_data_for_backward_step` is `pub`
- [ ] `SddpTrainHandler::eval_first_stage_bound` is `pub`
- [ ] `BackwardPhase1Timing` renamed/replaced with `CutComputationTiming`
- [ ] `ParallelHandlerCoordinator` in `src/algorithm/coordinator.rs`
- [ ] Full `BackwardStageProcessor` trait implementation
- [ ] No unsafe code
- [ ] All tests passing
- [ ] Golden tests pass
- [ ] Code reviewed and merged
