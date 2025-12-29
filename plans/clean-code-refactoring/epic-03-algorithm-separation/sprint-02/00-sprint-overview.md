# Sprint 2: Handler Coordination Infrastructure

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Duration**: 1.5-2 weeks
> **Status**: ⬜ Not Started (Blocked by T-021 rework)

---

## ⚠️ PREREQUISITE: T-021 Rework

**This sprint cannot start until [T-021](../sprint-01/ticket-021-forward-timing-integration.md) is reworked.**

T-021 establishes the **timing separation pattern** that must be applied to all context structs in this sprint.

---

## ⚠️ ARCHITECTURAL DECISION: Timing Separation

Per lesson learned from T-021:

**Timing must NOT be inside context structs.**

This applies to:
- `BackwardPassContext` - remove timing field
- `BackwardStageContext` - no timing field
- All processor trait methods - timing passed separately

**Rationale**: When timing is inside a context struct, `TimingGuard` creates a borrow that prevents mutable access to other context fields, defeating the purpose of RAII timing.

---

## Goals

1. **Primary**: Create `ParallelHandlerCoordinator` to encapsulate handler management
2. **Primary**: Define `BackwardStageProcessor` trait for 3-phase processing
3. **Primary**: Migrate `SddpTrainHandler` management into coordinator
4. **Prerequisite**: Apply timing separation pattern from T-021
5. **Validation**: Bit-for-bit identical outputs, no performance regression

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-023](./ticket-023-backward-pass-context.md) | Revise BackwardPassContext (remove timing) | 2 | Yes | T-021 rework | ⬜ |
| [T-024](./ticket-024-design-processor-trait.md) | Design BackwardStageProcessor trait | 3 | Yes | T-023 | ⬜ |
| [T-025](./ticket-025-implement-coordinator.md) | Implement ParallelHandlerCoordinator | 5 | Yes | T-024 | ⬜ |
| [T-026](./ticket-026-migrate-handlers.md) | Migrate SddpTrainHandler into coordinator | 5 | Yes | T-025 | ⬜ |
| [T-027](./ticket-027-coordinator-tests.md) | Unit tests for coordinator | 3 | Yes | T-026 | ⬜ |

**Total Points**: 18

---

## Sequence

```
T-021 rework ──→ T-023 (Context) ──→ T-024 (Trait) ──→ T-025 (Coordinator) ──→ T-026 (Migrate) ──→ T-027 (Tests)
     ↑
  BLOCKER
```

**IMPORTANT**: T-021 rework must complete before this sprint starts.

---

## Architectural Overview

### Handler Coordination

```
┌─────────────────────────────────────────────────────────────────┐
│                        sddp/mod.rs                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ParallelHandlerCoordinator                  │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │    │
│  │  │ Handler 0    │ │ Handler 1    │ │ Handler N    │     │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘     │    │
│  │                                                          │    │
│  │  impl BackwardStageProcessor                            │    │
│  │    ├── compute_cuts_parallel()                          │    │
│  │    ├── select_cuts_batch()                              │    │
│  │    └── apply_cuts_parallel()                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           algorithm/backward_pass.rs (Sprint 3)          │    │
│  │  pub fn execute<P: BackwardStageProcessor>(              │    │
│  │      processor: &mut P,                                  │    │
│  │      ctx: &BackwardPassContext,                          │    │
│  │      timing: &BackwardPassTimingAccumulator,  // SEPARATE│    │
│  │  )                                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Timing Separation Pattern

```rust
// CORRECT: Timing separate from context
pub fn execute<P: BackwardStageProcessor>(
    processor: &mut P,
    ctx: &BackwardPassContext,           // No timing inside
    timing: &BackwardPassTimingAccumulator,  // Separate parameter
) -> Result<BackwardPassResult, String> {
    for stage_idx in ctx.backward_stage_indices() {
        let stage_ctx = ctx.stage_context(stage_idx)?;
        
        {
            let _guard = TimingGuard::new(&timing.model_preprocessing);
            // Can still access processor and ctx mutably!
            processor.compute_cuts_parallel(&stage_ctx)?;
        }
    }
}
```

---

## Dependencies

- **From Sprint 1**:
  - `ForwardPassContext` pattern established ✅
  - Forward pass module complete ✅
  - **Timing separation pattern (T-021 rework)** ← BLOCKER
  
- **To Sprint 3**:
  - `ParallelHandlerCoordinator` ready for backward pass extraction
  - `BackwardStageProcessor` trait defined
  - Timing separation applied to all contexts

---

## Key Files

| File | Lines | Role |
|------|-------|------|
| `src/sddp/mod.rs` | 3,827 | Source: `SddpTrainHandler`, parallel coordination |
| `src/algorithm/coordinator.rs` | - | New: `ParallelHandlerCoordinator` |
| `src/algorithm/processor.rs` | - | New: `BackwardStageProcessor` trait |
| `src/algorithm/context.rs` | 523 | Update: contexts without timing |

---

## Verification Protocol

After EVERY ticket:

```bash
cargo build -j1 && RUST_TEST_THREADS=1 cargo test -j1 && ./scripts/golden-tests.sh verify
```

After sprint complete:
```bash
cargo bench -- --baseline before-refactoring
```

---

## Risks

| Risk | Mitigation |
|------|------------|
| T-021 rework delayed | Sprint cannot start until T-021 complete |
| Trait overhead | Use static dispatch via generics |
| Complex migration | Keep `SddpTrainHandler` unchanged, wrap in coordinator |
| Parallel behavior change | Verify thread count and scheduling unchanged |

---

## Definition of Done

- [ ] **T-021 rework complete** (prerequisite)
- [ ] `BackwardPassContext` has no timing field
- [ ] `BackwardStageContext` has no timing field
- [ ] `ParallelHandlerCoordinator` implemented
- [ ] `BackwardStageProcessor` trait defined
- [ ] `SddpTrainHandler` management moved to coordinator
- [ ] Coordinator unit tests passing
- [ ] Golden tests pass
- [ ] Benchmarks within 5%
- [ ] Code reviewed
