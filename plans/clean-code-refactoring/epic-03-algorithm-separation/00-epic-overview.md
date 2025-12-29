# Epic 3: Algorithm Separation

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 4-5 weeks (3 sprints)
> **Status**: ✅ Complete (Sprint 3 finished 2025-12-29)

---

## ⚠️ CRITICAL REMINDER

This epic touches the **core SDDP algorithm**. Read the [Critical Principles](../00-master-plan.md#️-critical-principles-correctness-first-then-performance) section carefully.

**The algorithm logic must remain EXACTLY the same.** We are separating structure, not modifying behavior. Golden tests must pass after every change.

---

## ⚠️ ARCHITECTURAL DECISION: Timing Separation (2025-12-29)

**Timing must NOT be inside context structs.**

### Problem Discovered

When timing is embedded inside context structs, `TimingGuard` creates a borrow that prevents mutable access to other context fields:

```rust
// PROBLEMATIC: timing inside context
pub struct ForwardPassContext<'a> {
    pub subproblem_graph: &'a mut DirectedGraph<Subproblem>,
    pub timing: &'a ForwardTiming,  // ❌ Borrowing this...
}

fn execute_stage(ctx: &mut ForwardPassContext) {
    let _guard = TimingGuard::new(&ctx.timing.field); // ...prevents this
    let node = ctx.subproblem_graph.get_node_mut(id)?; // ❌ BORROW CONFLICT
}
```

### Solution

**Pass timing as a separate parameter:**

```rust
// CORRECT: timing separate from context
pub struct ForwardPassContext<'a> {
    pub subproblem_graph: &'a mut DirectedGraph<Subproblem>,
    // NO timing field
}

pub fn execute(
    ctx: &mut ForwardPassContext,
    timing: &TrajectoryTiming,  // ✅ Separate parameter with Cell<Duration>
) -> Result<...> {
    {
        let _guard = TimingGuard::new(&timing.model_preprocessing);
        let node = ctx.subproblem_graph.get_node_mut(id)?; // ✅ No conflict!
    }
}
```

### Application

This pattern applies to ALL context structs:
- `ForwardPassContext` - no timing field
- `BackwardPassContext` - no timing field  
- `BackwardStageContext` - no timing field

Timing accumulators use `Cell<Duration>` for interior mutability, enabling `TimingGuard` to work.

---

## Summary

This epic separates the forward and backward passes from the monolithic `src/sddp/mod.rs` (3,827 lines) into dedicated modules in `src/algorithm/`. 

**Key architectural decisions**:
1. **Timing separation**: Timing passed as separate parameter, not inside context
2. **Handler coordination**: `ParallelHandlerCoordinator` encapsulates `Vec<SddpTrainHandler>`
3. **Trait-based processor**: `BackwardStageProcessor` trait abstracts 3-phase backward pass

---

## Scope

### Included

1. **Forward Pass Extraction** (Sprint 1)
   - Extract `forward()` to `src/algorithm/forward_pass.rs`
   - Create `ForwardPassContext` struct (no timing inside)
   - Integrate `TimingGuard` with timing as separate parameter

2. **Handler Coordination Infrastructure** (Sprint 2)
   - Create `ParallelHandlerCoordinator` to encapsulate handler management
   - Define `BackwardStageProcessor` trait for phase-based processing
   - Apply timing separation to backward contexts

3. **Backward Pass Extraction** (Sprint 3)
   - Extract backward pass loop to `src/algorithm/backward_pass.rs`
   - Extract cut computation to `src/algorithm/cut_computation.rs`
   - Integrate `TimingGuard` with timing as separate parameter

### Excluded

- Algorithm modifications
- State representation changes (Epic 4)
- Memory pool implementation (Epic 5)

---

## Sprints

### [Sprint 1: Forward Pass Extraction](./sprint-01/00-sprint-overview.md) ⚠️ T-021 Rework Needed

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-018 | Design context structs with preallocation awareness | 3 | ✅ |
| T-019 | Create ForwardPassContext | 3 | ✅ |
| T-020 | Extract forward pass step logic | 5 | ✅ |
| T-021 | Integrate forward timing infrastructure | 3 | ❌ **REWORK** |
| T-022 | Update sddp/mod.rs to use forward_pass module | 3 | ✅ |

**Sprint 1 Points**: 17  
**Status**: T-021 requires rework to implement timing separation

### [Sprint 2 Revised: Handler Coordination Infrastructure](./sprint-02-revised/00-sprint-overview.md) ✅ Complete

> **Note**: This sprint supersedes the original Sprint 2 after architectural challenges were discovered. See [T-025 Implementation Challenges](../../../docs/T-025-implementation-challenges.md).

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-023 | Revise BackwardPassContext (remove timing) | 2 | ✅ |
| T-024 | Design BackwardStageProcessor trait | 3 | ✅ |
| T-023A | Make handler methods and timing types public | 2 | ✅ |
| T-024A | Revise BackwardStageProcessor trait signatures | 3 | ✅ |
| T-025A | Implement ParallelHandlerCoordinator (no unsafe) | 5 | ✅ |
| T-026 | Migrate SddpTrainHandler into coordinator | 5 | ✅ |
| T-027 | Unit tests for coordinator | 3 | ✅ |

**Sprint 2 Points**: 23 (increased due to visibility changes and trait revision)  
**Key Changes**: FCF passed to methods (no unsafe), timing types unified


### [Sprint 3: Backward Pass Extraction](./sprint-03/00-sprint-overview.md) ✅ Complete

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-028 | Extract backward pass loop | 5 | ✅ |
| T-029 | Extract cut computation logic | 3 | ✅ |
| T-030 | Verify backward timing integration | 3 | ✅ |
| T-031 | Update sddp/mod.rs to use backward_pass module | 3 | ✅ |
| T-032 | Verify timing integration end-to-end | 2 | ✅ |

**Sprint 3 Points**: 16
**Key Outcome**: `sddp/mod.rs` reduced from 3881 to 3616 lines (-265 lines)

---

## Dependency Graph

```
Sprint 1 (Forward) ──→ Sprint 2 Revised (Coordinator) ──→ Sprint 3 (Backward)
                              │
                              ▼
              T-023 ✅ ─→ T-024 ✅ ─→ T-023A ─→ T-024A ─→ T-025A ─→ T-026 ─→ T-027
```

---

## Module Structure After Epic

```
src/algorithm/
├── mod.rs
├── context.rs           # ForwardPassContext, BackwardPassContext (NO timing)
├── forward_pass.rs      # Forward pass execution with TimingGuard
├── backward_pass.rs     # Backward pass execution with TimingGuard
├── cut_computation.rs   # Cut calculation logic
├── coordinator.rs       # ParallelHandlerCoordinator
└── processor.rs         # BackwardStageProcessor trait
```

---

## Acceptance Criteria

- [x] `src/algorithm/forward_pass.rs` contains forward pass logic
- [x] `src/algorithm/backward_pass.rs` contains backward pass logic
- [x] `src/algorithm/coordinator.rs` contains `ParallelHandlerCoordinator`
- [x] **No context struct contains timing** (timing passed separately)
- [x] **Hybrid timing pattern** (coordinator uses Instant::now(), module accumulates)
- [x] **Feature-gating works** (timing compiles to no-op when disabled)
- [x] All public functions have ≤4 parameters
- [x] Golden tests pass (bit-for-bit identical output)
- [ ] Benchmarks show no regression (within 5%) - pending formal benchmark run

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Subtle behavior change | Medium | **CRITICAL** | Golden tests after every change ✅ |
| Borrow checker issues | High | Medium | Resolved via timing separation ✅ |
| Parallel execution changes | Medium | High | Verified via golden tests ✅ |
| Complex refactoring | Medium | Medium | Incremental changes, verified each step ✅ |

---

## Definition of Done

- [x] All Sprint 1, 2, and 3 tickets complete
- [x] Forward pass in `src/algorithm/forward_pass.rs`
- [x] Backward pass in `src/algorithm/backward_pass.rs`
- [x] Handler coordination in `src/algorithm/coordinator.rs`
- [x] **Timing separated from all contexts**
- [x] **Hybrid timing pattern implemented**
- [x] **Feature-gating verified**
- [x] `sddp/mod.rs` reduced by ~265 lines (3881 → 3616)
- [x] Golden tests pass (all 7 examples)
- [ ] Benchmarks within 5% of baseline (pending)
- [x] All tests pass (542 tests)
- [ ] Code reviewed and merged
