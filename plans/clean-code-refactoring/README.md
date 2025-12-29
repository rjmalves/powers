# Clean Code Refactoring for HPC Performance

Refactoring the POWE.RS codebase into clean, modular Rust code to enable zero-allocation hot paths and improved performance while maintaining **bit-for-bit algorithmic correctness**.

> ⚠️ **CRITICAL**: Read the [Critical Principles](./00-master-plan.md#️-critical-principles-correctness-first-then-performance) section before starting ANY work.

---

## Quick Navigation

### Master Plan
- [00-master-plan.md](./00-master-plan.md) - Architecture overview, phases, and design decisions

### Epics

| Epic | Name | Duration | Status |
|------|------|----------|--------|
| 1 | [Foundation](./epic-01-foundation/00-epic-overview.md) | 2 weeks | ✅ Complete |
| 2 | [Core Extraction](./epic-02-core-extraction/00-epic-overview.md) | 3 weeks | ✅ Complete |
| 3 | [Algorithm Separation](./epic-03-algorithm-separation/00-epic-overview.md) | 4-5 weeks | 🔄 **T-021 Rework Needed** |
| 4 | [State Simplification](./epic-04-state-simplification/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 5 | [Memory Optimization](./epic-05-memory-optimization/00-epic-overview.md) | 3 weeks | ⬜ Not Started |
| 6 | [Test Modernization](./epic-06-test-modernization/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 7 | [Performance Validation](./epic-07-performance-validation/00-epic-overview.md) | 1 week | ⬜ Not Started |

**Total Duration**: ~17-18 weeks

---

## ⚠️ BLOCKING ISSUE: T-021 Rework Required

### Problem

T-021 (Forward Timing Integration) was implemented with a compromise that used `Instant::now()` instead of `TimingGuard` due to borrow checker conflicts.

**This was an unacceptable compromise that should have been escalated.**

### Root Cause

Timing was embedded inside `ForwardPassContext`, creating borrow conflicts when using `TimingGuard`.

### Solution (Architectural Decision 2025-12-29)

**Timing must NOT be inside context structs.** Pass timing as a separate parameter.

```rust
// BEFORE (problematic):
pub struct ForwardPassContext<'a> {
    pub timing: &'a ForwardTiming,  // ❌ Causes borrow conflicts
}

// AFTER (correct):
pub fn execute(
    ctx: &mut ForwardPassContext,  // No timing inside
    timing: &TrajectoryTiming,      // ✅ Separate parameter
) -> Result<...>
```

### Next Action

**Complete T-021 rework before proceeding with Sprint 2.**

See [T-021](./epic-03-algorithm-separation/sprint-01/ticket-021-forward-timing-integration.md) for full specification.

---

## Current Focus: Epic 3 - Algorithm Separation

### Sprint 1: Forward Pass Extraction ⚠️ T-021 Rework Needed
| ID | Title | Status |
|----|-------|--------|
| T-018 | Design context structs | ✅ |
| T-019 | Create ForwardPassContext | ✅ |
| T-020 | Extract forward pass step logic | ✅ |
| **T-021** | **Integrate forward timing** | **❌ REWORK** |
| T-022 | Update sddp/mod.rs forward | ✅ |

### Sprint 2: Handler Coordination Infrastructure (Blocked by T-021)
| ID | Title | Status |
|----|-------|--------|
| [T-023](./epic-03-algorithm-separation/sprint-02/ticket-023-backward-pass-context.md) | Revise BackwardPassContext | ⬜ |
| [T-024](./epic-03-algorithm-separation/sprint-02/ticket-024-design-processor-trait.md) | Design BackwardStageProcessor trait | ⬜ |
| [T-025](./epic-03-algorithm-separation/sprint-02/ticket-025-implement-coordinator.md) | Implement ParallelHandlerCoordinator | ⬜ |
| [T-026](./epic-03-algorithm-separation/sprint-02/ticket-026-migrate-handlers.md) | Migrate handlers into coordinator | ⬜ |
| [T-027](./epic-03-algorithm-separation/sprint-02/ticket-027-coordinator-tests.md) | Coordinator unit tests | ⬜ |

### Sprint 3: Backward Pass Extraction
| ID | Title | Status |
|----|-------|--------|
| [T-028](./epic-03-algorithm-separation/sprint-03/ticket-028-extract-backward-loop.md) | Extract backward pass loop | ⬜ |
| [T-029](./epic-03-algorithm-separation/sprint-03/ticket-029-extract-cut-computation.md) | Extract cut computation | ⬜ |
| [T-030](./epic-03-algorithm-separation/sprint-03/ticket-030-backward-timing.md) | Verify backward timing | ⬜ |
| [T-031](./epic-03-algorithm-separation/sprint-03/ticket-031-update-sddp-backward.md) | Update sddp/mod.rs backward | ⬜ |
| [T-032](./epic-03-algorithm-separation/sprint-03/ticket-032-verify-timing.md) | Verify timing end-to-end | ⬜ |

---

## Key Architectural Decisions

### Timing Separation (2025-12-29)

**Timing must NOT be inside context structs** to avoid borrow conflicts with `TimingGuard`.

```rust
// Timing accumulators use Cell for interior mutability
pub struct TrajectoryTiming {
    pub model_preprocessing: Cell<Duration>,
    pub solver: Cell<Duration>,
    // ...
}

// Pass timing as separate parameter
pub fn execute(
    ctx: &mut ForwardPassContext,
    timing: &TrajectoryTiming,  // Separate from context
) {
    {
        let _guard = TimingGuard::new(&timing.model_preprocessing);
        // Can access ctx mutably without conflict!
    }
}
```

### Handler Coordination

**`ParallelHandlerCoordinator`** encapsulates `Vec<SddpTrainHandler>` and implements `BackwardStageProcessor` trait for clean backward pass extraction.

---

## Before You Start Any Ticket

1. **Read the Critical Principles** in the [master plan](./00-master-plan.md)
2. **Run golden output tests**: `./scripts/golden-tests.sh verify`
3. **Read the epic and sprint overviews**
4. **Use single-threaded builds/tests**: `cargo build -j1`, `RUST_TEST_THREADS=1 cargo test -j1`

## After Completing Any Ticket

1. **Build**: `cargo build -j1`
2. **Test**: `RUST_TEST_THREADS=1 cargo test -j1`
3. **Feature test**: `cargo build -j1 --features timing`
4. **Golden tests**: `./scripts/golden-tests.sh verify` ✅ CRITICAL
5. **If ANY failure**: STOP and investigate

---

## Status Legend

- ⬜ Not Started
- 🔄 In Progress
- ⚠️ Needs Attention
- ❌ Requires Rework
- ✅ Complete
- 🔴 Blocked
