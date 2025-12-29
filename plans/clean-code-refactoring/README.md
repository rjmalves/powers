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
| 3 | [Algorithm Separation](./epic-03-algorithm-separation/00-epic-overview.md) | 4-5 weeks | 🔄 **Sprint 2 ✅, Sprint 3 Next** |
| 4 | [State Simplification](./epic-04-state-simplification/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 5 | [Memory Optimization](./epic-05-memory-optimization/00-epic-overview.md) | 3 weeks | ⬜ Not Started |
| 6 | [Test Modernization](./epic-06-test-modernization/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 7 | [Performance Validation](./epic-07-performance-validation/00-epic-overview.md) | 1 week | ⬜ Not Started |

**Total Duration**: ~17-18 weeks

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

### Sprint 2 Revised: Handler Coordination Infrastructure ✅ Complete

> **Note**: Sprint 2 was revised after T-025 implementation challenges. See [T-025 Implementation Challenges](../docs/T-025-implementation-challenges.md).

| ID | Title | Status |
|----|-------|--------|
| [T-023](./epic-03-algorithm-separation/sprint-02/ticket-023-backward-pass-context.md) | Revise BackwardPassContext | ✅ |
| [T-024](./epic-03-algorithm-separation/sprint-02/ticket-024-design-processor-trait.md) | Design BackwardStageProcessor trait | ✅ |
| [T-023A](./epic-03-algorithm-separation/sprint-02-revised/ticket-023a-handler-visibility.md) | Make handler methods public | ✅ |
| [T-024A](./epic-03-algorithm-separation/sprint-02-revised/ticket-024a-revise-processor-trait.md) | Revise processor trait signatures | ✅ |
| [T-025A](./epic-03-algorithm-separation/sprint-02-revised/ticket-025a-implement-coordinator.md) | Implement coordinator (no unsafe) | ✅ |
| [T-026](./epic-03-algorithm-separation/sprint-02-revised/ticket-026-migrate-handlers.md) | Migrate handlers into coordinator | ✅ |
| [T-027](./epic-03-algorithm-separation/sprint-02-revised/ticket-027-coordinator-tests.md) | Coordinator unit tests | ✅ |

### Sprint 3: Backward Pass Extraction ⬜ Ready to Start
| ID | Title | Status |
|----|-------|--------|
| [T-028](./epic-03-algorithm-separation/sprint-03/ticket-028-extract-backward-loop.md) | Extract backward pass loop | ⬜ **NEXT** |
| [T-029](./epic-03-algorithm-separation/sprint-03/ticket-029-extract-cut-computation.md) | Extract cut computation | ⬜ |
| [T-030](./epic-03-algorithm-separation/sprint-03/ticket-030-backward-timing.md) | Verify backward timing | ⬜ |
| [T-031](./epic-03-algorithm-separation/sprint-03/ticket-031-update-sddp-backward.md) | Update sddp/mod.rs backward | ⬜ |
| [T-032](./epic-03-algorithm-separation/sprint-03/ticket-032-verify-timing.md) | Verify timing end-to-end | ⬜ |

---

## Sprint 2 Revision Summary

### Problem Discovered

T-025 implementation revealed architectural challenges:
1. FCF internal structure access required unsafe raw pointers
2. Handler methods were `pub(crate)`, inaccessible from `algorithm` module
3. Timing types mismatched between modules

### Solution (2025-12-29)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Coordinator location | `src/algorithm/coordinator.rs` | Clean separation |
| FCF access | Pass to methods | Avoids unsafe code |
| Handler visibility | Make methods `pub` | Enable cross-module access |
| Timing types | Unify to `CutComputationTiming` | Clean API |

### New Tickets

- **T-023A**: Make handler methods and timing types public
- **T-024A**: Revise `BackwardStageProcessor` trait (FCF parameter)
- **T-025A**: Implement coordinator without unsafe code

---

## Key Architectural Decisions

### Timing Separation (2025-12-29)

**Timing must NOT be inside context structs** to avoid borrow conflicts with `TimingGuard`.

```rust
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

### Handler Coordination (2025-12-29 Revised)

**`ParallelHandlerCoordinator`** encapsulates `Vec<SddpTrainHandler>` and implements `BackwardStageProcessor`. FCF graph is passed to `select_cuts_batch()` to avoid unsafe code.

```rust
pub struct ParallelHandlerCoordinator {
    handlers: Vec<SddpTrainHandler>,
    // NO fcf_graph field - passed to methods
}

impl BackwardStageProcessor for ParallelHandlerCoordinator {
    fn select_cuts_batch(
        &mut self,
        cut_data: Vec<CutData>,
        stage_ctx: &BackwardStageContext,
        fcf_graph: &DirectedGraph<Mutex<FutureCostFunction>>,  // ✅ Safe
    ) -> Result<Phase2Result, String>;
}
```

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
