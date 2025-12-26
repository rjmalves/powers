# Preallocation Refactoring Implementation Plan

Complete memory preallocation for HPC scalability in POWE.RS.

## Quick Navigation

### [Master Plan](./00-master-plan.md)
Architecture overview, goals, and success metrics.

---

## Epics

### [Epic 1: HiGHS Constraint Preallocation](./epic-01-highs-constraint-preallocation/00-epic-overview.md)
**Priority**: 1 (Highest Impact)  
**Duration**: 2 weeks  
**Status**: ✅ Complete (2025-12-26)

Pre-allocate cut constraint slots in HiGHS solver to eliminate dynamic allocations during training.

- [Sprint 1: Core Infrastructure](./epic-01-highs-constraint-preallocation/sprint-01/00-sprint-overview.md) ✅
- [Sprint 2: Integration & Validation](./epic-01-highs-constraint-preallocation/sprint-02/00-sprint-overview.md) ✅

---

### [Epic 1b: Full Memory Determinism](./epic-01b-full-memory-determinism/00-epic-overview.md)
**Priority**: 1.5 (Required before continuing)  
**Duration**: 1 week  
**Status**: ✅ Complete (2025-12-26)

Remove dynamic allocation fallbacks and slot tracking complexity. Replace with deterministic slot calculation based on `(iteration, forward_pass_idx)`.

- [Sprint 1: Deterministic Cut Slot Management](./epic-01b-full-memory-determinism/sprint-01/00-sprint-overview.md) ✅

---

### [Epic 1c: Memory Module Cleanup](./epic-01c-memory-module-cleanup/00-epic-overview.md)
**Priority**: 1.6 (Required before Epic 2)  
**Duration**: 3-5 days  
**Status**: ✅ Complete (2025-12-26)

Clean up the `src/memory` module by removing ~3,100 lines of unused/broken code and hardening `CutComputationBuffers` to enforce zero-allocation guarantees.

- [Sprint 1: Cleanup and Hardening](./epic-01c-memory-module-cleanup/sprint-01/00-sprint-overview.md) ✅

---

### [Epic 2: FCF Full Preallocation](./epic-02-fcf-full-preallocation/00-epic-overview.md) ⭐ NEXT
**Priority**: 2  
**Duration**: 1 week  
**Status**: ⬜ Not Started

Ensure `FutureCostFunction::with_capacity()` is used everywhere, using runtime parameters (since `SizingInfo` was removed in Epic 1c).

- [Sprint 1: Complete FCF Preallocation](./epic-02-fcf-full-preallocation/sprint-01/00-sprint-overview.md)

---

### [Epic 3: Handler-Level SoA Blocks](./epic-03-handler-soa-blocks/00-epic-overview.md)
**Priority**: 3  
**Duration**: 2-3 weeks  
**Status**: ⬜ Not Started

Convert handler hot data to contiguous SoA blocks for cache efficiency.

- [Sprint 1: RealizationBlock Implementation](./epic-03-handler-soa-blocks/sprint-01/00-sprint-overview.md)
- [Sprint 2: Integration & Optimization](./epic-03-handler-soa-blocks/sprint-02/00-sprint-overview.md)

---

## Progress Tracking

### Epic 1: HiGHS Constraint Preallocation ✅
- [x] TICKET-001: Add HiGHS batch row API bindings ✅
- [x] TICKET-002: Extend SizingInfo with cut estimation ✅
- [x] TICKET-003: Implement Subproblem cut slot infrastructure ✅
- [x] TICKET-004: Update cut addition to use coefficient changes ✅
- [x] TICKET-005: Implement cut removal via bound relaxation ✅
- [x] TICKET-006: Add cut slot reuse for selection ✅
- [x] TICKET-007: Integration and validation ✅

**Epic 1 Complete!** 🎉 Measured ~23% performance improvement on example 07.

### Epic 1b: Full Memory Determinism ✅
- [x] TICKET-001b: Add slot_index to BendersCut and num_forward_passes to Subproblem ✅
- [x] TICKET-002b: Implement deterministic slot calculation ✅
- [x] TICKET-003b: Update cut addition flow to use deterministic slots ✅
- [x] TICKET-004b: Update cut deactivation to use stored slot ✅
- [x] TICKET-005b: Remove slot tracking data structures ✅
- [x] TICKET-006b: Update SDDP call sites ✅
- [x] TICKET-007b: Validation and testing ✅

**Epic 1b Complete!** 🎉 Full memory determinism achieved. Slot tracking overhead removed.

### Epic 1c: Memory Module Cleanup ✅
- [x] TICKET-001c: Remove SizingInfo and related dead code ✅
- [x] TICKET-002c: Remove DeepSizeEstimate trait and implementations ✅
- [x] TICKET-003c: Harden CutComputationBuffers with capacity enforcement ✅
- [x] TICKET-004c: Fix cut buffer initialization with correct dimensions ✅
- [x] TICKET-005c: Validation and testing ✅

**Epic 1c Complete!** 🎉 87% code reduction (3,424 → 437 lines). Capacity enforcement in place.

### Epic 2: FCF Full Preallocation ⬜ **NEXT**
- [ ] TICKET-008: Audit FCF instantiation sites
- [ ] TICKET-009: Update FCF creation to use with_capacity()
- [ ] TICKET-010: Validate memory profile

### Epic 3: Handler-Level SoA Blocks ⬜
- [ ] TICKET-011: Design RealizationBlock structure
- [ ] TICKET-012: Implement RealizationBlock
- [ ] TICKET-013: Integrate with SddpTrainHandler
- [ ] TICKET-014: Design SubproblemBlock structure
- [ ] TICKET-015: Implement SubproblemBlock
- [ ] TICKET-016: Performance validation

---

## Key Insight: Deterministic Cut Slot Formula

Since we know `num_iterations` and `num_forward_passes` at training start, each cut's slot is deterministic:

```
slot_index = (iteration - 1) * num_forward_passes + forward_pass_idx
```

| Iteration | Forward Pass | Slot (4 FPs) |
|-----------|--------------|--------------|
| 1 | 0 | 0 |
| 1 | 3 | 3 |
| 2 | 0 | 4 |
| 32 | 3 | 127 |

**Benefits**:
- Zero tracking overhead (no `Vec<Option<usize>>`, no free list)
- O(1) slot calculation (simple arithmetic)
- Predictable memory layout
- Simplified debugging

---

## Validation Commands

```bash
# Quick smoke test (examples 01 and 07)
cargo run --release -- run examples/01-deterministic
cargo run --release -- run examples/07-par-model-with-inflow-state

# Performance comparison
hyperfine --warmup 2 --runs 5 \
  'cargo run --release -- run examples/07-par-model-with-inflow-state'

# Memory profiling
valgrind --tool=massif --massif-out-file=massif.out \
  ./target/release/powers run examples/07-par-model-with-inflow-state
ms_print massif.out | head -100

# Verify memory module is minimal
wc -l src/memory/*.rs
# Should be ~437 lines

# Verify no dead code references
grep -rn "SizingInfo\|DeepSizeEstimate\|ThreadLocalBuffers" src/
# Should return nothing
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/memory/mod.rs` | Module re-exports |
| `src/memory/buffers.rs` | CutComputationBuffers (capacity-enforced) |
| `src/solver.rs` | HiGHS API bindings |
| `src/subproblem.rs` | Subproblem with cut handling |
| `src/fcf.rs` | FutureCostFunction with pools |
| `src/cut.rs` | BendersCut and BendersCutPool |
| `src/sddp/mod.rs` | SDDP algorithm and handlers |
