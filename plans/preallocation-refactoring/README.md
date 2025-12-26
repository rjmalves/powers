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
**Status**: ⬜ Not Started

Pre-allocate cut constraint slots in HiGHS solver to eliminate dynamic allocations during training.

- [Sprint 1: Core Infrastructure](./epic-01-highs-constraint-preallocation/sprint-01/00-sprint-overview.md)
- [Sprint 2: Integration & Validation](./epic-01-highs-constraint-preallocation/sprint-02/00-sprint-overview.md)

---

### [Epic 2: FCF Full Preallocation](./epic-02-fcf-full-preallocation/00-epic-overview.md)
**Priority**: 2  
**Duration**: 1 week  
**Status**: ⬜ Not Started

Ensure `FutureCostFunction::with_capacity()` is used everywhere.

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

### Epic 1: HiGHS Constraint Preallocation
- [ ] TICKET-001: Add HiGHS batch row API bindings
- [ ] TICKET-002: Extend SizingInfo with cut estimation
- [ ] TICKET-003: Implement Subproblem cut slot infrastructure
- [ ] TICKET-004: Update cut addition to use coefficient changes
- [ ] TICKET-005: Implement cut removal via bound relaxation
- [ ] TICKET-006: Add cut slot reuse for selection
- [ ] TICKET-007: Integration and validation

### Epic 2: FCF Full Preallocation
- [ ] TICKET-008: Audit FCF instantiation sites
- [ ] TICKET-009: Ensure with_capacity usage everywhere
- [ ] TICKET-010: Validate memory profile

### Epic 3: Handler-Level SoA Blocks
- [ ] TICKET-011: Design RealizationBlock structure
- [ ] TICKET-012: Implement RealizationBlock
- [ ] TICKET-013: Integrate with SddpTrainHandler
- [ ] TICKET-014: Design SubproblemBlock structure
- [ ] TICKET-015: Implement SubproblemBlock
- [ ] TICKET-016: Performance validation

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
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/memory/sizing.rs` | SizingInfo computation |
| `src/memory/buffers.rs` | Buffer and pool abstractions |
| `src/solver.rs` | HiGHS API bindings |
| `src/subproblem.rs` | Subproblem with cut handling |
| `src/fcf.rs` | FutureCostFunction with pools |
| `src/cut.rs` | BendersCut and BendersCutPool |
| `src/sddp/mod.rs` | SDDP algorithm and handlers |
