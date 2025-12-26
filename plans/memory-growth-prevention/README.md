# Memory Growth Prevention Plan

**Created**: 2025-12-26  
**Based on**: `MEMORY_GROWTH_ANALYSIS.md`

## Overview

This plan addresses the memory growth issues identified in the MEMORY_GROWTH_ANALYSIS.md report. The primary causes are:

1. **Solution/Basis allocations** (~4.9 GB churn) - Transient allocations per solve
2. **Cut cloning** (~80 MB churn) - Full struct copies for lock-free sharing
3. **HashMap cloning** (~120 MB churn) - Unused parameter cloning
4. **Memory fragmentation** - Allocator not returning memory to OS

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Peak RSS | ~5 GB | < 2 GB |
| Memory growth/iteration | ~375 MB | < 5 MB |
| Minor page faults | 3.4M | < 500K |

## Epics

| Epic | Priority | Effort | Impact |
|------|----------|--------|--------|
| [1: Solver Buffer Reuse](./epic-01-solver-buffer-reuse/00-epic-overview.md) | HIGH | 1 week | ~4.9 GB eliminated |
| [2: Cut Cloning Elimination](./epic-02-cut-cloning-elimination/00-epic-overview.md) | MEDIUM | 3-5 days | ~80 MB eliminated |
| [3: HashMap Cloning Elimination](./epic-03-hashmap-cloning-elimination/00-epic-overview.md) | LOW | 2-3 days | ~120 MB eliminated |
| [4: Custom Allocator](./epic-04-custom-allocator/00-epic-overview.md) | LOW | 1-2 days | RSS reduction |

## Status Tracking

### Epic 1: Solver Buffer Reuse
- [x] Sprint 1: Buffer-into implementation ✅ Complete
  - [x] [TICKET-001](./epic-01-solver-buffer-reuse/sprint-01/ticket-001-add-solution-buffer-into.md) - Add Solution buffer-into
  - [x] [TICKET-002](./epic-01-solver-buffer-reuse/sprint-01/ticket-002-add-basis-buffer-into.md) - Add Basis buffer-into
  - [x] [TICKET-003](./epic-01-solver-buffer-reuse/sprint-01/ticket-003-update-realize-and-solve.md) - Update realize_and_solve (solution + basis)

### Epic 2: Cut Cloning Elimination
- [ ] Sprint 1: Arc-wrapped cuts
  - [ ] [TICKET-001](./epic-02-cut-cloning-elimination/sprint-01/ticket-001-update-cutpool-to-arc.md) - Update BendersCutPool to Arc
  - [ ] [TICKET-002](./epic-02-cut-cloning-elimination/sprint-01/ticket-002-update-cut-consumers.md) - Update cut consumers

### Epic 3: HashMap Cloning Elimination
- [x] Sprint 1: HashMap removal ✅ Complete (dead code removed)
  - [x] [TICKET-001](./epic-03-hashmap-cloning-elimination/sprint-01/ticket-001-replace-hashmap-clone.md) - Removed unused HashMap clone

### Epic 4: Custom Allocator (Optional)
- [x] Sprint 1: Mimalloc feature ✅ Complete
  - [x] [TICKET-001](./epic-04-custom-allocator/sprint-01/ticket-001-add-mimalloc-feature.md) - Add mimalloc feature flag

## Implementation Order

1. **Week 1**: Epic 1 (highest impact - eliminates 95% of allocation churn)
2. **Week 2**: Epic 3 (quick win - unused parameter removal)
3. **Week 2**: Epic 2 (medium effort, requires API changes)
4. **Week 2**: Epic 4 (optional, 1-2 days)

## Validation

After each epic:

```bash
# Quick validation
cargo run --release -- run examples/01-deterministic

# Memory profiling
/usr/bin/time -v ./target/release/powers run examples/05-large-scale-brazilian 2>&1 | \
  grep "Maximum resident set size"

# Detailed allocation tracking
valgrind --tool=massif ./target/release/powers run examples/01-deterministic
```

## Dependencies

This plan is independent of:
- `plans/preallocation-refactoring/` - That plan addresses HiGHS constraint preallocation (completed)
- Epic 3: Handler-Level SoA Blocks - Can be done in parallel

## References

1. `MEMORY_GROWTH_ANALYSIS.md` - Root cause analysis
2. `src/solver.rs` - Solution/Basis structs
3. `src/sddp/mod.rs` - Training loop, cut cloning
4. `src/subproblem.rs` - realize_and_solve method
