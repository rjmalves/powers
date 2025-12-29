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
| 3 | [Algorithm Separation](./epic-03-algorithm-separation/00-epic-overview.md) | 4-5 weeks | ✅ Complete |
| 4 | [State Simplification](./epic-04-state-simplification/00-epic-overview.md) | 3 weeks | ⬜ **Ready to Start** |
| 5 | [Memory Optimization](./epic-05-memory-optimization/00-epic-overview.md) | 3 weeks | ⬜ Not Started |
| 6 | [Test Modernization](./epic-06-test-modernization/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 7 | [Performance Validation](./epic-07-performance-validation/00-epic-overview.md) | 1 week | ⬜ Not Started |

**Total Duration**: ~18-19 weeks

---

## Current Focus: Epic 4 - State Simplification

### Sprint 1: State Consolidation ⬜ Ready to Start

| ID | Title | Status |
|----|-------|--------|
| [T-038](./epic-04-state-simplification/sprint-01/ticket-038-analyze-state-structure.md) | Analyze state.rs structure and duplication | ⬜ **NEXT** |
| [T-039](./epic-04-state-simplification/sprint-01/ticket-039-document-state-cut-relationship.md) | Document State-Cut 1:1 relationship | ⬜ |
| [T-040](./epic-04-state-simplification/sprint-01/ticket-040-extract-state-utilities.md) | Extract common state utilities (StateCore) | ⬜ |
| [T-041](./epic-04-state-simplification/sprint-01/ticket-041-consolidate-storage-state.md) | Consolidate StorageState methods | ⬜ |
| [T-042](./epic-04-state-simplification/sprint-01/ticket-042-consolidate-inflow-state.md) | Consolidate StorageAndInflowState methods | ⬜ |
| [T-043](./epic-04-state-simplification/sprint-01/ticket-043-pool-compatible-extensions.md) | Add pool-compatible trait extensions | ⬜ |
| [T-044](./epic-04-state-simplification/sprint-01/ticket-044-state-extraction-module.md) | Create/document state extraction module | ⬜ |

### Sprint 2: FCF Graph Wrapper Removal ⬜ After Sprint 1

| ID | Title | Status |
|----|-------|--------|
| [T-045](./epic-04-state-simplification/sprint-02/ticket-045-analyze-fcf-access-patterns.md) | Analyze and document FCF access patterns | ⬜ |
| [T-046](./epic-04-state-simplification/sprint-02/ticket-046-remove-mutex-from-fcf-type.md) | Remove Mutex from FCF graph type | ⬜ |
| [T-047](./epic-04-state-simplification/sprint-02/ticket-047-update-coordinator-fcf-access.md) | Update coordinator FCF access | ⬜ |
| [T-048](./epic-04-state-simplification/sprint-02/ticket-048-update-output-fcf-access.md) | Update output modules FCF access | ⬜ |
| [T-049](./epic-04-state-simplification/sprint-02/ticket-049-verify-fcf-refactoring.md) | Verify FCF refactoring end-to-end | ⬜ |

---

## Epic 4 Overview

### Sprint 1: State Consolidation (19 points)

**Goal**: Reduce duplication in `state.rs` via `StateCore` composition pattern.

**Key Deliverables**:
- `StateCore` struct with common fields
- Both state types using composition
- Pool-compatible trait extensions
- State-Cut relationship documented

### Sprint 2: FCF Graph Wrapper Removal (12 points)

Based on [FCF_GRAPH_ARCHITECTURE_ANALYSIS.md](../docs/FCF_GRAPH_ARCHITECTURE_ANALYSIS.md):

**Problem**: `Arc<Mutex<FutureCostFunction>>` is unnecessary—FCF is only modified in single-threaded Phase 2.

**Solution**: Replace with just `FutureCostFunction` for:
- Cleaner type signatures
- Compile-time borrow checker guarantees
- Marginal performance improvement

---

## Key Architectural Decisions

### StateCore Composition Pattern (Sprint 1)

Extract common state fields into shared struct:

```rust
pub struct StateCore {
    pub dimension: usize,
    pub state_coefficients: Vec<f64>,
    pub dominating_objective: f64,
    pub dominating_cut_id: usize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
}

pub struct StorageState {
    core: StateCore,  // Embed shared fields
}
```

### FCF Graph Simplification (Sprint 2)

Remove `Arc<Mutex<>>` from FCF graph:
- No concurrent access exists
- All lock sites are single-threaded
- Borrow checker enforces safety at compile time

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
