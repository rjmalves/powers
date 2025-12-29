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
| 2 | [Core Extraction](./epic-02-core-extraction/00-epic-overview.md) | 3 weeks | ⬜ Not Started |
| 3 | [Algorithm Separation](./epic-03-algorithm-separation/00-epic-overview.md) | 3 weeks | ⬜ Not Started |
| 4 | [State Simplification](./epic-04-state-simplification/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 5 | [Memory Optimization](./epic-05-memory-optimization/00-epic-overview.md) | 3 weeks | ⬜ Not Started |
| 6 | [Test Modernization](./epic-06-test-modernization/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 7 | [Performance Validation](./epic-07-performance-validation/00-epic-overview.md) | 1 week | ⬜ Not Started |

**Total Duration**: ~16 weeks

---

## Current Focus: Epic 2 - Core Extraction

### Sprint 1: Solution Extraction
| ID | Title | Status |
|----|-------|--------|
| [T-006](./epic-02-core-extraction/sprint-01/ticket-006-analyze-extraction-points.md) | Analyze subproblem.rs extraction points | ⬜ |
| [T-007](./epic-02-core-extraction/sprint-01/ticket-007-variable-indices-struct.md) | Create VariableIndices and ConstraintIndices | ⬜ |
| [T-008](./epic-02-core-extraction/sprint-01/ticket-008-solution-extractor-scaffold.md) | Create SolutionExtractor scaffold | ⬜ |
| [T-009](./epic-02-core-extraction/sprint-01/ticket-009-extract-hydro-solution.md) | Extract hydro solution extraction | ⬜ |
| [T-010](./epic-02-core-extraction/sprint-01/ticket-010-extract-thermal-solution.md) | Extract thermal and exchange extraction | ⬜ |
| [T-011](./epic-02-core-extraction/sprint-01/ticket-011-extract-remaining-solutions.md) | Extract remaining solutions | ⬜ |

### Sprint 2: Constraint Extraction
| ID | Title | Status |
|----|-------|--------|
| [T-012](./epic-02-core-extraction/sprint-02/ticket-012-constraints-module-structure.md) | Create constraints module structure | ⬜ |
| [T-013](./epic-02-core-extraction/sprint-02/ticket-013-hydro-balance-constraints.md) | Extract hydro balance constraints | ⬜ |
| [T-014](./epic-02-core-extraction/sprint-02/ticket-014-bus-balance-constraints.md) | Extract bus balance constraints | ⬜ |
| [T-015](./epic-02-core-extraction/sprint-02/ticket-015-ar-dynamics-constraints.md) | Extract AR dynamics constraints | ⬜ |
| [T-016](./epic-02-core-extraction/sprint-02/ticket-016-bound-constraints.md) | Extract bound constraints (optional) | ⬜ |
| [T-017](./epic-02-core-extraction/sprint-02/ticket-017-refactor-subproblem-facade.md) | Refactor subproblem.rs to use new modules | ⬜ |

---

## Key Design Decisions

### Epic 1: Foundation ✅
- **Golden tests filter timing info** to avoid false failures from timing variance
- **Scripts use relative paths** for portability across setups
- **Memory benchmarks added** using example 05 for allocation profiling
- **Example 05 takes ~2 minutes** - plan timeouts accordingly
- **Timing module implemented** with feature-gated compilation

### Epic 2: Core Extraction
- **SoA-ready solution extraction**: All `extract_X()` methods have corresponding `extract_X_into()` for future slice-based extraction
- **Complete extraction list**: 11 solution extraction functions documented with specific source/target mapping
- **Constraint builders use ConstraintContext** for clean parameter passing
- **Facade pattern** preserves existing API while delegating to new modules

### Epic 3: Algorithm Separation
- **Context structs document preallocation opportunities**
- **Data sizes known from input** - context structs track this for future buffer sizing
- **Timing preserves precise values** - parallel overhead computed separately, never overwrites

### Epic 4: State Simplification
- **State-Cut 1:1 relationship documented** - each cut has exactly one originating state
- **Slot indexing uses (iteration, forward_pass_idx)** - already used in current code
- **Pool-compatible trait extensions** added for Epic 5 migration

### Epic 5: Memory Optimization
- **State pool is MANDATORY** (not optional) - paired with cut pool due to 1:1 relationship
- **Formal buffer structures**: `SolutionBuffer`, `BasisBuffer` with structured field access
- **Trajectory buffer**: stores all stages for single iteration
- **Works for training AND simulation** steps

---

## Dependency Graph

```
Epic 1: Foundation ✅
    │
    ├──→ Epic 2: Core Extraction ← CURRENT
    │        │
    │        └──→ Epic 3: Algorithm Separation
    │                  │
    │                  ├──→ Epic 4: State Simplification
    │                  │         │
    │                  │         └──→ Epic 5: Memory Optimization
    │                  │
    │                  └──────────────→ Epic 5: Memory Optimization
    │
    └──→ Epic 6: Test Modernization (can start after Epic 1)
              │
              └──→ Epic 7: Performance Validation (requires all epics)
```

---

## Before You Start Any Ticket

1. **Read the Critical Principles** in the [master plan](./00-master-plan.md#️-critical-principles-correctness-first-then-performance)
2. **Run golden output tests** to establish baseline: `./scripts/golden-tests.sh verify`
3. **Run benchmarks** to establish performance baseline
4. **Read the epic overview** for your ticket's epic
5. **Read the sprint overview** for context

## After Completing Any Ticket

1. **Run golden output tests** - must be bit-for-bit identical (timing filtered)
2. **Run all tests** - `cargo test`
3. **Run benchmarks** - no >5% regression allowed
4. **If ANY validation fails**: STOP and ask for clarification

---

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `src/sddp/mod.rs` | 3,913 | Main SDDP training loop, forward/backward passes |
| `src/subproblem.rs` | 6,631 | LP model building, constraint generation, solution extraction |
| `src/state.rs` | 3,087 | State representations, trajectory storage |
| `src/sddp/builder.rs` | 1,494 | SDDP instance construction |
| `src/solver.rs` | 1,235 | HiGHS solver interface |
| `src/model/` | - | New module for extracted code |

---

## Example Execution Times

⚠️ **Plan timeouts accordingly**:
- Examples 01-04, 06-07: < 30 seconds each
- **Example 05 (large-scale-brazilian): Up to 2 minutes**

---

## Status Legend

- ⬜ Not Started
- 🟡 In Progress
- ✅ Complete
- 🔴 Blocked
