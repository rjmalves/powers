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
| 1 | [Foundation](./epic-01-foundation/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 2 | [Core Extraction](./epic-02-core-extraction/00-epic-overview.md) | 3 weeks | ⬜ Not Started |
| 3 | [Algorithm Separation](./epic-03-algorithm-separation/00-epic-overview.md) | 3 weeks | ⬜ Not Started |
| 4 | [State Simplification](./epic-04-state-simplification/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 5 | [Memory Optimization](./epic-05-memory-optimization/00-epic-overview.md) | 3 weeks | ⬜ Not Started |
| 6 | [Test Modernization](./epic-06-test-modernization/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 7 | [Performance Validation](./epic-07-performance-validation/00-epic-overview.md) | 1 week | ⬜ Not Started |

**Total Duration**: ~16 weeks

---

## Key Design Decisions (Updated)

### Epic 1: Foundation
- **Golden tests filter timing info** to avoid false failures from timing variance
- **Scripts use relative paths** for portability across setups
- **Memory benchmarks added** using example 05 for allocation profiling
- **Example 05 takes ~2 minutes** - plan timeouts accordingly

### Epic 2: Core Extraction
- **SoA-ready solution extraction**: All `extract_X()` methods have corresponding `extract_X_into()` for future slice-based extraction
- **Complete extraction list**: 12 solution extraction functions documented with specific source/target mapping
- **Constraint builders support preallocation**

### Epic 3: Algorithm Separation
- **Context structs document preallocation opportunities**
- **Data sizes known from input** - context structs track this for future buffer sizing
- **Timing preserves precise values** - parallel overhead computed separately, never overwrites

### Epic 4: State Simplification
- **State-Cut 1:1 relationship documented** - each cut has exactly one originating state
- **Slot indexing uses (iteration, forward_pass_idx)** - already used in current code
- **Pool-compatible trait extensions** added for Epic 5 migration
- **Trait object allocation points documented** with file:line references

### Epic 5: Memory Optimization
- **State pool is MANDATORY** (not optional) - paired with cut pool due to 1:1 relationship
- **Formal buffer structures**: `SolutionBuffer`, `BasisBuffer` with structured field access
- **Solver integration**: `get_solution_into()` and `get_basis_into()` methods
- **Trajectory buffer**: stores all stages for single iteration
- **Works for training AND simulation** steps
- **SoA analysis included** - may implement direct SoA conversion if beneficial

---

## Dependency Graph

```
Epic 1: Foundation
    │
    ├──→ Epic 2: Core Extraction
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

## Progress Tracking

### Phase 1: Foundation (Epic 1)
- [ ] Sprint 1: Module structure, golden tests (with timing filtering), memory benchmarks

### Phase 2: Core Extraction (Epic 2)
- [ ] Sprint 1: Solution extraction (SoA-ready, all 12 functions)
- [ ] Sprint 2: Constraint building extraction

### Phase 3: Algorithm Separation (Epic 3)
- [ ] Sprint 1: Forward pass extraction (preallocation-aware context)
- [ ] Sprint 2: Backward pass extraction

### Phase 4: State Simplification (Epic 4)
- [ ] Sprint 1: State trait consolidation, 1:1 relationship docs, pool interfaces

### Phase 5: Memory Optimization (Epic 5)
- [ ] Sprint 1: Cut-State pool, SolutionBuffer, TrajectoryBuffer
- [ ] Sprint 2: Integration, SoA analysis, simulation step

### Phase 6: Test Modernization (Epic 6)
- [ ] Sprint 1: Behavior-focused tests

### Phase 7: Performance Validation (Epic 7)
- [ ] Sprint 1: Final benchmarking and validation

---

## Before You Start Any Ticket

1. **Read the Critical Principles** in the [master plan](./00-master-plan.md#️-critical-principles-correctness-first-then-performance)
2. **Run golden output tests** to establish baseline
3. **Run benchmarks** to establish performance baseline
4. **Read the epic overview** for your ticket's epic
5. **Read the sprint overview** for context

## After Completing Any Ticket

1. **Run golden output tests** - must be bit-for-bit identical (timing filtered)
2. **Run all tests** - no regressions allowed
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
| `src/fcf.rs` | ~800 | FCF management, CutStatePair |

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
