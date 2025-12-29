# Sprint 1: Solution Extraction

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

This sprint extracts code from `subproblem.rs`. **The algorithm logic must remain unchanged.** 

Run golden tests after EVERY extraction. If any test fails, **STOP immediately** and investigate before proceeding.

---

## Goals

1. **Primary**: Extract all `get_*_from_solution()` functions into `SolutionExtractor`
2. **Primary**: Create `VariableIndices` and `ConstraintIndices` structs for clean index management
3. **Secondary**: Reduce function sizes and parameter counts
4. **Validation**: Maintain bit-for-bit identical outputs

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-006](./ticket-006-analyze-extraction-points.md) | Analyze subproblem.rs extraction points | 2 | Yes | Epic 1 complete | ⬜ |
| [T-007](./ticket-007-variable-indices-struct.md) | Create VariableIndices and ConstraintIndices structs | 3 | Yes | T-006 | ⬜ |
| [T-008](./ticket-008-solution-extractor-scaffold.md) | Create SolutionExtractor scaffold with dual API | 2 | Yes | T-007 | ⬜ |
| [T-009](./ticket-009-extract-hydro-solution.md) | Extract hydro solution extraction | 3 | Yes | T-008 | ⬜ |
| [T-010](./ticket-010-extract-thermal-solution.md) | Extract thermal and exchange solution extraction | 2 | Yes | T-008 | ⬜ |
| [T-011](./ticket-011-extract-remaining-solutions.md) | Extract remaining solution extractions | 3 | Yes | T-009, T-010 | ⬜ |

**Total Points**: 15

---

## Parallelization

```
Week 1:
  T-006 (Analysis) ──→ T-007 (Indices) ──→ T-008 (Scaffold)
  
Week 2:
  T-008 ──→ T-009 (Hydro) ──────────────────────┐
       └──→ T-010 (Thermal) ───────────────────├──→ T-011 (Remaining)
```

- **T-009** and **T-010** can run in parallel after T-008
- **T-011** consolidates remaining extractions after T-009/T-010

---

## Dependencies

- **From Epic 1**: 
  - Golden test infrastructure (T-001)
  - Baseline benchmarks (T-002)
  - Module skeleton (T-005)
- **To Sprint 2**: 
  - VariableIndices and ConstraintIndices structs
  - SolutionExtractor pattern established

---

## Key Files

| File | Lines | Purpose in This Sprint |
|------|-------|------------------------|
| `src/subproblem.rs` | 6,631 | Source of extraction |
| `src/model/mod.rs` | - | Module home |
| `src/model/variable_indices.rs` | - | New: VariableIndices |
| `src/model/constraint_indices.rs` | - | New: ConstraintIndices |
| `src/model/solution_extract.rs` | - | New: SolutionExtractor |

---

## Verification Checklist

After EVERY ticket:

- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] Extracted functions are ≤50 lines
- [ ] Extracted functions have ≤4 parameters

After sprint complete:

- [ ] `cargo bench` shows no regression (within 5%)
- [ ] All solution extraction is in new module
- [ ] `subproblem.rs` can use new extractor

---

## Risks

| Risk | Mitigation |
|------|------------|
| Hidden state in subproblem.rs | Careful analysis in T-006 |
| Index mapping errors | Comprehensive tests, golden test validation |
| Performance from indirection | Benchmark after sprint, use `#[inline]` |

---

## Definition of Done

- [ ] All 6 tickets complete
- [ ] All tests passing
- [ ] Golden tests passing
- [ ] Benchmarks within 5% of baseline
- [ ] Code reviewed and merged
- [ ] SolutionExtractor fully functional
