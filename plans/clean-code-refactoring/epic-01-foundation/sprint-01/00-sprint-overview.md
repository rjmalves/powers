# Sprint 1: Infrastructure Setup

> **Epic**: [Epic 1: Foundation](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

This sprint establishes the validation infrastructure for the entire refactoring. **Correctness is paramount.** The golden tests created here will be the primary mechanism for detecting algorithmic regressions throughout the project.

If any test shows non-deterministic behavior or unexpected results, **STOP and investigate** before proceeding.

---

## Goals

1. **Primary**: Establish golden output test infrastructure for correctness validation
2. **Primary**: Capture baseline performance benchmarks (time AND memory)
3. **Secondary**: Create timing module foundation
4. **Secondary**: Set up module directory structure

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-001](./ticket-001-golden-test-infrastructure.md) | Create golden test infrastructure | 3 | Yes | None | ⬜ |
| [T-002](./ticket-002-baseline-benchmarks.md) | Capture baseline benchmarks | 3 | Yes | None | ⬜ |
| [T-003](./ticket-003-timing-module-guard.md) | Implement TimingGuard | 3 | Yes | T-001 | ⬜ |
| [T-004](./ticket-004-timing-module-collector.md) | Implement TimingCollector trait | 3 | Yes | T-003 | ⬜ |
| [T-005](./ticket-005-module-skeleton.md) | Create module directory skeleton | 1 | Yes | None | ⬜ |

**Total Points**: 13

---

## Parallelization

```
Week 1:
  T-001 (Golden Tests) ──────────────┐
                                     ├──→ T-003 (TimingGuard)
  T-002 (Benchmarks) ────────────────┘
  T-005 (Module Skeleton) ───────────────────────────────────→

Week 2:
  T-003 (TimingGuard) ──→ T-004 (TimingCollector)
```

- **T-001** and **T-002** can run in parallel (Week 1)
- **T-005** is independent, can run anytime
- **T-003** should wait for T-001 to ensure golden tests are available for validation
- **T-004** depends on T-003 (builds on TimingGuard)

---

## Dependencies

- **From Previous Sprint**: None (first sprint)
- **To Next Sprint (Epic 2, Sprint 1)**: 
  - Golden test infrastructure must be complete
  - Baseline benchmarks (time AND memory) must be documented
  - Module skeleton must exist

---

## Key Execution Time Notes

⚠️ **Example execution times to be aware of**:
- Examples 01-04, 06-07: < 30 seconds each
- **Example 05 (large-scale-brazilian): Up to 2 minutes**

Plan CI timeouts and manual testing accordingly.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Non-deterministic outputs | Verify 3 runs with same seed before accepting golden outputs; filter timing info |
| Benchmark high variance | Use criterion with sufficient samples, document variance |
| Feature flag complexity | Follow patterns from existing Cargo.toml features |
| Example 05 timeout | Set appropriate timeouts (5+ minutes for safety margin) |

---

## Verification Checklist

Before marking this sprint complete:

- [ ] Golden tests pass consistently (run 3x with same seed)
- [ ] Timing info properly filtered from golden test comparisons
- [ ] Baseline benchmarks documented with statistical confidence
- [ ] **Memory baseline captured** (Peak RSS, allocation patterns)
- [ ] Timing module compiles with `--features timing` and without
- [ ] All existing tests pass (`cargo test`)
- [ ] No changes to algorithm logic (diff shows only additions)

---

## Definition of Done

- [ ] All 5 tickets complete
- [ ] All tests passing
- [ ] Documentation updated (README for golden tests, timing module docs)
- [ ] Performance benchmarks (time + memory) baselined and documented
- [ ] Code reviewed and merged
- [ ] Golden test infrastructure verified working

---

## Notes

This sprint is **foundational**—all future sprints depend on the infrastructure created here. Take extra care to:

1. Ensure golden tests are truly deterministic (timing filtered out)
2. Document benchmark methodology clearly (including memory profiling)
3. Keep timing module simple and well-tested
4. Verify everything before proceeding to Epic 2
5. Be patient with example 05—it takes up to 2 minutes
