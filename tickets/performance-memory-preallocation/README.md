# Performance Memory Pre-allocation - Ticket Index

## Overview

This directory contains detailed implementation tickets for the Performance Implementation Plan (memory pre-allocation strategy). The plan targets 15-20% performance improvement by eliminating allocation overhead in hot paths.

**Target**: Reduce malloc overhead from 5.28% to <2%  
**Expected Impact**: Runtime improvement from 34.0s to ~28.9s (15-20% faster)  
**Timeline**: 4 weeks across 4 phases

## Phase 1: Core Buffer Management Infrastructure (Week 1)

**Goal**: Create foundational buffer management abstractions  
**Risk**: 🟢 LOW  
**Impact**: 🔴 HIGH (enables all subsequent work)

- [x] **TICKET-001**: Implement SizingInfo struct for buffer dimension computation (2 days)
  - Compute all buffer sizes from input configuration
  - Centralized sizing logic
  - Status: ⬜ Not Started

- [x] **TICKET-002**: Implement Buffer Pool abstractions for memory reuse (3 days)
  - Generic Buffer<T> and BufferPool<T>
  - ThreadLocalBuffers for parallel execution
  - Status: ⬜ Not Started

- [x] **TICKET-003**: Integrate memory module into main codebase (1 day)
  - Module exports and documentation
  - API design and organization
  - Status: ⬜ Not Started

- [x] **TICKET-004**: Create comprehensive test infrastructure for memory module (3 days)
  - Unit, integration, property, and performance tests
  - >90% coverage target
  - Status: ⬜ Not Started

**Phase 1 Total**: 9 story points (5 working days)

---

## Phase 2: Backward Pass Optimization (Week 2)

**Goal**: Eliminate allocations in backward pass (highest impact)  
**Risk**: 🟡 MEDIUM  
**Impact**: 🔴 HIGH (8-10% overall improvement)

- [x] **TICKET-005**: Implement BackwardPassBuffers for pre-allocated backward pass execution (2 days)
  - Buffer infrastructure for backward pass
  - Integration with SddpAlgorithm
  - Status: ⬜ Not Started

- [x] **TICKET-006**: Refactor backward_pass() to use pre-allocated buffers (3 days)
  - Eliminate ~60 allocations per iteration
  - Zero-allocation hot path
  - Status: ⬜ Not Started

- [x] **TICKET-007**: Performance validation and benchmarking for backward pass (2 days)
  - Profile before/after comparison
  - Benchmark suite
  - Status: ⬜ Not Started

**Phase 2 Total**: 10 story points (5 working days)

---

## Phase 3: Forward Pass and Subproblem Optimization (Week 3)

**Goal**: Eliminate allocations in forward pass and subproblem solve  
**Risk**: 🟡 MEDIUM  
**Impact**: 🟡 MEDIUM (5-7% additional improvement)

- [x] **TICKET-008**: Implement ForwardPassBuffers and optimize forward pass (3 days)
  - Trajectory buffer pre-allocation
  - State buffer reuse
  - Status: ⬜ Not Started

- [x] **TICKET-009**: Implement SubproblemBuffers and integrate with Subproblem (3 days)
  - Realization buffers
  - State extraction buffers
  - Status: ⬜ Not Started

- [x] **TICKET-010**: Audit and fix Vec::new() in hot paths (1 day)
  - Replace with Vec::with_capacity()
  - Target files: sddp, subproblem, fcf, scenario_generator
  - Status: ⬜ Not Started

**Phase 3 Total**: 11 story points (5.5 working days)

---

## Phase 4: Integration, Testing, and Validation (Week 4)

**Goal**: Comprehensive validation and performance measurement  
**Risk**: 🟢 LOW  
**Impact**: 🔴 HIGH (ensures correctness)

- [x] **TICKET-011**: Comprehensive integration testing (2 days)
  - Full training and simulation tests
  - Buffer reuse validation
  - Status: ⬜ Not Started

- [x] **TICKET-012**: Performance benchmarking suite (2 days)
  - Before/after comparisons
  - Multiple examples
  - Status: ⬜ Not Started

- [x] **TICKET-013**: Profiling validation and metrics collection (1 day)
  - Verify malloc overhead <2%
  - Collect final metrics
  - Status: ⬜ Not Started

- [x] **TICKET-014**: Documentation updates and finalization (1 day)
  - Update PERFORMANCE_REFACTORING_PLAN.md
  - Update README.md
  - Create architecture diagrams
  - Status: ⬜ Not Started

**Phase 4 Total**: 9 story points (4.5 working days)

---

## Summary

**Total Effort**: 39 story points (~20 working days, 4 weeks)

**Phase Breakdown**:
- Phase 1 (Infrastructure): 9 points (23%)
- Phase 2 (Backward Pass): 10 points (26%)
- Phase 3 (Forward Pass): 11 points (28%)
- Phase 4 (Validation): 9 points (23%)

**Risk Distribution**:
- Low Risk: 22 points (56%)
- Medium Risk: 17 points (44%)
- High Risk: 0 points (0%)

**Expected Outcomes**:
- Runtime: 34.0s → ~28.9s (15-20% faster)
- Malloc overhead: 5.28% → <2% (60% reduction)
- Memory peak: 2.4GB → 2.6GB (8% increase, acceptable)
- Code quality: +5 new modules, better organized

---

## Dependencies Graph

```
Phase 1:
  TICKET-001 (SizingInfo) ──┐
                            ├──> TICKET-003 (Integration) ──> TICKET-004 (Tests)
  TICKET-002 (Buffers) ─────┘

Phase 2:
  TICKET-003 ──> TICKET-005 (BackwardPassBuffers) ──> TICKET-006 (Refactor) ──> TICKET-007 (Validate)

Phase 3:
  TICKET-003 ──> TICKET-008 (ForwardPassBuffers) ──┐
  TICKET-003 ──> TICKET-009 (SubproblemBuffers) ───┼──> TICKET-010 (Vec audit)
                                                     │
Phase 4:                                            │
  All Phase 2 & 3 ────────────────────────────────┴──> TICKET-011, 012, 013, 014
```

---

## Sprint Planning Recommendation

### Sprint 1 (Week 1): Foundation
- TICKET-001: SizingInfo (2 days)
- TICKET-002: Buffer abstractions (3 days)

### Sprint 2 (Week 1-2): Integration & Backward Pass
- TICKET-003: Module integration (1 day)
- TICKET-004: Test infrastructure (3 days)
- TICKET-005: BackwardPassBuffers (2 days)

### Sprint 3 (Week 2): Backward Pass Optimization
- TICKET-006: Backward pass refactoring (3 days)
- TICKET-007: Performance validation (2 days)

### Sprint 4 (Week 3): Forward Pass & Subproblem
- TICKET-008: ForwardPassBuffers (3 days)
- TICKET-009: SubproblemBuffers (3 days)

### Sprint 5 (Week 3-4): Finalization
- TICKET-010: Vec::with_capacity audit (1 day)
- TICKET-011: Integration testing (2 days)
- TICKET-012: Benchmarking (2 days)

### Sprint 6 (Week 4): Validation & Documentation
- TICKET-013: Profiling validation (1 day)
- TICKET-014: Documentation (1 day)
- Buffer: 2 days for unexpected issues

---

## Success Criteria

**Must Have**:
- [ ] All tests pass (446 existing + new tests)
- [ ] Malloc overhead <2.5%
- [ ] Runtime improvement >10%
- [ ] No numerical regressions (results identical within 1e-10)
- [ ] No clippy warnings
- [ ] Documentation complete

**Nice to Have**:
- [ ] Runtime improvement >15%
- [ ] Malloc overhead <2%
- [ ] Memory usage <2.5GB peak
- [ ] Examples in documentation

---

## How to Use This Directory

1. **Start with Phase 1**: Complete tickets in order (001 → 002 → 003 → 004)
2. **Review after each phase**: Validate metrics before moving to next phase
3. **Track progress**: Update status in this file as tickets complete
4. **Document learnings**: Add notes to tickets as you discover insights
5. **Measure everything**: Profile and benchmark at each phase boundary

---

## Related Documents

- **PERFORMANCE_IMPLEMENTATION_PLAN.md**: Detailed implementation strategy
- **PERFORMANCE_REFACTORING_PLAN.md**: Overall performance optimization roadmap
- **PROFILING_ANALYSIS.md**: Profiling results and bottleneck analysis
- **REFACTORING_PLAN.md**: Code quality improvement plan (follows this)

---

## Contact & Questions

For questions about these tickets:
1. Read the detailed ticket (each has context, technical notes, and examples)
2. Check PERFORMANCE_IMPLEMENTATION_PLAN.md for architecture details
3. Review related code in the specified files
4. Consult profiling data in PROFILING_ANALYSIS.md

---

**Created**: 2025-11-10  
**Status**: Ready for Implementation  
**Owner**: Performance Optimization Team
