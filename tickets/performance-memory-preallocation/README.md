# Performance Memory Pre-allocation - Ticket Index

## Overview

This directory contains detailed implementation tickets for the Performance Implementation Plan (memory pre-allocation strategy). The plan targets 8-12% overall performance improvement by eliminating allocation overhead in hot paths.

**Target**: Reduce malloc overhead from 8-10% (actual) to <2%  
**Expected Impact**: Backward pass 10-15% faster, overall training 8-12% faster  
**Timeline**: 2-3 weeks across 4 phases  
**Status**: ⚠️ **REVISED STRATEGY** (2025-11-10)

---

## 🔴 **CRITICAL REVISION** (2025-11-10)

During TICKET-006 implementation, we discovered that current memory estimation **underestimates by 23×** because it only accounts for stack sizes, missing nested heap allocations.

**Key Finding**: Real malloc overhead is 8-10% (not ~2%), and nested allocations (inside `BendersCut::coefficients`, state vectors, etc.) dominate performance costs.

**Strategic Change**: 
1. Build accurate measurement foundation first (TICKET-000)
2. Systematically eliminate nested allocations using accurate sizing
3. Apply uniform pattern across backward pass, forward pass, and simulation

**See**: `MEMORY_OPTIMIZATION_STRATEGY.md` for complete technical analysis and rationale.

---

## Phase 0: Foundation (NEW - CRITICAL PRIORITY) 🚨

**Goal**: Implement accurate deep memory estimation to enable data-driven optimization  
**Risk**: 🟢 LOW (purely additive, well-defined pattern)  
**Impact**: 🔴 CRITICAL (blocks all subsequent optimization work)

- [x] **TICKET-000**: Implement Deep Memory Estimation with DeepSizeEstimate Trait (1-2 days) 🆕
  - Implement `DeepSizeEstimate` trait for recursive size computation
  - Account for nested heap allocations in all domain types
  - Validate accuracy (within 10% of actual memory usage)
  - Foundation for all buffer pre-allocation decisions
  - Status: 📋 **READY TO START IMMEDIATELY**
  - Priority: **P0 - CRITICAL**
  - Blocks: TICKET-006b, TICKET-007, all future optimization

**Phase 0 Total**: 8 story points (1-2 days)

---

## Phase 1: Core Buffer Management Infrastructure ✅

**Goal**: Create foundational buffer management abstractions  
**Risk**: 🟢 LOW  
**Impact**: 🔴 HIGH (enables all subsequent work)  
**Status**: ✅ **COMPLETE**

- [x] **TICKET-001**: Implement SizingInfo struct for buffer dimension computation (4 hours)
  - Compute all buffer sizes from input configuration
  - Centralized sizing logic with per-node heterogeneous support
  - Status: ✅ COMPLETE
  - Note: Provides shallow estimation; enhanced by TICKET-000 for deep estimation

- [x] **TICKET-002**: Implement Buffer Pool abstractions for memory reuse (2 hours)
  - Generic Buffer<T> and BufferPool<T>
  - ThreadLocalBuffers for parallel execution
  - Status: ✅ COMPLETE

- [x] **TICKET-003**: Integrate memory module into main codebase (1 hour)
  - Module exports and documentation
  - API design and organization
  - Status: ✅ COMPLETE

- [~] **TICKET-004**: Create comprehensive test infrastructure for memory module (3 days)
  - Status: ⏭️ SKIPPED (sufficient coverage from TICKET-001-003)

**Phase 1 Total**: 7 hours (< 1 day) ✅ **COMPLETE**

---

## Phase 2: Backward Pass Optimization (REVISED) 🔄

**Goal**: Eliminate ALL allocations in backward pass (outer + nested)  
**Risk**: 🟡 MEDIUM  
**Impact**: 🔴 HIGH (10-15% overall backward pass improvement)  
**Status**: 🔄 **PARTIALLY COMPLETE** (outer done, nested pending)

- [x] **TICKET-005**: Implement BackwardPassBuffers for pre-allocated backward pass execution (2 hours)
  - Buffer infrastructure for backward pass
  - Integration with SddpAlgorithm
  - Status: ✅ COMPLETE (but unused in final implementation)
  - Note: Different pattern chosen (thread-local buffers in TICKET-006b)

- [x] **TICKET-006**: Refactor backward_pass() to use pre-allocated buffers (4 hours)
  - Eliminate ~30,720 outer allocations (unzip operation)
  - Pre-allocation with exact capacity
  - Status: ✅ COMPLETE
  - Impact: Eliminates outer allocations, but nested allocations remain

- [ ] **TICKET-006b**: Eliminate Nested Allocations with Thread-Local Buffers (2 days) 🆕
  - Pre-allocate coefficient buffers (thread-local)
  - Pre-allocate state buffers (thread-local)
  - Eliminate ~61,440 nested allocations per training run
  - Reduce malloc overhead from 8-10% to <2%
  - Status: 📋 **BLOCKED BY TICKET-000**
  - Priority: **P1 - HIGH**
  - Expected: 10-15% backward pass improvement

- [ ] **TICKET-007**: Performance validation and benchmarking for backward pass (2 days) 📝 **UPDATED SCOPE**
  - Profile COMPLETE optimization (TICKET-006 + TICKET-006b)
  - Validate malloc overhead <2%
  - Measure 10-15% improvement at production scale (192 FPs)
  - Confirm ~92K allocations eliminated
  - Status: 📋 **BLOCKED BY TICKET-006b**

**Phase 2 Total**: 14 story points (3-4 days)  
**Status**: 30% complete (outer allocations done, nested allocations pending)

---

## Phase 3: Forward Pass and Simulation Optimization (REVISED) 📋

**Goal**: Apply thread-local buffer pattern to forward pass and simulation  
**Risk**: 🟡 MEDIUM  
**Impact**: 🟡 MEDIUM (5-10% additional improvement each)  
**Status**: 📋 **PLANNED** (blocked by Phase 2 completion)

- [ ] **TICKET-008**: Forward Pass Deep Pre-allocation (2 days) 📝 **UPDATED**
  - Apply thread-local buffer pattern from TICKET-006b
  - Pre-allocate action vectors
  - Pre-allocate realization buffers
  - Expected: 5-10% forward pass improvement
  - Status: 📋 **BLOCKED BY TICKET-007**
  - Uses: Deep sizing from TICKET-000

- [ ] **TICKET-009**: Simulation Buffer Pre-allocation (2 days) 📝 **UPDATED**
  - Apply same pattern to simulation
  - Pre-allocate trajectory buffers
  - Pre-allocate scenario result structures
  - Expected: 5-10% simulation improvement
  - Status: 📋 **BLOCKED BY TICKET-008**
  - Uses: Deep sizing from TICKET-000

- [ ] **TICKET-010**: Audit and fix Vec::new() in hot paths (1 day)
  - Replace with Vec::with_capacity() using deep sizing
  - Target files: sddp, subproblem, fcf, scenario_generator
  - Only if profiling reveals additional opportunities
  - Status: 📋 **CONDITIONAL** (based on TICKET-007 profiling)

**Phase 3 Total**: 11 story points (4-5 days)

---

## Phase 4: Integration, Testing, and Validation 📊

**Goal**: Comprehensive validation and performance measurement  
**Risk**: 🟢 LOW  
**Impact**: 🔴 HIGH (ensures correctness and documents improvements)  
**Status**: 📋 **PLANNED**

- [ ] **TICKET-011**: Comprehensive integration testing (2 days)
  - Full training and simulation tests with all optimizations
  - Numerical validation across all examples
  - Thread-safety stress testing
  - Status: 📋 **WAITING**

- [ ] **TICKET-012**: Create benchmark suite for performance regression prevention (2 days)
  - Criterion benchmarks for all hot paths
  - CI integration for performance monitoring
  - Baseline comparisons
  - Status: 📋 **WAITING**

- [ ] **TICKET-013**: Performance profiling and validation (2 days)
  - Comprehensive profiling report
  - Before/after comparison
  - Production-scale validation
  - Status: 📋 **WAITING**

- [ ] **TICKET-014**: Documentation finalization and examples (2 days)
  - Update all performance documentation
  - Create optimization guide
  - Document patterns for future work
  - Status: 📋 **WAITING**

**Phase 4 Total**: 11 story points (4-5 days)

---

## Sprint Timeline (REVISED)

### Week 1: Foundation + Backward Pass Nested Optimization
- **TICKET-000**: Deep memory estimation (1-2 days) ⚡ START IMMEDIATELY
- **TICKET-006b**: Nested pre-allocation (2 days)
- **TICKET-007**: Performance validation (2 days)

### Week 2: Forward Pass + Simulation
- **TICKET-008**: Forward pass optimization (2 days)
- **TICKET-009**: Simulation optimization (2 days)
- **TICKET-010**: Vec capacity audit (conditional, 1 day)

### Week 3: Integration & Validation
- **TICKET-011**: Integration testing (2 days)
- **TICKET-012**: Benchmark suite (2 days)
- **TICKET-013**: Profiling validation (2 days)

### Week 4: Documentation & Polish (if needed)
- **TICKET-014**: Documentation finalization (2 days)
- Buffer for unexpected issues

**Total Estimated Time**: 2-3 weeks (depends on TICKET-010 need)

---

## Current Status

| Phase | Status | Completion | Notes |
|-------|--------|-----------|-------|
| Phase 0 | 📋 Ready | 0% | CRITICAL - Start immediately |
| Phase 1 | ✅ Complete | 100% | Infrastructure done |
| Phase 2 | 🔄 In Progress | 30% | Outer done, nested pending |
| Phase 3 | 📋 Planned | 0% | Blocked by Phase 2 |
| Phase 4 | 📋 Planned | 0% | Blocked by Phase 3 |

**Overall Progress**: ~20% complete (infrastructure + partial optimization)

---

## Priority Order

1. **🚨 TICKET-000** (CRITICAL): Deep memory estimation - START NOW
2. **HIGH**: TICKET-006b - Completes backward pass optimization
3. **HIGH**: TICKET-007 - Validates optimization effectiveness
4. **MEDIUM**: TICKET-008, 009 - Extends pattern to other areas
5. **LOW**: TICKET-010-014 - Polish and validation

---

## Key Changes from Original Plan

### What Changed

1. **Added TICKET-000**: Deep memory estimation foundation (new Phase 0)
2. **Added TICKET-006b**: Nested allocation elimination (completes Phase 2)
3. **Updated TICKET-007**: Expanded scope to validate complete optimization
4. **Updated TICKET-008, 009**: Use deep sizing from TICKET-000
5. **Revised estimates**: More realistic based on actual findings

### Why Changed

- **Discovery**: Current memory estimation underestimates by 23×
- **Impact**: Real malloc overhead is 8-10% (not ~2%)
- **Strategy**: Build accurate foundation, then optimize systematically
- **Result**: Better optimization with data-driven decisions

### Expected Improvement

| Metric | Original Target | Revised Target | Rationale |
|--------|----------------|----------------|-----------|
| Malloc overhead | <2% | <2% | Same target, better path |
| Backward pass | 8-10% faster | 10-15% faster | More allocations found |
| Overall training | 15-20% faster | 8-12% faster | More realistic estimate |
| Memory estimation | Not scoped | Within 10% | New foundation |

---

## References

- **Strategy Document**: `MEMORY_OPTIMIZATION_STRATEGY.md` - Complete technical analysis
- **Sprint Status**: `SPRINT-STATUS.md` - Current progress and decisions
- **Performance Plan**: `PERFORMANCE_REFACTORING_PLAN.md` - Original plan (needs update)
- **Completion Reports**: `TICKET-00X-COMPLETE.md` - Detailed results per ticket

---

**Last Updated**: 2025-11-10 (Sprint Revision)  
**Next Review**: After TICKET-000 completion  
**Sprint Health**: 🟡 REVISED (strategy improved, timeline extended)
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
