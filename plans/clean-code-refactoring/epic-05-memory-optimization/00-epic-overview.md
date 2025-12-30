# Epic 5: Memory Optimization

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 2 weeks (1 sprint)
> **Status**: ⚠️ Infrastructure Complete - Integration Pending

---

## ⚠️ IMPLEMENTATION GAP IDENTIFIED

See [EPIC_05_IMPLEMENTATION_ANALYSIS.md](../../../docs/EPIC_05_IMPLEMENTATION_ANALYSIS.md) for details.

**Summary**: The zero-allocation APIs were implemented and tested, but the training loop (`backward_pass.rs`) was NOT updated to use them. The production code still uses the allocating `CutData` path.

**Action Required**: Complete T-055 to wire the new path into the training loop.

---

## ⚠️ CRITICAL REMINDER

This epic eliminates remaining dynamic allocations in the hot paths.

**Algorithm correctness is non-negotiable.** Memory optimization must not change any numerical results. Golden tests must pass after every change.

---

## Summary

This epic completes the zero-allocation hot path goal by eliminating the remaining Vec allocations identified in `REMAINING_ALLOCATIONS_ANALYSIS.md`. The infrastructure (pools, buffer methods, preallocated slots) was implemented in previous epics. This epic focuses on using that infrastructure to achieve true zero-allocation cut computation.

**Current State**: The `compute_cut_data()` path uses thread-local buffers for evaluation but still allocates via `CutData::from_refs()` which calls `.to_vec()` twice per cut.

**Target State**: Cut computation writes directly to preallocated pool slots without intermediate allocations.

---

## What's Already Implemented

The following was implemented in previous epics:

| Component | Location | Purpose |
|-----------|----------|---------|
| `BendersCutPool::preallocate()` | `src/cut.rs` | Preallocated cuts with slots |
| `VisitedStatePool` | `src/state.rs` | Preallocated states |
| `CutComputationBuffers` | `src/memory/buffers.rs` | Thread-local cut computation buffers |
| `compute_cut_data()` | `src/subproblem.rs` | Avoids `Box<dyn State>` allocation |
| `get_solution_into()` | `src/solver.rs` | Zero-alloc solution extraction |
| `get_basis_into()` | `src/solver.rs` | Zero-alloc basis extraction |
| `preallocate_cut_constraints()` | `src/subproblem.rs` | Preallocated HiGHS constraints |

---

## New APIs Implemented in This Epic

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| `update_cut_and_state_slots()` | `src/cut.rs` | Direct copy to slots | ✅ Implemented |
| `compute_cut_into_slot()` | `src/state.rs` | Zero-alloc cut computation | ✅ Implemented |
| `finalize_cut_at_slot()` | `src/fcf.rs` | Single slot finalization | ✅ Implemented |
| `finalize_cuts_batch()` | `src/fcf.rs` | Batch slot finalization | ✅ Implemented |
| `compute_cut_into_slot_for_backward_step()` | `src/sddp/mod.rs` | Full backward step | ✅ Implemented |
| `compute_cuts_into_slots()` | `src/algorithm/coordinator.rs` | Coordinator method | ✅ Implemented |
| **Training loop integration** | `src/algorithm/backward_pass.rs` | Wire into production | ❌ NOT DONE |

---

## Scope

### Included

1. **Direct Pool Update API** ✅
   - Add method to update preallocated cut pool directly from thread-local buffers
   - Eliminate `CutData` intermediate struct allocation

2. **Eliminate CutData Allocations** ⚠️ Partial
   - Replace `CutData::from_refs()` with direct copy to preallocated slots
   - APIs exist but training loop not updated

3. **Verify Zero Allocations** ❌ Blocked
   - Cannot verify until training loop uses new path

### Excluded

- SoA conversion (deferred - insufficient benefit vs complexity)
- Algorithm changes
- New dependencies

---

## Dependencies

- **Requires**:
  - Epic 4 complete ✅ (FCF simplified, pools ready)
- **Enables**:
  - Epic 7: Performance Validation (final verification)

---

## Acceptance Criteria

- [x] Zero-allocation APIs implemented
- [x] APIs tested for correctness
- [x] `CutData::from_refs()` marked deprecated
- [ ] **Training loop uses zero-allocation path** ❌
- [ ] DHAT profiling confirms zero allocations ❌
- [ ] Golden tests pass with new path ❌
- [ ] Performance improvement measured ❌

---

## Sprint 1: Zero-Allocation Cut Computation

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-050 | Add direct cut slot update method to BendersCutPool | 3 | ✅ |
| T-051 | Add compute_cut_into_slot to State trait | 5 | ✅ |
| T-052 | Update backward pass to use direct slot updates | 5 | ⚠️ Partial |
| T-053 | Remove CutData from hot path | 2 | ⚠️ Partial |
| T-054 | Verify zero allocations with profiling | 3 | ❌ Blocked |

**Total Points**: 18

---

## Sprint 2: Training Loop Integration (NEW - Required)

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-055 | Wire zero-allocation path into training loop | 5 | ⬜ |
| T-056 | Verify zero allocations with DHAT | 3 | ⬜ |

**Total Points**: 8

---

## Estimated Effort

- **Duration**: 1.5 sprints (3 weeks total)
- **Story Points**: 26 (18 infrastructure + 8 integration)
- **Risk Level**: Medium (touching hot path, must preserve correctness)

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical divergence | Low | **CRITICAL** | Golden tests after every change |
| Arc<BendersCut> mutation issues | Medium | Medium | May need unsafe or restructure |
| Performance regression | Low | Medium | Benchmark every change |
| Parallel execution complexity | Medium | Medium | Sequential fallback available |

---

## Definition of Done

- [x] All infrastructure APIs complete (T-050, T-051)
- [ ] Training loop updated to use zero-allocation path (T-055)
- [ ] Zero allocations in cut computation hot path verified
- [ ] `CutData::from_refs()` not called in production
- [x] All tests pass (549)
- [ ] Golden tests pass with new path
- [ ] Performance target met (+5% or no regression)
- [ ] Profiling data documented
