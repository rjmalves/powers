# PAR-V2-023: Performance Benchmarking

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 5 (Runtime & Optimization)  
**Story Points**: 2  
**Priority**: Medium  
**Status**: 🔵 Not Started

---

## Context

Benchmark PAR implementation to quantify performance impact:
- Training time vs non-PAR
- Memory usage
- LP solve time
- Scalability with AR order and number of hydros

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 4, Section 4.4
- Existing benchmarks: `benches/`

---

## Acceptance Criteria

- [ ] Benchmark suite for PAR models
- [ ] Comparison with non-PAR baseline
- [ ] Memory profiling
- [ ] Performance acceptable (< 50% slowdown)
- [ ] Results documented

---

## Tasks

- [ ] Create benchmark in `benches/par_benchmarks.rs`
- [ ] Measure training time for various problem sizes
- [ ] Measure LP solve time
- [ ] Measure memory usage
- [ ] Document results in plan

---

## Estimated Effort

**2 story points** (1 day)
