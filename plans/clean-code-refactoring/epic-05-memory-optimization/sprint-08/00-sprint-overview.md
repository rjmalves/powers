# Sprint 8: Model Rebuild Strategy & Deferred Optimizations

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint 7 Results**: [DHAT_SPRINT7_ANALYSIS.md](../../../../docs/DHAT_SPRINT7_ANALYSIS.md)
> **RSS Investigation**: [HIGHS_RSS_MEMORY_INVESTIGATION.md](../../../../docs/HIGHS_RSS_MEMORY_INVESTIGATION.md)
> **Duration**: 2 weeks
> **Status**: 🔵 Ready for Implementation

---

## ⚠️ CRITICAL REMINDER

**Algorithm correctness is non-negotiable.** Memory optimization must not change any numerical results. Golden tests must pass after every change.

If any test fails or results diverge: **STOP and investigate before proceeding.**

---

## Executive Summary

Sprint 8 implements the **Model Rebuild Strategy** to address RSS memory growth, plus completes two deferred optimization tickets from Sprint 7.

### Problem Statement

HiGHS internal buffers grow throughout SDDP training but never shrink. This causes:
- RSS memory increases ~3x from initial to peak
- Memory pressure on long-running training jobs
- Unpredictable resource requirements

### Solution

**Periodic Model Rebuild**: Every N iterations, destroy and recreate HiGHS models to force memory reclaim. Active cuts are preserved by extracting and re-adding them to the fresh model.

### Expected Outcomes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Peak RSS (300 iter) | ~8 GB | ~5.5 GB | ~30% reduction |
| Memory stability | Growing | Sawtooth (stable range) | Predictable |
| Training performance | Baseline | <2% overhead | Negligible |

---

## Sprint Goals

### Priority 1: Model Rebuild Strategy (Critical)

| Goal | Description |
|------|-------------|
| Implement rebuild infrastructure | `Subproblem::rebuild_model()` method |
| Integrate with training loop | Configurable rebuild interval |
| Add RSS monitoring | Visibility into memory behavior |
| Preserve correctness | Golden tests pass after rebuild |

### Priority 2: Deferred Optimizations (High)

| Goal | Description |
|------|-------------|
| Scenario indices buffer | Zero-allocation scenario sampling |
| CutIdSet type (Phase 1) | Efficient bit-vector set for cut IDs |

### Priority 3: Validation (Medium)

| Goal | Description |
|------|-------------|
| DHAT verification | No allocation regression |
| Performance benchmark | <2% overhead from rebuild |
| Documentation | Update memory behavior docs |

---

## Technical Architecture

### Model Rebuild Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                            │
│                                                             │
│  for iteration in 1..=max_iterations:                       │
│      forward_pass()                                         │
│      backward_pass()                                        │
│      update_fcf()                                           │
│                                                             │
│      if iteration % REBUILD_INTERVAL == 0:  ◄──── NEW      │
│          log_rss_before()                                   │
│          for handler in handlers:                           │
│              handler.rebuild_all_models(cut_pools)          │
│          log_rss_after()                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Rebuild Process Detail

```
┌─────────────────────────────────────────────────────────────┐
│              Subproblem::rebuild_model()                    │
│                                                             │
│  1. Extract active cut data from FCF                        │
│     - coefficients, RHS, iteration, forward_pass_idx        │
│                                                             │
│  2. Drop HiGHS model                                        │
│     - self.model = None (triggers Highs_destroy)            │
│     - Forces deallocation of HiGHS internal buffers         │
│                                                             │
│  3. Rebuild fresh model                                     │
│     - Create new Problem                                    │
│     - Add variables, constraints (same as constructor)      │
│     - set_default_solver_options()                          │
│                                                             │
│  4. Preallocate cut slots                                   │
│     - preallocate_cut_constraints(remaining_cuts)           │
│                                                             │
│  5. Restore active cuts                                     │
│     - For each active cut: update coefficients + bounds     │
│                                                             │
│  6. Warmup solver                                           │
│     - warmup_solver() to pre-allocate new HiGHS internals   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Rebuild at handler level**: Each `SddpTrainHandler` manages its own models, so rebuild is per-handler.

2. **Preserve FCF state**: Cut pools in `FutureCostFunction` are not affected by model rebuild - only the HiGHS model state.

3. **Configurable interval**: Default 100 iterations; tunable based on memory constraints.

4. **Parallel rebuild**: All handlers can rebuild in parallel (independent models).

---

## Sprint Tickets

### Priority 1: Model Rebuild Strategy

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-104 | Implement Subproblem::rebuild_model() | 5 | None |
| T-105 | Add RSS monitoring utilities | 2 | None |
| T-107 | Integrate rebuild into training loop | 3 | T-104, T-105 |
| T-108 | Add rebuild configuration options | 2 | T-107 |

### Priority 2: Deferred Optimizations

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-100-r | Scenario sampling indices buffer | 3 | None |
| T-102-r | CutIdSet type implementation | 3 | None |

### Priority 3: Validation

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-106 | DHAT verification & benchmarking | 3 | All above |

**Total**: 21 points

---

## Acceptance Criteria

### Model Rebuild

- [ ] `rebuild_model()` implemented with complete cut restoration
- [ ] RSS decreases after rebuild (verified with monitoring)
- [ ] Training produces identical results with/without rebuild
- [ ] Rebuild overhead < 2% of iteration time
- [ ] Golden tests pass

### Deferred Optimizations

- [ ] `sample_scenario_indices_into()` eliminates Vec allocation
- [ ] `CutIdSet` type implemented with comprehensive tests
- [ ] DHAT shows no regression

### Validation

- [ ] DHAT comparison documented
- [ ] Performance benchmark shows acceptable overhead
- [ ] All 567+ tests pass

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cut restoration breaks correctness | Medium | Critical | Golden tests, detailed validation |
| Rebuild overhead too high | Low | Medium | Tune interval, benchmark |
| RSS doesn't decrease as expected | Low | Medium | Analyze remaining allocations |
| Parallel rebuild causes issues | Low | Low | Already independent handlers |

---

## Key Files

| Component | Location | Purpose |
|-----------|----------|---------|
| Subproblem | `src/subproblem.rs` | Model rebuild logic |
| SddpTrainHandler | `src/sddp/mod.rs` | Handler-level rebuild |
| SddpAlgorithm | `src/sddp/mod.rs` | Training loop integration |
| RSS utilities | `src/memory/rss.rs` | RSS monitoring (NEW) |
| ScenarioTree | `src/scenario.rs` | Indices sampling |
| CutIdSet | `src/memory/cut_id_set.rs` | Bit-vector set (NEW) |

---

## Definition of Done

- [ ] All tickets complete and merged
- [ ] RSS monitoring shows memory reclaim after rebuild
- [ ] DHAT shows no regression
- [ ] Performance overhead < 2%
- [ ] No numerical divergence (golden tests pass)
- [ ] All 567+ tests pass
- [ ] Documentation updated

---

## Parallel Work Streams

Tickets can be worked on in parallel:

**Stream A** (Model Rebuild):
```
T-104 ──► T-107 ──► T-108
            │
            ▼
T-105 ──────┘
```

**Stream B** (Deferred Optimizations):
```
T-100-r (independent)
T-102-r (independent)
```

**Convergence**:
```
All streams ──► T-106 (validation)
```
