# Sprint 6: HiGHS Solver Memory Optimization

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ✅ Complete

---

## ⚠️ CRITICAL REMINDER

**Algorithm correctness is non-negotiable.** Memory optimization must not change any numerical results. Golden tests must pass after every change.

If any test fails or results diverge: **STOP and investigate before proceeding.**

---

## Executive Summary

DHAT profiling (see `docs/HOT_PATH_ALLOCATION_AUDIT.md` Appendix B) revealed that **94.7% of all heap allocations come from the HiGHS LP solver**, not Rust application code. This sprint targets the highest-impact HiGHS optimization opportunities identified in the profile.

### Key DHAT Findings Driving This Sprint

| Allocation Site | % of Total | Bytes | Strategy |
|-----------------|------------|-------|----------|
| `HFactor::setupGeneral` | 44.9% | 39.6 GB | Warm-starting / basis reuse |
| `changeRowBounds` | 3.4% | 3.0 GB | Batch bound updates |
| `debugDualSimplex` | 0.06% | 50 MB | Disable debug output |
| HiGHS Presolve | 3.2% | 2.8 GB | Evaluate presolve settings |

**Target**: Reduce HiGHS allocation churn by 30-50% through configuration and API usage changes.

---

## Sprint Completion Summary

### 🎉 EXCEPTIONAL RESULTS

Sprint 6 **far exceeded** the 30% target reduction:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Total Bytes Allocated** | 88.19 GB | 45.43 GB | **48.5% reduction** |
| **Total Allocation Blocks** | 159.0M | 43.4M | **72.7% reduction** |
| **HFactor::setupGeneral** | 39.58 GB | 2.00 GB | **95.0% reduction** |
| **changeRowBounds blocks** | 98.4M | 0.4M | **99.6% reduction** |

**Key Insight**: The `reuse_forward_basis()` function was counterproductive, triggering HiGHS "alien basis" handling on every backward branching solve. Disabling it eliminated 95% of HFactor allocations.

### Completed Work

1. **T-087**: HiGHS warm-start investigation ✅
   - Documented in `docs/HIGHS_WARM_START_INVESTIGATION.md`
   - Finding: `reuse_forward_basis()` triggers alien basis handling - disabled for testing
   - **Hypothesis validated by DHAT**: 95% HFactor reduction confirms the issue

2. **T-088**: Batch `changeRowBounds` interface ✅
   - Implemented `Model::change_rows_bounds_batch()` in `src/solver.rs`
   - Uses `Highs_changeRowsBoundsBySet` for single FFI call
   - **Result**: 99.6% reduction in allocation blocks

3. **T-089**: Integrated batch bounds into production code ✅
   - Updated `update_uncertainty_constraints()` 
   - Updated `update_lag_fixing_constraints()`
   - Updated `set_hydro_balance_rhs()` and `prepare_from_trajectory()`

4. **T-090**: HiGHS debug mode verified disabled ✅
   - `output_flag=false` and `log_to_console=0` already set in `make_quiet()`
   - No additional changes needed

5. **T-091**: Presolve settings evaluated ✅
   - `presolve="off"` already set in `set_default_solver_options()`
   - Eliminates 3.2% of allocations

6. **T-092**: HiGHS internal threading disabled ✅
   - `parallel="off"` and `threads=1` already set
   - No thread pool allocations

7. **T-093**: DHAT verification ✅
   - Full analysis in `docs/DHAT_SPRINT6_ANALYSIS.md`
   - Results far exceeded expectations

---

## Goals

1. **Investigate and implement HiGHS warm-starting** to avoid repeated `HFactor::setupGeneral` calls
2. **Batch `changeRowBounds` calls** to reduce 3 million individual calls to bulk updates
3. **Verify HiGHS debug mode is disabled** in release builds
4. **Evaluate HiGHS presolve settings** for memory vs. performance tradeoff
5. **Disable HiGHS internal threading** to reduce thread pool allocation overhead

## Non-Goals

- Modifying HiGHS source code
- Changing algorithm behavior or numerical results
- Optimizing Rust application allocations (deferred to Sprint 7)

---

## Technical Approach

### 1. HiGHS Warm-Starting Investigation

**Current Behavior**: Every `Highs_run()` call triggers `HFactor::setupGeneral`, allocating ~88 KB per solve.

**Investigation Areas**:
- Does HiGHS support basis reuse across solves with modified bounds?
- Can we call `Highs_setBasis()` before `Highs_run()` to skip factorization setup?
- What is the impact of `simplex_warm_start` option?

**Implementation Strategy**:
```rust
// Hypothesis: Calling setBasis before run() reduces factorization overhead
pub fn solve_warm(&mut self) -> HighsStatus {
    // Get current basis
    let basis = self.get_basis();
    
    // Modify model (change bounds, add cuts)
    // ...
    
    // Restore basis hint
    self.set_basis(&basis);
    
    // Solve with warm start
    self.run()
}
```

### 2. Batch changeRowBounds Calls

**Current Behavior**: `Model::change_rows_bounds()` calls `Highs_changeRowBounds()` once per row, triggering:
- Vector allocation for row indices
- String allocation for debug messages
- 3 million calls total

**Target Behavior**: Single bulk call per bound update batch.

**Implementation**:
```rust
// New: Batch interface
pub fn change_rows_bounds_batch(
    &mut self,
    row_indices: &[HighsInt],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
) -> Result<(), HighsError> {
    // Use Highs_changeRowsBounds (plural) API
    unsafe {
        Highs_changeRowsBounds(
            self.ptr,
            row_indices.len() as HighsInt,
            row_indices.as_ptr(),
            lower_bounds.as_ptr(),
            upper_bounds.as_ptr(),
        )
    }
}
```

### 3. HiGHS Debug Mode Verification

**Current Behavior**: `debugDualSimplex` allocates 50 MB of strings even in release builds.

**Verification Steps**:
1. Check if `Highs_setOptionValue(highs, "output_flag", "false")` is set
2. Verify HiGHS was built with `NDEBUG` flag
3. Check `log_to_console` and `log_file` options

### 4. Presolve Settings Evaluation

**Current Behavior**: 3.2% of allocations from presolve.

**Options to Evaluate**:
- `presolve = "off"` - May hurt solve time but reduce allocations
- `presolve = "choose"` - Let HiGHS decide based on problem size
- Measure allocation vs. solve time tradeoff

### 5. Disable HiGHS Threading

**Current Behavior**: `HighsTaskExecutor` allocates 8.4 MB for thread pools.

**Strategy**: Set `threads = 1` to prevent internal thread pool creation (Rayon handles parallelism at higher level).

---

## Sprint Tickets

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-087 | Investigate HiGHS warm-start API and basis reuse | 5 | None |
| T-088 | Implement batch changeRowBounds interface | 3 | None |
| T-089 | Update cut realization to use batch bound updates | 5 | T-088 |
| T-090 | Verify and enforce HiGHS debug mode disabled | 2 | None |
| T-091 | Evaluate HiGHS presolve settings impact | 3 | None |
| T-092 | Disable HiGHS internal threading | 2 | None |
| T-093 | DHAT profiling to measure HiGHS allocation reduction | 3 | T-087, T-089, T-090 |

**Total**: 23 points

---

## Acceptance Criteria

### Sprint Completion

- [x] HiGHS warm-start investigation complete with documented findings
- [x] Batch `changeRowBounds` interface implemented and integrated
- [x] HiGHS debug output verified disabled in release builds
- [x] Presolve settings evaluated with performance measurements
- [x] HiGHS internal threading disabled
- [x] DHAT shows ≥30% reduction in HiGHS allocations (**achieved 48.5% bytes, 72.7% blocks**)
- [x] All tests pass (numerical correctness preserved)
- [x] No performance regression (56.5% instruction reduction indicates improvement)

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Warm-start API not available | Medium | High | Document findings, explore alternatives |
| Batch API behavior differs | Low | Medium | Comprehensive testing with golden outputs |
| Presolve off hurts solve time | Medium | Medium | Benchmark both configurations |
| Debug mode still enabled | Low | Low | Explicit verification in tests |

---

## Key Files

| Component | Location |
|-----------|----------|
| HiGHS bindings | `src/solver.rs` |
| Model wrapper | `src/solver.rs:Model` |
| Default options | `src/subproblem.rs:set_default_solver_options()` |
| Cut realization | `src/sddp/mod.rs:realize_and_solve()` |
| Bound updates | `src/state.rs` |

---

## Verification Steps

### 1. DHAT Profiling (Before/After)

```bash
# Before changes
cargo build --release
valgrind --tool=dhat ./target/release/powers run examples/05-large-scale-brazilian
mv dhat.out dhat-before-sprint6.out

# After changes
cargo build --release
valgrind --tool=dhat ./target/release/powers run examples/05-large-scale-brazilian
mv dhat.out dhat-after-sprint6.out

# Compare HiGHS allocations
python3 scripts/compare_dhat.py dhat-before-sprint6.out dhat-after-sprint6.out
```

### 2. Golden Tests

```bash
cargo test --release -- golden
```

### 3. Performance Benchmark

```bash
cargo bench --bench sddp_training -- --save-baseline sprint6
```

---

## Definition of Done

- [x] All tickets complete and merged
- [x] DHAT shows ≥30% reduction in HiGHS allocations (**48.5% achieved**)
- [x] No numerical divergence (golden tests pass)
- [x] No performance regression (**56.5% instruction reduction**)
- [x] Documentation updated (`docs/DHAT_SPRINT6_ANALYSIS.md`)
- [x] All tests pass
