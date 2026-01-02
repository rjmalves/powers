# DHAT Sprint 6 Analysis Report

> **Sprint**: Epic 5, Sprint 6 - HiGHS Solver Memory Optimization
> **Date**: 2025-12-30
> **Previous Analysis**: [HOT_PATH_ALLOCATION_AUDIT.md](./HOT_PATH_ALLOCATION_AUDIT.md)
> **Investigation**: [HIGHS_WARM_START_INVESTIGATION.md](./HIGHS_WARM_START_INVESTIGATION.md)

---

## Executive Summary

Sprint 6 achieved **exceptional results**, far exceeding the 30% target reduction in HiGHS allocations:

| Metric | Before Sprint 6 | After Sprint 6 | Improvement |
|--------|-----------------|----------------|-------------|
| **Total Instructions** | 480.4 billion | 208.8 billion | **56.5% reduction** |
| **Total Bytes Allocated** | 88.19 GB | 45.43 GB | **48.5% reduction** |
| **Total Allocation Blocks** | 159.0 million | 43.4 million | **72.7% reduction** |
| **Allocation Throughput (tg)** | 460.9 GB | 190.6 GB | **58.6% reduction** |

The primary driver of these improvements was **disabling `reuse_forward_basis()`**, which eliminated the "alien basis" handling in HiGHS that was triggering full factorization rebuilds on every backward branching solve.

---

## Detailed Category Analysis

### HiGHS Allocation Breakdown

| Component | Before (GB) | After (GB) | Reduction | Before (blocks) | After (blocks) | Block Reduction |
|-----------|-------------|------------|-----------|-----------------|----------------|-----------------|
| **HFactor::setupGeneral** | 39.58 | 2.00 | **95.0%** | 2.4M | 0.1M | **95.8%** |
| HEkk::computeDual | 20.02 | 16.70 | 16.6% | 3.8M | 3.1M | 18.2% |
| HEkkDual | 18.32 | 17.96 | 2.0% | 31.3M | 19.1M | 39.1% |
| **changeRowBounds** | 0.98 | 0.17 | **82.6%** | 98.4M | 0.4M | **99.6%** |
| HSimplexNla | 3.07 | 3.05 | 0.5% | 0.8M | 0.4M | 51.4% |
| Rust/Powers | 6.12 | 5.44 | 11.1% | 22.2M | 20.2M | 8.9% |
| Parquet | 0.01 | 0.01 | ~0% | ~0 | ~0 | ~0% |

### Key Findings

#### 1. HFactor::setupGeneral: 95% Reduction ✅

**Hypothesis Confirmed**: The `reuse_forward_basis()` function was triggering HiGHS "alien basis" handling on every backward branching solve, forcing full factorization rebuilds.

**Before**: 39.58 GB, 2.4M blocks  
**After**: 2.00 GB, 0.1M blocks  
**Reduction**: 37.58 GB (95%)

The investigation in `docs/HIGHS_WARM_START_INVESTIGATION.md` correctly identified that:
- When `setBasis()` receives a basis with mismatched row counts (due to cuts being added), HiGHS marks it as "alien"
- Alien basis handling triggers `formSimplexLpBasisAndFactor()`, which rebuilds the full factorization
- Disabling `reuse_forward_basis()` allows HiGHS to use its default logical basis, avoiding the rebuild

#### 2. changeRowBounds: 99.6% Block Reduction ✅

**Batch API Working**: The implementation of `Model::change_rows_bounds_batch()` using `Highs_changeRowsBoundsBySet` dramatically reduced FFI call overhead.

**Before**: 98.4M allocation blocks (3 million calls × ~30 allocations each)  
**After**: 0.4M allocation blocks (~60 batch calls per stage)  
**Reduction**: 99.6% fewer allocation operations

While byte reduction was 82.6%, the block count reduction of 99.6% indicates dramatically improved allocation efficiency.

#### 3. HEkkDual Block Reduction: 39.1%

The 39.1% reduction in HEkkDual blocks (31.3M → 19.1M) is a secondary benefit of the reduced solve complexity when not using alien bases.

#### 4. Rust/Powers: 11.1% Reduction

Even without targeting Rust allocations in Sprint 6, we achieved an 11.1% reduction (6.12 GB → 5.44 GB), likely due to:
- Fewer solver invocations overall
- Reduced iteration counts from more efficient solves

---

## Validation of Investigation Hypotheses

### HIGHS_WARM_START_INVESTIGATION.md Predictions

| Prediction | Result | Status |
|------------|--------|--------|
| `reuse_forward_basis` triggers alien basis handling | Confirmed - 95% HFactor reduction when disabled | ✅ Validated |
| Batch bounds API reduces FFI overhead | Confirmed - 99.6% block reduction | ✅ Validated |
| HiGHS debug mode contributes 50 MB | Unchanged - suggests already disabled | ✅ Already optimized |
| HFactor allocations are inherent to HiGHS | Partially incorrect - alien basis was the cause | ⚠️ Revised |

**Major Insight**: The investigation initially concluded that "HFactor allocations are inherent to HiGHS and cannot be eliminated via API." This was **incorrect** - the allocations were caused by our misuse of the basis API, not HiGHS internals.

---

## Performance Impact

### Instruction Count Reduction

The 56.5% reduction in total instructions (480.4B → 208.8B) indicates:
- Significantly fewer solver iterations
- Reduced factorization overhead
- More efficient solve paths

This should translate to meaningful wall-clock time improvements (to be validated with benchmarks).

### Memory Pressure Reduction

The 58.6% reduction in allocation throughput (460.9 GB → 190.6 GB) means:
- Lower allocator pressure
- Better cache utilization
- Reduced memory fragmentation

---

## Comparison to Sprint Goals

| Sprint 6 Goal | Target | Achieved | Status |
|---------------|--------|----------|--------|
| Reduce HiGHS allocations | ≥30% | **48.5% bytes, 72.7% blocks** | ✅ Exceeded |
| Batch changeRowBounds | 90%+ reduction | **99.6% block reduction** | ✅ Exceeded |
| HiGHS debug disabled | Verify | Already disabled | ✅ Confirmed |
| Presolve evaluated | Document findings | presolve="off" confirmed | ✅ Complete |
| HiGHS threading disabled | Verify | threads=1 confirmed | ✅ Confirmed |

---

## Recommendations for Sprint 7

### Updated Priorities

Given Sprint 6's success, Sprint 7 can focus on Rust allocations with confidence:

1. **Keep `reuse_forward_basis()` Disabled**
   - The 95% HFactor reduction validates this approach
   - Consider removing the commented-out code entirely

2. **Rust Allocations Still Important**
   - 5.44 GB remaining from Rust/Powers code
   - `uniform_prob_by_count()` - preallocated buffers
   - `sample_scenario()` - thread-local buffers
   - HashSet allocations in FCF

3. **Potential Further HiGHS Optimizations**
   - HEkkDual still at 17.96 GB - investigate if further reduction possible
   - HSimplexNla at 3.05 GB - likely inherent to algorithm

### Sprint 7 Ticket Updates

Consider adding:
- **T-094a**: Remove `reuse_forward_basis()` code entirely (instead of keeping commented)
- **T-094b**: Document when basis reuse IS appropriate (if ever)

---

## Appendix A: Raw DHAT Comparison

```
=== DHAT Comparison: Sprint 06 Progress ===

Total Instructions:
  Old: 480,388,282,445
  New: 208,760,265,690
  Reduction: 56.5%

Total Bytes Allocated (tg):
  Old: 460,906,024,778 (460.91 GB)
  New: 190,627,406,401 (190.63 GB)
  Reduction: 58.6%

Unique Allocation Points (pps):
  Old: 12,026
  New: 12,338
  Change: +312

Total Bytes (sum of tb):
  Old: 88,189,970,461 (88.19 GB)
  New: 45,428,592,212 (45.43 GB)
  Reduction: 48.5%

Total Blocks (sum of tbk):
  Old: 159,007,634 (159.0M)
  New: 43,374,152 (43.4M)
  Reduction: 72.7%
```

---

## Appendix B: Files Referenced

- `dhat.out` - Before Sprint 6 (baseline)
- `dhat_new.out` - After Sprint 6
- `docs/HOT_PATH_ALLOCATION_AUDIT.md` - Original allocation audit
- `docs/HIGHS_WARM_START_INVESTIGATION.md` - Warm-start investigation
- `src/solver.rs` - Batch bounds implementation
- `src/sddp/mod.rs` - `reuse_forward_basis()` (disabled)

---

## Conclusion

Sprint 6 was a major success, achieving nearly **50% reduction in total allocations** and **73% reduction in allocation blocks**. The key insight was that our `reuse_forward_basis()` function was counterproductive, triggering expensive alien basis handling in HiGHS.

The investigation hypothesis in `HIGHS_WARM_START_INVESTIGATION.md` correctly identified the problem, and disabling the function validated the theory. Combined with the batch bounds API, Sprint 6 has fundamentally improved the memory behavior of SDDP training.
