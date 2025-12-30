# [T-093] DHAT Profiling to Measure HiGHS Allocation Reduction

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 6: HiGHS Solver Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: T-087, T-089, T-090, T-092
> **Blocks**: None

---

## Context

### Background

After implementing all HiGHS optimizations in Sprint 6 (warm-start, batch bounds, debug disabled, threading), we need to verify the allocation reduction with DHAT profiling.

### Baseline (Before Sprint 6)

From `docs/HOT_PATH_ALLOCATION_AUDIT.md`:

| Component | Bytes Allocated | Percentage |
|-----------|-----------------|------------|
| HiGHS Solver (HEkk/HFactor) | 83.5 GB | 94.7% |
| HiGHS Presolve | 2.8 GB | 3.2% |
| Rust Application | 1.8 GB | 2.0% |
| Total | 88.19 GB | 100% |

### Target

Reduce HiGHS allocations by **≥30%** (targeting ~60 GB or less).

### Relation to Epic

Final verification that Sprint 6 optimizations achieved their goals.

## Specification

### Tasks

1. **Run DHAT on example 05** with all Sprint 6 changes
2. **Parse and analyze results** using same methodology as baseline
3. **Compare to baseline** (dhat.out already exists)
4. **Document findings** with before/after tables

### Expected Outputs

- New DHAT output file: `dhat-sprint6.out`
- Analysis report in `docs/DHAT_SPRINT6_ANALYSIS.md`
- Updated `docs/HOT_PATH_ALLOCATION_AUDIT.md` with Sprint 6 results

### Behavior

- Profile should run to completion
- Results should be reproducible

## Acceptance Criteria

- [x] DHAT profiling completed on example 05
- [x] Results parsed and categorized by component
- [x] Before/after comparison table created
- [x] ≥30% reduction in HiGHS allocations achieved (**48.5% bytes, 72.7% blocks**)
- [x] If target not met: document why and next steps

**Status**: ✅ Complete

**Results Summary**:
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total Bytes | 88.19 GB | 45.43 GB | **48.5% reduction** |
| Total Blocks | 159.0M | 43.4M | **72.7% reduction** |
| HFactor::setupGeneral | 39.58 GB | 2.00 GB | **95.0% reduction** |
| changeRowBounds blocks | 98.4M | 0.4M | **99.6% reduction** |

**Key Finding**: Disabling `reuse_forward_basis()` eliminated 95% of HFactor allocations by avoiding HiGHS "alien basis" handling. This validated the hypothesis from `docs/HIGHS_WARM_START_INVESTIGATION.md`.

Full analysis: `docs/DHAT_SPRINT6_ANALYSIS.md`

## Implementation Guide

### Suggested Approach

1. **Build release binary**:
   ```bash
   cargo build --release
   ```

2. **Run DHAT profiling**:
   ```bash
   valgrind --tool=dhat \
       --dhat-out-file=dhat-sprint6.out \
       ./target/release/powers run examples/05-large-scale-brazilian
   ```

3. **Parse results with Python script**:
   ```python
   # scripts/analyze_dhat.py (create or reuse)
   import json
   
   with open('dhat-sprint6.out') as f:
       data = json.load(f)
   
   # Categorize and compare to baseline
   ```

4. **Create comparison report**:
   ```markdown
   # DHAT Sprint 6 Analysis
   
   ## Before/After Comparison
   
   | Component | Before (GB) | After (GB) | Reduction |
   |-----------|-------------|------------|-----------|
   | HFactor::setupGeneral | 39.6 | X | Y% |
   | changeRowBounds | 3.0 | X | Y% |
   | debugDualSimplex | 0.05 | X | Y% |
   | HighsTaskExecutor | 0.008 | X | Y% |
   | Total HiGHS | 83.5 | X | Y% |
   ```

5. **Update documentation**:
   - Add Sprint 6 section to `docs/HOT_PATH_ALLOCATION_AUDIT.md`
   - Create `docs/DHAT_SPRINT6_ANALYSIS.md` with detailed findings

### Key Analysis Points

1. **HFactor::setupGeneral** - Should reduce if warm-start helps
2. **changeRowBounds** - Should drop by 90%+ with batching
3. **debugDualSimplex** - Should be zero with debug disabled
4. **HighsTaskExecutor** - Should be zero with threads=1

### Pitfalls to Avoid

- ⚠️ DHAT runs ~20x slower; ensure example completes in reasonable time
- ⚠️ Use same example/seed as baseline for fair comparison
- ⚠️ Valgrind may not work on all platforms

## Testing Requirements

### Profiling

- [ ] DHAT profile completes without errors
- [ ] Output file is valid JSON
- [ ] Analysis script processes without errors

### Validation

- [ ] Results are reproducible (run twice)
- [ ] Categories match baseline methodology

## Documentation Requirements

- [ ] Create `docs/DHAT_SPRINT6_ANALYSIS.md`
- [x] Update `docs/HOT_PATH_ALLOCATION_AUDIT.md` with new section
- [x] Archive dhat files for future reference

## Dependencies

- **Blocked By**: T-087, T-089, T-090, T-092 (all Sprint 6 changes)
- **Blocks**: None
- **Related**: Sprint 7 planning depends on these results

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Analysis work with clear methodology

## Definition of Done

- [x] DHAT profiling complete
- [x] Results analyzed and documented
- [x] Comparison report created
- [x] ≥30% reduction verified (**48.5% bytes, 72.7% blocks**)
- [x] Documentation updated (`docs/DHAT_SPRINT6_ANALYSIS.md`)
- [x] PR merged
