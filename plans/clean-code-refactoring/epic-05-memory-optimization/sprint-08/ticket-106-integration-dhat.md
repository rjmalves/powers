# [T-106] DHAT Verification & Performance Benchmarking

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 8: Model Rebuild Strategy](./00-sprint-overview.md)
> **Dependencies**: T-104, T-105, T-107, T-108, T-100-r, T-102-r
> **Blocks**: None (final ticket)
> **Priority**: 3 (Validation)
> **Status**: 🔵 Ready

## Files to Read Before Starting

- `docs/DHAT_SPRINT7_ANALYSIS.md` - Baseline metrics
- All Sprint 8 ticket implementations

---

## Context

### Background

After implementing all Sprint 8 features, we need to:
1. Verify no allocation regression via DHAT
2. Confirm RSS decreases after model rebuild
3. Benchmark rebuild overhead
4. Document final state

### Success Criteria

| Metric | Target |
|--------|--------|
| DHAT total bytes | ≤ Sprint 7 (45.43 GB) + 5% tolerance |
| RSS after rebuild | ≤ RSS before rebuild × 0.85 (15% reduction) |
| Rebuild overhead | < 2% of iteration time |
| Golden tests | All pass |

---

## Specification

### DHAT Profiling

```bash
# Build release
cargo build --release

# Run DHAT with full training
valgrind --tool=dhat --dhat-out-file=dhat-sprint8.out \
    ./target/release/powers run examples/05-large-scale-brazilian

# Parse and compare
python3 scripts/compare_dhat.py dhat_updated.out dhat-sprint8.out
```

### RSS Verification

Run training and capture RSS logs:

```bash
./target/release/powers run examples/05-large-scale-brazilian 2>&1 | tee training.log
grep -i "rss\|rebuild" training.log
```

Expected output:
```
Iteration 100: RSS = 6.50 GB
Model rebuild at iter 100: 6.50 GB -> 4.80 GB (freed 1.70 GB) in 2.3s
Iteration 200: RSS = 5.20 GB
Model rebuild at iter 200: 5.20 GB -> 4.10 GB (freed 1.10 GB) in 2.1s
...
```

### Performance Benchmark

```bash
# Baseline (no rebuild)
hyperfine --warmup 1 --runs 3 \
    './target/release/powers run examples/05-large-scale-brazilian --rebuild-interval 0'

# With rebuild
hyperfine --warmup 1 --runs 3 \
    './target/release/powers run examples/05-large-scale-brazilian --rebuild-interval 100'

# Calculate overhead
```

---

## Deliverables

### 1. DHAT Sprint 8 Analysis Document

Create `docs/DHAT_SPRINT8_ANALYSIS.md`:

```markdown
# DHAT Sprint 8 Analysis Report

> **Sprint**: Epic 5, Sprint 8 - Model Rebuild Strategy
> **Date**: [DATE]
> **Previous**: [DHAT_SPRINT7_ANALYSIS.md](./DHAT_SPRINT7_ANALYSIS.md)

## Executive Summary

[Summary of results]

## Metrics Comparison

| Metric | Sprint 7 | Sprint 8 | Change |
|--------|----------|----------|--------|
| Total bytes | 45.43 GB | X GB | X% |
| Total blocks | 43.3 M | X M | X% |
| Unique allocation points | 12,651 | X | X |

## Model Rebuild Impact

### RSS Measurements

| Iteration | Before Rebuild | After Rebuild | Freed |
|-----------|----------------|---------------|-------|
| 100 | X GB | X GB | X GB |
| 200 | X GB | X GB | X GB |
| 300 | X GB | X GB | X GB |

### Rebuild Timing

Average rebuild time: X seconds
Percentage of iteration time: X%

## Allocation Breakdown

[Category breakdown similar to Sprint 7]

## Conclusions

[Summary of findings]
```

### 2. Performance Report

Document in the DHAT analysis:
- Rebuild overhead percentage
- RSS sawtooth pattern verification
- Memory stability across iterations

### 3. Updated Epic Overview

Update `plans/.../00-epic-overview.md` with Sprint 8 results.

---

## Acceptance Criteria

- [ ] DHAT profiling completed
- [ ] No significant allocation regression (< 5%)
- [ ] RSS decrease after rebuild verified
- [ ] Rebuild overhead < 2%
- [ ] `DHAT_SPRINT8_ANALYSIS.md` created
- [ ] Epic overview updated
- [ ] Golden tests pass
- [ ] All 567+ tests pass

---

## Implementation Guide

### Suggested Approach

1. **Run DHAT** with all Sprint 8 changes

2. **Analyze results**:
   ```python
   import json
   
   with open('dhat-sprint8.out') as f:
       data = json.load(f)
   
   total_bytes = sum(p['tb'] for p in data['pps'])
   print(f"Total bytes: {total_bytes / 1e9:.2f} GB")
   ```

3. **Verify RSS behavior** from training logs

4. **Run benchmarks** with hyperfine

5. **Create documentation**

6. **Update epic overview**

### Key Files to Create/Modify

| File | Action |
|------|--------|
| `docs/DHAT_SPRINT8_ANALYSIS.md` | Create |
| `plans/.../00-epic-overview.md` | Update |
| `dhat-sprint8.out` | Generate (gitignored) |

---

## Testing Requirements

### Verification Tests

- [ ] All golden tests pass
- [ ] All 567+ unit tests pass
- [ ] Training completes successfully

### Manual Verification

- [ ] DHAT output parsed correctly
- [ ] RSS logs show expected pattern
- [ ] Benchmarks show acceptable overhead

---

## Documentation Requirements

- [ ] DHAT Sprint 8 Analysis document
- [ ] Performance section in analysis
- [ ] Epic overview updated with Sprint 8 status

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Verification work with known scope

---

## Definition of Done

- [ ] DHAT profiling complete
- [ ] RSS verification complete
- [ ] Benchmarks complete
- [ ] Documentation complete
- [ ] Epic overview updated
- [ ] Sprint 8 declared complete
- [ ] PR merged
