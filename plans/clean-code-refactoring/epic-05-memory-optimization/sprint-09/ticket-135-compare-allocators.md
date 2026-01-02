# [T-135] Compare Allocator Results and Select Winner

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 9: RSS Stabilization](./00-sprint-overview.md)
> **Status**: ✅ Complete (glibc WINS)
> **Dependencies**: T-131, T-134 (T-133 optional)
> **Blocks**: T-136

---

## Context

### Background

After testing allocator options (mimalloc, malloc_trim, and optionally jemalloc), we need to analyze the results and select the best option for making the default.

### Dependencies (Revised)

**Required inputs:**
- T-131: mimalloc RSS analysis
- T-134: malloc_trim RSS analysis
- Sprint 8: glibc baseline (1,045 MB after 20 iterations)

**Optional input:**
- T-133: jemalloc RSS analysis (may be skipped due to compilation issues)

### jemalloc Status

⚠️ jemalloc has caused SIGBUS crashes during compilation. If T-133 was skipped:
- Proceed with comparison of mimalloc vs malloc_trim vs glibc
- Document jemalloc as "not tested due to compilation instability"
- Recommend mimalloc as default (most likely winner anyway)

## Specification

### Comparison Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| RSS Stability | 40% | Is RSS stable after warmup? |
| Final RSS | 25% | Lower is better |
| Performance | 20% | No regression vs baseline |
| Simplicity | 15% | Fewer dependencies, easier maintenance |

### Decision Matrix Template

| Allocator | RSS Stable? | Final RSS | Perf vs Base | Deps | Score |
|-----------|-------------|-----------|--------------|------|-------|
| glibc | No | 1,045 MB | baseline | 0 | - |
| glibc+trim | ? | ? MB | ? | +1 (libc) | ? |
| mimalloc | ? | ? MB | ? | +1 | ? |
| jemalloc | ? / N/A | ? MB | ? | +1 | ? / Skip |

### Selection Rules

1. **Must be RSS stable** - If not stable, disqualify
2. **Must not regress performance** - If >5% slower, disqualify
3. **Must compile reliably** - If causes SIGBUS, disqualify
4. **Among qualifying options**, prefer:
   - Lower final RSS
   - Better performance
   - Simpler (fewer/smaller dependencies)

### Fallback Decision Tree

```
┌─────────────────────────────────────┐
│ mimalloc stable AND performs well?  │
└──────────────────┬──────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
       YES                  NO
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌─────────────────────┐
│ mimalloc wins!  │  │ Try jemalloc if     │
│ Use as default  │  │ stable, else        │
└─────────────────┘  │ malloc_trim combo   │
                     └─────────────────────┘
```

### Output

- `docs/ALLOCATOR_COMPARISON.md` with full analysis
- Clear recommendation for default allocator
- Documented rationale (including why jemalloc skipped if applicable)

## Acceptance Criteria

- [ ] All available allocator results collected and compared
- [ ] Decision matrix completed with scores
- [ ] Clear winner identified (or documented if no option works)
- [ ] Performance impact documented
- [ ] jemalloc skip documented if applicable
- [ ] Recommendation approved for implementation

## Implementation Guide

### Suggested Approach

1. Collect all available RSS analysis documents
2. Run performance benchmarks for each stable allocator
3. Create comparison document with all metrics
4. Apply decision criteria and score each option
5. Document recommendation with rationale

### If jemalloc Was Skipped

Add this section to the comparison document:

```markdown
## jemalloc Status: NOT TESTED

jemalloc was excluded from comparison due to:
- SIGBUS crashes during test compilation
- System resource exhaustion
- Environment instability

**Recommendation**: jemalloc may be revisited in future if:
- Compilation issues are resolved upstream
- Testing is done in isolated environment with more resources
- Alternative jemalloc crate becomes available

For now, mimalloc is the recommended default.
```

### Performance Benchmark Commands

```bash
# Baseline (glibc)
cargo bench --bench sddp_e2e 2>&1 | tee bench_glibc.log

# mimalloc
cargo bench --bench sddp_e2e --features mimalloc 2>&1 | tee bench_mimalloc.log

# jemalloc (ONLY if T-133 succeeded, use -j1)
# cargo bench -j1 --bench sddp_e2e --features jemalloc 2>&1 | tee bench_jemalloc.log
```

### Analysis Document Template

```markdown
# Allocator Comparison Analysis

## Summary

Based on testing, **[WINNER]** is recommended as the default allocator.

## Test Coverage

| Allocator | Tested | Reason if Not |
|-----------|--------|---------------|
| glibc | ✅ Yes | Baseline |
| glibc+trim | ✅ Yes | T-134 |
| mimalloc | ✅ Yes | T-131 |
| jemalloc | ✅/❌ | T-133 / Compilation issues |

## RSS Stability Results

| Allocator | RSS Stable? | Final RSS | Delta vs glibc |
|-----------|-------------|-----------|----------------|
| glibc | ❌ No | 1,045 MB | baseline |
| glibc+trim | ? | ? MB | ? |
| mimalloc | ? | ? MB | ? |
| jemalloc | ?/N/A | ? MB | ? |

## Performance Results

| Allocator | Benchmark Time | vs Baseline |
|-----------|----------------|-------------|
| glibc | X.XX s | baseline |
| mimalloc | X.XX s | +/-Y% |
| jemalloc | X.XX s / N/A | +/-Y% |

## Decision Matrix

| Allocator | Stable (40%) | RSS (25%) | Perf (20%) | Simple (15%) | Total |
|-----------|--------------|-----------|------------|--------------|-------|
| glibc+trim | ? | ? | ? | 12 | ? |
| mimalloc | ? | ? | ? | 12 | ? |
| jemalloc | ?/0 | ? | ? | 12 | ? |

Scoring: 
- Stable = 40 if yes, 0 if no/N/A
- RSS = 25 * (1 - (rss - min_rss) / range)
- etc.

## Recommendation

**Default allocator**: [WINNER - likely mimalloc]

**Rationale**:
1. [Reason 1]
2. [Reason 2]
3. [Reason 3]

## Fallback Options

If the winner has platform issues, users can:
- `--features system-allocator` to use glibc
- `--features [alternative]` to use backup option

## Known Issues

### jemalloc Compilation (if applicable)
jemalloc was not included in comparison due to SIGBUS crashes during
test compilation. See T-133 for details. This does not affect the
recommendation since mimalloc is expected to perform well.
```

### Pitfalls to Avoid

- ⚠️ Don't block on jemalloc if it's causing instability
- ⚠️ Don't choose based on single metric - balance all criteria
- ⚠️ Ensure benchmark runs are consistent (multiple runs, warm cache)
- ⚠️ Consider cross-platform implications
- ⚠️ Document why options were skipped

## Testing Requirements

### Benchmark Consistency

- [ ] Run each benchmark 3+ times
- [ ] Use criterion's statistical analysis
- [ ] Document measurement conditions

## Documentation Requirements

- [ ] Create `docs/ALLOCATOR_COMPARISON.md`
- [ ] Clear recommendation with rationale
- [ ] Document jemalloc skip if applicable
- [ ] Document how to use alternative allocators

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Analysis and documentation, no code changes; jemalloc skip simplifies comparison

## Definition of Done

- [ ] All available allocator results compared
- [ ] Performance benchmarks complete (for stable allocators)
- [ ] Winner selected with documented rationale
- [ ] jemalloc exclusion documented (if applicable)
- [ ] Recommendation ready for T-136 implementation
