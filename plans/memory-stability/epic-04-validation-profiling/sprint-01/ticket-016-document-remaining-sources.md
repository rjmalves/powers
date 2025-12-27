# [TICKET-016] Document remaining memory sources

> **Epic**: [Epic 4: Validation & Profiling](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [TICKET-015](./ticket-015-algorithm-validation.md)
> **Blocks**: None (final ticket)

## Context

### Background

After implementing all preallocation changes, some memory sources will remain. These should be documented to set expectations and identify future optimization opportunities.

### Relation to Epic

Completes the memory stability work with comprehensive documentation.

## Specification

### Expected Remaining Sources

Based on MEMORY_STABILITY_ANALYSIS.md:

1. **HiGHS internal memory** (~50 MB/iteration?)
   - Basis factorization tables
   - Simplex iteration workspace
   - Sparse matrix index structures
   - Limited control (external library)

2. **HashMap entry storage** (~0.5 MB/iteration)
   - `active_cut_indices` entries grow with active cuts
   - Could be bounded but low priority

3. **Allocator fragmentation**
   - RSS may exceed actual usage
   - mimalloc helps (already implemented)

4. **Thread-local buffers**
   - Cut evaluation buffers (already optimized)
   - Scenario generation buffers

### Documentation Structure

```markdown
## Remaining Memory Growth Sources (Post-Implementation)

### 1. HiGHS Internal Memory
**Location**: External library
**Growth**: ~50 MB/iteration (estimated)
**Cause**: Basis factorization, working memory
**Mitigation**: Limited (external library)
**Future**: Consider HiGHS parameters (simplex_update_limit)

### 2. HashMap Entry Storage
**Location**: `fcf.cut_pool.active_cut_indices`
**Growth**: ~0.5 MB/iteration
**Cause**: New entries for active cuts
**Mitigation**: Could bound map size
**Future**: Low priority (minimal impact)

### 3. Allocator Behavior
**Location**: System allocator / mimalloc
**Growth**: Variable
**Cause**: Fragmentation, page retention
**Mitigation**: mimalloc already configured
**Future**: Monitor in production

### Summary
| Source | Per-Iteration | Controllable | Priority |
|--------|--------------|--------------|----------|
| HiGHS internal | ~50 MB | No | N/A |
| HashMap entries | ~0.5 MB | Yes | Low |
| Allocator | Variable | Partial | Low |
```

## Acceptance Criteria

- [ ] All remaining memory sources identified
- [ ] Per-source documentation complete
- [ ] Future optimization opportunities noted
- [ ] MEMORY_STABILITY_ANALYSIS.md updated

## Implementation Guide

### Suggested Approach

1. Review profiling results from TICKET-014
2. Identify sources of remaining growth
3. Investigate HiGHS memory behavior
4. Document findings in analysis file
5. Note future optimization opportunities

### Key Files to Update

- `MEMORY_STABILITY_ANALYSIS.md`: Add "Results" and "Remaining Sources" sections
- Optionally: Create `MEMORY_OPTIMIZATION_FUTURE.md` for future work

## Deliverables

1. Updated MEMORY_STABILITY_ANALYSIS.md with:
   - Implementation results
   - Remaining memory sources
   - Future optimization opportunities

2. Summary in README or CHANGELOG if appropriate

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Documentation and analysis task
