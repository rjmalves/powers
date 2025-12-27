# [TICKET-014] Memory profiling on example 05

> **Epic**: [Epic 4: Validation & Profiling](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: Epic 3 complete
> **Blocks**: [TICKET-015](./ticket-015-algorithm-validation.md)

## Context

### Background

After implementing all preallocation changes, we need to measure memory usage to verify the targets are met. Example 05 (large-scale Brazilian) is the primary benchmark due to its size (156 hydros, 121 thermals, 60 stages).

### Relation to Epic

Provides quantitative validation of the memory stability implementation.

## Specification

### Metrics to Measure

1. **Peak RSS**: Maximum resident set size during training
   - Target: < 3 GB (from ~5 GB baseline)
   
2. **Memory per iteration**: RSS delta between iterations
   - Target: < 10 MB/iteration (from ~375 MB baseline)
   
3. **Memory at initialization**: RSS after handler/FCF setup
   - Expected: ~2.5-3.5 GB (handlers + preallocated pools)

4. **Allocation count**: Number of heap allocations during training
   - Target: < 1 MB/iteration

### Measurement Commands

```bash
# Peak RSS
/usr/bin/time -v ./target/release/powers run examples/05-large-scale-brazilian 2>&1 | \
  grep "Maximum resident set size"

# Per-iteration memory (requires logging changes or external tool)
# Option 1: Add internal logging
# Option 2: Use massif
valgrind --tool=massif --pages-as-heap=yes --massif-out-file=massif.out \
  ./target/release/powers run examples/05-large-scale-brazilian

# Allocation tracking (detailed)
heaptrack ./target/release/powers run examples/05-large-scale-brazilian
```

### Expected Results

| Metric | Baseline | Target | Notes |
|--------|----------|--------|-------|
| Peak RSS | ~5 GB | < 3 GB | 40% reduction |
| Per-iteration growth | ~375 MB | < 10 MB | 97% reduction |
| Init memory | ~2 GB | ~3 GB | Preallocation adds ~1 GB |

## Acceptance Criteria

- [ ] Peak RSS < 3 GB measured
- [ ] Per-iteration growth < 10 MB measured
- [ ] Results documented with exact numbers
- [ ] Comparison table with baseline

## Implementation Guide

### Suggested Approach

1. Build release binary: `cargo build --release`
2. Run baseline measurement (if not already recorded)
3. Run post-implementation measurements
4. Create comparison table
5. Investigate any unexpected results

### Measurement Script

```bash
#!/bin/bash
# measure_memory.sh

echo "=== Memory Profiling: Example 05 ==="
echo "Date: $(date)"
echo ""

echo "Peak RSS:"
/usr/bin/time -v ./target/release/powers run examples/05-large-scale-brazilian 2>&1 | \
  grep -E "Maximum resident set size|Elapsed"

echo ""
echo "Detailed allocation (requires valgrind):"
# valgrind --tool=massif ...
```

## Testing Requirements

### Measurements

- [ ] Run 3 times and take average for consistency
- [ ] Record exact numbers for documentation
- [ ] Note any variance between runs

## Documentation Requirements

- [ ] Update MEMORY_STABILITY_ANALYSIS.md with results
- [ ] Create comparison table (before/after)
- [ ] Document measurement methodology

## Deliverables

1. Memory measurement report with:
   - Exact RSS values
   - Per-iteration growth (if measurable)
   - Comparison to baseline
   - Any unexpected findings

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Measurement and documentation task
