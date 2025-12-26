# [TICKET-010] Validate memory profile

> **Epic**: [Epic 2: FCF Full Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-009](./ticket-009-ensure-with-capacity.md)  
> **Blocks**: None (Epic 2 complete)

## Context

### Background

After updating FCF instantiation, validate that memory profile is flat and performance improved.

### Relation to Epic

Final validation ticket for Epic 2.

## Specification

### Validation Tasks

1. **Memory Profile**: Massif shows flat curve during training
2. **Performance**: Hyperfine shows ≥1% improvement
3. **Correctness**: Examples produce identical results

### Expected Results

With Epic 1 (HiGHS) + Epic 2 (FCF) complete:
- Memory growth during training: ≈0%
- Performance improvement: ≥4-5% cumulative

## Acceptance Criteria

- [ ] Memory profile flat (±1%) during training
- [ ] Performance improvement measurable (≥1% from FCF alone)
- [ ] Cumulative improvement ≥4% (Epic 1 + Epic 2)
- [ ] Examples produce identical results
- [ ] CHANGELOG.md updated

## Implementation Guide

### Validation Commands

```bash
# Memory profiling
valgrind --tool=massif --massif-out-file=massif_fcf.out \
  ./target/release/powers run examples/07-par-model-with-inflow-state
ms_print massif_fcf.out | head -100

# Performance benchmark
hyperfine --warmup 2 --runs 10 \
  'cargo run --release -- run examples/07-par-model-with-inflow-state'

# Compare with baseline (record before changes)
# baseline_time vs new_time

# Correctness check
cargo run --release -- run examples/01-deterministic 2>&1 | grep "lower"
cargo run --release -- run examples/07-par-model-with-inflow-state 2>&1 | tail -5
```

### Documentation

Update CHANGELOG.md:

```markdown
## [Unreleased]

### Performance

- **FCF Preallocation**: Ensure all FutureCostFunction instances use with_capacity()
  - Eliminates Vec/HashMap reallocations during training
  - ~1-3% performance improvement
- **HiGHS Constraint Preallocation**: Pre-allocate cut constraint slots
  - Eliminates dynamic row addition during training
  - ~3-8% performance improvement
- **Cumulative**: ~4-11% total improvement

### Memory

- Memory profile flat during training (±1%)
- Deterministic memory footprint for HPC deployment
```

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward validation tasks

## Definition of Done

- [ ] Memory profile validated as flat
- [ ] Performance improvement measured
- [ ] Correctness verified
- [ ] CHANGELOG.md updated
- [ ] Epic 2 complete ✅
