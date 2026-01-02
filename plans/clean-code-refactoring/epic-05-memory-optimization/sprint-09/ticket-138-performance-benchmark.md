# [T-138] Performance Benchmark with New Allocator

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 9: RSS Stabilization](./00-sprint-overview.md)
> **Dependencies**: T-136
> **Blocks**: None

---

## Context

### Background

Before finalizing the allocator change, we need comprehensive performance benchmarks to ensure no regression. The allocator selection in T-135 included preliminary benchmarks, but this ticket provides thorough validation.

### Current State

- Criterion benchmarks exist: `benches/sddp_e2e.rs`, `benches/simd_dot_product.rs`
- Baseline performance with glibc is known
- New allocator is default per T-136

## Specification

### Benchmarks to Run

1. **sddp_e2e**: End-to-end SDDP algorithm performance
2. **simd_dot_product**: Low-level numerical operations
3. **Training time**: Wall-clock time for example 05

### Metrics to Capture

| Metric | Baseline (glibc) | New Allocator | Delta |
|--------|------------------|---------------|-------|
| sddp_e2e mean time | X ms | Y ms | ±Z% |
| simd_dot_product | X ns | Y ns | ±Z% |
| Example 05 training | X s | Y s | ±Z% |
| Peak RSS | 1,045 MB | Y MB | -Z% |

### Acceptance Threshold

- **Performance**: No more than 5% regression in any benchmark
- **Memory**: Significant RSS improvement (primary goal)

## Acceptance Criteria

- [ ] sddp_e2e benchmark shows ≤5% regression
- [ ] simd_dot_product shows no regression
- [ ] Example 05 training time within 5% of baseline
- [ ] RSS improvement documented
- [ ] Results added to sprint documentation

## Implementation Guide

### Suggested Approach

1. Run baseline benchmarks with system allocator
2. Run benchmarks with new default allocator
3. Compare and document results
4. Investigate any significant regressions

### Commands

```bash
# Baseline (system allocator)
cargo bench --bench sddp_e2e --no-default-features 2>&1 | tee bench_baseline.log
cargo bench --bench simd_dot_product --no-default-features 2>&1 | tee bench_simd_baseline.log

# New allocator (default)
cargo bench --bench sddp_e2e 2>&1 | tee bench_new.log
cargo bench --bench simd_dot_product 2>&1 | tee bench_simd_new.log

# Training time comparison
time cargo run --release --no-default-features -- run examples/05-large-scale-brazilian
time cargo run --release -- run examples/05-large-scale-brazilian

# Memory comparison (use RSS harness)
cargo test --release test_rss_stability -- --nocapture
cargo test --release --no-default-features test_rss_stability -- --nocapture
```

### Criterion Report

Criterion generates HTML reports in `target/criterion/`. Compare:
- `target/criterion/sddp_e2e/report/index.html`

### Analysis Template

```markdown
# Performance Benchmark Results

## sddp_e2e Benchmark

| Allocator | Mean | Std Dev | vs Baseline |
|-----------|------|---------|-------------|
| glibc | X.XX ms | X.XX ms | baseline |
| [winner] | X.XX ms | X.XX ms | ±X.X% |

## simd_dot_product Benchmark

| Allocator | Mean | Std Dev | vs Baseline |
|-----------|------|---------|-------------|
| glibc | X ns | X ns | baseline |
| [winner] | X ns | X ns | ±X.X% |

## Training Time (Example 05, 20 iterations)

| Allocator | Time | vs Baseline |
|-----------|------|-------------|
| glibc | X.X s | baseline |
| [winner] | X.X s | ±X.X% |

## Memory Usage

| Allocator | Final RSS | vs Baseline |
|-----------|-----------|-------------|
| glibc | 1,045 MB | baseline |
| [winner] | X MB | -X.X% |

## Conclusion

Performance [meets/does not meet] acceptance criteria.
Memory improvement: X%.
```

### Pitfalls to Avoid

- ⚠️ Run benchmarks on quiet system (no other heavy processes)
- ⚠️ Run multiple iterations for statistical significance
- ⚠️ Use release builds only

## Testing Requirements

### Benchmark Runs

- [ ] At least 3 runs of each benchmark
- [ ] Criterion statistical analysis used
- [ ] Results consistent across runs

## Documentation Requirements

- [ ] Create `docs/ALLOCATOR_PERFORMANCE.md`
- [ ] Include all benchmark results
- [ ] Document measurement methodology

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Running benchmarks and analysis, clear procedure

## Definition of Done

- [ ] All benchmarks run and compared
- [ ] Results within acceptance threshold
- [ ] Documentation complete
- [ ] Any regressions investigated
