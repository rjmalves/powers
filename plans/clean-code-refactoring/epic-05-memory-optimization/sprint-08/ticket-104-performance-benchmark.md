# [T-104] Performance Benchmark Comparison

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8: Validation and Documentation](./00-sprint-overview.md)
> **Dependencies**: Sprint 7 complete
> **Blocks**: T-105

---

## Context

### Background

Verify that memory optimizations haven't caused performance regressions, and measure any performance improvements from reduced allocation overhead.

### Expected Outcome

- No regression (≤5% slowdown acceptable)
- Potential improvement from reduced allocation churn
- Document any tradeoffs

## Specification

### Tasks

1. **Run criterion benchmarks** before and after optimizations
2. **Run end-to-end training timing**
3. **Compare solve throughput**
4. **Document results**

### Benchmarks to Run

1. **End-to-end training** - Full example 05 training
2. **Solve throughput** - LP solves per second
3. **Cut computation** - Cut evaluation hot path
4. **Forward pass** - Single forward pass timing

## Acceptance Criteria

- [ ] All benchmarks completed
- [ ] Before/after comparison documented
- [ ] No regression ≥5% slowdown
- [ ] Any improvements quantified

## Implementation Guide

### Suggested Approach

1. **Checkout pre-optimization baseline**:
   ```bash
   git stash
   git checkout sprint5-baseline  # Or appropriate tag
   cargo build --release
   cargo bench --bench sddp_training -- --save-baseline pre-optimization
   git checkout main
   git stash pop
   ```

2. **Run current benchmarks**:
   ```bash
   cargo build --release
   cargo bench --bench sddp_training -- --baseline pre-optimization
   ```

3. **Run end-to-end timing**:
   ```bash
   # Before
   time ./target/release/powers run examples/05-large-scale-brazilian
   
   # After (current)
   cargo build --release
   time ./target/release/powers run examples/05-large-scale-brazilian
   ```

4. **Create comparison report**:
   ```markdown
   ## Performance Comparison
   
   | Benchmark | Pre-Optimization | Post-Optimization | Change |
   |-----------|------------------|-------------------|--------|
   | example-05 total | X.X sec | Y.Y sec | ±Z.Z% |
   | forward pass | X.X ms | Y.Y ms | ±Z.Z% |
   | backward pass | X.X ms | Y.Y ms | ±Z.Z% |
   | cut evaluation | X.X µs | Y.Y µs | ±Z.Z% |
   ```

5. **If regression detected**:
   - Profile to identify cause
   - Document tradeoff (memory vs. speed)
   - Decide if acceptable

### Key Benchmarks

```rust
// benches/sddp_training.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("sddp_training");
    group.sample_size(10);  // Training takes time
    
    group.bench_function("example_05", |b| {
        b.iter(|| {
            // Run training
        });
    });
    
    group.finish();
}
```

## Testing Requirements

- [ ] Benchmarks run without errors
- [ ] Results are reproducible (low variance)
- [ ] Comparison is fair (same hardware, same inputs)

## Documentation Requirements

- [ ] Add performance comparison to `docs/MEMORY_BEHAVIOR.md`
- [ ] Document any tradeoffs

## Effort Estimate

**Points**: 3
**Confidence**: High
