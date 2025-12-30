# [T-079] Benchmark Memory Usage and Performance

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 4: Pool Architecture Refinement](./00-sprint-overview.md)
> **Dependencies**: [T-078](./ticket-078-remove-concrete-state.md)
> **Blocks**: None (final ticket)

## Files to Read Before Starting

- `benches/` - Existing benchmark files
- `src/state.rs` - StateData and VisitedStatePool
- `docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md` - Architecture documentation

---

## Context

### Background

Sprint 4 refactored `VisitedStatePool` to eliminate layout duplication. This ticket validates that the refactoring achieved its goals:

1. **Memory reduction**: Layout stored once instead of per state
2. **No performance regression**: StateData access as fast as ConcreteState
3. **Correctness preserved**: All tests pass, results identical

---

## Specification

### Memory Measurements

Create a simple test that compares memory usage:

```rust
#[test]
fn test_memory_layout_efficiency() {
    use std::mem::size_of;
    
    // StateData should be minimal
    println!("StateData size: {} bytes", size_of::<StateData>());
    
    // VisitedStatePool overhead
    println!("VisitedStatePool size: {} bytes", size_of::<VisitedStatePool>());
    
    // For 500 states with 10 hydros (AR(1)):
    // Old: 500 × (StateCore + num_hydros + StateLayout)
    //    = 500 × (56 + 8 + ~160) = 500 × 224 = 112,000 bytes
    // New: 500 × StateData + 1 × StateLayout
    //    = 500 × 56 + ~160 = 28,160 bytes
    // Savings: ~84 KB (75% reduction in overhead)
}
```

### Performance Benchmarks

Use criterion to benchmark key operations:

1. **State coefficient access**: `state.coefficients()`
2. **State update**: `pool.update_state()`
3. **Domination evaluation**: Full FCF domination check
4. **Pool iteration**: `pool.pool.iter()`

### Comparison Points

| Metric | Before Sprint 4 | After Sprint 4 | Change |
|--------|-----------------|----------------|--------|
| StateData size | N/A (ConcreteState) | 56 bytes | Baseline |
| Pool overhead per state | ~168 bytes | 0 bytes | -100% |
| Layout allocations | 2 × N states | 2 total | -99.8% |
| State access time | ? ns | ? ns | Should be equal |

---

## Implementation Guide

### Step 1: Create memory size test

Add to `src/state.rs` tests:

```rust
#[test]
fn test_sprint_04_memory_efficiency() {
    use std::mem::size_of;
    
    // StateData is compact
    let state_data_size = size_of::<StateData>();
    assert!(state_data_size <= 64, "StateData should be <= 64 bytes, got {}", state_data_size);
    
    // VisitedStatePool with shared layout
    let config = StateConfig::StorageAndInflow {
        num_hydros: 10,
        per_hydro_state_dims: vec![2; 10],  // AR(1) for all
    };
    let pool = VisitedStatePool::preallocate_concrete(10, 50, &config);
    
    // 500 states
    assert_eq!(pool.pool.len(), 500);
    
    // Layout stored once (not 500 times)
    assert!(pool.layout.is_some());
    let layout = pool.layout.as_ref().unwrap();
    assert_eq!(layout.per_hydro_dims.len(), 10);
    
    // Estimate memory savings
    // Old: 500 × (~168 bytes layout overhead) = 84,000 bytes
    // New: 1 × (~168 bytes layout) = 168 bytes
    // This is verified by the type system - no layout in StateData
}
```

### Step 2: Create performance benchmark (optional)

If `benches/` directory exists and has criterion setup:

```rust
// benches/state_pool_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use powers_rs::state::{StateConfig, StateData, VisitedStatePool};

fn bench_state_access(c: &mut Criterion) {
    let config = StateConfig::Storage { num_hydros: 100 };
    let pool = VisitedStatePool::preallocate_concrete(10, 50, &config);
    
    c.bench_function("state_coefficient_access", |b| {
        b.iter(|| {
            let mut sum = 0.0;
            for state in &pool.pool {
                sum += black_box(state.coefficients().iter().sum::<f64>());
            }
            sum
        })
    });
}

fn bench_state_update(c: &mut Criterion) {
    let config = StateConfig::Storage { num_hydros: 100 };
    let mut pool = VisitedStatePool::preallocate_concrete(10, 50, &config);
    let coeffs: Vec<f64> = (0..100).map(|i| i as f64).collect();
    
    c.bench_function("state_update", |b| {
        b.iter(|| {
            for slot in 0..500 {
                black_box(pool.update_state(slot, &coeffs, 1, slot));
            }
        })
    });
}

criterion_group!(benches, bench_state_access, bench_state_update);
criterion_main!(benches);
```

### Step 3: Run golden tests

Ensure numerical correctness is preserved:

```bash
cargo test -j1 --test test_sddp_algorithm
cargo test -j1 --test test_integration_suite
```

### Step 4: Document results

Update architecture documentation with measured improvements.

---

## Acceptance Criteria

- [ ] Memory size test passes
- [ ] No performance regression (state access time)
- [ ] All golden tests pass
- [ ] Results documented

---

## Testing Requirements

### Memory Tests

```bash
cargo test -j1 --lib sprint_04_memory
```

### Performance Tests (if criterion available)

```bash
cargo bench -- state_pool
```

### Full Test Suite

```bash
cargo test -j1
```

---

## Pitfalls to Avoid

- ⚠️ **Don't block on criterion benchmarks** - Memory test is sufficient
- ⚠️ **Measure before celebrating** - Verify actual improvement

---

## Documentation Requirements

- [ ] Add memory efficiency notes to VisitedStatePool docs
- [ ] Update architecture documentation with Sprint 4 improvements
- [ ] Update CHANGELOG

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Mostly verification and documentation. Benchmarks are optional enhancement.

---

## Definition of Done

- [ ] Memory efficiency test passes
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Sprint 4 completion verified
