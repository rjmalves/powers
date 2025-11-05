# [TICKET-009] Performance Benchmarking and Optimization

**Sprint:** 4  
**Estimated Effort:** 3 story points (2 days)  
**Confidence:** High  
**Priority:** P2 - Medium

## Context

With the refactoring complete, we need to verify and document performance characteristics. The explicit separation should provide performance improvements through:
- Direct indexed access (no filtering)
- Better cache locality (contiguous memory for same entity type)
- Reduced branching (no entity type matching)

This ticket measures these improvements and optimizes any remaining hotspots.

## Acceptance Criteria

- [ ] Given baseline performance metrics from old implementation, when running with new implementation, then no operation is more than 5% slower
- [ ] Given cut generation benchmark, when measuring with new implementation, then performance is at least 15% faster
- [ ] Given dual extraction benchmark, when measuring with new implementation, then performance is at least 30% faster
- [ ] Given full SDDP iteration benchmark, when measuring with new implementation, then overhead from explicit structures is < 2%
- [ ] Performance: All measurements are reproducible (< 3% variance across runs)

## Tasks

### Implementation

- [ ] Create benchmark suite in `benches/explicit_lag_separation.rs`
  
- [ ] Benchmark: Subproblem variable creation
  - Measure time to create all lag variables
  - Test with various system sizes (10, 50, 100 entities)
  - Compare old unified vs new explicit approach
  
- [ ] Benchmark: Cut generation (add_cut_constraint_to_model)
  - Measure time to add 1000 cuts
  - Test with various AR order distributions
  - Compare old heuristic vs new explicit access
  - Expected: 15-30% improvement
  
- [ ] Benchmark: Dual extraction
  - Measure time to extract duals from 1000 solutions
  - Test with mixed entity types
  - Compare filtering vs direct access
  - Expected: 30-50% improvement
  
- [ ] Benchmark: State extraction
  - Measure time to extract lag values from trajectories
  - Test multi-stage scenarios
  - Expected: 10-20% improvement
  
- [ ] Benchmark: Full SDDP iteration
  - Measure end-to-end iteration time
  - Break down time by component
  - Verify refactoring overhead is minimal
  
- [ ] Benchmark: Memory usage
  - Measure heap allocation for explicit structures
  - Compare to unified structure
  - Verify no additional overhead
  
- [ ] Profile hotspots using `cargo flamegraph` or similar
  - Identify any unexpected bottlenecks
  - Optimize if needed

- [ ] Create performance regression tests for CI
  - Lightweight benchmarks that run on every commit
  - Alert if performance degrades > 10%

### Testing

- [ ] Verify benchmarks are reproducible
  - Run each benchmark 10 times
  - Calculate mean and standard deviation
  - Ensure stddev < 3% of mean
  
- [ ] Cross-platform validation
  - Run benchmarks on Linux, macOS (if available)
  - Verify improvements are consistent
  
- [ ] Validate under different compiler optimization levels
  - `--release` (primary target)
  - `--release` with `lto = "fat"` 
  - Profile-guided optimization if applicable

- [ ] Memory profiler validation
  - Use `valgrind --tool=massif` or similar
  - Verify no memory leaks
  - Confirm memory usage is identical or better

### Documentation

- [ ] Create `BENCHMARK_RESULTS.md` update with new metrics
- [ ] Document performance characteristics of explicit structures
- [ ] Add performance notes to README.md
- [ ] Create guide for running benchmarks
- [ ] Document any optimizations applied

## Technical Notes

### Benchmark Structure

```rust
// benches/explicit_lag_separation.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use powers::*;

fn bench_cut_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cut_generation");
    
    for n_entities in [10, 30, 50, 100].iter() {
        let system = create_benchmark_system(*n_entities);
        let subproblem = create_subproblem(&system);
        let state = StorageAndInflowState::new(&system);
        
        // Create test cut
        let cut = create_test_cut(&state);
        
        group.bench_with_input(
            BenchmarkId::new("add_cut", n_entities),
            n_entities,
            |b, _| {
                b.iter(|| {
                    let mut model = black_box(subproblem.model.clone());
                    state.add_cut_constraint_to_model(
                        black_box(&mut cut.clone()),
                        black_box(&subproblem.variables),
                        black_box(&mut model),
                    );
                });
            },
        );
    }
    
    group.finish();
}

fn bench_dual_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_extraction");
    
    for n_entities in [10, 30, 50, 100].iter() {
        let system = create_benchmark_system(*n_entities);
        let subproblem = create_subproblem(&system);
        let solution = create_test_solution(&subproblem);
        
        group.bench_with_input(
            BenchmarkId::new("extract_duals", n_entities),
            n_entities,
            |b, _| {
                b.iter(|| {
                    let (load_duals, inflow_duals) = 
                        subproblem.get_lag_duals_from_solution(
                            black_box(&solution),
                            black_box(&system),
                        );
                    black_box((load_duals, inflow_duals));
                });
            },
        );
    }
    
    group.finish();
}

fn bench_full_iteration(c: &mut Criterion) {
    let system = create_benchmark_system(50);
    let mut sddp = SDDP::new(system, default_params());
    sddp.set_seed(42);
    
    // Warm up
    for _ in 0..5 {
        sddp.run_iteration();
    }
    
    c.bench_function("full_sddp_iteration", |b| {
        b.iter(|| {
            sddp.run_iteration();
        });
    });
}

criterion_group!(
    benches,
    bench_cut_generation,
    bench_dual_extraction,
    bench_full_iteration,
);
criterion_main!(benches);
```

### Expected Performance Improvements

Based on analysis in architecture document:

| Operation | Before | After | Expected Improvement |
|-----------|--------|-------|---------------------|
| Cut generation | O(n_entities) with filtering | O(n_hydros) direct | 2-3x faster |
| Dual extraction | O(n_entities) with match | O(n_hydros + n_buses) | 1.5-2x faster |
| State extraction | O(n_entities) with match | O(n_hydros + n_buses) | 1.3-1.5x faster |
| Full iteration | Baseline | + overhead | < 2% overhead |

### System Configurations for Benchmarking

```rust
fn create_benchmark_system(n_entities: usize) -> System {
    // Split roughly 60% buses, 40% hydros
    let n_buses = (n_entities * 6) / 10;
    let n_hydros = n_entities - n_buses;
    
    let mut builder = SystemBuilder::new();
    
    // Add buses with varied AR orders
    for bus_id in 0..n_buses {
        builder = builder.add_bus(bus_id);
        let ar_order = match bus_id % 3 {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => unreachable!(),
        };
        builder = builder.add_load_temporal_model(
            bus_id,
            ar_order,
            mean: 50.0,
        );
    }
    
    // Add hydros with varied AR orders
    for hydro_id in 0..n_hydros {
        builder = builder.add_hydro(hydro_id, capacity: 100.0);
        let ar_order = match hydro_id % 3 {
            0 => 1,
            1 => 2,
            2 => 3,
            _ => unreachable!(),
        };
        builder = builder.add_inflow_temporal_model(
            hydro_id,
            ar_order,
            mean: 70.0,
        );
    }
    
    builder.build()
}
```

### Memory Profiling

```bash
# Using Valgrind Massif
valgrind --tool=massif --massif-out-file=massif.out \
    ./target/release/powers examples/07-ar-inflows

ms_print massif.out > memory_profile.txt

# Look for:
# - Peak memory usage
# - Memory allocation patterns
# - Any unexpected growth
```

### Optimization Opportunities

If benchmarks show unexpected slowness:

1. **Inline hot functions:**
   ```rust
   #[inline(always)]
   pub fn get_lags(&self, entity_id: usize) -> &[usize] {
       &self.lags_by_entity[entity_id]
   }
   ```

2. **Use iterators efficiently:**
   ```rust
   // Prefer:
   constraints.iter().map(|&idx| solution.rowdual[idx]).collect()
   
   // Over:
   let mut duals = Vec::new();
   for &idx in constraints {
       duals.push(solution.rowdual[idx]);
   }
   ```

3. **Avoid unnecessary clones:**
   ```rust
   // Use references where possible
   fn process_lags(&self, lags: &[usize]) { ... }
   // Not:
   fn process_lags(&self, lags: Vec<usize>) { ... }
   ```

4. **Consider `SmallVec` for short lag vectors:**
   ```rust
   use smallvec::{SmallVec, smallvec};
   
   // AR orders rarely exceed 4, avoid heap allocation
   type LagVec = SmallVec<[usize; 4]>;
   ```

### CI Performance Tests

Add lightweight performance regression test:

```rust
#[test]
fn test_cut_generation_performance() {
    let system = create_benchmark_system(30);
    let subproblem = create_subproblem(&system);
    let state = StorageAndInflowState::new(&system);
    let cut = create_test_cut(&state);
    
    let iterations = 1000;
    let start = std::time::Instant::now();
    
    for _ in 0..iterations {
        let mut model = subproblem.model.clone();
        state.add_cut_constraint_to_model(&mut cut.clone(), &subproblem.variables, &mut model);
    }
    
    let elapsed = start.elapsed();
    let per_cut = elapsed / iterations;
    
    // Should be very fast (< 10 microseconds per cut)
    assert!(
        per_cut.as_micros() < 10,
        "Cut generation too slow: {:?} per cut",
        per_cut
    );
}
```

### Reporting Template

```markdown
## Performance Results

### Hardware
- CPU: [processor model]
- RAM: [amount]
- OS: [operating system]
- Rust: [version]
- Compiler flags: `RUSTFLAGS="-C target-cpu=native"`

### Benchmark Results

#### Cut Generation
| Entities | Before (μs) | After (μs) | Improvement |
|----------|-------------|------------|-------------|
| 10       | 2.5         | 1.2        | 52%         |
| 30       | 7.8         | 3.1        | 60%         |
| 50       | 13.2        | 5.0        | 62%         |
| 100      | 27.5        | 9.8        | 64%         |

#### Dual Extraction
| Entities | Before (μs) | After (μs) | Improvement |
|----------|-------------|------------|-------------|
| 10       | 3.2         | 1.5        | 53%         |
| 30       | 9.5         | 4.2        | 56%         |
| 50       | 16.8        | 6.8        | 60%         |
| 100      | 35.2        | 13.1       | 63%         |

#### Full SDDP Iteration
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Mean   | 125ms  | 123ms | -1.6%  |
| Stddev | 3ms    | 3ms   | -      |

### Memory Usage
- Heap allocation: No change (0% increase)
- Peak memory: Identical
- Cache performance: Improved (fewer cache misses)
```

## Dependencies

- Blocked by: TICKET-001 through TICKET-007 (implementation complete)
- Blocks: None
- Related: TICKET-008 (integration tests validate correctness while this validates performance)

## Definition of Done

- [ ] All benchmarks implemented and documented
- [ ] Performance improvements measured and verified
- [ ] No operation regresses > 5%
- [ ] Memory usage verified identical or better
- [ ] Results documented in BENCHMARK_RESULTS.md
- [ ] CI includes lightweight performance tests
- [ ] Any optimizations applied and documented
