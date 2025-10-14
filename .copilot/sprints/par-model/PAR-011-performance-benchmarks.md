# PAR-011: Performance Benchmarks for PAR Implementation

## Context

Measure performance impact of PAR model vs stationary AR. Target: <10% overhead for typical use cases. Identify bottlenecks and optimize if needed.

## Acceptance Criteria

- [ ] Benchmark PAR(1) vs stationary AR(1) generation speed
- [ ] Benchmark PAR(2) vs stationary AR(2) generation speed
- [ ] Benchmark varying orders (PAR(p) with different p per period)
- [ ] Memory usage comparison (buffer sizes)
- [ ] Scenario generation end-to-end benchmark
- [ ] Regression: No impact on non-PAR pipelines
- [ ] Benchmark results documented in `benches/` directory

## Tasks

### Implementation

- [ ] Create `benches/par_performance.rs`
- [ ] Benchmark: PAR(1) single-series generation
- [ ] Benchmark: PAR(12) with varying orders
- [ ] Benchmark: Full scenario pipeline with PAR
- [ ] Benchmark: Memory usage with Criterion
- [ ] Compare against stationary AR baseline
- [ ] Profile with `cargo flamegraph` if overhead >10%

### Testing

- [ ] Run benchmarks on reference hardware
- [ ] Document performance characteristics
- [ ] Identify optimization opportunities if needed

### Documentation

- [ ] Benchmark results in `docs/performance/`
- [ ] Overhead analysis and recommendations
- [ ] Scaling characteristics (num_stages, period, AR order)

## Dependencies

- **Blocked by**: PAR-009 (scenario integration)
- **Blocks**: None (parallel with PAR-010)

## Estimated Effort

**2 story points** (1 day)
