# TICKET-007: Performance validation and benchmarking for backward pass optimization

## Context

This ticket validates that the backward pass optimization (TICKET-006) delivers the promised performance improvements. It includes profiling comparison, benchmark suite creation, and metrics collection. This is a critical validation step before moving to Phase 3 optimizations.

**Why this matters**: We need data-driven validation that our optimizations work. This ticket provides the evidence that malloc overhead decreased, execution time improved, and the optimization is worth keeping.

**Part of**: Performance Implementation Plan - Phase 2: Backward Pass Optimization

**Depends on**: TICKET-006 (backward pass refactoring must be complete)

## Acceptance Criteria

- [ ] Given profiling before/after comparison, when malloc overhead is measured, then it's reduced by >50% (from ~2% to <1%)
- [ ] Given benchmark comparison, when backward pass time is measured, then it's 8-10% faster
- [ ] Given criterion benchmarks, when run, then results show statistically significant improvement
- [ ] Given memory profiling, when backward pass executes, then zero allocations occur in hot path
- [ ] Given performance report, when created, then it includes concrete numbers and graphs
- [ ] All measurements are reproducible with documented methodology

## Tasks

### Implementation

- [ ] Create profiling script `scripts/profile_backward_pass.sh`:
  - [ ] Profile backward pass in isolation
  - [ ] Extract malloc overhead percentage
  - [ ] Generate flamegraph
  - [ ] Save results to timestamped directory
- [ ] Create benchmark suite `benches/backward_pass_bench.rs`:
  - [ ] Benchmark backward_pass() on 03-multistage
  - [ ] Benchmark backward_pass() on 05-large-scale-brazilian
  - [ ] Benchmark with different forward pass counts (10, 25, 50)
  - [ ] Use criterion for statistical analysis
- [ ] Create comparison script `scripts/compare_backward_pass_performance.sh`:
  - [ ] Run benchmarks on baseline (git main)
  - [ ] Run benchmarks on optimized branch
  - [ ] Generate comparison report
  - [ ] Output improvement percentages
- [ ] Create memory profiling script `scripts/profile_memory_backward_pass.sh`:
  - [ ] Use massif or custom allocator
  - [ ] Track allocations during backward pass
  - [ ] Verify zero allocations in hot path
  - [ ] Generate allocation timeline
- [ ] Create validation report template:
  - [ ] Performance metrics table
  - [ ] Profiling comparison
  - [ ] Benchmark results
  - [ ] Memory allocation analysis
  - [ ] Conclusion and recommendations

### Testing

- [ ] Sanity test: Verify profiling scripts work
  - [ ] Run on known baseline
  - [ ] Verify output format
  - [ ] Check for errors
- [ ] Sanity test: Verify benchmark scripts work
  - [ ] Run criterion benchmarks
  - [ ] Verify statistical analysis runs
  - [ ] Check output format
- [ ] Validation test: Reproducibility
  - [ ] Run profiling 3 times
  - [ ] Verify results are consistent (within 5%)
  - [ ] Document variance
- [ ] Validation test: Statistical significance
  - [ ] Run benchmarks with sufficient iterations
  - [ ] Verify p-value < 0.05 for improvement claim
  - [ ] Document confidence intervals

### Performance Validation Tasks

- [ ] Profile baseline (before optimization):
  - [ ] Checkout main branch
  - [ ] Build release with debug info: `CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release`
  - [ ] Run perf record on 05-large-scale-brazilian
  - [ ] Generate perf report
  - [ ] Save to `profiling_results/baseline_backward_pass/`
  - [ ] Extract malloc overhead percentage
- [ ] Profile optimized (after optimization):
  - [ ] Checkout optimization branch
  - [ ] Build release with debug info
  - [ ] Run perf record on same example
  - [ ] Generate perf report
  - [ ] Save to `profiling_results/optimized_backward_pass/`
  - [ ] Extract malloc overhead percentage
- [ ] Compare profiling results:
  - [ ] Calculate malloc overhead reduction
  - [ ] Calculate overall runtime improvement
  - [ ] Verify improvement meets target (>8%)
  - [ ] Document findings
- [ ] Run criterion benchmarks baseline:
  - [ ] Checkout main branch
  - [ ] Run: `cargo bench --bench backward_pass_bench -- --save-baseline before`
  - [ ] Save results
- [ ] Run criterion benchmarks optimized:
  - [ ] Checkout optimization branch
  - [ ] Run: `cargo bench --bench backward_pass_bench -- --baseline before`
  - [ ] Analyze improvement percentages
  - [ ] Verify statistical significance
- [ ] Memory profiling:
  - [ ] Run with massif on optimized version
  - [ ] Verify allocation count in backward pass loop
  - [ ] Compare with baseline
  - [ ] Document allocation elimination
- [ ] Create performance report:
  - [ ] Fill in metrics table
  - [ ] Add flamegraph comparisons
  - [ ] Add benchmark result graphs
  - [ ] Write conclusions
  - [ ] Make recommendations for Phase 3

### Documentation

- [ ] Create `BACKWARD_PASS_OPTIMIZATION_REPORT.md`:
  - [ ] Executive summary
  - [ ] Methodology section
  - [ ] Baseline measurements
  - [ ] Optimized measurements
  - [ ] Comparison and analysis
  - [ ] Conclusions
- [ ] Document profiling methodology:
  - [ ] Commands used
  - [ ] System configuration
  - [ ] Reproduction steps
- [ ] Document benchmarking methodology:
  - [ ] Benchmark setup
  - [ ] Iteration counts
  - [ ] Statistical analysis approach
- [ ] Update PERFORMANCE_REFACTORING_PLAN.md:
  - [ ] Mark Phase 2 as complete
  - [ ] Update metrics table with actual results
  - [ ] Add lessons learned
- [ ] Add results to CHANGELOG.md

## Technical Notes

### Profiling Methodology

**Tools**:
- `perf record` for CPU profiling
- `perf report` for analysis
- `flamegraph` for visualization
- `massif` for memory profiling

**Commands**:
```bash
# CPU profiling
CARGO_PROFILE_RELEASE_DEBUG=true cargo build --release
perf record --call-graph dwarf -F 999 ./target/release/powers examples/05-large-scale-brazilian
perf report --stdio > report.txt

# Extract malloc overhead
grep -E "(malloc|_int_malloc)" report.txt | awk '{sum+=$1} END {print sum "%"}'

# Flamegraph
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

**Memory profiling**:
```bash
valgrind --tool=massif --massif-out-file=massif.out ./target/release/powers examples/05-large-scale-brazilian
ms_print massif.out > massif_report.txt
```

### Benchmarking Methodology

**Criterion setup**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_backward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("backward_pass");
    
    for example in ["03-multistage", "05-large-scale-brazilian"].iter() {
        let mut sddp = load_instance(example).unwrap();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(example),
            example,
            |b, _| {
                b.iter(|| {
                    black_box(sddp.backward_pass().unwrap())
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_backward_pass);
criterion_main!(benches);
```

**Statistical Analysis**:
- Criterion automatically computes confidence intervals
- Look for non-overlapping intervals to claim significance
- Document mean, median, and standard deviation

### Metrics to Collect

**Profiling Metrics**:
- Total runtime (seconds)
- Malloc CPU overhead (%)
- Memset CPU overhead (%)
- Backward pass function time (seconds)
- HiGHS solver time (for reference)

**Benchmark Metrics**:
- Backward pass time (mean ± std dev)
- Improvement percentage (with confidence interval)
- Throughput (iterations per second)

**Memory Metrics**:
- Peak memory usage (MB)
- Allocation count in hot path
- Allocation count per iteration

### Expected Results

Based on PERFORMANCE_IMPLEMENTATION_PLAN.md:

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Runtime | 34.0s | ~31s | <32s |
| Malloc overhead | ~2% | <1% | <1% |
| Backward pass time | ~4.7s | ~4.0s | <4.3s |
| Allocations/iter | ~60 | ~1 | <5 |

### Validation Thresholds

**Pass Criteria**:
- ✅ Malloc overhead reduction >40%
- ✅ Backward pass time improvement >5%
- ✅ Overall runtime improvement >5%
- ✅ Zero allocations in backward pass loop (verified by profiler)

**Stretch Goals**:
- 🎯 Malloc overhead reduction >50%
- 🎯 Backward pass time improvement >8%
- 🎯 Overall runtime improvement >8%

### Report Template

```markdown
# Backward Pass Optimization Report

## Executive Summary

The backward pass optimization achieved:
- X% reduction in malloc overhead (from Y% to Z%)
- X% improvement in backward pass execution time
- X% improvement in overall training runtime
- Zero allocations verified in hot path

## Methodology

### System Configuration
- CPU: [details]
- RAM: [details]
- OS: [details]
- Rust: [version]

### Profiling Setup
[Commands and configuration]

### Benchmark Setup
[Criterion configuration]

## Results

### Profiling Comparison

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Runtime | ... | ... | ... |
| Malloc % | ... | ... | ... |

### Benchmark Results

[Criterion output graphs and tables]

### Memory Analysis

[Massif output and allocation counts]

## Conclusions

[Interpretation of results]

## Recommendations

[Next steps for Phase 3]
```

### Integration with CI/CD

Consider adding benchmark step to CI:
```yaml
- name: Run benchmarks
  run: cargo bench --bench backward_pass_bench -- --baseline ci-baseline
```

### References

- See `PERFORMANCE_IMPLEMENTATION_PLAN.md` Section 2.3
- Criterion documentation: https://bheisler.github.io/criterion.rs/
- Perf tutorial: https://perf.wiki.kernel.org/index.php/Tutorial

## Dependencies

- Blocked by: TICKET-006 (backward pass refactoring must be complete)
- Blocks: TICKET-008 (Phase 3 work depends on Phase 2 validation)
- Related: TICKET-004 (testing infrastructure)

## Estimated Effort

**3 story points** (2 days)

**Confidence**: High

**Breakdown**:
- Profiling: 0.5 day (run profiling, compare results)
- Benchmarking: 0.5 day (create benchmarks, run comparisons)
- Analysis: 0.5 day (interpret results, create graphs)
- Documentation: 0.5 day (write report, update plans)

## Validation Checklist

Before marking this ticket complete:

- [ ] Profiling scripts created and tested
- [ ] Benchmark suite created and runs successfully
- [ ] Baseline measurements collected
- [ ] Optimized measurements collected
- [ ] Malloc overhead reduced by >50%
- [ ] Backward pass time improved by >8%
- [ ] Zero allocations verified in hot path
- [ ] Performance report created with concrete numbers
- [ ] Results are reproducible (documented methodology)
- [ ] PERFORMANCE_REFACTORING_PLAN.md updated
- [ ] CHANGELOG.md updated
- [ ] Code reviewed by team member

## Notes

**Data Quality**: Ensure measurements are taken under consistent conditions:
- Same hardware
- Same system load (no other heavy processes)
- Multiple runs to account for variance
- Document any outliers

**Reporting**: Be honest about results. If targets aren't met, document why and adjust strategy for Phase 3.

**Celebration**: If targets are met, take a moment to appreciate the data-driven success! This is concrete evidence that the optimization works.
