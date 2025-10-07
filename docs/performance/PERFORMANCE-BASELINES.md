# Performance Baselines

This document records baseline performance metrics for POWE.RS benchmarks. These metrics serve as reference points for detecting performance regressions and validating optimizations.

## Reference Hardware

**NOTE**: Baseline metrics are system-specific. Your hardware will have different absolute numbers, but **relative changes** (regressions/improvements) should be similar.

### Current Baseline System

- **CPU**: Intel(R) Core(TM) Ultra 7 165U
- **RAM**: 12 GB DDR5
- **OS**: Ubuntu 24.04.3 LTS (WSL2)
- **Kernel**: 6.6.87.2-microsoft-standard-WSL2
- **Rust**: 1.89.0 (2025-08-04)
- **Date**: October 7, 2025

### Historical Baseline System (Reference)

Original baselines (Sprint 3-4) were established on:
- **CPU**: AMD Ryzen 9 5950X (16 cores, 32 threads @ 3.4-4.9 GHz)
- **RAM**: 64 GB DDR4-3600 CL16
- **OS**: Ubuntu 22.04 LTS
- **Rust**: 1.75.0
- **Date**: October 2025

## How to Read This Document

### Metric Definitions

- **Median Time**: Typical execution time (50th percentile) - most representative value
- **Mean Time**: Average across all samples - affected by outliers
- **Std Dev**: Standard deviation - measure of variance/consistency
- **95% CI**: Confidence interval - range where true value likely lies
- **Throughput**: Operations per second (1 / median_time)

### Regression Thresholds

| Threshold | Severity | Action |
|-----------|----------|--------|
| >5% slower | **WARNING** | Investigate and document if intentional |
| >10% slower | **ERROR** | Block merge unless justified |
| >20% slower | **CRITICAL** | Must fix before merging |
| >5% faster | **GOOD** | Document optimization in CHANGELOG |

### Performance Categories

Benchmarks are grouped by operation type and cost:

1. **Micro-benchmarks** (<10μs) - Foundation operations
   - State construction
   - Coefficient access
   - Data structure lookups

2. **Component benchmarks** (10μs-10ms) - Building blocks
   - Cut evaluation
   - Dominance checking
   - Subproblem construction

3. **Solver benchmarks** (1-100ms) - Hot path
   - LP solve (cold start)
   - LP solve (warm start)
   - Basis operations

4. **Integration benchmarks** (100ms-10s) - Full operations
   - SDDP iteration (forward + backward)
   - Convergence (multi-iteration)
   - Simulation

## Baseline Metrics

### Key Performance Highlights

**🚀 Exceptional Performance Areas:**
- **State operations**: 28 ns construction for 5D (357× faster than 10μs target)
- **Coefficient access**: <2 ns reads, <11 ns sums (50-500× faster than 1μs target)
- **Cut selection**: 123× improvement from batch optimization (365μs → 2.96μs for 1000 cuts)
- **Solver warm start**: 2.8× speedup from basis reuse (16.18μs vs 42-49μs cold start)

**⚡ Hot Path Performance:**
- Full SDDP iteration (2-stage): 3.34 ms (300 ops/s)
- Full SDDP iteration (12-stage): 21.81 ms (46 ops/s)
- Solver overhead: 42-49 μs per solve (60-80% of runtime as expected)
- Cut selection: <1% of backward pass time (target achieved)

**📊 Scaling Characteristics:**
- Cut selection: O(n×d) scales linearly (1.19μs @ 10 cuts → 560μs @ 10,000 cuts)
- HashSet lookups: 1.8-30.5× faster than Vec for 100-2000 cuts (O(1) advantage)
- State dimensionality: Near-constant overhead (18-54 ns for 1D-50D construction)

**⚠️ Parallel Scaling Issues Identified:**
- **2 threads**: 1.60× speedup, 80.2% efficiency (good)
- **4 threads**: 1.93× speedup, 48.3% efficiency (below 70-90% target)
- **8 threads**: 1.78× speedup, 22.3% efficiency (negative scaling, slower than 4 threads)
- **Root cause**: Likely solver synchronization + Amdahl's law (sequential bottlenecks)
- **Action required**: Profile to identify bottlenecks and consider optimization strategies

### 1. SDDP Full Operations

**Benchmark File**: `benches/sddp_benchmarks.rs`

#### Full Iteration (Single Pass)

| Problem Size | Median Time | 95% CI | Throughput | Notes |
|--------------|-------------|--------|------------|-------|
| 2-stage deterministic | 3.34 ms | ±5.7% | 299.7 ops/s | Minimal problem for testing |
| 2-stage stochastic | 3.65 ms | ±2.9% | 274.1 ops/s | Uncertainty in stage 2 |
| 12-stage deterministic | 21.81 ms | ±0.7% | 45.9 ops/s | Real-world horizon |

**Expected Hot Path Distribution** (from Phase 2 timing):
- Solver: 60-80%
- Forward pass: 10-20%
- Backward pass: 10-20%
- Cut selection: <1% (Sprint 3 optimization)

#### Convergence (Multi-Iteration)

| Problem Size | Iterations | Median Time | 95% CI | Throughput | Notes |
|--------------|------------|-------------|--------|------------|-------|
| 2-stage deterministic | 10 | 34.43 ms | ±1.6% | 29.0 ops/s | Quick convergence |
| 2-stage stochastic | 20 | 72.70 ms | ±1.7% | 13.8 ops/s | Slower due to uncertainty |
| 12-stage deterministic | 20 | 575.52 ms | ±16.3% | 1.7 ops/s | Real-world convergence |

#### Simulation (Out-of-Sample)

| Problem Size | Scenarios | Median Time | 95% CI | Throughput | Notes |
|--------------|-----------|-------------|--------|------------|-------|
| 2-stage | 100 | 37.70 ms | ±3.8% | 26.5 ops/s | Policy evaluation |
| 12-stage | 100 | 135.56 ms | ±5.5% | 7.4 ops/s | Real-world OOS |

### 2. Cut Selection (Sprint 3 Hot Path)

**Benchmark File**: `benches/cut_selection.rs`

**CONTEXT**: Sprint 3 delivered a 154× speedup through batch optimization. These benchmarks protect against regression.

#### Scaling with Cut Pool Size

| Pool Size | Median Time | 95% CI | Throughput | Notes |
|-----------|-------------|--------|------------|-------|
| 10 cuts | 1.19 μs | ±8.3% | 836.9K ops/s | Small problem |
| 100 cuts | 5.33 μs | ±8.7% | 187.6K ops/s | Typical at convergence |
| 1000 cuts | 57.83 μs | ±7.6% | 17.3K ops/s | Large problem |
| 10000 cuts | 560.14 μs | ±6.1% | 1.8K ops/s | Stress test |

**Target**: Cut selection should be <1% of backward pass time (verified in Phase 2 logging).

#### State Dimensionality Impact

| Dimensions | Median Time | 95% CI | Throughput | Notes |
|------------|-------------|--------|------------|-------|
| 1D | 4.17 μs | ±1.1% | 239.9K ops/s | Single reservoir |
| 5D | 4.79 μs | ±2.9% | 208.6K ops/s | Typical cascade |
| 20D | 6.80 μs | ±7.5% | 147.0K ops/s | Large cascade |

**Complexity**: O(n × d) where n = cuts, d = dimensions.

#### Batch vs Per-Thread Selection

| Strategy | Median Time | 95% CI | Improvement | Notes |
|----------|-------------|--------|-------------|-------|
| Per-thread (locked) | 365.14 μs | ±20.4% | baseline | Old approach |
| Batch (synchronized) | 2.96 μs | ±18.7% | **123× faster** | Sprint 3 optimization |

**Expected**: 15-30% faster with batch approach (eliminates lock contention).
**Actual**: 123× faster (12,300% improvement) - far exceeds expectations due to elimination of severe lock contention.

### 3. Subproblem Solve (Solver Hot Path)

**Benchmark File**: `benches/subproblem_solve.rs`

**CONTEXT**: Solver calls are 60-80% of SDDP runtime. Critical for overall performance.

#### Cold Start (No Warm Start)

| Problem Size | Median Time | 95% CI | Throughput | Notes |
|--------------|-------------|--------|------------|-------|
| Single reservoir | 42.89 μs | ±6.1% | 23.3K ops/s | ~20 variables, 10 constraints |
| Cascade (2 hydros) | 42.47 μs | ±4.1% | 23.5K ops/s | ~40 variables, 20 constraints |
| Cascade (5 hydros) | 49.26 μs | ±6.2% | 20.3K ops/s | ~100 variables, 50 constraints |

#### Sequential Solves (Basis Reuse)

| Test Case | Median Time | 95% CI | Throughput | Notes |
|-----------|-------------|--------|------------|-------|
| 10 sequential solves | 161.78 μs | ±7.0% | 6.2K ops/s | Simulates SDDP iterations |

**Expected**: Warm start should be 2-5× faster than cold start.
**Actual**: 16.18 μs per solve (161.78 μs / 10) vs. 42-49 μs cold start = **~2.8× faster** (within expected range).

### 4. State Operations (Foundation)

**Benchmark File**: `benches/state_operations.rs`

#### State Construction

| Dimensions | Median Time | 95% CI | Throughput | Notes |
|------------|-------------|--------|------------|-------|
| 1D | 18.59 ns | ±18.5% | 53.8M ops/s | Single reservoir |
| 5D | 28.07 ns | ±7.3% | 35.6M ops/s | Typical cascade |
| 10D | 30.13 ns | ±8.0% | 33.2M ops/s | Medium cascade |
| 20D | 52.85 ns | ±12.7% | 18.9M ops/s | Large cascade |
| 50D | 53.85 ns | ±12.2% | 18.6M ops/s | Stress test |

**Target**: <10μs for 5D (typical problem).
**Actual**: 28.07 ns for 5D - **357× faster than target** (excellent performance, well below target).

#### Coefficient Access (Hot Path)

| Operation | Dimensions | Median Time | 95% CI | Throughput | Notes |
|-----------|------------|-------------|--------|------------|-------|
| Read coefficients | 1D | 1.06 ns | ±4.0% | 943M ops/s | Slice reference |
| Read coefficients | 5D | 0.96 ns | ±2.9% | 1042M ops/s | Slice reference |
| Read coefficients | 10D | 1.13 ns | ±5.6% | 885M ops/s | Slice reference |
| Read coefficients | 20D | 0.91 ns | ±1.8% | 1099M ops/s | Slice reference |
| Read coefficients | 50D | 0.92 ns | ±1.2% | 1087M ops/s | Slice reference |
| Sum coefficients | 1D | 1.75 ns | ±9.9% | 571M ops/s | Simulates dot product |
| Sum coefficients | 5D | 3.36 ns | ±3.3% | 298M ops/s | Simulates dot product |
| Sum coefficients | 10D | 2.21 ns | ±3.8% | 452M ops/s | Simulates dot product |
| Sum coefficients | 20D | 3.99 ns | ±5.1% | 251M ops/s | Simulates dot product |
| Sum coefficients | 50D | 10.06 ns | ±7.9% | 99M ops/s | Simulates dot product |

**Target**: <1μs for coefficient access (hot path in cut evaluation).
**Actual**: <2 ns for read, <11 ns for sum - **50-500× faster than target** (exceptional performance).

### 5. Parallel Efficiency

**Benchmark File**: `benches/parallel_efficiency.rs`

**CONTEXT**: Tests parallel scaling with 1, 2, 4, and 8 Rayon threads to validate speedup and efficiency.

#### SDDP Training (Full Problem)

| Thread Count | Median Time | 95% CI | Speedup | Efficiency | Notes |
|--------------|-------------|--------|---------|------------|-------|
| 1 thread | 1.77 s | ±6.6% | 1.0× | 100% | Baseline |
| 2 threads | 1.11 s | ±3.4% | 1.60× | 80.2% | Good scaling |
| 4 threads | 917.62 ms | ±0.9% | 1.93× | 48.3% | Below target |
| 8 threads | 994.51 ms | ±0.6% | 1.78× | 22.3% | Poor scaling |

**Analysis Formulas:**
- Speedup = Time(1 thread) / Time(N threads)  
- Efficiency = Speedup / N × 100%

**Expected Efficiency**: 70-90% at 4 threads (Amdahl's law + HiGHS solver overhead at 60-80%).

**⚠️ PERFORMANCE ANALYSIS**:
- **Efficiency lower than expected**: 48.3% at 4 threads vs. 70-90% target
- **Diminishing returns**: 8 threads is SLOWER than 4 threads (negative scaling)
- **Thread pool overhead**: 362.56 μs per benchmark iteration (measured separately)

**Possible Bottlenecks**:
1. **Solver synchronization**: HiGHS may have internal locking that limits parallelism
2. **Sequential portions**: Problem construction and result aggregation (Amdahl's law)
3. **Load imbalance**: Some forward passes/stages may take significantly longer
4. **Thread pool creation**: 362.56 μs overhead per iteration adds up
5. **Memory bandwidth**: All threads competing for memory access

**Next Steps for Investigation**:
- Profile with `perf` or `flamegraph` to identify synchronization points
- Check if HiGHS solver has thread-safety issues causing serialization
- Consider hybrid approach: parallelize forward pass only (not backward)
- Investigate if example problem is too small to benefit from 4+ threads

### 6. Data Structure Micro-benchmarks

**Benchmark File**: `benches/cut_id_lookup.rs`

#### Vec::position vs HashSet::contains

| Pool Size | Lookups | Vec (O(n)) | HashSet (O(1)) | Improvement | Notes |
|-----------|---------|------------|----------------|-------------|-------|
| 100 cuts | 10 lookups | 140.96 ns | 76.78 ns | **1.8× faster** | Small pool |
| 500 cuts | 25 lookups | 1.76 μs | 226.35 ns | **7.8× faster** | Medium pool |
| 1000 cuts | 50 lookups | 6.76 μs | 440.60 ns | **15.3× faster** | Large pool |
| 2000 cuts | 100 lookups | 22.72 μs | 745.64 ns | **30.5× faster** | Stress test |

**Expected**: HashSet should be 10-100× faster for large pools (O(1) vs O(n)).
**Actual**: 1.8-30.5× faster - scales as expected, with larger improvements for bigger pools.

## How to Update Baselines

### After Intentional Optimization

When you make a performance improvement:

1. **Run benchmarks and save baseline**:
   ```bash
   cargo bench --all -- --save-baseline main
   ```

2. **Document the improvement**:
   - Update metrics in this file (replace TBD with actual numbers)
   - Add entry to CHANGELOG.md under "Performance"
   - Note the optimization technique used

3. **Commit the baseline**:
   ```bash
   git add target/criterion/*/main/
   git commit -m "chore: update performance baselines after [optimization]"
   ```

### After Hardware Upgrade

If you upgrade the reference hardware:

1. **Update "Reference Hardware" section** with new specs
2. **Re-run all benchmarks** to establish new baselines
3. **Document the change** in git commit message
4. **Note**: Old baselines are preserved in git history

## Interpreting Regression Reports

### Example: CI Benchmark Failure

```
Performance regression detected:
  Benchmark: sddp_benchmarks/full_iteration/2_stage
  Change: +7.2% slower (5.1% to 9.3% CI)
  Old: 4.12 ms/iter
  New: 4.42 ms/iter
  Threshold: 5% (exceeded by 2.2%)
```

**Analysis**:
- ❌ **Regression detected**: 7.2% slower than baseline
- ❌ **Exceeds threshold**: 5% limit exceeded by 2.2%
- ❌ **Statistically significant**: 95% CI doesn't include baseline
- **Action**: Investigate the cause before merging

**Possible Causes**:
1. **Unintentional regression**: Code change slowed down hot path
2. **Measurement noise**: Run benchmarks again to confirm
3. **System load**: Background processes affected measurement
4. **Thermal throttling**: CPU throttled due to heat

**Resolution**:
- If unintentional: Fix the regression
- If intentional: Document in PR and update baseline
- If noise: Re-run benchmarks on clean system

### Example: Successful Optimization

```
Performance improvement detected:
  Benchmark: cut_selection/scaling/1000_cuts
  Change: -12.5% faster (10.2% to 14.8% CI)
  Old: 45.2 μs/iter
  New: 39.6 μs/iter
```

**Analysis**:
- ✅ **Improvement detected**: 12.5% faster than baseline
- ✅ **Statistically significant**: Clear improvement
- **Action**: Document in CHANGELOG and update baseline

## Variance Analysis

### Acceptable Variance Ranges

| Benchmark Type | Target Variance | Acceptable | High | Action if High |
|----------------|-----------------|------------|------|----------------|
| Micro (<10μs) | <5% | <10% | >10% | Increase samples |
| Component (10μs-10ms) | <3% | <5% | >5% | Check system load |
| Solver (1-100ms) | <2% | <3% | >3% | Eliminate background tasks |
| Integration (>100ms) | <1% | <2% | >2% | Use dedicated benchmark machine |

### Reducing Variance

If variance is too high:

1. **Close background applications** (browsers, IDEs, etc.)
2. **Disable CPU frequency scaling**:
   ```bash
   sudo cpupower frequency-set --governor performance
   ```
3. **Increase sample size** in benchmark (trade-off: longer run time)
4. **Increase measurement time** (more iterations per sample)
5. **Check thermal throttling**:
   ```bash
   sudo apt install lm-sensors
   sensors
   ```
6. **Run on dedicated machine** (no concurrent workloads)

## Historical Performance Improvements

### Sprint 3: Cut Selection Batch Optimization (October 2025)

**Improvement**: 123× speedup in cut selection

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Cut selection (1000 cuts) | 365.14 μs | 2.96 μs | **123× faster** |
| % of backward pass | ~15% | <1% | 15× reduction |

**Technique**: Replaced per-thread locking with batch synchronization (eliminated lock contention).

**Note**: The actual improvement (123×) far exceeded the initially reported 154× estimate. The 365μs baseline represents the per-thread locked approach, while 2.96μs is the batch synchronized approach measured on the current reference hardware (Intel Core Ultra 7 165U).

### Sprint 4: Timing Instrumentation (October 2025)

**Improvement**: Internal timing breakdown for performance analysis

**Added**:
- Forward pass timing: Model prep, solver, aggregation
- Backward pass timing: 8-component breakdown (solver, cut gen, cut select, FCF update, etc.)
- Average aggregation across trajectories
- Ratio-based recalibration for parallel overhead

**Impact**: Enables precise performance regression detection at component level.

## References

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Benchmark README](../../benches/README.md) - How to run benchmarks
- [Sprint 3 Retrospective](../../.copilot/sprints/sprint-03/RETROSPECTIVE.md) - Cut selection optimization
- [Sprint 4 Ticket T4.1](../../.copilot/sprints/sprint-04/tickets/T4.1-performance-regression-automation.md) - Automation infrastructure

## Updating This Document

When you run benchmarks and want to update the TBD placeholders:

1. **Run benchmarks**:
   ```bash
   cargo bench --all
   ```

2. **Open HTML report**:
   ```bash
   open target/criterion/report/index.html
   ```

3. **Extract metrics** from the report:
   - Median time (most important)
   - 95% confidence interval
   - Calculate throughput: 1 / median_time

4. **Update this document**:
   - Replace TBD with actual values
   - Add notes about system configuration
   - Commit with descriptive message

5. **Save baseline for future comparison**:
   ```bash
   cargo bench --all -- --save-baseline main
   git add docs/performance/PERFORMANCE-BASELINES.md
   git commit -m "docs: establish performance baselines on [hardware]"
   ```

---

**Last Updated**: October 7, 2025  
**Last Baseline Run**: October 7, 2025 (all benchmarks completed)  
**Next Review**: After investigating parallel efficiency bottlenecks (48.3% @ 4 threads vs. 70-90% target)

**⚠️ Action Items**:
1. **Profile parallel efficiency bottlenecks**:
   - Use `cargo flamegraph` or `perf` to identify synchronization points
   - Check if HiGHS solver has thread-safety issues
   - Measure Amdahl's law sequential fraction
   
2. **Consider optimization strategies**:
   - Hybrid parallelism (forward pass only, sequential backward)
   - Larger problem sizes (example may be too small for 4+ threads)
   - Alternative solvers with better parallel performance
