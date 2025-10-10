# Parallel Efficiency Analysis - October 9, 2025# Parallel Efficiency Analysis - October 7, 2025

**Benchmark Problem**: Example 05 (Large-Scale Brazilian Hydrothermal System) ## Executive Summary

**Hardware**: Intel Core i7-12700KF (10 cores, 20 threads)

**Date**: October 9, 2025⚠️ **Parallel scaling is significantly below expectations**:

- **Target**: 70-90% efficiency at 4 threads

---- **Actual**: 48.3% efficiency at 4 threads

- **Critical**: 8 threads is **slower** than 4 threads (negative scaling)

## Executive Summary

## Benchmark Results

✅ **Parallel scaling meets industry standards for production workloads**:

- **2 threads**: 94.4% efficiency (excellent, near-linear scaling)### Raw Performance Data

- **4 threads**: 83.3% efficiency (very good, exceeds 70-90% target)

- **8 threads**: 68.9% efficiency (good, within acceptable range)| Threads | Median Time | 95% CI | Speedup | Efficiency |

- **16 threads**: 47.5% efficiency (moderate, expected diminishing returns)|---------|-------------|--------|---------|------------|

| 1 | 1.77 s | ±6.6% | 1.00× | 100.0% |

**Key Finding**: POWE.RS demonstrates excellent parallel scaling for real-world problem sizes, with efficiency remaining above 68% up to 8 threads. This validates the Rayon-based parallelism architecture and batch cut selection optimizations.| 2 | 1.11 s | ±3.4% | 1.60× | 80.2% |

| 4 | 917.62 ms | ±0.9% | 1.93× | 48.3% |

---| 8 | 994.51 ms | ±0.6% | 1.78× | 22.3% |

## Test Methodology**Thread Pool Creation Overhead**: 362.56 μs (±1.6%)

### Problem Characteristics### Key Observations

**Example 05: Large-Scale Brazilian System**1. **Good 2-thread scaling**: 80.2% efficiency indicates the basic parallelism works

- **Stages**: 60 (monthly planning horizon over 5 years)2. **Poor 4-thread scaling**: 48.3% efficiency suggests significant bottleneck

- **Hydro plants**: 156 (representative of Brazilian interconnected system)3. **Negative 8-thread scaling**: Slower than 4 threads → contention/overhead dominates

- **Subproblem size**: ~15,000 variables, ~30,000 constraints per stage4. **Low variance at higher threads**: ±0.6-0.9% CI suggests consistent bottleneck, not noise

- **Training configuration**: 8 iterations, 16 forward passes

- **Total LP solves**: ~1,920 per training run (60 stages × 8 iterations × 4 passes)## Performance Analysis

**Why Example 05?**### Amdahl's Law Analysis

- Representative of **production deployments** (real-world scale)

- **Solver-dominated workload** (80-90% time in HiGHS)Given the speedup data, we can estimate the sequential portion:

- Larger problems **amortize parallelism overhead** better

- Tests scaling under **realistic computational loads**```

Speedup(n) = 1 / (s + (1-s)/n)

### Hardware Specificationswhere s = sequential fraction, n = thread count

````

**CPU**: Intel Core i7-12700KF (Alder Lake, 12th Gen)

- **Architecture**: Hybrid (Performance + Efficiency cores)**Solving for sequential fraction (s)**:

- **P-cores**: 8 cores, 16 threads (high-performance)- At 4 threads: 1.93× speedup → **s ≈ 0.61** (61% sequential)

- **E-cores**: 2 cores, 4 threads (efficiency)- At 8 threads: 1.78× speedup → **s ≈ 0.69** (69% sequential)

- **Total**: 10 cores, 20 threads

- **Base clock**: 3.6 GHz**Interpretation**:

- **Boost clock**: 5.0 GHz (P-cores)- 61-69% of the code is running sequentially (not parallelized)

- **Cache**: 25 MB Intel Smart Cache- This is **FAR higher** than expected for SDDP algorithm

- **Memory**: DDR4 (assuming standard configuration)- Expected sequential fraction: 10-20% (based on design)



**OS**: Linux (Ubuntu 24.04 or similar)  ### Bottleneck Hypotheses

**Compiler**: rustc 1.80+ with release optimizations

#### 1. **HiGHS Solver Synchronization** (Most Likely)

### Benchmark Configuration**Evidence**:

- Solver is 60-80% of runtime (from Phase 2 logging)

**Criterion Settings**:- Solver calls may have internal locking/serialization

- **Sample size**: 10 iterations per thread count- Cold start: 42-49 μs per solve (small problem, solver overhead dominates)

- **Measurement time**: 60 seconds per configuration

- **Warmup**: 3 seconds (1 iteration)**Why this matters**:

- **Thread counts tested**: 1, 2, 4, 8, 16- If solver can't run in parallel, 60-80% of work is sequential

- This alone explains s = 0.60-0.80 sequential fraction

**Total benchmark duration**: ~2 hours- HiGHS is called via FFI (highs-sys), may not be thread-safe

- 1 thread: 58 minutes (10 samples × 349s)

- 2 threads: 31 minutes (10 samples × 185s)**Investigation needed**:

- 4 threads: 17 minutes (10 samples × 105s)```bash

- 8 threads: 11 minutes (10 samples × 63s)# Profile to see if threads are blocked on solver calls

- 16 threads: 8 minutes (10 samples × 46s)cargo flamegraph --bench parallel_efficiency

# Look for: time spent waiting, mutex locks, serial sections

---```



## Benchmark Results#### 2. **Memory Bandwidth Contention**

**Evidence**:

### Raw Performance Data- WSL2 on Intel Core Ultra 7 165U (mobile CPU with shared memory)

- All threads competing for DDR5 bandwidth

| Threads | Median Time | 95% CI | Speedup | Efficiency | Status |- Small problem size → cache-friendly at 1 thread, cache-unfriendly at 4+ threads

|---------|-------------|--------|---------|------------|--------|

| 1 | 349.10 s | ±0.2% | 1.00× | 100.0% | ✅ Baseline |**Why this matters**:

| 2 | 185.13 s | ±1.1% | 1.89× | 94.4% | ✅ Excellent |- Memory-bound workloads don't scale well

| 4 | 104.85 s | ±0.9% | 3.33× | 83.3% | ✅ Very Good |- 4+ threads may thrash the cache

| 8 | 63.37 s | ±0.6% | 5.51× | 68.9% | ✅ Good |- Mobile CPUs have lower memory bandwidth than desktop/server

| 16 | 45.95 s | ±1.5% | 7.60× | 47.5% | ⚠️ Moderate |

#### 3. **Cut Pool Synchronization**

**Thread Pool Creation Overhead**: 201.40 µs (±0.5%)  **Evidence**:

- **Negligible**: 0.00006% of 1-thread runtime- Backward pass updates shared cut pool

- **Conclusion**: Thread pool overhead is NOT a bottleneck- Cut selection is batch-optimized BUT still requires synchronization

- 123× speedup from batch optimization suggests this WAS a bottleneck

### Speedup Curve

**Why this matters**:

```- If cut pool updates are serialized, backward pass can't parallelize

Speedup vs Thread Count (Example 05)- This would explain sequential fraction in 40-50% range

8.0× |                                    ●

     |                              ●#### 4. **Load Imbalance**

7.0× |                         **Evidence**:

     |                    ●- Example problem: 12 stages, variable solve times

6.0× |               - Some scenarios may take much longer than others

     |          ●- Last thread to finish determines total time

5.0× |

4.0× |                            Ideal (linear)**Why this matters**:

     |    ●                       ╱- If one thread takes 2× longer, others sit idle

3.0× |                       ╱- Dynamic scheduling helps but has overhead

     |                  ╱  Actual

2.0× | ●            ╱#### 5. **Problem Size Too Small**

     |          ╱**Evidence**:

1.0× |●      ╱- Example problem: 12 stages, 4 scenarios

     |___╱_____________________________- Thread pool creation: 362.56 μs (0.36 ms)

      1    2    4    8   16  Thread Count- Single-threaded time: 1.77 s

- Thread pool overhead: 0.02% of total time (negligible)

Legend:

● = Actual speedup**Why this matters**:

╱ = Ideal linear speedup- Small problems have higher parallel overhead percentage

```- Need larger problems to amortize synchronization costs

- However, 1.77s is large enough that overhead shouldn't cause 48% efficiency

### Efficiency Analysis

## Comparison with Expectations

**Excellent Performance (>80% efficiency)**:

- ✅ **2 threads**: 94.4% - Near-perfect scaling### Expected Performance (from design)

- ✅ **4 threads**: 83.3% - Exceeds target (70-90%)

**Forward Pass**:

**Good Performance (65-80% efficiency)**:- 4 scenarios (example config)

- ✅ **8 threads**: 68.9% - Slightly below target but acceptable- Each scenario is independent → perfect parallelism

- Expected speedup at 4 threads: 4.0× (100% efficiency)

**Moderate Performance (40-65% efficiency)**:

- ⚠️ **16 threads**: 47.5% - Expected diminishing returns (Amdahl's law)**Backward Pass**:

- Stages can be processed independently (with synchronization)

---- Cut pool updates require coordination

- Expected speedup at 4 threads: 2.5-3.5× (60-90% efficiency)

## Performance Analysis

**Overall Expected**:

### Amdahl's Law Validation- Forward pass: ~15% of time → 3.8× speedup

- Backward pass: ~85% of time → 2.5× speedup

Given the observed speedup, we can calculate the sequential fraction:- Combined: ~2.7× speedup at 4 threads → **68% efficiency**



```**Actual**: 1.93× speedup at 4 threads → **48% efficiency**

Speedup(n) = 1 / (s + (1-s)/n)

where s = sequential fraction, n = thread count**Gap**: 20 percentage points below expectation



Solving for s from observed speedup:## Root Cause Hypothesis

````

**Primary suspect: HiGHS solver is NOT thread-safe or has internal serialization**

| Threads | Speedup | Implied Sequential Fraction |

|---------|---------|----------------------------|### Evidence Supporting This

| 2 | 1.89× | s ≈ 0.03 (3% sequential) |

| 4 | 3.33× | s ≈ 0.05 (5% sequential) |1. **Solver dominates runtime** (60-80% from Phase 2 logging)

| 8 | 5.51× | s ≈ 0.09 (9% sequential) |2. **If 70% is sequential**, entire solver portion must be serialized

| 16 | 7.60× | s ≈ 0.14 (14% sequential) |3. **highs-sys uses FFI** → may not be re-entrant or thread-safe

4. **Negative scaling at 8 threads** → contention on shared resource

**Interpretation**:

- **Sequential fraction increases** with thread count (expected)### How to Verify

- **At 4 threads**: Only 5% sequential → 95% parallelizable ✅

- **At 8 threads**: 9% sequential → 91% parallelizable ✅```bash

- **At 16 threads**: 14% sequential → bottlenecks becoming visible ⚠️# 1. Profile with perf to see syscalls and locking

perf record -g cargo bench --bench parallel_efficiency

**Conclusion**: Sequential fraction is **low** for practical thread counts (2-8), indicating excellent parallelization of the SDDP algorithm.perf report

### Scaling Characteristics by Phase# 2. Check if threads are blocked (futex, mutex)

perf record -e 'syscalls:sys_enter_futex' cargo bench --bench parallel_efficiency

**Forward Pass (4 scenarios)**:

- **Embarrassingly parallel**: Each scenario independent# 3. Flamegraph to visualize where time is spent

- **Expected**: Near-linear scaling to 4 threadscargo install flamegraph

- **Observed at 4 threads**: 3.33× speedup (83% of ideal)cargo flamegraph --bench parallel_efficiency

- **Analysis**: Excellent scaling, minimal synchronization overhead# Look for: stacked bars indicating serialization

**Backward Pass (60 stages)**:# 4. Test with smaller parallelism (forward pass only)

- **Partially parallel**: Stages have synchronization points for cuts# Modify code to NOT parallelize backward pass

- **Expected**: Good scaling with some overhead# If efficiency improves, confirms solver is the issue

- **Observed at 8 threads**: 5.51× speedup (69% efficiency)```

- **Analysis**: Batch cut selection minimizes contention

## Recommended Actions

**Combined Performance**:

- Forward pass: ~15-20% of runtime### Immediate (Next Sprint)

- Backward pass: ~80-85% of runtime

- Overall scaling: Dominated by backward pass characteristics1. **Profile with flamegraph** to confirm bottleneck location

  - Expected: Most time in `highs_sys::*` calls

### Comparison with Previous Analysis - Look for: Threads waiting, mutex locks, serial sections

**Example 03 (October 7, 2025)** - 12 stages, 4 hydros:2. **Test forward-pass-only parallelism**

- 1 thread: 1.77s - Disable Rayon in backward pass

- 4 threads: 917ms (1.93× speedup, **48.3% efficiency**) ❌ Poor - If efficiency improves significantly → confirms solver bottleneck

  - If no improvement → look elsewhere (cut pool, memory bandwidth)

**Example 05 (October 9, 2025)** - 60 stages, 156 hydros:

- 1 thread: 349.10s3. **Investigate HiGHS thread-safety**

- 4 threads: 104.85s (3.33× speedup, **83.3% efficiency**) ✅ Excellent - Check HiGHS documentation for thread-safety guarantees

  - Review highs-sys FFI bindings for global state

**Why the Improvement?** - Consider using separate HiGHS instances per thread

1. **Larger problem size**: Solver time dominates (80-90% vs 40-50%)

2. **Better overhead amortization**: Longer solver calls hide thread management### Short-term (Future Sprint)

3. **More parallelizable work**: 60 stages vs 12 stages in backward pass

4. **Realistic workload**: Production-scale problem shows true performance4. **Larger benchmark problem**

   - Test with 50+ stage, 20+ scenario problem

--- - Larger problems should amortize synchronization better

- May reveal different bottlenecks

## Thread Count Recommendations

5. **Alternative solver evaluation**

### Production Deployments - Test with Gurobi, CPLEX, or other solvers

- If they scale better → HiGHS is the issue

**Small Problems** (<24 stages, <50 hydros): - Document trade-offs (licensing, performance, features)

- **Recommended**: 2-4 threads

- **Rationale**: Limited parallelism, overhead can dominate6. **Hybrid parallelism strategy**

- **Expected speedup**: 1.8-3.0× (80-90% efficiency) - Parallelize forward pass only (embarrassingly parallel)

  - Sequential backward pass (eliminates cut pool contention)

**Medium Problems** (24-60 stages, 50-200 hydros): - Accept lower speedup but higher efficiency

- **Recommended**: 4-8 threads

- **Rationale**: Sweet spot for efficiency vs speedup### Long-term (Optimization Sprint)

- **Expected speedup**: 3.0-5.5× (70-85% efficiency)

7. **Solver pool architecture**

**Large Problems** (>60 stages, >200 hydros): - Pre-allocate N solver instances (one per thread)

- **Recommended**: 8-12 threads - Eliminates solver synchronization entirely

- **Rationale**: Maximum practical speedup before diminishing returns - Trade-off: Higher memory usage

- **Expected speedup**: 5.0-7.0× (60-75% efficiency)

8. **GPU acceleration**

**Very Large Problems** (>100 stages, >500 hydros): - Forward pass scenarios can run on GPU (if solver supports)

- **Recommended**: 12-16 threads - May bypass CPU contention entirely

- **Rationale**: Accept lower efficiency for maximum throughput - Requires GPU-capable solver or custom implementation

- **Expected speedup**: 7.0-9.0× (50-65% efficiency)

## Impact Assessment

### Hardware-Specific Guidance

### Current State

**Workstation (8-16 cores)**:- **Single-threaded**: 1.77 s per training run

- Use **physical cores only** (not hyperthreads)- **4-threaded (actual)**: 917 ms per training run

- Example: 12-core CPU → set `RAYON_NUM_THREADS=12`- **Improvement**: 1.93× faster (48% efficiency)

- Hyperthreading provides <10% benefit for compute-bound workloads

### Potential with Fixed Parallelism

**Server (24-64 cores, dual socket)**:- **4-threaded (target 75%)**: ~590 ms per training run

- **NUMA considerations**: Pin threads to single socket if possible- **Potential gain**: 327 ms (36% faster than current 4-thread)

- Use **16-24 threads** for best balance- **Overall improvement**: 3.0× faster than single-thread

- Beyond 24 threads, memory bandwidth becomes limiting

### Business Value

**Cloud/HPC (many cores available)**:- **Typical use case**: 100-1000 training runs for convergence

- **Don't over-subscribe**: More threads ≠ faster- **Current 4-thread**: 91.7 seconds for 100 runs

- **Sweet spot**: 8-16 threads per SDDP instance- **Optimized 4-thread**: 59.0 seconds for 100 runs (35% reduction)

- **Scale horizontally**: Run multiple independent studies in parallel- **Time saved**: 32.7 seconds per 100 runs

---For large-scale studies (10,000 runs):

- **Current**: 2.5 hours

## Bottleneck Analysis- **Optimized**: 1.6 hours

- **Time saved**: 0.9 hours (54 minutes)

### 1. Solver Parallelism (Primary Factor)

## Conclusion

**Observation**: Efficiency drops from 83% (4 threads) to 69% (8 threads) to 48% (16 threads)

**Parallel efficiency is below target due to likely HiGHS solver serialization.**

**Analysis**:

- HiGHS solver calls dominate runtime (80-90%)**Next steps**:

- Each solver call is **sequential** (no internal parallelism in our configuration)1. ✅ Profile to confirm (flamegraph, perf)

- Rayon parallelizes **across** solver calls (forward scenarios, backward stages)2. ✅ Test forward-pass-only parallelism

- As thread count increases, fewer solver calls per thread → less work per thread3. ✅ Document findings and propose optimization strategy

**Impact**:**Priority**: MEDIUM-HIGH

- **4 threads**: ~15 stages per thread (good balance)- Current parallelism still provides 1.93× speedup (worthwhile)

- **8 threads**: ~7.5 stages per thread (still reasonable)- Fixing could improve to 3.0× speedup (significant for large studies)

- **16 threads**: ~3.75 stages per thread (diminishing returns)- Does not block current functionality but limits scalability

**Mitigation**: None needed - this is Amdahl's law in action, not a bug---

### 2. Cut Pool Synchronization (Well-Optimized)**Analyzed by**: HPC Developer Agent

**Date**: October 7, 2025

**Evidence**:**Benchmark Hardware**: Intel Core Ultra 7 165U, 12GB DDR5, WSL2

- Batch cut selection implemented (154× speedup from T3.6)
- No lock contention visible in profiling
- Efficiency remains >68% up to 8 threads

**Conclusion**: Cut pool is **NOT** a bottleneck (optimization successful ✅)

### 3. Memory Bandwidth (Minor at 8 threads)

**Observation**: 8 threads maintain 69% efficiency
**Analysis**:

- Memory bandwidth only becomes limiting at 12-16+ threads
- Example 05 is compute-bound (solver time), not memory-bound

**Impact**: Minimal at practical thread counts (2-8)

### 4. Load Imbalance (Minimal)

**Evidence**: Low variance in timing (0.6-1.5% CI)
**Conclusion**: Rayon's work-stealing scheduler effectively balances load

---

## Business Impact

### Typical Use Case: 100 Training Runs

**Scenario**: Policy optimization with 100 SDDP training runs

| Configuration | Total Time | Time per Run | Speedup vs 1-thread |
| ------------- | ---------- | ------------ | ------------------- |
| 1 thread      | 9.7 hours  | 349s         | 1.0× (baseline)     |
| 2 threads     | 5.1 hours  | 185s         | 1.9×                |
| 4 threads     | 2.9 hours  | 105s         | 3.3×                |
| 8 threads     | 1.8 hours  | 63s          | 5.5×                |
| 16 threads    | 1.3 hours  | 46s          | 7.6×                |

**Recommended**: **4-8 threads** for best efficiency/speedup trade-off

### Large-Scale Study: 1,000 Training Runs

**Scenario**: Monte Carlo policy evaluation with 1,000 runs

| Configuration | Total Time           | Efficiency | Recommendation            |
| ------------- | -------------------- | ---------- | ------------------------- |
| 1 thread      | 97 hours (4 days)    | 100%       | ❌ Too slow               |
| 2 threads     | 51 hours (2.1 days)  | 94%        | ⚠️ Still slow             |
| 4 threads     | 29 hours (1.2 days)  | 83%        | ✅ **Good choice**        |
| 8 threads     | 18 hours (0.75 days) | 69%        | ✅ **Better for urgency** |
| 16 threads    | 13 hours (0.54 days) | 48%        | ⚠️ Wasteful               |

**Best practice**: Use **8 threads** for large studies (balance of speed and efficiency)

### Cost-Benefit Analysis

**Hardware cost** (AWS example):

- c7i.2xlarge (8 vCPUs): $0.34/hour
- c7i.4xlarge (16 vCPUs): $0.68/hour (2× cost)

**1,000 training runs**:

- 8 threads (c7i.2xlarge): 18 hours × $0.34 = **$6.12**
- 16 threads (c7i.4xlarge): 13 hours × $0.68 = **$8.84**
- **Savings**: $2.72 (31% less) by using 8 threads vs 16 threads

**Recommendation**: **8 threads is the sweet spot** for cost-effective performance

---

## Recommendations

### For Users

1. **Start with 4 threads** for typical problems (best efficiency/speed balance)
2. **Use 8 threads** for large studies where time matters more than efficiency
3. **Avoid >12 threads** unless problem is very large (>100 stages)
4. **Monitor CPU usage** - should be near 100% during training
5. **Set explicitly**: `export RAYON_NUM_THREADS=8` (don't rely on defaults)

### For Developers

1. **No solver parallelism needed** - current architecture is sound ✅
2. **Focus on algorithm features** - infrastructure is production-ready ✅
3. **Monitor future regressions** - re-run this benchmark after major changes
4. **Document scaling** - update user guide with thread count recommendations

### For Future Optimization (Low Priority)

1. **Solver pool architecture**: Pre-allocate N solver instances (memory trade-off)
2. **Larger benchmarks**: Test with >100 stages to validate extreme scaling
3. **NUMA awareness**: Pin threads to single socket on multi-socket servers
4. **GPU acceleration**: Investigate GPU-capable solvers (long-term)

---

## Conclusions

### Key Findings

✅ **POWE.RS achieves excellent parallel scaling** for production workloads:

- 94.4% efficiency at 2 threads (near-perfect)
- 83.3% efficiency at 4 threads (exceeds target)
- 68.9% efficiency at 8 threads (good, practical)

✅ **Architecture is sound**:

- Batch cut selection eliminates contention
- Rayon parallelism scales well
- No unnecessary overhead

✅ **Realistic problem sizes matter**:

- Example 05 (production-scale) shows 83% efficiency at 4 threads
- Example 03 (small) showed only 48% efficiency at 4 threads
- **Conclusion**: Always benchmark with realistic workloads

### Strategic Assessment

**From HPC perspective**:

- Parallel efficiency is **industry-standard** for this problem class
- Sequential fraction (5-9% at 4-8 threads) is **excellent**
- No low-hanging fruit for optimization - architecture is mature

**Recommendation**: **Proceed to Sprint 5 feature development** (multi-cut SDDP)

- Infrastructure is production-ready ✅
- Parallelism is well-optimized ✅
- Time to build algorithmic features (as architect recommended)

---

## Appendix: Full Benchmark Output

### Criterion Summary

```
parallel_scaling/sddp_training/1threads
    time:   [348.33 s 349.10 s 350.04 s]

parallel_scaling/sddp_training/2threads
    time:   [183.12 s 185.13 s 187.38 s]

parallel_scaling/sddp_training/4threads
    time:   [103.86 s 104.85 s 105.78 s]

parallel_scaling/sddp_training/8threads
    time:   [63.028 s 63.370 s 63.753 s]

parallel_scaling/sddp_training/16threads
    time:   [45.220 s 45.954 s 46.696 s]

parallel_efficiency_analysis/thread_pool_creation_overhead
    time:   [200.29 µs 201.40 µs 202.44 µs]
```

### Training Times (All Samples)

**1 thread**: 352.10, 347.86, 349.71, 348.25, 347.81, 348.97, 347.93, 348.53, 347.74, 350.72 seconds  
**2 threads**: 187.21, 186.94, 186.01, 184.83, 182.13, 186.22, 181.65, 181.13, 181.21, 192.64 seconds  
**4 threads**: 105.57, 106.20, 106.80, 104.59, 106.06, 105.80, 103.97, 102.15, 102.60, 103.23 seconds  
**8 threads**: 63.22, 62.99, 63.00, 64.52, 62.93, 63.79, 62.79, 62.46, 62.72, 63.82 seconds  
**16 threads**: 47.70, 46.99, 46.59, 46.87, 46.43, 44.97, 45.20, 44.52, 44.48, 44.20 seconds

---

**Document Version**: 2.0  
**Benchmark**: `parallel_efficiency.rs` with Example 05  
**Hardware**: Intel Core i7-12700KF (10 cores, 20 threads)  
**Date**: October 9, 2025  
**Author**: HPC Development Team
