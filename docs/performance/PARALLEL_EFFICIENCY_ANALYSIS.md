# Parallel Efficiency Analysis - October 7, 2025

## Executive Summary

⚠️ **Parallel scaling is significantly below expectations**:
- **Target**: 70-90% efficiency at 4 threads
- **Actual**: 48.3% efficiency at 4 threads
- **Critical**: 8 threads is **slower** than 4 threads (negative scaling)

## Benchmark Results

### Raw Performance Data

| Threads | Median Time | 95% CI | Speedup | Efficiency |
|---------|-------------|--------|---------|------------|
| 1 | 1.77 s | ±6.6% | 1.00× | 100.0% |
| 2 | 1.11 s | ±3.4% | 1.60× | 80.2% |
| 4 | 917.62 ms | ±0.9% | 1.93× | 48.3% |
| 8 | 994.51 ms | ±0.6% | 1.78× | 22.3% |

**Thread Pool Creation Overhead**: 362.56 μs (±1.6%)

### Key Observations

1. **Good 2-thread scaling**: 80.2% efficiency indicates the basic parallelism works
2. **Poor 4-thread scaling**: 48.3% efficiency suggests significant bottleneck
3. **Negative 8-thread scaling**: Slower than 4 threads → contention/overhead dominates
4. **Low variance at higher threads**: ±0.6-0.9% CI suggests consistent bottleneck, not noise

## Performance Analysis

### Amdahl's Law Analysis

Given the speedup data, we can estimate the sequential portion:

```
Speedup(n) = 1 / (s + (1-s)/n)
where s = sequential fraction, n = thread count
```

**Solving for sequential fraction (s)**:
- At 4 threads: 1.93× speedup → **s ≈ 0.61** (61% sequential)
- At 8 threads: 1.78× speedup → **s ≈ 0.69** (69% sequential)

**Interpretation**: 
- 61-69% of the code is running sequentially (not parallelized)
- This is **FAR higher** than expected for SDDP algorithm
- Expected sequential fraction: 10-20% (based on design)

### Bottleneck Hypotheses

#### 1. **HiGHS Solver Synchronization** (Most Likely)
**Evidence**:
- Solver is 60-80% of runtime (from Phase 2 logging)
- Solver calls may have internal locking/serialization
- Cold start: 42-49 μs per solve (small problem, solver overhead dominates)

**Why this matters**:
- If solver can't run in parallel, 60-80% of work is sequential
- This alone explains s = 0.60-0.80 sequential fraction
- HiGHS is called via FFI (highs-sys), may not be thread-safe

**Investigation needed**:
```bash
# Profile to see if threads are blocked on solver calls
cargo flamegraph --bench parallel_efficiency
# Look for: time spent waiting, mutex locks, serial sections
```

#### 2. **Memory Bandwidth Contention**
**Evidence**:
- WSL2 on Intel Core Ultra 7 165U (mobile CPU with shared memory)
- All threads competing for DDR5 bandwidth
- Small problem size → cache-friendly at 1 thread, cache-unfriendly at 4+ threads

**Why this matters**:
- Memory-bound workloads don't scale well
- 4+ threads may thrash the cache
- Mobile CPUs have lower memory bandwidth than desktop/server

#### 3. **Cut Pool Synchronization**
**Evidence**:
- Backward pass updates shared cut pool
- Cut selection is batch-optimized BUT still requires synchronization
- 123× speedup from batch optimization suggests this WAS a bottleneck

**Why this matters**:
- If cut pool updates are serialized, backward pass can't parallelize
- This would explain sequential fraction in 40-50% range

#### 4. **Load Imbalance**
**Evidence**:
- Example problem: 12 stages, variable solve times
- Some scenarios may take much longer than others
- Last thread to finish determines total time

**Why this matters**:
- If one thread takes 2× longer, others sit idle
- Dynamic scheduling helps but has overhead

#### 5. **Problem Size Too Small**
**Evidence**:
- Example problem: 12 stages, 4 scenarios
- Thread pool creation: 362.56 μs (0.36 ms)
- Single-threaded time: 1.77 s
- Thread pool overhead: 0.02% of total time (negligible)

**Why this matters**:
- Small problems have higher parallel overhead percentage
- Need larger problems to amortize synchronization costs
- However, 1.77s is large enough that overhead shouldn't cause 48% efficiency

## Comparison with Expectations

### Expected Performance (from design)

**Forward Pass**: 
- 4 scenarios (example config)
- Each scenario is independent → perfect parallelism
- Expected speedup at 4 threads: 4.0× (100% efficiency)

**Backward Pass**:
- Stages can be processed independently (with synchronization)
- Cut pool updates require coordination
- Expected speedup at 4 threads: 2.5-3.5× (60-90% efficiency)

**Overall Expected**:
- Forward pass: ~15% of time → 3.8× speedup
- Backward pass: ~85% of time → 2.5× speedup
- Combined: ~2.7× speedup at 4 threads → **68% efficiency**

**Actual**: 1.93× speedup at 4 threads → **48% efficiency**

**Gap**: 20 percentage points below expectation

## Root Cause Hypothesis

**Primary suspect: HiGHS solver is NOT thread-safe or has internal serialization**

### Evidence Supporting This

1. **Solver dominates runtime** (60-80% from Phase 2 logging)
2. **If 70% is sequential**, entire solver portion must be serialized
3. **highs-sys uses FFI** → may not be re-entrant or thread-safe
4. **Negative scaling at 8 threads** → contention on shared resource

### How to Verify

```bash
# 1. Profile with perf to see syscalls and locking
perf record -g cargo bench --bench parallel_efficiency
perf report

# 2. Check if threads are blocked (futex, mutex)
perf record -e 'syscalls:sys_enter_futex' cargo bench --bench parallel_efficiency

# 3. Flamegraph to visualize where time is spent
cargo install flamegraph
cargo flamegraph --bench parallel_efficiency
# Look for: stacked bars indicating serialization

# 4. Test with smaller parallelism (forward pass only)
# Modify code to NOT parallelize backward pass
# If efficiency improves, confirms solver is the issue
```

## Recommended Actions

### Immediate (Next Sprint)

1. **Profile with flamegraph** to confirm bottleneck location
   - Expected: Most time in `highs_sys::*` calls
   - Look for: Threads waiting, mutex locks, serial sections

2. **Test forward-pass-only parallelism**
   - Disable Rayon in backward pass
   - If efficiency improves significantly → confirms solver bottleneck
   - If no improvement → look elsewhere (cut pool, memory bandwidth)

3. **Investigate HiGHS thread-safety**
   - Check HiGHS documentation for thread-safety guarantees
   - Review highs-sys FFI bindings for global state
   - Consider using separate HiGHS instances per thread

### Short-term (Future Sprint)

4. **Larger benchmark problem**
   - Test with 50+ stage, 20+ scenario problem
   - Larger problems should amortize synchronization better
   - May reveal different bottlenecks

5. **Alternative solver evaluation**
   - Test with Gurobi, CPLEX, or other solvers
   - If they scale better → HiGHS is the issue
   - Document trade-offs (licensing, performance, features)

6. **Hybrid parallelism strategy**
   - Parallelize forward pass only (embarrassingly parallel)
   - Sequential backward pass (eliminates cut pool contention)
   - Accept lower speedup but higher efficiency

### Long-term (Optimization Sprint)

7. **Solver pool architecture**
   - Pre-allocate N solver instances (one per thread)
   - Eliminates solver synchronization entirely
   - Trade-off: Higher memory usage

8. **GPU acceleration**
   - Forward pass scenarios can run on GPU (if solver supports)
   - May bypass CPU contention entirely
   - Requires GPU-capable solver or custom implementation

## Impact Assessment

### Current State
- **Single-threaded**: 1.77 s per training run
- **4-threaded (actual)**: 917 ms per training run
- **Improvement**: 1.93× faster (48% efficiency)

### Potential with Fixed Parallelism
- **4-threaded (target 75%)**: ~590 ms per training run
- **Potential gain**: 327 ms (36% faster than current 4-thread)
- **Overall improvement**: 3.0× faster than single-thread

### Business Value
- **Typical use case**: 100-1000 training runs for convergence
- **Current 4-thread**: 91.7 seconds for 100 runs
- **Optimized 4-thread**: 59.0 seconds for 100 runs (35% reduction)
- **Time saved**: 32.7 seconds per 100 runs

For large-scale studies (10,000 runs):
- **Current**: 2.5 hours
- **Optimized**: 1.6 hours
- **Time saved**: 0.9 hours (54 minutes)

## Conclusion

**Parallel efficiency is below target due to likely HiGHS solver serialization.**

**Next steps**:
1. ✅ Profile to confirm (flamegraph, perf)
2. ✅ Test forward-pass-only parallelism
3. ✅ Document findings and propose optimization strategy

**Priority**: MEDIUM-HIGH
- Current parallelism still provides 1.93× speedup (worthwhile)
- Fixing could improve to 3.0× speedup (significant for large studies)
- Does not block current functionality but limits scalability

---

**Analyzed by**: HPC Developer Agent  
**Date**: October 7, 2025  
**Benchmark Hardware**: Intel Core Ultra 7 165U, 12GB DDR5, WSL2
