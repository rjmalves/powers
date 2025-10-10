# Performance Tuning Guide

**Last Updated**: October 9, 2025  
**Audience**: POWE.RS users optimizing for production deployments  
**Prerequisites**: Basic understanding of SDDP algorithm and hydrothermal dispatch

---

## Table of Contents

1. [Overview](#overview)
2. [Thread Configuration](#thread-configuration)
3. [Solver Tuning](#solver-tuning)
4. [Memory Management](#memory-management)
5. [Problem-Specific Optimizations](#problem-specific-optimizations)
6. [Benchmarking and Validation](#benchmarking-and-validation)
7. [Troubleshooting](#troubleshooting)
8. [Quick Reference](#quick-reference)

---

## Overview

POWE.RS is designed for **high performance out-of-the-box**, with optimizations including:

- ✅ **Batch cut selection**: 154× faster than naive implementation (Sprint 3 optimization)
- ✅ **Basis warm-starting**: 30-50% solver speedup between stages
- ✅ **Rayon parallelism**: Near-linear scaling for forward passes
- ✅ **Zero-copy solver interface**: Direct FFI to HiGHS (no wrapper overhead)
- ✅ **Efficient memory usage**: Minimal allocations, model reuse

**When to Tune**: Production deployments, large problems (>60 stages), resource-constrained environments, or time-critical applications.

**Expected Gains from Tuning**:

- **Thread optimization**: 2-8× speedup (1 thread → 4-8 threads)
- **Problem formulation**: 2-5× speedup (state space reduction, if applicable)
- **Solver tuning**: 10-20% speedup (tolerance relaxation, problem-dependent)
- **Memory optimization**: Minimal gains (already efficient)

**Performance Baseline** (Intel Core i7-12700KF, Example 05):

- **1 thread**: 349 seconds (baseline)
- **4 threads**: 105 seconds (3.33× speedup, 83% efficiency)
- **8 threads**: 63 seconds (5.51× speedup, 69% efficiency)

**Key Principle**: **Profile before and after** - use benchmarks to validate tuning decisions.

---

## Thread Configuration

### Understanding POWE.RS Parallelism

POWE.RS uses **Rayon** for shared-memory parallelism with two main parallel sections:

1. **Forward Pass** (Embarrassingly Parallel):

   - Evaluates multiple scenarios independently
   - Near-linear scaling (94% efficiency at 2 threads)
   - No synchronization between scenarios

2. **Backward Pass** (Stage-Wise Parallel):

   - Solves subproblems for each stage in parallel
   - Synchronization for cut aggregation
   - Good scaling (83% efficiency at 4 threads)

3. **Cut Selection** (Batched):
   - Dominance checking parallelized
   - Minimal lock contention (batch synchronization)
   - Negligible overhead (<1% of backward pass time)

### Setting Thread Count

#### Using Environment Variable

```bash
# Set before running POWE.RS
export RAYON_NUM_THREADS=8
cargo run --release -- examples/05-large-scale-brazilian

# Or inline
RAYON_NUM_THREADS=4 ./target/release/powers examples/04-cascade
```

#### Using Configuration File

```json
{
  "num_iterations": 100,
  "num_forward_passes": 10,
  "seed": 42,
  "num_threads": 8
}
```

**Note**: Configuration file takes precedence over environment variable. If omitted or `null`, auto-detects available cores.

#### Using SddpInstanceBuilder (Programmatic)

```rust
use powers_rs::sddp::SddpInstanceBuilder;

let mut sddp = SddpInstanceBuilder::from_paths(
    "examples/05-large-scale-brazilian/config.json",
    "examples/05-large-scale-brazilian/system.json",
    "examples/05-large-scale-brazilian/graph.json",
    "examples/05-large-scale-brazilian/recourse.json",
)?
.with_num_threads(8)  // Override config
.build()?;

sddp.train()?;
```

### Recommendations by Hardware

#### Single-Socket Workstation (4-16 cores)

**Recommended**: Use **physical cores only** (not hyperthreads)

| CPU Cores | Recommended Threads | Rationale               |
| --------- | ------------------- | ----------------------- |
| 4 cores   | 4 threads           | Full utilization        |
| 6 cores   | 4-6 threads         | Optimal balance         |
| 8 cores   | 4-8 threads         | Good efficiency         |
| 12 cores  | 8-12 threads        | Diminishing returns     |
| 16 cores  | 8-12 threads        | Avoid over-subscription |

**Example**: Intel Core i7-12700KF (10 cores)

```bash
export RAYON_NUM_THREADS=8  # Conservative, high efficiency
# or
export RAYON_NUM_THREADS=10 # Maximum throughput
```

**Why Not Hyperthreads?**

- POWE.RS is **compute-bound** (80-90% time in LP solver)
- Hyperthreading provides minimal benefit (<5% speedup)
- May reduce efficiency due to resource contention

#### Dual-Socket Server (24-48 cores)

**Recommended**: Start with **single socket** to avoid NUMA effects

| Configuration | Threads | Notes                                      |
| ------------- | ------- | ------------------------------------------ |
| Single socket | 12-14   | Optimal (no NUMA)                          |
| Both sockets  | 24-28   | May need thread pinning                    |
| Full capacity | 48      | Only for very large problems (>100 stages) |

**NUMA Awareness** (Advanced):

```bash
# Pin to socket 0 (Intel)
numactl --cpunodebind=0 --membind=0 ./target/release/powers ...

# Or use Rayon with thread affinity (requires custom build)
```

#### Cloud/Virtualized Environments

**Recommended**: Test with different thread counts (vCPU != physical core)

```bash
# Test with 2, 4, 8 threads
for threads in 2 4 8; do
    RAYON_NUM_THREADS=$threads time ./target/release/powers examples/05-large-scale-brazilian
done
```

**Cloud-Specific Issues**:

- Shared physical cores (noisy neighbors)
- Variable CPU frequency (burst credits)
- NUMA on multi-socket VMs

### Recommendations by Problem Size

| Problem Size | Stages | Scenarios | Recommended Threads | Expected Efficiency |
| ------------ | ------ | --------- | ------------------- | ------------------- |
| Small        | <24    | <50       | 2-4                 | 80-95%              |
| Medium       | 24-60  | 50-200    | 4-8                 | 70-85%              |
| Large        | >60    | >200      | 8-12                | 60-75%              |
| Very Large   | >100   | >500      | 12-16               | 50-65%              |

**Rationale**:

- **Small problems**: Overhead dominates, more threads inefficient
- **Medium problems**: Good balance of speedup and efficiency
- **Large problems**: Maximum throughput, acceptable efficiency trade-off

### Parallel Efficiency Reference

From [Parallel Efficiency Analysis](../performance/PARALLEL_EFFICIENCY_ANALYSIS.md):

| Threads | Speedup | Efficiency | Status       |
| ------- | ------- | ---------- | ------------ |
| 1       | 1.00×   | 100%       | Baseline     |
| 2       | 1.89×   | 94.4%      | ✅ Excellent |
| 4       | 3.33×   | 83.3%      | ✅ Very Good |
| 8       | 5.51×   | 68.9%      | ✅ Good      |
| 16      | 7.60×   | 47.5%      | ⚠️ Moderate  |

**Key Insight**: **4 threads is the sweet spot** for most production deployments (83% efficiency).

### Validating Thread Configuration

```bash
# Monitor CPU usage during training
htop  # or top

# Should see:
# - All cores at ~100% during forward/backward passes
# - Brief dips during sequential sections (problem setup, result aggregation)

# If cores idle: Increase thread count
# If efficiency poor: Decrease thread count
```

### Common Thread Configuration Mistakes

❌ **Using too many threads**:

```bash
export RAYON_NUM_THREADS=32  # On 16-core CPU → poor efficiency
```

❌ **Relying on auto-detection**:

```bash
# Auto-detection may use hyperthreads (suboptimal)
# Always set explicitly for production
```

✅ **Correct approach**:

```bash
export RAYON_NUM_THREADS=8  # Explicit, matches physical cores
./target/release/powers examples/05-large-scale-brazilian
```

---

## Solver Tuning

### HiGHS Solver Role

The LP solver (HiGHS) **dominates runtime** (60-80% of total time):

- **Forward pass**: One LP solve per scenario per stage
- **Backward pass**: One LP solve per stage
- **Cut generation**: Dual variables extracted from solver

**Total LP solves per training**:

```
Solves = iterations × (forward_passes × stages + stages)
```

Example: 8 iterations × (16 fwd passes × 60 stages + 60 stages) = **8,160 solves**

**Key Optimization**: Solver tuning provides 10-20% speedup for specific problem classes.

### Automatic Optimizations (Already Enabled)

✅ **Basis Warm-Starting** (Automatic):

- Reuses basis between stages in backward pass
- **30-50% solver speedup** vs. cold start
- No user action required

✅ **Solver Retry Strategy** (Automatic):

- 5-level retry for numerical issues:
  1. Tight tolerances (1e-7)
  2. Relaxed tolerances (1e-6, 1e-5)
  3. Alternative simplex strategy
  4. Interior point method + presolve
  5. Error (infeasible problem)
- Handles 99.9% of numerical issues gracefully
- No user action required

### Presolve Strategy

**Default**: Presolve **enabled** (usually beneficial for large problems)

**When to Consider Disabling**:

- Very small problems (<10 stages, <10 scenarios)
- Presolve overhead > LP solve time
- Debugging infeasibility (presolve may obscure cause)

**How to Configure** (Future API - not yet exposed):

```rust
// Hypothetical API (Phase 2 enhancement)
solver.set_presolve(false);  // Disable presolve
```

**Expected Impact**: ±5-15% (problem-dependent)

### Tolerance Tuning

**Feasibility Tolerance** (default: 1e-7):

- Controls constraint satisfaction accuracy
- **Tighter** (1e-8): More accurate, slower convergence, numerical instability
- **Relaxed** (1e-6, 1e-5): Faster convergence, slight accuracy loss

**When to Relax**:

- Solver warnings about numerical issues
- Acceptable accuracy trade-off (e.g., hydro storage ±0.1% is negligible)
- Speed-critical applications (real-time dispatch)

**Expected Speedup**: 10-20% (if applicable)

**Optimality Tolerance** (default: 1e-7):

- Controls dual bound accuracy
- Rarely needs tuning (1e-7 is appropriate for most problems)

**Trade-off**: Speed vs. numerical accuracy

### Solver Settings (Not Yet Exposed)

The following settings are **planned for Phase 2** (user configuration):

1. **Simplex vs. Interior Point**:

   - Simplex: Default, better for warm-starting
   - IPM: Faster for very large problems (>100K constraints)

2. **Parallelism within Solver**:

   - HiGHS supports multi-threaded simplex
   - Currently disabled (Rayon handles parallelism)
   - May be useful for single-scenario problems

3. **Scaling Strategy**:
   - Default: Automatic scaling (equilibration)
   - Advanced users may want manual control

**Recommendation**: Wait for Phase 2 API before attempting solver tuning. Current defaults are well-tested.

### Monitoring Solver Performance

```rust
// Training output shows solver time (future enhancement)
// For now, use benchmarks:
cargo bench --bench subproblem_solve
```

Check `docs/performance/PERFORMANCE-BASELINES.md` for solver baseline times:

- Single reservoir: ~42-49 μs per solve (cold start)
- With warm-start: ~16 μs per solve (2.8× faster)

---

## Memory Management

### Current Memory Efficiency

POWE.RS has **excellent memory efficiency** (see [Memory Profiling](../performance/MEMORY-PROFILING.md)):

- **Small problems** (12 stages): ~17 MB
- **Medium problems** (24 stages): ~28 MB
- **Large problems** (60 stages): ~50-80 MB (depends on forward passes)
- **No memory leaks**: Delta RSS = 0 after warmup
- **Linear scaling**: ~0.73 MB per stage

**Memory Formula** (estimated):

```
Peak Memory (MB) ≈ 0.5 + (0.73 × stages) + (0.03 × scenarios) + (0.08 × iterations)
```

**Examples**:

- 24 stages, 50 scenarios, 100 iterations: ~28 MB
- 60 stages, 100 scenarios, 200 iterations: ~64 MB
- 100 stages, 200 scenarios, 500 iterations: ~120 MB

### Memory-Constrained Systems

**Guidelines** (<4 GB available):

- Keep `stages × iterations < 30,000`
- Reduce scenario count if needed (accuracy trade-off)
- Monitor RSS during training

**Monitoring Memory**:

```bash
# Watch memory usage during training
watch -n 5 'ps aux | grep powers'

# Or use built-in Linux tools
/usr/bin/time -v ./target/release/powers examples/05-large-scale-brazilian
```

### Cut Pool Management

**Current Implementation**: Unbounded cut pool (grows with iterations)

**Memory per Cut** (1 state variable):

- BendersCut struct: 57 bytes (id + coefficients + rhs)
- HashMap entry: 24 bytes (active_cut_indices)
- **Total**: ~81 bytes per cut

**Memory per Cut** (N state variables):

```
Memory = 81 + (N-1) × 8 bytes
```

Examples:

- 5 variables: 113 bytes/cut
- 10 variables: 153 bytes/cut
- 50 variables: 473 bytes/cut

**Expected Cut Count**:

```
Total cuts ≈ stages × iterations × active_cuts_per_stage
```

Typical: 10-100 active cuts per stage → 6,000-60,000 total cuts for 60 stages, 100 iterations

**Memory for Cut Pool** (60 stages, 100 iterations, 50 active cuts/stage):

```
300,000 cuts × 81 bytes = ~24 MB
```

**Future Enhancement** (Phase 3): Cut purging/aggregation for long runs (>1,000 iterations)

### Memory Optimization Strategies

✅ **Already Implemented**:

- Model reuse (avoids solver re-allocation)
- Cut selection (prevents unbounded growth)
- Basis warm-starting (eliminates re-allocation)
- Efficient data structures (minimize overhead)

⏳ **Future Enhancements** (Low Priority):

- Scenario pooling: ~19 KB savings per iteration (negligible)
- f32 for non-critical data: ~25% coefficient savings (small impact)
- Cut pool compaction: ~5% savings (not worth complexity)

**Recommendation**: No memory optimizations needed for typical problems. Current efficiency is excellent.

### Handling Large-Scale Problems

**Problem Size Limits** (32 GB RAM system):

| Stages | Iterations | Forward Passes | Estimated Memory | Status              |
| ------ | ---------- | -------------- | ---------------- | ------------------- |
| 60     | 100        | 16             | ~80 MB           | ✅ Safe             |
| 100    | 200        | 16             | ~180 MB          | ✅ Safe             |
| 120    | 500        | 32             | ~450 MB          | ✅ Safe             |
| 200    | 1000       | 32             | ~1.2 GB          | ⚠️ Monitor          |
| 500    | 2000       | 64             | ~8 GB            | ⚠️ Memory-intensive |

**If Memory Exceeds Limits**:

1. Reduce iteration count (convergence may suffer)
2. Reduce forward passes (policy quality may degrade)
3. Reduce scenario count (SAA approximation worse)
4. Consider staged training (Phase 3 feature)

---

## Problem-Specific Optimizations

### State Space Reduction

**Highest Impact Optimization**: Reduce state variable dimensions

**Examples**:

1. **Aggregate storage states**:

   ```
   Before: 10 small reservoirs → 10 state variables
   After: 1 aggregated reservoir → 1 state variable
   Impact: 10× reduction in cut dimensions, 2-5× speedup
   ```

2. **Eliminate redundant states**:

   ```
   Before: Track both storage and turbined water
   After: Track storage only (turbined = inflow - spill)
   Impact: 2× reduction in state space
   ```

3. **Coarser discretization**:
   ```
   Before: 1% storage precision → 100 discrete levels
   After: 5% storage precision → 20 discrete levels
   Impact: 5× reduction (but coarser decisions)
   ```

**Trade-off**: Solution quality vs. computational speed

### Scenario Selection

**Trade-off**: Fewer scenarios = faster, less accurate

**Guidelines**:

| Problem Type          | Recommended Scenarios | Rationale                |
| --------------------- | --------------------- | ------------------------ |
| Deterministic         | 1                     | No uncertainty           |
| Low uncertainty       | 20-50                 | Quick convergence        |
| Medium uncertainty    | 50-100                | Balanced                 |
| High uncertainty      | 100-200               | Better SAA approximation |
| Very high uncertainty | 200-500               | Diminishing returns >200 |

**Example**: Hydrothermal dispatch with inflow uncertainty

```json
{
  "num_forward_passes": 16, // Typical for production
  "num_simulation_scenarios": 100 // Out-of-sample validation
}
```

**Validating Scenario Count**:

1. Train with N scenarios
2. Simulate with 10×N scenarios (out-of-sample)
3. Compare policy quality
4. If policy degrades: Increase N

### Stage Aggregation

**Trade-off**: Fewer stages = faster, coarser decisions

**When to Consider**:

- Planning horizon allows coarser time steps (monthly vs. weekly)
- Decision granularity not critical (long-term vs. real-time)
- Rapid prototyping (validate problem formulation)

**Example**:

```
Before: 60 monthly stages (5 years)
After: 20 quarterly stages (5 years)
Impact: 3× faster (but coarser decisions)
```

**Recommendation**: Start with finest granularity, aggregate only if speed critical.

### Problem Formulation Best Practices

1. **Minimize state variables**: Each additional state variable increases cut dimensions exponentially
2. **Use tight bounds**: Narrow bounds improve solver performance
3. **Avoid redundant constraints**: Remove constraints implied by others
4. **Leverage problem structure**: Cascade topology, network structure
5. **Validate formulation**: Use small examples first (Example 01-04)

---

## Benchmarking and Validation

### Using the Criterion Benchmark Suite

POWE.RS includes **6 comprehensive benchmark suites**:

1. `comprehensive_benchmarks` - Production examples (Example 04, 05)
2. `memory_profiling` - Memory scaling analysis
3. `parallel_efficiency` - Thread scaling validation
4. `subproblem_solve` - Solver performance
5. `cut_selection` - Cut selection overhead
6. `state_operations` - Foundation operations

**Running Benchmarks**:

```bash
# Run all benchmarks (time: ~30 minutes)
cargo bench

# Run specific suite
cargo bench --bench comprehensive_benchmarks

# Run with baseline comparison
cargo bench --bench comprehensive_benchmarks -- --save-baseline main
# After tuning:
cargo bench --bench comprehensive_benchmarks -- --baseline main

# View HTML reports
open target/criterion/report/index.html
```

### Validating Tuning Changes

**Step 1: Baseline** (before tuning)

```bash
export RAYON_NUM_THREADS=4
cargo bench --bench comprehensive_benchmarks -- --save-baseline before
```

**Step 2: Apply Tuning**

```bash
export RAYON_NUM_THREADS=8
```

**Step 3: Measure** (after tuning)

```bash
cargo bench --bench comprehensive_benchmarks -- --baseline before
```

**Step 4: Compare Results**

Criterion shows relative performance:

```
training_iteration_large_scale/single_iteration_cold_start
    time:   [27.441 s 28.126 s 29.073 s]
    change: [-8.5% -4.4% +0.3%]  ← 4.4% faster!
```

### Metrics to Track

| Metric                  | Description             | Target                    |
| ----------------------- | ----------------------- | ------------------------- |
| **Iteration time**      | Time per SDDP iteration | Minimize                  |
| **Forward pass time**   | Scenario evaluation     | Should scale with threads |
| **Backward pass time**  | Cut generation          | Should scale with threads |
| **Solver time**         | LP solve duration       | 60-80% of total           |
| **Convergence quality** | Gap reduction           | Should not degrade        |

### Regression Detection

**Performance Regression Thresholds** (from CI):

| Change      | Severity    | Action                |
| ----------- | ----------- | --------------------- |
| >5% slower  | ⚠️ WARNING  | Investigate           |
| >10% slower | ❌ ERROR    | Block merge           |
| >20% slower | 🚨 CRITICAL | Must fix              |
| >5% faster  | ✅ GOOD     | Document optimization |

**Monitoring**:

```bash
# Check for regressions after code changes
cargo bench --bench comprehensive_benchmarks -- --baseline main
```

---

## Troubleshooting

### Poor Parallel Scaling

**Symptoms**: Speedup < 0.5 × thread count (efficiency <50%)

**Diagnosis**:

1. **Check CPU usage**:

   ```bash
   htop  # Should see ~100% on all cores during training
   ```

2. **Check thread affinity**:

   ```bash
   # Are threads migrating across sockets?
   perf stat -e 'sched:sched_migrate_task' ./target/release/powers ...
   ```

3. **Check background processes**:

   ```bash
   ps aux --sort=-%cpu | head -20  # Are other processes competing?
   ```

4. **Check memory bandwidth** (advanced):
   ```bash
   perf stat -e 'LLC-load-misses,LLC-store-misses' ./target/release/powers ...
   ```

**Solutions**:

✅ **Eliminate background processes**:

```bash
# Close browsers, IDEs, etc.
systemctl stop unnecessary-services
```

✅ **Pin threads to single socket** (dual-socket servers):

```bash
numactl --cpunodebind=0 --membind=0 ./target/release/powers ...
```

✅ **Reduce thread count if over-subscribed**:

```bash
export RAYON_NUM_THREADS=8  # From 16 (if efficiency improves)
```

✅ **Check for hyperthreading** (disable if enabled):

```bash
lscpu | grep -i thread  # Should match core count
```

### Slow Convergence

**Symptoms**: Many iterations needed, gap not decreasing, poor policy quality

**Diagnosis**:

1. **Check convergence plots** (future enhancement - manual for now):

   ```
   Iteration | Lower Bound | Upper Bound | Gap
   -----------------------------------------------
   1         | 1000        | 2000        | 50%
   10        | 1500        | 1800        | 16.7%
   20        | 1700        | 1750        | 2.9%
   ...
   ```

2. **Check scenario count**:

   ```json
   "num_forward_passes": 4  // Too few? Try 16
   ```

3. **Check problem formulation**:
   - State space too large?
   - Constraints feasible?
   - Objective function bounded?

**Solutions**:

✅ **Increase scenario count**:

```json
"num_forward_passes": 16  // Better SAA approximation
```

✅ **Increase iteration count**:

```json
"num_iterations": 200  // More cuts = better policy
```

✅ **Verify problem formulation**:

```bash
# Test with simple example first
./target/release/powers examples/01-deterministic
# Should converge in <10 iterations
```

✅ **Check for numerical issues**:

- Review solver warnings
- Check input data (unrealistic parameters?)
- Validate constraint feasibility

### High Memory Usage

**Symptoms**: RSS growing unexpectedly, approaching system limit, swapping to disk

**Diagnosis**:

1. **Monitor RSS during training**:

   ```bash
   watch -n 5 'ps aux | grep powers | awk "{print \$6/1024\" MB\"}"'
   ```

2. **Compare to formula**:

   ```
   Expected: 0.5 + (0.73 × 60) + (0.03 × 100) + (0.08 × 100) = ~55 MB
   Actual: 150 MB → Unexpected growth!
   ```

3. **Check cut pool size**:
   ```
   Total cuts = 60 stages × 100 iters × 50 cuts/stage = 300,000 cuts
   Memory = 300,000 × 81 bytes = ~24 MB (reasonable)
   ```

**Solutions**:

✅ **Reduce iteration count**:

```json
"num_iterations": 50  // From 100 (limits cut pool growth)
```

✅ **Reduce problem size**:

```json
"num_forward_passes": 8  // From 16
```

✅ **Monitor for unexpected growth** (potential bug):

```bash
# If delta RSS > 0 after warmup → file a bug report
```

✅ **Increase system RAM** (if needed):

- Large problems (>100 stages) may need 16-32 GB
- Cloud: Scale up VM instance

### Numerical Issues

**Symptoms**: Solver failures, infeasibility warnings, NaN values, solution quality degradation

**Diagnosis**:

1. **Review error messages**:

   ```
   Error: Solver failed: Infeasible problem after 5 retries
   Suggestion: Check input validation (unrealistic parameters?)
   ```

2. **Check input validation**:

   ```bash
   # Review input files
   cat examples/05-large-scale-brazilian/system.json | jq '.hydros[].max_capacity'
   # Are all values realistic?
   ```

3. **Check constraint feasibility**:
   - Is problem solvable with given constraints?
   - Are bounds consistent (min < max)?
   - Is objective function bounded?

**Solutions**:

✅ **Solver retry usually handles this automatically** (5-level retry)

✅ **Relax feasibility tolerance** (if needed):

```rust
// Future API (Phase 2)
solver.set_feasibility_tolerance(1e-6);  // From 1e-7
```

✅ **Review input files**:

```bash
# Validate JSON schema
cargo run --release -- --validate examples/05-large-scale-brazilian
```

✅ **Check for infeasible problem formulation**:

- Run small example first (Example 01)
- Gradually increase complexity
- Identify which constraint causes infeasibility

### Performance Not Improving

**Symptoms**: Tuning changes have minimal impact, benchmarks show no difference

**Diagnosis**:

1. **Verify tuning was applied**:

   ```bash
   echo $RAYON_NUM_THREADS  # Should show new value
   ```

2. **Check baseline comparison**:

   ```bash
   cargo bench --bench comprehensive_benchmarks -- --baseline before
   # Should show "change: [...]"
   ```

3. **Identify bottleneck**:
   ```bash
   # Profile to see where time is spent
   perf record -g ./target/release/powers examples/05-large-scale-brazilian
   perf report
   ```

**Solutions**:

✅ **Profile to find bottleneck**:

```bash
cargo flamegraph --bin powers -- examples/05-large-scale-brazilian
# Open flamegraph.svg to see where time is spent
```

✅ **Focus on hot paths**:

- If 80% time in solver: Solver tuning won't help much
- If 20% time in cut selection: Already optimized (154× speedup)
- If 10% time in forward pass: Thread count may help

✅ **Consider problem formulation**:

- State space reduction (biggest impact)
- Scenario count (accuracy vs. speed trade-off)

---

## Quick Reference

### Thread Configuration Cheat Sheet

| Hardware           | Problem Size             | Recommended Threads | Efficiency |
| ------------------ | ------------------------ | ------------------- | ---------- |
| 4-core laptop      | Small (<24 stages)       | 4                   | ~85%       |
| 8-core workstation | Medium (24-60 stages)    | 4-8                 | ~80%       |
| 16-core server     | Large (>60 stages)       | 8-12                | ~70%       |
| 32-core server     | Very large (>100 stages) | 12-16               | ~60%       |

**Command**:

```bash
export RAYON_NUM_THREADS=8
./target/release/powers examples/05-large-scale-brazilian
```

### Memory Estimation Cheat Sheet

```
Memory (MB) ≈ 0.5 + (0.73 × stages) + (0.03 × scenarios) + (0.08 × iterations)
```

**Examples**:

- 24 stages, 50 scenarios, 100 iters: ~28 MB
- 60 stages, 100 scenarios, 200 iters: ~64 MB

### Benchmarking Cheat Sheet

```bash
# Baseline
cargo bench --bench comprehensive_benchmarks -- --save-baseline before

# After tuning
cargo bench --bench comprehensive_benchmarks -- --baseline before

# View report
open target/criterion/report/index.html
```

### Performance Targets

| Metric                          | Target        | Notes             |
| ------------------------------- | ------------- | ----------------- |
| Parallel efficiency (4 threads) | >70%          | 83% achieved      |
| Memory stability                | Delta RSS = 0 | No leaks          |
| Cut selection overhead          | <1%           | Already optimized |
| Solver warm-start speedup       | 2-5×          | 2.8× achieved     |

---

## Summary

### Quick Wins (High Impact, Low Effort)

1. ✅ **Set explicit thread count**: `export RAYON_NUM_THREADS=8`
2. ✅ **Use production-scale examples**: Example 04 or 05 for realistic benchmarking
3. ✅ **Monitor convergence**: Track gap reduction, not just speed
4. ✅ **Validate with benchmarks**: Use Criterion to quantify gains

### Advanced Tuning (Medium Impact, Medium Effort)

1. **State space reduction**: 2-5× speedup (if problem allows)
2. **Scenario count optimization**: Balance accuracy vs. speed
3. **Memory-constrained systems**: Reduce iterations or problem size

### Expert-Level (Low Impact, High Effort)

1. **Solver tolerance tuning**: 10-20% speedup (problem-specific)
2. **NUMA-aware thread pinning**: 5-10% gain on dual-socket servers
3. **Custom solver configuration**: Wait for Phase 2 API

### Monitoring Best Practices

1. ✅ **Track iteration time** during training
2. ✅ **Monitor CPU usage** (should be ~100%)
3. ✅ **Monitor memory** (should be stable after warmup)
4. ✅ **Validate convergence quality** (gap reduction)
5. ✅ **Use benchmarks** to quantify tuning gains

### When to Seek Help

❓ **Performance issues?**

- Check troubleshooting section first
- Review error messages (usually actionable)
- Run benchmarks to identify bottleneck

❓ **Still stuck?**

- File an issue: https://github.com/rjmalves/powers/issues
- Include: Hardware specs, problem size, benchmark results
- Expected vs. actual performance

---

## References

- [Parallel Efficiency Analysis](../performance/PARALLEL_EFFICIENCY_ANALYSIS.md): Thread scaling validation
- [Memory Profiling Report](../performance/MEMORY-PROFILING.md): Memory usage analysis
- [Performance Baselines](../performance/PERFORMANCE-BASELINES.md): Regression thresholds
- [Benchmark Suite](../../benches/README.md): How to run benchmarks
- [Example Problems](../../examples/README.md): Production-ready examples

---

**Document Version**: 1.0  
**Last Updated**: October 9, 2025  
**Author**: HPC Development Team  
**Status**: Production-ready

**Feedback**: Please report issues or suggest improvements via GitHub Issues.
