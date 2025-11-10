# 📊 SDDP Benchmark Guide

## Current Situation Summary

**Previous State**: Only micro-benchmarks on isolated components were available:
- ✅ `cut_selection.rs` - Dominance checking algorithms  
- ✅ `correlation_application.rs` - Scenario correlation
- ✅ `memory_profiling.rs` - Memory tracking (synthetic)
- ⚠️ **Problem**: These tested on trivial problem sizes (1-2 hydros)
- ⚠️ **Missing**: End-to-end SDDP algorithm benchmarks

**What Changed**: Created comprehensive E2E benchmark suite (`sddp_e2e.rs`) that measures:
- Full iteration performance on **production-scale problems** (156 hydros!)
- Forward/backward pass breakdown
- Simulation performance
- Scalability with training effort

**Critical Insight**: Small test cases (1-2 hydros) give **misleading results**:
- Cache effects don't materialize
- Allocation overhead is hidden by noise
- Parallelism opportunities invisible
- O(n²) algorithms appear fast when n=2

---

## 🎯 Benchmark Suite Overview

### File: `benches/sddp_e2e.rs`

Uses **Example 05** (Large-scale Brazilian system) as realistic test case:
- **156 hydro plants** (realistic cascade)
- **121 thermal plants** (large generation mix)
- **5 buses** (regional interconnections)
- **60 stages** (5 years monthly planning)
- **State space: ~156 dimensions**

This is representative of **production-scale** hydrothermal dispatch problems.
Performance optimizations **must** be validated at this scale to be meaningful.

### Benchmark Groups

| Group | Measures | Runtime | Purpose |
|-------|----------|---------|---------|
| **single_iteration** | One forward + backward (hot path!) | 10-15 min | **Most important** - iteration efficiency |
| **training_phases** | Forward vs backward breakdown | 5-10 min | Identify bottleneck |
| **simulation** | Out-of-sample policy simulation | 5-8 min | Post-training performance |
| **problem_scaling** | Performance vs training effort | 10-15 min | Scaling behavior |

**Total runtime**: ~30-50 minutes for full suite

---

## 🚀 Quick Start

### Run All E2E Benchmarks (WARNING: ~30-50 minutes with 156 hydros!)

```bash
cargo bench --bench sddp_e2e
```

### Run Specific Group (Recommended)

```bash
# Single iteration tests (MOST IMPORTANT - ~10-15 minutes)
cargo bench --bench sddp_e2e single_iteration

# Training phases breakdown (~5-10 minutes)
cargo bench --bench sddp_e2e training_phases

# Problem scaling (~10-15 minutes)
cargo bench --bench sddp_e2e problem_scaling
```

### Save Baseline Before Optimization

```bash
# Run benchmarks and save as baseline
cargo bench --bench sddp_e2e -- --save-baseline before_refactoring

# After making changes, compare
cargo bench --bench sddp_e2e -- --baseline before_refactoring
```

### Quick Smoke Test (Just compile, don't run)

```bash
cargo bench --bench sddp_e2e --no-run
```

---

## 📈 Performance Metrics Captured

### Primary Metrics
- **Wall-clock time**: Total execution time
- **Solver calls/sec**: Throughput of LP solver
- **Cuts generated/sec**: Cut generation rate
- **Forward/backward split**: Time breakdown by phase

### Detailed Timing (via TrainingResult)
- `forward_timing.total_time` - Forward pass duration
- `backward_timing.total_time` - Backward pass duration
- `backward_timing.solver_time` - Time in HiGHS solver
- `backward_timing.cut_selection_time` - Cut management overhead
- `num_solver_calls` - Total LP solves
- `num_cuts_added` - Cuts generated

---

## 🎓 How to Use for Optimization

### Step 1: Establish Baseline

```bash
# Run and save baseline
cargo bench --bench sddp_e2e -- --save-baseline baseline_v0.2.0

# Check results
open target/criterion/report/index.html
```

**What to look for**:
- Which group takes most time?
- Is forward or backward pass slower?
- How does performance scale with iterations?

### Step 2: Identify Bottleneck

Look at timing breakdown in benchmark output:

```
Phase Breakdown:
  Forward:  2.5s
  Backward: 4.8s
```

If **backward > forward**: Focus on:
- Cut generation (`backward_timing.solver_time`)
- Cut selection (`backward_timing.cut_selection_time`)
- State updates (`backward_timing.fcf_state_update_time`)

If **forward > backward**: Focus on:
- Scenario sampling
- Forward solve efficiency
- State propagation

### Step 3: Make Optimization

Example: Pre-allocate buffers in backward pass

```rust
// Before: Allocates every iteration
for scenario in scenarios {
    let temp = vec![0.0; size];  // 🔴 Allocation!
    // use temp
}

// After: Reuse buffer
let mut temp = vec![0.0; size];
for scenario in scenarios {
    temp.fill(0.0);  // ✅ Reuse
    // use temp
}
```

### Step 4: Measure Impact

```bash
# Compare with baseline
cargo bench --bench sddp_e2e -- --baseline baseline_v0.2.0
```

**Interpreting results**:
```
sddp_single_iteration/forward_passes/10
                        time:   [450.20 ms 455.30 ms 460.50 ms]
                        change: [-15.234% -12.456% -9.678%] (p = 0.00 < 0.05)
                        Performance has improved.
```

- **Negative change** = improvement (faster) ✅
- **Positive change** = regression (slower) ❌
- **p < 0.05** = statistically significant

### Step 5: Validate Correctness

```bash
# Always run tests after optimization
cargo test --release
```

---

## 📝 Benchmark Design Principles

### Why Example 05?

✅ **Production-scale**: 156 hydros, 121 thermals, 60 stages - real-world size  
✅ **Representative**: Exercises all SDDP bottlenecks (solver, cuts, state)  
✅ **Available**: Ships with the codebase  
✅ **Realistic complexity**: O(n²) algorithms actually hurt at this scale

❌ **Why NOT small examples?** (1-2 hydros):
- Cache always hits (unrealistic)
- Allocations hidden in noise
- Parallelism opportunities invisible
- Linear vs quadratic algorithms both "fast"
- Optimizations that help at scale show no benefit

**Rule**: If optimization doesn't help with 156 hydros, it's not worth doing.

### Benchmark Configuration

```rust
group.sample_size(10);  // 10 iterations per benchmark
group.measurement_time(std::time::Duration::from_secs(60));  // 60s measurement
```

- **sample_size**: Trade-off between accuracy and runtime
  - Micro-benchmarks: 50-100 samples
  - E2E benchmarks: 10-20 samples (slower)
  
- **measurement_time**: How long to collect samples
  - Fast operations: 5-10 seconds
  - Training/simulation: 30-60 seconds

### Baseline Management

```bash
# Save baseline before starting optimization work
cargo bench --bench sddp_e2e -- --save-baseline phase1_start

# After Phase 1 optimizations
cargo bench --bench sddp_e2e -- --baseline phase1_start --save-baseline phase1_done

# After Phase 2 optimizations
cargo bench --bench sddp_e2e -- --baseline phase1_done --save-baseline phase2_done

# Compare end-to-end improvement
cargo bench --bench sddp_e2e -- --baseline phase1_start
```

---

## ⚠️ Warning: Micro-Benchmarks Can Mislead

**Other benchmarks in `benches/` directory**:
- `cut_selection.rs` - Tests cut dominance on synthetic data
- `correlation_application.rs` - Tests correlation on small matrices  
- `memory_profiling.rs` - Tracks memory on small problems
- `simulation_memory.rs` - Memory during simulation (small scale)
- `parallel_efficiency.rs` - Parallelism on tiny problems

**Critical Problem**: These all use **trivial problem sizes** (1-5 hydros, synthetic data).

### Why This Matters

**1. Hidden Complexity**
- O(n²) vs O(n log n) both look "fast" when n=2
- Linear search appears competitive with binary search
- Sorting 5 items takes nanoseconds either way

**2. Cache Effects**
- Everything fits in L1 cache (unrealistic)
- No cache misses to measure
- Data layout optimizations appear irrelevant

**3. Allocation Noise**
- Small allocations are fast (<1µs)
- Reusing buffers shows no benefit in micro-benchmarks
- Memory pressure never manifests

**4. No Parallelism**
- Problems too small to benefit from threads
- Rayon overhead dominates actual work
- Single-threaded code wins micro-benchmarks

**5. False Confidence**
- "Optimized" code may regress at scale
- Improvements disappear or reverse with real data
- Time wasted on irrelevant optimizations

### Real Example

```rust
// Micro-benchmark (5 hydros): HashMap 20% faster than Vec
// Production (156 hydros): Vec 3x faster due to cache locality!
```

**A cut selection "optimization" that improves micro-benchmark by 20% might actually regress by 10% on the full 156-hydro problem due to cache pressure.**

### Recommendation

- ✅ **Always validate with `sddp_e2e`** - The only benchmark that matters
- ⚠️ **Use micro-benchmarks** only for:
  - Comparing algorithm variants in isolation
  - Unit testing specific functions
  - Quick feedback during development
- ❌ **Never trust micro-benchmark** improvements without E2E validation
- ❌ **Never commit optimizations** based only on micro-benchmarks

**Rule of thumb**: If it doesn't help the 156-hydro case, it doesn't help.

---

## 🔧 Troubleshooting

### Benchmarks Take Too Long

**Reduce sample size** (less accurate but faster):

```rust
// In benches/sddp_e2e.rs
group.sample_size(5);  // Reduce from 10 to 5
group.measurement_time(std::time::Duration::from_secs(30));  // Reduce from 60s
```

### High Variance in Results

**Causes**:
- Background processes competing for CPU
- Thermal throttling
- Insufficient warm-up

**Solutions**:
```bash
# Close other applications
# Run on a quiet system

# Increase warm-up time in code:
group.warm_up_time(std::time::Duration::from_secs(5));
```

### Criterion Can't Find Gnuplot

**Not a problem** - Criterion will use plotters backend (pure Rust).

To install gnuplot (optional):
```bash
sudo apt install gnuplot
```

---

## 📊 Interpreting Benchmark Output

### Example Output

```
sddp_single_iteration/forward_passes/10
                        time:   [450.20 ms 455.30 ms 460.50 ms]
                        thrpt:  [  2.17 elem/s   2.20 elem/s   2.22 elem/s]
Found 2 outliers among 10 measurements (20.00%)
  1 (10.00%) high mild
  1 (10.00%) low mild
```

**Reading the numbers**:
- `[450.20 ms 455.30 ms 460.50 ms]` = [lower_bound, estimate, upper_bound]
- `455.30 ms` = best estimate of mean time
- `[450.20, 460.50]` = 95% confidence interval
- `2.20 elem/s` = throughput (iterations per second)

**Outliers**:
- High mild/severe: Some runs were slower
- Low mild/severe: Some runs were faster
- <20% outliers is normal
- >50% outliers suggests unstable benchmark

### Regression Detection

```
                        change: [-15.234% -12.456% -9.678%] (p = 0.00 < 0.05)
```

- **change < 0**: Improvement (faster)
- **change > 0**: Regression (slower)
- **p-value < 0.05**: Statistically significant
- **p-value > 0.05**: Could be noise

---

## 🎯 Recommended Optimization Workflow

### For QUICKSTART_REFACTORING.md

```bash
# Phase 0: Establish baseline
cargo bench --bench sddp_e2e -- --save-baseline baseline_start
cargo test --release  # Validate correctness

# Phase 1: Memory optimization (buffer pre-allocation)
# ... make changes ...
cargo bench --bench sddp_e2e -- --baseline baseline_start
cargo test --release  # Validate correctness
# Expected: 15-25% improvement in backward pass

# Phase 2: Cut selection optimization
# ... make changes ...
cargo bench --bench sddp_e2e -- --baseline baseline_start
cargo test --release  # Validate correctness
# Expected: Additional 10-15% improvement

# Phase 3: Parallel optimization
# ... make changes ...
cargo bench --bench sddp_e2e -- --baseline baseline_start
cargo test --release  # Validate correctness
# Expected: 20-30% improvement with multi-core

# Final check: Compare end-to-end
cargo bench --bench sddp_e2e -- --baseline baseline_start
```

**Target improvements**:
- Overall runtime: **-40% to -60%**
- Backward pass: **-30%**
- Memory allocations: **-80%**
- Solver calls/sec: **+50%**

---

## 🔗 Related Documentation

- **QUICKSTART_REFACTORING.md** - Step-by-step optimization guide
- **PERFORMANCE_REFACTORING_PLAN.md** - Detailed optimization phases
- **REFACTORING_SUMMARY.md** - Overview of approaches

---

## 💡 Tips

### For Performance Work

1. **Always profile first** - Don't guess bottlenecks
2. **Benchmark before and after** - Measure actual impact
3. **Keep tests passing** - Correctness > speed
4. **Document trade-offs** - Explain performance changes
5. **Focus on hot paths** - 80% effort on 20% of code

### For Accurate Benchmarks

1. **Run on quiet system** - Close other apps
2. **Disable CPU scaling** - Lock frequency if possible
3. **Multiple runs** - Check consistency
4. **Warm up properly** - First run is always slower
5. **Compare apples to apples** - Same system, same conditions

---

## ✅ Checklist: Before Committing Optimization

- [ ] Benchmark shows improvement (negative change %)
- [ ] p-value < 0.05 (statistically significant)
- [ ] All tests pass (`cargo test --release`)
- [ ] No regressions in other benchmarks
- [ ] Code remains readable
- [ ] Performance comments added
- [ ] Baseline comparison saved

---

**Ready to start?**

```bash
# 1. Run baseline
cargo bench --bench sddp_e2e -- --save-baseline before_optimization

# 2. Make your optimization changes

# 3. Compare
cargo bench --bench sddp_e2e -- --baseline before_optimization

# 4. Validate
cargo test --release
```

Happy optimizing! 🚀
