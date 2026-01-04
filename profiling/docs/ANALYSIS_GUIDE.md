# Analysis Guide

How to interpret profiling results and diagnose performance issues.

## Table of Contents

- [Reading Profiling Results](#reading-profiling-results)
- [Timing Analysis](#timing-analysis)
- [Memory Analysis](#memory-analysis)
- [CPU Analysis](#cpu-analysis)
- [Parallel Scaling Analysis](#parallel-scaling-analysis)
- [Common Performance Patterns](#common-performance-patterns)
- [Decision Trees](#decision-trees)

---

## Reading Profiling Results

### Run Summary

Every profiling run creates a summary in `run.json`:

```json
{
  "run_id": "20260103-180250-abc1234",
  "timestamp": "2026-01-03T18:02:50Z",
  "status": "complete",
  "total_duration_seconds": 25.34,
  "collectors_run": ["timing", "rss", "cpu"],
  "git_info": {
    "branch": "main",
    "commit_short": "abc1234",
    "is_dirty": false
  }
}
```

**Key Fields:**
- `status` - "complete", "failed", or "partial"
- `total_duration_seconds` - Wall-clock time (includes profiling overhead)
- `collectors_run` - Which collectors succeeded

---

## Timing Analysis

### Interpreting Timing Metrics

Timing metrics show how long each phase takes:

```json
{
  "timings": {
    "initialization": 0.523,
    "graph_construction": 12.456,
    "optimization": 8.234,
    "output": 1.123
  }
}
```

**Analysis:**
1. **Identify bottlenecks**: Which phase consumes most time?
2. **Check distribution**: Is one phase dominating? (>50% of total)
3. **Compare versions**: Has a phase gotten slower?

**Example Interpretation:**
```
initialization:      0.5s (2%)   ← Fast, OK
graph_construction: 12.5s (56%)  ← BOTTLENECK!
optimization:        8.2s (37%)  ← Significant
output:              1.1s (5%)   ← Fast, OK
Total:              22.3s
```

**Actionable Insights:**
- If **graph_construction** dominates → optimize data structures, reduce allocations
- If **optimization** dominates → improve algorithm, add early termination
- If **output** dominates → use buffered writes, compress data

---

## Memory Analysis

### RSS (Resident Set Size)

RSS shows physical memory usage over time.

#### Key Metrics

```json
{
  "summary": {
    "peak_mb": 253.4,
    "mean_mb": 180.2,
    "final_mb": 45.6,
    "growth_mb": 207.8
  }
}
```

**Interpretation:**

| Metric | Meaning | Good Range | Warning Signs |
|--------|---------|------------|---------------|
| `peak_mb` | Maximum memory | Depends on problem | >80% of system RAM |
| `mean_mb` | Average usage | <50% of peak | Consistent high usage |
| `final_mb` | Memory at exit | Low (if freed) | High (possible leak) |
| `growth_mb` | Peak - initial | Varies | Continuous growth |

**Memory Leak Detection:**

```
peak_mb: 500 MB     ← High peak
mean_mb: 450 MB     ← High average
final_mb: 480 MB    ← High at exit (LEAK!)
growth_mb: 475 MB   ← Large growth
```

**Healthy Memory Profile:**

```
peak_mb: 200 MB     ← Reasonable peak
mean_mb: 120 MB     ← Lower average
final_mb: 10 MB     ← Low at exit (freed)
growth_mb: 190 MB   ← Controlled growth
```

### DHAT (Heap Allocation)

DHAT shows dynamic memory allocation patterns.

```json
{
  "total_bytes": 5242880000,
  "total_blocks": 123456,
  "max_bytes": 268435456,
  "max_blocks": 45678
}
```

**Analysis:**

1. **Allocation Churn**: High `total_blocks` relative to `max_blocks`
   ```
   total_blocks: 1,000,000
   max_blocks:        10,000
   → 99% of allocations are temporary (churn!)
   ```

2. **Memory Efficiency**: `total_bytes` vs `max_bytes`
   ```
   total_bytes: 10 GB
   max_bytes:    500 MB
   → Allocating 20x peak usage (inefficient!)
   ```

**Actionable Insights:**
- **High churn** → Pool/reuse allocations, use arena allocator
- **High total/max ratio** → Reduce temporary allocations, use stack where possible

### Massif (Heap Timeline)

Massif shows heap usage over time.

```json
{
  "peak_bytes": 268435456,
  "peak_snapshot": 42,
  "snapshots": [...]
}
```

**Visualization:** View `massif.out` with `ms_print`:
```bash
ms_print memory/massif/massif.out | head -50
```

**Patterns to Look For:**
- **Steady climb** → Memory leak
- **Sawtooth** → Healthy allocation/deallocation
- **Plateaus** → Caching (may be intentional)
- **Spikes** → Temporary large allocations

### Cachegrind (Cache Efficiency)

Cache misses hurt performance significantly.

```json
{
  "Ir": 10000000000,     # Instructions
  "I1mr": 1234567,       # L1 instruction misses
  "D1mr": 9876543,       # L1 data read misses
  "D1mw": 5432109        # L1 data write misses
}
```

**Miss Rates:**
```
L1 I-cache miss rate: I1mr / Ir = 0.012% (good < 1%)
L1 D-cache miss rate: (D1mr + D1mw) / Ir = 0.15% (good < 5%)
```

**Actionable Insights:**
- **High I-cache misses** → Code bloat, reduce binary size, improve locality
- **High D-cache misses** → Improve data layout (SoA vs AoS), reduce working set

---

## CPU Analysis

### Hotspot Identification

Hotspots are functions consuming the most CPU time.

```json
{
  "hotspots": [
    {
      "function": "powers::graph::builder::construct",
      "self_percent": 23.45,
      "total_percent": 45.67,
      "samples": 12345
    }
  ]
}
```

**Metrics:**
- `self_percent` - Time in this function (excluding callees)
- `total_percent` - Time in this function + callees
- `samples` - Number of perf samples

**Analysis Priority:**
1. **High self%** → Optimize this function directly
2. **High total%, low self%** → Optimize callees
3. **Many samples** → Hot path (profile-guided optimization)

**Example:**
```
Function                      Self%  Total%  Samples
graph::builder::construct     23.5%   45.7%   12,345  ← Optimize directly
graph::add_edge               15.2%   15.4%    8,012  ← Optimize directly
HashMap::insert                8.1%    8.1%    4,267  ← Called frequently
alloc::alloc                   2.3%    2.3%    1,213  ← Allocation overhead
```

**Actionable Insights:**
- `graph::builder::construct` - Hot function, optimize algorithm
- `graph::add_edge` - Frequently called, consider batching
- `HashMap::insert` - Many insertions, pre-allocate capacity
- `alloc::alloc` - Reduce allocations, reuse memory

### FlameGraph Analysis

FlameGraphs show the call stack consuming CPU.

**How to Read:**
- **Width** = CPU time (wider = more time)
- **Height** = Call depth (taller = deeper stack)
- **Color** = Function type (depends on coloring scheme)

**Patterns:**
- **Wide plateaus** → Hot function (optimize it!)
- **Towers** → Deep call stacks (inline/devirtualize?)
- **Many thin slices** → Function call overhead

---

## Parallel Scaling Analysis

### Speedup and Efficiency

```json
{
  "speedup_metrics": [
    {"thread_count": 1, "speedup": 1.00, "efficiency": 1.00},
    {"thread_count": 2, "speedup": 1.95, "efficiency": 0.975},
    {"thread_count": 4, "speedup": 3.70, "efficiency": 0.925},
    {"thread_count": 8, "speedup": 6.80, "efficiency": 0.850}
  ]
}
```

**Interpretation:**

| Threads | Speedup | Efficiency | Assessment |
|---------|---------|------------|------------|
| 1 | 1.00x | 100% | Baseline |
| 2 | 1.95x | 97.5% | 🟢 Excellent |
| 4 | 3.70x | 92.5% | 🟢 Excellent |
| 8 | 6.80x | 85.0% | 🟡 Good |
| 16 | 11.2x | 70.0% | 🟠 Fair |
| 32 | 16.0x | 50.0% | 🔴 Poor |

**Efficiency Ranges:**
- **>90%** - Excellent, near-linear scaling
- **70-90%** - Good, acceptable overhead
- **50-70%** - Fair, noticeable overhead
- **<50%** - Poor, investigate bottlenecks

### Amdahl's Law

```json
{
  "amdahl_estimate": {
    "serial_fraction": 0.085,
    "parallel_fraction": 0.915,
    "predicted_max_speedup": 11.76,
    "confidence": "high"
  }
}
```

**Interpretation:**
- **Serial fraction 8.5%** → 91.5% of work is parallelizable
- **Max speedup 11.76x** → Diminishing returns beyond ~12 threads
- **High confidence** → Estimate is reliable

**Actionable Insights:**
- If serial fraction > 20% → Parallelize more of the code
- If max speedup < target threads → Consider algorithmic changes
- If efficiency drops sharply → Check for lock contention

### Contention Detection

```json
{
  "contention_analysis": {
    "wait_events": 123456,
    "futex_events": 45678,
    "high_contention_growth": true
  }
}
```

**Warning Signs:**
- `wait_events` increasing super-linearly with threads
- `high_contention_growth: true`
- Efficiency cliff at specific thread count

**Solutions:**
- Reduce lock scope
- Use lock-free data structures
- Partition data to avoid sharing
- Consider NUMA-aware allocation

---

## Common Performance Patterns

### Pattern 1: Allocation Bottleneck

**Symptoms:**
- High `alloc::alloc` time in CPU profile
- High DHAT `total_blocks` relative to `max_blocks`
- RSS growing continuously

**Diagnosis:**
```
CPU Hotspot: alloc::alloc - 15% self time
DHAT: total_blocks: 10M, max_blocks: 100K
→ 99% allocation churn!
```

**Solutions:**
1. Object pooling
2. Arena allocator
3. Pre-allocate with capacity
4. Use stack allocation (SmallVec, ArrayVec)

### Pattern 2: Cache Misses

**Symptoms:**
- High Cachegrind D1 miss rate (>5%)
- Poor speedup despite parallelism
- Memory bandwidth bottleneck

**Diagnosis:**
```
Cachegrind: D1mr = 10M, Ir = 100M
→ 10% D1 miss rate (bad!)
```

**Solutions:**
1. Structure of Arrays (SoA) layout
2. Tiling/blocking for cache locality
3. Reduce working set size
4. Prefetching hints

### Pattern 3: Lock Contention

**Symptoms:**
- Efficiency drops sharply at high thread counts
- High `futex_events` in contention analysis
- Speedup plateaus early

**Diagnosis:**
```
Threads: 8, Efficiency: 90%
Threads: 16, Efficiency: 60% ← CLIFF!
futex_events: 2x increase
→ Lock contention
```

**Solutions:**
1. Fine-grained locking
2. Lock-free data structures (crossbeam)
3. Thread-local accumulation
4. Work partitioning

### Pattern 4: Amdahl's Law Limit

**Symptoms:**
- Good efficiency initially, then plateau
- Serial fraction >10%
- Speedup far below thread count

**Diagnosis:**
```
Serial fraction: 20%
Max predicted speedup: 5x
Actual speedup at 8 threads: 4.8x
→ Hitting Amdahl limit
```

**Solutions:**
1. Parallelize serial sections
2. Algorithmic change (e.g., different graph traversal)
3. Accept limitation if serial work is unavoidable

---

## Decision Trees

### "My Program is Slow"

```
Is total_duration high?
├─ Yes → Check timing breakdown
│  ├─ One phase dominates (>50%)?
│  │  ├─ Yes → Profile that phase deeply (CPU + Memory)
│  │  └─ No → Multiple bottlenecks, prioritize by impact
│  └─ All phases slow → Check CPU profile for algorithmic issues
└─ No → Overhead acceptable, consider optimization ROI
```

### "Memory Usage is High"

```
Is peak_mb concerning?
├─ Yes → Check RSS timeline
│  ├─ Continuous growth?
│  │  ├─ Yes → MEMORY LEAK! Use DHAT to find source
│  │  └─ No → High working set, may be normal
│  └─ Spiky usage?
│     ├─ Yes → Temporary allocations, check DHAT churn
│     └─ No → Stable usage, may be caching (check if intentional)
└─ No → Memory usage acceptable
```

### "Parallel Performance is Poor"

```
Is efficiency <70% at low thread counts (2-4)?
├─ Yes → Parallelization overhead
│  ├─ Task granularity too small?
│  │  ├─ Yes → Increase work per task
│  │  └─ No → Check for excessive synchronization
│  └─ High contention?
│     ├─ Yes → Reduce lock scope, use lock-free structures
│     └─ No → Algorithm may not parallelize well
└─ No → Check scaling at higher thread counts
   ├─ Efficiency cliff at specific thread count?
   │  ├─ Yes → Check for contention, NUMA effects
   │  └─ No → Smooth degradation (Amdahl's law)
   └─ Amdahl serial fraction?
      ├─ High (>20%) → Parallelize more code
      └─ Low (<10%) → Good parallelization, diminishing returns expected
```

---

## Performance Checklist

### Before Optimization

- [ ] Profile first (don't guess!)
- [ ] Establish baseline metrics
- [ ] Identify top 3 bottlenecks
- [ ] Estimate optimization potential

### During Optimization

- [ ] Focus on hot paths (>10% time)
- [ ] One optimization at a time
- [ ] Re-profile after each change
- [ ] Document what changed and why

### After Optimization

- [ ] Compare with baseline
- [ ] Verify correctness (tests still pass)
- [ ] Check for regressions in other metrics
- [ ] Document performance gains

---

## See Also

- **[Tools Reference](TOOLS_REFERENCE.md)** - Complete collector documentation
- **[Comparison Guide](COMPARISON_GUIDE.md)** - Detecting regressions
- **[Scaling Guide](../SCALING_GUIDE.md)** - Multi-socket optimization
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues
