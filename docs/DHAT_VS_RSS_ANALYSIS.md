# DHAT vs RSS Analysis: Understanding Memory Behavior

## Executive Summary

**Critical Finding**: DHAT allocation data and RSS (Resident Set Size) tell **different stories** about memory behavior. This document explains why and what each metric means.

### Key Insights

1. **DHAT tracks allocations** - Total bytes allocated over program lifetime
2. **RSS tracks actual memory usage** - Physical memory currently in use
3. **glibc + malloc_trim achieves stable RSS** despite high DHAT allocations
4. **Alternative allocators show high RSS** despite similar DHAT patterns

## DHAT Allocation Comparison

All runs use example `05-large-scale-brazilian`, 8 iterations, release build:

| Run | Total Allocated | Total Blocks | Instructions | Notes |
|-----|-----------------|--------------|--------------|-------|
| baseline | 82.13 GB | 159,007,634 | 480 B | Before optimizations |
| new | 42.31 GB | 43,374,152 | 209 B | After Sprint 6 |
| updated | 42.31 GB | 43,292,811 | 209 B | After Sprint 7 |
| sprint08 | 46.87 GB | 46,646,039 | 217 B | Model lifecycle + malloc_trim |
| **trim** | **46.87 GB** | **46,646,301** | **217 B** | **Same as sprint08** |

### DHAT Analysis

- **Baseline → New**: -48.5% allocations (Sprint 6 HiGHS optimizations)
- **Sprint08 vs Trim**: Essentially identical (+0.0% difference)
- **Both Sprint08 and Trim**: 46.87 GB total allocations

**Conclusion from DHAT**: The `malloc_trim()` call doesn't change allocation patterns (expected).

## RSS (Resident Set Size) Comparison

Same workload, measured via `/proc/self/status` VmRSS:

### glibc (System Allocator) - With malloc_trim

| Iteration | Start (MB) | End (MB) | Delta (KB) | malloc_trim Called |
|-----------|------------|----------|------------|-------------------|
| 1 | 204.6 | 246.5 | +42,800 | ✅ |
| 2 | 246.5 | 248.4 | +1,956 | ✅ |
| 3 | 248.4 | 249.5 | +1,104 | ✅ |
| 4 | 249.5 | 249.7 | +220 | ✅ |
| 5 | 249.7 | 250.4 | +660 | ✅ |
| 6 | 250.4 | 250.9 | +516 | ✅ |
| 7 | 250.9 | 251.2 | +300 | ✅ |
| 8 | 251.2 | 253.4 | +2,288 | ✅ |

**Final RSS: ~253 MB** | **Stable after iteration 2**

### mimalloc

| Iteration | Start (MB) | End (MB) | Delta (KB) |
|-----------|------------|----------|------------|
| 1 | 268.6 | 493.8 | +230,548 |
| 2 | 493.8 | 552.7 | +60,284 |
| 3 | 552.7 | 606.7 | +55,292 |
| 4 | 606.7 | 653.2 | +47,668 |
| 5 | 653.2 | 687.4 | +34,976 |
| 6 | 687.4 | 760.7 | +75,124 |
| 7 | 760.7 | 798.4 | +38,624 |
| 8 | 798.4 | 838.1 | +40,600 |

**Final RSS: ~838 MB** | **Continuous growth**

### jemalloc

| Iteration | Start (MB) | End (MB) | Delta (KB) |
|-----------|------------|----------|------------|
| 1 | 230.8 | 449.3 | +223,784 |
| 2 | 449.3 | 481.6 | +33,032 |
| 3 | 481.6 | 535.8 | +55,544 |
| 4 | 535.8 | 581.3 | +46,616 |
| 5 | 581.3 | 628.4 | +48,208 |
| 6 | 628.2 | 664.5 | +37,116 |
| 7 | 664.5 | 728.1 | +65,184 |
| 8 | 728.1 | 775.4 | +48,416 |

**Final RSS: ~775 MB** | **Continuous growth**

## Why DHAT and RSS Tell Different Stories

### DHAT Measures Cumulative Allocations

```
Iteration 1: Allocate 100 MB, free 100 MB
Iteration 2: Allocate 100 MB, free 100 MB
...
Iteration 8: Allocate 100 MB, free 100 MB

DHAT Total: 800 MB (sum of all allocations)
```

### RSS Measures Current Memory Usage

```
Iteration 1: Allocate 100 MB, free 100 MB
  - glibc: Returns 100 MB to OS → RSS stays low
  - mimalloc: Caches 100 MB → RSS stays 100 MB

Iteration 2: Allocate 100 MB, free 100 MB
  - glibc: Reuses+returns → RSS stays ~100 MB
  - mimalloc: Allocates new 100 MB, caches both → RSS now 200 MB

...

After 8 iterations:
  - glibc RSS: ~100 MB (stable, returns to OS)
  - mimalloc RSS: ~800 MB (caches everything)
```

## Why malloc_trim() Works for glibc

The `malloc_trim(0)` call after `finalize_iteration()` tells glibc to:
1. Consolidate freed memory
2. Return memory above the "top" of the heap to the OS via `sbrk()`

**Effect**:
- DHAT: No change (allocations still happen)
- RSS: Significant improvement (memory returned to OS)

## Why Alternative Allocators Don't Help

### mimalloc Design Philosophy

- **Goal**: Fast allocations by caching freed memory
- **Strategy**: Thread-local heaps, aggressive retention
- **Trade-off**: Higher RSS for lower latency

For our use case:
- ✅ DHAT shows similar allocation patterns
- ❌ RSS grows because cached memory isn't returned to OS

### jemalloc Design Philosophy

- **Goal**: Balance performance and fragmentation
- **Strategy**: Size-class bins, delayed coalescing
- **Trade-off**: Better than mimalloc, worse than glibc+trim

For our use case:
- ✅ DHAT shows similar allocation patterns
- ❌ RSS grows due to retention policies

## Recommendations

### 1. Use DHAT for Allocation Profiling

DHAT is excellent for:
- Finding allocation hotspots
- Reducing total allocations
- Optimizing allocation patterns

**Sprint 6-7 achievements** (baseline → updated):
- -48.5% total allocations
- -73% allocation blocks
- Major reduction in HiGHS and Rust allocations

### 2. Use RSS for Memory Footprint

RSS is the metric that matters for:
- Memory-constrained systems
- OOM (Out of Memory) prevention
- Long-running processes

**Sprint 8-9 achievement**:
- glibc + malloc_trim: Stable 250 MB RSS
- mimalloc/jemalloc: Growing 775-838 MB RSS (rejected)

### 3. Keep glibc as Default

The data definitively shows:
- **DHAT**: All allocators have similar patterns (46-47 GB)
- **RSS**: glibc is 3x better (250 MB vs 775-838 MB)

### 4. malloc_trim is Critical

Without malloc_trim (tested in earlier sprints):
- glibc RSS would grow similarly to alternative allocators
- The Model lifecycle alone isn't enough

With malloc_trim (current):
- glibc returns freed HiGHS memory to OS
- RSS stable after warmup

## Conclusion

| Metric | Purpose | Finding |
|--------|---------|---------|
| **DHAT Total Allocations** | Optimize allocation patterns | 46.87 GB (Sprint 6-7 reduced from 82 GB) |
| **RSS (Memory Footprint)** | Optimize memory usage | 250 MB stable with glibc + malloc_trim |

**Both metrics are important, but they measure different things.**

The combination of:
1. Reduced allocations (Sprint 6-7)
2. Per-iteration Model lifecycle (Sprint 8)
3. malloc_trim() after finalize (Sprint 8)

...achieves both low allocations AND low RSS with the system allocator.

---

*Generated: 2026-01-01*
*Epic 5, Sprint 9: Memory Optimization Complete*
