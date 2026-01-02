# Allocator Comparison Analysis

## Summary

**CRITICAL FINDING**: glibc is the best allocator for this workload!

Based on testing, **system allocator (glibc)** is recommended as the default because it shows **stable RSS** after warmup, while both mimalloc and jemalloc show **continuous RSS growth**.

This is the opposite of initial expectations.

## Test Configuration

- **Example**: 05-large-scale-brazilian
- **Iterations**: 8
- **Forward passes**: 4  
- **Threads**: 1
- **Build**: release mode with `--log-level debug`

## Results Summary

| Allocator | Initial RSS | Final RSS | Behavior | Recommendation |
|-----------|-------------|-----------|----------|----------------|
| glibc | 205 MB | 251 MB | **Stable** ±1 MB | ✅ **Default** |
| mimalloc | 269 MB | 838 MB | Growing ~40-60 MB/iter | ❌ Not recommended |
| jemalloc | 231 MB | 775 MB | Growing ~40-60 MB/iter | ❌ Not recommended |

## Detailed Per-Iteration Data

### glibc (System Allocator)

| Iter | Start (MB) | End (MB) | Delta (KB) |
|------|------------|----------|------------|
| 1 | 204.6 | 247.4 | +43,884 |
| 2 | 247.4 | 248.2 | +836 |
| 3 | 248.2 | 249.0 | +760 |
| 4 | 249.0 | 250.2 | +1,252 |
| 5 | 250.2 | 249.5 | -672 |
| 6 | 249.5 | 251.1 | +1,632 |
| 7 | 251.1 | 250.9 | -204 |
| 8 | 250.9 | 250.7 | -248 |

**Observation**: After iteration 2, RSS is stable around 250 MB with ±1.5 MB variation.

### mimalloc

| Iter | Start (MB) | End (MB) | Delta (KB) |
|------|------------|----------|------------|
| 1 | 268.6 | 493.8 | +230,548 |
| 2 | 493.8 | 552.7 | +60,284 |
| 3 | 552.7 | 606.7 | +55,292 |
| 4 | 606.7 | 653.2 | +47,668 |
| 5 | 653.2 | 687.4 | +34,976 |
| 6 | 687.4 | 760.7 | +75,124 |
| 7 | 760.7 | 798.4 | +38,624 |
| 8 | 798.4 | 838.1 | +40,600 |

**Observation**: RSS grows continuously, ~40-60 MB per iteration. Final RSS is 3.3x glibc.

### jemalloc

| Iter | Start (MB) | End (MB) | Delta (KB) |
|------|------------|----------|------------|
| 1 | 230.8 | 449.3 | +223,784 |
| 2 | 449.3 | 481.6 | +33,032 |
| 3 | 481.6 | 535.8 | +55,544 |
| 4 | 535.8 | 581.3 | +46,616 |
| 5 | 581.3 | 628.4 | +48,208 |
| 6 | 628.2 | 664.5 | +37,116 |
| 7 | 664.5 | 728.1 | +65,184 |
| 8 | 728.1 | 775.4 | +48,416 |

**Observation**: RSS grows continuously, similar pattern to mimalloc. Final RSS is 3.1x glibc.

## Analysis

### Why is glibc Better?

1. **Model Lifecycle Works**: The per-iteration Model drop in `finalize_iteration()` 
   effectively releases HiGHS memory back to glibc.

2. **Memory Return to OS**: glibc's `malloc_trim(0)` call (added in Sprint 8) 
   successfully returns freed memory to the OS.

3. **No Over-Retention**: glibc doesn't aggressively cache freed memory like 
   mimalloc and jemalloc do.

### Why Do Alternative Allocators Perform Worse?

1. **Aggressive Caching**: Both mimalloc and jemalloc are designed to cache 
   freed memory for faster subsequent allocations. This is beneficial for 
   short-lived allocations but detrimental for our use case where we want 
   to actually return memory to the OS.

2. **Thread-Local Caches**: Both allocators use thread-local heaps that don't 
   immediately return memory to the global pool.

3. **Fragmentation Handling**: Their defragmentation strategies may not trigger 
   as quickly as needed for our allocation pattern.

## Recommendations

### 1. Keep System Allocator as Default

Do NOT enable mimalloc or jemalloc by default. The current configuration 
with glibc + `malloc_trim(0)` provides the best RSS behavior.

### 2. Feature Flags for Experimentation

Keep mimalloc and jemalloc as optional features for users who want to 
experiment:

```bash
# Default (recommended)
cargo build --release

# Optional (not recommended for memory-constrained systems)
cargo build --release --features mimalloc
cargo build --release --features jemalloc
```

### 3. Document Memory Behavior

Update documentation to explain that:
- glibc provides stable RSS after warmup (~250 MB for this workload)
- Alternative allocators may use 3x more memory
- The Model lifecycle and `malloc_trim(0)` are critical for memory efficiency

## Conclusion

The Sprint 8 work on per-iteration Model lifecycle combined with `malloc_trim(0)` 
has made the system allocator (glibc) the optimal choice. This is an unexpected 
but positive outcome - we don't need to change the default allocator.

**Action Items**:
- [x] T-131: Test mimalloc ✅ (shows RSS growth - not recommended)
- [x] T-133: Test jemalloc ✅ (shows RSS growth - not recommended)
- [ ] T-134: Skip - malloc_trim already implemented and working
- [x] T-135: Compare allocators ✅ (glibc wins)
- [ ] T-136: Skip - glibc is already the default
- [ ] Update documentation with findings

---

*Generated: 2026-01-01*
*Sprint 9: RSS Stabilization via Allocator Strategy*

---

## Understanding DHAT vs RSS

**Important**: DHAT measures total allocations over the program's lifetime, while RSS measures actual physical memory usage at any point in time.

See [DHAT_VS_RSS_ANALYSIS.md](./DHAT_VS_RSS_ANALYSIS.md) for a detailed explanation of why:
- DHAT shows ~47 GB allocations for ALL allocators (glibc, mimalloc, jemalloc)
- RSS shows 250 MB for glibc, but 775-838 MB for alternative allocators

The key insight: glibc + `malloc_trim()` returns memory to the OS, while alternative allocators cache it for performance.

For our long-running SDDP workload, **low RSS is more important than fast allocations**.

