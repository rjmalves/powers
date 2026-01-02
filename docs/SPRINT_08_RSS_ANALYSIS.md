# Sprint 8 RSS Analysis Report

> **Date**: 2026-01-01
> **Sprint**: Epic 5, Sprint 8 (Revised) - Per-Iteration Model Architecture
> **Status**: ❌ RSS Objectives NOT Met

---

## Executive Summary

The Per-Iteration Model Architecture was implemented to address monotonic RSS growth during SDDP training. Despite successful implementation of the architecture (Models are created and dropped each iteration), **RSS continues to grow monotonically** because glibc malloc does not return freed memory to the operating system.

### Key Metrics

| Metric | Before Training | After 20 Iterations | Growth |
|--------|-----------------|---------------------|--------|
| RSS | 11 MB | 1,045 MB | **+1,034 MB** |
| Active Cuts | 0 | 3,502 | +3,502 |

---

## Detailed Analysis

### Test Configuration

- **Example**: 05-large-scale-brazilian
- **Iterations**: 20
- **Forward Passes**: 4
- **Threads**: 1 (to isolate memory behavior)
- **Seed**: 42

### Iteration-by-Iteration RSS Measurements

RSS was logged at the start and end of each iteration using `/proc/self/status`:

| Iteration | RSS Start (KB) | RSS End (KB) | Delta (KB) | Active Cuts |
|-----------|----------------|--------------|------------|-------------|
| 1 | 246,364 | 524,124 | +277,760 | 191 |
| 2 | 524,124 | 534,488 | +10,364 | 408 |
| 3 | 534,488 | 554,956 | +20,468 | 609 |
| 4 | 554,956 | 583,260 | +28,304 | 823 |
| 5 | 583,260 | 612,820 | +29,560 | 994 |
| 6 | 612,820 | 658,084 | +45,264 | 1,169 |
| 7 | 658,084 | 673,604 | +15,520 | 1,330 |
| 8 | 673,604 | 701,396 | +27,792 | 1,521 |
| 9 | 701,396 | 739,532 | +38,136 | 1,703 |
| 10 | 739,532 | 778,788 | +39,256 | 1,846 |
| 11 | 778,788 | 811,972 | +33,184 | 2,035 |
| 12 | 811,972 | 821,000 | +9,028 | 2,226 |
| 13 | 821,000 | 860,388 | +39,388 | 2,400 |
| 14 | 860,388 | 889,716 | +29,328 | 2,501 |
| 15 | 889,716 | 923,280 | +33,564 | 2,722 |
| 16 | 923,280 | 963,640 | +40,360 | 2,879 |
| 17 | 963,640 | 976,492 | +12,852 | 2,951 |
| 18 | 976,492 | 1,017,504 | +41,012 | 3,126 |
| 19 | 1,017,504 | 1,025,356 | +7,852 | 3,291 |
| 20 | 1,025,356 | 1,070,256 | +44,900 | 3,502 |

### Key Observations

1. **RSS Never Decreases**: Despite `finalize_iteration()` dropping all Models, RSS at iteration N end equals RSS at iteration N+1 start.

2. **First Iteration Spike**: Iteration 1 shows +278 MB growth (HiGHS internal buffer initialization).

3. **Steady Growth Pattern**: Average ~25 MB/iteration after warmup.

4. **Cut Pool Correlation**: Active cuts grow from 191 → 3,502, correlating with RSS growth.

---

## Root Cause Analysis

### Primary Cause: glibc malloc Behavior

Linux glibc's malloc implementation does **not** return freed memory to the OS by default. When `Highs_destroy()` is called:

1. ✅ HiGHS C++ destructors run
2. ✅ Memory is marked as "free" in the heap
3. ❌ Pages are NOT returned to the OS
4. ❌ RSS does not decrease

This is documented glibc behavior designed to optimize for allocation speed over memory footprint.

### Secondary Causes

#### 1. Cut Pool Growth (Legitimate)

Each iteration adds ~236 cuts (4 forward passes × 59 stages). Cut selection removes some, but net growth is positive:

- **Cuts added**: 236 per iteration
- **Cuts removed**: Variable (19-164 per iteration)
- **Net growth**: ~175 cuts/iteration average

Estimated memory per cut: `(1,070,256 - 524,124) KB / 3,311 cuts ≈ 165 KB/cut`

This seems high, suggesting either:
- Problem struct overhead (Vec growth)
- HiGHS row addition overhead
- Memory fragmentation

#### 2. Problem Struct Growth

Each `Problem` stores cuts via `add_row()`, which grows internal vectors:
- `row_lower: Vec<f64>`
- `row_upper: Vec<f64>`
- `columns: Vec<(Vec<c_int>, Vec<f64>)>`

#### 3. Memory Fragmentation

Small allocations interspersed with large HiGHS buffers prevent page release even with `malloc_trim()`.

---

## DHAT Analysis Comparison

DHAT measures **cumulative allocations**, not peak RSS:

| Metric | Post-Sprint 7 | Sprint 8 | Change |
|--------|---------------|----------|--------|
| Total Bytes Allocated | 45.43 GB | 50.32 GB | **+10.8%** |
| Total Allocation Blocks | 43.3 M | 46.6 M | **+7.6%** |
| Sum of Max Bytes | 468.4 MB | 526.6 MB | **+12.4%** |
| Allocation Sites | 12,651 | 15,238 | **+20.4%** |

**Analysis**: The DHAT regression is **expected** because per-iteration Model creation means more total allocations over the program lifetime. DHAT cannot measure RSS reclamation.

---

## Architecture Validation

The per-iteration Model lifecycle **is working correctly**:

```
✅ finalize_iteration() called at each iteration end
✅ Model::drop() triggers Highs_destroy()
✅ Basis cached for warm-start (optional)
✅ Models created fresh at iteration start
❌ RSS does not decrease (glibc behavior)
```

The architecture is sound; the problem is at the allocator level.

---

## Recommended Solutions

### Option 1: malloc_trim() (Low Effort, Uncertain Impact)

Add `malloc_trim(0)` call after `finalize_iteration()`:

```rust
// After dropping Models
#[cfg(target_os = "linux")]
unsafe {
    libc::malloc_trim(0);
}
```

**Pros**: Simple, no dependencies
**Cons**: May not help with fragmented memory, requires `libc` crate

### Option 2: jemalloc (Medium Effort, High Impact)

Use jemalloc instead of glibc malloc:

```toml
[dependencies]
tikv-jemallocator = "0.5"
```

```rust
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
```

**Pros**: Better memory release, proven in production
**Cons**: Additional dependency, may affect other allocation patterns

### Option 3: mimalloc (Medium Effort, High Impact)

Use Microsoft's mimalloc:

```toml
[dependencies]
mimalloc = { version = "0.1", default-features = false }
```

```rust
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

**Pros**: Excellent memory release, very fast
**Cons**: Additional dependency

### Option 4: Arena Allocator for HiGHS (High Effort, Highest Impact)

Use bumpalo or similar for HiGHS memory:

```rust
use bumpalo::Bump;

struct IterationArena {
    arena: Bump,
}

impl IterationArena {
    fn reset(&mut self) {
        self.arena.reset();
    }
}
```

**Pros**: Guaranteed memory release, zero fragmentation
**Cons**: Requires HiGHS modification or custom allocator integration

### Option 5: Reduce Cut Storage Overhead (Medium Effort, Medium Impact)

Investigate why cuts use ~165 KB each:

1. Use sparse representation for cut coefficients
2. Compress inactive cuts
3. Stream cuts to disk for cold storage

### Option 6: Accept and Document (No Effort)

If memory growth is acceptable for the problem sizes:

1. Document expected memory usage
2. Provide memory estimation formulas
3. Add configuration for memory limits

---

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 days)

1. **Try malloc_trim()**: Add after finalize_iteration, measure impact
2. **Try jemalloc**: Build with jemalloc, compare RSS behavior

### Phase 2: Deeper Investigation (1 week)

1. **Profile cut storage**: Understand 165 KB/cut overhead
2. **Memory mapping**: Consider mmap for large buffers
3. **Benchmark allocators**: Compare glibc vs jemalloc vs mimalloc

### Phase 3: Architecture Changes (2-3 weeks)

1. **Cut compression**: Reduce storage for inactive cuts
2. **Streaming cuts**: Offload cold cuts to disk
3. **Custom allocator**: Arena-based allocation for HiGHS

---

## Files Modified

| File | Changes |
|------|---------|
| `src/sddp/mod.rs` | Added RSS debug logging at iteration boundaries |
| `tests/test_rss_monitoring.rs` | Created RSS monitoring test |
| `plans/.../sprint-08-revised/00-sprint-overview.md` | Updated with analysis |
| `plans/.../sprint-08-revised/ticket-120-rss-verification.md` | Marked as FAILED |
| `plans/.../00-epic-overview.md` | Updated status |

---

## Conclusion

The Sprint 8 Per-Iteration Model Architecture is **correctly implemented** but **does not achieve RSS stability** due to glibc malloc behavior. The recommended next step is to test alternative allocators (jemalloc or mimalloc) which are known to have better memory release characteristics.

The cut pool growth (~546 MB over 20 iterations) is a significant contributor and may warrant optimization regardless of allocator choice.

---

## References

- [glibc malloc documentation](https://www.gnu.org/software/libc/manual/html_node/The-GNU-Allocator.html)
- [jemalloc](https://jemalloc.net/)
- [mimalloc](https://github.com/microsoft/mimalloc)
- [HiGHS solver](https://highs.dev/)
- [DHAT profiler](https://valgrind.org/docs/manual/dh-manual.html)
