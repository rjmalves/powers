# Sprint 9: RSS Stabilization via Allocator Strategy

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ✅ Complete (Unexpected Positive Outcome)

---

## Executive Summary

Sprint 8 revealed that the per-iteration Model architecture is correctly implemented, but initial hypothesis was that RSS growth was due to glibc malloc not returning freed memory. **This hypothesis was WRONG.**

### Critical Finding

**glibc is the BEST allocator for this workload!**

Testing revealed:
- **glibc**: Stable RSS ~250 MB after warmup (±1 MB)
- **mimalloc**: Growing RSS, final ~838 MB (3.3x worse)
- **jemalloc**: Growing RSS, final ~775 MB (3.1x worse)

The `malloc_trim(0)` call added in Sprint 8 successfully returns freed HiGHS memory to the OS.

## Goals (Revised)

1. ~~**Primary**: Achieve stable RSS between iterations~~ ✅ Already achieved with glibc!
2. ~~**Secondary**: Make the best-performing allocator the default~~ ✅ glibc is already default
3. **Tertiary**: Document memory behavior ✅ See [ALLOCATOR_COMPARISON.md](../../../../docs/ALLOCATOR_COMPARISON.md)

## Success Criteria

- [x] RSS at iteration N end ≈ RSS at iteration N+1 start (within 5% tolerance) ✅ glibc achieves this
- [x] RSS does not grow monotonically after iteration 2 (warmup complete) ✅ glibc stable after iter 2
- [x] Best allocator becomes default in Cargo.toml ✅ glibc already default (no change needed)
- [x] All existing tests pass with new default allocator ✅ No change to allocator
- [x] Performance benchmarks show no regression ✅ No change to allocator

---

## Ticket Status (Revised)

| ID | Title | Points | Status | Notes |
|----|-------|--------|--------|-------|
| T-130 | Create RSS measurement test harness | 3 | ✅ Complete | Harness works |
| T-131 | Test mimalloc allocator RSS behavior | 3 | ✅ Complete | **Result: NOT RECOMMENDED** |
| T-132 | Add jemalloc as optional dependency | 2 | ✅ Complete | Dependency added |
| T-133 | Test jemalloc allocator RSS behavior | 3 | ✅ Complete | **Result: NOT RECOMMENDED** |
| T-134 | Test malloc_trim after finalize_iteration | 2 | ⏭️ Skipped | Already implemented in Sprint 8 |
| T-135 | Compare allocator results and select winner | 2 | ✅ Complete | **Winner: glibc** |
| T-136 | Make winning allocator the default | 3 | ⏭️ Skipped | glibc already default |
| T-137 | Validate all tests pass with new default | 2 | ⏭️ Skipped | No change needed |
| T-138 | Performance benchmark with new allocator | 3 | ⏭️ Skipped | No change needed |
| T-139 | Document allocator configuration | 2 | ✅ Complete | See ALLOCATOR_COMPARISON.md |
| T-140 | Add RSS stability CI check | 3 | 📋 Optional | Can add later |

**Completed**: 18 points (T-130, T-131, T-132, T-133, T-135, T-139)
**Skipped**: 10 points (T-134, T-136, T-137, T-138 - no longer needed)

---

## Key Results

### RSS Comparison (8 iterations, 05-large-scale-brazilian)

| Allocator | Initial RSS | Final RSS | Stable? | Recommendation |
|-----------|-------------|-----------|---------|----------------|
| glibc | 205 MB | 251 MB | ✅ Yes | **Default** |
| mimalloc | 269 MB | 838 MB | ❌ No | Not recommended |
| jemalloc | 231 MB | 775 MB | ❌ No | Not recommended |

### Why glibc Wins

1. **Model Lifecycle Works**: Per-iteration Model drop releases HiGHS memory
2. **malloc_trim Effective**: `malloc_trim(0)` returns memory to OS
3. **No Over-Retention**: glibc doesn't aggressively cache like mimalloc/jemalloc

### Why Alternative Allocators Fail

1. **Aggressive Caching**: Designed to cache freed memory for performance
2. **Thread-Local Heaps**: Don't immediately return memory to global pool
3. **Wrong Use Case**: Optimized for short-lived allocations, not memory return

---

## Documentation

- [ALLOCATOR_COMPARISON.md](../../../../docs/ALLOCATOR_COMPARISON.md) - RSS comparison and recommendation
- [DHAT_VS_RSS_ANALYSIS.md](../../../../docs/DHAT_VS_RSS_ANALYSIS.md) - Why DHAT and RSS tell different stories
- [SPRINT_09_FINAL_REPORT.md](../../../../docs/SPRINT_09_FINAL_REPORT.md) - Comprehensive final report

---

## Definition of Done

- [x] All essential tickets complete (T-130 through T-135, T-139)
- [x] RSS stable between iterations ✅ (already was with glibc!)
- [x] Best allocator is default ✅ (glibc already default)
- [x] Documentation updated ✅ (ALLOCATOR_COMPARISON.md created)
- [ ] CHANGELOG.md updated (pending)

---

## Lessons Learned

1. **Test Before Assuming**: Initial hypothesis about glibc was wrong
2. **Simpler is Sometimes Better**: No need for fancy allocators
3. **malloc_trim Works**: The Sprint 8 addition was the key fix
4. **Measure Everything**: Per-iteration RSS data was crucial
