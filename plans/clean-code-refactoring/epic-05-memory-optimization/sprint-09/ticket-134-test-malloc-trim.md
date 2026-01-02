# [T-134] Test malloc_trim After finalize_iteration

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 9: RSS Stabilization](./00-sprint-overview.md)
> **Dependencies**: T-130
> **Blocks**: T-135

---

## Context

### Background

`malloc_trim(0)` is a glibc-specific function that attempts to return freed memory to the OS. This is a low-effort option that doesn't require changing the allocator. However, its effectiveness is limited when memory is fragmented.

### Current State

- No malloc_trim calls in the codebase
- glibc is the default allocator
- RSS grows monotonically despite Models being dropped

## Specification

### Changes Required

1. Add `libc` crate as dependency (for `malloc_trim`)
2. Call `malloc_trim(0)` after `finalize_iteration()` in training loop
3. Measure RSS behavior with this change

### Implementation

```rust
// In src/sddp/mod.rs, after finalize_iteration() calls

// Attempt to release freed memory to the OS (glibc only)
#[cfg(all(target_os = "linux", not(any(feature = "mimalloc", feature = "jemalloc"))))]
unsafe {
    libc::malloc_trim(0);
}
```

### Expected Behavior

- **Best case**: RSS decreases after malloc_trim, stabilizes between iterations
- **Likely case**: Partial improvement, fragmented memory prevents full release
- **Worst case**: No improvement (fragmentation too severe)

### Metrics to Capture

| Metric | glibc (no trim) | glibc + malloc_trim |
|--------|-----------------|---------------------|
| RSS after 20 iter | 1,045 MB | ? |
| Avg inter-iteration delta | +25 MB | ? |
| Is stable after warmup | No | ? |

## Acceptance Criteria

- [ ] `libc` crate added to dependencies
- [ ] `malloc_trim(0)` called after finalize_iteration (conditionally)
- [ ] RSS stability test runs with malloc_trim
- [ ] Results documented and compared to baseline
- [ ] Result recorded: PASS (stable) or FAIL

## Implementation Guide

### Suggested Approach

1. Add `libc` to Cargo.toml
2. Add malloc_trim call after finalize_iteration in training loop
3. Build without allocator features (use glibc)
4. Run RSS stability test
5. Document results

### Key Files to Modify

- `Cargo.toml`: Add libc dependency
- `src/sddp/mod.rs`: Add malloc_trim call after finalize_iteration

### Cargo.toml Changes

```toml
[dependencies]
libc = "0.2"
```

### src/sddp/mod.rs Changes

```rust
// After the finalize_iteration loop (around line 2274)
for handler in coordinator.handlers_mut() {
    handler.finalize_iteration(&lifecycle_config);
}

// Attempt to release freed memory to the OS (glibc only, no effect with other allocators)
#[cfg(all(target_os = "linux", not(any(feature = "mimalloc", feature = "jemalloc"))))]
{
    // SAFETY: malloc_trim is safe to call, it only affects the calling process's heap
    unsafe {
        libc::malloc_trim(0);
    }
    log::trace!("Called malloc_trim(0) after finalize_iteration");
}
```

### Test Commands

```bash
# Build without allocator features (uses glibc + malloc_trim)
cargo build --release

# Run RSS stability test
cargo test --release test_rss_stability -- --nocapture 2>&1 | tee malloc_trim_rss.log
```

### Pitfalls to Avoid

- ⚠️ malloc_trim is glibc-specific, not available on all platforms
- ⚠️ Only call when not using alternative allocators
- ⚠️ Effect may be minimal with fragmented memory

## Testing Requirements

### Integration Tests

- [ ] RSS stability test with malloc_trim
- [ ] All existing tests pass
- [ ] No performance regression

### Unit Tests

- [ ] Verify malloc_trim is only called on Linux without alternative allocators

## Documentation Requirements

- [ ] Create `docs/MALLOC_TRIM_RSS_ANALYSIS.md` with results
- [ ] Document conditional compilation in code comments

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple change, clear measurement procedure

## Definition of Done

- [ ] malloc_trim call implemented
- [ ] RSS behavior measured and documented
- [ ] Comparison to other allocators complete
- [ ] Result (PASS/FAIL) recorded
