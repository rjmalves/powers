# Epic 01c: Memory Module Cleanup and Cut Buffer Hardening

## Summary

Clean up the `src/memory` module by removing ~3,100 lines of unused/broken code and hardening the `CutComputationBuffers` to enforce zero-allocation guarantees. The current module contains dead abstractions (`SizingInfo`, `ThreadLocalBuffers`, `DeepSizeEstimate`) that are never used in production and have incorrect assumptions about state dimensions.

## Motivation

Analysis in `MEMORY_MODULE_ANALYSIS.md` revealed:

| Issue | Severity | Lines Affected |
|-------|----------|----------------|
| `SizingInfo` never used in production | Critical | 1,574 lines |
| `DeepSizeEstimate` has broken assumptions | High | 474 lines |
| `ThreadLocalBuffers` never initialized | High | ~460 lines |
| `CutComputationBuffers` allows silent reallocation | Medium | ~130 lines |

**Key Problem**: The `CutComputationBuffers` currently allows dynamic growth if preallocated sizes are wrong:

```rust
// reset_for_cut() silently reallocates if capacity insufficient
while self.contributions_outer.len() < num_scenarios {
    self.contributions_outer.push(Vec::with_capacity(state_dim));  // ALLOCATION!
}
```

This violates our goal of 100% memory determinism.

## Scope

### Included

1. Remove unused code: `SizingInfo`, `NodeSizing`, `MemoryBreakdown`, `DeepSizeEstimate`
2. Remove unused buffers: `Buffer<T>`, `BufferPool<T>`, `ThreadLocalBuffers`
3. Remove `DeepSizeEstimate` implementations from `cut.rs`, `fcf.rs`, `state.rs`
4. Harden `CutComputationBuffers` to panic on insufficient capacity
5. Extract correct state dimensions from node data (handle lagged inflows)
6. Initialize cut buffers in all Rayon worker threads

### Excluded

- Changes to HiGHS preallocation (Epic 1, 1b)
- FCF preallocation (Epic 2)
- Handler SoA blocks (Epic 3)

## Dependencies

- **Requires**: Epic 1b complete ✅
- **Enables**: Epic 2 (cleaner codebase)

## Acceptance Criteria

- [x] `src/memory/sizing.rs` removed (1,574 lines)
- [x] `src/memory/deep_sizing.rs` removed (474 lines)
- [x] `Buffer<T>`, `BufferPool<T>`, `ThreadLocalBuffers` removed from `buffers.rs`
- [x] `DeepSizeEstimate` impls removed from `cut.rs`, `fcf.rs`, `state.rs`
- [x] `CutComputationBuffers` panics if capacity exceeded
- [x] Cut buffer initialization uses correct state dimensions (including inflow lags)
- [x] All Rayon worker threads have initialized buffers
- [x] All examples pass with identical results
- [x] No clippy warnings
- [x] Code compiles

## Technical Approach

### Phase 1: Remove Dead Code

Remove files and trait implementations that are never called in production.

### Phase 2: Harden CutComputationBuffers

1. Add `max_state_dim` and `max_scenarios` fields to track preallocated capacity
2. Replace silent reallocation with panic in `reset_for_cut()`
3. Add `validate_capacity()` method for debugging

### Phase 3: Fix Initialization

1. Extract true `max_state_dim` from node data (accounting for inflow lags)
2. Use `rayon::broadcast()` to initialize buffers in all worker threads
3. Remove lazy initialization fallback (panic if uninitialized)

## Success Metrics

- [x] ~3,100 lines of code removed (actual: ~2,987 lines removed)
- [x] `src/memory` reduced from 3,424 lines to ~300 lines (actual: 437 lines)
- [x] Zero allocations in cut computation (validated with profiler)
- [x] All examples produce identical results
- [x] No performance regression

## Estimated Effort

**Duration**: 1 sprint (3-5 days)
**Story Points**: 8

## Completion Date

**Completed**: 2025-12-26
