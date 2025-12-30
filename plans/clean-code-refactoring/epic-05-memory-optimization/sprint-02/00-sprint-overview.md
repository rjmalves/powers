# Sprint 2: Training Loop Integration

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ✅ Complete

---

## Goals

1. Wire the new parallel-then-sequential path into `backward_pass.rs`
2. Validate correctness with golden tests
3. Benchmark to confirm parallel execution is preserved
4. Verify zero allocations with DHAT profiling

---

## Prerequisites

- Sprint 1 complete (all T-060 through T-064) ✅
- `compute_cuts_parallel_into_slots()` method available ✅

---

## Tickets

| ID | Title | Points | Dependencies | Status |
|----|-------|--------|--------------|--------|
| [T-065](./ticket-065-wire-backward-pass.md) | Update backward_pass.rs to use staging path | 5 | T-064 | ✅ |
| [T-066](./ticket-066-golden-tests.md) | Golden tests validation | 2 | T-065 | ✅ |
| [T-067](./ticket-067-benchmark-parallel.md) | Benchmark parallel vs sequential | 3 | T-065 | ⏭️ Deferred |
| [T-068](./ticket-068-dhat-profiling.md) | DHAT profiling to verify zero allocations | 3 | T-065 | ⏭️ Deferred |

**Total Points**: 13

**Note**: T-067 and T-068 are deferred as they require specialized tooling (DHAT, criterion benchmarks) that are validation tasks rather than core implementation.

---

## Key Changes

### Before (Current)

```rust
// backward_pass.rs - Uses sequential zero-allocation path
let phase1 = processor.compute_cuts_into_slots(stage_ctx, fcf_graph)?;
let phase2 = processor.select_cuts_from_slots(phase1.slots, stage_ctx, fcf_graph)?;
```

### After (Implemented)

```rust
// backward_pass.rs - Uses parallel-then-sequential zero-allocation path
let phase1 = processor.compute_cuts_parallel_into_slots(stage_ctx, fcf_graph)?;
let phase2 = processor.select_cuts_from_slots(phase1.slots, stage_ctx, fcf_graph)?;
```

---

## Validation Strategy

1. **Golden tests**: Bit-for-bit identical results to baseline ✅
2. **Unit tests**: All 554 tests pass ✅
3. **Benchmarks**: Deferred to Epic 7
4. **Profiling**: Deferred to Epic 7

---

## Definition of Done

- [x] Training loop uses new parallel path
- [x] All tests pass (554)
- [x] Code is clippy-clean
- [ ] Benchmarks show ≥ parity (deferred to Epic 7)
- [ ] DHAT shows zero allocations (deferred to Epic 7)
