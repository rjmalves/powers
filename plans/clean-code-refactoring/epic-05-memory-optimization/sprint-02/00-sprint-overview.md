# Sprint 2: Training Loop Integration

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## Goals

1. Wire the new parallel-then-sequential path into `backward_pass.rs`
2. Validate correctness with golden tests
3. Benchmark to confirm parallel execution is preserved
4. Verify zero allocations with DHAT profiling

---

## Prerequisites

- Sprint 1 complete (all T-060 through T-064)
- `compute_cuts_parallel_into_slots()` method available

---

## Tickets

| ID | Title | Points | Dependencies | Status |
|----|-------|--------|--------------|--------|
| [T-065](./ticket-065-wire-backward-pass.md) | Update backward_pass.rs to use staging path | 5 | T-064 | ⬜ |
| [T-066](./ticket-066-golden-tests.md) | Golden tests validation | 2 | T-065 | ⬜ |
| [T-067](./ticket-067-benchmark-parallel.md) | Benchmark parallel vs sequential | 3 | T-065 | ⬜ |
| [T-068](./ticket-068-dhat-profiling.md) | DHAT profiling to verify zero allocations | 3 | T-065 | ⬜ |

**Total Points**: 13

---

## Key Changes

### Before (Current)

```rust
// backward_pass.rs - Uses allocating parallel path
let phase1 = processor.compute_cuts_parallel(stage_ctx)?;
let phase2 = processor.select_cuts_batch(phase1.cut_data, stage_ctx, fcf_graph)?;
```

### After (Target)

```rust
// backward_pass.rs - Uses zero-allocation parallel path
let phase1 = processor.compute_cuts_parallel_into_slots(stage_ctx, fcf_graph)?;
let phase2 = processor.select_cuts_from_slots(phase1.slots, stage_ctx, fcf_graph)?;
```

---

## Validation Strategy

1. **Golden tests**: Bit-for-bit identical results to baseline
2. **Unit tests**: All 549+ tests pass
3. **Benchmarks**: No performance regression (should be faster)
4. **Profiling**: DHAT confirms zero CutData allocations

---

## Definition of Done

- [ ] Training loop uses new path
- [ ] Golden tests pass
- [ ] Benchmarks show ≥ parity (ideally faster)
- [ ] DHAT shows zero allocations in cut computation
- [ ] All tests pass
- [ ] Deprecated methods marked
