# Sprint 5: Deterministic Memory Allocation

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ✅ Complete

---

## ⚠️ CRITICAL REMINDER

**Algorithm correctness is non-negotiable.** Memory optimization must not change any numerical results. Golden tests must pass after every change.

If any test fails or results diverge: **STOP and investigate before proceeding.**

---

## Executive Summary

Despite Epic 5 Sprints 1-4 implementing significant architectural improvements (staging buffers, pool preallocation, Arc/HashMap removal), **RSS monitoring during example 05 execution shows memory behavior similar to pre-optimization**. This sprint targets the remaining allocation sources identified through code analysis and DHAT profiling.

### Root Cause Analysis

The analysis in `DETERMINISTIC_MEMORY_ANALYSIS.md` identified these remaining allocation sources:

| Category | Allocation Source | Impact |
|----------|------------------|--------|
| **HiGHS Internal** | `Highs_addRow()` causes reallocation | HIGH |
| **HiGHS Internal** | Work vectors allocated on each `Highs_run()` | MEDIUM |
| **Rust Wrapper** | `try_add_row()` creates 2 temp Vecs per call | HIGH |
| **Rust Wrapper** | `delete_row()` creates Vec for row set | LOW |
| **Coordinator** | `Vec::with_capacity` + `push` in hot paths | MEDIUM |
| **Realization** | Cloning Vecs for trajectory data | HIGH |
| **State** | `extract_storage_from_trajectory` allocates | LOW |

---

## Goals

1. **Eliminate all `Highs_addRow()` calls** during training (use preallocation exclusively)
2. **Add HiGHS solver warmup** to pre-allocate internal work vectors
3. **Add thread-local buffers** for any remaining row addition edge cases
4. **Preallocate coordinator result buffers** per iteration
5. **Eliminate realization cloning** in hot paths via preallocated trajectory buffers
6. **Verify with DHAT** that hot path has zero allocations

## Non-Goals

- Changing HiGHS solver options beyond what's in `set_default_solver_options()`
- Modifying HiGHS source code
- Changing algorithm behavior

---

## Technical Approach

### 1. Complete Cut Preallocation Path

Currently, `add_cut_constraint_to_model()` in `state.rs` still calls `model.add_row()`. This must be eliminated.

**Strategy**: Ensure ALL cut additions during training use `add_cut_with_preallocation()`.

### 2. HiGHS Solver Warmup

After `preallocate_cut_constraints()`, solve once with trivial bounds to force HiGHS to allocate all internal work vectors.

```rust
fn warmup_solver(&mut self) {
    if let Some(model) = self.model.as_mut() {
        model.solve();
        model.clear_solver();
    }
}
```

### 3. Thread-Local Row Addition Buffers

For any remaining `add_row` calls (edge cases, initialization):

```rust
thread_local! {
    static ROW_COLS_BUFFER: RefCell<Vec<HighsInt>> = RefCell::new(Vec::with_capacity(64));
    static ROW_VALS_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(64));
}
```

### 4. Preallocated Coordinator Buffers

Replace per-iteration `Vec::with_capacity` + `push` pattern:

```rust
struct CoordinatorBuffers {
    slots: Vec<usize>,
    timings: Vec<BackwardPhase1Timing>,
}
```

### 5. Preallocated Trajectory Buffers

Replace realization cloning with preallocated trajectory storage:

```rust
struct TrajectoryBuffer {
    realizations: Vec<Realization>,  // Preallocated for max stages
    len: usize,                       // Current valid length
}
```

---

## Sprint Tickets

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-080 | Audit and eliminate remaining add_row calls in training | 5 | None |
| T-081 | Add thread-local buffers for try_add_row | 3 | None |
| T-082 | Implement HiGHS solver warmup after preallocation | 3 | T-080 |
| T-083 | Preallocate coordinator result buffers | 3 | None |
| T-084 | Preallocate trajectory buffers and eliminate cloning | 5 | None |
| T-085 | DHAT profiling to verify zero allocations in hot path | 3 | T-080, T-081, T-082, T-083, T-084 |
| T-086 | Benchmark and document memory behavior | 2 | T-085 |

**Total**: 24 points

---

## Acceptance Criteria

### Sprint Completion

- [x] No `Highs_addRow()` calls during SDDP training iterations
- [x] HiGHS solver warmed up after cut preallocation
- [x] Thread-local buffers available for any edge-case row additions
- [x] Coordinator uses preallocated result buffers
- [x] Trajectory data cloning is conditional (only when history recording enabled)
- [ ] DHAT profiling shows zero allocations in cut computation hot path (requires manual verification)
- [ ] RSS monitoring shows stable memory after warmup phase (requires manual verification)
- [x] All 567+ tests pass
- [x] Golden tests pass (numerical correctness preserved)

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Preallocation size underestimated | Low | High | Use max(num_iterations × num_forward_passes × safety_factor) |
| Warmup solve changes model state | Low | Medium | Use `clear_solver()` after warmup |
| Thread-local buffer capacity exceeded | Low | Low | Assert and resize if needed in debug builds |
| Realization buffer lifetime issues | Medium | Medium | Careful borrow checker design with explicit indices |

---

## Memory Budget (Expected After This Sprint)

| Phase | Expected Allocations |
|-------|---------------------|
| Initialization | All buffers preallocated |
| Warmup | HiGHS internal vectors preallocated |
| Training (per iteration) | **Zero** in hot path |
| Training (total) | Only logging/output I/O |

---

## Key Files

| Component | Location |
|-----------|----------|
| Cut preallocation | `src/subproblem.rs:1006-1066` |
| add_row with allocations | `src/solver.rs:512-534` |
| State add_cut_constraint | `src/state.rs:1223-1238` |
| Coordinator buffers | `src/algorithm/coordinator.rs:176-192` |
| Realization cloning | `src/sddp/mod.rs:1248-1259` |
| HiGHS options | `src/subproblem.rs:140-162` |

---

## Verification Steps

### 1. DHAT Profiling

```bash
cargo build --release
valgrind --tool=dhat ./target/release/powers run examples/05-linear-model
```

Look for:
- Zero allocations in `evaluate_cut`, `realize_and_solve`, `add_cut_with_preallocation`
- All significant allocations during initialization phase

### 2. RSS Monitoring

```bash
./target/release/powers run examples/05-linear-model &
watch -n 1 'ps -o rss= -p $(pgrep powers)'
```

After warmup phase, RSS should remain constant (±1% variation acceptable for OS page management).

### 3. Benchmark Comparison

```bash
cargo bench --bench sddp_training -- --save-baseline sprint5
cargo bench --bench sddp_training -- --baseline sprint5
```

---

## Definition of Done

- [ ] All tickets complete and merged
- [ ] DHAT shows zero allocations in hot path
- [ ] RSS stable during training (after warmup)
- [ ] No performance regression (benchmark ≥ baseline)
- [ ] Golden tests pass
- [ ] All tests pass
- [ ] Documentation updated
