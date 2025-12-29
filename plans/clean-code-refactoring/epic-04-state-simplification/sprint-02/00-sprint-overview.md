# Sprint 2: FCF Graph Wrapper Removal

> **Epic**: [Epic 4: State Simplification](../00-epic-overview.md)
> **Duration**: 1 week
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

The `Arc<Mutex<FutureCostFunction>>` wrapper removal is a **mechanical refactoring** with no algorithm changes.

**All numerical results must remain bit-for-bit identical.** Golden tests must pass after every change.

---

## Background

Based on the analysis in [FCF_GRAPH_ARCHITECTURE_ANALYSIS.md](../../../../docs/FCF_GRAPH_ARCHITECTURE_ANALYSIS.md):

- The `Mutex` in `Arc<Mutex<FutureCostFunction>>` is **unnecessary**
- FCF is only modified in Phase 2 (single-threaded batch cut selection)
- Parallel phases don't access FCF directly—they receive pre-cloned `Arc<BendersCut>` references
- Manual synchronization (sorting by `forward_pass_idx`) already ensures deterministic ordering

---

## Goals

1. **Primary**: Remove `Arc<Mutex<>>` wrapper from FCF graph
2. **Primary**: Replace `.lock().unwrap()` calls with direct `&mut` access
3. **Secondary**: Simplify type signatures throughout the codebase
4. **Validation**: Bit-for-bit identical outputs, no performance regression

---

## Prerequisites

From Sprint 1:
- ✅ State consolidation complete (not blocking, but ordered logically)

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-045](./ticket-045-analyze-fcf-access-patterns.md) | Analyze and document FCF access patterns | 2 | Yes | None | ⬜ |
| [T-046](./ticket-046-remove-mutex-from-fcf-type.md) | Remove Mutex from FCF graph type | 3 | Yes | T-045 | ⬜ |
| [T-047](./ticket-047-update-coordinator-fcf-access.md) | Update coordinator FCF access | 3 | Yes | T-046 | ⬜ |
| [T-048](./ticket-048-update-output-fcf-access.md) | Update output modules FCF access | 2 | Yes | T-046 | ⬜ |
| [T-049](./ticket-049-verify-fcf-refactoring.md) | Verify FCF refactoring end-to-end | 2 | Yes | T-047, T-048 | ⬜ |

**Total Points**: 12

---

## Parallelization

```
T-045 (Analyze) ──→ T-046 (Remove Mutex) ──→ T-047 (Coordinator) ────┐
                                         └──→ T-048 (Output) ────────├──→ T-049 (Verify)
```

T-047 and T-048 can run in parallel after T-046.

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| FCF graph type | `DirectedGraph<Arc<Mutex<FutureCostFunction>>>` | `DirectedGraph<FutureCostFunction>` |
| Lock calls removed | ~8 locations | 0 |
| Lock contention | Zero (already) | N/A (no locks) |
| Type signature complexity | High | Low |
| Compile-time guarantees | Runtime (Mutex) | Static (borrow checker) |

---

## Key Files to Modify

| File | Change |
|------|--------|
| `src/sddp/mod.rs` | Change FCF graph type, remove `.lock()` calls |
| `src/algorithm/coordinator.rs` | Update `select_cuts_batch` FCF access |
| `src/algorithm/backward_pass.rs` | Update FCF parameter type (if any) |
| `src/output/csv/*.rs` | Remove `.lock()` calls |
| `src/output/parquet/*.rs` | Remove `.lock()` calls |

---

## Verification Protocol

After EVERY ticket:

```bash
cargo build -j1 && RUST_TEST_THREADS=1 cargo test -j1 && ./scripts/golden-tests.sh verify
```

---

## Definition of Done

- [ ] `Arc<Mutex<FutureCostFunction>>` replaced with `FutureCostFunction`
- [ ] All `.lock().unwrap()` calls removed from FCF access
- [ ] Borrow checker enforces safe access (no runtime locking)
- [ ] Golden tests pass (bit-for-bit identical)
- [ ] No performance regression
- [ ] All tests pass
