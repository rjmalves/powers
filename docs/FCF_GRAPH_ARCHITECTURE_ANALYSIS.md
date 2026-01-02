# FCF Graph Architecture Analysis Report

> **Date**: 2025-12-29  
> **Context**: Post Epic 03 completion - Algorithm Separation  
> **Purpose**: Determine if `Arc<Mutex<>>` wrapper on FCF graph is necessary

---

## Executive Summary

**Conclusion**: The `Mutex` in `Arc<Mutex<FutureCostFunction>>` is **unnecessary** and can be removed. The current architecture already implements **manual synchronization** to ensure deterministic reproducibility, and **FCF is never accessed concurrently** in a way that would cause data races.

**Recommendation**: Replace `Arc<Mutex<FutureCostFunction>>` with just `FutureCostFunction` (no wrapper) or `RefCell<FutureCostFunction>` (for interior mutability without runtime locking overhead).

---

## Current Architecture Analysis

### FCF Graph Type

```rust
// Current (src/sddp/mod.rs:1472-1474)
pub struct SddpAlgorithm {
    pub future_cost_function_graph: graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    // ...
}
```

### Threading Model

The SDDP algorithm has a **strict 3-phase architecture** per backward stage:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ITERATION LOOP                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ FORWARD PASS (PARALLEL)                                               │   │
│  │                                                                        │   │
│  │   handler[0].forward() ──┐                                            │   │
│  │   handler[1].forward() ──┼── par_iter_mut                             │   │
│  │   handler[2].forward() ──┤   (NO FCF ACCESS)                          │   │
│  │   ...                    │                                            │   │
│  │                          ▼                                            │   │
│  │   [Handlers modify their OWN subproblem_graph]                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ BACKWARD PASS (Stage loop - SEQUENTIAL over stages)                   │   │
│  │                                                                        │   │
│  │   for stage_idx in (0..num_stages).rev() {                            │   │
│  │                                                                        │   │
│  │     ┌────────────────────────────────────────────────────────────┐    │   │
│  │     │ PHASE 1: Parallel Cut Computation                          │    │   │
│  │     │                                                             │    │   │
│  │     │   handler[0].compute_cut_data() ──┐                        │    │   │
│  │     │   handler[1].compute_cut_data() ──┼── par_iter_mut         │    │   │
│  │     │   handler[2].compute_cut_data() ──┤   (NO FCF ACCESS)      │    │   │
│  │     │   ...                              │                        │    │   │
│  │     │                                    ▼                        │    │   │
│  │     │   [Returns CutData, not accessing FCF]                      │    │   │
│  │     └────────────────────────────────────────────────────────────┘    │   │
│  │                              │                                         │   │
│  │                              ▼                                         │   │
│  │     ┌────────────────────────────────────────────────────────────┐    │   │
│  │     │ PHASE 2: Sequential Batch Cut Selection (SINGLE-THREADED)  │    │   │
│  │     │                                                             │    │   │
│  │     │   cut_data.sort_by(forward_pass_idx)  // DETERMINISM       │    │   │
│  │     │   fcf.lock().add_cuts_batch()         // FCF WRITE         │    │   │
│  │     │   fcf.lock().update_state()           // FCF WRITE         │    │   │
│  │     │                                                             │    │   │
│  │     │   [THIS IS THE ONLY FCF MODIFICATION POINT]                 │    │   │
│  │     └────────────────────────────────────────────────────────────┘    │   │
│  │                              │                                         │   │
│  │                              ▼                                         │   │
│  │     ┌────────────────────────────────────────────────────────────┐    │   │
│  │     │ PHASE 3b: Parallel Cut Application                         │    │   │
│  │     │                                                             │    │   │
│  │     │   handler[0].apply_cuts() ──┐                              │    │   │
│  │     │   handler[1].apply_cuts() ──┼── par_iter_mut               │    │   │
│  │     │   handler[2].apply_cuts() ──┤   (FCF READ via Arc clone)   │    │   │
│  │     │   ...                        │                              │    │   │
│  │     │                              ▼                              │    │   │
│  │     │   [Handlers modify their OWN subproblem models]             │    │   │
│  │     └────────────────────────────────────────────────────────────┘    │   │
│  │   }                                                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Observations

#### 1. FCF is ONLY Modified in Phase 2 (Single-Threaded)

```rust
// src/algorithm/coordinator.rs:201-230
fn select_cuts_batch(...) -> Result<Phase2Result, String> {
    // Sort for deterministic ordering (CRITICAL for reproducibility)
    cut_data.sort_unstable_by_key(|data| data.forward_pass_idx);  // ← Manual sync

    // Access FCF and perform batch selection
    let batch_result: BatchCutSelectionResult = {
        let mut fcf_locked = parent_fcf_node.data.lock().unwrap();  // ← Single-threaded
        fcf_locked.add_cuts_batch_from_data(cut_data, enable_cut_selection)
    };
    // ...
}
```

This is the **only** place FCF is mutated. It's already **single-threaded** and protected by **manual synchronization** (sorting by `forward_pass_idx`).

#### 2. Parallel Phases DON'T Access FCF Directly

**Phase 1** (Parallel Cut Computation):
```rust
// src/algorithm/coordinator.rs:157-178
fn compute_cuts_parallel(...) -> Result<Phase1Result, String> {
    let results: Vec<(CutData, BackwardPhase1Timing)> = self.handlers
        .par_iter_mut()
        .map(|(fp_idx, handler)| {
            handler.compute_cut_data_for_backward_step(...)  // ← NO FCF ACCESS
        })
        .collect()?;
    // ...
}
```

**Phase 3b** (Parallel Cut Application):
```rust
// src/algorithm/coordinator.rs:309-329
fn apply_cuts_parallel(...) -> Result<Duration, String> {
    self.handlers
        .par_iter_mut()
        .map(|handler| {
            handler.apply_aggregated_cut_result(
                parent_id,
                &phase2_result.aggregated,  // ← Read-only aggregate
                &phase2_result.cuts,        // ← Arc<BendersCut> clones (read-only)
            )
        })
        .collect()?;
    // ...
}
```

The cuts are pre-cloned as `Arc<BendersCut>` during Phase 2, so Phase 3b only **reads** shared data.

#### 3. Forward Pass NEVER Accesses FCF

```rust
// src/sddp/mod.rs:1310-1343
pub fn forward(...) -> Result<(f64, ForwardPassTimingAccumulator), String> {
    let mut ctx = ForwardPassContext::new(
        &mut self.subproblem_graph,      // ← Handler's own graph
        &mut self.realization_graph,     // ← Handler's own graph
        // ... NO FCF reference
    );
    forward_pass::execute(&mut ctx, &timing)?
}
```

Cuts are already incorporated into each handler's `subproblem_graph` model (the HiGHS LP).

---

## Why the Mutex Exists (Historical Context)

The `Mutex` was likely introduced for one of these reasons:

1. **Conservative Safety**: Early development assumed parallel access might occur
2. **Rust's Borrow Checker**: `Mutex` provides interior mutability for shared references
3. **Future Parallelism**: Anticipated parallel cut selection (never implemented due to reproducibility)

However, the **reproducibility requirement** forced manual synchronization, making the `Mutex` redundant:

> "We had to implement manual synchronization in many steps... to ensure floating point operation ordering... always indexed by the forward pass index, branching index, iteration number"

---

## Evidence: Lock Contention is Zero

All `.lock().unwrap()` calls on FCF happen in **single-threaded context**:

| Location | Context | Lock Duration |
|----------|---------|---------------|
| `coordinator.rs:225` | Phase 2 (single-threaded) | ~microseconds |
| `coordinator.rs:248` | Phase 3a (single-threaded) | ~microseconds |
| `sddp/mod.rs:1643` | Pre-training init (single-threaded) | ~milliseconds |
| `output/*.rs` | Post-training export (single-threaded) | ~milliseconds |

**No lock contention occurs** because locks are never held during parallel phases.

---

## Proposed Architecture Change

### Option A: Remove All Wrappers (Simplest)

```rust
// Proposed
pub struct SddpAlgorithm {
    pub future_cost_function_graph: graph::DirectedGraph<FutureCostFunction>,
    // ...
}
```

**Pros**:
- Zero runtime overhead
- Simpler type signatures
- No lock/unlock machinery

**Cons**:
- Requires passing `&mut` references through the call stack
- May require splitting borrows in some cases

### Option B: Use RefCell (Interior Mutability Without Locking)

```rust
// Proposed
pub struct SddpAlgorithm {
    pub future_cost_function_graph: graph::DirectedGraph<RefCell<FutureCostFunction>>,
    // ...
}
```

**Pros**:
- Interior mutability without Mutex overhead
- Minimal code changes (replace `.lock().unwrap()` with `.borrow_mut()`)
- Runtime borrow checking catches bugs in debug builds

**Cons**:
- `RefCell` panics on double-borrow (but this can't happen with current architecture)
- Not `Sync`, so can't be shared across threads (but we don't need this)

### Option C: Keep Arc, Remove Mutex

```rust
// Proposed
pub struct SddpAlgorithm {
    pub future_cost_function_graph: graph::DirectedGraph<Arc<RefCell<FutureCostFunction>>>,
    // ...
}
```

**Pros**:
- Allows cheap cloning of references for parallel phases
- Interior mutability for single-threaded phases

**Cons**:
- Arc overhead for reference counting (minimal)
- RefCell not Sync (can't be used directly in parallel, but we don't need this)

### Recommendation: Option A (Remove All Wrappers)

The current architecture already ensures:
1. **No concurrent FCF writes** (Phase 2 is single-threaded)
2. **No concurrent FCF reads during writes** (phases are sequential)
3. **Deterministic ordering** (manual synchronization by forward_pass_idx)

Therefore, no synchronization primitive is needed. The `&mut` borrow checker provides compile-time guarantees.

---

## Implementation Impact

### Files Requiring Changes

| File | Change |
|------|--------|
| `src/sddp/mod.rs` | Change FCF graph type, remove `.lock()` calls |
| `src/algorithm/context.rs` | Update `BackwardPassContext` FCF reference type |
| `src/algorithm/backward_pass.rs` | Update FCF parameter type |
| `src/algorithm/coordinator.rs` | Update `select_cuts_batch` FCF parameter, remove `.lock()` |
| `src/algorithm/processor.rs` | Update trait signature |
| `src/output/csv/*.rs` | Remove `.lock()` calls |
| `src/output/parquet/*.rs` | Remove `.lock()` calls |

### Estimated Effort

- **Lines Changed**: ~50-100
- **Risk**: Low (mechanical replacement, no algorithm changes)
- **Testing**: Golden tests will verify correctness

---

## Benefits of Removing Mutex

### 1. Performance

While actual lock contention is zero, there's still overhead:
- **Lock acquisition/release**: ~20-50 CPU cycles per `.lock()` call
- **Memory barriers**: Implicit synchronization fences
- **Cache line bouncing**: Mutex state updates invalidate caches

### 2. Code Clarity

```rust
// Before: Noisy, hides intent
let mut fcf = fcf_node.data.lock().unwrap();
fcf.add_cuts_batch_from_data(cut_data, enable_cut_selection)

// After: Clear ownership
let fcf = fcf_graph.get_node_mut(parent_id)?;
fcf.data.add_cuts_batch_from_data(cut_data, enable_cut_selection)
```

### 3. Compile-Time Guarantees

The borrow checker will **statically enforce** that:
- Only one phase can modify FCF at a time
- Parallel phases cannot access FCF mutably
- No data races are possible

This is **stronger** than Mutex, which only provides runtime guarantees.

### 4. Reduced Cognitive Load

Developers no longer need to:
- Wonder about potential deadlocks
- Think about lock ordering
- Handle `PoisonError` from panicking threads

---

## Alternative Architecture: Structure of Arrays (SoA)

If we're refactoring the FCF graph, consider a more radical redesign:

### Current: Array of Structures (AoS)

```rust
DirectedGraph<FutureCostFunction>
// Each node contains: cut_pool + state_pool
```

### Alternative: Structure of Arrays (SoA)

```rust
pub struct FcfGraphSoA {
    // Indexed by node_id
    cut_pools: Vec<BendersCutPool>,
    state_pools: Vec<VisitedStatePool>,
    
    // Precomputed for O(1) lookup
    node_id_to_index: HashMap<usize, usize>,
}
```

**Benefits**:
- Better cache locality when iterating cut_pools
- Easier to parallelize operations on all nodes
- Simpler lifetime management

**Drawback**:
- Larger refactoring effort
- May not fit `DirectedGraph` abstraction

This could be considered for Epic 4 or 5 if memory optimization is a priority.

---

## Validation Plan

After implementing the change:

1. **Unit Tests**: Existing tests should pass unchanged
2. **Golden Tests**: Bit-for-bit identical outputs required
3. **Performance**: Run benchmarks to confirm no regression
4. **Memory**: Profile to confirm no increase in allocations

---

## Conclusion

The `Arc<Mutex<FutureCostFunction>>` wrapper is an artifact of defensive programming that is no longer needed given the current architecture. The manual synchronization for reproducibility has made it redundant.

**Recommended Action**: Add a ticket to Epic 4 (State Simplification) or create a small focused Epic to:

1. Replace `Arc<Mutex<FutureCostFunction>>` with `FutureCostFunction`
2. Update all access sites to use direct `&mut` references
3. Verify with golden tests
4. Benchmark to confirm performance improvement

This change will simplify the codebase, improve performance marginally, and provide stronger compile-time guarantees through Rust's ownership system.

---

## Appendix: All FCF Access Points

### Writes (Mutations)

| Location | Context | Current Code |
|----------|---------|--------------|
| `coordinator.rs:225-229` | Phase 2 batch add | `fcf_locked.add_cuts_batch_from_data()` |
| `coordinator.rs:254-255` | Phase 3a set inactive | `cut.set_active(false)` |
| `coordinator.rs:258-261` | Phase 3a index update | `active_cut_indices.remove()` |
| `coordinator.rs:268-274` | Phase 3a index adjust | `active_cut_indices.iter_mut()` |
| `sddp/mod.rs:1643-1666` | Pre-training init | `*fcf = FutureCostFunction::preallocate_pools()` |

### Reads

| Location | Context | Current Code |
|----------|---------|--------------|
| `coordinator.rs:284-291` | Phase 3a clone cuts | `fcf_locked.cut_pool.pool.get()` |
| `output/csv/*.rs` | Post-training export | Iteration over cuts/states |
| `output/parquet/*.rs` | Post-training export | Iteration over cuts/states |
| `sddp/mod.rs:1845-1859` | Active cut count | `active_cut_indices.len()` |

All accesses occur in **single-threaded context**, confirming the Mutex is unnecessary.
