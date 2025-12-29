# [T-026] Migrate SddpTrainHandler into Coordinator

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2 Revised](./00-sprint-overview.md)
> **Dependencies**: [T-025A](./ticket-025a-implement-coordinator.md)
> **Blocks**: [T-027](./ticket-027-coordinator-tests.md)

---

## ⚠️ CRITICAL: Golden Tests Required

This ticket changes how handlers are managed in `train()`. **Algorithm behavior must remain identical.** Run golden tests after EVERY change.

---

## Files to Read Before Starting

- `src/algorithm/coordinator.rs` - `ParallelHandlerCoordinator` (from T-025A)
- `src/sddp/mod.rs:1650-1670` - Current handler creation
- `src/sddp/mod.rs:1699-1704` - Forward pass parallel execution
- `src/sddp/mod.rs:1762-2054` - Backward pass with handler usage

---

## Context

### Current State

Handlers are created and managed directly in `train()`:

```rust
let mut train_handlers: Vec<SddpTrainHandler> = (0..num_forward_passes)
    .map(|_| SddpTrainHandler::new(...))
    .collect::<Result<_, _>>()?;

// Forward pass
let forward_results = train_handlers
    .par_iter_mut()
    .zip(all_sampled_noises.par_iter())
    .map(|(handler, noises)| self.forward(noises.to_vec(), handler))
    .collect()?;

// Backward pass phases use train_handlers directly
```

### Target State

Handlers are owned by the coordinator:

```rust
let mut coordinator = ParallelHandlerCoordinator::new(
    (0..num_forward_passes)
        .map(|_| SddpTrainHandler::new(...))
        .collect::<Result<_, _>>()?,
);

// Forward pass uses handlers_mut()
let forward_results = coordinator.handlers_mut()
    .par_iter_mut()
    .zip(all_sampled_noises.par_iter())
    .map(|(handler, noises)| self.forward(noises.to_vec(), handler))
    .collect()?;

// Backward pass (for now, still inline - extraction in Sprint 3)
// But uses coordinator methods for Phase 1, 2, 3
```

---

## Specification

### Changes to `src/sddp/mod.rs`

#### 1. Import coordinator

```rust
use crate::algorithm::coordinator::ParallelHandlerCoordinator;
```

#### 2. Replace handler creation (around line 1650)

**Before:**
```rust
let mut train_handlers: Vec<SddpTrainHandler> = (0..num_forward_passes)
    .map(|_| {
        SddpTrainHandler::new(
            &self.node_data_graph,
            initial_condition,
            saa,
            preserve_forward_detail,
            preserve_backward_detail,
            num_forward_passes,
            num_iterations,
        )
    })
    .collect::<Result<_, _>>()?;
```

**After:**
```rust
let handlers: Vec<SddpTrainHandler> = (0..num_forward_passes)
    .map(|_| {
        SddpTrainHandler::new(
            &self.node_data_graph,
            initial_condition,
            saa,
            preserve_forward_detail,
            preserve_backward_detail,
            num_forward_passes,
            num_iterations,
        )
    })
    .collect::<Result<_, _>>()?;

let mut coordinator = ParallelHandlerCoordinator::new(handlers);
```

**Note:** Unlike the original T-026, the coordinator no longer takes an FCF reference in `new()`. FCF is passed to `select_cuts_batch()` when called.

#### 3. Update forward pass call (around line 1699)

**Before:**
```rust
let forward_results: Vec<(f64, ForwardPassTimingAccumulator)> = train_handlers
    .par_iter_mut()
    .zip(all_sampled_noises.par_iter())
    .map(|(handler, noises)| self.forward(noises.to_vec(), handler))
    .collect::<Result<Vec<_>, String>>()?;
```

**After:**
```rust
let forward_results: Vec<(f64, ForwardPassTimingAccumulator)> = coordinator
    .handlers_mut()
    .par_iter_mut()
    .zip(all_sampled_noises.par_iter())
    .map(|(handler, noises)| self.forward(noises.to_vec(), handler))
    .collect::<Result<Vec<_>, String>>()?;
```

#### 4. Update other handler references

All other `train_handlers` references should become `coordinator.handlers_mut()`:
- Line ~1753: trajectory capture
- Line ~2222: backward detail extraction
- Line ~2233: forward detail extraction

Search for all occurrences of `train_handlers` and update:
```bash
grep -n "train_handlers" src/sddp/mod.rs
```

**Note**: The backward pass loop (lines 1762-2054) remains unchanged in this ticket. It will be refactored in Sprint 3 to use `BackwardStageProcessor` methods.

---

## Acceptance Criteria

- [ ] `ParallelHandlerCoordinator` created in `train()` with `new(handlers)`
- [ ] All `train_handlers` references updated to use coordinator
- [ ] Forward pass still uses `par_iter_mut()` on handlers
- [ ] Backward pass still works (unchanged for now)
- [ ] `cargo build -j1` succeeds
- [ ] `cargo test -j1` passes
- [ ] Golden tests pass ✅ CRITICAL
- [ ] No performance regression (benchmark within 5%)

---

## Implementation Guide

### Step 1: Add import

Add `use crate::algorithm::coordinator::ParallelHandlerCoordinator;` to imports.

### Step 2: Create coordinator

Replace handler vec creation with coordinator creation:
```rust
let mut coordinator = ParallelHandlerCoordinator::new(handlers);
```

### Step 3: Update all references

Find all `train_handlers` and replace:
- Read access: `coordinator.handlers()`
- Mutable access: `coordinator.handlers_mut()`

### Step 4: Verify

```bash
cargo build -j1 && RUST_TEST_THREADS=1 cargo test -j1 && ./scripts/golden-tests.sh verify
```

---

## Key Files to Modify

| File | Action |
|------|--------|
| `src/sddp/mod.rs` | UPDATE handler management to use coordinator |

---

## Pitfalls to Avoid

- ⚠️ Don't change backward pass loop logic yet—that's Sprint 3
- ⚠️ Ensure all handler references are updated (search for `train_handlers`)
- ⚠️ Run golden tests after EVERY change
- ⚠️ Coordinator `new()` takes only handlers, not FCF reference

---

## Effort Estimate

**Points**: 5  
**Confidence**: Medium  
**Rationale**: Many references to update; main risk is missing one

---

## Definition of Done

- [ ] Coordinator created and used in `train()`
- [ ] All handler references updated
- [ ] Forward pass unchanged in behavior
- [ ] Backward pass unchanged in behavior
- [ ] Golden tests pass
- [ ] Benchmarks within 5%
- [ ] Code reviewed
