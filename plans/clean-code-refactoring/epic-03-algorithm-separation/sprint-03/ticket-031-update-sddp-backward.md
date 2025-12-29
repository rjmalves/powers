# [T-031] Update sddp/mod.rs to Use backward_pass Module

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 3: Backward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-030](./ticket-030-backward-timing.md)
> **Blocks**: [T-032](./ticket-032-verify-timing.md)

---

## ⚠️ CRITICAL: Golden Tests Required

This ticket replaces the inline backward pass with the extracted module. **Golden tests MUST pass.**

---

## Files to Read Before Starting

- `src/algorithm/backward_pass.rs` - Extracted backward pass
- `src/sddp/mod.rs:1762-2054` - Current inline backward pass
- `src/algorithm/context.rs` - `BackwardPassContext`

---

## Specification

### Changes to `src/sddp/mod.rs`

#### 1. Add import

```rust
use crate::algorithm::backward_pass;
```

#### 2. Create BackwardPassContext before backward loop

```rust
let backward_ctx = BackwardPassContext::new(
    &self.node_data_graph,
    &self.future_cost_function_graph,
    saa,
    &self.graph_bfs_table,
    &self.study_period_ids,
    index + 1,  // iteration (1-indexed)
    enable_cut_selection,
    &iteration_timing.backward,  // or create new timing
);
```

#### 3. Replace backward loop with module call

**Before (lines 1762-2054, ~300 lines):**
```rust
// --- Parallel Backward Pass with Stage-wise Synchronization ---
let backward_begin = Instant::now();
let num_study_periods = self.study_period_ids.len();
let mut lower_bound = 0.0;

for rev_idx in 0..num_study_periods {
    // ... 290 lines of backward pass logic ...
}
```

**After (~20 lines):**
```rust
// --- Backward Pass ---
let backward_begin = Instant::now();

let (backward_result, backward_timing) = backward_pass::execute(
    &mut coordinator,
    &backward_ctx,
)?;

let lower_bound = backward_result.lower_bound;
backward_cuts_added = backward_result.cuts_added;
backward_cuts_removed = backward_result.cuts_removed;
backward_cuts_returned = backward_result.cuts_returned;

// Convert timing
total_backward_preprocessing_time = backward_timing.preprocessing_time;
total_backward_model_preprocessing_time = backward_timing.model_preprocessing_time;
total_backward_solver_time = backward_timing.solver_time;
// ... etc
```

#### 4. Remove old backward pass code

Delete the ~300 lines that are now in `backward_pass.rs`.

---

## Acceptance Criteria

- [x] Backward pass replaced with `backward_pass::execute()` call
- [x] `BackwardPassContext` created correctly
- [x] Timing values converted to match original structure
- [x] Cut statistics populated correctly
- [x] `sddp/mod.rs` reduced by ~265 lines (from 3881 to 3616)
- [x] `cargo build -j1` succeeds
- [x] `cargo test -j1` passes
- [x] `./scripts/golden-tests.sh verify` passes ✅ CRITICAL

---

## Implementation Guide

### Step 1: Add imports and create context ✅

### Step 2: Replace loop with execute() call ✅

### Step 3: Convert result and timing to existing variables ✅

### Step 4: Delete old code ✅

### Step 5: Verify ✅

```bash
cargo build -j1 && RUST_TEST_THREADS=1 cargo test -j1 && ./scripts/golden-tests.sh verify
```

---

## Key Files to Modify

| File | Action | Status |
|------|--------|--------|
| `src/sddp/mod.rs` | REPLACE backward loop with module call, DELETE ~296 lines | ✅ |

---

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Integration, need to match timing exactly

---

## Definition of Done

- [x] Backward pass uses extracted module
- [x] Old code removed (296 lines)
- [x] sddp/mod.rs reduced significantly (3881 → 3616)
- [x] Golden tests pass (all 7 examples)
- [x] Bit-for-bit identical results verified
- [ ] Code reviewed
