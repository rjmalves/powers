# [T-123] Integrate Per-Iteration Model Lifecycle into Training Loop

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-117, T-118 (infrastructure complete)
> **Blocks**: T-125, T-126
> **Priority**: 6 (Production Integration)
> **Status**: ✅ Complete

## Files to Read Before Starting

- `src/sddp/mod.rs` - Current training loop implementation (lines 1828-2233)
- `src/sddp/mod.rs` - `IterationLifecycleConfig` (lines 72-106)
- `src/sddp/mod.rs` - `SddpTrainHandler::create_iteration_models()` (lines 701-715)
- `src/sddp/mod.rs` - `SddpTrainHandler::finalize_iteration()` (lines 724-732)
- `src/subproblem.rs` - `create_iteration_model()`, `finalize_iteration()` (lines 1167-1220)

---

## Context

### Background

The per-iteration Model lifecycle infrastructure is complete (T-110 through T-118), but the production training loop still uses persistent Models that are created once and reused across all iterations. This means HiGHS memory is NOT being reclaimed between iterations.

### Current State

```rust
// Current: Handlers created ONCE, Models persist for entire training
let handlers: Vec<SddpTrainHandler> = (0..num_forward_passes)
    .map(|_| SddpTrainHandler::new(...))  // Creates Models here
    .collect();

for iteration in 0..num_iterations {
    // Forward pass uses existing Models
    // Backward pass uses existing Models
    // Models NOT dropped - memory accumulates!
}
```

### Target State

```rust
// Target: Models created/dropped each iteration
let handlers: Vec<SddpTrainHandler> = (0..num_forward_passes)
    .map(|_| SddpTrainHandler::new(...))  // Creates Problems, NOT Models
    .collect();

let lifecycle_config = IterationLifecycleConfig::training();

for iteration in 0..num_iterations {
    // Create Models from Problems (with basis warm-start)
    coordinator.create_iteration_models(&lifecycle_config)?;
    
    // Forward pass
    // Backward pass
    
    // Drop Models, cache basis, free HiGHS memory
    coordinator.finalize_iteration(&lifecycle_config);
}
```

---

## Specification

### Changes Required

1. **Modify `SddpTrainHandler::new()`** to NOT create Models initially
   - Create Problems only
   - Models created later via `create_iteration_model()`

2. **Add lifecycle calls to training loop**
   - Call `create_iteration_models()` at start of each iteration
   - Call `finalize_iteration()` at end of each iteration

3. **Update `ParallelHandlerCoordinator`** (if needed)
   - Add `create_iteration_models()` and `finalize_iteration()` methods
   - These delegate to each handler

4. **Update warmup logic**
   - Warmup should happen AFTER first `create_iteration_models()`
   - Or warmup should be part of iteration lifecycle

### Behavior

- **Iteration Start**: Fresh Models created from Problems
- **Basis Warm-Start**: If `use_basis=true`, apply cached basis from previous iteration
- **Iteration End**: Models dropped, basis cached for next iteration
- **Memory**: HiGHS memory reclaimed between iterations

### Error Handling

- If Model creation fails: Return error, abort training
- If basis incompatible: Log warning, cold-start (already handled in infrastructure)

---

## Acceptance Criteria

- [ ] Training loop calls `create_iteration_models()` at iteration start
- [ ] Training loop calls `finalize_iteration()` at iteration end
- [ ] Basis is cached between iterations (training mode)
- [ ] All 589+ lib tests pass
- [ ] No numerical result changes (determinism preserved)
- [ ] Memory usage stabilizes across iterations (no monotonic growth)

---

## Implementation Guide

### Step 1: Modify SddpTrainHandler::new()

Current implementation creates Model immediately. Change to create Problem only:

```rust
// In SddpTrainHandler::new()
// BEFORE: Creates Subproblem with Model
// AFTER: Creates Subproblem with Problem only, model=None

// The Subproblem::new() already creates a Model via create_model().
// Need to refactor to separate Problem creation from Model creation.
```

**Option A**: Add `new_without_model()` constructor to Subproblem
**Option B**: Modify `new()` to take a `create_model: bool` parameter
**Option C**: Create Model in `new()` but drop it immediately, rely on `create_iteration_model()`

Recommend **Option C** for minimal changes - the overhead of creating and dropping once is negligible.

### Step 2: Add Coordinator Methods

```rust
impl ParallelHandlerCoordinator {
    pub fn create_iteration_models(&mut self, config: &IterationLifecycleConfig) -> Result<(), String> {
        for handler in &mut self.handlers {
            handler.create_iteration_models(config)?;
        }
        Ok(())
    }
    
    pub fn finalize_iteration(&mut self, config: &IterationLifecycleConfig) {
        for handler in &mut self.handlers {
            handler.finalize_iteration(config);
        }
    }
}
```

### Step 3: Modify Training Loop

```rust
pub fn train(...) -> Result<TrainingResult, String> {
    // ... initialization ...
    
    let handlers: Vec<SddpTrainHandler> = ...;
    let mut coordinator = ParallelHandlerCoordinator::new(handlers);
    
    // Preallocate cuts (operates on Problems)
    for handler in coordinator.handlers_mut() {
        handler.preallocate_cut_constraints(max_cuts_per_node, num_forward_passes)?;
    }
    
    let lifecycle_config = IterationLifecycleConfig::training();
    
    for index in 0..num_iterations {
        let iter_begin = Instant::now();
        
        // ===== NEW: Create Models for this iteration =====
        coordinator.create_iteration_models(&lifecycle_config)?;
        
        // Warmup on first iteration (now that Models exist)
        if index == 0 {
            for handler in coordinator.handlers_mut() {
                handler.warmup_solvers()?;
            }
        }
        
        // ... forward pass ...
        // ... backward pass ...
        
        // ===== NEW: Finalize iteration, free HiGHS memory =====
        coordinator.finalize_iteration(&lifecycle_config);
        
        // ... logging ...
    }
    
    // ... return results ...
}
```

### Step 4: Handle Warmup

Warmup currently happens before the iteration loop. With per-iteration Models, warmup should happen after `create_iteration_models()`:

**Option A**: Move warmup inside first iteration
**Option B**: Do warmup during `create_iteration_model()` if first call
**Option C**: Make warmup idempotent (can be called multiple times safely)

Recommend **Option A** for clarity.

---

## Testing Requirements

### Unit Tests

- [ ] `create_iteration_models()` creates Models for all handlers
- [ ] `finalize_iteration()` drops all Models
- [ ] Basis is cached when `cache_basis=true`
- [ ] Basis is applied when `use_basis=true`

### Integration Tests

- [ ] Full training run with lifecycle integration
- [ ] Results match pre-integration results (determinism)

### Performance Tests

- [ ] Run benchmark before/after integration
- [ ] Verify overhead < 5% of total training time

### Memory Tests

- [ ] RSS stable across iterations (no monotonic growth)
- [ ] Peak RSS reduced compared to persistent Model approach

---

## Pitfalls to Avoid

⚠️ **Don't break cut preallocation**: Cuts are preallocated in Problem AND Model. If Model is recreated each iteration, preallocated cuts in Problem will be copied to new Model.

⚠️ **Don't break backward pass cut updates**: During backward pass, cuts are added to Problem (permanent) and Model (current iteration). The `add_cut_dual()` method already handles this.

⚠️ **Don't break warmup**: Warmup must happen after Model creation.

⚠️ **Don't break FCF updates**: FCF updates happen at graph level, not subproblem level. Should be unaffected.

---

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Requires careful integration with existing training loop, multiple touchpoints

---

## Definition of Done

- [ ] Training loop uses per-iteration lifecycle
- [ ] All tests pass (589+ lib tests)
- [ ] No numerical result changes
- [ ] Memory usage stable across iterations
- [ ] Benchmark shows acceptable overhead
- [ ] Code reviewed and merged
