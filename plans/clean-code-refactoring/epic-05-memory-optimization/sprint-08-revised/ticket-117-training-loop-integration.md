# [T-117] Integrate Iteration Lifecycle into Training Loop

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-114, T-115, T-116
> **Blocks**: T-118, T-119, T-120, T-121, T-122
> **Priority**: 1 (Critical Path)
> **Status**: 📋 Planned

## Files to Read Before Starting

- `src/sddp/mod.rs` - Training loop implementation
- T-114, T-115, T-116 implementations

---

## Context

### Background

The SDDP training loop must be updated to:
1. Create Models for all stages at iteration start
2. Use `use_basis=true` for warm-starting during training
3. Update cuts via `update_cut_dual()` during backward pass
4. Cache basis and drop Models at iteration end

---

## Specification

```rust
impl SddpAlgorithm {
    /// Create Models for all stages at start of iteration.
    fn create_iteration_models(&mut self, use_basis: bool) -> Result<(), String> {
        for (stage, handler) in self.handlers.iter_mut().enumerate() {
            handler.subproblem.create_iteration_model(use_basis)
                .map_err(|e| format!("Stage {} model creation failed: {}", stage, e))?;
        }
        Ok(())
    }
    
    /// Finalize iteration: cache basis and drop Models.
    fn finalize_iteration(&mut self, cache_basis: bool) {
        for handler in &mut self.handlers {
            handler.subproblem.finalize_iteration(cache_basis);
        }
    }
    
    /// Main training loop with per-iteration Model lifecycle.
    pub fn train(&mut self, config: &TrainingConfig) -> TrainingResult {
        for iteration in 1..=config.max_iterations {
            // 1. Create Models with warm-start
            self.create_iteration_models(true)?;
            
            // 2. Forward pass
            let forward_result = self.forward_pass(iteration)?;
            
            // 3. Backward pass (uses update_cut_dual)
            let backward_result = self.backward_pass(iteration)?;
            
            // 4. Finalize with basis caching
            self.finalize_iteration(true);
            
            // Check convergence...
        }
        // ...
    }
}
```

### Backward Pass Integration

```rust
fn backward_pass(&mut self, iteration: usize) -> Result<BackwardPassResult, String> {
    for stage in (1..self.num_stages).rev() {
        for (forward_pass_idx, branching_result) in branchings.iter().enumerate() {
            let (coefficients, rhs) = self.compute_cut(stage, branching_result)?;
            let slot = self.compute_cut_slot(iteration, forward_pass_idx);
            
            // DUAL UPDATE: both Problem and Model
            self.handlers[stage - 1]
                .subproblem
                .update_cut_dual(slot, &coefficients, rhs)?;
        }
    }
    Ok(backward_result)
}
```

---

## Acceptance Criteria

- [ ] `create_iteration_models()` called at start of each iteration
- [ ] `finalize_iteration()` called at end of each iteration
- [ ] Backward pass uses `update_cut_dual()`
- [ ] Training uses `use_basis=true`, `cache_basis=true`
- [ ] Memory reclaimed between iterations
- [ ] All SDDP tests pass
- [ ] Golden tests pass

---

## Testing Requirements

```rust
#[test]
fn test_training_loop_lifecycle() {
    let mut algorithm = create_test_algorithm();
    
    // Before training
    for handler in &algorithm.handlers {
        assert!(!handler.subproblem.has_model());
    }
    
    // Create models
    algorithm.create_iteration_models(true).unwrap();
    
    for handler in &algorithm.handlers {
        assert!(handler.subproblem.has_model());
    }
    
    // Finalize
    algorithm.finalize_iteration(true);
    
    for handler in &algorithm.handlers {
        assert!(!handler.subproblem.has_model());
        assert!(handler.subproblem.cached_basis.is_some());
    }
}

#[test]
fn test_full_training_run() {
    let mut algorithm = create_test_algorithm();
    let result = algorithm.train(&TrainingConfig { max_iterations: 5, .. });
    
    assert!(result.is_ok());
    // Verify convergence behavior matches pre-refactoring
}
```

---

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Integration work with many touchpoints

---

## Definition of Done

- [ ] Lifecycle integrated into training loop
- [ ] Forward pass uses iteration Model
- [ ] Backward pass uses dual cut update
- [ ] Training with `use_basis=true`
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] PR merged
