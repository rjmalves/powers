# [T-114] Implement Per-Iteration Model Lifecycle with Optional Basis

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-113
> **Blocks**: T-115, T-116, T-117, T-118
> **Priority**: 1 (Critical Path)
> **Status**: 📋 Planned

## Files to Read Before Starting

- `src/subproblem.rs` - With T-113 changes
- [Sprint Overview](./00-sprint-overview.md) - Mode-aware lifecycle diagram

---

## Context

### Background

The lifecycle methods control Model creation and destruction per iteration, with **optional basis** support:
- **Training**: `use_basis=true` → warm-start for performance
- **Simulation**: `use_basis=false` → cold-start for reproducibility from loaded FCF

### Why Optional Basis?

When users load a persisted FCF, they won't have the basis (too large to store). Simulation must produce identical results whether:
- Run immediately after training (basis available)
- Run from loaded FCF (no basis)

---

## Specification

```rust
impl Subproblem {
    /// Create Model for a new iteration.
    ///
    /// # Arguments
    ///
    /// * `use_basis` - If true, apply cached basis for warm-starting.
    ///                 If false, cold-start (simulation mode).
    pub fn create_iteration_model(&mut self, use_basis: bool) -> Result<(), String> {
        if self.model.is_some() {
            return Err("Model already exists - call finalize_iteration first".into());
        }
        
        let mut model = self.problem
            .create_model(solver::Sense::Minimise)
            .map_err(|e| format!("Model creation failed: {:?}", e))?;
        
        set_default_solver_options(&mut model);
        
        // OPTIONAL: Apply cached basis only if requested
        if use_basis {
            if let Some(ref basis) = self.cached_basis {
                if basis.is_compatible(model.num_cols(), model.num_rows()) {
                    let _ = model.apply_stored_basis(basis);
                }
            }
        }
        
        self.model = Some(model);
        Ok(())
    }
    
    /// Finalize iteration with optional basis caching.
    ///
    /// # Arguments
    ///
    /// * `cache_basis` - If true, cache basis for next iteration (training).
    ///                   If false, don't cache (simulation, saves memory).
    pub fn finalize_iteration(&mut self, cache_basis: bool) {
        if cache_basis {
            if let Some(ref model) = self.model {
                self.cached_basis = Some(model.get_stored_basis());
            }
        }
        self.model = None;  // Drop Model, free HiGHS memory
    }
    
    /// Clear cached basis (for transitioning to simulation).
    pub fn clear_cached_basis(&mut self) {
        self.cached_basis = None;
    }
    
    /// Check if Model is available.
    #[inline]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }
    
    /// Get mutable reference to Model (panics if not created).
    #[inline]
    pub fn model_mut(&mut self) -> &mut solver::Model {
        self.model.as_mut().expect("Model not created for iteration")
    }
    
    /// Get immutable reference to Model (panics if not created).
    #[inline]
    pub fn model_ref(&self) -> &solver::Model {
        self.model.as_ref().expect("Model not created for iteration")
    }
}
```

---

## Acceptance Criteria

- [ ] `create_iteration_model(use_basis: bool)` implemented
- [ ] `finalize_iteration(cache_basis: bool)` implemented
- [ ] `clear_cached_basis()` implemented
- [ ] Basis only applied when `use_basis=true`
- [ ] Basis only cached when `cache_basis=true`
- [ ] Model dropped on finalize (HiGHS freed)
- [ ] Unit tests for all modes

---

## Testing Requirements

```rust
#[test]
fn test_create_with_basis() {
    let mut subproblem = create_test_subproblem();
    subproblem.preallocate_cut_constraints(100, 10).unwrap();
    
    // First iteration - no basis yet
    subproblem.create_iteration_model(true).unwrap();
    subproblem.model_mut().solve();
    subproblem.finalize_iteration(true);  // Cache basis
    
    assert!(subproblem.cached_basis.is_some());
    
    // Second iteration - should apply basis
    subproblem.create_iteration_model(true).unwrap();
    assert!(subproblem.has_model());
    subproblem.finalize_iteration(true);
}

#[test]
fn test_create_without_basis() {
    let mut subproblem = create_test_subproblem();
    subproblem.preallocate_cut_constraints(100, 10).unwrap();
    
    // Train first
    subproblem.create_iteration_model(true).unwrap();
    subproblem.model_mut().solve();
    subproblem.finalize_iteration(true);
    
    assert!(subproblem.cached_basis.is_some());
    
    // Simulate - don't use basis
    subproblem.create_iteration_model(false).unwrap();
    subproblem.model_mut().solve();
    subproblem.finalize_iteration(false);  // Don't cache
    
    // Basis unchanged (still from training)
    assert!(subproblem.cached_basis.is_some());
}

#[test]
fn test_clear_cached_basis() {
    let mut subproblem = create_test_subproblem();
    subproblem.preallocate_cut_constraints(100, 10).unwrap();
    
    subproblem.create_iteration_model(true).unwrap();
    subproblem.model_mut().solve();
    subproblem.finalize_iteration(true);
    
    assert!(subproblem.cached_basis.is_some());
    
    subproblem.clear_cached_basis();
    
    assert!(subproblem.cached_basis.is_none());
}

#[test]
fn test_finalize_without_cache() {
    let mut subproblem = create_test_subproblem();
    subproblem.preallocate_cut_constraints(100, 10).unwrap();
    
    subproblem.create_iteration_model(true).unwrap();
    subproblem.model_mut().solve();
    subproblem.finalize_iteration(false);  // Don't cache
    
    assert!(subproblem.cached_basis.is_none());
    assert!(!subproblem.has_model());
}

#[test]
fn test_create_model_twice_errors() {
    let mut subproblem = create_test_subproblem();
    subproblem.preallocate_cut_constraints(100, 10).unwrap();
    
    subproblem.create_iteration_model(true).unwrap();
    assert!(subproblem.create_iteration_model(true).is_err());
}
```

---

## Effort Estimate

**Points**: 5
**Confidence**: High

---

## Definition of Done

- [ ] All lifecycle methods implemented
- [ ] Optional basis logic correct
- [ ] Tests for all modes
- [ ] Documentation complete
- [ ] PR merged
