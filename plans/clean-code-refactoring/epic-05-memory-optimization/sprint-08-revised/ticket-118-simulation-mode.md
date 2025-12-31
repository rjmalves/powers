# [T-118] Add Basis Configuration for Simulation Mode

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-117
> **Blocks**: T-119
> **Priority**: 2
> **Status**: 📋 Planned

## Files to Read Before Starting

- `src/sddp/mod.rs` - Training and simulation loops
- T-117 implementation

---

## Context

### Background

Simulation mode must use `use_basis=false` to ensure reproducibility when running from a loaded FCF. This ticket adds the configuration and ensures simulation properly cold-starts.

### Why This Matters for FCF Persistence (Future Feature)

When users load a persisted FCF:
1. They don't have the basis (too large to store)
2. Simulation must produce identical results whether:
   - Run immediately after training (basis available but unused)
   - Run from loaded FCF (no basis)

---

## Specification

### Configuration

```rust
/// Configuration for iteration lifecycle.
#[derive(Clone, Copy, Debug)]
pub struct IterationLifecycleConfig {
    /// Whether to use cached basis for warm-starting.
    /// - Training: true (performance)
    /// - Simulation: false (reproducibility)
    pub use_basis: bool,
    
    /// Whether to cache basis at end of iteration.
    /// - Training: true (for next iteration)
    /// - Simulation: false (not needed)
    pub cache_basis: bool,
}

impl IterationLifecycleConfig {
    /// Configuration for training iterations.
    pub fn training() -> Self {
        Self { use_basis: true, cache_basis: true }
    }
    
    /// Configuration for simulation iterations.
    pub fn simulation() -> Self {
        Self { use_basis: false, cache_basis: false }
    }
}
```

### Simulation Loop

```rust
impl SddpAlgorithm {
    pub fn simulate(&mut self, config: &SimulationConfig) -> SimulationResult {
        let lifecycle = IterationLifecycleConfig::simulation();
        
        // Clear any cached basis from training
        for handler in &mut self.handlers {
            handler.subproblem.clear_cached_basis();
        }
        
        for scenario in 0..config.num_scenarios {
            // Create models WITHOUT basis
            self.create_iteration_models(lifecycle.use_basis)?;
            
            // Forward pass only (simulation)
            let result = self.forward_pass_simulation(scenario)?;
            
            // Finalize WITHOUT caching
            self.finalize_iteration(lifecycle.cache_basis);
        }
        
        // Aggregate results...
    }
    
    /// Helper method using lifecycle config.
    fn run_iteration(&mut self, lifecycle: &IterationLifecycleConfig) -> Result<(), String> {
        self.create_iteration_models(lifecycle.use_basis)?;
        // ... forward/backward passes ...
        self.finalize_iteration(lifecycle.cache_basis);
        Ok(())
    }
}
```

---

## Acceptance Criteria

- [ ] `IterationLifecycleConfig` type defined
- [ ] Training uses `training()` config
- [ ] Simulation uses `simulation()` config
- [ ] Simulation clears cached basis before starting
- [ ] Simulation cold-starts every iteration
- [ ] Results identical with/without prior basis
- [ ] Unit tests for both modes

---

## Testing Requirements

```rust
#[test]
fn test_simulation_mode_cold_start() {
    let mut algorithm = create_test_algorithm();
    
    // Train first (caches basis)
    algorithm.train(&TrainingConfig { max_iterations: 5, .. }).unwrap();
    
    for handler in &algorithm.handlers {
        assert!(handler.subproblem.cached_basis.is_some());
    }
    
    // Simulate (should clear and not use basis)
    algorithm.simulate(&SimulationConfig { num_scenarios: 3, .. }).unwrap();
    
    // Basis should be cleared
    for handler in &algorithm.handlers {
        assert!(handler.subproblem.cached_basis.is_none());
    }
}

#[test]
fn test_lifecycle_config_training() {
    let config = IterationLifecycleConfig::training();
    assert!(config.use_basis);
    assert!(config.cache_basis);
}

#[test]
fn test_lifecycle_config_simulation() {
    let config = IterationLifecycleConfig::simulation();
    assert!(!config.use_basis);
    assert!(!config.cache_basis);
}
```

---

## Effort Estimate

**Points**: 3
**Confidence**: High

---

## Definition of Done

- [ ] Lifecycle config type
- [ ] Simulation mode implemented
- [ ] Basis cleared before simulation
- [ ] Tests passing
- [ ] PR merged
