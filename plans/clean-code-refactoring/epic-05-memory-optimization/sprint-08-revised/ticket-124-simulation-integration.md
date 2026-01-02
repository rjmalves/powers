# [T-124] Integrate Per-Iteration Model Lifecycle into Simulation

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-123 (training integration)
> **Blocks**: T-125
> **Priority**: 6 (Production Integration)
> **Status**: ✅ Complete (Verified - no changes needed)

## Files to Read Before Starting

- `src/sddp/mod.rs` - Current simulation implementation (lines 2363-2449)
- `src/sddp/mod.rs` - `SddpSimulationHandler` struct and methods
- `src/sddp/mod.rs` - `IterationLifecycleConfig::simulation()` (lines 93-99)

---

## Context

### Background

Simulation runs after training to evaluate the policy. Currently, simulation creates fresh handlers per scenario using `par_iter().map_init()`. Each handler creates its own Model.

For FCF persistence (future feature), simulation must be reproducible WITHOUT basis from training. The `IterationLifecycleConfig::simulation()` mode already exists for this purpose.

### Current State

```rust
// Current: Each parallel scenario creates its own handler with Model
let trajectories: Vec<SimulationTrajectory> = all_sampled_noises
    .par_iter()
    .enumerate()
    .map_init(
        || SddpSimulationHandler::new(...),  // Creates Model per thread
        |handler, (scenario_id, noises)| {
            // Use Model
        },
    )
    .collect();
```

### Target State

```rust
// Target: Same pattern but with explicit lifecycle control
// (Models created in handler::new() with simulation config)

// Option A: Keep current pattern, modify handler to use simulation config
// Option B: Create handlers once, reuse across scenarios with lifecycle

// Simulation typically runs N scenarios in parallel, each scenario needs
// independent solver state. Current pattern with map_init is appropriate.
```

### Key Insight

Simulation is DIFFERENT from training:
- Training: Same handlers reused across iterations → lifecycle per iteration
- Simulation: Different scenarios run in parallel → one handler per scenario (current pattern)

The main change needed is to ensure simulation handlers use `IterationLifecycleConfig::simulation()` (no basis).

---

## Specification

### Changes Required

1. **Verify simulation uses cold-start** (no basis warm-starting)
   - `SddpSimulationHandler::new()` should NOT use cached basis
   - This ensures reproducibility from persisted FCF

2. **Optional: Add lifecycle config parameter** to simulation
   - Allow caller to specify simulation vs training mode
   - Default to simulation mode (no basis)

3. **Document simulation memory behavior**
   - Each scenario gets its own handler/Model
   - Models dropped after `par_iter` completes
   - Memory is already reclaimed (Rayon pattern)

### Behavior

- **Simulation Mode**: `use_basis=false`, `cache_basis=false`
- **Models**: Created per scenario, dropped after scenario completes
- **Memory**: Reclaimed after each scenario (natural from Rayon pattern)
- **Reproducibility**: Same results regardless of training basis state

---

## Acceptance Criteria

- [ ] Simulation handlers use cold-start (no basis)
- [ ] Simulation results identical whether run immediately after training or from fresh load
- [ ] All simulation tests pass
- [ ] Memory reclaimed after simulation completes

---

## Implementation Guide

### Step 1: Verify Handler Behavior

Check `SddpSimulationHandler::new()`:

```rust
impl SddpSimulationHandler {
    pub fn new(...) -> Result<Self, String> {
        // Verify: Does this create Models without basis?
        // If Subproblem::new() creates Model, it's cold-start by default
        // No cached_basis available in fresh handler
    }
}
```

If handler creation already cold-starts (no cached basis), no changes needed.

### Step 2: Optional Enhancement

Add explicit lifecycle config to simulation:

```rust
pub fn simulate(
    &mut self,
    num_simulation_scenarios: usize,
    saa: &scenario::ScenarioTree,
) -> Result<Vec<SimulationTrajectory>, String> {
    let lifecycle_config = IterationLifecycleConfig::simulation();
    
    // ... rest of implementation ...
    // Handlers created with simulation config
}
```

### Step 3: Clear Cached Basis Before Simulation

If training cached basis in handlers, clear before simulation:

```rust
// If using same handlers for training and simulation (we don't, but for completeness)
for handler in &mut self.handlers {
    handler.clear_cached_basis();
}
```

This is NOT needed if simulation creates fresh handlers (current pattern).

---

## Testing Requirements

### Unit Tests

- [ ] Simulation handler creates cold-start Models
- [ ] No basis cached in simulation handlers

### Integration Tests

- [ ] Simulation after training produces same results as simulation from fresh instance
- [ ] Memory reclaimed after simulation completes

### Reproducibility Tests

- [ ] Run training → simulate → record results
- [ ] Create fresh instance → load FCF → simulate → compare results
- [ ] Results should match exactly

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Simulation already uses appropriate pattern; mainly verification and documentation

---

## Definition of Done

- [ ] Simulation uses cold-start (verified)
- [ ] Reproducibility test passes
- [ ] Documentation updated
- [ ] Code reviewed and merged
