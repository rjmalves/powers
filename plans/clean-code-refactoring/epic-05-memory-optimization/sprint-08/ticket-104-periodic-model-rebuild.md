# [T-104] Implement Subproblem::rebuild_model()

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 8: Model Rebuild Strategy](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-107, T-108
> **Priority**: 1 (Critical)
> **Status**: 🔵 Ready

## Files to Read Before Starting

- `docs/HIGHS_RSS_MEMORY_INVESTIGATION.md` - Root cause analysis of RSS growth
- `src/subproblem.rs` - Current model management (`new_from_temporal_models`, `preallocate_cut_constraints`)
- `src/solver.rs` - HiGHS Model wrapper and lifecycle
- `src/cut.rs` - BendersCut structure

---

## Context

### Background

DHAT and RSS analysis confirmed that HiGHS internal buffers grow throughout training but never shrink. The `rebuild_model()` method forces memory reclaim by:

1. Destroying the current HiGHS model (triggers `Highs_destroy()`)
2. Creating a fresh model with the same structure
3. Restoring only the currently active cuts

### Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Peak RSS (300 iter) | ~8 GB | ~5.5 GB |
| HiGHS internal buffers | Grow indefinitely | Reset every N iterations |

### Key Insight

The cut data is stored externally in `FutureCostFunction::cut_pool`. The HiGHS model only needs the cut constraints for solving. We can reconstruct the model without losing any algorithmic state.

---

## Specification

### New Types

```rust
/// Data needed to restore a cut after model rebuild.
/// Lightweight snapshot of cut state for reconstruction.
#[derive(Clone, Debug)]
pub struct ActiveCutData {
    /// Cut slot index (used for coefficient updates)
    pub slot: usize,
    /// Cut coefficients (including alpha coefficient for future cost variable)
    pub coefficients: Vec<f64>,
    /// Right-hand side value
    pub rhs: f64,
    /// Iteration when cut was created (for slot computation)
    pub iteration: usize,
    /// Forward pass index (for slot computation)
    pub forward_pass_idx: usize,
}
```

### New Methods

```rust
impl Subproblem {
    /// Rebuild the HiGHS model to reclaim memory.
    ///
    /// This operation destroys the current HiGHS model and creates a fresh one,
    /// then restores all active cuts from the provided cut data.
    ///
    /// # Arguments
    ///
    /// * `active_cuts` - Cuts to restore after rebuild
    /// * `system` - System specification (needed to rebuild constraints)
    /// * `temporal_models` - Temporal models for uncertainty handling
    /// * `remaining_iterations` - Remaining iterations (for cut slot sizing)
    /// * `num_forward_passes` - Forward passes per iteration
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if rebuild fails.
    ///
    /// # Performance
    ///
    /// This is expensive (~50-100ms per subproblem). Call every 50-100 iterations.
    pub fn rebuild_model(
        &mut self,
        active_cuts: &[ActiveCutData],
        system: &system::System,
        temporal_models: &[temporal_model::TemporalModel],
        remaining_iterations: usize,
        num_forward_passes: usize,
    ) -> Result<(), String>;
    
    /// Extract data for all active cuts in this subproblem.
    ///
    /// Returns a vector of ActiveCutData for cuts that are currently active
    /// (bounds set to enable the constraint, not deactivated).
    ///
    /// # Arguments
    ///
    /// * `cut_pool` - Reference to the FCF cut pool for this stage
    ///
    /// # Returns
    ///
    /// Vector of active cut data that can be used for `rebuild_model()`.
    pub fn extract_active_cuts(
        &self,
        cut_pool: &[cut::BendersCut],
    ) -> Vec<ActiveCutData>;
}
```

### Behavior

1. **Extract Phase**: Before rebuild, caller extracts active cuts via `extract_active_cuts()`
2. **Destroy Phase**: `self.model = None` triggers `Highs_destroy()` via `Drop`
3. **Rebuild Phase**: Create fresh `solver::Problem`, add variables/constraints
4. **Preallocate Phase**: Call `preallocate_cut_constraints()` with remaining capacity
5. **Restore Phase**: For each active cut, update coefficients and bounds
6. **Warmup Phase**: Call `warmup_solver()` to pre-allocate HiGHS internals

---

## Acceptance Criteria

- [ ] `ActiveCutData` struct defined
- [ ] `extract_active_cuts()` correctly identifies active cuts
- [ ] `rebuild_model()` creates structurally identical model
- [ ] Active cuts are correctly restored after rebuild
- [ ] Solving after rebuild produces same results as before
- [ ] Unit tests for rebuild logic
- [ ] Integration test: solve → rebuild → solve produces consistent results

---

## Implementation Guide

### Suggested Approach

#### Step 1: Add ActiveCutData struct

```rust
// In src/subproblem.rs, near top of file

/// Data needed to restore a cut after model rebuild.
#[derive(Clone, Debug)]
pub struct ActiveCutData {
    pub slot: usize,
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    pub iteration: usize,
    pub forward_pass_idx: usize,
}
```

#### Step 2: Implement extract_active_cuts()

```rust
impl Subproblem {
    pub fn extract_active_cuts(
        &self,
        cut_pool: &[cut::BendersCut],
    ) -> Vec<ActiveCutData> {
        let mut active_cuts = Vec::new();
        
        for cut in cut_pool.iter() {
            // Skip unpopulated or inactive cuts
            if !cut.is_populated() || !cut.is_active() {
                continue;
            }
            
            let slot = self.compute_cut_slot(cut.iteration, cut.forward_pass_idx);
            
            active_cuts.push(ActiveCutData {
                slot,
                coefficients: cut.coefficients.clone(),
                rhs: cut.rhs,
                iteration: cut.iteration,
                forward_pass_idx: cut.forward_pass_idx,
            });
        }
        
        active_cuts
    }
}
```

#### Step 3: Implement rebuild_model()

```rust
pub fn rebuild_model(
    &mut self,
    active_cuts: &[ActiveCutData],
    system: &system::System,
    temporal_models: &[temporal_model::TemporalModel],
    remaining_iterations: usize,
    num_forward_passes: usize,
) -> Result<(), String> {
    // 1. Preserve state that survives rebuild
    let state_choice = self.state.name().to_string();
    let season_id = self.season_id;
    
    // 2. Destroy old model (forces HiGHS deallocation)
    self.model = None;
    
    // 3. Rebuild from scratch (similar to new_from_temporal_models)
    let state = state::factory(&state_choice, system, temporal_models);
    let mut pb = solver::Problem::new();
    
    let variables = Self::add_variables(&mut pb, system, state.as_ref(), temporal_models);
    let constraints = Self::add_constraints(
        &mut pb, &variables, system, state.as_ref(), temporal_models, season_id
    );
    Self::add_offset_to_subproblem(&mut pb, system);
    
    let mut model = pb.optimise(solver::Sense::Minimise);
    set_retry_solver_options(&mut model, 0);
    
    // 4. Update self with new components
    self.model = Some(model);
    self.state = state;
    self.variables = variables;
    self.constraints = constraints;
    
    // Rebuild uncertainty observation data
    self.uncertainty_observation_data = Self::build_uncertainty_observation_data(
        temporal_models, &self.constraints, season_id
    );
    
    // Rebuild lag data structures
    self.rebuild_lag_data(system, temporal_models);
    
    // 5. Preallocate cut slots for remaining iterations
    let remaining_cuts = remaining_iterations * num_forward_passes;
    self.preallocate_cut_constraints(remaining_cuts, num_forward_passes)?;
    
    // 6. Restore active cuts
    for cut_data in active_cuts {
        self.restore_cut(cut_data)?;
    }
    
    // 7. Warmup solver
    self.warmup_solver()?;
    
    Ok(())
}

fn restore_cut(&mut self, cut_data: &ActiveCutData) -> Result<(), String> {
    let model = self.model.as_mut().ok_or("Model not initialized")?;
    let row = self.slot_to_row(cut_data.slot);
    
    // Update coefficients
    for (i, &var_idx) in self.cut_var_indices.iter().enumerate() {
        if i < cut_data.coefficients.len() {
            model.change_coefficient(row, var_idx, cut_data.coefficients[i])?;
        }
    }
    
    // Activate cut by setting bounds
    model.change_rows_bounds(row, cut_data.rhs, f64::INFINITY);
    
    Ok(())
}

fn rebuild_lag_data(
    &mut self,
    system: &system::System,
    temporal_models: &[temporal_model::TemporalModel],
) {
    // Rebuild load_lag_data if present
    if let Some(ref load_constraints) = self.constraints.load_lag_constraints {
        let mut data = LoadLagData::new(system.buses.len(), 0);
        data.constraints = load_constraints.clone();
        if let Some(ref lag_vars) = self.variables.lagged_state {
            for (entity_idx, model) in temporal_models.iter().enumerate() {
                if model.entity_type == crate::input::UncertaintyType::Load {
                    let bus_id = model.entity_id;
                    data.variables.lags_by_bus[bus_id] = lag_vars[entity_idx].clone();
                    data.allocate_buffer(bus_id, model.max_ar_order);
                }
            }
        }
        self.load_lag_data = Some(data);
    }
    
    // Rebuild inflow_lag_data if present
    if let Some(ref inflow_constraints) = self.constraints.inflow_lag_constraints {
        let mut data = InflowLagData::new(system.hydros.len(), 0);
        data.constraints = inflow_constraints.clone();
        if let Some(ref lag_vars) = self.variables.lagged_state {
            for (entity_idx, model) in temporal_models.iter().enumerate() {
                if model.entity_type == crate::input::UncertaintyType::Inflow {
                    let hydro_id = model.entity_id;
                    data.variables.lags_by_hydro[hydro_id] = lag_vars[entity_idx].clone();
                    data.allocate_buffer(hydro_id, model.max_ar_order);
                }
            }
        }
        self.inflow_lag_data = Some(data);
    }
}
```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/subproblem.rs` | Add `ActiveCutData`, `extract_active_cuts()`, `rebuild_model()`, `restore_cut()`, `rebuild_lag_data()` |

### Pitfalls to Avoid

- ⚠️ **Don't forget lag data**: The `load_lag_data` and `inflow_lag_data` must be rebuilt with fresh constraint references
- ⚠️ **Preserve state choice**: Extract state type name before dropping state
- ⚠️ **Order matters**: Preallocate cuts BEFORE restoring them
- ⚠️ **Slot consistency**: Use same slot computation logic as original

---

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_extract_active_cuts_empty() {
    // No cuts → empty result
}

#[test]
fn test_extract_active_cuts_filters_inactive() {
    // Only returns cuts where is_active() && is_populated()
}

#[test]
fn test_rebuild_model_preserves_structure() {
    // After rebuild, num_rows/num_cols should match
}

#[test]
fn test_rebuild_model_restores_cuts() {
    // Active cuts should be solvable after rebuild
}

#[test]
fn test_solve_after_rebuild_same_result() {
    // Solving same problem before/after rebuild gives identical objective
}
```

### Integration Tests

- [ ] Golden tests pass with rebuild enabled
- [ ] Training with rebuild produces same lower bounds

---

## Documentation Requirements

- [ ] Doc comments for `ActiveCutData`
- [ ] Doc comments for `rebuild_model()` with performance note
- [ ] Doc comments for `extract_active_cuts()`
- [ ] Update `HIGHS_RSS_MEMORY_INVESTIGATION.md` with implementation status

---

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Core infrastructure with multiple interacting components; needs careful state management

---

## Definition of Done

- [ ] `ActiveCutData` struct added
- [ ] `extract_active_cuts()` implemented and tested
- [ ] `rebuild_model()` implemented and tested
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation complete
- [ ] PR merged
