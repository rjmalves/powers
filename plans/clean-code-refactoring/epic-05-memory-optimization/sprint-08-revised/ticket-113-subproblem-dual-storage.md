# [T-113] Refactor Subproblem for Dual Problem+Model Storage

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-110, T-111, T-112
> **Blocks**: T-114, T-115, T-116
> **Priority**: 1 (Critical Path)
> **Status**: 📋 Planned

## Files to Read Before Starting

- `src/subproblem.rs` - Current Subproblem structure
- `src/solver.rs` - Problem and Model types (with T-110, T-111, T-112)

---

## Context

### Background

The new architecture requires Subproblem to hold:
1. **Problem**: Persistent source of truth
2. **Model**: Transient per-iteration
3. **Cached Basis**: Optional, for warm-starting (training only)

---

## Specification

### Struct Changes

```rust
pub struct Subproblem {
    /// Persistent LP problem definition (source of truth).
    pub problem: solver::Problem,
    
    /// Transient Model for current iteration.
    /// Created via `create_iteration_model()`, dropped via `finalize_iteration()`.
    pub model: Option<solver::Model>,
    
    /// Cached basis for optional warm-starting.
    /// Used in training mode, not in simulation mode.
    pub cached_basis: Option<solver::StoredBasis>,
    
    // Unchanged fields
    pub state: Box<dyn state::State>,
    pub variables: Variables,
    pub constraints: Constraints,
    pub season_id: usize,
    pub load_lag_data: Option<LoadLagData>,
    pub inflow_lag_data: Option<InflowLagData>,
    pub uncertainty_observation_data: Vec<UncertaintyObservationData>,
    first_preallocated_cut_row: usize,
    num_preallocated_cuts: usize,
    num_forward_passes: usize,
    cut_var_indices: Vec<usize>,
}
```

### Constructor Changes

```rust
pub fn new_from_temporal_models(
    system: &system::System,
    state_choice: &str,
    temporal_models: &[temporal_model::TemporalModel],
    season_id: usize,
) -> Self {
    let state = state::factory(state_choice, system, temporal_models);
    let mut problem = solver::Problem::new();
    
    let variables = Self::add_variables(&mut problem, ...);
    let constraints = Self::add_constraints(&mut problem, ...);
    Self::add_offset_to_subproblem(&mut problem, system);
    
    // Build other data as before...
    
    Self {
        problem,            // NEW
        model: None,        // NEW: No Model yet
        cached_basis: None, // NEW: No cached basis yet
        state,
        variables,
        constraints,
        // ... rest unchanged
    }
}
```

### Preallocate on Problem

```rust
pub fn preallocate_cut_constraints(
    &mut self,
    max_cuts: usize,
    num_forward_passes: usize,
) -> Result<(), String> {
    let cut_var_indices = self.state.get_cut_variable_indices(&self.variables);
    let first_cut_row = self.problem.num_row;
    
    // Add rows to Problem
    for _ in 0..max_cuts {
        let row_factors: Vec<(usize, f64)> = cut_var_indices
            .iter()
            .map(|&var| (var, 0.0))
            .collect();
        
        self.problem.add_row(f64::NEG_INFINITY..f64::INFINITY, row_factors);
    }
    
    self.first_preallocated_cut_row = first_cut_row;
    self.num_preallocated_cuts = max_cuts;
    self.num_forward_passes = num_forward_passes;
    self.cut_var_indices = cut_var_indices;
    
    Ok(())
}
```

---

## Acceptance Criteria

- [ ] Subproblem has `problem`, `model`, `cached_basis` fields
- [ ] Constructor stores Problem, not Model
- [ ] `preallocate_cut_constraints()` works on Problem
- [ ] Code compiles
- [ ] Existing tests compile

---

## Testing Requirements

```rust
#[test]
fn test_subproblem_structure() {
    let subproblem = create_test_subproblem();
    
    assert!(subproblem.problem.num_col > 0);
    assert!(subproblem.model.is_none());
    assert!(subproblem.cached_basis.is_none());
}

#[test]
fn test_preallocate_on_problem() {
    let mut subproblem = create_test_subproblem();
    let initial_rows = subproblem.problem.num_row;
    
    subproblem.preallocate_cut_constraints(100, 10).unwrap();
    
    assert_eq!(subproblem.problem.num_row, initial_rows + 100);
}
```

---

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Large struct change, many methods to audit

---

## Definition of Done

- [ ] Struct changed
- [ ] Constructor updated
- [ ] `preallocate_cut_constraints` on Problem
- [ ] Code compiles
- [ ] Tests pass
- [ ] PR merged
