# TICKET-002: Implement LP Constraint Generation for Unified AR Model

**Sprint:** 1 - Foundation  
**Phase:** 1 - Create Unified Inflow Model  
**Estimated Effort:** 3 days (8 story points)  
**Confidence:** High  
**Status:** Not Started

## Context

With the UnifiedInflowModel struct in place, this ticket implements the core functionality: generating LP constraints that represent AR dynamics explicitly, even for independent noise. This is the heart of the unified representation that eliminates conditional logic in hot paths.

The constraint formulation always includes:

1. **AR dynamics constraint**: `Z'_t - Σ(φ_k * Z'_{t-k}) = ε_t` (even when φ is empty for independent case)
2. **Observation transformation**: `Y_t = μ_s + σ_s * Z'_t` (links residual to observation space)

## Acceptance Criteria

- [ ] Given a UnifiedInflowModel with AR coefficients, when constraints are added to LP, then AR dynamics constraints are created with correct coefficients
- [ ] Given a UnifiedInflowModel with empty coefficients (independent), when constraints are added to LP, then simplified AR constraints are created (Z'\_t = ε_t)
- [ ] Given any UnifiedInflowModel, when constraints are added, then observation transformation constraints are created (Y = μ + σZ')
- [ ] Given a subproblem in season s, when constraints are added, then correct seasonal parameters (μ_s, σ_s) are used
- [ ] Given multiple hydros with different lag orders, when constraints are added, then each hydro gets correct number of lag terms
- [ ] Performance: Constraint generation should be O(n\*p) where n=hydros, p=max_lag

## Tasks

### Implementation

- [ ] Add `ConstraintIndices` struct to hold constraint indices for each hydro:
  - `ar_dynamics: Vec<usize>` (one per hydro)
  - `observation_transform: Vec<usize>` (one per hydro)
- [ ] Implement `add_constraints_to_lp()` method:
  - Takes `&self`, `pb: &mut solver::Problem`, `vars: &Variables`, `season_id: usize`
  - Returns `ConstraintIndices`
- [ ] For each hydro, add AR dynamics constraint:
  - LHS: `Z'_t[hydro] - Σ(φ_k[hydro] * Z'_{t-k}[hydro])`
  - RHS: `ε_t[hydro]` (set to 0.0 initially, updated at solve time)
  - Handle empty coefficients (independent case): just `Z'_t[hydro] = ε_t[hydro]`
- [ ] For each hydro, add observation transformation constraint:
  - LHS: `Y_t[hydro] - σ_s[hydro] * Z'_t[hydro]`
  - RHS: `μ_s[hydro]`
  - Lookup μ, σ from seasonal_params using season_id
- [ ] Add bounds to innovation variables (ε_t): typically [-∞, +∞] for unbounded
- [ ] Add bounds to residual variables (Z'\_t): typically [-∞, +∞] for unbounded
- [ ] Add bounds to lag residual variables (Z'\_{t-k}): typically fixed via RHS update
- [ ] Optimize coefficient vector construction to avoid repeated allocations

### Testing

- [ ] Unit test: AR(1) model with φ=0.8, verify constraint has correct coefficient
- [ ] Unit test: AR(2) model with φ=[0.6, 0.3], verify constraint has both lag terms
- [ ] Unit test: Independent model (empty φ), verify constraint simplifies to Z'=ε
- [ ] Unit test: Mixed models (2 hydros AR, 1 independent), verify all constraints correct
- [ ] Unit test: Observation transform uses correct seasonal params for season_id
- [ ] Unit test: Verify ConstraintIndices returns correct row indices
- [ ] Integration test: Add constraints to actual solver::Problem and verify solvability
- [ ] Performance test: Benchmark constraint generation for 50 hydros with AR(3)

### Documentation

- [ ] Add detailed doc comment for `add_constraints_to_lp()` explaining:
  - AR dynamics formulation in residual space
  - Observation space transformation
  - How independent case is handled (empty coefficients)
- [ ] Add doc comment for `ConstraintIndices` struct
- [ ] Add inline comments explaining constraint construction
- [ ] Add example to module-level docs showing constraint structure
- [ ] Update CHANGELOG.md with "Added: LP constraint generation for unified AR model"

## Technical Notes

### Constraint Formulation

**AR Dynamics Constraint (per hydro h):**

```
Z'_t[h] - φ₁[h]*Z'_{t-1}[h] - φ₂[h]*Z'_{t-2}[h] - ... - φₚ[h]*Z'_{t-p}[h] = ε_t[h]
```

For independent case (empty φ):

```
Z'_t[h] = ε_t[h]
```

**Observation Transformation (per hydro h):**

```
Y_t[h] = μ_s[h] + σ_s[h] * Z'_t[h]

Rearranged for LP:
Y_t[h] - σ_s[h]*Z'_t[h] = μ_s[h]
```

### Implementation Pattern

```rust
impl UnifiedInflowModel {
    pub fn add_constraints_to_lp(
        &self,
        pb: &mut solver::Problem,
        vars: &Variables,
        season_id: usize,
    ) -> ConstraintIndices {
        let mut ar_dynamics = Vec::with_capacity(self.dimension);
        let mut observation_transform = Vec::with_capacity(self.dimension);

        for hydro in 0..self.dimension {
            // AR dynamics: Z'_t - Σ(φ_k * Z'_{t-k}) = ε_t
            let mut factors = vec![(vars.inflow_residual[hydro], 1.0)];

            for (lag_idx, &coeff) in self.ar_coefficients[hydro].iter().enumerate() {
                factors.push((vars.lag_residual[hydro][lag_idx], -coeff));
            }

            // RHS will be set to innovation at solve time
            let ar_row = pb.add_row(0.0..=0.0, &factors);
            ar_dynamics.push(ar_row);

            // Observation transform: Y - σZ' = μ
            let (mu, sigma) = self.seasonal_params.get(hydro, season_id);
            let obs_factors = vec![
                (vars.inflow[hydro], 1.0),
                (vars.inflow_residual[hydro], -sigma),
            ];
            let obs_row = pb.add_row(mu..=mu, &obs_factors);
            observation_transform.push(obs_row);
        }

        ConstraintIndices { ar_dynamics, observation_transform }
    }
}
```

### Edge Cases

- **Zero coefficients**: If φ_k = 0.0, should still include in constraint (solver handles efficiently)
- **Numerical stability**: Use exact equality constraints (RHS bounds: val..=val)
- **Large lag orders**: Pre-allocate factors vector to avoid repeated allocations
- **Missing seasonal params**: Should panic early with clear error message

### Performance Considerations

- **Allocation**: Pre-allocate constraint indices vectors (with_capacity)
- **Coefficient lookup**: O(1) via vector indexing
- **Seasonal params**: O(1) lookup via cached HashMap or Vec
- **Total complexity**: O(n\*p) unavoidable - must create all constraints

### Variables Required

This method assumes `Variables` struct has:

- `inflow: Vec<usize>` (observation space, Y_t)
- `inflow_residual: Vec<usize>` (residual space, Z'\_t)
- `lag_residual: Vec<Vec<usize>>` (residual space, Z'\_{t-k})
- `innovation: Vec<usize>` (white noise, ε_t)

These will be added in TICKET-004 (Variable Structure Refactoring).

## Dependencies

- **Blocked by**: TICKET-001 (needs UnifiedInflowModel struct)
- **Blocks**: TICKET-005 (realize_uncertainties needs constraint indices)
- **Related**: TICKET-004 (requires updated Variables struct)

## References

- `src/solver.rs` - Solver interface for adding constraints
- `src/seasonal_params.rs` - SeasonalParamsCache interface
- UNIFIED_AR_ROADMAP.md - Section 2.1 (Space Consistency decision)

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All unit tests pass
- [ ] Integration test with real solver passes
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Performance benchmark shows O(n\*p) scaling
- [ ] Documentation builds without warnings
- [ ] Code reviewed by at least one team member
