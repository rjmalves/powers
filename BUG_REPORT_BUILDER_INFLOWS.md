# Bug Report: SddpAlgorithm::builder() Missing Inflow Variables

**Date**: 2025-11-01  
**Severity**: HIGH - Causes incorrect optimization results  
**Status**: Identified, Fix Needed  
**Affects**: Programmatic builder API (`SddpAlgorithm::builder()`)

---

## 🐛 Bug Summary

When using `SddpAlgorithm::builder()` with `.deterministic_inflows()` and `.deterministic_loads()`, the SDDP algorithm produces **incorrect results** with costs approximately **7x higher** than expected.

**Example**:
- Expected cost: ~$2,500
- Actual cost: ~$17,500

This affects all tests using the programmatic builder:
- `test_deterministic_single_reservoir_convergence`
- `test_stochastic_single_reservoir_convergence`
- `test_two_reservoir_cascade_convergence`

---

## 🔍 Root Cause Analysis

### The Problem Chain

1. **Builder creates empty uncertainty models**
   - `build_graph()` calls `builder_empty_unified_specs()` which returns `Arc::new(vec![])`
   - Graph nodes are created with NO uncertainty models

2. **No inflow variables created**
   - `add_inflow_variables()` counts uncertainty models with `entity_type == Inflow`
   - With empty vector, count = 0, so **zero inflow variables** are added to LP

3. **Inflows not added to water balance**
   - Hydro balance constraint checks: `if hydro.id < variables.inflow.len()`
   - With empty inflow vector, this condition is always false
   - **Inflow term is never added** to the water balance constraint

4. **Hydros receive no water**
   - Hydro plants can only use initial storage
   - Inflows specified via `.deterministic_inflows()` are **completely ignored**

5. **Wrong optimization results**
   - Without inflows, hydro generation is severely limited
   - Thermal generation must compensate, increasing costs dramatically

### Code Locations

**File**: `src/sddp/builder.rs` (line ~254)
```rust
fn build_graph(
    system_factory: &dyn Fn() -> System,
    num_stages: usize,
    state_choice: &str,
) -> Result<DirectedGraph<NodeData>, String> {
    // ...
    
    // BUG: Creates nodes with empty uncertainty models!
    let uncertainty_models = builder_empty_unified_specs(); // Returns Arc::new(vec![])
    
    let node_data = NodeData::new(
        node_id,
        stage_id,
        StudyPeriodKind::Study,
        season_id,
        system_factory,
        state_choice,
        uncertainty_models, // Empty vec!
    );
    // ...
}
```

**File**: `src/subproblem.rs` (line 393-403)
```rust
fn add_inflow_variables(
    pb: &mut solver::Problem,
    uncertainty_models: &[crate::uncertainty_model::UncertaintyModel],
) -> (Vec<usize>, Vec<usize>, Vec<Vec<usize>>, Vec<usize>) {
    // BUG: With empty uncertainty_models, n_hydros = 0!
    let n_hydros = uncertainty_models
        .iter()
        .filter(|m| matches!(m.entity_type(), UncertaintyType::Inflow))
        .count(); // Returns 0 when uncertainty_models is empty
    
    // Creates empty vectors → no inflow variables in LP
    let mut inflow_obs = Vec::with_capacity(n_hydros); // capacity = 0
    // ...
}
```

**File**: `src/subproblem.rs` (line 573-576)
```rust
// BUG: This condition is always false when variables.inflow is empty!
if hydro.id < variables.inflow.len() { // Always false if len() == 0
    factors.push((variables.inflow[hydro.id], -1.0));
}
```

---

## 🎯 Expected Behavior

When using the programmatic builder:

```rust
SddpAlgorithm::builder()
    .system_factory(create_single_reservoir_system)
    .initial_storage(vec![20.0])
    .num_stages(2)
    .deterministic_inflows(vec![
        vec![15.0], // Stage 1: 15 MWh inflow
        vec![25.0], // Stage 2: 25 MWh inflow
    ])
    .deterministic_loads(vec![vec![50.0], vec![50.0]])
    .seed(42)
    .build_with_saa()
```

**Expected**: 
- Inflow variables created for each hydro
- Inflows added to water balance: `stored_volume[t] = stored_volume[t-1] + inflow[t] - turbined[t] - spillage[t]`
- Optimization uses available water (20 + 15 + 25 = 60 MWh)
- Cost: ~$2,500

**Actual**: 
- No inflow variables created
- Inflows NOT added to water balance
- Optimization only uses initial storage (20 MWh)
- Cost: ~$17,500 (7x higher due to excessive thermal generation)

---

## ✅ Solution

The builder needs to create **basic Independent uncertainty models** when `.deterministic_inflows()` is used, even though the actual inflow values come from the SAA.

### Fix Strategy

Modify `build_graph()` in `src/sddp/builder.rs` to:

1. **Detect if builder is being used programmatically** (no recourse.json)
2. **Create Independent uncertainty models** for each hydro
3. **Use standard normal parameters** (μ=0, σ=1) so residual = observation

### Detailed Fix

**File**: `src/sddp/builder.rs`

**Current code** (line ~254):
```rust
fn build_graph(
    system_factory: &dyn Fn() -> System,
    num_stages: usize,
    state_choice: &str,
) -> Result<DirectedGraph<NodeData>, String> {
    // ...
    let uncertainty_models = builder_empty_unified_specs(); // BUG: Empty!
    // ...
}
```

**Fixed code**:
```rust
fn build_graph(
    system_factory: &dyn Fn() -> System,
    num_stages: usize,
    state_choice: &str,
) -> Result<DirectedGraph<NodeData>, String> {
    // ...
    
    // FIX: Create basic Independent uncertainty models for programmatic builder
    let system = system_factory();
    let uncertainty_models = create_default_uncertainty_models(&system);
    
    // ...
}

/// Create default Independent uncertainty models for programmatic builder.
///
/// These models enable inflow variables to be created in the LP, allowing
/// inflows from SAA to be properly incorporated into water balance.
///
/// Uses standard normal parameters (μ=0, σ=1) so that:
/// - Transform: Y_t = 0 + 1*Z'_t = Z'_t
/// - AR (independent): Z'_t = ε_t  
/// - Result: Physical inflow Y_t = innovation ε_t from SAA
fn create_default_uncertainty_models(
    system: &System,
) -> Arc<Vec<UncertaintyModel>> {
    use crate::uncertainty_model::{
        MarginalDistribution, SeasonalParams, TemporalModelSpec, UncertaintyModel,
        UncertaintyType,
    };
    
    let num_seasons = 12; // Default monthly seasons
    let mut models = Vec::new();
    
    // Create Independent model for each hydro
    for hydro_id in 0..system.meta.hydros_count {
        // Standard normal seasonal params (μ=0, σ=1)
        // This means: observation = residual = innovation
        let seasonal_params: Vec<SeasonalParams> = (0..num_seasons)
            .map(|season_id| SeasonalParams {
                season_id,
                mean: 0.0,           // Zero mean
                std_dev: 1.0,        // Unit std dev
                ar_coefficients: vec![], // Independent (no AR)
                marginal_override: Some(MarginalDistribution::Normal {
                    mean: 0.0,
                    std_dev: 1.0,
                }),
            })
            .collect();
        
        models.push(UncertaintyModel::Independent {
            entity_type: UncertaintyType::Inflow,
            entity_id: hydro_id,
            seasonal_params,
        });
    }
    
    Arc::new(models)
}
```

### Why This Fix Works

With standard normal parameters (μ=0, σ=1):

1. **Transform constraint**: `Y_t = μ + σ*Z'_t = 0 + 1*Z'_t = Z'_t`
2. **AR constraint** (independent): `Z'_t = ε_t`
3. **Innovation** from SAA: `ε_t = 15.0` (from deterministic_inflows)
4. **Physical inflow**: `Y_t = Z'_t = ε_t = 15.0 MWh` ✓

This correctly maps the inflow values from `.deterministic_inflows()` to physical water balance!

---

## 🧪 Testing the Fix

After implementing the fix, verify with:

```bash
# Run the failing tests
cargo test test_deterministic_single_reservoir_convergence -- --nocapture
cargo test test_stochastic_single_reservoir_convergence -- --nocapture
cargo test test_two_reservoir_cascade_convergence -- --nocapture

# Should now pass with costs in expected range
```

**Expected results**:
- `test_deterministic_single_reservoir_convergence`: Cost in $1,500-$2,700 range
- `test_stochastic_single_reservoir_convergence`: Cost in $1,200-$3,300 range
- `test_two_reservoir_cascade_convergence`: Cost in $1,200-$2,150 range

---

## 📝 Alternative Approaches (Not Recommended)

### Alternative 1: Make SAA inject inflows directly
**Issue**: Violates separation of concerns. SAA provides noise samples, LP formulation handles constraint structure.

### Alternative 2: Special-case handling for empty uncertainty models
**Issue**: Adds complexity. Better to use the existing uncertainty model infrastructure properly.

### Alternative 3: Require recourse.json even for simple cases
**Issue**: Defeats the purpose of the programmatic builder API for ease of use.

---

## 🎯 Impact Assessment

### Who is affected?
- **Programmatic builder users**: Anyone using `SddpAlgorithm::builder()` without recourse.json
- **Test suite**: 3 benchmark tests currently failing
- **Examples**: Any examples using the builder programmatically (not affected if using `from_files()`)

### Who is NOT affected?
- **JSON-based API**: `SddpAlgorithm::from_files()` works correctly (provides recourse.json)
- **Existing production code**: If it uses recourse.json files

### Severity justification
**HIGH** because:
- Produces silently incorrect results (no error, just wrong answer)
- Cost difference is dramatic (7x higher)
- Affects core algorithm correctness
- Could lead to incorrect operational decisions

---

## 📚 Related Information

### Related Files
- `src/sddp/builder.rs` - Builder implementation (needs fix)
- `src/subproblem.rs` - Subproblem variable creation
- `src/uncertainty_model.rs` - Uncertainty model definitions
- `tests/fixtures/benchmarks.rs` - Affected tests

### Related Concepts
- **Observation space** (Y): Physical values (MWh)
- **Residual space** (Z'): Normalized values for AR dynamics
- **Innovation** (ε): Random component / noise term
- **Transform constraint**: Links observation and residual: Y = μ + σ*Z'
- **AR constraint**: Defines residual dynamics: Z'_t = Σφ*Z'_{t-k} + ε

### Documentation to Update
After fix:
- [ ] Update `docs/TESTING.md` to mark tests as passing
- [ ] Update `BENCHMARK_BASELINE.md` with correct expected values
- [ ] Add example of programmatic builder usage to documentation
- [ ] Consider adding validation warning if uncertainty_models is empty

---

## 🔄 Related Issues

This bug was introduced during the "simplifying recourse input" refactor (commit c102e6a) which:
- Removed `base_noise` module
- Simplified `unified_noise_spec`  
- Changed how uncertainty models are handled

The builder was not updated to create appropriate default uncertainty models for the programmatic case.

---

**Priority**: HIGH  
**Assignee**: To be assigned  
**Estimated effort**: 2-4 hours (implementation + testing)  
**Complexity**: Medium (requires understanding of uncertainty model structure)

---

## ✅ Acceptance Criteria

1. [ ] All 3 failing benchmark tests pass
2. [ ] Costs are within expected ranges
3. [ ] No regressions in other tests
4. [ ] Solution works for both single and multiple hydros
5. [ ] Solution works for cascade systems
6. [ ] Documentation updated
7. [ ] Code review completed

---

**End of Bug Report**
