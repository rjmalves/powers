# Phase 1 Progress: Fix Test Compilation

**Started**: November 1, 2025  
**Status**: IN PROGRESS  
**Current Task**: Task 1.1 - Fix Source File Test Modules

---

## Task 1.1: Fix Source File Test Modules ⏱️ 2h

### src/state.rs

**Status**: ✅ PARTIAL - First 5 tests fixed

**Fixed Tests**:
- ✅ test_new_storage_state
- ✅ test_factory_storage_state
- ✅ test_factory_storage_and_inflow_state
- ✅ test_factory_invalid_choice
- ✅ test_factory_preserves_system_dimension

**Fixed Helper Functions**:
- ✅ create_uncertainty_model_independent() (was create_noise_spec_independent)
- ✅ create_uncertainty_model_par() (was create_noise_spec_par)

**Remaining Tests** (need StateLayout API review):
- [ ] test_extract_max_ar_order_for_hydro_independent
- [ ] test_extract_max_ar_order_for_hydro_par
- [ ] test_extract_max_ar_order_for_hydro_not_found
- [ ] test_state_layout_homogeneous_naive
- [ ] test_state_layout_homogeneous_ar1
- [ ] test_state_layout_heterogeneous
- [ ] test_state_layout_hydro_slice
- [ ] test_state_layout_hydro_dim
- [ ] test_state_layout_hydro_storage_offset
- [ ] test_state_layout_hydro_lag_count
- [ ] test_state_layout_empty_noise_models

**Note**: These tests use `StateLayout::from_unified_specs()` which may not exist in new API.
Need to check if StateLayout still exists and what its constructor is.

### src/subproblem.rs

**Status**: ⏸️ NOT STARTED  
**Estimate**: ~20 occurrences of old API

### src/fcf.rs  

**Status**: ⏸️ NOT STARTED  
**Estimate**: Several occurrences of old API

---

## Compilation Status

**Before fixes**: 125 errors  
**Current**: ~90 errors (estimated based on grep output)  
**Progress**: ~28% (35 errors fixed)

**Remaining error sources**:
- src/state.rs - More tests using StateLayout
- src/subproblem.rs - Test modules
- src/fcf.rs - Test modules

---

## Key Learnings

### API Changes Discovered

1. **StorageState::new()** signature changed:
   - OLD: `new(&System, &[UnifiedNoiseSpec])`
   - NEW: `new(&System)` - No longer needs uncertainty models!

2. **factory()** signature:
   - OLD: `factory(kind, &System, &[UnifiedNoiseSpec])`
   - NEW: `factory(kind, &System, &[UncertaintyModel])`

3. **UncertaintyModel enum variants**:
   - `Independent { seasonal_params: Vec<SeasonalParams> }`
   - `PeriodicAR { par_params: PARParams }`

4. **No more TemporalModelSpec**:
   - Concept merged into UncertaintyModel enum variants

### Migration Patterns Established

**Pattern: Independent Model**:
```rust
uncertainty_model::UncertaintyModel::Independent {
    seasonal_params: vec![uncertainty_model::SeasonalParams {
        mean: 100.0,
        std_dev: 20.0,
        distribution: uncertainty_model::DistributionType::Normal,
    }],
}
```

**Pattern: PAR Model**:
```rust
uncertainty_model::UncertaintyModel::PeriodicAR {
    par_params: uncertainty_model::PARParams {
        ar_orders: vec![1, 1, 1],
        ar_coefficients: vec![vec![0.7], vec![0.7], vec![0.7]],
        seasonal_means: vec![100.0; 3],
        seasonal_stds: vec![20.0; 3],
        seasonal_distributions: vec![
            uncertainty_model::DistributionType::LogNormal3 {
                gamma: 1.0,
                mu: 4.5,
                sigma: 0.3,
            };
            3
        ],
        period: 3,
    },
}
```

---

## Next Steps

### Immediate (Next 30 min)

1. ✅ Document progress (this file)
2. ⏭️ Check if StateLayout API still exists
3. ⏭️ Fix remaining state.rs tests OR
4. ⏭️ Move to subproblem.rs if StateLayout tests need rewrite

### Decision Point

**If StateLayout tests are salvageable**:
  → Fix them using new API
  → Complete state.rs
  → Move to subproblem.rs

**If StateLayout tests need complete rewrite**:
  → Skip for now (mark as Task 4.2)
  → Move to subproblem.rs (more impactful)
  → Return to StateLayout tests in Phase 2

---

## Files Created

- ✅ API_MIGRATION.md - Comprehensive migration guide
- ✅ PHASE1_PROGRESS.md - This file

---

## Time Tracking

- **Task 1.1 Start**: 18:24 UTC
- **Progress Check**: 18:45 UTC (~20 min)
- **Estimated Completion**: 19:30 UTC (1h remaining)

---

**Next update**: After completing state.rs OR starting subproblem.rs
