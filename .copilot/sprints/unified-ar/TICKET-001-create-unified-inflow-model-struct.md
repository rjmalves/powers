# TICKET-001: Create UnifiedInflowModel Struct and Core Interface

**Sprint:** 1 - Foundation  
**Phase:** 1 - Create Unified Inflow Model  
**Estimated Effort:** 2 days (5 story points)  
**Confidence:** High  
**Status:** Not Started

## Context

This is the foundational ticket for the unified AR model refactoring. The current architecture has multiple state implementations (StorageState, StorageAndInflowState) with conditional logic throughout the codebase. This ticket creates a new `UnifiedInflowModel` struct that will handle all inflow modeling (both independent and AR) in a single, consistent way.

The key insight is that **AR(0) = Independent**, allowing us to use a single representation with variable coefficients rather than multiple code paths. This eliminates complexity while preserving all functionality.

## Acceptance Criteria

- [ ] Given a set of unified noise specifications with AR coefficients, when UnifiedInflowModel is created, then it stores coefficients in residual space
- [ ] Given a set of unified noise specifications with empty AR coefficients (independent case), when UnifiedInflowModel is created, then it handles it as AR(0) with empty coefficient vectors
- [ ] Given a hydro with AR(p) model, when the model is queried for lag order, then it returns p
- [ ] Given a hydro with independent model, when the model is queried for lag order, then it returns 0
- [ ] Given seasonal parameters for each hydro/season, when UnifiedInflowModel is created, then it caches them efficiently via Arc
- [ ] Performance: UnifiedInflowModel construction should be O(n\*s) where n=hydros, s=seasons

## Tasks

### Implementation

- [ ] Create new file `src/unified_inflow_model.rs`
- [ ] Define `UnifiedInflowModel` struct with fields:
  - `dimension: usize` (number of hydros)
  - `ar_coefficients: Vec<Vec<f64>>` (per hydro, empty = independent)
  - `seasonal_params: Arc<SeasonalParamsCache>` (shared reference)
  - `lag_buffer: Vec<Vec<f64>>` (current lag values in residual space)
  - `max_lag: usize` (maximum lag order across all hydros)
- [ ] Implement `UnifiedInflowModel::new()` constructor
- [ ] Implement `UnifiedInflowModel::from_spec()` factory method that extracts AR coefficients from UnifiedNoiseSpec
- [ ] Implement `lag_order(&self, hydro: usize) -> usize` method
- [ ] Implement `max_lag(&self) -> usize` method
- [ ] Implement `dimension(&self) -> usize` method
- [ ] Implement `has_ar_dynamics(&self, hydro: usize) -> bool` helper
- [ ] Add comprehensive inline documentation explaining residual space representation
- [ ] Add `mod unified_inflow_model;` to `src/lib.rs`

### Testing

- [ ] Unit test: Create UnifiedInflowModel with independent noises (AR coefficients empty)
- [ ] Unit test: Create UnifiedInflowModel with AR(1) coefficients for all hydros
- [ ] Unit test: Create UnifiedInflowModel with mixed AR(1), AR(2), and independent hydros
- [ ] Unit test: Verify lag_order() returns correct values for each hydro
- [ ] Unit test: Verify max_lag() returns maximum across all hydros
- [ ] Unit test: Verify dimension() returns correct number of hydros
- [ ] Unit test: Verify has_ar_dynamics() correctly identifies AR vs independent
- [ ] Unit test: Verify seasonal_params are shared via Arc (test Arc::strong_count)

### Documentation

- [ ] Add module-level doc comment explaining unified AR representation
- [ ] Document the AR(0) = Independent design decision
- [ ] Add doc comments for each public method with examples
- [ ] Add section to `docs/architecture/` explaining unified inflow model design
- [ ] Update CHANGELOG.md with "Added: UnifiedInflowModel for consistent AR representation"

## Technical Notes

### Design Decisions

1. **Empty vector = Independent**: For independent hydros, `ar_coefficients[hydro]` is an empty Vec. This allows uniform code paths without conditionals.

2. **Residual Space Native**: All AR coefficients and lag buffers are in residual space (normalized, zero-mean). Transformation to observation space happens in LP constraints.

3. **Shared Seasonal Params**: Use `Arc<SeasonalParamsCache>` to avoid cloning large parameter arrays across subproblems.

### Implementation Approach

```rust
pub struct UnifiedInflowModel {
    dimension: usize,
    ar_coefficients: Vec<Vec<f64>>,
    seasonal_params: Arc<SeasonalParamsCache>,
    lag_buffer: Vec<Vec<f64>>,
    max_lag: usize,
}

impl UnifiedInflowModel {
    pub fn from_spec(
        unified_specs: &[UnifiedNoiseSpec],
        n_hydros: usize,
        seasonal_params: Arc<SeasonalParamsCache>,
    ) -> Self {
        let mut ar_coefficients = vec![Vec::new(); n_hydros];
        let mut max_lag = 0;

        for (hydro, spec) in unified_specs.iter().enumerate() {
            if let Some(ar_params) = &spec.ar_parameters {
                ar_coefficients[hydro] = ar_params.phi.clone();
                max_lag = max_lag.max(ar_params.phi.len());
            }
            // else: empty Vec = independent (AR(0))
        }

        let lag_buffer = vec![vec![0.0; max_lag]; n_hydros];

        Self {
            dimension: n_hydros,
            ar_coefficients,
            seasonal_params,
            lag_buffer,
            max_lag,
        }
    }
}
```

### Edge Cases

- **All independent**: max_lag = 0, all ar_coefficients are empty vectors
- **Mixed orders**: Some hydros AR(1), some AR(2), some independent
- **Single hydro**: dimension = 1, should work correctly
- **No seasonal variation**: seasonal_params contains same μ, σ for all seasons

### Performance Considerations

- **Memory**: O(n\*p) where p is max lag order
- **Construction**: O(n) - just extracting coefficients from specs
- **No allocations in hot paths**: All vectors pre-allocated during construction

## Dependencies

- **Blocked by**: None (first ticket in sequence)
- **Blocks**: TICKET-002 (constraint generation needs this struct)
- **Related**: TICKET-003 (lag buffer management uses this struct)

## References

- `src/unified_noise_spec.rs` - UnifiedNoiseSpec definition
- `src/seasonal_params.rs` - SeasonalParamsCache definition
- `UNIFIED_AR_ROADMAP.md` - Overall architecture design

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All unit tests pass
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Documentation builds without warnings (`cargo doc`)
- [ ] Code reviewed by at least one team member
