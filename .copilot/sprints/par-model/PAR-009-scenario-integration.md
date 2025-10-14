# PAR-009: Integrate PAR Generator into Scenario Pipeline

## Context

**INTEGRATION TICKET**: Wire PAR generator into existing scenario generation pipeline (`src/scenario.rs`). Refactor current pipeline order:

**Current**: Noise → Correlation → Marginal → AR  
**CEPEL**: Noise → Correlation → Residual Transform → PAR

This requires updating `generate_scenarios()` to detect `PeriodicAutoregressive` variant and route through new pipeline. The seasonal index (period m) is determined by the `season_id` field in graph nodes, allowing flexible periodicity (monthly, quarterly, etc.).

## Acceptance Criteria

- [ ] `generate_scenarios()` detects PAR temporal model
- [ ] PAR pipeline: noise → correlation → residual transform → PAR generator
- [ ] Stationary AR/Independent pipelines unchanged
- [ ] Backward compatible (existing configs work)
- [ ] Integration tests with PAR fixture pass
- [ ] No performance regression (<10% overhead)

## Tasks

### Implementation

- [ ] Update `src/scenario.rs::generate_scenarios()` to branch on TemporalModel variant
- [ ] Add PAR-specific pipeline path
- [ ] Integrate `PeriodicARGenerator` for PAR cases
- [ ] Integrate `transform_to_residuals()` for residual transformation
- [ ] Preserve initial conditions handling
- [ ] Add warning logs when using PAR mode

### Testing

- [ ] Integration test: PAR fixture generates valid scenarios
- [ ] Integration test: Compare PAR vs stationary AR output structure
- [ ] Integration test: Verify spatial correlation preserved
- [ ] Integration test: Multi-inflow PAR generation
- [ ] Regression test: Existing Independent/AR fixtures unchanged

### Documentation

- [ ] Pipeline diagram update
- [ ] Migration guide for users

## Technical Notes

```rust
// Pseudocode for scenario.rs integration

pub fn generate_scenarios(...) -> Result<Vec<Scenario>, PowersError> {
    match &noise_model.temporal_model {
        TemporalModel::Independent => {
            // Existing: direct marginal transformation
        }
        TemporalModel::Autoregressive { .. } => {
            // Existing: stationary AR path
        }
        TemporalModel::PeriodicAutoregressive { .. } => {
            // NEW: CEPEL PAR pipeline

            // 1. Generate base noise
            let base_noise = generate_base_noise(...);

            // 2. Apply spatial correlation
            let correlated = apply_correlation(base_noise, cholesky);

            // 3. Transform to residuals
            let residuals = transform_to_residuals(
                &correlated,
                &noise_model.residual_distribution.unwrap()
            );

            // 4. Apply PAR generator
            let seasonal_params = SeasonalParams::try_from(&noise_model.temporal_model)?;
            let mut generator = PeriodicARGenerator::new(seasonal_params, initial_residuals);

            let series: Vec<f64> = residuals.iter()
                .map(|&a_t| generator.generate_next(a_t))
                .collect();

            series
        }
    }
}
```

## Dependencies

- **Blocked by**: PAR-006 (generator), PAR-007 (residual transform)
- **Blocks**: PAR-010 (validation tests)
- **Related**: PAR-004 (fixtures)

## Estimated Effort

**2 story points** (1 day)
